# type: ignore
import torch
from torch.optim import Optimizer
from torch import Tensor


from typing import List, Optional, Tuple, Union, Callable, Dict
from math import pow, sqrt
from functools import partial
from contextlib import contextmanager
import torch.distributed as dist
import copy
import os

from .soap import SOAP


def get_param_groups(
    named_parameters,
    no_weight_decay_for_bias_and_norm_params=True,
    weight_lr_scale=1.0,
    no_wd_on_embedding=False,  # for tied models this needs to be false to have wd on the logit layer
    verbose=False,
):
    param_groups = []

    weights_group = []  # default group
    embedding_group = []
    scale_and_norm_group = []
    # readout_group = [] # unused
    for name, param in named_parameters:
        lname = name.lower()
        if "wte" in lname or "embedding" in lname or "abacus" in lname or "lm_head" in lname:
            embedding_group.append(param)
            if verbose:
                print(f"Matched {name} to embedding group")
        elif "ln_f" in lname or "norm" in lname or "bias" in lname or "diff_lmb" in name:
            scale_and_norm_group.append(param)
            if verbose:
                print(f"Matched {name} to scale and norm_group")
        elif "proj" in lname or "qkv" in lname or "fc" in lname or param.ndim == 2:
            weights_group.append(param)
            if verbose:
                print(f"Matched {name} to main weight group")
        else:
            raise ValueError(f"param {name} could not be matched to an optim group in recpre/optim.py")

    param_groups.append({"params": weights_group, "base_lr": weight_lr_scale})
    param_groups.append({"params": embedding_group, "base_lr": 1.0})
    if no_wd_on_embedding:
        param_groups[-1]["weight_decay"] = 0.0
    param_groups.append({"params": scale_and_norm_group, "base_lr": 1.0})
    if no_weight_decay_for_bias_and_norm_params:
        param_groups[-1]["weight_decay"] = 0.0

    return param_groups


def get_optimizer(
    optimizer_name,
    model=None,
    pytorch_optimizer_sharding: bool = False,
    allow_fusion: bool = True,
    use_apex_adamw: bool = False,
):
    if hasattr(torch.optim, optimizer_name):
        optim_class = getattr(torch.optim, optimizer_name)  # read all torch optimizers
    elif optimizer_name == "LionW":
        optim_class = LionW
    elif optimizer_name == "SophiaG":
        optim_class = SophiaG
    elif optimizer_name == "Lilith":
        optim_class = Lilith
    elif optimizer_name == "ELLISAdam":
        optim_class = ELLISAdam
    elif optimizer_name == "IVON":
        optim_class = IVON
    elif optimizer_name == "simo-shampoo":
        optim_class = ZeroShampooWithAdamGraftingOptimizer
    elif optimizer_name == "meta-shampoo":
        try:
            from distributed_shampoo.distributed_shampoo import DistributedShampoo
            from distributed_shampoo.shampoo_types import AdamGraftingConfig, CommunicationDType, DDPShampooConfig
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Run `pip install git+https://github.com/JonasGeiping/meta-shampoo` first!")

        optim_class = partial(
            DistributedShampoo,
            grafting_config=AdamGraftingConfig(
                beta2=0.95,
                epsilon=1e-8,
            ),
            distributed_config=DDPShampooConfig(
                communication_dtype=CommunicationDType.FP32,
                num_trainers_per_group=torch.cuda.device_count(),
                communicate_params=False,
            ),
        )
    elif optimizer_name == "SOAP":
        optim_class = SOAP
    elif optimizer_name == "Kellers":
        optim_class = OrthogonalNesterov
    else:
        raise ValueError(f"Invalid optimizer {optimizer_name} requested.")

    if optimizer_name == "AdamW" and use_apex_adamw:
        try:
            from apex.optimizers import FusedAdam
        except ModuleNotFoundError:
            raise ValueError("Need to install apex!")

        optim_class = FusedAdam
        print("Using apex.optimizers.FusedAdam")

    if allow_fusion:
        import inspect

        if "fused" in inspect.signature(optim_class).parameters:
            # llm.c trick to fish for fused implementations
            optim_class = partial(optim_class, fused=True)

    if pytorch_optimizer_sharding and torch.distributed.is_initialized():
        # from torch.distributed.optim import ZeroRedundancyOptimizer

        # return partial(ZeroRedundancyOptimizer, optimizer_class=optim_class, overlap_with_ddp=False)
        return partial(SimpleZeroRedundancyOptimizer, optimizer_class=optim_class)
    else:
        return optim_class


class SimpleZeroRedundancyOptimizer(Optimizer):
    """
    Works with the following magic numbers: param group 0 is sharded across the 8 devices on each node.
                                            param group 1 is kept only on device 0
                                            all other groups are not sharded
    """

    verbose = False
    single_rank_return = False

    def __init__(
        self,
        params,
        optimizer_class: Optimizer,
        **optimizer_class_kwargs,
    ):
        self.arg_lr = optimizer_class_kwargs["lr"]
        if not torch.distributed.is_initialized():
            self.local_optim_group = optimizer_class(params, **optimizer_class_kwargs)
            self.local_rank = 0
        else:
            global_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
            self.global_rank = global_rank
            self.local_devices = min(int(os.getenv("SLURM_NTASKS_PER_NODE", torch.cuda.device_count())), world_size)
            self.local_rank = int(os.getenv("LOCAL_RANK", global_rank % self.local_devices))
            # Form a local process group
            node_rank = global_rank // self.local_devices
            local_ranks = [rank for rank in range(world_size) if rank // self.local_devices == node_rank]
            self.local_rank_zero_in_global_rank = local_ranks[0]
            if self.verbose:
                print(f"ZERO: Local ranks on {global_rank}: {local_ranks}", flush=True)
            self.local_pg = torch.distributed.new_group(ranks=local_ranks)
            # Assign parameters
            assert self.local_devices > 1
            assert len(params[0]["params"]) > self.local_devices
            assert self.local_rank_zero_in_global_rank + self.local_rank == global_rank
            local_param_groups = []
            if self.local_rank == 0:
                local_param_groups.append(params[1])  # embeddings
                if self.verbose:
                    print(f"ZERO: Placing {len(params[1]['params'])} embedding params on {self.local_rank}")
            else:
                selected_ints = list(range(len(params[0]["params"])))[self.local_rank - 1 :: self.local_devices - 1]
                if self.verbose:
                    print(f"ZERO: Placing weights {selected_ints} on {self.local_rank}")
                local_params = params[0].copy()
                local_params["params"] = params[0]["params"][self.local_rank - 1 :: self.local_devices - 1]
                local_param_groups.append(local_params)
            local_param_groups.extend(params[2:])  # append all higher groups (scalers) to all local ranks

            self.local_optim_group = optimizer_class(local_param_groups, **optimizer_class_kwargs)
            torch.distributed.barrier()

        super().__init__(params, {})  # this way, super covers all (non-sharded parameters!)
        if self.verbose:
            print(f"ZERO: Optimizer initialized on local_rank {self.local_rank}.")

    @torch.no_grad()
    def step(self, closure=None):
        """Step only on local parameters"""
        self.local_optim_group.step(closure)
        self.sync_parameters()

    @torch.no_grad()
    def sync_parameters(self):
        if torch.distributed.is_initialized():
            # Sync embeddings from rank 0
            for param in self.param_groups[1]["params"]:  # group 1 (embeddings)
                torch.distributed.broadcast(param.data, src=self.local_rank_zero_in_global_rank, group=self.local_pg)

            # Sync weights - each rank broadcasts its chunk
            weight_params = self.param_groups[0]["params"]
            for i in range(self.local_devices - 1):
                for param in weight_params[i :: self.local_devices - 1]:
                    torch.distributed.broadcast(
                        param.data, src=self.local_rank_zero_in_global_rank + i + 1, group=self.local_pg
                    )

    # def zero_grad(self, set_to_none=True):
    """zero_grad automatically executes against all parameters (which is necessary as grads are not sharded)"""

    def state_dict(self, cpu_before_gather=True):
        """Returns the state of the optimizer. Only rank 0 will have the complete state.
        Would it be safer if this didn't even execute on all of the other nodes?"""
        if not torch.distributed.is_initialized():
            return [self._cpu_state_dict(self.local_optim_group.state_dict())]

        if not self.single_rank_return:
            local_state = self.local_optim_group.state_dict()
            return [local_state]
        else:
            torch.cuda.empty_cache()  # thanks HIP
            # Get local state and move to CPU before communication
            local_state = self.local_optim_group.state_dict()
            if cpu_before_gather:
                local_state = self._cpu_state_dict(local_state)
            output_states = [None] * self.local_devices if self.local_rank == 0 else None
            torch.distributed.gather_object(
                local_state, output_states, dst=self.local_rank_zero_in_global_rank, group=self.local_pg
            )
            if not cpu_before_gather:
                output_states = [self._cpu_state_dict(state) for state in output_states]

            return output_states if self.global_rank == 0 else [{"state": {}, "param_groups": []}]

    def load_state_dict(self, state_dict):
        """Each rank loads its relevant parts from the checkpoint."""
        if not self.single_rank_return:
            local_state = state_dict[0]
            self.local_optim_group.load_state_dict(local_state)
        else:
            local_state = state_dict[self.local_rank]
            self.local_optim_group.load_state_dict(local_state)

    def _cpu_state_dict(self, state_dict):
        """Helper to move optimizer state dict to CPU"""
        cpu_state = {}
        for k, v in state_dict.items():
            if k == "state":
                cpu_state[k] = {
                    param_id: {name: val.cpu() if torch.is_tensor(val) else val for name, val in param_state.items()}
                    for param_id, param_state in v.items()
                }
            else:
                cpu_state[k] = v
        return cpu_state

    def __repr__(self):
        return self.__class__.__name__ + self.local_optim_group.__repr__()

    # def __getattr__(self, name):
    #     """Call this only if all other attributes are exhausted."""
    #     return getattr(self.local_optim_group, name)


class LionW(Optimizer):
    """
    Adapted from https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    and further modified from https://github.com/allenai/OLMo/blob/829f1d69d001b67a8a9845cc75c9a5edc8432d29/olmo/optim.py
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        **kwargs,
    ):
        assert lr > 0.0
        assert all(0.0 <= beta <= 1.0 for beta in betas)
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]
        self._update_total_dot_prod: Optional[torch.Tensor] = None
        self._update_total_norm: Optional[torch.Tensor] = None
        self._signed_update_total_norm: Optional[torch.Tensor] = None

    def get_post_step_metrics(
        self, module: torch.nn.Module, process_group: Optional[dist.ProcessGroup] = None
    ) -> Dict[str, torch.Tensor]:
        update_total_dot_prod = self._update_total_dot_prod
        update_total_norm = self._update_total_norm
        signed_update_total_norm = self._signed_update_total_norm
        if update_total_dot_prod is None or update_total_norm is None or signed_update_total_norm is None:
            return {}

        # if is_distributed() and isinstance(module, FullyShardedDataParallel):
        #     # Reduce total dot prod and norms across all ranks.
        #     update_total_norm = update_total_norm**2.0
        #     signed_update_total_norm = signed_update_total_norm**2.0
        #     # Reduce all together to avoid multiple communication calls.
        #     all_together = torch.stack([update_total_dot_prod, update_total_norm, signed_update_total_norm])
        #     # Only need the final result on rank0, since that's where we log from.
        #     dist.reduce(
        #         all_together,
        #         0 if process_group is None else dist.get_global_rank(process_group, 0),
        #         group=process_group,
        #     )
        #     update_total_dot_prod, update_total_norm, signed_update_total_norm = all_together
        #     update_total_norm = update_total_norm**0.5
        #     signed_update_total_norm = signed_update_total_norm**0.5

        update_cos_sim = update_total_dot_prod / torch.max(
            update_total_norm * signed_update_total_norm, torch.tensor(1e-8, device=torch.device("cuda"))
        )
        return {"update_cos_sim": update_cos_sim}

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        update_total_dot_prod = torch.tensor(0.0, dtype=torch.float32)
        update_norms = []
        signed_update_norms = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform step weight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                signed_update = torch.sign(update)
                p.add_(signed_update, alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                # Track dot product and norms of update vs signed update in order to calculate
                # their cosine similarity.
                update_total_dot_prod = update_total_dot_prod.to(update.device)
                update_total_dot_prod += torch.tensordot(update, signed_update, dims=len(update.shape))
                update_norms.append(torch.linalg.vector_norm(update, 2.0, dtype=torch.float32))
                signed_update_norms.append(torch.linalg.vector_norm(signed_update, 2.0, dtype=torch.float32))

        # Compute cosine similarity between update and signed update.
        self._update_total_dot_prod = update_total_dot_prod.to(torch.device("cuda"))
        self._update_total_norm = torch.linalg.vector_norm(
            torch.stack(update_norms),
            2.0,
            dtype=torch.float32,
        ).to(torch.device("cuda"))
        self._signed_update_total_norm = torch.linalg.vector_norm(
            torch.stack(signed_update_norms),
            2.0,
            dtype=torch.float32,
        ).to(torch.device("cuda"))


"""
# Sophia would require this training loop
for epoch in range(epochs):
    for X, Y in data_loader:
        # standard training code
        logits, loss = model(X, Y)
        loss.backward()
        optimizer.step(bs=bs)
        optimizer.zero_grad(set_to_none=True)
        iter_num += 1

        if iter_num % k != k - 1:
            continue
        else:
            # update hessian EMA
            logits, _ = model(X, None)
            samp_dist = torch.distributions.Categorical(logits=logits)
            y_sample = samp_dist.sample()
            loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
            loss_sampled.backward()
            optimizer.update_hessian()
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad()
"""
"""IVON would need
+optimizer = ivon.IVON(model.parameters(), lr=0.1, ess=len(train_dataset))

for X, y in train_loader:

+    for _ in range(train_samples):
+       with optimizer.sampled_params(train=True)

to be incorporated
"""


# stolen from https://github.com/Liuhong99/Sophia/blob/main/sophia.py
class SophiaG(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.965, 0.99),
        rho=0.04,
        weight_decay=1e-1,
        *,
        maximize: bool = False,
        capturable: bool = False,
        **kwargs,
    ):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not rho >= 0.0:
            raise ValueError("Invalid rho parameter at index 1: {}".format(rho))
        if not weight_decay >= 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, maximize=maximize, capturable=capturable
        )
        super(SophiaG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]["step"])
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError("Hero does not support sparse gradients")
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                state_steps.append(state["step"])
                hessian.append(state["hessian"])

                if self.defaults["capturable"]:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(
                params_with_grad,
                grads,
                exp_avgs,
                hessian,
                state_steps,
                bs=bs,
                beta1=beta1,
                beta2=beta2,
                rho=group["rho"],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
                capturable=group["capturable"],
            )

        return loss


def sophiag(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    hessian: List[Tensor],
    state_steps: List[Tensor],
    capturable: bool = False,
    *,
    bs: int,
    beta1: float,
    beta2: float,
    rho: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
    **kwargs,
):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    func = _single_tensor_sophiag

    func(
        params,
        grads,
        exp_avgs,
        hessian,
        state_steps,
        bs=bs,
        beta1=beta1,
        beta2=beta2,
        rho=rho,
        lr=lr,
        weight_decay=weight_decay,
        maximize=maximize,
        capturable=capturable,
    )


def _single_tensor_sophiag(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    hessian: List[Tensor],
    state_steps: List[Tensor],
    *,
    bs: int,
    beta1: float,
    beta2: float,
    rho: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
    capturable: bool,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if capturable:
            step_size = lr
            step_size_neg = step_size.neg()

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        else:
            step_size_neg = -lr

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)


# stolen from https://github.com/euclaise/supertrainer2000/blob/master/src/supertrainer2k/optim/lilith.py
class Lilith(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        eps: float = 1e-8,
        beta1_m: float = 0.9,
        beta2_m: float = 0.99,
        beta_v: float = 0.999,
        weight_decay: float = 0.01,
        g_norm_min: float = 1e-4,
        ratio_min: float = 1e-4,
        acceleration: float = 1,
        ema_k: int = 0,
        ema_beta: float = 0.99,
        **kwargs,
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            beta1_m=beta1_m,
            beta2_m=beta2_m,
            beta_v=beta_v,
            weight_decay=weight_decay,
            g_norm_min=g_norm_min,
            ratio_min=ratio_min,
            acceleration=acceleration,
            ema_k=ema_k,
            ema_beta=ema_beta,
        )

        super(Lilith, self).__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m_avg1"] = torch.zeros_like(grad)
                    state["m_avg2"] = torch.zeros_like(grad)
                    state["v_avg"] = torch.zeros_like(grad)
                    if group["ema_k"] > 0:
                        state["ema"] = p.data.clone()

                state["step"] += 1

                if sum(grad.shape) > 1:
                    trust_ratio = (p.data.norm() / grad.norm().clip(min=group["g_norm_min"])).clip(
                        min=group["ratio_min"]
                    )
                    grad.mul_(trust_ratio)

                m_avg1_prev = state["m_avg1"].clone()
                state["m_avg1"].add_(state["m_avg2"]).lerp_(grad, 1 - group["beta1_m"])
                state["m_avg2"].lerp_(state["m_avg1"] - m_avg1_prev, 1 - group["beta2_m"])

                u = state["m_avg1"] + group["acceleration"] * state["m_avg2"]

                state["v_avg"].lerp_(u.square(), 1 - group["beta_v"])
                v_avg = state["v_avg"] / (1 - group["beta_v"] ** state["step"])

                u.div_((v_avg + group["eps"]).sqrt())

                u.add_(p, alpha=group["weight_decay"])

                p.data.add_(u, alpha=-group["lr"])

                if group["ema_k"] != 0:
                    state["ema"].lerp_(p.data, 1 - group["ema_beta"])
                    if state["step"] % group["ema_k"] == 0:
                        p.data.copy_(state["ema"])

        return loss


def _parse_str_to_dtype(string_rep: str):
    if "bf16" in string_rep:
        return torch.bfloat16
    elif "f16" in string_rep or "fp16" in string_rep:
        return torch.float16
    else:
        return torch.float32


# an apple cobbler of many sources
class ELLISAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-6,
        weight_decay: float = 1e-2,
        *,
        foreach: Optional[bool] = None,
        nesterov: bool = False,
        eps_adjustment: bool = False,
        update_clipping: bool = False,
        kahan_sum_compensation: bool = False,
        buffer_dtype: Optional[Union[torch.dtype, str]] = None,  # can be torch.float16 / torch.bfloat16
        running_init: bool = False,
        tensor_wise_finite_check: bool = False,
        tensor_wise_gradient_normalization: bool = False,
        adafactor_like_beta_corrections: bool = False,
        atan_adam: bool = False,
        decouple_wd: bool = True,
        brute_force_clip: Optional[float] = None,
        poly_ema_p: Optional[float] = None,
    ):
        defaults = dict(
            lr=torch.tensor(lr, dtype=torch.float32),
            init_lr=copy.deepcopy(lr),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            nesterov=nesterov,
            eps_adjustment=eps_adjustment,
            update_clipping=update_clipping,
            kahan_sum_compensation=kahan_sum_compensation,
            buffer_dtype=_parse_str_to_dtype(buffer_dtype) if isinstance(buffer_dtype, str) else buffer_dtype,
            running_init=running_init,
            tensor_wise_finite_check=tensor_wise_finite_check,
            tensor_wise_gradient_normalization=tensor_wise_gradient_normalization,
            adafactor_like_beta_corrections=adafactor_like_beta_corrections,
            atan_adam=atan_adam,
            decouple_wd=decouple_wd,
            brute_force_clip=brute_force_clip,
            poly_ema_p=poly_ema_p,
        )
        self.arg_lr = lr
        if foreach:
            raise ValueError("Todo: reinstate a foreach version, minimizing additional mem alloc")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=torch.float32)

    @torch.no_grad()
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        kahan_comps,
        running_init: bool = False,
        buffer_dtype=None,
        kahan_sum_compensation: bool = False,
        tensor_wise_gradient_normalization: bool = False,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                _tensor_constructors = dict(memory_format=torch.preserve_format)
                if buffer_dtype is not None:
                    _tensor_constructors["dtype"] = buffer_dtype

                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0, dtype=torch.long)

                if kahan_sum_compensation:
                    state["kahan_comps"] = torch.zeros_like(p, **_tensor_constructors)
                else:
                    state["kahan_comps"] = None
                if running_init:
                    grad = p.grad if not tensor_wise_gradient_normalization else p.grad / p.grad.norm()
                    # Exponential moving average of gradient values
                    state["exp_avg"] = grad.clone().to(**_tensor_constructors)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = grad.pow(2).clone().to(**_tensor_constructors)
                else:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, **_tensor_constructors)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, **_tensor_constructors)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])
            kahan_comps.append(state["kahan_comps"])

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            kahan_comps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                kahan_comps,
                running_init=group["running_init"],
                kahan_sum_compensation=group["kahan_sum_compensation"],
                buffer_dtype=group["buffer_dtype"],
                tensor_wise_gradient_normalization=group["tensor_wise_gradient_normalization"],
            )
            _single_tensor_modded_adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                kahan_comps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                init_lr=group["init_lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                nesterov=group["nesterov"],
                eps_adjustment=group["eps_adjustment"],
                update_clipping=group["update_clipping"],
                kahan_sum_compensation=group["kahan_sum_compensation"],
                buffer_dtype=group["buffer_dtype"],
                tensor_wise_finite_check=group["tensor_wise_finite_check"],
                tensor_wise_gradient_normalization=group["tensor_wise_gradient_normalization"],
                adafactor_like_beta_corrections=group["adafactor_like_beta_corrections"],
                atan_adam=group["atan_adam"],
                decouple_wd=group["decouple_wd"],
                brute_force_clip=group["brute_force_clip"],
                poly_ema_p=group["poly_ema_p"],
            )

        return loss


def _single_tensor_modded_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    kahan_comps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    init_lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    nesterov: bool = False,
    eps_adjustment: bool = False,
    update_clipping: bool = False,
    kahan_sum_compensation: bool = False,
    buffer_dtype=Optional[torch.dtype],
    tensor_wise_finite_check: bool = False,
    tensor_wise_gradient_normalization: bool = False,
    adafactor_like_beta_corrections: bool = False,
    atan_adam: bool = False,
    decouple_wd: bool = False,
    brute_force_clip: Optional[float] = None,
    poly_ema_p: Optional[float] = None,
):
    if adafactor_like_beta_corrections:
        # update group step
        step_t = state_steps[0]  # crime
        step_t += 1
        beta1 = (beta1**step_t - beta1) / (beta1**step_t - 1)
        beta2 = (beta2**step_t - beta2) / (beta2**step_t - 1)

    if poly_ema_p is not None:
        step_t = state_steps[0]  # crime
        # beta1 = step_t / (step_t + poly_ema_p)
        beta2 = step_t / (step_t + poly_ema_p)  # palm: 1 - step_t ** -0.8

    if nesterov:
        alpha = 2 * (1 - beta1) - (1 - beta1) ** 2  # only for nesterov to fuse the two lerps

    for i, param in enumerate(params):
        grad = grads[i].to(buffer_dtype)
        if tensor_wise_finite_check and (~torch.isfinite(grad)).sum() > 0:
            continue

        if tensor_wise_gradient_normalization:
            grad = grad / grad.norm()
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        kahan_comp = kahan_comps[i]

        # Decay the first and second moment running average coefficient
        if nesterov:
            # Only difference between NAdamW and AdamW in this implementation.
            # The official PyTorch implementation of NAdam uses a different algorithm.
            # We undo these ops later on, which could cause numerical issues but saves
            # us from having to make an extra copy of the gradients.
            exp_avg.lerp_(grad, alpha)
        else:
            exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step_size = lr.clone() if isinstance(lr, torch.Tensor) else lr

        if update_clipping:
            rms = grad.pow(2).div_(exp_avg_sq.clamp_(min=eps**2)).mean().sqrt()  # impl like optimi
            step_size = step_size / rms.clamp(min=1.0)

        if not adafactor_like_beta_corrections:
            step_t += 1
            bias_correction1 = 1 - beta1**step_t
            bias_correction2 = 1 - beta2**step_t
            bias_correction2_sqrt = sqrt(bias_correction2)

            step_size = step_size / bias_correction1
        else:
            bias_correction2 = 1.0
            bias_correction2_sqrt = 1.0

        # Actual adam step
        if kahan_sum_compensation:
            # Perform stepweight decay
            if decouple_wd:
                kahan_comp.mul_(1 - lr / init_lr * weight_decay)
            else:
                kahan_comp.mul_(1 - lr * weight_decay)
            if atan_adam:
                # a = b = 1
                kahan_comp.add_(torch.atan2(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt)), alpha=-step_size)
            elif eps_adjustment:
                kahan_comp.addcdiv_(exp_avg, exp_avg_sq.div(bias_correction2).add_(eps**2).sqrt(), value=-step_size)
            else:
                kahan_comp.addcdiv_(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps), value=-step_size)
            # update weights with kahan compensation using grad as temp buffer
            grad.copy_(param.detach())
            param.add_(kahan_comp)
            # save error back to kahan compensation for next iteration
            kahan_comp.add_(grad.sub_(param))
        else:
            # Perform stepweight decay
            if decouple_wd:
                param.mul_(1 - lr / init_lr * weight_decay)
            else:
                param.mul_(1 - lr * weight_decay)
            if atan_adam:
                update = torch.atan2(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt))
                if brute_force_clip is not None:
                    param.add_(update / torch.clamp(update.norm(), min=brute_force_clip), alpha=-step_size)
                else:
                    param.add_(update, alpha=-step_size)
            elif eps_adjustment:
                if brute_force_clip is not None:
                    update = exp_avg / exp_avg_sq.div(bias_correction2).add_(eps**2).sqrt()
                    param.add_(update / torch.clamp(update.norm(), min=brute_force_clip), alpha=-step_size)
                else:
                    param.addcdiv_(exp_avg, exp_avg_sq.div(bias_correction2).add_(eps**2).sqrt(), value=-step_size)
            else:
                if brute_force_clip is not None:
                    update = exp_avg / exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)
                    param.add_(update / torch.clamp(update.norm(), min=brute_force_clip), alpha=-step_size)
                else:
                    param.addcdiv_(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps), value=-step_size)

        # undo nadam
        if nesterov:
            exp_avg.lerp_(grad, 1 - 1 / beta1)


# from https://github.com/team-approx-bayes/ivon/blob/main/ivon/_ivon.py
ClosureType = Callable[[], Tensor]


def _welford_mean(avg: Optional[Tensor], newval: Tensor, count: int) -> Tensor:
    return newval if avg is None else avg + (newval - avg) / count


class IVON(Optimizer):
    hessian_approx_methods = (
        "price",
        "gradsq",
    )

    def __init__(
        self,
        params,
        lr: float,  # 0.2 suggested for GPT-2
        ess: float,  # better not set this too small, but can it be too large?
        hess_init: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 1 - 10**-5,
        weight_decay: float = 1e-6,  # from Appendix A
        mc_samples: int = 1,
        hess_approx: str = "price",
        clip_radius: float = 1e-3,  # clip a lot ? # float("inf")
        sync: bool = False,
        debias: bool = True,
        rescale_lr: bool = True,
    ):
        if not lr > 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not mc_samples >= 1:
            raise ValueError("Invalid number of MC samples: {}".format(mc_samples))
        if not weight_decay > 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if not hess_init > 0.0:
            raise ValueError("Invalid Hessian initialization: {}".format(hess_init))
        if not ess > 0.0:
            raise ValueError("Invalid effective sample size: {}".format(ess))
        if not clip_radius > 0.0:
            raise ValueError("Invalid clipping radius: {}".format(clip_radius))
        if not 0.0 < beta1 < 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if not 0.0 < beta2 < 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        if hess_approx not in self.hessian_approx_methods:
            raise ValueError("Invalid hess_approx parameter: {}".format(beta2))

        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            hess_init=hess_init,
            ess=ess,
            clip_radius=clip_radius,
        )
        super().__init__(params, defaults)

        self.mc_samples = mc_samples
        self.hess_approx = hess_approx
        self.sync = sync
        self._numel, self._device, self._dtype = self._get_param_configs()
        self.current_step = 0
        self.debias = debias
        self.rescale_lr = rescale_lr

        # set initial temporary running averages
        self._reset_samples()
        # init all states
        self._init_buffers()

    def _get_param_configs(self):
        all_params = []
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
            all_params += [p for p in pg["params"] if p is not None]
        if len(all_params) == 0:
            return 0, torch.device("cpu"), torch.get_default_dtype()
        devices = {p.device for p in all_params}
        if len(devices) > 1:
            raise ValueError(f"Parameters are on different devices: {[str(d) for d in devices]}")
        device = next(iter(devices))
        dtypes = {p.dtype for p in all_params}
        if len(dtypes) > 1:
            raise ValueError(f"Parameters are on different dtypes: {[str(d) for d in dtypes]}")
        dtype = next(iter(dtypes))
        total = sum(pg["numel"] for pg in self.param_groups)
        return total, device, dtype

    def _reset_samples(self):
        self.state["count"] = 0
        self.state["avg_grad"] = None
        self.state["avg_nxg"] = None
        self.state["avg_gsq"] = None

    def _init_buffers(self):
        for group in self.param_groups:
            hess_init, numel = group["hess_init"], group["numel"]

            group["momentum"] = torch.zeros(numel, device=self._device, dtype=self._dtype)
            group["hess"] = torch.zeros(numel, device=self._device, dtype=self._dtype).add(torch.as_tensor(hess_init))

    @contextmanager
    def sampled_params(self, train: bool = False):
        param_avg, noise = self._sample_params()
        yield
        self._restore_param_average(train, param_avg, noise)

    def _restore_param_average(self, train: bool, param_avg: Tensor, noise: Tensor):
        param_grads = []
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                p_slice = slice(offset, offset + p.numel())

                p.data = param_avg[p_slice].view(*p.shape)
                if train:
                    if p.requires_grad:
                        param_grads.append(p.grad.flatten())
                    else:
                        param_grads.append(torch.zeros_like(p).flatten())
                offset += p.numel()
        assert offset == self._numel  # sanity check

        if train:  # collect grad sample for training
            grad_sample = torch.cat(param_grads, 0)
            count = self.state["count"] + 1
            self.state["count"] = count
            self.state["avg_grad"] = _welford_mean(self.state["avg_grad"], grad_sample, count)
            if self.hess_approx == "price":
                self.state["avg_nxg"] = _welford_mean(self.state["avg_nxg"], noise * grad_sample, count)
            elif self.hess_approx == "gradsq":
                self.state["avg_gsq"] = _welford_mean(self.state["avg_gsq"], grad_sample.square(), count)

    @torch.no_grad()
    def step(self, closure: ClosureType = None) -> Optional[Tensor]:
        if closure is None:
            loss = None
        else:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        if self.sync and dist.is_initialized():  # explicit sync
            self._sync_samples()
        self._update()
        self._reset_samples()
        return loss

    def _sync_samples(self):
        world_size = dist.get_world_size()
        dist.all_reduce(self.state["avg_grad"])
        self.state["avg_grad"].div_(world_size)
        dist.all_reduce(self.state["avg_nxg"])
        self.state["avg_nxg"].div_(world_size)

    def _sample_params(self) -> Tuple[Tensor, Tensor]:
        noise_samples = []
        param_avgs = []

        offset = 0
        for group in self.param_groups:
            gnumel = group["numel"]
            noise_sample = (
                torch.randn(gnumel, device=self._device, dtype=self._dtype)
                / (group["ess"] * (group["hess"] + group["weight_decay"])).sqrt()
            )
            noise_samples.append(noise_sample)

            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue

                p_avg = p.data.flatten()
                numel = p.numel()
                p_noise = noise_sample[offset : offset + numel]

                param_avgs.append(p_avg)
                p.data = (p_avg + p_noise).view(*p.shape)
                goffset += numel
                offset += numel
            assert goffset == group["numel"]  # sanity check
        assert offset == self._numel  # sanity check

        return torch.cat(param_avgs, 0), torch.cat(noise_samples, 0)

    def _update(self):
        self.current_step += 1

        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["numel"])

            param_avg = torch.cat([p.flatten() for p in group["params"] if p is not None], 0)

            group["momentum"] = self._new_momentum(self.state["avg_grad"][pg_slice], group["momentum"], b1)

            group["hess"] = self._new_hess(
                self.hess_approx,
                group["hess"],
                self.state["avg_nxg"],
                self.state["avg_gsq"],
                pg_slice,
                group["ess"],
                b2,
                group["weight_decay"],
            )

            param_avg = self._new_param_averages(
                param_avg,
                group["hess"],
                group["momentum"],
                lr * (group["hess_init"] + group["weight_decay"]) if self.rescale_lr else lr,
                group["weight_decay"],
                group["clip_radius"],
                1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0,
                group["hess_init"],
            )

            # update params
            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    p.data = param_avg[pg_offset : pg_offset + p.numel()].view(*p.shape)
                    pg_offset += p.numel()
            assert pg_offset == group["numel"]  # sanity check
            offset += group["numel"]
        assert offset == self._numel  # sanity check

    @staticmethod
    def _get_nll_hess(method: str, hess, avg_nxg, avg_gsq, pg_slice) -> Tensor:
        if method == "price":
            return avg_nxg[pg_slice] * hess
        elif method == "gradsq":
            return avg_gsq[pg_slice]
        else:
            raise NotImplementedError(f"unknown hessian approx.: {method}")

    @staticmethod
    def _new_momentum(avg_grad, m, b1) -> Tensor:
        return b1 * m + (1.0 - b1) * avg_grad

    @staticmethod
    def _new_hess(method, hess, avg_nxg, avg_gsq, pg_slice, ess, beta2, wd) -> Tensor:
        f = IVON._get_nll_hess(method, hess + wd, avg_nxg, avg_gsq, pg_slice) * ess
        return beta2 * hess + (1.0 - beta2) * f + (0.5 * (1 - beta2) ** 2) * (hess - f).square() / (hess + wd)

    @staticmethod
    def _new_param_averages(param_avg, hess, momentum, lr, wd, clip_radius, debias, hess_init) -> Tensor:
        return param_avg - lr * torch.clip(
            (momentum / debias + wd * param_avg) / (hess + wd),
            min=-clip_radius,
            max=clip_radius,
        )


"a sane reimplementation of dist shampoo via https://github.com/cloneofsimo/zeroshampoo/tree/main"

import numpy as np
import torch
import torch.distributed as dist


class ZeroShampooWithAdamGraftingOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        shampoo_eps=1e-6,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        precondition_frequency=None,
        start_preconditioning=4,
        independent_weight_decay=True,
        weight_decay=0.001,
        device=None,
        dtype=None,
        block_size=128,
    ):
        if isinstance(params, (list, tuple)) and isinstance(params[0], dict):
            self.param_groups = params
        else:
            self.param_groups = [{"params": list(params)}]

        self.defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_eps=shampoo_eps,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            precondition_frequency=precondition_frequency or start_preconditioning,
            start_preconditioning=start_preconditioning,
            independent_weight_decay=independent_weight_decay,
            weight_decay=weight_decay,
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        self.state = {}
        self.block_size = block_size

        # Distributed training setup
        try:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.is_distributed = True
        except Exception:
            print("Distributed training not initialized, setting rank and world size to 0")
            self.rank = 0
            self.world_size = 1
            self.is_distributed = False

        self.param_stats = {}
        self._make_lookup_and_enumeratables()
        self._init_state()

    @torch.no_grad()
    def _make_lookup_and_enumeratables(self):
        self.lookup = {}
        self.enumeratables = []
        global_counter = 0
        total_params = 0

        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    name = "param"
                    s1, s2 = self._get_left_right_shape(param)
                    for i1 in range(0, s1, self.block_size):
                        i1r = min(i1 + self.block_size, s1)
                        for i2 in range(0, s2, self.block_size):
                            i2r = min(i2 + self.block_size, s2)
                            block_name = f"{name}_{global_counter}_{i1}_{i1r}_{i2}_{i2r}"
                            self.enumeratables.append(
                                (
                                    global_counter,
                                    block_name,
                                    param,
                                    (s1, s2),
                                    (i1, i1r),
                                    (i2, i2r),
                                    group,
                                )
                            )
                            total_params += (i1r - i1) * (i2r - i2)
                            if param not in self.param_stats:
                                self.param_stats[param] = []

                            self.param_stats[param].append((i1, i1r, i2, i2r, s1, s2, block_name))

                    global_counter += 1

            # make default
            for k, v in self.defaults.items():
                group[k] = v

        total_param_in_model = 0
        for group in self.param_groups:
            for param in group["params"]:
                total_param_in_model += param.numel()

        assert total_params == total_param_in_model, f"Total params: {total_params} != {total_param_in_model}"

    def _enumerate_sharded_params(self):
        for (
            global_counter,
            block_name,
            param,
            (s1, s2),
            (i1, i1r),
            (i2, i2r),
            group,
        ) in self.enumeratables:
            if global_counter % self.world_size != self.rank:
                continue
            yield block_name, param, (s1, s2), (i1, i1r), (i2, i2r), group

    def _init_state(self):
        for (
            block_name,
            param,
            (s1, s2),
            (i1, i1r),
            (i2, i2r),
            group,
        ) in self._enumerate_sharded_params():
            block_param = param.view(s1, s2)[i1:i1r, i2:i2r]
            print(
                f"Rank {self.rank} is managing parameter {block_name}, shape: {block_param.shape}, dtype: {block_param.dtype}, range {i1}:{i1r}, {i2}:{i2r}"
            )
            assert self.state.get(block_name, None) is None, f"State for {block_name} already exists"
            self.state[block_name] = {}
            state = self.state[block_name]
            state["step"] = 0
            state["m_adam"] = torch.zeros_like(block_param, device=self.device, dtype=self.dtype)
            state["v_adam"] = torch.zeros_like(block_param, device=self.device, dtype=self.dtype)
            state["left_preconditioner_accum"] = group["shampoo_eps"] * torch.eye(
                i1r - i1, device=self.device, dtype=self.dtype
            )
            state["right_preconditioner_accum"] = group["shampoo_eps"] * torch.eye(
                i2r - i2, device=self.device, dtype=self.dtype
            )
            state["left_preconditioner"] = None
            state["right_preconditioner"] = None

    def _get_left_right_shape(self, param):
        if param.ndim == 1:
            return (param.shape[0], 1)
        else:
            return (np.prod(param.shape[:-1]), param.shape[-1])

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.detach_()
                        param.grad.zero_()

    @torch.no_grad()
    def step(self):
        # self._reduce_gradients() # assume this is handled in fsdp/ddp/axonn

        for (
            block_name,
            param,
            (s1, s2),
            (i1, i1r),
            (i2, i2r),
            group,
        ) in self._enumerate_sharded_params():
            grad = param.grad
            assert grad is not None, f"Gradient is None for {block_name}"
            state = self.state[block_name]

            block_param = param.view(s1, s2)[i1:i1r, i2:i2r]
            block_grad = grad.view(s1, s2)[i1:i1r, i2:i2r]

            assert block_param.shape == block_grad.shape, (
                block_param.shape,
                block_grad.shape,
            )

            left_shape, right_shape = block_param.shape

            # Update step count
            state["step"] += 1

            # Get group-specific hyperparameters
            lr = group["lr"]

            weight_decay = group["weight_decay"]
            independent_weight_decay = group["independent_weight_decay"]
            shampoo_beta1, shampoo_beta2 = group["betas"]
            adam_beta1, adam_beta2 = group["adam_betas"]
            adam_eps = group["adam_eps"]
            start_preconditioning = group["start_preconditioning"]
            precondition_frequency = group["precondition_frequency"]

            # Perform stepweight decay
            if independent_weight_decay:
                block_param.data.mul_(1 - lr * weight_decay)

            # Update preconditioners
            state["left_preconditioner_accum"].mul_(shampoo_beta1).add_(
                block_grad @ block_grad.t(), alpha=1 - shampoo_beta1
            )
            state["right_preconditioner_accum"].mul_(shampoo_beta1).add_(
                block_grad.t() @ block_grad, alpha=1 - shampoo_beta1
            )

            # Update Adam state
            state["m_adam"].mul_(adam_beta1).add_(block_grad, alpha=1 - adam_beta1)
            state["v_adam"].mul_(adam_beta2).addcmul_(block_grad, block_grad, value=1 - adam_beta2)

            m_hat = state["m_adam"] / (1 - adam_beta1 ** state["step"])
            v_hat = state["v_adam"] / (1 - adam_beta2 ** state["step"])
            adam_update_dir = m_hat / (torch.sqrt(v_hat) + adam_eps)

            if state["step"] >= start_preconditioning:
                if state["step"] % precondition_frequency == 0:
                    state["left_preconditioner"] = self._matrix_pth_power_via_eigendecompsition(
                        state["left_preconditioner_accum"], p=-1 / 4
                    )
                    state["right_preconditioner"] = self._matrix_pth_power_via_eigendecompsition(
                        state["right_preconditioner_accum"], p=-1 / 4
                    )

                fnorm_of_adam_update_dir = torch.linalg.norm(adam_update_dir)
                grad_momentum = state["m_adam"]

                shampoo_update_dir = state["left_preconditioner"] @ grad_momentum @ state["right_preconditioner"]

                fnorm_of_shampoo_update_dir = torch.linalg.norm(shampoo_update_dir)

                update_dir = fnorm_of_adam_update_dir * shampoo_update_dir / fnorm_of_shampoo_update_dir
            else:
                update_dir = adam_update_dir

            assert update_dir.shape == block_param.shape
            assert update_dir.shape == block_grad.shape

            param.view(s1, s2)[i1:i1r, i2:i2r].data.add_(update_dir, alpha=-lr)

        self._sync_params()

    def _check_momentum_and_variance(self):
        num_total_params = 0
        # iterate over all params
        for group in self.param_groups:
            for param in group["params"]:
                num_total_params += param.numel()

        num_non_zero_params = 0
        for (
            block_name,
            param,
            (s1, s2),
            (i1, i1r),
            (i2, i2r),
            group,
        ) in self._enumerate_sharded_params():
            state = self.state[block_name]
            # check if the values are very-close to non-zero or not
            assert not torch.allclose(state["m_adam"], torch.zeros_like(state["m_adam"]), atol=1e-8), (
                f"Momentum is zero for {block_name}: average var: {state['m_adam'].abs().mean()}, state: {state['m_adam']}"
            )
            assert not torch.allclose(state["v_adam"], torch.zeros_like(state["v_adam"]), atol=1e-8), (
                f"Variance is zero for {block_name}: average var: {state['v_adam'].abs().mean()}, state: {state['v_adam']}"
            )
            num_non_zero_params += (i1r - i1) * (i2r - i2)

        assert num_non_zero_params == num_total_params, (
            f"Num non-zero params: {num_non_zero_params} != {num_total_params}"
        )
        print("All momentum and variance are non-zero")

    def build_global_state_for_debug_purposes(self, device=None):
        constructors = {} if device is None else {"device": device}

        global_state = {}
        for (
            global_counter,
            block_name,
            param,
            (s1, s2),
            (i1, i1r),
            (i2, i2r),
            group,
        ) in self.enumeratables:
            if global_counter % self.world_size != self.rank:
                continue

            if param not in global_state:
                global_state[param] = {}
            # make exp_avg, exp_avg_sq
            if "exp_avg" not in global_state[param]:
                global_state[param]["exp_avg"] = torch.ones_like(param.data, **constructors).view(s1, s2)
            if "exp_avg_sq" not in global_state[param]:
                global_state[param]["exp_avg_sq"] = torch.ones_like(param.data, **constructors).view(s1, s2)

            # print(f"Doing {block_name}, {i1}:{i1r}, {i2}:{i2r}")
            assert self.state[block_name]["m_adam"].shape == (i1r - i1, i2r - i2)
            # fill in
            global_state[param]["exp_avg"][i1:i1r, i2:i2r] = self.state[block_name]["m_adam"].to(**constructors)
            global_state[param]["exp_avg_sq"][i1:i1r, i2:i2r] = self.state[block_name]["v_adam"].to(**constructors)
        return global_state

    @torch.no_grad()
    def _matrix_pth_power_via_eigendecompsition(self, mat, p=-1 / 4):
        try:
            eigvals, eigvecs = torch.linalg.eigh(mat)
        except Exception:
            print("RuntimeError in _matrix_pth_power_via_eigendecompsition")
            print("mat", mat)
            print("p", p)
            print("trace", mat.trace().item())
            print("rank", self.rank)

            raise

        mineig = min(eigvals.min().item(), 0)

        eigvals = eigvals - mineig + 1e-8
        eigvals = eigvals**p

        return eigvecs @ torch.diag(eigvals) @ eigvecs.t()

    @torch.no_grad()
    def _sync_params(self):
        if not self.is_distributed:
            return
        did_broadcast_list = set()
        for (
            global_counter,
            block_name,
            param,
            (s1, s2),
            (i1, i1r),
            (
                i2,
                i2r,
            ),
            group,
        ) in self.enumeratables:
            if global_counter in did_broadcast_list:
                continue

            if global_counter % self.world_size == self.rank:
                dist.broadcast(param.data, src=self.rank)
            else:
                dist.broadcast(param.data, src=global_counter % self.world_size)

            did_broadcast_list.add(global_counter)

    @torch.no_grad()
    def _reduce_gradients(self):
        if not self.is_distributed:
            return
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

    @torch._disable_dynamo
    @torch.no_grad
    def state_dict(self):
        r"""Return the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * ``state``: a Dict holding current optimization state. Its content
            differs between optimizer classes, but some common characteristics
            hold. For example, state is saved per parameter, and the parameter
            itself is NOT saved. ``state`` is a Dictionary mapping parameter ids
            to a Dict with state corresponding to each parameter.
        * ``param_groups``: a List containing all parameter groups where each
            parameter group is a Dict. Each parameter group contains metadata
            specific to the optimizer, such as learning rate and weight decay,
            as well as a List of parameter IDs of the parameters in the group.

        NOTE: The parameter IDs may look like indices but they are just IDs
        associating state with param_group. When loading from a state_dict,
        the optimizer will zip the param_group ``params`` (int IDs) and the
        optimizer ``param_groups`` (actual ``nn.Parameter`` s) in order to
        match state WITHOUT additional verification.

        A returned state dict might look something like:

        .. code-block:: text

            {
                'state': {
                    0: {'momentum_buffer': tensor(...), ...},
                    1: {'momentum_buffer': tensor(...), ...},
                    2: {'momentum_buffer': tensor(...), ...},
                    3: {'momentum_buffer': tensor(...), ...}
                },
                'param_groups': [
                    {
                        'lr': 0.01,
                        'weight_decay': 0,
                        ...
                        'params': [0]
                    },
                    {
                        'lr': 0.001,
                        'weight_decay': 0.5,
                        ...
                        'params': [1, 2, 3]
                    }
                ]
            }

        """

        # Save order indices instead of Tensors
        param_mappings: Dict[int, int] = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "params"}
            param_mappings.update(
                {id(p): i for i, p in enumerate(group["params"], start_index) if id(p) not in param_mappings}
            )
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys

        global_state = self.build_global_state_for_debug_purposes(device=torch.device("cpu"))
        packed_state = {
            (param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v for k, v in global_state.items()
        }

        state_dict = {
            "state": packed_state,
            "param_groups": param_groups,
        }

        return state_dict


# -----------------------------------------------------------------------------
# OrthgonalNesterov optimizer


class OrthogonalNesterov(torch.optim.Optimizer):
    """
    Some warnings: This optimizer assumes that all parameters passed in are 2D.
    It shouldn't be used for the embedding layer, the final fully connected layer, or {0,1}-D
    parameters; those should be optimized by a standard method (e.g., AdamW).
    To use it with 4D convolutional filters, it works well to flatten their last 3 dimensions.
    """

    def __init__(
        self,
        params,
        lr=4e-4,
        momentum=0.95,
        nesterov=True,
        zeropower_iters=5,
        eps=1e-5,
        betas=[0.9, 0.95],
        weight_decay=1e-5,
        vocab_dim=32768,  # hack for now
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            zeropower_iters=zeropower_iters,
            eps=eps,
            betas=betas,
            weight_decay=weight_decay,
            vocab_dim=vocab_dim,
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                state = self.state[p]
                g = p.grad
                if g is None:
                    continue
                if p.ndim == 2 and p.shape[0] != group["vocab_dim"] and p.shape[1] != group["vocab_dim"]:
                    # Newton-Schulz mode
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                    update = zeroth_power_via_newtonschulz5(g, steps=group["zeropower_iters"])
                    scale = update.numel() ** 0.5 / update.norm()
                    p.data.add_(update, alpha=-lr * 0.1 * scale)
                else:
                    # adam mode
                    if "step" not in state:
                        state["step"] = 0  # torch.tensor(0.0, dtype=torch.float32)
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    # update step
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    # wd
                    p.mul_(1 - lr * group["weight_decay"])

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]

                    step_size = lr / bias_correction1

                    bias_correction2_sqrt = bias_correction2**0.5
                    denom = (state["exp_avg"].sqrt() / bias_correction2_sqrt).add_(group["eps"])
                    p.addcdiv_(state["exp_avg"], denom, value=-step_size)


@torch.compile
def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. It turns out
    to be empirically effective to keep increasing the slope of the quintic at zero even beyond the
    point where it no longer converges to one everywhere after repeated application (so long as it
    stays relatively close to 1 across the interval). Our usage of a Newton-Schulz iteration as the
    orthogonalization method traces to Bernstein & Newhouse (2024) https://arxiv.org/abs/2409.20325
    who suggested its use for computing the preconditioners of Shampoo.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)
