"""Try to keep all inits in this single file for our sanity.

The param_init_fn should be callable by fsdp or by later module.apply()
and replace all meta tensors with actually initialized tensors on the correct devices,
supporting tensor-parallel ...

Could also support workarounds like
* https://github.com/pytorch/pytorch/issues/105840
* https://github.com/pytorch/pytorch/blob/bffcfa9628d4c8e858ef5f2aeab34e021885e682/torch/distributed/fsdp/api.py#L280
where only rank0 devices actually initialize CPU parameters, and the others are faking it,
or rely on preexisting, or provided globals to initialize directly onto the accelerators

Contract:
* module.layer_id needs to be provided if layer_id dependent inits are to be used
* non-standard options need to be wrapped as partial(param_init_fn, *options) before applying the fn/passing to fsdp
"""

import torch
import math
from math import sqrt

from typing import Optional


# Init lookup from trivialname of init -> prescription.
# if no prescription for ["embedding", "normalization", "attention", "mlp", "head"] is given, then it is taken from default
def get_factor_table(dim, intermed_dim, attn_head_dim, layer_idx=0, num_layers=16):
    """A bit weird to have this as a fn that just defines the dict, but I like the compact summary that this setup provides."""
    layer_idx = layer_idx % num_layers
    lookup = {
        "mitchell": {
            "embedding": 1 / sqrt(dim),  # or 1
            "head": 1 / sqrt(dim),
            "in_proj": 1 / sqrt(dim),
            "out_proj": 1 / sqrt(dim) / sqrt(2 * (layer_idx + 1)),
        },  # from Zhang-Titov-Sennrich
        "normal": {"std": 1 / sqrt(dim)},
        "llama": {
            "embedding": 1.0,  # even without truncation, just a normal normal_
            "in_proj": 0.02,
            "out_proj": 0.02 / sqrt(2 * (layer_idx + 1)),
            "head": 1 / sqrt(dim),
        },  # apply in_proj definitely per q,k,v # small variation of zhang-titov-sennrich
        "llama-by-dim": {
            "embedding": 1.0,  # even without truncation, just a normal normal_
            "in_proj": 1 / sqrt(dim),
            "out_proj": 1 / sqrt(dim) / sqrt(2 * (layer_idx + 1)),
            "head": 1 / sqrt(dim),
        },  # apply in_proj definitely per q,k,v # small variation of zhang-titov-sennrich
        "llama-by-dim-ls": {
            "embedding": 1.0,  # even without truncation in the original, just a normal normal_
            "in_proj": 1 / sqrt(dim),
            "out_proj": 1 / sqrt(dim) / sqrt(2 * (layer_idx + 1)),
            "head": 1.0,
            "logit_scale": 1 / sqrt(dim),
        },
        # the ffn gate (w2) also counts as out_proj (but not in "olmo"/mitchell)
        "kaiming": {"std": sqrt(2.0 / dim)},  # need to account for intermed_dim on ffn out
        "bert": {"std": 0.02},
        "megatron": {"std": 0.02, "out_proj": 0.02 / sqrt(dim), "embedding": 0.02, "head": 1 / sqrt(dim)},
        "megatron2": {"std": sqrt(1 / (3 * dim))},
        "small": {"std": sqrt(2 / (5 * dim))},  # nguyen & salazar
        "scaled": {
            "std": sqrt(2 / (5 * dim)),
            "out_proj": sqrt(2 / (5 * dim)) / sqrt(2 * num_layers),
        },  # Le Scao, Biderman,
        "scaled-stuck": {
            "std": sqrt(2 / (5 * dim)),
            "out_proj": sqrt(2 / (5 * dim)) / sqrt(2 * 16),
        },  # Le Scao, Biderman,
        "takase": {
            "std": sqrt(2 / (5 * dim)),
            "out_proj": sqrt(2 / (5 * dim)) / sqrt(2 * num_layers),
            "embedding": sqrt(2 / (5 * dim)),
            "embed_scale": sqrt(dim),
            # "logit_scale": sqrt(2 / 5) / sqrt(dim),  # if weight-tied
        },  # spike-no-more, Takase et al.
        "takase-scaled": {
            "std": sqrt(2 / (5 * dim)),
            "out_proj": sqrt(2 / (5 * dim)) / sqrt(2 * num_layers),
            "embedding": sqrt(2 / (5 * dim)),
            "embed_scale": sqrt(dim),
            "logit_scale": sqrt(2 / 5) / sqrt(dim),  # if weight-tied
        },
        "wang": {"std": 2 / num_layers / sqrt(dim)},  # Wang& Komatsuzaki
        "deepnorm-straight": {
            "embedding": 0.02,  # undef in original, taken from megatron
            "gain": pow(8 * num_layers, -0.25),
            "skip": pow(2 * num_layers, 0.25),
            "mlp": _xavier_gain_to_std(pow(8 * num_layers, -0.25), dim, intermed_dim),
            "out_proj": _xavier_gain_to_std(pow(8 * num_layers, -0.25), dim, intermed_dim),
            "v": _xavier_gain_to_std(pow(8 * num_layers, -0.25), dim, dim),
            "out_attn": _xavier_gain_to_std(pow(8 * num_layers, -0.25), dim, dim),
            "q": _xavier_gain_to_std(1.0, dim, attn_head_dim),
            "k": _xavier_gain_to_std(1.0, dim, attn_head_dim),
            "head": 1 / sqrt(dim),  # undef in original, taken from megatron
        },
        "deepnorm-subln": {
            "embedding": 0.02,  # undef in original, taken from megatron
            "gain": sqrt(math.log(2 * num_layers)),
            "skip": pow(2 * num_layers, 0.25),
            "mlp": _xavier_gain_to_std(sqrt(math.log(2 * num_layers)), dim, intermed_dim),
            "v": _xavier_gain_to_std(sqrt(math.log(2 * num_layers)), dim, dim),
            "out_attn": _xavier_gain_to_std(sqrt(math.log(2 * num_layers)), dim, dim),
            "q": _xavier_gain_to_std(1.0, dim, attn_head_dim),
            "k": _xavier_gain_to_std(1.0, dim, attn_head_dim),
            "head": 1 / sqrt(dim),  # undef in original, taken from megatron
        },
        "noci-anagnostidis": {
            "residual": 1 / sqrt(num_layers),
            "mlp": _xavier_gain_to_std(1.0, dim, intermed_dim),
            "v": _xavier_gain_to_std(1.0, dim, dim),
            "out_attn": _xavier_gain_to_std(1.0, dim, dim),
            "q": _xavier_gain_to_std(1.0, dim, attn_head_dim),
            "k": _xavier_gain_to_std(1.0, dim, attn_head_dim),
            "std": 1.0,
            "logit_scale": 1 / sqrt(dim),
        },
        "shaped": {
            "residual": 0.2,  # gamma=0.1 from appendix, or around 0.2 from main?
            "skip": sqrt(1 - 0.2**2),  # needs to fulfill residual**2 + skip**2 = 1
            "in_proj": 1 / sqrt(dim),
            "out_proj": 1 / sqrt(dim),
            "std": 1.0,
            "q": 1 / sqrt(dim),
            "k": 1 / sqrt(dim),
            "logit_scale": 1 / sqrt(dim),
        },  # should go with identity-shaped activation functions
        "deep-scale-simple": {
            "residual": sqrt(2 / num_layers),
            "skip": sqrt(1 - 2 / num_layers),
            "std": sqrt(1 / dim * sqrt(1 / 2)),
            "q": sqrt(1 / dim),
            "k": sqrt(1 / dim),
            "embedding": sqrt(1 / 3),
            "head": 1.0,
            "logit_scale": 1 / sqrt(dim),  # mentioned for BERT in 5.1
        },
        "deep-scale-full": {
            "residual": sqrt(2 / num_layers),
            "skip": sqrt(1 - 2 / num_layers),
            "std": sqrt(1 / dim * sqrt(1 / 2)),
            "q": sqrt(1 / dim),
            "k": sqrt(1 / dim),
            "v": _get_deepscale_value_std(dim, num_layers, layer_idx),  # compare to sqrt(1/d * sqrt(sigma)) ?
            "out_attn": _get_deepscale_value_std(dim, num_layers, layer_idx),
            "embedding": sqrt(1 / 3),  # technically just one because we have only one embedding type
            "head": 1.0,
            "logit_scale": 1 / sqrt(dim),  # mentioned for BERT in 5.1
        },
        "scaled-and-logit-scale": {
            "std": sqrt(2 / (5 * dim)),
            "out_proj": sqrt(2 / (5 * dim)) / sqrt(2 * num_layers),
            "embedding": 1.0,
            "head": 1 / sqrt(dim),  # not used due to weight tying
            "logit_scale": 1 / sqrt(dim),
        },  # Le Scao, Biderman,
        "bernstein": {"std": 1.0},  # handled elsewhere  # a special in the twitter to code pipeline
        "illiterate": {
            "embedding": 1.0,
            "std": sqrt(1 / dim),
            "out_proj": 0.0,
            "head": 0.0,
        },
        "scaled-large-embed": {
            "embedding": 1.0,
            "std": sqrt(1 / dim),
            "out_proj": sqrt(1 / dim) / sqrt(2 * num_layers),
        },
    }
    return lookup


def _get_deepscale_value_std(dims, num_layers, layer_idx):
    def attn_block(r):
        r_out = 1 - p
        sigma_w1 = math.sqrt(math.sqrt((1 - p) / r) / dims)
        return sigma_w1, r_out

    def ffn_block(r):
        r_out = (1 - p) * (r + ((1 - r**2) ** 0.5 - r * math.acos(r)) / math.pi)
        return r_out

    p = 0.0  # no dropout
    lambda_sq = 1 - 2 / num_layers
    beta_sq = 2 / num_layers
    r0 = 0.221  # deus ex machina from Zipf
    sigma_new = 1.0
    sigma_attn_out_v_list = []
    r_in = r0 * (1 - p)

    r_list = []
    r_list.append(r_in)
    for _ in range(num_layers):
        sigma_attn_out_v, r_out = attn_block(r_in)
        r_in = (lambda_sq * r_in * sigma_new + beta_sq * r_out * 1.0) / (lambda_sq * sigma_new + beta_sq * 1.0)
        sigma_new = lambda_sq * sigma_new + beta_sq * 1.0
        sigma_attn_out_v_list.append(sigma_attn_out_v)
        r_out = ffn_block(r_in)
        r_in = (lambda_sq * r_in * sigma_new + beta_sq * r_out * 1.0) / (lambda_sq * sigma_new + beta_sq * 1.0)
        sigma_new = lambda_sq * sigma_new + beta_sq * 1.0
        r_list.append(r_in)

    return sigma_attn_out_v_list[layer_idx]


@torch.no_grad
def trunc_orthogonal_(
    tensor,
    gain: float = 1.0,
):
    r"""simplified truncated orthogonal, no guarantees"""

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new_empty(rows, cols)
    torch.nn.init.trunc_normal_(flattened, mean=0.0, std=1.0)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.linalg.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()

    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def _xavier_gain_to_std(gain, dim0, dim1):
    return gain * sqrt(2.0 / float(dim1 + dim0))


def wrapped_trunc_normal(tensor, std):
    torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)


def wrapped_ortho(tensor, std):
    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    torch.nn.init.orthogonal_(tensor, gain=std * math.sqrt(max(rows, cols)))


def wrapped_trunc_ortho(tensor, std):
    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    trunc_orthogonal_(tensor, gain=std * math.sqrt(max(rows, cols)))


def wrapped_trunc_ortho_natural_scale(tensor, std=1.0):
    fan_out = tensor.shape[0]
    fan_in = tensor.numel() // fan_out
    trunc_orthogonal_(tensor, gain=std * math.sqrt(fan_out / fan_in))


@torch.no_grad()
def init_qkv(qkv_tensor, init_fn, qk_std, v_std, dim, head_dim):
    s = qkv_tensor.shape[0]
    n_kv_heads = (s - dim) // (2 * head_dim)
    shapes = [dim, n_kv_heads * head_dim, n_kv_heads * head_dim]

    Q, K, V = (
        qkv_tensor.new_empty([shapes[0], dim]),
        qkv_tensor.new_empty([shapes[1], dim]),
        qkv_tensor.new_empty([shapes[2], dim]),
    )
    init_fn(Q, qk_std)
    init_fn(K, qk_std)
    init_fn(V, v_std)
    qkv_tensor.data.copy_(torch.cat([Q, K, V], dim=0).contiguous())


@torch.no_grad()
def init_qk_diagonal(qkv_tensor, init_fn, qk_std, v_std, dim, head_dim):
    s = qkv_tensor.shape[0]
    n_kv_heads = (s - dim) // (2 * head_dim)
    shapes = [dim, n_kv_heads * head_dim, n_kv_heads * head_dim]
    assert n_kv_heads == dim // head_dim

    Q = torch.eye(dim, dtype=qkv_tensor.dtype, device=qkv_tensor.device) * qk_std
    K = torch.eye(dim, dtype=qkv_tensor.dtype, device=qkv_tensor.device) * qk_std
    V = qkv_tensor.new_empty([shapes[2], dim])
    init_fn(V, v_std)
    qkv_tensor.data.copy_(torch.cat([Q, K, V], dim=0).contiguous())


@torch.no_grad()
def init_glu(glu_tensor, init_fn, w1_std, w2_std):
    g, h = glu_tensor.shape
    W1, W2 = (
        glu_tensor.new_empty([g // 2, h]),
        glu_tensor.new_empty([g // 2, h]),
    )
    init_fn(W1, w1_std)
    init_fn(W2, w2_std)
    glu_tensor.data.copy_(torch.cat([W1, W2], dim=0).contiguous())


@torch.no_grad()
def normalization_init(tensor):
    torch.nn.init.ones_(tensor)


class Init:
    """Construct a separate Init object that can dispatch all the various inits based on names.
    All biases are always initialized to zero, this just handles weights."""

    def __init__(
        self,
        init_strategy: str = "mitchell",
        dim: int = 1024,
        dim2: int = 4096,
        head_dim: int = 64,
        num_layers=32,
        mup_model_scaling_factor: int = 1,
        truncate_normals: bool = True,
        orthogonal: bool = False,
        verbose: bool = True,
        skip_reinitializing: bool = False,
    ):
        self.init_strategy = init_strategy
        self.dim = dim
        self.dim2 = dim2
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.mup_model_scaling_factor = mup_model_scaling_factor
        if "bernstein" in init_strategy:
            self.normal_ = wrapped_trunc_ortho_natural_scale
        else:
            if not orthogonal:
                self.normal_ = wrapped_trunc_normal if truncate_normals else torch.nn.init.normal_
            else:
                self.normal_ = wrapped_trunc_ortho if truncate_normals else wrapped_ortho
        self.verbose = verbose
        self.skip_reinitializing = skip_reinitializing

    def fn(self, name_of_layer: str, layer_idx: int = 0):
        """Return init function as Callable to evaluated later on a module.weight,
        and to be stored for reset_parameters()"""
        if self.skip_reinitializing:
            return lambda tensor: None  # skip the entire thing if we know we are going to load a ckpt anyway

        mu = self.mup_model_scaling_factor
        # Dividing by mu here means retrieving stddevs as if the model was smaller
        # these are then later scaled by /mu, if appropriate (see below)
        init_table = get_factor_table(self.dim // mu, self.dim2 // mu, self.head_dim, layer_idx, self.num_layers)[
            self.init_strategy
        ]
        # lookup hierarchy needs to consider five cases
        # -1) if normalization layer, return the trivial initializer for that
        # 0) if the layer is fused, make sure all subcomponents are correctly initialized
        # 1) if layer is named directly as q,k,v,embedding, head, mlp, init from table
        # 2) if only "out_proj" or "in_proj" are defined, use those
        # 3) if layer name is not defined, look up the default std value

        if "normalization" in name_of_layer:
            init = normalization_init
        elif "qkv" in name_of_layer:
            qk_std = next((init_table.get(key) for key in ["q", "in_proj", "std"] if key in init_table), None)
            v_std = next((init_table.get(key) for key in ["v", "in_proj", "std"] if key in init_table), None)
            if qk_std is None or v_std is None:
                raise ValueError(f"Could not resolve init of layer{name_of_layer}")
            qk_std /= mu
            v_std /= mu

            if "diagonal" not in name_of_layer:

                def init(tensor):
                    if self.verbose:
                        print(f"Init layer {layer_idx} {name_of_layer} with qk_std={qk_std:2.4f}, v_std={v_std:2.4f}.")
                    init_qkv(tensor, self.normal_, float(qk_std), float(v_std), self.dim, self.head_dim)

            else:

                def init(tensor):
                    if self.verbose:
                        print(f"Init layer {layer_idx} {name_of_layer} diag s={qk_std:2.4f}, v_std={v_std:2.4f}.")
                    init_qk_diagonal(tensor, self.normal_, float(qk_std), float(v_std), self.dim, self.head_dim)

        elif "glu" in name_of_layer:
            w1_std = next((init_table.get(key) for key in ["w1", "mlp", "in_proj", "std"] if key in init_table), None)
            w2_std = next((init_table.get(key) for key in ["w2", "mlp", "out_proj", "std"] if key in init_table), None)
            if w1_std is None or w2_std is None:
                raise ValueError(f"Could not resolve init of layer {name_of_layer}")
            w1_std /= mu
            w2_std /= mu

            def init(tensor):
                if self.verbose:
                    print(f"Init layer {layer_idx} {name_of_layer} with w1_std={w1_std:2.4f}, w2_std={w2_std:2.4f}.")
                init_glu(tensor, self.normal_, float(w1_std), float(w1_std))

        else:
            if name_of_layer in init_table:
                std = init_table[name_of_layer]
                if name_of_layer in [
                    "out_attn",
                    "w2",
                    "w3",
                    "q",
                    "k",
                    "v",
                    "w1",
                    "w1",
                    "w2",
                    "w3",
                    "mlp",
                    "in_proj",
                    "out_proj",
                ]:
                    std /= mu
            elif "out_proj" in init_table and name_of_layer in ["out_attn", "w2", "w3"]:
                std = init_table["out_proj"] / mu
            elif "in_proj" in init_table and name_of_layer in ["q", "k", "v", "w1"]:  # v is debated
                std = init_table["in_proj"] / mu
            elif "mlp" in init_table and name_of_layer in ["w1", "w2", "w3", "mlp"]:
                std = init_table["mlp"] / mu
            else:
                try:
                    std = init_table["std"]
                except KeyError:
                    raise ValueError(
                        f"Layer {name_of_layer}-{layer_idx} accessing undefined init key 'std'. Rename layer to match init table."
                    )

            def init(tensor):
                if self.verbose:
                    print(f"Init layer {layer_idx} {name_of_layer} with std={std:2.4f}.")
                self.normal_(tensor, std=float(std))

        return init

    def apply(self, module, name_of_layer: Optional[str] = None, layer_idx: int = 0):
        """Directly apply the init to an already constructed module"""
        if name_of_layer is not None and hasattr(module, "weight"):
            self.fn(name_of_layer, layer_idx)(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def get_scales(self, layer_idx: int = -1):
        init_table = get_factor_table(self.dim, self.dim2, layer_idx, self.num_layers)[self.init_strategy]
        residual_scale = float(init_table.get("residual", 1.0))
        skip_scale = float(init_table.get("skip", 1.0))
        return residual_scale, skip_scale

    def get_std(self, name_of_layer: str, layer_idx: int = -1):
        init_table = get_factor_table(self.dim, self.dim2, layer_idx, self.num_layers)[self.init_strategy]
        if name_of_layer in init_table:
            std = init_table[name_of_layer]
        elif "out_proj" in init_table and name_of_layer in ["out_attn", "w2", "w3"]:
            std = init_table["out_proj"]
        elif "in_proj" in init_table and name_of_layer in ["q", "k", "v", "w1"]:  # v is debated
            std = init_table["in_proj"]
        elif "mlp" in init_table and name_of_layer in ["w1", "w2", "w3", "mlp"]:
            std = init_table["mlp"]
        else:
            std = init_table["std"]
        return std

    @property
    def logit_scale(self):
        init_table = get_factor_table(self.dim, self.dim2, 0, self.num_layers)[self.init_strategy]
        return float(init_table.get("logit_scale", 1.0)) / self.mup_model_scaling_factor

    @property
    def embedding_scale(self):
        init_table = get_factor_table(self.dim, self.dim2, 0, self.num_layers)[self.init_strategy]
        return float(init_table.get("embed_scale", 1.0))

    def __repr__(self):
        return f"{self.init_strategy} Initializer {self.dim}x{self.dim2}x{self.head_dim}-{self.num_layers}"


def param_init_fn(module: torch.nn.Module, init: Init) -> None:
    """This would only be for compat with certain FSDP variants, but is too coarse for us"""
    for name, module in module.named_modules():
        if isinstance(module, torch.nn.Embedding):
            init.apply(module, "embedding")
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif "attn" in name:
            if "qkv" in name:
                init.apply(module, "qkv", module.layer_id)
            else:
                init.apply(module, "out_attn", module.layer_id)
        elif "mlp" in name:
            if "fc" in name:
                init.apply(module, "w1", module.layer_id)
            elif "proj" in name:
                init.apply(module, "w2", module.layer_id)
            else:
                init.apply(module, "mlp", module.layer_id)
        elif "head" in name:
            init.apply(module, "head", module.layer_id)


########################## Old/Default inits #######################################################33


# mitchell init just like olmo
def init_normal(n_embd):
    def init(tensor):
        std = 1 / math.sqrt(n_embd)
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)

    return init


# mitchell init just like olmo
# layer_id ranges [from 1 to num_layers]
def scaled_init_normal(n_embd, layer_id):
    # this is the same as the GPT-J 6B paper
    # https://arxiv.org/pdf/2312.16903.pdf says this is bad
    def init(tensor):
        std = 1 / math.sqrt(n_embd)
        std = std / math.sqrt(2 * (layer_id + 1))
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)

    return init


def init_weights(module: torch.nn.Module, n_layer: int, n_embd: int):
    # Follows GPT-NeoX: https://arxiv.org/abs/2204.06745
    if isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / n_embd))
    elif isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / n_embd))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
