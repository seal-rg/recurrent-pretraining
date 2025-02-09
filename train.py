"""
This script is originally adapted from and inspired by the tinyllama.py and
redpajama.py scripts in the lit-gpt/pretrain directory, but not too much of the original structure remains.
"""

####################################################################################################
# Imports.
####################################################################################################

import time

global_start_time = time.time()
import math
import os
import socket

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Optional
import json

import torch
import torch.nn as nn

# Check device health immediately after loading torch and standard libraries without loading cuda/hip:
nvml_count = torch.cuda._device_count_amdsmi() if torch.version.hip else torch.cuda._device_count_nvml()
if nvml_count < 1:
    raise ValueError(f"Node failure! Device manager init failed on {socket.gethostname()}")


if TYPE_CHECKING:
    import torch.distributed
    import torch.version
    import torch._dynamo.config
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.loggers import WandbLogger
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from torch.distributed.checkpoint import state_dict as state_dict_helpers

import warnings

warnings.filterwarnings("ignore", message="The config.capture_autograd_function flag is deprecated")  # pytorch nightly
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*")  # our weights

from recpre.settings import CLISettings


from recpre.tokenizer import Tokenizer
from recpre.huggingface_dataset import HuggingfaceDataset, ParquetStream, ParquetStreamPure, RandomTokensDataset
from recpre.data_loading_utils import generic_collate_fn
import recpre.utils
from recpre.data_scheduler_utils import DataSchedulerTracker, DataScheduler
from recpre.monitor import (
    enable_monitoring_on_step,
    disable_monitoring_and_retrieve_metrics,
    track_gradient_metrics,
    get_MFU_metrics,
)

from dataclasses import asdict, is_dataclass
from jsonargparse import CLI
import re

RETRY_CACHE_INDUCTOR = False

if RETRY_CACHE_INDUCTOR:
    import torch._inductor.codecache

    torch._inductor.codecache.PyCodeCache.load_by_key_path = classmethod(recpre.utils.load_by_key_path_with_retry)

end_time = time.time()
if int(os.getenv("SLURM_PROCID", "0")) == 0:
    print(f"{time.ctime()[:-5]}: Time to load libraries: {end_time - global_start_time:.02f} seconds.")

####################################################################################################
# Setup functions.
####################################################################################################
Fabric = recpre.utils.LightningFabric | recpre.utils.SimpleFabric


def set_torch_flags(cfg):
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    # Do they AMD cards pick up on any of this? :
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway

    # Dynamo + DDP primitives:
    torch._dynamo.config.optimize_ddp = cfg.dynamo_ddp_config
    # compilation choices
    torch._dynamo.config.compiled_autograd = cfg.compiled_autograd
    if cfg.fail_instead_of_recompile:
        torch._dynamo.config.error_on_recompile = True


def setup_fabric(cfg: CLISettings) -> Fabric:
    """Sets up the fabric and logger based on the cfg."""
    # Instantiate the logger.
    if cfg.logger_name == "wandb":
        # set offline dynamically from environemtn
        logger = WandbLogger(
            entity="tomg-group-umd", project=cfg.logger_project, name=cfg.run_name, save_dir=cfg.out_dir
        )
    else:
        raise ValueError(f"`logger={cfg.logger_name}` is not a valid option.")

    # Instantiate the fabric.
    if cfg.fabric_strategy == "simple-ddp":
        fabric = recpre.utils.SimpleFabric(precision=cfg.fabric_precision, loggers=[logger])
        fabric.print("Using simple fabric.")
    else:
        if "fsdp" in cfg.fabric_strategy:
            if "grad" in cfg.fabric_strategy:
                sharding_strategy = "SHARD_GRAD_OP"
                """ Gradients and optimizer states are sharded during computation, and additionally, parameters are sharded outside computation. 
                    For the parameters, this strategy unshards before the forward, does not reshard them after the forward, and only reshards them
                    after the backward computation. The sharded optimizer states are updated locally per rank. Inside no_sync(), the parameters
                    are not resharded after the backward computation.
                """
            elif "full" in cfg.fabric_strategy:
                sharding_strategy = "FULL_SHARD"  # choose FULL_SHARD if oom
            elif "hybrid2" in cfg.fabric_strategy:
                sharding_strategy = "_HYBRID_SHARD_ZERO2"
                """  Apply SHARD_GRAD_OP within a node, and replicate parameters across nodes. This is like HYBRID_SHARD, except this may 
                provide even higher throughput since the unsharded parameters are not freed after the forward pass, saving the all-gathers 
                in the pre-backward.
                """
            else:
                sharding_strategy = "HYBRID_SHARD"  #  USE "HYBRID_SHARD" AT SCALE
            precision_strategy = derive_precision(cfg.fabric_precision, cfg.fabric)
            strategy = FSDPStrategy(
                auto_wrap_policy={cfg.model_config.Block},
                mixed_precision=precision_strategy,
                activation_checkpointing_policy={cfg.model_config.Block} if cfg.gradient_checkpointing else None,
                state_dict_type="full",
                sharding_strategy=sharding_strategy,  # type: ignore
                param_init_fn=((lambda x: x.to_empty(recurse=False)) if cfg.model_impl == "huggingface" else None),
                # use_orig_params=cfg.fabric.fsdp_use_original_params, # no
            )
        elif cfg.fabric_strategy == "ddp":
            strategy = DDPStrategy(find_unused_parameters=True)  # required for recurrent models with TBPTT
        elif cfg.fabric_strategy == "single":
            strategy = SingleDeviceStrategy(device=torch.device("cuda:0") if torch.cuda.is_available() else "cpu")
        elif cfg.fabric_strategy == "axonn_tp":
            from axonn.lightning import AxonnStrategy
            # dataloader world size mod not merged yet

            strategy = AxonnStrategy(
                G_intra_r=cfg.fabric.row_tensor_parallel_size,
                G_intra_c=cfg.fabric.col_tensor_parallel_size,  # this needs more integration!
                G_intra_d=cfg.fabric.depth_tensor_parallel_size,
                overlap_communication=cfg.fabric.optimize_communication,
            )
        else:
            raise ValueError(f"`fabric_strategy={cfg.fabric_strategy}` is not a valid option.")

        # Instantiate and launch/initialize the fabric distributed environment management.
        fabric = recpre.utils.LightningFabric(
            devices=cfg.devices,
            strategy=strategy,
            precision=cfg.fabric_precision,
            loggers=[logger],
            num_nodes=cfg.num_nodes,
        )
        fabric.print(f"Using Lightning Fabric with strategy {cfg.fabric_strategy} ")
        fabric.launch()

    fabric.print(f"> gradient_accumulation_steps = {cfg.gradient_accumulation_steps}")
    fabric.print(f"> micro_batch_size = {cfg.micro_batch_size}")
    fabric.print(f"> global_batch_size = {cfg.world_batch_size}")

    return fabric


####################################################################################################
# Main driver functions.
####################################################################################################


def startup(fabric: Fabric, cfg: CLISettings):
    """The main driver function for the training script."""
    start_time = time.time()

    # Get job remaining time
    if cfg.save_n_min_before_job_done is not None:
        if fabric.global_rank == 0:
            global_total_time = _get_time_from_slurm()
            fabric.print(f"Total job time: {global_total_time:.02f} seconds.")
        else:
            global_total_time = 0

        global_total_time = fabric.broadcast(global_total_time, 0)  # does this have to be a broadcast?
        cfg.global_total_time = global_total_time

    # Prepare directories for logging
    if fabric.global_rank == 0:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.out_dir) / fabric.get_prefix_for_checkpoint()).mkdir(parents=True, exist_ok=True)
        # Last step before we move on is to dump the cfg to a file in the out_dir.
        # This is is itself loadable as a config by passing like train.py --config run_config.json
        with open(f"{cfg.out_dir}/run_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=4)
        with open(f"{cfg.out_dir}/model_config.json", "w") as f:
            json.dump(asdict(cfg.model_config) if is_dataclass(cfg.model_config) else cfg.model_config, f, indent=4)
    # Load tokenizer
    tokenizer = Tokenizer(cfg.tokenizer_path)
    if tokenizer.pad_id is None:
        tokenizer.pad_id = -1
    if cfg.cache_attn:
        assert tokenizer.cache_token_id is not None
    if cfg.doc_block_attn:
        assert tokenizer.eod_token_id is not None

    # Create data objects
    t0 = time.time()
    # On block size, moved this here to be more explicit that this is happening ...
    if not cfg.ignore_block_size_mismatch:
        assert cfg.block_size == cfg.model_config.block_size, "cfg.block_size must match config.block_size"
    # Increase by one to actually be supervising "block_size" tokens in every update after rshift.
    train_dataloader, val_dataloader, data_scheduler_tracker = create_dataloaders(
        batch_size=cfg.micro_batch_size,
        block_size=cfg.loader_block_size,
        fabric=fabric,
        seed=(cfg.seed + fabric.global_rank),
        cfg=cfg,
        tokenizer=tokenizer,
    )
    # train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    # setup_dataloaders would also 1) introduce a distributed sampler, and 2) move tensors to a device.
    # We would rather do both manually
    if data_scheduler_tracker is not None:
        data_scheduler = DataScheduler(data_scheduler_tracker, cfg.data_config["train_data"], cfg)
        data_scheduler.step(0)
    else:
        data_scheduler = None
    fabric.print(f"{time.ctime()[:-5]}: Time to instantiate and setup dataloaders: {time.time() - t0:.02f} seconds.")

    # Construct the model
    fabric.seed_everything(cfg.seed)  # same seed for every process to init model (FSDP)
    # if cfg.model_checkpoint is not None:
    #     recpre.utils.check_valid_checkpoint_dir(Path(cfg.model_checkpoint))
    fabric.print(f"Loading model with {cfg.model_config.__dict__}")

    # Set the objective
    objective = {
        "op": recpre.utils.chunked_cross_entropy,
        "label_smoothing": cfg.label_smoothing,
        "ignore_index": tokenizer.pad_id,
        "z_regularization": cfg.z_regularization,
    }

    # Initialize the model
    t0 = time.time()
    with fabric.init_module(empty_init="fsdp" in cfg.fabric_strategy):
        model = cfg.model_config.construct_model(
            objective=objective,
            tokenizer=tokenizer.processor,
            gradient_checkpointing=cfg.gradient_checkpointing and "fsdp" not in cfg.fabric_strategy,
        )
    fabric.print(f"{time.ctime()[:-5]}: Time to instantiate model: {time.time() - t0:.02f} seconds.")
    num_params = recpre.utils.num_parameters(model)
    fabric.log_to_summary({"num_parameters": num_params, "device": torch.cuda.get_device_name()})

    # With fabric and the model up, we can compute a few last derived cfg
    if cfg.max_steps is None:
        cfg.max_tokens_per_device = cfg.max_tokens // fabric.world_size
        cfg.tokens_per_step = cfg.micro_batch_size * cfg.block_size
        cfg.max_steps = cfg.max_tokens_per_device // cfg.tokens_per_step
        fabric.print(
            f"Based on block size {cfg.block_size}, expecting to take {cfg.max_steps} steps to reach "
            f"{cfg.max_tokens / 1e12:4.2f}T tokens.\nRunning {cfg.tokens_per_step * fabric.world_size} tok/step "
            f"for {cfg.max_tokens_per_device / 1e9:4.2f}B in total per device."
        )

    # Set up the final fabric+model details
    t0 = time.time()
    model = fabric.setup(model, compile=cfg.compile_model)
    fabric.print(f"Model with full setup is {model}")
    fabric.print(f"Total parameters: {num_params:,}")
    if hasattr(model.transformer, "core_block"):
        rec_params = sum([p.numel() for p in model.transformer.core_block.parameters()])
        static_params = num_params - rec_params
        if "fsdp" in cfg.fabric_strategy:
            rec_params, static_params = rec_params * cfg.devices, static_params * cfg.devices
        r = model.config.mean_recurrence
        unfolded_params_mean = static_params + rec_params * r
        unfolded_params_max = static_params + rec_params * 2 * r
        fabric.print(f"Model initialized with {int(rec_params / 1e6):,}m parameters in recurrent block.")
        fabric.print(f"Will unfold to {int(unfolded_params_mean // 1e6):,}m mean parameters at test time ({r} rec).")
        fabric.print(f"Could unfold to {int(unfolded_params_max // 1e6):,}m parameters at test time ({2 * r} rec).")
    fabric.print(f"{time.ctime()[:-5]}: Time to setup model: {time.time() - t0:.02f} seconds.")
    t0 = time.time()
    # Set up the optimizer and training state object.
    param_groups = recpre.optim.get_param_groups(
        model.named_parameters(),
        cfg.no_weight_decay_for_bias_and_norm_params,
        weight_lr_scale=1 / getattr(model.config, "mup_model_scaling_factor", 1.0),
    )
    optimizer = recpre.optim.get_optimizer(
        cfg.optimizer,
        model,
        cfg.fabric.optim_sharding,
        allow_fusion=not torch.version.hip and "bf16" in cfg.fabric_precision,
        use_apex_adamw=cfg.fabric.use_apex_adamw,
    )(param_groups, **cfg.optim_config)
    optimizer = fabric.setup_optimizers(optimizer)
    fabric.print(f"{time.ctime()[:-5]}: Time to instantiate and setup optimizers: {time.time() - t0:.02f} seconds.")

    state = {
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        "data_scheduler": data_scheduler,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,  # val loader actually doesnt need state
        "microbatch_step": 0,  # mbs steps
        "optimizer_step": 0,  # optimizer updates taken
        "is_accumulating": False,
        "metrics": {"lr": 0.0, "grad_norm": 0.0, "current_batch_size": 0},
        "should_exit_training": False,
        "model_config": asdict(cfg.model_config) if is_dataclass(cfg.model_config) else cfg.model_config,
    }

    # If resuming or reloading, determines the checkpoint to resume from and loads it into the fabric
    load_checkpoint(fabric, state, cfg.out_dir, cfg.run_name, cfg.model_checkpoint, cfg.resume)
    if asdict(cfg.model_config) != state["model_config"]:
        fabric.print("-------------Warning, model config difference between checkpoint and model config!-------------")

    # Report the full cfg set for the run.
    fabric.print(f"cmdline + derived cfg:\n{json.dumps(cfg.__dict__, default=lambda x: x.__dict__, indent=4)}")
    fabric.logger.log_hyperparams(cfg.__dict__)

    fabric.barrier()
    end_time = time.time()
    fabric.print(f"{time.ctime()[:-5]}: Total time to run main func setups: {end_time - start_time:.02f} seconds.")

    return state


@torch.no_grad()
def validate(
    fabric: Fabric, model: nn.Module, val_dataloader: DataLoader, tokenizer: Tokenizer, cfg
) -> dict[str, torch.Tensor]:
    fabric.print(f"Validating for {cfg.eval_iters} steps ...")
    model.eval()

    def loss_fn(logits, labels):
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.view(-1), ignore_index=model.objective["ignore_index"]
        )

    metrics = {}

    # Don't merge this block into main, works only for a few models
    losses = torch.zeros(cfg.eval_iters, len(cfg.partial_depth_eval) + 1, device=fabric.device)
    for idx, depth in enumerate(cfg.partial_depth_eval + [model.config.mean_recurrence]):
        for k, (input_ids, labels, _) in enumerate(val_dataloader):
            if k >= cfg.eval_iters:
                break

            input_ids = input_ids.to(fabric.device, non_blocking=True)
            labels = labels.to(fabric.device, non_blocking=True)

            mask, positions = get_attention_mask(input_ids, tokenizer, cfg.cache_attn, cfg.doc_block_attn)
            rec_steps = torch.as_tensor([depth, 0])
            outputs = model(
                input_ids, position_ids=positions, attention_mask=mask, return_logits=True, num_steps_pair=rec_steps
            )
            losses[k, idx] = loss_fn(outputs["logits"], labels)

    # print(f"{time.ctime()[:-5]}: Validation forward passes complete on rank ({fabric.global_rank}/{fabric.world_size})")
    # Communicate
    global_val_loss = fabric.all_reduce(losses.mean(dim=0))  # dim-0 is the mbs dimension, dim-1 is kept after comms
    metrics["val_loss"] = global_val_loss[-1]
    metrics["val_ppl"] = global_val_loss[-1].exp()
    for idx, depth in enumerate(
        cfg.partial_depth_eval + [model.config.mean_recurrence]
    ):  # duplicate mean depth value key for ease of use
        metrics[f"val_loss_{depth}"] = global_val_loss[idx]
        metrics[f"val_ppl_{depth}"] = global_val_loss[idx].exp()

    model.train()
    return metrics


def train_step(input_ids, labels, fabric, state, running_loss, running_ppl, cfg):
    """Separate scope for a single train step, encapsulating the part that is actual work"""
    model = state["model"]
    optimizer = state["optimizer"]
    data_scheduler = state["data_scheduler"]
    tokenizer = state["tokenizer"]
    metrics = state["metrics"]

    state["microbatch_step"] += 1
    model.step = state["microbatch_step"]  # propagate this to the model
    # Goldfishing on CPU
    if cfg.goldfish.strategy is not None:
        labels, _ = recpre.utils.apply_tld(labels=labels, settings=cfg.goldfish, ignore_index=tokenizer.pad_id)

    # Realize the input and labels tensors.
    input_ids = input_ids.to(fabric.device, non_blocking=True)
    labels = labels.to(fabric.device, non_blocking=True)
    mask, positions = get_attention_mask(input_ids, tokenizer, cfg.cache_attn, cfg.doc_block_attn)

    # Prepare for step
    if state["microbatch_step"] < cfg.shape_watching_steps:
        bsz, seq_len = input_ids.shape
        fabric.print(f"bsz: {bsz} | seq_len: {seq_len}")
        fabric.print(f"input_ids.shape: {input_ids.shape} | labels.shape: {labels.shape}")
    elif state["microbatch_step"] == cfg.shape_watching_steps and cfg.shape_watching_steps > 0:
        fabric.print("Silencing shape watching ...")
    state["is_accumulating"] = state["microbatch_step"] % cfg.gradient_accumulation_steps != 0
    monitor_step = cfg.model_telemetry and state["microbatch_step"] % cfg.log_step_interval == 0
    if monitor_step and not state["is_accumulating"]:
        model.module.apply(enable_monitoring_on_step)

    # The actual compute step of  Forward, loss, and backward computation:
    def tightly_scoped_fwd_bwd(model, input_ids, positions, labels, mask):
        with fabric.no_backward_sync(model, enabled=state["is_accumulating"]):
            outputs = model(input_ids, position_ids=positions, labels=labels, attention_mask=mask)
            fabric.backward(outputs["loss"] / cfg.gradient_accumulation_steps, model=model)
            return outputs["loss"].detach(), outputs["log_ppl"].detach()

    loss, log_ppl = tightly_scoped_fwd_bwd(model, input_ids, positions, labels, mask)
    # Record metrics
    metrics["mbs_loss"] = loss
    running_loss.update(loss)
    running_ppl.update(log_ppl)

    # Guardrails
    if not cfg.allow_nonfinite_loss and not torch.isfinite(loss):
        fabric.print(f"Loss is {loss} on {socket.gethostname()}. Terminating ...")
        state["should_exit_training"] = True

    # Take an optimization step if not accumulating.
    if not state["is_accumulating"]:
        # LR scheduler (now a pre-increment, so that this step runs on exactly this scheduled lr)
        current_step_lr = get_lr(step=state["microbatch_step"], max_steps=cfg.max_steps, cfg=cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = torch.as_tensor(current_step_lr * param_group["base_lr"])

        metrics["grad_norm"] = fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_clip, error_if_nonfinite=False)
        if torch.isfinite(metrics["grad_norm"]):
            if state["optimizer_step"] > 0:  # Skip first step if compiling or autotuning
                if cfg.compile_optimizer:
                    for param_group in optimizer.param_groups:
                        for param in param_group["params"]:
                            if param.grad is not None:
                                torch._dynamo.decorators.mark_static_address(param.grad)  # yolo
                    torch.compile(optimizer.step, mode="max-autotune-no-cudagraphs")()
                else:
                    optimizer.step()
        else:
            if cfg.skip_nonfinite_grads:
                fabric.print(f"Grad norm non-finite! Optimizer step {state['optimizer_step'] + 1} skipped.")
            else:
                fabric.print(f"Grad norm non-finite! Optimizer step {state['optimizer_step'] + 1}. Terminating ...")
                state["should_exit_training"] = True

        if monitor_step:  # Monitor triggers after update (to check it), but before grads are wiped
            track_gradient_metrics(model, optimizer, metrics)
            model.module.apply(partial(disable_monitoring_and_retrieve_metrics, metrics=metrics))
        optimizer.zero_grad(set_to_none=not (cfg.fabric.use_apex_adamw or cfg.compile_optimizer))
        state["optimizer_step"] += 1
        # Data scheduler
        if data_scheduler is not None:
            data_scheduler.step(state["microbatch_step"])
        # Batch size scheduler
        cfg.gradient_accumulation_steps = get_batch_size(state["microbatch_step"], cfg)

        metrics["lr"] = current_step_lr
        metrics["current_batch_size"] = cfg.gradient_accumulation_steps * cfg.micro_batch_size * cfg.replicas


def train(fabric, state, cfg):
    """The main training loop."""
    warmup_or_early_fail_allreduce(fabric)
    state["initial_step"] = state["last_logged_step"] = state["microbatch_step"]  # "initial_step" in this chunk
    train_iterator = iter(state["train_dataloader"])

    # Set up global loss monitor.
    running_loss = RunningMean(window=cfg.log_step_interval, sync_on_compute=False).to(fabric.device)
    running_log_ppl = RunningMean(window=cfg.log_step_interval, sync_on_compute=False).to(fabric.device)
    first_validation_passed = False
    fabric.barrier()
    state["total_t0"] = time.time()  # this is the start time for this chunk of training
    fabric.print(f"{time.ctime()[:-5]}: Training preparations finished, starting to iterate train data now.")

    # Main training loop.
    step_time = 0
    for input_ids, labels, _ in train_iterator:
        # Main train work
        t0 = time.time()  # measure average time over last log_step steps
        train_step(input_ids, labels, fabric, state, running_loss, running_log_ppl, cfg=cfg)
        step_time += time.time() - t0
        step = state["microbatch_step"]

        # Regular validation (comes before log)
        validate_regular = not state["is_accumulating"] and step % cfg.eval_step_interval == 0
        validate_at_the_end = step >= cfg.max_steps - 1
        if validate_regular or validate_at_the_end:
            t0 = time.time()
            val_metrics = validate(fabric, state["model"], state["val_dataloader"], state["tokenizer"], cfg=cfg)
            td = time.time() - t0
            val_metrics["val_time"] = torch.as_tensor(td)

            fabric.print(f"Step {step}: Val loss {val_metrics['val_loss'].item():.4f}, Val time: {td:.2f}s")
            state["metrics"] |= val_metrics
            if not first_validation_passed:
                # This is the first moment that all potential compilation calls have resolved
                fabric.log_to_summary({"first_validation_passed": time.time() - global_start_time})
                first_validation_passed = True
            fabric.barrier()

        # Communicate exit flags from all devices BEFORE logging conditional on flag
        if torch.distributed.is_initialized() and (state["microbatch_step"] % 32) == 0:  # Exit is comm'd every 32 steps
            state["should_exit_training"] = torch.as_tensor([state["should_exit_training"]], device=fabric.device)
            torch.distributed.all_reduce(state["should_exit_training"], torch.distributed.ReduceOp.MIN, async_op=False)

        # Log at an interval.
        if step % cfg.log_step_interval == 0 or (state["should_exit_training"] and (step % 32) == 0):
            log_step(fabric, state, running_loss, running_log_ppl, step_time, state["data_scheduler"], cfg)
            step_time = 0

        if state["should_exit_training"] and (state["microbatch_step"] % 32) == 0:  # Exit is checked every 32 steps.
            fabric.print(f"{time.ctime()[:-5]}: Exiting training early in step {step} due to error signal received.")
            break

        # Maybe save, this needs to come after the error signal exit
        maybe_save_checkpoint(fabric, state, cfg, is_accumulating=state["is_accumulating"])

        if step >= cfg.max_steps - 1:
            fabric.print(f"{time.ctime()[:-5]}: Exiting training orderly after completion of {step + 1} steps.")
            break


####################################################################################################
# Train loop sub-routines.
####################################################################################################


def log_step(
    fabric: Fabric,
    state: dict,
    running_loss: RunningMean,
    running_log_ppl: RunningMean,
    accumulated_step_time: float,
    data_scheduler: Optional[DataScheduler],
    cfg: CLISettings,
):
    """Log at this microbatch step and compute the throughput."""
    loss = running_loss.compute()
    log_ppl = running_log_ppl.compute()
    t1 = time.time()

    # Load metrics here:
    metrics = state["metrics"]

    avg_time_per_step = accumulated_step_time / (state["microbatch_step"] - state["last_logged_step"])
    tokens_per_step = cfg.micro_batch_size * cfg.block_size * fabric.world_size
    tokens_per_second = tokens_per_step / avg_time_per_step

    metrics |= {
        "local_loss": loss,
        "local_ppl": log_ppl.exp(),
        "microbatch_step": state["microbatch_step"],
        "optimizer_step": state["optimizer_step"],
        "steps/second": 1 / avg_time_per_step,
        "seconds/step": avg_time_per_step,
        "tokens/second": tokens_per_second,
        "remaining_time": (
            (t1 - state["total_t0"])
            / (state["microbatch_step"] - state["initial_step"])
            * (cfg.max_steps - state["microbatch_step"])
        ),
        "total_tokens": state["microbatch_step"] * tokens_per_step,
        "total_time": t1 - state["total_t0"],
    }
    if cfg.measure_utilization:
        max_memory_allocated_per_gpu = torch.cuda.max_memory_allocated(fabric.device) / 1024**3
        max_mem_reserved_per_gpu = torch.cuda.max_memory_reserved(fabric.device) / 1024**3
        torch.cuda.reset_peak_memory_stats(fabric.device)
        model_flops, tflops, mfu = get_MFU_metrics(tokens_per_second, fabric, state["model"], cfg.fabric_precision)
        metrics |= {
            "total_FLOPs": state["microbatch_step"] * tokens_per_step * model_flops,
            "FLOP/S": tflops,
            "model_flop_utilization": mfu,
            "max_mem_per_gpu": max_memory_allocated_per_gpu,
            "max_mem_reserved_per_gpu": max_mem_reserved_per_gpu,
        }

    # Update loss and grad_norm with all_reduce
    if "grad_norm" in metrics and metrics["grad_norm"] is not None:
        grad_norm = fabric.all_reduce(metrics["grad_norm"])
        metrics["global_grad_norm"] = grad_norm
    else:
        metrics["global_grad_norm"] = None
    metrics["global_loss"] = fabric.all_reduce(loss)

    # This is the only place where this guardrail can be checked without incurring another sync
    if cfg.loss_guardrail_active:
        total_tokens = state["microbatch_step"] * cfg.micro_batch_size * cfg.block_size * fabric.world_size
        if total_tokens > 10_000_000_000 and metrics["global_loss"] > 6:  # after 10b tokens we're in slow descent
            fabric.print(
                f"Loss guard activated with loss {metrics['global_loss']} in step {state['microbatch_step']}. "
                f"Terminating ..."
            )
            state["should_exit_training"] = True

    metrics["global_train_ppl"] = fabric.all_reduce(log_ppl).exp()

    if data_scheduler is not None:
        curr_data_weights = data_scheduler.get_data_weights()
        curr_data_weights = dict(zip(cfg.dataset_names, curr_data_weights))

        curr_sample_count = data_scheduler.get_sample_count()
        curr_sample_count = fabric.all_reduce(curr_sample_count, reduce_op="sum")

        curr_epoch_count = data_scheduler.get_epoch_count()
        curr_epoch_count = fabric.all_reduce(curr_epoch_count, reduce_op="mean")

        for i, x in enumerate(curr_data_weights.keys()):
            metrics["data_scheduler_weight/" + x] = curr_data_weights[x]
            metrics["data_scheduler_norm_weight/" + x] = curr_data_weights[x] / sum(list(curr_data_weights.values()))
            metrics["data_scheduler_sample_count/" + x] = curr_sample_count[i]
            metrics["data_scheduler_epoch_count/" + x] = curr_epoch_count[i]

            state["data_scheduler_weight/" + x] = metrics["data_scheduler_weight/" + x]
            state["data_scheduler_norm_weight/" + x] = metrics["data_scheduler_norm_weight/" + x]
            state["data_scheduler_sample_count/" + x] = metrics["data_scheduler_sample_count/" + x]
            state["data_scheduler_epoch_count/" + x] = metrics["data_scheduler_epoch_count/" + x]

    fabric.log_dict(metrics, step=state["microbatch_step"])
    state["last_logged_step"] = state["microbatch_step"]

    # Log some metrics to the console.
    step_timing = (
        f" steps/sec: {metrics['steps/second']:4.2f}  |"
        if metrics["steps/second"] >= 1.0
        else f" secs/step: {metrics['seconds/step']:4.2f}  |"
    )
    lr_str = f"{metrics['lr']:2.4e}" if "lr" in metrics and metrics["lr"] is not None else ""
    grad_norm_str = f"{metrics['global_grad_norm']:6.4e}" if metrics["global_grad_norm"] is not None else ""

    fabric.print(
        f"{time.ctime()[:-5]}\n"
        f"Step {metrics['microbatch_step']:>8}    | Loss: {metrics['global_loss']:7.4f} | {metrics['global_train_ppl']:9.2f} PPL     |"
        f" Update {metrics['optimizer_step']:>8}     |\n"
        f"{'(optimizer.step)' if not state['is_accumulating'] else ' ' * 16}"
        f" | LR: {lr_str:>10}| Grad norm: {grad_norm_str:>11} |{' ' * 19}|\n"
        f"                 | MFU : {metrics.get('model_flop_utilization', 0):6.2%}  | TFLOP/S : {metrics.get('FLOP/S', 0):5.2f}  |"
        f" tok/sec: {metrics['tokens/second']:8.1f} | {step_timing}\n"
        f"                 | Max mem allocated: {metrics.get('max_mem_per_gpu', 0):4.2f} GB       "
        f"| Max mem reserved: {metrics.get('max_mem_reserved_per_gpu', 0):4.2f} GB            |\n"
        f"                 | Tokens: {metrics['total_tokens'] / 1e9: 4.1f}B | exaFLOP: {metrics.get('total_FLOPs', 0) / 1e18:8.5f} |"
        f" Remaining time: {metrics['remaining_time'] / 3600 / 24:.2f} days             |"
    )
    # Reset metrics after logging them
    state["metrics"] = {}


####################################################################################################
# Data utility functions.
####################################################################################################


def create_dataloader(
    data_config: list[recpre.settings.DataEntry],
    batch_size: int,
    block_size: int,
    n_chunks: int,
    data_dir: str,
    fabric: Fabric,
    seed: int = 1337,
    *,
    cfg: CLISettings,
    tokenizer: Tokenizer,
    stateful: bool = True,
) -> tuple[StatefulDataLoader | DataLoader, Optional[DataSchedulerTracker]]:
    global_data_dir = data_dir
    datasets = []
    for curr_config in data_config:
        if curr_config.type == "hfds":
            assert tokenizer is not None, "tokenizer must be provided for HuggingfaceDataset"
            assert curr_config.data_dir is not None, "data_dir must be provided for HuggingfaceDataset"
            dataset = HuggingfaceDataset(
                ds_name_or_path=curr_config.data_dir,  # this is a path to a previously save_to_disk'd hfds
                seed=seed,
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
                data_id=curr_config.prefix,  # this is provided for logging, and schedule purposes
                return_data_id=curr_config.return_data_id,
                data_signature=curr_config.data_signature or cfg.data_signature,  # specification of the data fmt
                repetitions=curr_config.repetitions,  # repeat the dataset a number of times
            )
        elif "pqds" in curr_config.type:
            ParquetImpl = ParquetStreamPure if curr_config.type == "pqds-pure" else ParquetStream
            dataset = ParquetImpl(
                dataset_folder_path=curr_config.data_dir if curr_config.data_dir is not None else global_data_dir,
                seed=seed,
                shuffle=cfg.shuffle_blocks,
                shuffle_filenames=cfg.shuffle_filenames,
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
                data_id=curr_config.prefix,
                data_signature=curr_config.data_signature or cfg.data_signature,
                repetitions=None,
                return_data_id=curr_config.return_data_id,
                prefix=curr_config.prefix,
                stateful=stateful,
            )
        elif curr_config.type == "rngds":  # debug option
            dataset = RandomTokensDataset(seed=seed, vocab_size=tokenizer.vocab_size, block_size=block_size)
        else:
            raise ValueError(f"Unsupported dataset type: {curr_config.type}")

        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(f"No data found at {data_dir}.")

    if len(datasets) > 1:
        raise ValueError("Not exported")
    else:
        combined_dataset = datasets[0]
        data_scheduler_tracker = None

    parametrized_collate_fn = partial(
        generic_collate_fn,
        tokenizer=tokenizer,
        block_size=cfg.loader_block_size,
        pad_to_block_size=cfg.pad_to_block_size,
        add_bos=cfg.add_bos,
        add_eos=cfg.add_eos,
        collate_checks_enabled=cfg.collate_checks_enabled,
        all_block_size_tensors=cfg.all_block_size_tensors,
    )

    loader_class = StatefulDataLoader if stateful else DataLoader
    return (
        loader_class(
            combined_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=parametrized_collate_fn,
            num_workers=cfg.dataloader_num_workers,
            prefetch_factor=4 if cfg.dataloader_num_workers > 0 else None,
        ),
        data_scheduler_tracker,
    )


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric: Fabric,
    seed: int = 1337,
    *,
    cfg: CLISettings,
    tokenizer: Tokenizer,
) -> Tuple[StatefulDataLoader, Optional[DataLoader], DataSchedulerTracker]:
    fabric.print(f"Creating dataloaders with seed: {seed}")
    train_dataloader, data_scheduler_tracker = create_dataloader(
        cfg.data_config["train_data"],
        batch_size=batch_size,
        block_size=block_size,
        n_chunks=cfg.n_chunks,
        fabric=fabric,
        data_dir=cfg.train_data_dir,
        seed=seed,
        cfg=cfg,
        tokenizer=tokenizer,
    )
    val_dataloader, _ = (
        create_dataloader(
            cfg.data_config["val_data"],
            batch_size=batch_size,
            block_size=block_size,
            n_chunks=cfg.n_chunks,
            fabric=fabric,
            data_dir=cfg.val_data_dir,
            seed=seed,
            cfg=cfg,
            tokenizer=tokenizer,
            stateful=False,
        )
        if "val_data" in cfg.data_config
        else (None, None)
    )
    return train_dataloader, val_dataloader, data_scheduler_tracker  # type: ignore


####################################################################################################
# Train utility functions.
####################################################################################################


def derive_precision(precision, strategy_details):
    """ "Precision setup for torch fsdp"""
    import torch.distributed.fsdp

    param_dtype = torch.bfloat16 if "bf16" in precision else torch.float16 if "16" in precision else torch.float32
    reduce_dtype = torch.float32 if "mixed" in precision else param_dtype
    if r := strategy_details.all_reduce_dtype is not None:
        reduce_dtype = (
            torch.float16
            if r in ["16", "fp16", "fp16-mixed"]
            else torch.bfloat16
            if r in ["bf16", "bf16-mixed"]
            else torch.float32
        )
    return torch.distributed.fsdp.MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=torch.float32,
        keep_low_precision_grads=False,
        # cast_forward_inputs=False,
    )


def get_attention_mask(input_ids, tokenizer, cache_attn=True, doc_block_attn=True):
    mask, position_ids = None, None
    return mask, position_ids


# learning rate decay schedulers
def get_lr(step: int, max_steps: int, cfg: CLISettings) -> float:
    base_lr = cfg.optim_config["lr"]
    # 1) linear warmup and cooldown
    if step < cfg.warmup_steps:
        return base_lr * step / cfg.warmup_steps
    if step > (max_steps - cfg.cooldown_steps):
        return max(base_lr * (max_steps - step) / cfg.cooldown_steps, cfg.min_lr)
    # 2) if step > max_steps, return min learning rate
    if step > max_steps:
        return cfg.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - cfg.warmup_steps) / (max_steps - cfg.warmup_steps)
    assert 0 <= decay_ratio <= 1
    if cfg.lr_schedule == "linear":
        return base_lr - decay_ratio * (base_lr - cfg.min_lr)
    elif cfg.lr_schedule in ["constant", "trapezoid"]:
        return base_lr
    elif cfg.lr_schedule == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return cfg.min_lr + coeff * (base_lr - cfg.min_lr)
    else:
        raise ValueError(f"Unsupported lr_schedule: {cfg.lr_schedule}")


# only linear batch size schedules for now
def get_batch_size(step: int, cfg: CLISettings) -> int:
    if step > cfg.batch_size_ramp:
        gradient_accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    else:
        slope = step / cfg.batch_size_ramp
        gradient_accumulation_steps = math.ceil(slope * cfg.batch_size / cfg.micro_batch_size)
    return gradient_accumulation_steps


def load_checkpoint(fabric, state, out_dir, run_name, model_checkpoint, resume=True):
    resume_ckpt = None
    t0 = time.time()
    if resume:
        fabric.print("-------------------- Model Load triggered ------------------------------")
        base_for_glob = Path(out_dir) / fabric.get_prefix_for_checkpoint()
        fabric.print(f"Globbing for checkpoint files in {base_for_glob}")
        ckpt_pattern = f"*/*-{run_name}_*.pth" if fabric.strategy_name == "axonn_tp" else f"*-{run_name}_*.pth"
        ckpt_paths = list(base_for_glob.glob(ckpt_pattern))
        if len(ckpt_paths) == 0:
            fabric.print(f"No checkpoint found in {out_dir} to resume from.")
        else:
            resume_ckpt = max(ckpt_paths, key=(lambda p: int(p.name.split("-")[1].split(f"-{run_name}_")[0])))
            filename, directory = str(resume_ckpt.name), resume_ckpt.parents[0]
            filename = filename[filename.find("step") :]
            filename = filename.split(f"-{run_name}_")[0] + f"-{run_name}"  # split off rank info and .pth
            if fabric.strategy_name == "axonn_tp":
                directory = Path(out_dir) / fabric.get_prefix_for_checkpoint()
            resume_ckpt = directory / filename

            fabric.print(f"Resuming training from {resume_ckpt}")
            # load_save_state(state, resume_ckpt) # torch.distributed.checkpoint integration
            fabric.load(resume_ckpt, state, strict=False)

    if resume_ckpt is None and model_checkpoint is not None:
        fabric.print("-------------------- Pretrained Checkpoint Load triggered ------------------------------")
        fabric.print(f"Loaded full checkpoint (including optim and data state) from {model_checkpoint}")
        fabric.load(model_checkpoint, state, strict=False)
    if resume_ckpt or model_checkpoint:
        # Dataloaders should have resumed automatically, due to reloading the entire previous state. Let's print the state
        # as well as dataloader internal state:
        fabric.print(f"Loaded state is from step {state['microbatch_step']}")
        # fabric.print(f"Train Data Loader State: {state['train_dataloader'].state_dict()}")# deadlock
        fabric.print(f"{time.ctime()[:-5]} : Time to load ckpt state: {time.time() - t0:.02f} seconds.")
        fabric.print("-------------------- Checkpoint loaded    ------------------------------")
    else:
        fabric.print("-------------------- No Checkpoint loaded ------------------------------")
    return resume_ckpt


def maybe_save_checkpoint(fabric, state, cfg, is_accumulating=False, force_save=False):
    # Pathing for various save conditions.
    t0 = time.time()
    prefix = fabric.get_prefix_for_checkpoint()
    fully_qualified_checkpoint_path = f"{cfg.out_dir}/{prefix}/step-{state['microbatch_step']:08d}-{cfg.run_name}"

    # Check the three save conditions:
    save_at_interval = not is_accumulating and state["microbatch_step"] % cfg.save_step_interval == 0
    if cfg.save_n_min_before_job_done is not None and (state["microbatch_step"] % 32) == 0:
        time_spent = time.time() - global_start_time
        remaining_time = cfg.global_total_time - time_spent
        remaining_time = remaining_time / 60.0
        remaining_time = fabric.all_reduce(remaining_time, reduce_op="mean")  # slowdown?
        save_before_timeout = remaining_time <= cfg.save_n_min_before_job_done
        if save_before_timeout:
            fabric.print(f"{time.ctime()[:-5]}: Saving at {remaining_time:.02f} minutes left")
            cfg.save_n_min_before_job_done = None  # reset
    else:
        save_before_timeout = False

    save_at_first_step = cfg.save_first_step and (state["microbatch_step"] == 0)
    save_at_last_step = cfg.save_last_step and (state["microbatch_step"] >= (cfg.max_steps - 1))

    if save_at_interval or save_at_last_step or save_at_first_step or save_before_timeout or force_save:
        fabric.print(f"--------------------- {time.ctime()[:-5]} Model Save triggered --------")
        fabric.print(f"Saving to {str(fully_qualified_checkpoint_path)!r}")

        fabric.save(fully_qualified_checkpoint_path, state)
        fabric.print(f"------------------- {time.ctime()[:-5]} Checkpoint saved ({time.time() - t0:.02f} seconds)")


def form_save_state(state):
    """Could make this explicit, currently unused"""
    save_state_dict = {}
    model_state, optim_state = state_dict_helpers.get_state_dict(state["model"], state["optimizer"])
    save_state_dict["model"] = model_state
    save_state_dict["optimizer"] = optim_state
    save_state_dict["train_dataloader"] = state["train_dataloader"].state_dict()

    for key, value in state.items():
        if key not in ["optimizer", "model"] and "dataloader" not in key:
            save_state_dict[key] = value
    return save_state_dict


def load_save_state(state, resume_ckpt):
    checkpoint_state = torch.load(resume_ckpt, map_location=torch.device("cpu"))
    state_dict_helpers.set_state_dict(
        state["model"],
        state["optimizer"],
        model_state_dict=checkpoint_state["model"],
        optim_state_dict=checkpoint_state["optimizer"],
        options=None,
    )
    state["train_dataloader"].load_state_dict(checkpoint_state["train_dataloader"])
    for key, value in checkpoint_state.items():
        if key not in ["optimizer", "model"] and "dataloader" not in key:
            state[key] = value


def warmup_or_early_fail_allreduce(fabric):
    if torch.distributed.is_initialized():
        fabric.print("Staging allreduce warmup")
        device = fabric.device
        # Creating random data for warmup
        flat_params = torch.randn(128 * 1024 * 1024 // 4, device=device)
        num_stages = 8
        chunk_size = flat_params.numel() // num_stages

        for i in range(num_stages):
            end = min((i + 1) * chunk_size, flat_params.numel())
            chunk = flat_params[:end]
            torch.distributed.all_reduce(chunk)
            torch.cuda.current_stream().synchronize()  # Force completion
            fabric.print(f"Warmup stage {i} [{chunk.numel() // (1024 * 1024 // 4)} MB] really completed")

        torch.distributed.barrier()
        fabric.print(f"{time.ctime()[:-5]}: All warmup stages passed")


def _get_time_from_slurm() -> int:
    try:
        global_total_str_parse = os.popen("squeue -h -j $SLURM_JOBID -o %L").read()  # this is slow
        global_total_str_parse = global_total_str_parse.strip("\n")
        global_total_str_parse = [int(i) for i in re.split(":|-", global_total_str_parse)]
        if len(global_total_str_parse) == 4:
            global_total_time = (
                24 * 3600 * global_total_str_parse[0]
                + 3600 * global_total_str_parse[1]
                + 60 * global_total_str_parse[2]
                + global_total_str_parse[3]
            )
        elif len(global_total_str_parse) == 3:
            global_total_time = (
                3600 * global_total_str_parse[0] + 60 * global_total_str_parse[1] + global_total_str_parse[2]
            )
        elif len(global_total_str_parse) == 2:
            global_total_time = 60 * global_total_str_parse[0] + global_total_str_parse[1]
    except Exception as e:
        print(e)
        global_total_time = 9999999999999999
    return global_total_time


####################################################################################################
# Main control loop
####################################################################################################
import sys
import datetime


def main():
    """Encapsulates main scope away from import calls."""

    # Configuration loader
    cfg: CLISettings = CLI(CLISettings)  # type: ignore

    # Print system setup
    if int(os.getenv("SLURM_PROCID", "0")) == 0:
        print("--------------------------------------------------------------------")
        print(f"------------------ Launching run {cfg.run_name}------------------")
        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")
        print(f"Platform: {sys.platform}, Python: {sys.version.split(' (')[0]}, PyTorch: {torch.__version__}")
        print(f"CPU threads: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
        driver = f"HIP/ROCM {torch.version.hip}" if torch.version.hip else f"CUDA: {torch.version.cuda}"
        print(f"GPU : {torch.cuda.get_device_name()}. {driver}.")

    set_torch_flags(cfg)  # should come before fabric setup
    # Next we set up the fabric and logger.
    fabric = setup_fabric(cfg)

    # Now we call the main function with the fabric and cfg.
    state = startup(fabric, cfg)

    # Now we call the train function with the fabric, state, and dataloaders.
    train_time = time.time()
    train(fabric, state, cfg)

    # Now exit
    fabric.print("--------------------------------------------------------------------")
    fabric.print(f"Training time: {str(datetime.timedelta(seconds=time.time() - train_time))} ")
    fabric.log_to_summary(
        {"train_time": time.time() - global_start_time, "total_time": time.time() - global_start_time}
    )
    if fabric.device.type == "cuda":
        max_alloc = f"{torch.cuda.max_memory_allocated(fabric.device) / float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(fabric.device) / float(1024**3):,.3f} GB"
        fabric.print(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
    fabric.print("--------------------------------------------------------------------")
    if torch.distributed.is_initialized():
        # torch.distributed.barrier()  # this could be very good or very bad
        torch.distributed.destroy_process_group()  # Force a clean exit
    if int(os.getenv("SLURM_PROCID", "0")) == 0:
        print(f"Run {cfg.run_name} finished without error.")
        print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
        print("-----------------Shutdown complete.--------------------------")


def guarded_main():
    try:
        main()
    except BaseException:  # gate around hell to guarantee NCCL deconstruction
        if torch.distributed.is_initialized():
            # torch.distributed.barrier()  # this could be very good or very bad
            torch.distributed.destroy_process_group()  # Force a clean exit
        if int(os.getenv("SLURM_PROCID", "0")) == 0:
            print("Run finished with errors.")
            print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
            print("-----------------Shutdown complete.--------------------------")

            raise


if __name__ == "__main__":
    guarded_main()
