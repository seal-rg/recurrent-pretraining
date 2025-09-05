"""Simple(ish), self-contained finetuning script. Training on GSM8k like in this example will not improve flexible extract,
but the model will quickly learn the format and strict match will rise.

built around minimal train.py variant

Almost all of the credit for this file goes to SeanMcLeish.
"""

####################################################################################################
# Imports.
####################################################################################################

import time

global_start_time = time.time()
import os
import socket
import json

from typing import TYPE_CHECKING, Any, Optional
import sys
import datetime
import shutil

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset, Dataset, load_from_disk
from contextlib import nullcontext

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

USE_LOCAL_CODE = True
if USE_LOCAL_CODE:
    import litgpt  # noqa


# Check device health immediately after loading torch and standard libraries without loading cuda/hip/dist:
nvml_count = torch.cuda._device_count_amdsmi() if torch.version.hip else torch.cuda._device_count_nvml()
if nvml_count < 1:
    raise ValueError(f"Node failure! Device manager init failed on {socket.gethostname()}")


if TYPE_CHECKING:
    import torch.distributed
    import torch.version
    import torch._dynamo.config


from dataclasses import dataclass, field
from jsonargparse import CLI


end_time = time.time()
if int(os.getenv("SLURM_PROCID", "0")) == 0:
    print(f"{time.ctime()[:-5]}: Time to load libraries: {end_time - global_start_time:.02f} seconds.")


@dataclass
class CLISettings:
    run_name: str = "default-run"
    out_path: str = "outputs"
    # data
    dataset_location: str = "openai/gsm8k"
    model_name: str = "tomg-group-umd/huginn-0125"
    dataset_args: dict[str, Any] = field(default_factory=lambda: dict(q_col="question", a_col="answer"))
    dataset_config: str = "main"
    max_seq_length: int = 128
    max_samples: Optional[int] = None
    # impl
    micro_batch_size: int = 2
    compile: bool = False
    # log_interval: int = 8
    # training
    max_steps: int = 0
    epochs: int = 1
    batch_size: int = 32
    optim_config: dict[str, Any] = field(
        default_factory=lambda: dict(lr=5e-7, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
    )
    scheduler_args: dict[float, Any] = field(default_factory=lambda: dict(warmup=0.1, cooldown=0.1, min_lr_ratio=0.001))  # type: ignore # min_lr = min_lr_ratio * lr
    eval_interval: int = 1_000_000_000
    seed: int = 74
    take_loss_over_all_tokens: bool = False  # for chat templated datasets default is to only supervise assistant tokens
    max_grad_norm: float = 1_000_000.0  # i.e. unused unless something is going very wrong
    precision: str = "bf16-true"
    gradient_checkpointing: bool = False
    save_final_checkpoint: bool = False

    def __post_init__(self):
        pass


@dataclass
class Message:
    role: str
    content: str


def is_main_process():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True


def seed_everything(seed):
    import random  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_unwrapped_model(state):
    return state["model"].module if state["distributed"] else state["model"]


####################################################################################################
# Main driver functions.
####################################################################################################
DEFAULT_SYS_PROMPT = "You are a helpful assistant that can assist users with mathematical reasoning."


def startup(cfg: CLISettings):
    """The main setup function for the training script."""
    seed_everything(cfg.seed)
    ##########    Comms              ##############
    rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
    local_device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        distributed = True
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=int(os.getenv("SLURM_NTASKS", os.getenv("WORLD_SIZE", -1))),
            device_id=local_device,
            timeout=datetime.timedelta(hours=2),
        )
        world_size = torch.distributed.get_world_size()
        print(f"Comms formed on rank {rank} with device {local_device} out of world size {world_size}.")
    else:
        world_size = 1
        distributed = False
    torch.cuda.set_device(local_device)

    if cfg.precision == "bf16-true":
        torch.set_default_dtype(torch.bfloat16)
        weight_dtype = torch.bfloat16
        autocast_args = {"device_type": "cuda", "enabled": False, "dtype": torch.bfloat16}
    elif cfg.precision == "bf16-mixed":
        torch.set_default_dtype(torch.float32)
        weight_dtype = torch.float32
        autocast_args = {"device_type": "cuda", "enabled": True, "dtype": torch.bfloat16}

    ########## Model and tokenizer ##############
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=not USE_LOCAL_CODE,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # model.config.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ##########  Distribute model   ##############
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_device], find_unused_parameters=not cfg.compile, gradient_as_bucket_view=True
        )
    if cfg.compile:
        model = torch.compile(model, fullgraph=False, dynamic=False, mode="max-autotune-no-cudagraphs")
    ##########     Optimizer       ##############
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optim_config)

    ##########     Data            ##############
    def format_and_tokenize_examples(examples):
        conversations = []
        for idx in range(len(examples[cfg.dataset_args["q_col"]])):
            if cfg.dataset_args["q_col"] != "text":
                messages = [
                    Message(role="system", content=DEFAULT_SYS_PROMPT),
                    Message(role="user", content=examples[cfg.dataset_args["q_col"]][idx].strip()),
                    Message(role="Huginn", content=examples[cfg.dataset_args["a_col"]][idx].strip()),
                ]
            else:
                messages = tokenizer.bos_token + examples[cfg.dataset_args["q_col"]][idx].strip()
            conversations.append(messages)

        if cfg.dataset_args["q_col"] != "text":
            chat_encoding = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                padding="max_length",
                max_length=cfg.max_seq_length + 1,
                return_tensors="pt",
                return_dict=True,
                truncation=True,
            )
            if cfg.take_loss_over_all_tokens:
                chat_encoding["assistant_masks"] = chat_encoding["attention_mask"]
        else:
            chat_encoding = tokenizer(
                conversations,
                padding="max_length",
                max_length=cfg.max_seq_length + 1,
                return_tensors="pt",
                truncation=True,
            )
            chat_encoding["assistant_masks"] = chat_encoding["attention_mask"].clone()

        return {
            "input_ids": chat_encoding["input_ids"],
            "mask": chat_encoding["assistant_masks"],
            "attention_mask": chat_encoding["attention_mask"],
        }

    cfg.token_id_col_name = "input_ids"  # type: ignore
    dataset_save_dir = f"{cfg.out_path}/{cfg.run_name}/dataset"
    if is_main_process():  # only do mapping on rank 0
        try:
            dataset: Dataset = load_dataset(cfg.dataset_location, cfg.dataset_config)["train"]  # type: ignore
        except BaseException:
            dataset: Dataset = load_from_disk(cfg.dataset_location, cfg.dataset_config)  # type: ignore

        if cfg.max_samples is not None:
            dataset = dataset.select(range(cfg.max_samples))

        if os.path.exists(dataset_save_dir):  # delete any old dataset
            shutil.rmtree(dataset_save_dir)

        tokenized_dataset = dataset.map(
            format_and_tokenize_examples,
            num_proc=16,
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=1024,
        )

    if distributed:  # load the dataset to other ranks
        if is_main_process():
            tokenized_dataset.save_to_disk(dataset_save_dir)
        torch.distributed.barrier()
        tokenized_dataset = load_from_disk(dataset_save_dir)
        torch.distributed.barrier()

    if rank == 0:
        idx = int(torch.randint(len(tokenized_dataset), (1,)))
        print(f"-----------------------------------Processed Data example idx {idx}:----------------------------")
        print(tokenized_dataset[idx])
        print(tokenizer.decode(tokenized_dataset[idx]["input_ids"], skip_special_tokens=False))
        print("--------------------------------------------------------------------------------------------")
    tokenized_dataset.set_format("pt")
    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            tokenized_dataset,  # type: ignore
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            seed=cfg.seed,
        )
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,  # type: ignore
            batch_size=cfg.micro_batch_size,
            sampler=sampler,
            pin_memory=True,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,  # type: ignore
            batch_size=cfg.micro_batch_size,
            shuffle=True,
            pin_memory=True,
        )

    ##########     Scheduler       ##############
    accumulation_steps = max(1, cfg.batch_size // cfg.micro_batch_size)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
    max_training_steps = cfg.epochs * num_update_steps_per_epoch
    num_warmup_steps = math.ceil(cfg.scheduler_args["warmup"] * max_training_steps)  # type: ignore
    num_decay_steps = math.ceil(cfg.scheduler_args["cooldown"] * max_training_steps)  # type: ignore
    scheduler = get_scheduler(
        name="warmup_stable_decay",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_training_steps,
        scheduler_specific_kwargs={
            "num_decay_steps": num_decay_steps,
            "min_lr_ratio": cfg.scheduler_args["min_lr_ratio"],  # type: ignore
        },
    )

    state = {
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        "dataloader": dataloader,
        "distributed": distributed,
        "rank": rank,
        "scheduler": scheduler,
        "autocast_args": autocast_args,
    }

    cfg.world_size = world_size  # type: ignore
    return state, local_device


def train(state, device, cfg):
    model, optimizer = state["model"], state["optimizer"]
    model.train()

    accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    optimizer_step = 0
    step_time = time.time()
    total_tokens = 0
    total_tokens_with_loss = 0
    tokens_in_step = 0

    metrics_to_agg_data_step = {
        "loss": [],
        "log_ppl": [],
    }

    for epoch in range(cfg.epochs):
        for data_step, inputs in enumerate(state["dataloader"], start=1):
            # Realize the input and labels tensors.
            input_ids = inputs[cfg.token_id_col_name][:, :-1].to(dtype=torch.long, device=device, non_blocking=True)
            # Need to take into account the assistant and attention if sequences are being padded
            mask = ~(inputs["mask"].bool() & inputs["attention_mask"].bool())

            labels = torch.where(mask[:, 1:], -100, inputs[cfg.token_id_col_name][:, 1:]).to(
                dtype=torch.long, device=device, non_blocking=True
            )
            total_tokens_with_loss += (labels != -100).sum().item()
            tokens_in_step += input_ids.numel()
            is_accumulating = data_step % accumulation_steps != 0

            # The actual compute step of  Forward, loss, and backward computation:
            def tightly_scoped_fwd_bwd(model, input_ids, labels):
                with model.no_sync() if is_accumulating and state["distributed"] else nullcontext():
                    with torch.autocast(**state["autocast_args"]):
                        outputs = model(input_ids, labels=labels)
                    (outputs["loss"] / accumulation_steps).backward()
                    return (outputs["loss"].detach(), outputs["log_ppl"].detach())

            loss, log_ppl = tightly_scoped_fwd_bwd(model, input_ids, labels)

            # logging
            metrics_to_agg_data_step["loss"].append(loss.item())
            metrics_to_agg_data_step["log_ppl"].append(log_ppl.item())

            if not is_accumulating:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.max_grad_norm, norm_type=2.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state["scheduler"].step()
                optimizer_step += 1

                if state["rank"] == 0:
                    time_interval = (time.time() - step_time) / accumulation_steps
                    tok_sec = tokens_in_step * cfg.world_size / (time.time() - step_time)
                    print(
                        f"GPU: {model.device} | Step: {data_step:4d} | Updates: {optimizer_step:4d} | Time/step: {time_interval:2.4f}"
                        f" | Tok/sec={tok_sec:9.2f} | Loss: {loss:2.4f} / log-ppl: {log_ppl:2.4f} | Grad-Norm {total_norm.item():2.4f}"
                    )
                    total_tokens += tokens_in_step * cfg.world_size
                    step_time = time.time()
                    tokens_in_step = 0

            if optimizer_step and (optimizer_step % cfg.eval_interval == 0):
                validate(state, optimizer_step, cfg)

            if cfg.max_steps and data_step >= cfg.max_steps:
                break

    model.eval()
    return state


####################################################################################################
# Main control loop
####################################################################################################


def validate(state, step, cfg, task="gsm8k"):
    # eval on-the-fly
    unwrapped_model = get_unwrapped_model(state)
    unwrapped_model.eval()
    results = evaluator.simple_evaluate(
        model=HFLM(
            pretrained=unwrapped_model,
            tokenizer=state["tokenizer"],
            batch_size=16,  # 16: 4:41
        ),
        tasks=[task],
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        system_instruction=DEFAULT_SYS_PROMPT,
        limit=100,
        # batch_size=13,
        num_fewshot=0,
        gen_kwargs={"num_steps": unwrapped_model.config.mean_recurrence},
    )
    print(make_table(results))
    results_by_step = {}
    if results is not None:
        if "groups" in results:
            print(make_table(results, "groups"))
        results_by_step[str(step)] = results["results"][task]

    os.makedirs(f"{cfg.out_path}/{cfg.run_name}", exist_ok=True)
    with open(f"{cfg.out_path}/{cfg.run_name}/eval.json", "a") as f:
        json.dump(results_by_step, f)

    unwrapped_model.train()


def main():
    """Encapsulates main scope away from import calls."""

    # Configuration loader
    cfg: CLISettings = CLI(CLISettings)

    # Print system setup
    if is_main_process():
        print("--------------------------------------------------------------------")
        print(f"------------------ Launching run {cfg.run_name}------------------")
        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")
        print(f"Platform: {sys.platform}, Python: {sys.version.split(' (')[0]}, PyTorch: {torch.__version__}")
        print(f"CPU threads: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
        driver = f"HIP/ROCM {torch.version.hip}" if torch.version.hip else f"CUDA: {torch.version.cuda}"
        print(f"GPU : {torch.cuda.get_device_name()}. {driver}.")

    # set flags
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway
    torch._dynamo.config.optimize_ddp = "python_reducer"
    torch._dynamo.config.compiled_autograd = False

    train_time = time.time()

    state, device = startup(cfg)
    state = train(state, device, cfg)
    # validate(state, "final", cfg)

    if cfg.save_final_checkpoint:
        unwrapped_model = get_unwrapped_model(state)
        unwrapped_model.save_pretrained(f"{cfg.out_path}/{cfg.run_name}/final_checkpoint")
        state["tokenizer"].save_pretrained(f"{cfg.out_path}/{cfg.run_name}/final_checkpoint")

    # Now exit
    if is_main_process():
        print("--------------------------------------------------------------------")
        print(f"Training time: {str(datetime.timedelta(seconds=time.time() - train_time))} ")
        max_alloc = f"{torch.cuda.max_memory_allocated(device) / float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(device) / float(1024**3):,.3f} GB"
        print(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
        print("--------------------------------------------------------------------")
        dataset_save_dir = f"{cfg.out_path}/{cfg.run_name}/dataset"
        if os.path.exists(dataset_save_dir):
            shutil.rmtree(dataset_save_dir)


def shutdown():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
    print("-----------------Shutdown complete.--------------------------")


def guarded_main():
    try:
        run_name = main()
        print("--------------------------------------------------------------------")
        print(f"Run {run_name} finished without error.")
    except BaseException:
        print("--------------------------------------------------------------------")
        print("Run finished with errors.")
        raise
    finally:
        shutdown()  # guarantee NCCL deconstruction


if __name__ == "__main__":
    guarded_main()
