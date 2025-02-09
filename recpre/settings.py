import os
import json
import torch

from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Union, Optional, Any, Literal

from recpre.config_dynamic import Config as DynamicConfig, AnyConfig

from transformers import AutoModelForCausalLM, AutoConfig


@dataclass
class HuggingfaceConfig:
    """need to properly merge HF one day"""

    name: str
    checkpoint: Optional[str]
    block_size: Optional[int] = None
    strategy: Optional[str] = None
    abacus_ids: list[int] = field(default_factory=lambda: list(range(10)))  # Will be initialized correctly later

    @property
    def Block(self):
        if "llama" in self.name.lower():
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer

            return LlamaDecoderLayer
        else:
            raise ValueError("Provide the block name for this architecture.")

    def construct_model(self, objective, gradient_checkpointing: bool, **kwargs) -> torch.nn.Module:
        from axonn.models.transformers import parallelize

        source = self.checkpoint or self.name
        with parallelize(source) if self.strategy == "axonn_tp" else nullcontext():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(source))

        if gradient_checkpointing:
            model.enable_gradient_checkpointing()
        return model


@dataclass
class DataEntry:
    type: str
    prefix: str
    weight: int = 1
    data_signature: Optional[dict[str, list[str] | str]] = None
    name: Optional[str] = None
    data_dir: Optional[str] = None
    text_key: Optional[str] = None
    repetitions: Optional[int] = None
    max_epoch: Optional[int] = None
    scheduler: Optional[tuple[str, int]] = None
    return_data_id: bool = False


@dataclass
class GoldfishConfig:
    k_token_loss_dropout: int = 4
    start_position: int = 0
    context_width: int = 13
    strategy: Optional[str] = None  # off by default, set to "hash-table" or "hash-avalanche" to enable


@dataclass
class FabricConfig:
    optimize_communication: bool = False
    all_reduce_dtype: Optional[str] = None
    row_tensor_parallel_size: int = 1
    col_tensor_parallel_size: int = 1
    depth_tensor_parallel_size: int = 1
    optim_sharding: bool = False
    allow_optim_fusion: bool = False
    use_apex_adamw: bool = False
    fsdp_use_original_params: bool = False


@dataclass
class CLISettings:
    # Main settings
    run_name: str = "default-run"  # The name for logging.
    out_dir: str = None  # type: ignore # The directory to save checkpoints. Required to be given or set as OUT_DIR
    resume: bool = True  # Whether to resume from a checkpoint in the out_dir.
    max_tokens: int = 1_000_000_000_000  # The maximum number of tokens to train on (determines max_iters).
    max_steps: Optional[int] = None  # Set max_tokens=0 if setting max_steps
    seed: int = 1337  # The random seed to use for reproducibility.

    # Model configuration
    model_name: str = "tiny-llama-1.1b"  # The model name to use when creating the model from config.py / config_dynamic
    model_impl: str = "recpre"  # The model name to use when creating the model from config.py
    block_size: int = 2048  # The block size to use (lit-gpt-ese for sequence length).
    ignore_block_size_mismatch: bool = False  # Whether to ignore block size mismatch.
    model_checkpoint: Optional[str] = None  # The model checkpoint to load. Else, from config.
    doc_block_attn: bool = False  # Whether to mask out the attention between tokens from different documents.
    cache_attn: bool = False  # Whether to train the model with cache attention with cache tokens randomly inserted.
    eod_token: Optional[str] = None  # 'eos','bos','pad' The end-of-document token name (used for doc-block-attn).

    # Training hyperparameters
    world_batch_size: int = 2048  # The total batch size across all devices and nodes.
    batch_size_ramp: int = 0  # Over how many mbs steps to linearly increase the batch size to world_batch_size
    optimizer: str = "AdamW"
    optim_config: dict[str, Any] = field(
        default_factory=lambda: dict(
            lr=0.0004,  # The learning rate.
            weight_decay=0.1,  # The weight decay.
            betas=(0.9, 0.95),  # The beta parameters for the Adam optimizer.
            eps=1e-8,  # The eps parameter for the Adam optimizer
        )
    )
    grad_clip: float = 1.0  # The gradient clipping value.
    warmup_steps: int = 0  # The number of warmup steps.
    cooldown_steps: int = 0  # The number of cooldown steps.
    lr_schedule: str = "cosine"  # The learning rate schedule to use.
    min_lr: float = 0.00004  # The minimum learning rate to decay to.
    no_weight_decay_for_bias_and_norm_params: bool = False  # do not use weight decay for bias and norm params

    # Objective and Regularization
    label_smoothing: float = 0.0
    z_regularization: float = 0.0
    goldfish: GoldfishConfig = field(default_factory=lambda: GoldfishConfig())

    # Implementation and backend
    fabric_strategy: str = "ddp"  # The fabric strategy to use: ddp, fsdp, axonn_tp.
    fabric_precision: Literal["bf16-true", "bf16-mixed", "16-mixed", "16", "32"] = "bf16-mixed"
    fabric_use_lightning_environment: bool = False  # If False, use the auto setting, True, use LightningEnvironment.
    fabric: FabricConfig = field(
        default_factory=lambda: FabricConfig(
            optimize_communication=False,
            all_reduce_dtype=None,
            row_tensor_parallel_size=1,  # The size of the row tensor parallel dimension
            col_tensor_parallel_size=1,  # The size of the col tensor parallel dimension
            depth_tensor_parallel_size=1,  # The size of the depth tensor parallel dimension
            optim_sharding=False,  # zero-1, activated directly in pytorch. May not play nicely with non-ddp
            allow_optim_fusion=False,  # fishes for fusion opportunities in the optimizer
            fsdp_use_original_params=False,
        )
    )
    micro_batch_size: int = 4  # The micro batch size to use.
    compile_model: bool = False  # Whether to compile the model.
    matmul_precision: str = "high"  # enable tf32 acc on cuda with this
    dataloader_num_workers: int = 0  # The number of workers to use for the dataloaders.
    n_chunks: int = 4  # The number of chunks to preload at a time from packed dataset.
    gradient_checkpointing: bool = False  # Whether to use activation checkpointing
    allow_nonfinite_loss: bool = False  # whether to end training immediately if non-finite loss is encountered
    compiled_autograd: bool = False
    compile_optimizer: bool = False
    dynamo_ddp_config: Literal["ddp_optimizer", "python_reducer", "no_optimization"] = "ddp_optimizer"
    loss_guardrail_active: bool = False
    skip_nonfinite_grads: bool = False
    fail_instead_of_recompile: bool = False  # code fails instead of recompiling
    # us this option to prevent dist jobs wasting time with cache failures

    # Logging
    logger_name: str = "wandb"  # The logger to use for logging, only supports "wandb" for now.
    logger_project: str = "tinyllama"  # The logger/wandb project to log to.
    data_telemetry: bool = False  # Data telemetry switch, set based on needs.
    model_telemetry: bool = True  # Whether to monitor important model values to look for spikes. May increase overhead
    shape_watching_steps: int = 3  # Number of iterations to watch shapes for. Set to 0 to disable.
    log_step_interval: int = 1  # The base interval for logging (scales with gradient_accumulation_steps).
    eval_iters: int = 100  # The number of iterations to process during a validation loop.
    save_step_interval: int = 2000  # The number of iterations between saving.
    eval_step_interval: int = 2000  # The number of iterations between evaluating.
    save_first_step: bool = False  # Whether to save the checkpoint at the first step
    save_last_step: bool = False  # Whether to save the checkpoint at the last step
    save_n_min_before_job_done: Optional[int] = None  # Save the checkpoint n minutes before current job done
    measure_utilization: bool = True  # Print FLOPs and MFU
    partial_depth_eval: list[int] = field(default_factory=list)  # don't merge this into main

    # Data Handling
    # PKDS arguments:
    shuffle_filenames: bool = True  # (PKDS only.) Shuffle filenames glob'd up for each prefix
    shuffle_blocks: bool = True  # (PKDS only.) Whether to shuffle the blocks in files.
    # HFDS arguments:
    pad_to_block_size: bool = False  # Whether to pad to the block size (HFDS only).
    add_bos: bool = True  # Whether to add the BOS token to the input (HFDS only).
    add_eos: bool = True  # Whether to add the EOS token to the input (HFDS only).
    data_signature: dict[str, list[str] | str] = field(
        default_factory=lambda: {"keys": ["text"], "format_fn": "pass_text"}
    )  # The data signature to use for processing rows of the dataset. Can be set individually per dataset. (HFDS only).
    # For both backends:
    collate_checks_enabled: bool = True  # Enable checks for the collate function.
    all_block_size_tensors: bool = False  # Assume all datasets return tensors with the same size, may reduce latency.
    use_chat_template: bool = False  # Whether to use the chat template in the collator.
    data_config: dict[str, list[DataEntry]] = field(
        default_factory=lambda: {
            "train_data": [DataEntry("pkds", "", 1)],
            "val_data": [DataEntry("pkds", "", 1)],
        }
    )
    # The directories containing the training/validation data.
    train_data_dir: str = ""
    val_data_dir: str = ""
    # The path to the tokenizer to use [required to identify pad_token_id even for pkds]
    tokenizer_path: str = "/lustre/orion/csc569/scratch/jgeiping/tokenizers/huginn_tokenizer_65k"
    model_config: Union[AnyConfig, HuggingfaceConfig] = field(init=False)
    model_overwrite: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate arguments
        if self.out_dir is None:
            self.out_dir = os.getenv("OUTPUT_DIR", "NOT_FOUND")
        assert self.out_dir != "NOT_FOUND"
        assert self.tokenizer_path, "Tokenizer has to be specified."

        # If data_config is a string, load it from a file.
        self.data_config = self._validate_data_config()
        self._expand_paths()

        # Tensor parallelism is implemented by the AxoNN fabric only.
        if (
            self.fabric.depth_tensor_parallel_size > 1
            or self.fabric.row_tensor_parallel_size > 1
            or self.fabric.col_tensor_parallel_size > 1
        ):
            assert self.fabric_strategy == "axonn_tp", "x_tensor_parallel_size > 1 implies use of axonn_tp."

        self._parse_environment_variables()

        # Add any derived cfg here
        self.node_batch_size = self.world_batch_size // self.num_nodes
        self.loader_block_size = self.block_size + 1
        self.global_total_time = 0
        self.max_tokens_per_device = 0
        self.tokens_per_step = 0

        self.batch_size = self.node_batch_size // self.devices
        if self.batch_size_ramp == 0:
            self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size
        else:
            self.gradient_accumulation_steps = 1
        self.replicas = self.devices * self.num_nodes

        self.dataset_names = [i.prefix for i in self.data_config["train_data"]]
        self.train_dataset_prefixes = [ds.prefix for ds in self.data_config["train_data"]]
        self.val_dataset_prefixes = (
            [ds.prefix for ds in self.data_config["val_data"]] if "val_data" in self.data_config else []
        )

        self._validate_args()

        # Finally, store model config object itself
        self.model_config = DynamicConfig.from_name(self.model_name, **self.model_overwrite)

        # Set strategy
        self.model_config.strategy = self.fabric_strategy

    def _validate_args(self):
        assert (self.max_steps is not None) ^ (self.max_tokens > 0), (
            f"only max_steps ({self.max_steps}) xor max_tokens ({self.max_tokens}) can be specified"
        )
        assert len(set(self.dataset_names)) == len(self.data_config["train_data"]), (
            "please provide different names for each subset"
        )

        # Any additional sanity checks here.
        assert self.gradient_accumulation_steps > 0, "derived gradient_accumulation_steps must be > 0"
        if self.batch_size_ramp == 0:
            assert (
                self.world_batch_size
                == self.micro_batch_size * self.gradient_accumulation_steps * self.devices * self.num_nodes
            ), "world batch size should be: micro_batch_size * gradient_accumulation_steps * devices * num_nodes"
        else:
            assert self.world_batch_size % (self.micro_batch_size * self.devices * self.num_nodes) == 0

    def _expand_paths(self):
        self.train_data_dir = os.path.expandvars(self.train_data_dir) if self.train_data_dir is not None else ""
        self.val_data_dir = os.path.expandvars(self.val_data_dir) if self.val_data_dir is not None else ""
        for entry in self.data_config["train_data"] + self.data_config["val_data"]:
            if entry.data_dir is not None:
                entry.data_dir = os.path.expandvars(entry.data_dir)

    def _parse_environment_variables(self):
        """Parse env variables and directly store as non-field attributes"""
        self.SLURM_JOB_ID = int(os.getenv("SLURM_JOB_ID", 0))
        self.SLURM_ARRAY_JOB_ID = int(os.getenv("SLURM_ARRAY_JOB_ID", 0))
        self.SLURM_ARRAY_TASK_ID = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
        self.SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))
        self.MASTER_ADDR = os.getenv("MASTER_ADDR", "0")
        self.MASTER_PORT = int(os.getenv("MASTER_PORT", 0))
        self.WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
        self.RANK = int(os.getenv("SLURM_PROCID", "0"))
        self.devices = int(os.getenv("SLURM_NTASKS_PER_NODE", torch.cuda.device_count()))
        self.num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES", 1))

    def _validate_data_config(self) -> dict[str, list[DataEntry]]:
        if isinstance(self.data_config, str):
            try:
                with open(self.data_config, mode="r") as json_file:
                    self.data_config = json.load(json_file)
            except Exception as e:
                raise ValueError(
                    f"data_config passed was a string, but failed to load as a json object from {self.data_config}: {e}"
                )
        return self.data_config
