# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union, Callable, Self, TYPE_CHECKING
from functools import partial
from collections import defaultdict
import contextlib

import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy


if TYPE_CHECKING:
    import torch.utils.checkpoint

from recpre.utils import find_multiple
from recpre.model_registry import name_to_config, configs
from recpre.init import Init, init_normal
from recpre.attention_backends import select_attention_implementation

# patch checkpoint+meta
from .checkpoint_patch import _checkpoint_without_reentrant_generator

torch.utils.checkpoint._checkpoint_without_reentrant_generator = _checkpoint_without_reentrant_generator


@dataclass
class RoPESettings:
    use_rope: bool = True
    rope_condense_ratio: int = 1
    rope_base: int = 50_000


@dataclass
class Config:
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    architecture_class_name: Literal["GPT", "RecurrentGPT"] = "GPT"

    # Core
    block_size: int = 4096  # max_seq_len
    n_embd: int = 4096
    intermediate_size: Optional[int] = None  # type: ignore
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # type: ignore # new GQA notation oriented at Llama

    # Word+Pos Embedding
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    rope_settings: RoPESettings = field(default_factory=lambda: RoPESettings())
    use_abacus: bool = False
    abacus_ids: list[int] = field(default_factory=lambda: list(range(10)))  # Will be initialized correctly later
    # set randomize_positions_from to an integer greater than block_size to draw pos_ids from the entire range:
    randomize_positions_from: Optional[int] = None

    # Main blocks
    block_class_name: str = "TransformerPreNormBlock"
    norm_class_name: str = "LayerNorm"

    # Block details
    attn_impl: Literal["sdpa", "amd", "openai", "triton-kernels", "mosaic", "tridao", "debug-skip"] = "sdpa"
    norm_eps: float = 1e-5
    mlp_class_name: str = "BaseMLP"
    nonlin_name: str = "GELU"  # draws from torch.nn first
    bias: bool = False
    qk_bias: bool = False
    lm_head_bias: bool = False
    init_strategy: str = "scaled"
    init_orthogonal: bool = False
    skip_initialization: bool = False

    mup_model_scaling_factor: int = 1  # use this to scale model width+lr+logit_scale

    use_fused_head: Literal["hhe", "cce", "full-triton", "pytorch"] = "pytorch"

    # Specific to some model families, usually unused:
    loss_shape: str = "none"
    mod_capacity_factor: float = 0.125

    debias_attention: bool = False
    center_attention: bool = False

    clip_qkv: Optional[float] = None  # 8 in olmo1.7
    qk_norm: bool = False

    # Implementation handles
    activation_checkpoint_impl: str = "per-block"
    simple_ops: bool = False  # Choose naive implementations where flops can be traced more easily (never use in prod)
    strategy: str = "single"  # which device strategy is being used

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # Validate kv heads versus all heads
        if self.num_key_value_heads is not None:
            assert self.num_attention_heads % self.num_key_value_heads == 0
        else:
            self.num_key_value_heads: int = self.num_attention_heads
        assert self.n_embd % self.num_attention_heads == 0
        self.head_size = self.n_embd // self.num_attention_heads
        self.n_head = self.num_attention_heads  # for compatibility with config.py
        self.n_query_groups = self.num_key_value_heads  # for compatibility with config.py

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size: int = 4 * self.n_embd

        # SCALE architecture definition
        self.n_embd *= self.mup_model_scaling_factor
        self.intermediate_size *= self.mup_model_scaling_factor
        self.n_query_groups *= self.mup_model_scaling_factor
        self.num_key_value_heads *= self.mup_model_scaling_factor
        self.n_head *= self.mup_model_scaling_factor

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> "AnyConfig":
        if name not in name_to_config:
            # search through all `config['hf_config']['name']`
            try:
                conf_dict = next(config for config in configs if name == config["hf_config"]["name"])
            except StopIteration:
                raise ValueError(f"{name!r} is not a supported config name")
        else:
            conf_dict = name_to_config[name]

        conf_dict = conf_dict.copy()
        rope_settings = {}
        for key, value in kwargs.items():
            if "rope_settings" in key:
                rope_key = key.split(".", 1)[1]
                rope_settings[rope_key] = value
            else:
                conf_dict[key] = value
        if rope_settings:
            conf_dict["rope_settings"] = RoPESettings(**rope_settings)

        if conf_dict["architecture_class_name"] == "GPT":
            return GPTConfig(**conf_dict)
        elif conf_dict["architecture_class_name"] == "RecurrentGPT":
            return RecurrentConfig(**conf_dict)
        else:
            raise ValueError(f"Invalid architecture {conf_dict['architecture_class_name']} provided.")

    def construct_model(self, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    @classmethod
    def from_json(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            json_kwargs = json.load(fp)
        json_kwargs.update(kwargs)
        return cls(**json_kwargs)

    @classmethod
    def from_checkpoint(cls, path: Path, **kwargs: Any):
        """Automatically load `lit_config.json` and if it doesn't exist - a matching config from `recpre/config.py`."""
        if (config_path := path / "lit_config.json").is_file():
            return cls.from_json(config_path, **kwargs)
        if (model_name := path.name) in name_to_config:
            return cls.from_name(model_name, **kwargs)
        raise FileNotFoundError(f"For {str(path)!r} neither 'lit_config.json' nor matching config exists.")

    @property
    def MLP(self) -> Type[torch.nn.Module]:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        import recpre.model_dynamic

        return getattr(recpre.model_dynamic, self.mlp_class_name)

    @property
    def Linear(self) -> Type[torch.nn.Module]:
        if self.strategy == "axonn_tp" and not self.simple_ops:
            # Load different module for axonn tensor parallel
            from axonn.intra_layer import Linear as TensorParallelLinear

            return TensorParallelLinear
        else:
            return Linear

    @property
    def Block(self) -> Type[torch.nn.Module]:
        import recpre.model_dynamic

        return getattr(recpre.model_dynamic, self.block_class_name)

    @property
    def Nonlin(self) -> Type[torch.nn.Module]:
        try:
            return getattr(torch.nn, self.nonlin_name)
        except AttributeError:
            if self.nonlin_name == "ReLU2":
                return Relu2
            else:
                raise ValueError(f"Could not identify nonlinearity {self.nonlin_name}")

    @property
    def Norm(self) -> Union[Type[torch.nn.Module], Callable]:
        if not self.simple_ops:
            try:
                import recpre.norms

                norm_fn = getattr(recpre.norms, self.norm_class_name)
                if "Gemma" in self.name:
                    return partial(norm_fn, add_unit_offset=True)
                else:
                    return norm_fn
            except AttributeError:
                return getattr(torch.nn, self.norm_class_name)
        else:
            import recpre.norms

            return recpre.norms.RMSNorm

    @property
    def attn_nonlin_fn(self):
        provider = str(self.attn_impl)
        center_attention = bool(self.center_attention)
        debias_attention = bool(self.debias_attention)
        with contextlib.suppress(ModuleNotFoundError):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)  # flag is not yet implemented on earlier pytorch versions

        if not self.simple_ops:
            return partial(
                select_attention_implementation,
                provider=provider,
                center=center_attention,
                debias=debias_attention,
            )
        else:
            return partial(
                select_attention_implementation,
                provider="sdpa",
                center=center_attention,
                debias=debias_attention,
            )

    @property
    def checkpoint(self) -> Callable:
        """Run SAC at your own risk :<"""
        attn_ops = [
            torch.ops.aten._scaled_dot_product_efficient_attention.default,  # type: ignore
            torch.ops.aten._scaled_dot_product_flash_attention.default,  # type: ignore
        ]
        try:
            from flash_attn import flash_attn_func  # type: ignore

            attn_ops.append(flash_attn_func)
        except ImportError:
            pass
        ops_to_save = [
            torch.ops.aten.mm.default,  # type: ignore
            *attn_ops,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,  # type: ignore # from comms
        ]

        if "sac%" in self.activation_checkpoint_impl:
            frequency = int(self.activation_checkpoint_impl.split("sac%")[1][0])

            def _get_custom_policy(meta):
                def _custom_policy(ctx, func, *args, **kwargs):
                    mode = "recompute" if ctx.is_recompute else "forward"
                    mm_count_key = f"{mode}_mm_count"
                    if func == torch.ops.aten.mm.default:  # type: ignore
                        meta[mm_count_key] += 1
                    # Saves output of all compute ops, except every freq- mm
                    to_save = func in ops_to_save and not (
                        func == torch.ops.aten.mm.default and meta[mm_count_key] % frequency == 0  # type: ignore
                    )
                    return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

                return _custom_policy

            def context_fn():
                meta = defaultdict(int)
                return create_selective_checkpoint_contexts(_get_custom_policy(meta))

            return partial(
                checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
                determinism_check="none",
                context_fn=context_fn,
            )
        elif "sac-attn" in self.activation_checkpoint_impl:

            def policy_fn(ctx, op, *args, **kwargs):
                if op in attn_ops:
                    return CheckpointPolicy.MUST_SAVE
                else:
                    return CheckpointPolicy.PREFER_RECOMPUTE

            context_fn = partial(create_selective_checkpoint_contexts, policy_fn)
            return partial(
                checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
                determinism_check="none",
                context_fn=context_fn,
            )

        elif "sac" in self.activation_checkpoint_impl:

            def policy_fn(ctx, op, *args, **kwargs):
                if op in ops_to_save:
                    return CheckpointPolicy.MUST_SAVE
                else:
                    return CheckpointPolicy.PREFER_RECOMPUTE

            context_fn = partial(create_selective_checkpoint_contexts, policy_fn)
            return partial(
                checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
                determinism_check="none",
                context_fn=context_fn,
            )
        else:
            # returning context_fn can break inductor in funny ways, best not to provide it if not necessary
            return partial(checkpoint, use_reentrant=False, preserve_rng_state=False, determinism_check="none")

    # this is a bit of slop code, but ok for now
    def __getstate__(self):
        state = asdict(self)
        state["_class_name"] = self.__class__.__name__
        return state

    def __setstate__(self, state):
        if state["_class_name"] == self.__class__.__name__:
            rope_settings = RoPESettings(**state.pop("rope_settings"))
            state.pop("_class_name")
            self.__dict__.update(state)
            self.__dict__["rope_settings"] = rope_settings
            self.__post_init__()
        else:
            raise ValueError(f"Saved Architecture class name {state['_class_name']} does not match saved config.")


@dataclass
class GPTConfig(Config):
    n_layer: int = 16
    n_expert: int = 0
    n_expert_per_token: int = 0
    num_memories: int = 2048
    mod_capacity_factor: int = 1

    # olmoe:
    moe_top_k: int = 8
    moe_num_experts: int = 64
    moe_dropless: bool = True
    moe_mlp_impl: str = "sparse"
    moe_zloss_weight: float = 0.001
    moe_loss_weight: float = 0.01

    def __post_init__(self):
        super().__post_init__()

        # Define initializer object from strategy
        self.init = Init(
            self.init_strategy,
            self.n_embd,
            self.intermediate_size,
            self.head_size,
            self.n_layer,
            self.mup_model_scaling_factor,
            orthogonal=self.init_orthogonal,
            verbose=False,
            skip_reinitializing=self.skip_initialization,
        )

    def construct_model(self, **kwargs) -> torch.nn.Module:
        from recpre.model_dynamic import GPT

        return GPT(self, **kwargs)


@dataclass
class RecurrentConfig(Config):
    # Arch
    injection_type: Literal["none", "add", "linear", "ffn"] = "add"
    embed_step: bool = False  # whether to provide step information
    randomize_embed_step: bool = False
    normalize_rec: str = ""
    intermediate_noise_injection: float = 0.0
    geom_noise_injection: str = "geom"
    n_layers_in_recurrent_block: int = 4
    n_layers_in_prelude: int = 1
    n_layers_in_coda: int = 1
    state_init: str = "like-init"
    # Sampling
    sampling_scheme: str = "poisson-unbounded"
    mean_recurrence: int = 32
    mean_backprop_depth: int = 8
    lockstep_n: bool = False
    lockstep_k: bool = False
    # Objective Modification
    mcleish_throttle: bool = False
    elbayad_weighing: bool = False
    elbayad_exponent: float = 1.0  # with what power should future steps be penalized
    seers: int = 0  # not functional
    qk_norm: bool = False
    # sac to force saving of mm and sdpa, # per-iteration / per-block to change granularity:
    activation_checkpoint_impl: str = "per-iteration"
    tie_embeddings: bool = False

    def __post_init__(self):
        super().__post_init__()

        effective_expected_depth = (
            self.n_layers_in_prelude + self.n_layers_in_coda + self.n_layers_in_recurrent_block * self.mean_recurrence
        )
        self.n_layer = self.n_layers_in_recurrent_block * self.mean_backprop_depth  # for compat
        # Define initializer object from strategy
        self.init = Init(
            self.init_strategy,
            self.n_embd,
            self.intermediate_size,
            self.head_size,
            effective_expected_depth,
            self.mup_model_scaling_factor,
            orthogonal=self.init_orthogonal,
            verbose=False,
            skip_reinitializing=self.skip_initialization,
        )

    def construct_model(self, **kwargs) -> torch.nn.Module:
        from recpre.model_dynamic import RecurrentGPT

        return RecurrentGPT(self, **kwargs)


AnyConfig = Union[GPTConfig, RecurrentConfig]


class Linear(torch.nn.Linear):
    """Linear layer wrapper that unifies tensor-parallel implementation and default implementations."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, init_method=None
    ):
        self.init_method = init_method if init_method else init_normal(in_features)
        super().__init__(in_features, out_features, bias, device, dtype)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.init_method(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input, **kwargs):
        """Additional args like scatter_input from axonn are ignored in this wrapper."""
        return super().forward(input)


class Relu2(torch.nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.relu(x, inplace=self.inplace).pow(2).mul(0.5)  # mul just to be difficult? :<
