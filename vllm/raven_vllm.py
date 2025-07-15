# type: ignore
"""vLLM-compatible Huginn-3.5B model implementation.

The reference HF implementation can be found in litgpt/raven_modeling_minimal.py
"""

from typing import Iterable, Optional, Tuple
import torch
import torch.nn as nn

from vllm.attention import Attention
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear, MergedColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.compilation.decorators import support_torch_compile  # noqa: F401


from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from transformers import PretrainedConfig
from math import sqrt

"""The HuggingFace-style model configuration, replicated for vllm plugin packaging"""


class RavenConfig(PretrainedConfig):
    model_type = "huginn_raven"
    keys_to_ignore_at_inference = [""]
    attribute_map = {"num_attention_heads": "n_heads", "hidden_size": "n_embd", "num_hidden_layers": "n_layers"}

    def __init__(
        self,
        n_embd: int = 5280,
        n_heads: int = 55,
        n_layers: int = 8,  # total of prelude + recurrent + coda
        block_size: int = 4096,
        vocab_size: int = 65536,
        padding_multiple: int = 4096,
        tie_embeddings: bool = True,
        intermediate_size: int = 17920,
        bias: bool = False,
        architecture_class_name: str = "RecurrentGPT",
        block_class_name: str = "SandwichBlock",
        norm_class_name: str = "RMSNorm_llama",
        norm_eps: float = 0.000001,
        mlp_class_name: str = "GatedMLP",
        nonlin_name: str = "SiLU",
        init_strategy: str = "takase",
        init_orthogonal: bool = False,
        state_init: str = "like-init",
        injection_type: str = "linear",
        n_layers_in_recurrent_block: int = 4,
        mean_recurrence: int = 32,
        sampling_scheme: str = "poisson-lognormal-filling",
        mean_backprop_depth: int = 8,
        n_layers_in_prelude: int = 2,
        n_layers_in_coda: int = 2,
        test_time_noise: float = 0.0,
        test_time_noise_type: str = "none",
        qk_bias: bool = True,
        activation_checkpoint_impl: str = "per-iteration",
        rope_base: float = 50_000,
        torch_dtype: str = "bfloat16",
        transformers_version: str = "4.47.1",
        **kwargs,
    ):
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.vocab_size = self.padded_vocab_size = vocab_size
        self.padding_multiple = padding_multiple
        self.tie_embeddings = tie_embeddings
        self.intermediate_size = intermediate_size
        self.bias = bias
        self.architecture_class_name = architecture_class_name
        self.block_class_name = block_class_name
        self.norm_class_name = norm_class_name
        self.norm_eps = norm_eps
        self.mlp_class_name = mlp_class_name
        self.nonlin_name = nonlin_name
        self.init_strategy = init_strategy
        self.init_orthogonal = init_orthogonal
        self.state_init = state_init
        self.injection_type = injection_type
        self.n_layers_in_recurrent_block = n_layers_in_recurrent_block
        self.mean_recurrence = mean_recurrence
        self.sampling_scheme = sampling_scheme
        self.mean_backprop_depth = mean_backprop_depth
        self.n_layers_in_prelude = n_layers_in_prelude
        self.n_layers_in_coda = n_layers_in_coda
        self.qk_bias = qk_bias
        self.activation_checkpoint_impl = activation_checkpoint_impl
        self.rope_base = rope_base
        self.torch_dtype = torch_dtype  # Added from JSON
        self.transformers_version = transformers_version  # Added from JSON
        # inference
        self.test_time_noise = test_time_noise
        self.test_time_noise_type = test_time_noise_type
        # Derived
        self.num_key_value_heads = n_heads
        self.num_attention_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.effective_expected_depth = (
            self.n_layers_in_prelude + self.n_layers_in_coda + self.n_layers_in_recurrent_block * self.mean_recurrence
        )
        self.init_values = {
            "std": sqrt(2 / (5 * self.n_embd)),
            "out_proj": sqrt(2 / (5 * self.n_embd)) / sqrt(2 * self.effective_expected_depth),
            "embedding": sqrt(2 / (5 * self.n_embd)),
            "embed_scale": sqrt(self.n_embd),
        }

        super().__init__(
            pad_token_id=65509,
            bos_token_id=65504,
            eos_token_id=[65505, 65508],
            tie_word_embeddings=tie_embeddings,
            **kwargs,
        )


class RavenAttention(nn.Module):
    def __init__(
        self,
        config: RavenConfig,
        vllm_config: Optional[VllmConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.total_num_heads

        # Tensor parallel setup
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Combined QKV projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.qk_bias = nn.Parameter(torch.zeros(2, self.num_heads, self.head_dim))
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            quant_config=quant_config,
            prefix=prefix,
        )
        # self.attn.use_direct_call = True

    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply QK bias and rotary on head dim
        total_tokens = hidden_states.shape[0]
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)

        q_bias, k_bias = self.qk_bias.split(1, dim=0)
        q = (q + q_bias).to(q.dtype)
        k = (k + k_bias).to(q.dtype)
        q, k = self._apply_rotary_emb_complex_like(q, k, freqs_cis)
        # Flatten back for vllm attention
        q = q.view(total_tokens, -1)
        k = k.view(total_tokens, -1)
        attn_output = self.attn(q, k, v)
        # print(attn_output.flatten()[:5])
        output, _ = self.o_proj(attn_output)
        return output

    def _apply_rotary_emb_complex_like(
        self, q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=False):
            # Concatenate q and k on head dimension (dim=1)
            qk_concat = torch.cat([q, k], dim=1)
            qk_r2 = qk_concat.unflatten(dim=-1, sizes=(-1, 2)).float()  # cast to float32 for smooth skin
            rotated_qk_r2 = torch.stack(
                [
                    qk_r2[..., 0] * freqs_cis[..., 0] - qk_r2[..., 1] * freqs_cis[..., 1],
                    qk_r2[..., 1] * freqs_cis[..., 0] + qk_r2[..., 0] * freqs_cis[..., 1],
                ],
                -1,
            ).flatten(-2)
            q_rotated, k_rotated = torch.split(rotated_qk_r2.type_as(q), q.shape[1], dim=1)
            return q_rotated, k_rotated


class RavenMLP(nn.Module):
    def __init__(
        self,
        config: RavenConfig,
        vllm_config: Optional[VllmConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = config.intermediate_size

        # Gate and up projections combined
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        # Down projection
        self.down_proj = RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        # Activation function
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class RavenDecoderLayer(nn.Module):
    """Single decoder layer with sandwich normalization."""

    def __init__(
        self,
        config: RavenConfig,
        layer_idx: int,
        vllm_config: Optional[VllmConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Sandwich normalization layers
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm_3 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm_4 = RMSNorm(config.n_embd, eps=config.norm_eps)

        # Attention and MLP
        self.self_attn = RavenAttention(config, vllm_config, quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = RavenMLP(config, vllm_config, quant_config, prefix=f"{prefix}.mlp")

    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(self.norm_1(hidden_states), freqs_cis)
        hidden_states = self.norm_2(attn_output + hidden_states)
        hidden_states = self.norm_4(self.mlp(self.norm_3(hidden_states)) + hidden_states)

        return hidden_states


@support_torch_compile
class RavenModel(nn.Module):
    """The Raven model consisting of prelude, adaptive core, and coda layers."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

        # Embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.n_embd,
            quant_config=quant_config,
        )

        # Embedding scale
        self.embed_scale = config.init_values["embed_scale"]

        # Create all layers in a flat structure
        total_layers = config.n_layers_in_prelude + config.n_layers_in_recurrent_block + config.n_layers_in_coda

        self.layers = nn.ModuleList(
            [
                RavenDecoderLayer(config, i, vllm_config, quant_config, prefix=f"{prefix}.layers.{i}")
                for i in range(total_layers)
            ]
        )
        # Adapter layer (concatenates embeddings with current state)
        self.adapter = RowParallelLinear(
            input_size=config.n_embd * 2,
            output_size=config.n_embd,
            bias=config.bias,
            quant_config=quant_config,
            prefix=f"{prefix}.adapter",
        )

        # Final norm
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        # rope
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=False)

    def _precompute_freqs_cis(self):
        dim = self.config.n_embd // self.config.num_attention_heads
        end = self.config.block_size
        theta = self.config.rope_base
        with torch.autocast("cuda", enabled=False):
            inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            t = torch.arange(end, dtype=torch.float32, device=inv_freqs.device)
            freqs = torch.outer(t, inv_freqs).float()
            return torch.stack([torch.cos(freqs)[:, None, :], torch.sin(freqs)[:, None, :]], dim=3)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def initialize_state(self, input_embeds, scale: float = 1.0):
        """Initialize adaptive state exactly like the reference implementation."""
        x = torch.randn_like(input_embeds)
        std = self.config.init_values["std"] * scale
        if std > 0:
            with torch.no_grad():
                torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
                if self.embed_scale != 1:
                    x = x * self.embed_scale
        else:
            x.zero_()
        return x

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get embeddings
        if inputs_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        if self.embed_scale != 1.0:
            input_embeds *= self.embed_scale
        # Get rope frequencies
        freqs_cis = self.freqs_cis.index_select(0, positions)

        # Prelude layers
        for i in range(self.config.n_layers_in_prelude):
            input_embeds = self.layers[i](input_embeds, freqs_cis)

        # Initialize recurrent state
        hidden_states = self.initialize_state(input_embeds)
        for recurrent_step in range(self.config.mean_recurrence):
            # Concatenate adaptive state with input embeddings (reference pattern)
            hidden_states, _ = self.adapter(torch.cat([hidden_states, input_embeds], dim=-1))

            for i in range(self.config.n_layers_in_recurrent_block):
                hidden_states = self.layers[self.config.n_layers_in_prelude + i](hidden_states, freqs_cis)

        # Apply final norm to core
        hidden_states = self.ln_f(hidden_states)

        # Coda layers
        coda_start = self.config.n_layers_in_prelude + self.config.n_layers_in_recurrent_block
        for i in range(self.config.n_layers_in_coda):
            layer = self.layers[coda_start + i]
            hidden_states = layer(hidden_states, freqs_cis)

        return self.ln_f(hidden_states)


class RavenForvLLM(nn.Module):
    """Raven model for causal language modeling with vLLM support."""

    _supports_attention_backend = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        # Main model
        self.model = RavenModel(vllm_config=vllm_config, prefix="model" if prefix == "" else prefix)

        # Language modeling head
        if config.tie_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size, config.n_embd, quant_config=vllm_config.quant_config)

        # Logits processor and sampler
        self.logits_processor = LogitsProcessor(config.vocab_size, config.vocab_size, 1.0)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model_output = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.sampler(logits, sampling_metadata)

    def get_input_embeddings(self) -> nn.Module:
        """Get input embeddings for vLLM compatibility."""
        return self.model.embed_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from a state dict."""
        params_dict = dict(self.named_parameters())

        missing_params = []
        loaded_params = []

        for name, loaded_weight in weights:
            if "freqs_cis" in name:
                continue  # is that in there? Will be recomputed

            # Handle parameter name mapping for compatibility
            if "transformer.wte" in name:
                name = name.replace("transformer.wte.weight", "model.embed_tokens.weight")
            elif "lm_head.weight" in name and self.config.tie_embeddings:
                # If weights are tied, lm_head shares weights with embeddings
                name = "model.embed_tokens.weight"
            elif "lm_head.weight" in name:
                name = "lm_head.weight"
            elif "transformer.prelude" in name:
                # Map prelude layers to flat layer structure
                parts = name.split(".")
                layer_idx = int(parts[2])  # transformer.prelude.X.rest
                rest = ".".join(parts[3:])
                name = f"model.layers.{layer_idx}.{rest}"
            elif "transformer.core_block" in name:
                # Map core layers to flat layer structure
                parts = name.split(".")
                layer_idx = int(parts[2])  # transformer.core_block.X.rest
                rest = ".".join(parts[3:])
                # Core layers start after prelude layers
                flat_idx = self.config.n_layers_in_prelude + layer_idx
                name = f"model.layers.{flat_idx}.{rest}"
            elif "transformer.coda" in name:
                # Map coda layers to flat layer structure
                parts = name.split(".")
                layer_idx = int(parts[2])  # transformer.coda.X.rest
                rest = ".".join(parts[3:])
                # Coda layers start after prelude + core layers
                flat_idx = self.config.n_layers_in_prelude + self.config.n_layers_in_recurrent_block + layer_idx
                name = f"model.layers.{flat_idx}.{rest}"
            elif "transformer.ln_f" in name:
                name = name.replace("transformer.ln_f", "model.ln_f")
            elif "transformer.adapter" in name:
                name = name.replace("transformer.adapter", "model.adapter")

            if "attn.Wqkv.weight" in name:
                name = name.replace("attn.Wqkv.weight", "self_attn.qkv_proj.weight")
            elif "attn.proj.weight" in name:
                name = name.replace("attn.proj.weight", "self_attn.o_proj.weight")
            elif "attn.qk_bias" in name:
                name = name.replace("attn.qk_bias", "self_attn.qk_bias")

            if "mlp.fc.weight" in name:
                # HF fc layer contains both gate and up weights concatenated
                name = name.replace("mlp.fc.weight", "mlp.gate_up_proj.weight")
            elif "mlp.proj.weight" in name:
                name = name.replace("mlp.proj.weight", "mlp.down_proj.weight")

            if name in params_dict:
                param = params_dict[name]
                # Handle special case for qk bias
                if "attn.qk_bias" in name:
                    default_weight_loader(param, loaded_weight.squeeze())
                else:
                    default_weight_loader(param, loaded_weight)
                loaded_params.append(name)
            else:
                missing_params.append(name)

        if missing_params:
            print(f"Missing parameters: {missing_params[:10]}...")

    @property
    def base_model_tp_plan(self):
        """Tensor parallel plan for the base model Not required for now."""
        return {}


from vllm import ModelRegistry


def register():
    # Lazy-import is safer in multi-process contexts
    ModelRegistry.register_model("RavenForCausalLM", "raven_vllm:RavenForvLLM")


from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def compare_intermediate_outputs(hf_model, vllm_model, input_ids, num_steps=1, init_scale=0.0):
    """Compare intermediate outputs between HF and vLLM models at each stage."""
    print("\n🔍 Detailed Intermediate State Comparison")
    print("-" * 60)

    with torch.no_grad():
        # Capture HF intermediate states using hooks
        hf_states = {}

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hf_states[name] = output[0].clone().detach()
                else:
                    hf_states[name] = output.clone().detach()

            return hook

        # Register hooks for different stages
        hooks = []

        # 1. After embeddings
        hook = hf_model.transformer.wte.register_forward_hook(make_hook("embeddings"))
        hooks.append(hook)

        # 2. First layer in prelude
        hook = hf_model.transformer.prelude[0].register_forward_hook(make_hook("prelude_first"))
        hooks.append(hook)

        # 2a. QKV hook in first prelude layer
        hook = hf_model.transformer.prelude[0].attn.Wqkv.register_forward_hook(make_hook("prelude_qkv"))
        hooks.append(hook)
        # 2a. QKV hook in 2nd prelude layer
        hook = hf_model.transformer.prelude[1].attn.Wqkv.register_forward_hook(make_hook("prelude_qkv2"))
        hooks.append(hook)
        # hook adapter
        hook = hf_model.transformer.adapter.register_forward_hook(make_hook("adapter"))
        # 3. First layer in core adaptive block
        hook = hf_model.transformer.core_block[0].register_forward_hook(make_hook("core_first"))
        hooks.append(hook)
        hook = hf_model.transformer.core_block[0].attn.Wqkv.register_forward_hook(make_hook("core_qkv"))
        hooks.append(hook)

        # 4. First layer in coda
        hook = hf_model.transformer.coda[0].register_forward_hook(make_hook("coda_first"))
        hooks.append(hook)

        # Run HF model to capture states
        print("🔄 Running HF model to capture intermediate states...")
        hf_model(input_ids, num_steps=num_steps, init_scale=init_scale, cache_lookup_strategy="latest-m4-compress-s4")

        # Remove hooks
        for hook in hooks:
            hook.remove()

        vllm_states = {}

        print("Accessing existing vLLM model for hooks...")

        # Add hooks to existing vLLM model to capture intermediate states
        def make_vllm_hook(name):
            def hook(module, input, output):
                state = output[0].clone().detach() if isinstance(output, tuple) else output.clone().detach()
                state = state.unsqueeze(0)  # [1, seq_len, hidden_size]

                vllm_states[name] = state

            return hook

        # Get the vLLM model (multiprocessing disabled for testing)
        actual_model = vllm_model.llm_engine.model_executor.driver_worker.model_runner.model

        # Add hooks to corresponding stages
        vllm_hooks = []
        vllm_hooks.append(actual_model.model.embed_tokens.register_forward_hook(make_vllm_hook("embeddings")))

        # Hook first prelude layer
        vllm_hooks.append(actual_model.model.layers[0].register_forward_hook(make_vllm_hook("prelude_first")))

        # Hook QKV in first prelude layer
        vllm_hooks.append(
            actual_model.model.layers[0].self_attn.qkv_proj.register_forward_hook(make_vllm_hook("prelude_qkv"))
        )
        # Hook QKV in 2nd prelude layer
        vllm_hooks.append(
            actual_model.model.layers[1].self_attn.qkv_proj.register_forward_hook(make_vllm_hook("prelude_qkv2"))
        )
        # hook adapter
        vllm_hooks.append(actual_model.model.adapter.register_forward_hook(make_vllm_hook("adapter")))

        # Hook first core layer
        core_first_idx = actual_model.config.n_layers_in_prelude
        vllm_hooks.append(actual_model.model.layers[core_first_idx].register_forward_hook(make_vllm_hook("core_first")))

        # Hook QKV in core
        vllm_hooks.append(
            actual_model.model.layers[core_first_idx].self_attn.qkv_proj.register_forward_hook(
                make_vllm_hook("core_qkv")
            )
        )
        # Hook first coda layer
        coda_first_idx = actual_model.config.n_layers_in_prelude + actual_model.config.n_layers_in_recurrent_block
        vllm_hooks.append(actual_model.model.layers[coda_first_idx].register_forward_hook(make_vllm_hook("coda_first")))

        sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=1)

        token_prompt = TokensPrompt(prompt_token_ids=input_ids[0].tolist())
        print("🔄 Running vLLM model to capture intermediate states...")
        vllm_outputs = vllm_model.generate([token_prompt], sampling_params)

        print(f"✓ Captured {len(vllm_states)} vLLM intermediate states. Outputs were {vllm_outputs}")

        # Remove vLLM hooks
        for hook in vllm_hooks:
            hook.remove()

    # Step 3: Compare the captured states
    print("\n📊 Intermediate State Comparison Results:")
    print("=" * 60)

    stages = [
        "embeddings",
        "prelude_first",
        "prelude_qkv",
        "prelude_qkv2",
        "adapter",
        "core_first",
        "core_qkv",
        "coda_first",
    ]
    success = True

    print("\n📋 States captured:")
    print(f"   HF: {list(hf_states.keys())}")
    print(f"   vLLM: {list(vllm_states.keys())}")

    with torch.no_grad():
        for stage in stages:
            print(f"\n🔍 {stage} Stage:")
            print(f"   HF Shape: {hf_states[stage].shape}")
            print(f"   vllm Shape: {vllm_states[stage].shape}")

            hf_state = hf_states[stage].flatten()
            vllm_state = vllm_states[stage][0, : hf_states[stage].shape[1]].flatten()
            # Compute differences
            mse = torch.nn.functional.mse_loss(hf_state, vllm_state).item()
            cosine_sim = torch.nn.functional.cosine_similarity(hf_state, vllm_state, dim=0).item()

            # Relative error
            rel_error = (hf_state - vllm_state).norm() / hf_state.norm()

            print(f"   MSE: {mse:.8f}")
            print(f"   Cosine similarity: {cosine_sim:.6f}")
            print(f"   Relative error: {rel_error:.6f}")
            print(f"   HF norm: {hf_state.norm().item():.6f}")
            print(f"   vLLM norm: {vllm_state.norm().item():.6f}")

            # Determine if this stage is good
            if cosine_sim < 0.8 or rel_error > 0.2:
                print("❌ NO MATCH")
                success = False

            # Show first few values for debugging
            print(f"   First 5 HF values: {hf_state.flatten()[:5].tolist()}")
            print(f"   First 5 vLLM values: {vllm_state.flatten()[:5].tolist()}")

    return success


def compare_single_token_gen(hf_model, vllm_model, input_ids, num_steps=1, init_scale=0.0):
    print(f"Testing with same input: {input_ids.shape} -> {input_ids.tolist()}")

    # Single token forward pass comparison
    print("Comparing single token forward passes...")
    with torch.no_grad():
        # HF forward pass with detailed intermediate outputs
        hf_outputs = hf_model(
            input_ids, init_scale=init_scale, num_steps=num_steps, cache_lookup_strategy="latest-m4-compress-s4"
        )
        hf_logits = hf_outputs.logits
        print(f"HF logits shape: {hf_logits.shape}")

        # vLLM forward pass
        sampling_params = SamplingParams(
            temperature=0.0,  # deterministic
            max_tokens=1,
            logprobs=5,  # get top 5 logprobs only
        )
        vllm_outputs = vllm_model.generate([TokensPrompt(prompt_token_ids=input_ids[0].tolist())], sampling_params)
        vllm_logprobs = vllm_outputs[0].outputs[0].logprobs[0]  # first token logprobs

        # Compare last token logits from HF
        hf_last_logits = hf_logits[0, -1, :]
        hf_top5 = torch.topk(hf_last_logits, 5)
        hf_log_probs = torch.log_softmax(hf_last_logits, dim=-1)

        # Get vLLM top 5 tokens and logprobs
        vllm_tokens = list(vllm_logprobs.keys())
        vllm_logprob_values = [vllm_logprobs[token] for token in vllm_tokens]

        print(f"HF top 5 tokens: {hf_top5.indices.tolist()}")
        print(f"HF top 5 logprobs: {[hf_log_probs[idx].item() for idx in hf_top5.indices]}")
        print(f"vLLM top 5 tokens: {vllm_tokens}")
        print(f"vLLM top 5 logprobs: {vllm_logprob_values}")

        # Detailed single token validation
        tokens_match = len(vllm_tokens) > 0 and hf_top5.indices[0].item() == vllm_tokens[0]
        if tokens_match:
            print("✅ Top token prediction matches!")
            # Compare actual logprob values
            hf_top_logprob = hf_log_probs[hf_top5.indices[0]].item()
            vllm_top_logprob = vllm_logprob_values[0].logprob
            logprob_diff = abs(hf_top_logprob - vllm_top_logprob)
            print(f"   HF top logprob: {hf_top_logprob:.6f}")
            print(f"   vLLM top logprob: {vllm_top_logprob:.6f}")
            print(f"   Difference: {logprob_diff:.6f}")
            if logprob_diff < 0.01:
                print("✅ Logprob values match closely!")
            else:
                print("⚠️ Logprob values differ significantly")
        else:
            print("❌ Top token predictions differ")

        # Compare logprob values for common tokens
        common_tokens = set(hf_top5.indices.tolist()) & set(vllm_tokens)
        if common_tokens:
            print(f"✅ Common tokens in top 5: {list(common_tokens)}")
            for token in common_tokens:
                hf_logprob = hf_log_probs[token].item()
                vllm_idx = vllm_tokens.index(token)
                vllm_logprob = vllm_logprob_values[vllm_idx].logprob
                print(
                    f"   Token {token}: HF={hf_logprob:.4f}, vLLM={vllm_logprob:.4f}, diff={abs(hf_logprob - vllm_logprob):.4f}"
                )
        else:
            print("❌ No common tokens in top 5")


import os


def vllm_test():
    """Test vLLM integration against Huggingface reference implementation"""
    ModelRegistry.register_model("RavenForCausalLM", RavenForvLLM)
    print("🧪 Testing vLLM Raven model implementation...")
    model_name = "tomg-group-umd/huginn-0125"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Disable multiprocessing for simpler model access during testing
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    vllm_model = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        tensor_parallel_size=1,
        max_model_len=512,  # for testing
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        # compilation_config=CompilationConfig(level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=[1]),
    )
    print("vLLM model created successfully")

    # Test HuggingFace reference model
    print("Loading HF model")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    hf_model.eval()
    print("\n" + "=" * 60)
    print("\n🔍 Generation Comparison")
    print("-" * 50)

    # Test setup:
    num_steps = 32

    # Note: This may fail if the fundamental issue isn't fixed, but we want to see the behavior
    generation_prompt = "The capital of Westphalia is"

    # HF generation
    hf_inputs = tokenizer(generation_prompt, return_tensors="pt", add_special_tokens=True).input_ids.cuda()
    hf_generated = hf_model.generate(
        hf_inputs,
        max_new_tokens=10,
        do_sample=False,
        num_steps=num_steps,
        init_scale=0.0,
        cache_lookup_strategy="latest-m4-compress-s4",
    )
    hf_text = tokenizer.decode(hf_generated[0], skip_special_tokens=True)

    # vLLM generation
    vllm_sampling = SamplingParams(temperature=0.0, max_tokens=10)
    vllm_generated = vllm_model.generate([generation_prompt], vllm_sampling)
    vllm_text = generation_prompt + vllm_generated[0].outputs[0].text

    # compare result
    print(f"HF generated:   '{hf_text}'")
    print(f"vLLM generated: '{vllm_text}'")

    if hf_text.strip() == vllm_text.strip():
        print("✅ Generation outputs match exactly!")
    else:
        print("⚠️ Generation outputs differ")
        # Check if they start the same way
        common_prefix = ""
        for i, (h, v) in enumerate(zip(hf_text, vllm_text)):
            if h == v:
                common_prefix += h
            else:
                break
        print(f"Common prefix: '{common_prefix}' ({len(common_prefix)} chars)")

        # Check if at least the first few tokens are the same
        hf_tokens = tokenizer.encode(hf_text)
        vllm_tokens = tokenizer.encode(vllm_text)
        matching_tokens = 0
        for ht, vt in zip(hf_tokens, vllm_tokens):
            if ht == vt:
                matching_tokens += 1
            else:
                break
        print(f"Matching tokens from start: {matching_tokens}/{min(len(hf_tokens), len(vllm_tokens))}")

    print("\n" + "=" * 60)
    test_prompt = "Are you standing silent vigil now? Until the heavens fall?"
    input_ids = tokenizer(test_prompt, return_tensors="pt", add_special_tokens=True).input_ids.cuda()
    print(f"Test input: '{test_prompt}' -> {input_ids.shape} -> {input_ids.tolist()}")
    compare_single_token_gen(hf_model, vllm_model, input_ids, num_steps=num_steps, init_scale=0.0)
    print("\n" + "=" * 60)
    compare_intermediate_outputs(hf_model, vllm_model, input_ids, num_steps=num_steps, init_scale=0.0)


if __name__ == "__main__":
    vllm_test()
