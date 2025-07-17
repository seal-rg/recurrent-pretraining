"""Modeling file for HF compatibility and zero-shot experiments."""

import torch
import math

from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, BlockMask, flex_attention
from torch.nn.attention import bias as attn_bias
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from typing import Union, Optional, Any, Tuple, Callable, List
from functools import cache, cached_property

from .raven_config_minimal import RavenConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache

###################### Huggingface Glue code I ##################################################################
from transformers import PreTrainedModel, GenerationMixin
from transformers.utils import ModelOutput
from transformers.generation.utils import GenerateDecoderOnlyOutput

import torch.nn.functional as F
from transformers import GenerationConfig


class RavenPreTrainedModel(PreTrainedModel):
    config_class = RavenConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SandwichBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _tied_weights_keys = ["lm_head.weight"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = False
    _supports_static_cache = True
    _tp_plan = {}

    @cache
    def _init_func(self, dim, num_layers):
        return {
            "std": math.sqrt(2 / (5 * dim)),
            "out_proj": math.sqrt(2 / (5 * dim)) / math.sqrt(2 * num_layers),
            "embedding": math.sqrt(2 / (5 * dim)),
            "embed_scale": math.sqrt(dim),
        }

    @property
    def emb_scale(self):
        return self._init_func(self.config.n_embd, self.config.effective_expected_depth)["embed_scale"]

    def _normal_(self, tensor, std):
        return torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)

    @torch.no_grad()
    def init_qkv(self, qkv_tensor, init_fn, qk_std, v_std, dim, head_dim):
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
    def init_glu(self, glu_tensor, init_fn, w1_std, w2_std):
        g, h = glu_tensor.shape
        W1, W2 = (
            glu_tensor.new_empty([g // 2, h]),
            glu_tensor.new_empty([g // 2, h]),
        )
        init_fn(W1, w1_std)
        init_fn(W2, w2_std)
        glu_tensor.data.copy_(torch.cat([W1, W2], dim=0).contiguous())

    @cached_property
    def _full_name_of_module_lookup(self):
        return {id(m): n for n, m in self.named_modules()}

    @torch.no_grad()
    def _init_weights(self, module):
        _init_values = self._init_func(self.config.n_embd, self.config.effective_expected_depth)
        name = self._full_name_of_module_lookup[id(module)]
        if isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, torch.nn.Linear):
            if "Wqkv" in name:
                self.init_qkv(
                    module.weight,
                    self._normal_,
                    float(_init_values["std"]),
                    float(_init_values["std"]),
                    self.config.n_embd,
                    self.config.head_dim,
                )
            elif "fc" in name:
                self.init_glu(module.weight, self._normal_, float(_init_values["std"]), float(_init_values["out_proj"]))
            elif "mlp.proj" in name or "attn.proj" in name:
                self._normal_(module.weight, std=float(_init_values["out_proj"]))
            elif "adapter" in name or "lm_head" in name:
                self._normal_(module.weight, std=float(_init_values["std"]))
        elif isinstance(module, torch.nn.Embedding):
            self._normal_(module.weight, std=float(_init_values["embedding"]))


@dataclass
class CausalLMOutputRecurrentLatents(ModelOutput):
    loss: Optional[torch.Tensor] = None
    log_ppl: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    latent_states: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attention_maps: Optional[dict[int, torch.Tensor]] = None
    stats: Optional[dict] = None


###################### Minimal implementation from here ############################################################


class RMSNorm(torch.nn.Module):
    """Saner dtype handling and slightly better for fusion"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        with torch.autocast(enabled=False, device_type=x.device.type if x.device.type != "meta" else "cuda"):
            return self._norm(x.float()).type_as(x) * self.weight

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


class HuginnDynamicCache(DynamicCache):
    def __init__(self, lookup_strategy: str = "full") -> None:
        super().__init__()
        self._seen_tokens = 0
        self.key_cache: dict[int, dict[int, torch.Tensor]] = {}
        self.value_cache: dict[int, dict[int, torch.Tensor]] = {}
        # structure: cache[index_of_layer_or_recurrent_step][index_in_sequence]
        # the cache is held uncoalesced because certain recurrent steps may be missing for some sequence ids if using
        # per-token adaptive compute. In those cases, the "lookup_strategy" determines how to proceed
        # Also, It is critical that the head indices do not overlap with the recurrent iteration indices
        self.lookup_strategy = lookup_strategy

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        step_idx_tensor: torch.Tensor,
        lookup_strategy: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_idx: int = int(step_idx_tensor)  # todo: fix dicts with tensor step_idx, currently the memberships fail
        lookup_strategy = self.lookup_strategy if lookup_strategy is None else lookup_strategy
        if "compress-" in self.lookup_strategy and step_idx > 1:  # hardcode for current model!
            if "compress-s" in self.lookup_strategy:
                compression_stage = int(self.lookup_strategy.split("compress-")[1][1:])
                new_step_idx = (step_idx - 2) % compression_stage + 2
            elif "compress-anchor" in self.lookup_strategy:
                if step_idx - 2 < 4 * 8:  # anchor onto first 8 recurrence steps  # noqa: SIM108
                    new_step_idx = step_idx
                else:  # then re-use the next 4 KV states = one recurrence for all future recurrence
                    new_step_idx = 34 + (step_idx - 34) % 4
                # print(step_idx, new_step_idx)
            else:  # compress-r
                compression_stage = int(self.lookup_strategy.split("compress-")[1][1:])
                new_step_idx = (step_idx - 2) // compression_stage + 2
            step_idx = new_step_idx
        # Init
        if step_idx not in self.key_cache:
            self.key_cache[step_idx] = {}
            self.value_cache[step_idx] = {}
        # Update the number of seen tokens, we assume that step_idx=0 (first prelude) is always hit
        if step_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        # Add entries to cache
        for idx, entry in enumerate(key_states.unbind(dim=-2)):
            if "compress-" not in self.lookup_strategy:
                assert step_idx < 0 or self._seen_tokens - key_states.shape[-2] + idx not in self.key_cache[step_idx]
            self.key_cache[step_idx][self._seen_tokens - key_states.shape[-2] + idx] = entry
        for idx, entry in enumerate(value_states.unbind(dim=-2)):
            self.value_cache[step_idx][self._seen_tokens - value_states.shape[-2] + idx] = entry

        # Materialize past state based on lookup strategy:
        if len(self.key_cache[step_idx]) == self._seen_tokens or self.lookup_strategy == "full":
            # All entries are present, materialize cache as normal
            return (
                torch.stack(list(self.key_cache[step_idx].values()), dim=-2),
                torch.stack(list(self.value_cache[step_idx].values()), dim=-2),
            )
        else:  # some entries were not previously computed
            if lookup_strategy.startswith("latest-m4"):
                latest_keys = []
                latest_values = []
                for token_pos in range(self._seen_tokens):
                    # For steps >= 2, use modulo 4, this hard-codes the huginn block structure for now
                    if step_idx >= 2:
                        # Find valid steps for this token position
                        valid_steps = [s for s in range(step_idx + 1) if token_pos in self.key_cache[s]]
                        max_step = max([s for s in valid_steps if s >= 2 and s % 4 == step_idx % 4])
                    else:
                        max_step = step_idx if token_pos in self.key_cache[step_idx] else 0
                    latest_keys.append(self.key_cache[max_step][token_pos])
                    latest_values.append(self.value_cache[max_step][token_pos])
                return torch.stack(latest_keys, dim=-2), torch.stack(latest_values, dim=-2)
            elif lookup_strategy.startswith("available-m4"):
                latest_keys = []
                latest_values = []
                for token_pos in range(self._seen_tokens):
                    if token_pos in self.key_cache[step_idx]:
                        step = step_idx
                    else:
                        # Find valid steps for this token position
                        valid_steps = [s for s in range(step_idx + 1) if token_pos in self.key_cache[s]]
                        step = max([s for s in valid_steps if s >= 2 and s % 4 == step_idx % 4])
                    latest_keys.append(self.key_cache[step][token_pos])
                    latest_values.append(self.value_cache[step][token_pos])
                return torch.stack(latest_keys, dim=-2), torch.stack(latest_values, dim=-2)
            elif lookup_strategy.startswith("always-last-m4"):
                latest_keys = []
                latest_values = []
                for token_pos in range(self._seen_tokens):
                    # For steps >= 2, use modulo 4, this hard-codes the huginn block structure for now
                    if step_idx >= 2:
                        # Find valid steps for this token position
                        valid_steps = [key_step for key_step in self.key_cache if token_pos in self.key_cache[key_step]]
                        max_step = max([s for s in valid_steps if s >= 2 and s % 4 == step_idx % 4])
                    else:
                        max_step = step_idx if token_pos in self.key_cache[step_idx] else 0
                    latest_keys.append(self.key_cache[max_step][token_pos])
                    latest_values.append(self.value_cache[max_step][token_pos])
                return torch.stack(latest_keys, dim=-2), torch.stack(latest_values, dim=-2)
            elif lookup_strategy.startswith("skip"):
                existing_keys = []
                existing_values = []
                for token_pos in range(self._seen_tokens):
                    if token_pos in self.key_cache[step_idx]:
                        existing_keys.append(self.key_cache[step_idx][token_pos])
                        existing_values.append(self.value_cache[step_idx][token_pos])
                return torch.stack(existing_keys, dim=-2), torch.stack(existing_values, dim=-2)
            elif lookup_strategy.startswith("randomized"):  # sanity check
                rand_keys = []
                rand_values = []
                for token_pos in range(self._seen_tokens):
                    if step_idx < 2:  # For prelude steps
                        max_step = step_idx if token_pos in self.key_cache[step_idx] else 0
                    else:  # Get all steps from same block position
                        curr_modulo = (step_idx - 2) % 4 + 2
                        valid_steps = [
                            s
                            for s in range(2, step_idx + 1)
                            if (s - 2) % 4 + 2 == curr_modulo and token_pos in self.key_cache[s]
                        ]
                        max_step = valid_steps[torch.randint(len(valid_steps), (1,))]
                    rand_keys.append(self.key_cache[max_step][token_pos])
                    rand_values.append(self.value_cache[max_step][token_pos])
                return torch.stack(rand_keys, dim=-2), torch.stack(rand_values, dim=-2)
            else:
                raise ValueError(f"Unknown lookup strategy: {lookup_strategy}")

    def reset(self) -> None:
        """Reset the cache state."""
        self._seen_tokens = 0
        self.key_cache.clear()
        self.value_cache.clear()

    def clear_last_k_entries(self, k: int = 0):
        """Partially clear cache."""
        assert self._seen_tokens >= k
        self._seen_tokens = self._seen_tokens - k
        # self.key_cache[step_idx][self._seen_tokens - key_states.shape[-2] + idx] = entry
        self.key_cache = {
            step: {seq: seq_cache for seq, seq_cache in cache.items() if seq < self._seen_tokens}
            for step, cache in self.key_cache.items()
        }
        self.value_cache = {
            step: {seq: seq_cache for seq, seq_cache in cache.items() if seq < self._seen_tokens}
            for step, cache in self.value_cache.items()
        }

    def get_seq_length(self, step_idx: int = 0) -> int:
        return self._seen_tokens

    def get_memory_usage(self) -> float:
        total_bytes = 0
        # For each recurrent step/layer index
        for step_idx in self.key_cache:
            # Get the sequence cache for this step
            key_seq_cache = self.key_cache[step_idx]
            for seq_idx in key_seq_cache:
                key_tensor = key_seq_cache[seq_idx]
                # Add memory for of key tensors, assuming value is the same
                total_bytes += key_tensor.nelement() * key_tensor.element_size()
        return total_bytes * 2 / (1024 * 1024)


class HuginnStaticCache(Cache):
    """Static Cache for the recurrent model"""

    is_compileable = False  # this is todo

    def __init__(
        self,
        max_length: int,
        max_num_steps: int,
        num_heads: int,
        hidden_dim: int,
        batch_size: int = 1,
        lookup_strategy: str = "full",
        device: Optional[Union[torch.device, str]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self._seen_tokens = 0
        self.max_length = max_length
        self.lookup_strategy = lookup_strategy

        # Adjust max_num_steps based on compression strategy
        if "compress-" in lookup_strategy:
            compression_stage = int(lookup_strategy.split("compress-")[1][1:])
            if "compress-s" in lookup_strategy:
                # For modulo compression (s), we need steps for 0,1 + compressed steps
                self.max_num_steps = 4 + compression_stage
            else:
                # For relative compression, we need steps for 0,1 + compressed steps
                self.max_num_steps = 4 + (max_num_steps - 4 + compression_stage - 1) // compression_stage
        else:
            self.max_num_steps = max_num_steps

        # Pre-allocate cache tensors [steps, batch, heads, seq_len, head_dim]
        device = torch.device(device) if device is not None else None
        cache_shape = (self.max_num_steps, batch_size, num_heads, max_length, hidden_dim)

        self.key_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.valid_mask = torch.zeros((self.max_num_steps, max_length), dtype=torch.bool, device=device)
        # Mark tensors as static for compile
        torch._dynamo.mark_static_address(self.key_cache)
        torch._dynamo.mark_static_address(self.value_cache)
        torch._dynamo.mark_static_address(self.valid_mask)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        step_idx: torch.Tensor,
        lookup_strategy: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if step_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Adjust step_idx for compression
        lookup_strategy = self.lookup_strategy if lookup_strategy is None else lookup_strategy
        if "compress-" in lookup_strategy and step_idx > 1:
            compression_stage = int(lookup_strategy.split("compress-")[1][1:])
            if "compress-s" in lookup_strategy:
                step_idx = (step_idx - 2) % compression_stage + 2
            else:
                step_idx = (step_idx - 2) // compression_stage + 2

        start_idx = self._seen_tokens - key_states.shape[-2]

        indices = torch.arange(start_idx, start_idx + key_states.shape[-2], device=key_states.device)
        self.key_cache[step_idx].index_copy_(2, indices, key_states)
        self.value_cache[step_idx].index_copy_(2, indices, value_states)
        self.valid_mask[step_idx, start_idx : start_idx + key_states.shape[-2]] = True

        # Return based on lookup strategy
        if lookup_strategy == "full":
            return (
                self.key_cache[step_idx, :, :, : self._seen_tokens],
                self.value_cache[step_idx, :, :, : self._seen_tokens],
            )
        elif lookup_strategy.startswith("latest-m4"):
            if step_idx >= 2:
                pattern_steps = torch.arange(2, step_idx.item() + 1, 4, device=self.valid_mask.device)
                pattern_valid = self.valid_mask[pattern_steps]
                max_valid_step = pattern_steps[pattern_valid.to(torch.long).argmax(dim=0)]
                return (
                    self.key_cache[max_valid_step, torch.arange(self._seen_tokens)],
                    self.value_cache[max_valid_step, torch.arange(self._seen_tokens)],
                )
            return self.key_cache[step_idx, :, :, : self._seen_tokens], self.value_cache[
                step_idx, :, :, : self._seen_tokens
            ]
        elif lookup_strategy == "skip":
            valid_mask = self.valid_mask[step_idx, : self._seen_tokens]
            return (
                self.key_cache[step_idx, :, :, : self._seen_tokens][valid_mask],
                self.value_cache[step_idx, :, :, : self._seen_tokens][valid_mask],
            )
        elif lookup_strategy.startswith("randomized"):
            if step_idx < 2:
                max_step = step_idx
            else:
                curr_modulo = (step_idx - 2) % 4 + 2
                valid_steps = (
                    torch.where(
                        (torch.arange(2, step_idx.item() + 1, device=self.valid_mask.device) - 2) % 4 + 2 == curr_modulo
                    )[0]
                    + 2
                )
                rand_idx = torch.randint(len(valid_steps), (1,), device=valid_steps.device)
                max_step = valid_steps[rand_idx]
            return self.key_cache[max_step, : self._seen_tokens], self.value_cache[max_step, : self._seen_tokens]
        else:
            raise ValueError(f"Unknown lookup strategy: {lookup_strategy}")

    def reset(self) -> None:
        self._seen_tokens = 0
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.valid_mask.zero_()

    def get_seq_length(self, step_idx: int = 0) -> int:
        return self._seen_tokens

    def get_memory_usage(self) -> float:
        return (self.key_cache.nelement() + self.value_cache.nelement()) * self.key_cache.element_size() / (1024 * 1024)


ValidCache = HuginnDynamicCache | HuginnStaticCache


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config: RavenConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.n_embd // self.n_head

        shape = (self.n_head + 2 * self.n_kv_heads) * self.head_dim
        self.chunks = [config.n_embd, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim]
        self.Wqkv = torch.nn.Linear(config.n_embd, shape, bias=False)
        if config.qk_bias:
            self.qk_bias = torch.nn.Parameter(torch.zeros(2, 1, self.n_head, self.head_dim))
        self.proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        block_idx: torch.Tensor,
        mask: Optional[BlockMask] = None,
        past_key_values: Optional[ValidCache] = None,
    ) -> Tensor:
        B, S, E = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v = self.Wqkv(x).split(self.chunks, dim=2)
        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_kv_heads, self.head_dim)
        v = v.view(B, S, self.n_kv_heads, self.head_dim)
        # bias?
        if self.config.qk_bias:
            q_bias, k_bias = self.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
        # apply rotary
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2)  # (B, nh, S, hs)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, block_idx)

        if mask is not None:
            y: torch.Tensor = flex_attention(q, k, v, block_mask=mask)  # type: ignore
        else:
            if q.shape[2] < k.shape[2]:
                if q.shape[2] > 1:
                    bias = attn_bias.causal_lower_right(q.shape[2], k.shape[2])
                    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, bias, dropout_p=0.0)
                else:
                    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            else:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).reshape(B, S, E).contiguous()  # reshape is a view if possible (it mostly is)
        return self.proj(y)


class GatedMLP(torch.nn.Module):
    def __init__(self, config: RavenConfig, in_features: int = 0) -> None:
        super().__init__()
        in_features = config.n_embd if in_features == 0 else in_features
        self.fc = torch.nn.Linear(in_features, config.intermediate_size * 2, bias=False)

        self.proj = torch.nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.nonlin = torch.nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        # modified to single FC layer to improve parallelism
        x_fc_1, x_fc_2 = self.fc(x).chunk(2, dim=-1)
        x = self.nonlin(x_fc_1) * x_fc_2
        return self.proj(x)


class SandwichBlock(torch.nn.Module):
    expanded = False

    def __init__(self, config: RavenConfig, layer_id: int) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = GatedMLP(config)
        self.norm_3 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm_4 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.layer_id = layer_id

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        step_idx: int,
        mask: Optional[BlockMask] = None,
        past_key_values: Optional[ValidCache] = None,
    ) -> Tensor:
        attn_out = self.attn(self.norm_1(x), freqs_cis, step_idx, mask, past_key_values)
        x = self.norm_2(attn_out + x)
        x = self.norm_4(self.mlp(self.norm_3(x)) + x)
        return x


#################################### Main Model ##################################################################


class RavenForCausalLM(RavenPreTrainedModel, GenerationMixin):
    freqs_cis: torch.Tensor

    def __init__(
        self,
        config: RavenConfig,
    ) -> None:
        super().__init__(config)
        self.config = config

        # Transformer layers
        prelude = torch.nn.ModuleList(SandwichBlock(config, layer_id=i) for i in range(config.n_layers_in_prelude))
        adapter = torch.nn.Linear(config.n_embd * 2, config.n_embd, bias=config.bias)
        core_block = torch.nn.ModuleList(
            SandwichBlock(config, layer_id=i + config.n_layers_in_prelude)
            for i in range(config.n_layers_in_recurrent_block)
        )
        o = config.n_layers_in_prelude + config.n_layers_in_recurrent_block * config.mean_recurrence
        coda = torch.nn.ModuleList(SandwichBlock(config, layer_id=i + o) for i in range(config.n_layers_in_coda))

        self.transformer = torch.nn.ModuleDict(
            dict(
                wte=torch.nn.Embedding(config.padded_vocab_size, config.n_embd),
                prelude=prelude,
                adapter=adapter,
                core_block=core_block,
                coda=coda,
                ln_f=RMSNorm(config.n_embd, eps=config.norm_eps),  # used twice :>
            )
        )
        # Head
        self.lm_head = torch.nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        if self.config.tie_embeddings:
            self.tie_weights()
        # rope
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)
        self.gradient_checkpointing = False
        # Call weight init through HF post init:
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def get_output_embeddings(self):
        return self.lm_head

    def _precompute_freqs_cis(self):
        return precompute_freqs_cis(
            self.config.n_embd // self.config.num_attention_heads, self.config.block_size, self.config.rope_base, 1
        )

    def compile_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[ValidCache] = None,
        pad_token_id=65509,
    ) -> Optional[BlockMask]:
        batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]

        # If no padding and no attention mask, no need for a mask
        if attention_mask is None and (input_ids == pad_token_id).sum() == 0:
            return None

        if past_key_values is not None and seq_len == 1:
            return None

        # Get total sequence length including cache
        cache_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_length = cache_len + seq_len

        if attention_mask is None:

            def mask_mod(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx & (input_ids[b, kv_idx] != pad_token_id)
        else:

            def mask_mod(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & (input_ids[b, kv_idx] != pad_token_id) & attention_mask[b, q_idx, kv_idx]

        kv_length = past_key_values.get_seq_length() if past_key_values is not None else seq_len
        if kv_length == 0:
            kv_length = seq_len  # prefill
        block_mask = create_block_mask(
            mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=kv_length,
            device=str(input_ids.device),
        )

        return block_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
        input_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # binary  mask of shape q x kv, True=valid position
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_steps: Optional[torch.Tensor] = None,
        past_key_values: Optional[ValidCache] = None,
        output_details: dict = {
            "return_logits": True,
            "return_latents": True,
            "return_head": False,
            "return_stats": False,
        },
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        init_scale: float = 1.0,
        **kwargs,
    ) -> CausalLMOutputRecurrentLatents:
        # Support multiple position formats:
        if position_ids is None and cache_position is None:
            freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
        elif position_ids is not None:
            freqs_cis = self.freqs_cis.index_select(1, position_ids.squeeze())
        elif cache_position is not None:
            freqs_cis = self.freqs_cis[:, cache_position]

        if input_embeds is None:
            input_embeds = self.transformer.wte(input_ids)  # type: ignore # types broken in 2.6+

        if self.emb_scale != 1:
            input_embeds = input_embeds * self.emb_scale  # type: ignore

        if use_cache and past_key_values is None:
            past_key_values = HuginnDynamicCache()

        prepared_attn_mask = None  # self.compile_mask(input_ids, attention_mask, past_key_values)
        block_idx = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)  # count in tensors for compile
        # Non-recurrent prelude
        for block in self.transformer.prelude:  # type: ignore # types broken in 2.6+
            block_idx += 1
            input_embeds = block(input_embeds, freqs_cis, block_idx, prepared_attn_mask, past_key_values)

        # Main recurrence
        x, num_steps_no_grad, num_steps_with_grad, xk, block_idx = self.iterate_forward(
            input_embeds,  # type: ignore # mystery typing error
            input_states,
            freqs_cis,
            block_idx,
            prepared_attn_mask,
            past_key_values,
            num_steps,
            init_scale,
        )
        latent_states = x.clone().detach()

        # Coda layers
        block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)  # use negative indices for head
        for block in self.transformer.coda:  # type: ignore # types broken in 2.6+
            block_idx -= 1
            x = block(x, freqs_cis, block_idx, prepared_attn_mask, past_key_values)
        x = self.transformer.ln_f(x)  # type: ignore # types broken in 2.6+

        # Prediction head, assuming labels really are labels and not equal to input_ids
        if labels is not None:
            logits = self.lm_head(x).float()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100
            )
            log_ppl = loss.clone().detach().exp()
        else:
            logits = self.lm_head(x).float()
            loss, log_ppl = torch.as_tensor(0.0), torch.as_tensor(0.0)

        return CausalLMOutputRecurrentLatents(
            loss=loss,
            log_ppl=log_ppl,
            logits=logits if output_details["return_logits"] else None,
            past_key_values=past_key_values,
            hidden_states=x if output_details["return_head"] else None,
            latent_states=latent_states if output_details["return_latents"] else None,
            stats=self.get_stats(logits, x, latent_states, xk, input_embeds, num_steps_no_grad, num_steps_with_grad)
            if output_details["return_stats"]
            else None,
        )

    @torch._dynamo.disable(recursive=False)  # type: ignore
    def iterate_forward(
        self,
        input_embeds: torch.Tensor,
        input_states: torch.Tensor,
        freqs_cis,
        block_idx: torch.Tensor,
        mask: Optional[BlockMask],
        past_key_values: Optional[ValidCache] = None,
        num_steps: Optional[torch.Tensor] = None,
        init_scale: float = 1.0,
    ):
        x = xk = self.initialize_state(input_embeds, scale=init_scale) if input_states is None else input_states.clone()
        if num_steps is None:
            num_steps_no_grad, num_steps_with_grad = self.randomized_iteration_sampler()  # type: ignore
        elif hasattr(num_steps, "__len__") and len(num_steps) > 1:
            num_steps_no_grad, num_steps_with_grad = num_steps
        else:
            num_steps_no_grad, num_steps_with_grad = num_steps, torch.tensor(0) if not x.is_meta else 0

        with torch.no_grad():
            # ultra annoying in ddp due to
            # https://discuss.pytorch.org/t/does-distributeddataparallel-work-with-torch-no-grad-and-find-unused-parameters-false/122594
            # for now running with find_unused_params=True enabled even though the graph structure is (technically) clear
            # and all parameters are always used
            for no_grad_step in range(num_steps_no_grad):
                xk = x
                x, block_idx = self.core_block_forward(
                    xk, input_embeds, freqs_cis, mask, past_key_values, block_idx, no_grad_step
                )

        for grad_step in range(num_steps_with_grad):
            xk = x
            x, block_idx = self._maybe_checkpoint_core_block(
                xk, input_embeds, freqs_cis, mask, past_key_values, block_idx, num_steps_no_grad + grad_step
            )
        return self.transformer.ln_f(x), num_steps_no_grad, num_steps_with_grad, xk.detach(), block_idx  # type: ignore # types broken in 2.6+

    def core_block_forward(
        self,
        x,
        input_embeds,
        freqs_cis,
        mask: Optional[BlockMask],
        past_key_values,
        block_idx: torch.Tensor,
        current_step: int | Tensor,
    ):
        block_idx = block_idx.detach().clone()  # line only included to convince torch.checkpointing
        x = self._maybe_inject_noise(x, current_step)
        x = self.transformer.adapter(torch.cat([x, input_embeds.to(x.device)], dim=-1))  # type: ignore # types broken in 2.6+
        for block in self.transformer.core_block:  # type: ignore # types broken in 2.6+
            block_idx += 1
            x = block(x, freqs_cis, block_idx, mask, past_key_values)

        return x, block_idx

    @torch._dynamo.disable(recursive=False)  # type: ignore
    def randomized_iteration_sampler(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Outputs are long tensors so that they can be passed through compiled functions"""
        t = max(self.config.mean_recurrence - self.config.mean_backprop_depth, 0)
        s = self.config.mean_backprop_depth
        if torch.rand((1,)).is_meta:  # annoying clause to make meta-tensor-based flop counting work
            # these values are only the mean TFLOPs of the randomized sampler
            # Note that this clause also breaks the contract, and returns ints in meta tensor mode
            return t, s  # type: ignore
        if self.training:
            sigma = 0.5
            mu = math.log(t + s) - (sigma**2 / 2)
            rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma)
            p = torch.poisson(torch.tensor([rate], dtype=torch.float)) + 1
            n = torch.clamp(p - s, min=0)
            k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))
        else:
            n, k = torch.as_tensor(self.config.mean_recurrence), torch.as_tensor(0)

        return n.to(dtype=torch.long), k.to(dtype=torch.long)

    def initialize_state(self, input_embeds, scale: float = 1.0):
        x = torch.randn_like(input_embeds)
        std = self.config.init_values["std"] * scale
        if std > 0:
            torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
            if self.emb_scale != 1:
                x = x * self.emb_scale
        else:
            x.zero_()
        return x

    def _maybe_inject_noise(self, x, current_step, renorm=True):
        if self.config.test_time_noise > 0:
            n = self.config.test_time_noise * self.config.init_values["std"] * self.emb_scale
            if self.config.test_time_noise_type == "geom":
                step1 = torch.as_tensor(current_step + 1, device=x.device)  # need to cast for compile
                x = x * (1 - n / step1) + torch.randn_like(x) * n / step1
            elif self.config.test_time_noise_type == "sqrt":
                step1sqrt = torch.as_tensor(current_step + 1, device=x.device).sqrt()  # need to cast for compile
                x = x * (1 - n / step1sqrt) + torch.randn_like(x) * n / step1sqrt
            elif self.config.test_time_noise_type == "line":
                noise = max(n, (self.config.mean_recurrence - current_step) / self.config.mean_recurrence)  # type: ignore
                x = x * (1 - noise) + torch.randn_like(x) * noise
            elif self.config.test_time_noise_type == "chi":
                noise = 2 * torch.rand(1, device=x.device, dtype=x.dtype) * n
                x = x * (1 - noise) + torch.randn_like(x) * noise
            elif self.config.test_time_noise_type == "fixed":
                x = x * (1 - n) + torch.randn_like(x) * n
            else:
                raise ValueError()

            if renorm:
                x = self.transformer.core_block[-1].norm_4(x)  # type: ignore moduledict types still broken in pytorch
        return x

    """ ------------------ Alternative interfaces into the model forward ---------------------------------------- """

    @torch.no_grad()
    def iterate_one_step(
        self,
        input_embeds,
        input_states,
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        block_idx: torch.Tensor = torch.tensor(0, dtype=torch.long),
        attention_mask: Optional[BlockMask] = None,
        past_key_values: Optional[ValidCache] = None,
        current_step: int = 0,
    ):
        if position_ids is None and cache_position is None:
            freqs_cis = self.freqs_cis[:, : input_embeds.shape[1]]
        elif position_ids is not None:
            freqs_cis = self.freqs_cis.index_select(1, position_ids.squeeze())
        elif cache_position is not None:
            freqs_cis = self.freqs_cis[:, cache_position]
        x, block_idx = self.core_block_forward(
            input_states,
            input_embeds,
            freqs_cis,
            attention_mask,
            past_key_values,
            block_idx,
            current_step=current_step,
        )
        return x, block_idx, current_step + 1

    def predict_from_latents(
        self,
        latents,
        attention_mask: Optional[BlockMask] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_values: Optional[ValidCache] = None,
    ):
        if position_ids is None and cache_position is None:
            freqs_cis = self.freqs_cis[:, : latents.shape[1]]
        elif position_ids is not None:
            freqs_cis = self.freqs_cis.index_select(1, position_ids.squeeze())
        elif cache_position is not None:
            freqs_cis = self.freqs_cis[:, cache_position]
        x = self.transformer.ln_f(latents)  # type: ignore # types broken in 2.6+
        # Coda layers
        block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)  # use negative indices for head
        for block in self.transformer.coda:  # type: ignore # types broken in 2.6+
            block_idx -= 1
            x = block(x, freqs_cis, block_idx, attention_mask, past_key_values)
        x = self.transformer.ln_f(x)  # type: ignore # types broken in 2.6+

        logits = self.lm_head(x).float()

        return CausalLMOutputRecurrentLatents(
            loss=torch.as_tensor(0.0),
            log_ppl=torch.as_tensor(0.0),
            logits=logits,
            past_key_values=past_key_values,
            latent_states=x,
        )

    def embed_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[ValidCache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Support multiple position formats:
        if position_ids is None and cache_position is None:
            freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
        elif position_ids is not None:
            freqs_cis = self.freqs_cis.index_select(1, position_ids.squeeze())
        elif cache_position is not None:
            freqs_cis = self.freqs_cis[:, cache_position]

        input_embeds = self.transformer.wte(input_ids)  # type: ignore # types broken in 2.6+
        prepared_attn_mask = self.compile_mask(input_ids, attention_mask)

        if self.emb_scale != 1:
            input_embeds = input_embeds * self.emb_scale  # type: ignore

        if use_cache and past_key_values is None:
            past_key_values = HuginnDynamicCache()

        block_idx = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)  # count in tensors for compile
        # Non-recurrent prelude
        for block in self.transformer.prelude:  # type: ignore # types broken in 2.6+
            block_idx += 1
            input_embeds = block(input_embeds, freqs_cis, block_idx, prepared_attn_mask, past_key_values)
        return input_embeds, block_idx

    @torch.no_grad()
    def _prefill_with_varied_exit_steps(
        self,
        input_ids: torch.Tensor,
        exit_evaluator: "PerIterationExitEvaluator",
        past_key_values: Optional[ValidCache] = None,
        init_scale: float = 1.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, ValidCache, List[int]]:
        """ "
        Note that this the opposite of a real prefill, it goes token-by token and can adaptively exit on each.
        Use for scientific experiments.
        """
        # currently the cache doesn't support batching with adaptive compute
        assert input_ids.shape[0] == 1

        if past_key_values is None:
            past_key_values = HuginnDynamicCache()
        attention_mask = None
        output = torch.empty(
            (input_ids.shape[0], 0, self.config.vocab_size), device=input_ids.device, dtype=torch.float
        )
        compute_steps = []
        for pos in range(input_ids.shape[1]):
            aux_inputs = {
                "cache_position": pos,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }
            freqs_cis = self.freqs_cis[:, pos]
            embedded_inputs, block_idx = self.embed_inputs(input_ids[:, pos].unsqueeze(1), **aux_inputs)

            current_latents = self.initialize_state(embedded_inputs, scale=init_scale)
            exit_evaluator.init(current_latents)

            # Main recurrence
            for compute_step in range(self.config.mean_recurrence):
                current_latents, block_idx, _ = self.iterate_one_step(
                    embedded_inputs,
                    current_latents,
                    block_idx=block_idx,
                    **aux_inputs,
                    current_step=compute_step,
                )
                new_exits, _, _ = exit_evaluator.check(self, current_latents, aux_inputs)
                if new_exits.any():
                    break
            compute_steps.append(compute_step + 1)

            x = self.transformer.ln_f(current_latents)  # type: ignore

            # Coda layers
            block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)  # use negative indices for head
            for block in self.transformer.coda:  # type: ignore # types broken in 2.6+
                block_idx -= 1
                x = block(x, freqs_cis, block_idx, attention_mask, past_key_values)

            x = self.transformer.ln_f(x)  # type: ignore
            logits = self.lm_head(x).float()
            output = torch.cat([output, logits], dim=1)
        return output, past_key_values, compute_steps  # type: ignore

    @torch.no_grad()
    def forward_with_adaptive_compute(
        self,
        input_ids: torch.Tensor,
        exit_evaluator: "PerIterationExitEvaluator",
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[ValidCache] = None,
        output_details: dict = {
            "return_logits": True,
            "return_latents": True,
            "return_head": False,
            "return_stats": False,
        },
        init_scale: float = 1.0,
        **kwargs,
    ) -> CausalLMOutputRecurrentLatents:
        """This forward call does not make use of the causal nature of transformers, it runs token-by token!
        Do not use this function for anything other than scientific experiments with adaptive compute!
        """
        logits, past_key_values, compute_steps = self._prefill_with_varied_exit_steps(
            input_ids, exit_evaluator, past_key_values, init_scale
        )
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
            log_ppl = loss.clone().detach()
        else:
            loss, log_ppl = torch.as_tensor(0.0), torch.as_tensor(0.0)

        return CausalLMOutputRecurrentLatents(
            loss=loss,
            log_ppl=log_ppl,
            logits=logits if output_details["return_logits"] else None,
            past_key_values=None,
            hidden_states=None,
            latent_states=None,
            attention_maps=None,
            stats={"compute_steps": compute_steps},
        )

    def get_stats(self, logits, x, latent_states, xk, input_embeds, num_steps_no_grad, num_steps_with_grad):
        probs = torch.softmax(logits.float(), dim=-1)
        prob_entropy = torch.where(probs > 0, -probs * probs.log(), 0).sum(dim=-1)
        residual_diff = (x - latent_states).norm(dim=-1)
        rel_residual = residual_diff / latent_states.norm(dim=-1)
        stats = {
            "entropy": prob_entropy,
            "residual_diff": residual_diff,
            "rel_residual": rel_residual,
            "num_steps_no_grad": num_steps_no_grad,
            "num_steps_with_grad": num_steps_with_grad,
        }
        return stats

    def _maybe_checkpoint_core_block(self, *args, **kwargs) -> tuple[Tensor, Tensor]:
        if self.gradient_checkpointing:
            return checkpoint(
                self.core_block_forward,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
                determinism_check="none",
                **kwargs,
            )  # type: ignore
        else:
            return self.core_block_forward(*args)

    """"------------------------------------------Generation Utilities from here----------------------------------"""

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        cache_lookup_strategy: str = "full",
        **kwargs,
    ):
        model_inputs = {}
        model_inputs["cache_position"] = cache_position
        current_input_length = input_ids.shape[1]

        if past_key_values is not None:
            if not isinstance(past_key_values, (HuginnDynamicCache, HuginnStaticCache)):
                assert past_key_values.get_seq_length() == 0  # only replace empty caches
                # Need to use custom cache, detect and replace HF cache if generate injects it
                if isinstance(past_key_values, StaticCache):
                    past_key_values = HuginnStaticCache(
                        max_length=getattr(self.generation_config, "max_length", self.config.block_size),
                        max_num_steps=4 + kwargs.get("num_steps", self.config.mean_recurrence) * 4,
                        num_heads=self.config.num_key_value_heads,
                        hidden_dim=self.config.n_embd // self.config.num_attention_heads,
                        dtype=torch.bfloat16,
                        device=input_ids.device,
                        lookup_strategy=cache_lookup_strategy,
                    )
                else:
                    past_key_values = HuginnDynamicCache(lookup_strategy=cache_lookup_strategy)
            model_inputs["past_key_values"] = past_key_values if kwargs["use_cache"] else None
            input_ids = input_ids[:, cache_position]  # type: ignore

        model_inputs["input_ids"] = input_ids.clone(memory_format=torch.contiguous_format)
        if cache_position is None:
            position_ids = torch.arange(current_input_length)[None, :].to(input_ids.device)
            model_inputs["position_ids"] = position_ids[:, -current_input_length:].clone(
                memory_format=torch.contiguous_format
            )  # some form of position_ids is a critical argument for the model to correctly apply rope!

        # forward all other entries
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value
        return model_inputs

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """Dispatcher - use HF generate in all normal cases."""
        self.generation_config = args[1] if len(args) > 1 else self.generation_config
        if any(k in kwargs for k in ("criterion", "exit_threshold", "exit_evaluator")):
            return self.generate_with_adaptive_compute(*args, **kwargs)
        elif any(k in kwargs for k in ("draft_steps", "lookahead_for_draft", "verification_threshold")):
            return self.generate_speculative(*args, **kwargs)
        elif "continuous_compute" in kwargs:
            return self.generate_minimal(*args, **kwargs)
        else:
            return super().generate(*args, **kwargs)

    @torch.no_grad()
    def _prep_generate_args(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,  # type: ignore
        cache_lookup_strategy: str = "full",
        model_kwargs: dict = {},
    ):
        # Setup
        if generation_config is None:
            generation_config: GenerationConfig = self.generation_config  # type: ignore
        if "max_new_tokens" in model_kwargs:
            max_new_tokens = model_kwargs["max_new_tokens"]
            if "max_length" in model_kwargs:
                max_new_tokens = min(max_new_tokens, model_kwargs["max_length"] - input_ids.shape[1])
        else:
            max_length = model_kwargs.get("max_length", generation_config.max_length)
            max_new_tokens = max_length - input_ids.shape[1]

        if "cache_implementation" not in model_kwargs or model_kwargs["cache_implementation"] == "dynamic":
            model_kwargs["past_key_values"] = HuginnDynamicCache(lookup_strategy=cache_lookup_strategy)
        else:
            model_kwargs["past_key_values"] = HuginnStaticCache(
                max_length=max_length,
                max_num_steps=4 + model_kwargs.get("num_steps", self.config.mean_recurrence) * 4,
                num_heads=self.config.num_key_value_heads,
                hidden_dim=self.config.n_embd // self.config.num_attention_heads,
                batch_size=input_ids.shape[0],
                dtype=torch.bfloat16,
                device=input_ids.device,
                lookup_strategy=cache_lookup_strategy,
            )
        model_kwargs["use_cache"] = True
        model_kwargs = self._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)
        return model_kwargs, generation_config, max_new_tokens

    @torch.no_grad()
    def generate_minimal(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,  # type: ignore
        tokenizer=None,
        streamer=None,
        continuous_compute=False,  # warm-start state / continuous CoT
        init_scale: float = 1.0,
        cache_lookup_strategy: str = "full",
        **model_kwargs,
    ) -> Union[torch.Tensor, dict[str, Any]]:
        """Minimal single-sequence generation. Template for more complicated generate tasks"""
        model_kwargs, generation_config, max_new_tokens = self._prep_generate_args(
            input_ids, generation_config, cache_lookup_strategy, model_kwargs
        )
        stop_tokens = self._get_stops(generation_config, tokenizer, model_kwargs).to(input_ids.device)
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # Set up continuous compute if enabled
        if continuous_compute:
            embedded_inputs, _ = self.embed_inputs(input_ids)
            model_kwargs["input_states"] = self.initialize_state(embedded_inputs, scale=init_scale)

        # Generate tokens
        batch_size = input_ids.shape[0]
        for _ in range(max_new_tokens):
            # Forward pass
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs, init_scale=init_scale)

            # Get next token
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token = self._sample_next_token(next_token_logits, generation_config)

            # Append token to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if streamer:
                streamer.put(next_token.cpu())

            # Update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            if continuous_compute:
                model_kwargs["input_states"] = outputs.latent_states[:, -1:, :]

            if stop_tokens is not None:
                for i in range(batch_size):
                    if unfinished_sequences[i] and next_token[i, 0].item() in stop_tokens:
                        unfinished_sequences[i] = 0
            if "stopping_criteria" in model_kwargs:
                unfinished_sequences = unfinished_sequences & ~model_kwargs["stopping_criteria"](input_ids, None)
            if unfinished_sequences.max() == 0:
                break

        if streamer:
            streamer.end()

        if generation_config.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,  # type: ignore
                scores=None,
                logits=None,
                attentions=None,
                hidden_states=None,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return input_ids

    @torch.no_grad()
    def generate_with_adaptive_compute(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,  # type: ignore
        tokenizer=None,
        streamer=None,
        continuous_compute=False,  # warm-start state / continuous CoT
        criterion="none",  # adaptive compute is off by default, turn on by choosing an exit criterion
        exit_threshold: Union[str, float, int] = "auto",
        init_scale: float = 1.0,
        cache_lookup_strategy: str = "full",
        do_not_exit_in_prefill: bool = False,
        min_steps: int = 0,
        check_criterion_every_n_steps=1,
        exit_evaluator: "Optional[PerIterationExitEvaluator]" = None,  # optional plugin of a new exit eval object
        **model_kwargs,
    ) -> Union[torch.Tensor, GenerateDecoderOnlyOutput]:
        """
        Generate tokens with adaptive compute. This is NOT the most efficient implementation.
        For batches, on each token, we iterate until the entire batch finishes.
        Note: While the method can be used batched, and will produce sensible results, this cannot be used to evaluate
              the success of adaptive compute methods, which should only ever be benchmarked with batch_size=1.
              This is because the KV-cache entries are necessarily batched and so contain entries equal to the sequence
              with the largest number of steps in the whole batch, and these KV states, which would not have been computed
              if there was only one (short compute) sequence in the batch, will be picked up by later compute steps,
              making early exits look better than they are.
        """
        model_kwargs, generation_config, max_new_tokens = self._prep_generate_args(
            input_ids, generation_config, cache_lookup_strategy, model_kwargs
        )
        max_steps = model_kwargs.get("num_steps", self.config.mean_recurrence)
        stop_tokens = self._get_stops(generation_config, tokenizer, model_kwargs).to(input_ids.device)
        logit_type = dict(copy=True, dtype=torch.float32, device=input_ids.device)
        batch_size = input_ids.shape[0]
        compute_steps = []

        # Set up continuous compute if enabled
        if continuous_compute:
            embedded_inputs, _ = self.embed_inputs(input_ids)
            model_kwargs["input_states"] = self.initialize_state(embedded_inputs, scale=init_scale)

        # Track which sequences have finished (using unfinished_sequences to match generate_minimal)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        if exit_evaluator is None:
            exit_evaluator = get_adaptive_exit_evaluator(self, criterion, exit_threshold)

        # Generate tokens
        for token_step_in_sequence in range(max_new_tokens):
            # Adaptive compute forward
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            aux_inputs = {
                k: model_inputs[k] for k in ["cache_position", "past_key_values", "attention_mask"] if k in model_inputs
            }
            embedded_inputs, block_idx = self.embed_inputs(model_inputs["input_ids"], **aux_inputs)
            current_latents = (
                self.initialize_state(embedded_inputs, scale=init_scale)
                if not continuous_compute
                else model_kwargs["input_states"]
            )

            # Initialize next_states for continuous compute
            if continuous_compute:
                next_states = current_latents[:, -1:, :].clone()

            # Initialize criterion tracking for each sequence in batch
            exit_values_per_seq = [[] for _ in range(batch_size)]
            compute_steps_per_seq = [0] * batch_size
            exit_reached = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

            outputs, next_token_logits = None, None
            exit_evaluator.init(current_latents)

            # Iterate through compute steps
            for compute_step in range(max_steps):
                current_latents, block_idx, _ = self.iterate_one_step(
                    embedded_inputs,
                    current_latents,
                    block_idx=block_idx,
                    **aux_inputs,
                    current_step=compute_step,
                )

                # Skip checking exit conditions if min_steps not met, or not checking this step, or in prefill
                if (
                    compute_step < min_steps
                    or (compute_step - min_steps) % check_criterion_every_n_steps != 0
                    or (do_not_exit_in_prefill and token_step_in_sequence == 0)
                ):
                    continue

                # Otherwise check for new exits, potentially by evaluating the coda:
                new_exits, outputs, exit_values = exit_evaluator.check(self, current_latents, aux_inputs)

                # Record values and check exits for each sequence
                for i in range(batch_size):
                    if not exit_reached[i] and unfinished_sequences[i].bool():
                        exit_values_per_seq[i].append(exit_values[i].item())

                new_exits = new_exits & ~exit_reached & unfinished_sequences.bool()

                if new_exits.any():
                    exit_reached = exit_reached | new_exits
                    if outputs is not None:
                        logits = outputs.logits
                    else:
                        # For latent-based criteria, compute outputs when we need them
                        outputs = self.predict_from_latents(current_latents, **aux_inputs)
                        logits = outputs.logits

                    if next_token_logits is None:
                        next_token_logits = logits[:, -1, :].to(**logit_type)  # type: ignore
                    else:
                        next_token_logits[new_exits] = logits[new_exits, -1, :].to(**logit_type)  # type: ignore

                    for i in range(batch_size):
                        if new_exits[i]:
                            compute_steps_per_seq[i] = compute_step + 1

                    # Update continuous compute states for newly exited sequences
                    if continuous_compute:
                        next_states[new_exits] = current_latents[new_exits, -1:, :]

                # If all sequences have exited or finished, break early
                if (exit_reached | ~unfinished_sequences.bool()).all():
                    break

            # This else triggers if the for loop finishes without breaking:
            else:
                if outputs is None:
                    outputs = self.predict_from_latents(current_latents, **aux_inputs)

                # For sequences that didn't exit early, use the final logits
                if next_token_logits is None:
                    next_token_logits = outputs.logits[:, -1, :].to(**logit_type)  # type: ignore
                    for i in range(batch_size):
                        compute_steps_per_seq[i] = max_steps
                else:
                    for i in range(batch_size):
                        if not exit_reached[i] and unfinished_sequences[i].bool():
                            next_token_logits[i] = outputs.logits[i, -1, :].to(**logit_type)  # type: ignore
                            compute_steps_per_seq[i] = max_steps
            # Save latent states for continuous compute if enabled
            if continuous_compute:
                still_running = ~exit_reached & unfinished_sequences.bool()
                next_states[still_running] = current_latents[still_running, -1:, :]
                model_kwargs["input_states"] = next_states

            # Record compute steps for this token generation
            compute_steps.append([compute_steps_per_seq, exit_values_per_seq])

            # Sample or select next token based on generation config
            next_token = self._sample_next_token(next_token_logits, generation_config)

            # Append token to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if streamer:
                streamer.put(next_token.cpu())

            # Update model kwargs for next iteration
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)  # type: ignore

            # Check for stop tokens and update unfinished sequences
            for i in range(batch_size):
                if (
                    unfinished_sequences[i].bool()
                    and stop_tokens is not None
                    and next_token[i, 0].item() in stop_tokens
                ):
                    unfinished_sequences[i] = 0

            # Apply any custom stopping criteria
            if "stopping_criteria" in model_kwargs:
                unfinished_sequences = unfinished_sequences & ~model_kwargs["stopping_criteria"](input_ids, None)

            # Break if all sequences are finished
            if unfinished_sequences.max() == 0:
                break

        if streamer:
            streamer.end()

        if generation_config.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,  # type: ignore
                scores=compute_steps,  # type: ignore
                logits=None,
                attentions=None,
                hidden_states=None,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return input_ids

    @torch.no_grad()
    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,  # type: ignore
        tokenizer=None,
        streamer=None,
        continuous_compute=False,  # warm-start state / continuous CoT
        init_scale: float = 1.0,
        cache_lookup_strategy: str = "full",
        draft_steps=32,
        lookahead_for_draft=8,
        verification_threshold=1,
        num_steps: int = 32,  # intercept deliberately
        **model_kwargs,
    ) -> Union[torch.Tensor, dict[str, Any]]:
        """Batched speculative decoding with per-sequence acceptance."""
        assert lookahead_for_draft > 0
        pad_id = 65509
        model_kwargs, generation_config, max_new_tokens = self._prep_generate_args(
            input_ids, generation_config, cache_lookup_strategy, model_kwargs
        )
        stop_tokens = self._get_stops(generation_config, tokenizer, model_kwargs).to(input_ids.device)
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # Set up continuous compute if enabled
        if continuous_compute:
            embedded_inputs, _ = self.embed_inputs(input_ids)
            model_kwargs["input_states"] = self.initialize_state(embedded_inputs, scale=init_scale)

        tokens_generated = 0
        # Prefill cache with full num_steps
        if model_kwargs["past_key_values"].get_seq_length() == 0:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs, num_steps=num_steps, init_scale=init_scale)
            next_token = self._sample_next_token(
                outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32), generation_config
            )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            tokens_generated += 1
            if streamer:
                streamer.put(next_token.cpu())
            model_kwargs["cache_position"] = torch.as_tensor(
                [model_inputs["past_key_values"].get_seq_length()], device=input_ids.device
            )
            if continuous_compute:
                model_kwargs["input_states"] = outputs.latent_states[:, -1:, :]

        # Generate tokens
        batch_size, prefix_seq_len = input_ids.shape[0], input_ids.shape[1]
        accepted_tokens = []

        while tokens_generated < max_new_tokens:
            ### Run the next draft ####
            drafted_inputs = input_ids.clone()
            current_len = input_ids.shape[1]

            for _ in range(lookahead_for_draft):
                model_inputs = self.prepare_inputs_for_generation(drafted_inputs, **model_kwargs)
                outputs = self(**model_inputs, num_steps=draft_steps, init_scale=init_scale)
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32)
                next_token = self._sample_next_token(next_token_logits, generation_config)
                drafted_inputs = torch.cat([drafted_inputs, next_token], dim=-1)
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
                if continuous_compute:
                    model_kwargs["input_states"] = outputs.latent_states[:, -1:, :]

            model_kwargs["past_key_values"].clear_last_k_entries(lookahead_for_draft)

            ## Verify drafted tokens ###
            model_kwargs["cache_position"] = torch.arange(
                current_len - 1, current_len + lookahead_for_draft - 1, device=input_ids.device
            )
            model_inputs = self.prepare_inputs_for_generation(drafted_inputs, **model_kwargs)
            outputs = self(**model_inputs, num_steps=num_steps, init_scale=init_scale)
            verified_next_token_preds = outputs.logits.argmax(dim=-1)

            if verification_threshold >= 1:
                mismatched_tokens = (
                    verified_next_token_preds[:, -lookahead_for_draft:] != drafted_inputs[:, current_len:]
                )
                not_all_matched, first_mismatch = torch.max(mismatched_tokens, dim=1)
            else:
                verified_logits = outputs.logits[:, -lookahead_for_draft:, :]
                verified_probs = F.softmax(verified_logits, dim=-1)
                drafted_token_probs = torch.gather(
                    verified_probs, -1, drafted_inputs[:, current_len:].unsqueeze(-1)
                ).squeeze(-1)
                max_probs = verified_probs.max(dim=-1)[0]
                verification_passed = drafted_token_probs >= verification_threshold * max_probs
                not_all_matched, first_mismatch = torch.max(~verification_passed, dim=1)

            # Per-sequence acceptance handling
            acceptance_lengths = torch.where(not_all_matched, first_mismatch, lookahead_for_draft)

            # Build next_tokens for each sequence
            next_tokens_batch = []
            for i in range(batch_size):
                seq_acceptance = acceptance_lengths[i].item()
                if not_all_matched[i] and seq_acceptance < lookahead_for_draft:
                    # Accept up to mismatch + sample final token
                    accepted_part = drafted_inputs[i : i + 1, current_len : current_len + seq_acceptance]
                    final_token_logits = outputs.logits[i : i + 1, seq_acceptance, :].to(copy=True, dtype=torch.float32)
                    final_token = self._sample_next_token(final_token_logits, generation_config)
                    seq_tokens = torch.cat([accepted_part, final_token], dim=-1) if seq_acceptance > 0 else final_token
                else:
                    # Accept all drafted tokens
                    seq_tokens = drafted_inputs[i : i + 1, current_len : current_len + seq_acceptance]
                next_tokens_batch.append(seq_tokens)

            # Clean up KV cache - only if any sequence had mismatches
            if not_all_matched.any():
                min_first_mismatch = first_mismatch.min().item()
                model_inputs["past_key_values"].clear_last_k_entries(lookahead_for_draft - min_first_mismatch - 1)

            # Concatenate accepted tokens to input_ids
            batch_accepted_counts = [tokens.shape[1] for tokens in next_tokens_batch]
            max_len = max(batch_accepted_counts)
            padded_tokens = [
                torch.cat(
                    [
                        tokens,
                        pad_id * torch.ones((1, max_len - tokens.shape[1]), dtype=tokens.dtype, device=tokens.device),
                    ],
                    dim=-1,
                )
                if tokens.shape[1] < max_len
                else tokens
                for tokens in next_tokens_batch
            ]
            next_tokens = torch.cat(padded_tokens, dim=0)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            accepted_tokens.append(batch_accepted_counts)
            tokens_generated += max(batch_accepted_counts)

            if streamer:
                streamer.put(next_tokens_batch[0].cpu())

            model_kwargs["cache_position"] = torch.as_tensor(
                [model_inputs["past_key_values"].get_seq_length()], device=input_ids.device
            )
            if continuous_compute:
                model_kwargs["input_states"] = outputs.latent_states[:, -1:, :]

            # Check stopping conditions
            if stop_tokens is not None:
                for i in range(batch_size):
                    if unfinished_sequences[i] and torch.isin(next_tokens_batch[i], stop_tokens).any():
                        unfinished_sequences[i] = 0
            if "stopping_criteria" in model_kwargs:
                unfinished_sequences = unfinished_sequences & ~model_kwargs["stopping_criteria"](input_ids, None)
            if unfinished_sequences.max() == 0:
                break

        if streamer:
            streamer.end()

        # Cut off extraneous parts of the sequence per batch element
        if stop_tokens is not None:
            for i in range(batch_size):
                stop_positions = torch.isin(input_ids[i, prefix_seq_len:], stop_tokens).nonzero()
                if len(stop_positions) > 0:
                    input_ids[i, prefix_seq_len + stop_positions[0].item() + 1 :] = pad_id
        # Trim tensor to remove columns that are pad_id across all sequences
        non_pad_mask = input_ids != pad_id
        last_real_token = non_pad_mask.any(dim=0).nonzero()
        if len(last_real_token) > 0:
            input_ids = input_ids[:, : last_real_token[-1].item() + 1]

        if generation_config.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,  # type: ignore
                scores=accepted_tokens,  # type: ignore
                logits=None,
                attentions=None,
                hidden_states=None,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return input_ids

    def _get_stops(self, generation_config, tokenizer, model_kwargs):
        stop_tokens = {65504, 65505, 65508}  # begin_text, end_text, end_turn
        if generation_config.eos_token_id is not None:
            try:
                stop_tokens.update(generation_config.eos_token_id)
            except TypeError:
                stop_tokens.add(generation_config.eos_token_id)
        if "stopping_criteria" in model_kwargs and tokenizer is None:
            tokenizer = model_kwargs["stopping_criteria"][0].tokenizer
        if hasattr(generation_config, "stop_strings") and tokenizer and generation_config.stop_strings:
            for s in generation_config.stop_strings:
                token_id = tokenizer(s, add_special_tokens=False)["input_ids"][0]
                stop_tokens.add(token_id)
        return torch.tensor(list(stop_tokens))

    def _sample_next_token(self, next_token_logits, generation_config):
        """Helper function to sample the next token."""
        if generation_config.do_sample:
            if generation_config.temperature:
                next_token_logits = next_token_logits.float() / generation_config.temperature

            probs = F.softmax(next_token_logits, dim=-1)

            # Apply top_k
            if generation_config.top_k:
                top_k_values, _ = torch.topk(probs, generation_config.top_k, dim=-1)
                min_values = top_k_values[:, -1].unsqueeze(-1).expand_as(probs)
                probs = torch.where(probs < min_values, torch.zeros_like(probs), probs)

            # Apply top_p (nucleus sampling)
            if generation_config.top_p:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Create mask for probs to keep
                remove_indices = cumulative_probs > generation_config.top_p
                remove_indices[:, 0] = False  # Keep at least the top probability

                # Convert sorted indices mask back to original indices mask
                mask = torch.zeros_like(probs, dtype=torch.bool)
                for i in range(probs.shape[0]):
                    mask[i, sorted_indices[i, remove_indices[i]]] = True

                probs = torch.where(mask, torch.zeros_like(probs), probs)

            # Apply min_p
            if generation_config.min_p:
                max_probs = probs.max(dim=-1, keepdim=True)[0]
                min_p_threshold = generation_config.min_p * max_probs
                probs = torch.where(probs < min_p_threshold, torch.zeros_like(probs), probs)

            # Renormalize probabilities
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

            # Sample from the distribution
            return torch.multinomial(probs, num_samples=1)
        else:
            return torch.argmax(next_token_logits, dim=-1, keepdim=True)


################################ Model Utils #######################################################################


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, condense_ratio: int = 1):
    with torch.autocast("cuda", enabled=False):
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(end, dtype=torch.float32, device=inv_freqs.device) / condense_ratio
        freqs = torch.outer(t, inv_freqs).float()
        return torch.stack([torch.cos(freqs)[None, :, None, :], torch.sin(freqs)[None, :, None, :]], dim=4)
        # equivalent to
        # freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        # cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)


def apply_rotary_emb_complex_like(q: Tensor, k: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    with torch.autocast("cuda", enabled=False):
        qk_r2 = torch.cat([q, k], dim=2).unflatten(dim=-1, sizes=(-1, 2)).float()  # cast to float32 for smooth skin
        rotated_qk_r2 = torch.stack(
            [
                qk_r2[..., 0] * freqs_cis[..., 0] - qk_r2[..., 1] * freqs_cis[..., 1],
                qk_r2[..., 1] * freqs_cis[..., 0] + qk_r2[..., 0] * freqs_cis[..., 1],
            ],
            -1,
        ).flatten(3)
        rotated_qk = rotated_qk_r2
        return torch.split(rotated_qk.type_as(q), q.shape[2], dim=2)  # type: ignore


#################################### Adaptive Compute Exit Evaluators ##########################################

Exit = Tuple[torch.Tensor, Optional[CausalLMOutputRecurrentLatents], torch.Tensor]


class PerIterationExitEvaluator:
    """Base class for exit evaluators that check after each recurrent step."""

    def init(self, initial_latents: torch.Tensor):
        """Initialize evaluator state."""

    def check(self, model: "RavenForCausalLM", latents: torch.Tensor, aux_inputs: dict) -> Exit:
        """Returns (should_exit, outputs (or None), exit_values)"""
        raise NotImplementedError()


class NoOpExitEvaluator(PerIterationExitEvaluator):
    """Exit evaluator that never exits early."""

    def check(self, model: "RavenForCausalLM", latents: torch.Tensor, aux_inputs: dict) -> Exit:
        return (
            torch.zeros(latents.shape[0], device=latents.device, dtype=torch.bool),
            None,
            torch.zeros(latents.shape[0], device=latents.device),
        )


class EntropyDiffExitEvaluator(PerIterationExitEvaluator):
    """Exit based on change in output entropy."""

    def __init__(self, exit_threshold: Union[str, float] = "auto"):
        self.exit_threshold = 1e-3 if exit_threshold == "auto" else float(exit_threshold)

    def init(self, initial_latents: torch.Tensor):
        batch_size = initial_latents.shape[0]
        self.prev_entropy = torch.ones(batch_size, device=initial_latents.device) * 100.0

    def check(self, model: "RavenForCausalLM", latents: torch.Tensor, aux_inputs: dict) -> Exit:
        outputs = model.predict_from_latents(latents, **aux_inputs)
        logits: torch.Tensor = outputs.logits  # type: ignore
        probs = F.softmax(logits[:, -1, :], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        exit_values = (entropy - self.prev_entropy).abs()
        self.prev_entropy = entropy
        return exit_values < self.exit_threshold, outputs, exit_values


class LatentDiffExitEvaluator(PerIterationExitEvaluator):
    """Exit based on change in latent states."""

    def __init__(self, exit_threshold: Union[str, float] = "auto"):
        self.exit_threshold = 0.03 if exit_threshold == "auto" else float(exit_threshold)

    def init(self, initial_latents: torch.Tensor):
        self.prev_latents = initial_latents.clone().detach()

    def check(self, model: "RavenForCausalLM", latents: torch.Tensor, aux_inputs: dict) -> Exit:
        exit_values = ((latents - self.prev_latents).norm(dim=-1) / latents.norm(dim=-1)).mean(dim=-1)
        self.prev_latents = latents.clone().detach()
        return exit_values < self.exit_threshold, None, exit_values


class KLExitEvaluator(PerIterationExitEvaluator):
    """Exit based on KL divergence between successive outputs."""

    def __init__(self, model: "RavenForCausalLM", exit_threshold: Union[str, float] = "auto"):
        self.exit_threshold = 0.001 if exit_threshold == "auto" else float(exit_threshold)
        self.V = model.config.padded_vocab_size

    def init(self, initial_latents: torch.Tensor):
        batch_size = initial_latents.shape[0]
        self.prev_log_probs = ((1 / self.V) * torch.ones(batch_size, self.V, device=initial_latents.device)).log()

    def check(self, model: "RavenForCausalLM", latents: torch.Tensor, aux_inputs: dict) -> Exit:
        outputs = model.predict_from_latents(latents, **aux_inputs)
        logits: torch.Tensor = outputs.logits  # type: ignore
        log_probs = F.log_softmax(logits[:, -1, :].float(), dim=-1)
        exit_values = F.kl_div(log_probs, self.prev_log_probs, reduction="none", log_target=True).sum(dim=-1)
        self.prev_log_probs = log_probs
        return exit_values < self.exit_threshold, outputs, exit_values


class MinKLExitEvaluator(PerIterationExitEvaluator):
    """Exit based on min-p filtered KL divergence."""

    def __init__(self, model: "RavenForCausalLM", exit_threshold: Union[str, float] = "auto"):
        self.exit_threshold = 1e-5 if exit_threshold == "auto" else float(exit_threshold)
        self.V = model.config.padded_vocab_size

    def init(self, initial_latents: torch.Tensor):
        batch_size = initial_latents.shape[0]
        self.prev_log_probs = ((1 / self.V) * torch.ones(batch_size, self.V, device=initial_latents.device)).log()

    def _calc_minp_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits[:, -1, :], dim=-1)
        max_probs = probs.max(dim=-1, keepdim=True)[0]
        probs_mask = probs < (0.1 * max_probs)
        masked_probs = probs
        masked_probs[probs_mask] = 1 / self.V
        probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
        return probs.log()

    def check(self, model: "RavenForCausalLM", latents: torch.Tensor, aux_inputs: dict) -> Exit:
        outputs = model.predict_from_latents(latents, **aux_inputs)
        logits: torch.Tensor = outputs.logits  # type: ignore
        log_probs = self._calc_minp_log_probs(logits)
        exit_values = F.kl_div(log_probs, self.prev_log_probs, reduction="none", log_target=True).sum(dim=-1)
        self.prev_log_probs = log_probs
        return exit_values < self.exit_threshold, outputs, exit_values


class ArgmaxStabilityExitEvaluator(PerIterationExitEvaluator):
    """Exit based on argmax stability over consecutive steps."""

    def __init__(self, exit_threshold: Union[str, int] = "auto"):
        self.exit_threshold = 5 if exit_threshold == "auto" else int(exit_threshold)

    def init(self, initial_latents: torch.Tensor):
        batch_size = initial_latents.shape[0]
        self.prev_argmax = torch.ones(batch_size, dtype=torch.long, device=initial_latents.device) * -1
        self.stable_for_n_steps = torch.zeros(batch_size, dtype=torch.long, device=initial_latents.device)

    def check(self, model: "RavenForCausalLM", latents: torch.Tensor, aux_inputs: dict) -> Exit:
        outputs = model.predict_from_latents(latents, **aux_inputs)
        logits: torch.Tensor = outputs.logits  # type: ignore
        current_argmax = logits[:, -1, :].argmax(dim=-1)
        stable_for_n_steps = torch.where(
            current_argmax == self.prev_argmax, self.stable_for_n_steps + 1, torch.zeros_like(self.stable_for_n_steps)
        )
        exit_values = stable_for_n_steps
        self.prev_argmax = current_argmax
        self.stable_for_n_steps = stable_for_n_steps
        return exit_values >= self.exit_threshold, outputs, exit_values


class CosineExitEvaluator(PerIterationExitEvaluator):
    """Exit based on cosine similarity between successive latent states."""

    def __init__(self, exit_threshold: Union[str, float] = "auto"):
        self.exit_threshold = 1e-3 if exit_threshold == "auto" else float(exit_threshold)

    def init(self, initial_latents: torch.Tensor):
        self.prev_latents = initial_latents.clone().detach()

    def check(self, model: "RavenForCausalLM", latents: torch.Tensor, aux_inputs: dict) -> Exit:
        cosine_sim = (
            (latents * self.prev_latents).sum(dim=-1) / latents.norm(dim=-1) / self.prev_latents.norm(dim=-1)
        ).mean(dim=1)
        exit_values = 1 - cosine_sim
        self.prev_latents = latents.clone().detach()
        return exit_values < self.exit_threshold, None, exit_values


class NumStepsGenerator(PerIterationExitEvaluator):
    def __init__(self, steps_fn: Callable):
        self.steps_fn = steps_fn
        self.counter = 0
        self.target_steps = 0
        self.current_step = 0

    def init(self, initial_latents):
        self.target_steps = self.steps_fn(self.counter)
        self.counter += 1
        self.current_step = 0

    def check(self, model: "RavenForCausalLM", latents: torch.Tensor, aux_inputs: dict) -> Exit:
        self.current_step += 1
        should_exit = self.current_step >= self.target_steps
        return (
            torch.full((latents.shape[0],), should_exit, dtype=torch.bool, device=latents.device),
            None,
            torch.zeros(latents.shape[0], device=latents.device),
        )


def get_adaptive_exit_evaluator(
    model: "RavenForCausalLM", criterion: str, exit_threshold: Union[str, float, int]
) -> PerIterationExitEvaluator:
    """Factory function to create appropriate exit evaluator."""
    if criterion == "entropy-diff":
        return EntropyDiffExitEvaluator(exit_threshold)
    elif criterion == "latent-diff":
        return LatentDiffExitEvaluator(exit_threshold)
    elif criterion == "cosine":
        return CosineExitEvaluator(exit_threshold)
    elif "kl" in criterion:
        if criterion == "minp-kl":
            return MinKLExitEvaluator(model, exit_threshold)
        else:
            return KLExitEvaluator(model, exit_threshold)
    elif criterion == "argmax-stability":
        return ArgmaxStabilityExitEvaluator(exit_threshold)  # type: ignore
    elif criterion == "none":
        return NoOpExitEvaluator()
    else:
        raise ValueError(f"Invalid adaptive compute strategy: {criterion}")


#################################### HF registration ############################################################

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# New
RavenConfig.register_for_auto_class()

RavenForCausalLM.register_for_auto_class("AutoModel")
RavenForCausalLM.register_for_auto_class("AutoModelForCausalLM")

# Old?
AutoConfig.register("huginn_raven", RavenConfig)
AutoModel.register(RavenConfig, RavenForCausalLM)
AutoModelForCausalLM.register(RavenConfig, RavenForCausalLM)
