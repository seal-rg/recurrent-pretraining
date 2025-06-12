"""This is an old modeling file, the updated version is in raven_modeling_minimal.py"""

import torch
import math

from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Union, Any

from .raven_config_minimal import RavenConfig
from transformers.cache_utils import Cache, DynamicCache

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
    _supports_static_cache = False

    def _init_weights(self, module):
        if not torch.rand((1,)).is_meta:
            print("Random Initialization not implemented.")


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
        step_idx: int,
        lookup_strategy: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lookup_strategy = self.lookup_strategy if lookup_strategy is None else lookup_strategy
        if "compress-" in self.lookup_strategy and step_idx > 1:  # hardcode for current model!
            compression_stage = int(self.lookup_strategy.split("compress-")[1][1:])
            if "compress-s" in self.lookup_strategy:
                new_step_idx = (step_idx - 2) % compression_stage + 2
            else:
                new_step_idx = (step_idx - 2) // compression_stage + 2
            # @ print(step_idx, new_step_idx, compression_stage)
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
            # print(f"Overwrote cache entry for step_idx {step_idx}") # likely the head
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
        else:  # some entries where not previously computed
            # if lookup_strategy.startswith("latest"):
            #     latest_keys = []
            #     latest_values = []
            #     for token_pos in range(self._seen_tokens):
            #         # Find the latest step that has this token position
            #         max_step = max((s for s in range(step_idx + 1) if token_pos in self.key_cache[s]), default=None)
            #         if max_step is None:
            #             raise ValueError(f"No cache entry found for token position {token_pos}")
            #         latest_keys.append(self.key_cache[max_step][token_pos])
            #         latest_values.append(self.value_cache[max_step][token_pos])
            #     return torch.stack(latest_keys, dim=-2), torch.stack(latest_values, dim=-2)
            if lookup_strategy.startswith("latest-m4"):
                latest_keys = []
                latest_values = []
                for token_pos in range(self._seen_tokens):
                    # For steps >= 2, use modulo 4
                    if step_idx >= 2:
                        # Find valid steps for this token position
                        valid_steps = [s for s in range(step_idx + 1) if token_pos in self.key_cache[s]]
                        max_step = max([s for s in valid_steps if s >= 2 and s % 4 == step_idx % 4])
                    else:
                        max_step = step_idx if token_pos in self.key_cache[step_idx] else 0
                    if max_step is None:
                        raise ValueError(f"No cache entry found for token position {token_pos}")
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
        step_idx: int,
        mask: Optional[Tensor] = None,
        past_key_values: Optional[Cache] = None,
        return_attn: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
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
            k, v = past_key_values.update(k, v, step_idx)

        if return_attn:
            y, attention_map = self.compute_eager_sdpa(q, k, v, attn_mask=mask)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=q.shape[2] > 1
            )
        y = y.transpose(1, 2).reshape(B, S, E).contiguous()  # reshape is a view if possible (it mostly is)
        return self.proj(y), attention_map if return_attn else None

    def compute_eager_sdpa(self, q, k, v, attn_mask):
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            scores = scores + attn_mask
        if q.shape[2] > 1:
            causal_mask = torch.triu(torch.ones(q.shape[2], q.shape[2]), diagonal=1).bool()
            scores.masked_fill_(causal_mask.to(scores.device), float("-inf"))

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        y = torch.matmul(attention_weights, v)
        return y, attention_weights.max(dim=1)[0]


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
        mask: Optional[Tensor] = None,
        past_key_values: Optional[Cache] = None,
        return_attn: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        attn_out, attn_map = self.attn(self.norm_1(x), freqs_cis, step_idx, mask, past_key_values, return_attn)
        x = self.norm_2(attn_out + x)
        x = self.norm_4(self.mlp(self.norm_3(x)) + x)
        return x, attn_map


class RavenForCausalLM(RavenPreTrainedModel, GenerationMixin):
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
        self.emb_scale = config.init_values["embed_scale"]
        # Head
        self.lm_head = torch.nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        if self.config.tie_embeddings:
            self.tie_weights()
        # rope
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

    def get_input_embeddings(self):
        return self.transformer.wte

    def get_output_embeddings(self):
        return self.lm_head

    def _precompute_freqs_cis(self):
        # can actually be a buffer now, and remains in fp32! (at least in the settings I tested)
        freqs_cis = precompute_freqs_cis(
            self.config.n_embd // self.config.num_attention_heads, self.config.block_size, self.config.rope_base, 1
        )
        return freqs_cis

    def forward(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
        input_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_steps: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_details: dict = {
            "return_logits": True,
            "return_latents": True,
            "return_attention": False,
            "return_head": False,
            "return_stats": False,
        },
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
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
            input_embeds = self.transformer.wte(input_ids)

        if self.emb_scale != 1:
            input_embeds = input_embeds * self.emb_scale  # type: ignore

        if use_cache and past_key_values is None:
            past_key_values = HuginnDynamicCache()
        attn_maps = {}
        return_attn = output_details["return_attention"]

        # Non-recurrent prelude
        for block_idx, block in enumerate(self.transformer.prelude):
            input_embeds, attn_map = block(
                input_embeds, freqs_cis, block_idx, attention_mask, past_key_values, return_attn=return_attn
            )
            attn_maps[block_idx] = attn_map

        # Main recurrence
        x, num_steps_no_grad, num_steps_with_grad, xk, block_idx, attn_maps = self.iterate_forward(
            input_embeds,  # type: ignore
            input_states,
            freqs_cis,
            block_idx,
            attention_mask,
            past_key_values,
            num_steps,
            attn_maps,
            return_attn=return_attn,
        )
        latent_states = x.clone().detach()

        # Coda layers
        for block_idx, block in enumerate(self.transformer.coda, start=1):
            x, attn_map = block(x, freqs_cis, -block_idx, attention_mask, past_key_values, return_attn=return_attn)
            attn_maps[-block_idx] = attn_map
        x = self.transformer.ln_f(x)

        # Prediction head, assuming labels really are labels and not equal to input_ids
        if labels is not None:
            logits = self.lm_head(x).float()
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
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
            attention_maps=attn_maps if output_details["return_attention"] else None,  # type: ignore
            stats=self.get_stats(logits, x, latent_states, xk, input_embeds, num_steps_no_grad, num_steps_with_grad)
            if output_details["return_stats"]
            else None,
        )

    @torch._dynamo.disable(recursive=False)  # type: ignore
    def iterate_forward(
        self,
        input_embeds,
        input_states,
        freqs_cis,
        block_idx,
        mask,
        past_key_values: Optional[Cache] = None,
        num_steps: Optional[torch.Tensor] = None,
        attn_maps: dict = {},
        return_attn: bool = False,
    ):
        x = xk = self.initialize_state(input_embeds) if input_states is None else input_states.clone()
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
            for step in range(num_steps_no_grad):
                xk = x
                x, block_idx, attn_maps = self.core_block_forward(
                    xk, input_embeds, freqs_cis, mask, past_key_values, block_idx, attn_maps, return_attn
                )

        for step in range(num_steps_with_grad):
            xk = x
            x, block_idx, attn_maps = self.core_block_forward(
                xk, input_embeds, freqs_cis, mask, past_key_values, block_idx, attn_maps, return_attn
            )
        return self.transformer.ln_f(x), num_steps_no_grad, num_steps_with_grad, xk.detach(), block_idx, attn_maps

    def core_block_forward(
        self,
        x,
        input_embeds,
        freqs_cis,
        mask,
        past_key_values,
        block_idx: Union[torch.Tensor, int],
        attn_maps: dict = {},
        return_attn: bool = False,
    ):
        x = self.transformer.adapter(torch.cat([x, input_embeds.to(x.device)], dim=-1))
        for idx, block in enumerate(self.transformer.core_block, start=1):
            x, attn_map = block(x, freqs_cis, block_idx + idx, mask, past_key_values, return_attn=return_attn)
            attn_maps[block_idx + idx] = attn_map
        return x, block_idx + idx, attn_maps

    @torch.no_grad()
    def iterate_one_step(
        self,
        input_embeds,
        input_states,
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        block_idx: Union[torch.Tensor, int] = 0,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Cache] = None,
        attn_maps: dict = {},
    ):
        if position_ids is None and cache_position is None:
            freqs_cis = self.freqs_cis[:, : input_embeds.shape[1]]
        elif position_ids is not None:
            freqs_cis = self.freqs_cis.index_select(1, position_ids.squeeze())
        elif cache_position is not None:
            freqs_cis = self.freqs_cis[:, cache_position]
        x, block_idx, attn_maps = self.core_block_forward(
            input_states, input_embeds, freqs_cis, attention_mask, past_key_values, block_idx, attn_maps
        )
        return x, block_idx, attn_maps

    def predict_from_latents(
        self,
        latents,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        return_attn: bool = False,
        attn_maps: dict = {},
    ):
        if position_ids is None and cache_position is None:
            freqs_cis = self.freqs_cis[:, : latents.shape[1]]
        elif position_ids is not None:
            freqs_cis = self.freqs_cis.index_select(1, position_ids.squeeze())
        elif cache_position is not None:
            freqs_cis = self.freqs_cis[:, cache_position]
        x = self.transformer.ln_f(latents)
        # Coda layers
        for block_idx, block in enumerate(self.transformer.coda, start=1):
            x, attn_map = block(x, freqs_cis, -block_idx, attention_mask, past_key_values)
        attn_maps[block_idx] = attn_map
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x).float()

        return CausalLMOutputRecurrentLatents(
            loss=torch.as_tensor(0.0),
            log_ppl=torch.as_tensor(0.0),
            logits=logits,
            past_key_values=past_key_values,
            attention_maps=attn_maps if len(attn_maps) > 0 else None,
        )

    def embed_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, int, dict[int, Tensor]]:
        # Support multiple position formats:
        if position_ids is None and cache_position is None:
            freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
        elif position_ids is not None:
            freqs_cis = self.freqs_cis.index_select(1, position_ids.squeeze())
        elif cache_position is not None:
            freqs_cis = self.freqs_cis[:, cache_position]

        input_embeds = self.transformer.wte(input_ids)

        if self.emb_scale != 1:
            input_embeds = input_embeds * self.emb_scale  # type: ignore

        if use_cache and past_key_values is None:
            past_key_values = HuginnDynamicCache()

        # Non-recurrent prelude
        attn_maps = {}
        for block_idx, block in enumerate(self.transformer.prelude):
            input_embeds, attn_maps = block(
                input_embeds, freqs_cis, block_idx, attention_mask, past_key_values, return_attn
            )
        return input_embeds, block_idx, attn_maps

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

    def initialize_state(self, input_embeds, deterministic: bool = False):
        x = torch.randn_like(input_embeds)
        std = self.config.init_values["std"]
        torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if self.emb_scale != 1:
            x = x * self.emb_scale
        return x if not deterministic else x.zero_()

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = {}
        model_inputs["cache_position"] = cache_position
        current_input_length = input_ids.shape[1]
        if past_key_values is not None:
            if type(past_key_values) != HuginnDynamicCache:
                # Need to use custom cache, detect and replace HF dynamic cache if generate injects it
                assert past_key_values.get_seq_length() == 0
                past_key_values = HuginnDynamicCache()
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
        if any(
            k in kwargs
            for k in ("continuous_compute", "latent_dampening", "criterion", "exit_threshold", "cache_kwargs")
        ):
            print("Dispatching to custom generate function call")
            return self.generate_with_adaptive_compute(*args, **kwargs)
        else:
            return super().generate(*args, **kwargs)

    @torch.no_grad()
    def generate_minimal(
        self,
        input_ids: torch.LongTensor,
        generation_config: Optional[GenerationConfig] = None,  # type: ignore
        tokenizer=None,
        streamer=None,
        continuous_compute=False,  # warm-start state / continuous CoT
        cache_kwargs: dict = {},
        **model_kwargs,
    ) -> Union[torch.Tensor, dict[str, Any]]:
        """Minimal single-sequence generation. Template for more complicated generate tasks"""
        # Setup
        if generation_config is None:
            generation_config: GenerationConfig = self.generation_config  # type: ignore
        model_kwargs["past_key_values"] = HuginnDynamicCache(**cache_kwargs)
        model_kwargs["use_cache"] = True
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        stop_tokens = self._get_stops(generation_config, tokenizer).to(input_ids.device)
        if continuous_compute:
            embedded_inputs, _, _ = self.embed_inputs(input_ids)
            model_kwargs["input_states"] = self.initialize_state(embedded_inputs)
        # Generate tokens
        for _ in range(generation_config.max_length - input_ids.shape[1]):
            # Forward pass
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs)
            next_token_logits = outputs.logits[0, -1, :]
            if continuous_compute:
                current_last_latent = outputs.latent_states[:, -1:, :]

            # Sample or select next token
            if generation_config.do_sample:
                if generation_config.temperature:
                    next_token_logits = next_token_logits / generation_config.temperature

                probs = F.softmax(next_token_logits, dim=-1)

                # Apply top_k
                if generation_config.top_k:
                    top_k_probs, _ = torch.topk(probs, generation_config.top_k)
                    probs[probs < top_k_probs[-1]] = 0
                # Apply top_p
                if generation_config.top_p:
                    sorted_probs = torch.sort(probs, descending=True)[0]
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    probs[cumsum > generation_config.top_p] = 0
                # Apply min_p
                if generation_config.min_p:
                    probs[probs < generation_config.min_p * probs.max()] = 0

                probs = probs / probs.sum()
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token[None, :]], dim=-1)  # type: ignore

            if streamer:
                streamer.put(next_token.cpu())

            # Update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            if continuous_compute:
                model_kwargs["input_states"] = current_last_latent

            # Check if we hit a stop token
            if stop_tokens is not None and next_token in stop_tokens:
                break

        if streamer:
            streamer.end()

        if generation_config.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
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
        input_ids: torch.LongTensor,
        generation_config: Optional[GenerationConfig] = None,  # type: ignore
        tokenizer=None,
        streamer=None,
        continuous_compute=False,  # warm-start state / continuous CoT
        latent_dampening=False,
        criterion="entropy-diff",
        exit_threshold: Union[str, float, int] = "auto",
        cache_kwargs: dict = {},
        **model_kwargs,
    ) -> Union[torch.Tensor, GenerateDecoderOnlyOutput]:
        """
        Generate tokens with adaptive compute. This is NOT the most efficient implementation.
        For batches, on each token, we iterate until the entire batch finishes.
        """
        # Setup
        if generation_config is None:
            generation_config: GenerationConfig = self.generation_config  # type: ignore
        model_kwargs["past_key_values"] = HuginnDynamicCache(**cache_kwargs)
        model_kwargs["use_cache"] = True
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        stop_tokens = self._get_stops(generation_config, tokenizer).to(input_ids.device)
        batch_size = input_ids.shape[0]
        compute_steps = []

        # Set up continuous compute if enabled
        if continuous_compute:
            embedded_inputs, _, _ = self.embed_inputs(input_ids)
            current_last_latents = self.initialize_state(embedded_inputs)

        # Track which sequences have finished
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # Generate tokens
        for step in range(generation_config.max_length - input_ids.shape[1]):
            # Adaptive compute forward
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            aux_inputs = {
                k: model_inputs[k] for k in ["cache_position", "past_key_values", "attention_mask"] if k in model_inputs
            }
            embedded_inputs, block_idx, _ = self.embed_inputs(model_inputs["input_ids"], **aux_inputs)
            if not continuous_compute:
                current_latents = self.initialize_state(embedded_inputs, deterministic=False)
            else:
                current_latents = current_last_latents

            # Initialize criterion tracking for each sequence in batch
            exit_values_per_seq = [[] for _ in range(batch_size)]
            compute_steps_per_seq = [0] * batch_size
            exit_reached = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

            # Set up criterions based on selected strategy
            if criterion == "entropy-diff":
                entropy = torch.ones(batch_size, device=input_ids.device) * 100.0
                exit_threshold = 1e-3 if exit_threshold == "auto" else float(exit_threshold)
            elif criterion in ["latent-diff", "none"]:
                exit_threshold = 0.03 if exit_threshold == "auto" else float(exit_threshold)
            elif "kl" in criterion:
                V = self.config.padded_vocab_size
                log_probs = ((1 / V) * torch.ones(batch_size, V, device=input_ids.device)).log()
                if criterion == "minp-kl":
                    exit_threshold = 1e-6 if exit_threshold == "auto" else float(exit_threshold)
                else:
                    exit_threshold = 5e-4 if exit_threshold == "auto" else float(exit_threshold)
            elif criterion == "argmax-stability":
                stable_for_n_steps = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
                current_argmax = torch.ones(batch_size, dtype=torch.long, device=input_ids.device) * -1
                exit_threshold = 5 if exit_threshold == "auto" else int(exit_threshold)
            else:
                raise ValueError("Invalid adaptive compute strategy.")

            all_latents = []
            next_token_logits = None

            # Iterate through compute steps
            for compute_step in range(model_inputs["num_steps"]):
                prev_latents = current_latents.clone()
                current_latents, block_idx, _ = self.iterate_one_step(
                    embedded_inputs, current_latents, block_idx=block_idx, **aux_inputs
                )

                if latent_dampening:
                    all_latents.append(current_latents)

                if step > 0:  # do not exit in prefill:
                    # Check exit condition for each sequence in batch
                    if criterion == "entropy-diff":
                        prev_entropy = entropy
                        outputs = self.predict_from_latents(current_latents, **aux_inputs)
                        logits: torch.Tensor = outputs.logits  # type: ignore
                        probs = F.softmax(logits[:, -1, :], dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                        exit_values = (entropy - prev_entropy).abs()

                    elif criterion == "latent-diff":
                        norm_diff = (prev_latents - current_latents).norm(dim=-1) / current_latents.norm(dim=-1)
                        exit_values = norm_diff.mean(dim=-1)

                    elif "kl" in criterion:
                        outputs = self.predict_from_latents(current_latents, **aux_inputs)
                        logits: torch.Tensor = outputs.logits  # type: ignore
                        prev_log_probs = log_probs
                        if criterion == "minp-kl":
                            probs = F.softmax(logits[:, -1, :], dim=-1)
                            max_probs = probs.max(dim=-1, keepdim=True)[0]
                            probs_mask = probs < (0.1 * max_probs)
                            masked_probs = probs
                            masked_probs[probs_mask] = 1 / V
                            probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
                            log_probs = probs.log()
                        else:
                            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                        exit_values = F.kl_div(log_probs, prev_log_probs, reduction="none", log_target=True).sum(dim=-1)

                    elif criterion == "argmax-stability":
                        prev_argmax = current_argmax
                        outputs = self.predict_from_latents(current_latents, **aux_inputs)
                        logits: torch.Tensor = outputs.logits  # type: ignore
                        current_argmax = logits[:, -1, :].argmax(dim=-1)
                        stable_for_n_steps = torch.where(
                            current_argmax == prev_argmax, stable_for_n_steps + 1, torch.zeros_like(stable_for_n_steps)
                        )
                        exit_values = stable_for_n_steps

                    # Record values and check exits for each sequence
                    for i in range(batch_size):
                        if not exit_reached[i] and not finished_sequences[i]:
                            exit_values_per_seq[i].append(exit_values[i].item())

                    new_exits = (
                        exit_values < exit_threshold
                        if criterion != "argmax-stability"
                        else exit_values >= exit_threshold
                    )
                    new_exits = new_exits & ~exit_reached & ~finished_sequences

                    if new_exits.any():
                        exit_reached = exit_reached | new_exits
                        if criterion == "latent-diff":
                            # Normally we don't compute the output for latent-diff, but when there is an exit,
                            # we need to compute and save the output
                            outputs = self.predict_from_latents(current_latents, **aux_inputs)
                            logits: torch.Tensor = outputs.logits  # type: ignore
                        if next_token_logits is None:
                            next_token_logits = logits[:, -1, :].clone()
                        else:
                            next_token_logits = torch.where(
                                new_exits.unsqueeze(1).expand_as(logits[:, -1, :]), logits[:, -1, :], next_token_logits
                            )
                        for i in range(batch_size):
                            if new_exits[i]:
                                compute_steps_per_seq[i] = compute_step + 1

                    # If all sequences have exited, break early
                    if (exit_reached | finished_sequences).all():
                        break
            # This else is if the for loop finished without breaking
            else:
                if not latent_dampening:
                    outputs = self.predict_from_latents(current_latents, **aux_inputs)
                else:
                    dampened_latents = torch.sum(torch.cat(all_latents, dim=0), dim=0, keepdim=True)
                    outputs = self.predict_from_latents(dampened_latents, **aux_inputs)

                # For sequences that didn't exit early, use the final logits
                if next_token_logits is None:
                    next_token_logits = outputs.logits[:, -1, :]  # type: ignore
                else:
                    # Only update logits for sequences that didn't exit early
                    non_exit_mask = ~exit_reached & ~finished_sequences
                    next_token_logits = torch.where(
                        non_exit_mask.unsqueeze(1).expand_as(next_token_logits),
                        outputs.logits[:, -1, :],  # type: ignore
                        next_token_logits,
                    )

                    # Record compute steps for non-exited sequences
                    for i in range(batch_size):
                        if non_exit_mask[i]:
                            compute_steps_per_seq[i] = model_inputs["num_steps"]

            # Save latent states for continuous compute if enabled
            if continuous_compute:
                current_last_latents = current_latents[:, -1:, :]

            # Record compute steps for this token generation
            compute_steps.append([compute_steps_per_seq, exit_values_per_seq])

            # Sample or select next token based on generation config
            if generation_config.do_sample:
                next_token = self._sample_next_token(
                    next_token_logits,
                    generation_config.temperature,
                    generation_config.top_k,
                    generation_config.top_p,
                    generation_config.min_p,
                )
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # type: ignore

            input_ids = torch.cat([input_ids, next_token], dim=-1)  # type: ignore

            if streamer:
                streamer.put(next_token.cpu())

            # Update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            if continuous_compute:
                model_kwargs["input_states"] = current_last_latents

            # Check for finished sequences
            for i in range(batch_size):
                if not finished_sequences[i] and stop_tokens is not None and next_token[i, 0] in stop_tokens:
                    finished_sequences[i] = True

            # Break if all sequences are finished
            if finished_sequences.all():
                break

        if streamer:
            streamer.end()

        if generation_config.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=compute_steps,  # type: ignore
                logits=None,
                attentions=None,
                hidden_states=None,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return input_ids

    def _get_stops(self, generation_config, tokenizer):
        stop_tokens = set()
        if generation_config.eos_token_id is not None:
            stop_tokens.add(generation_config.eos_token_id)
        if hasattr(generation_config, "stop_strings") and tokenizer and generation_config.stop_strings:
            for s in generation_config.stop_strings:
                token_id = tokenizer(s, add_special_tokens=False)["input_ids"][0]
                stop_tokens.add(token_id)
        return torch.tensor(list(stop_tokens))

    def _sample_next_token(self, next_token_logits, temperature=None, top_k=None, top_p=None, min_p=None):
        """Helper function to sample the next token."""
        if temperature:
            next_token_logits = next_token_logits / temperature

        probs = F.softmax(next_token_logits, dim=-1)

        # Apply top_k
        if top_k:
            top_k_values, _ = torch.topk(probs, top_k, dim=-1)
            min_values = top_k_values[:, -1].unsqueeze(-1).expand_as(probs)
            probs = torch.where(probs < min_values, torch.zeros_like(probs), probs)

        # Apply top_p (nucleus sampling)
        if top_p:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create mask for probs to keep
            remove_indices = cumulative_probs > top_p
            remove_indices[:, 0] = False  # Keep at least the top probability

            # Convert sorted indices mask back to original indices mask
            mask = torch.zeros_like(probs, dtype=torch.bool)
            for i in range(probs.shape[0]):
                mask[i, sorted_indices[i, remove_indices[i]]] = True

            probs = torch.where(mask, torch.zeros_like(probs), probs)

        # Apply min_p
        if min_p:
            max_probs = probs.max(dim=-1, keepdim=True)[0]
            min_p_threshold = min_p * max_probs
            probs = torch.where(probs < min_p_threshold, torch.zeros_like(probs), probs)

        # Renormalize probabilities
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

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


#################################### Utils #######################################################################
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
