"""Implementation of all possible dynamic model variants.
Not closely following the model_axonn/model separation, writing for unified implementation
"""

import math
from typing import Any, Optional, Union


import torch
from torch import Tensor

from recpre.config_dynamic import Config, GPTConfig, RecurrentConfig
from .ops import LinearCrossEntropyLoss


AnyConfig = Union[GPTConfig, RecurrentConfig]


############################ Full Model Wrapper ########################################################################


class GPT(torch.nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        objective: dict[str, Any],
        gradient_checkpointing=False,
    ) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.emb_scale = config.init.embedding_scale
        self.transformer = torch.nn.ModuleDict(
            dict(
                wte=torch.nn.Embedding(config.padded_vocab_size, config.n_embd),
                abacus=(
                    torch.nn.Embedding(config.block_size, config.n_embd)
                    if self.config.use_abacus
                    else torch.nn.Identity()
                ),
                h=torch.nn.ModuleList(config.Block(config, layer_id=i) for i in range(config.n_layer)),
                ln_f=config.Norm(config.n_embd, eps=config.norm_eps),
            )
        )
        self.objective = objective
        if self.config.use_fused_head:
            self.lm_head = LinearCrossEntropyLoss(
                config.n_embd,
                config.padded_vocab_size,
                ignore_index=objective["ignore_index"],
                z_regularization=objective["z_regularization"],
                logit_scale=config.init.logit_scale,
                init_method=config.init.fn("head"),
            )
        else:
            self.lm_head = config.Linear(
                config.padded_vocab_size, config.n_embd, bias=False, init_method=config.init.fn("head")
            )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[Tensor] = None
        self.gradient_checkpointing = gradient_checkpointing

        self.step = 0
        self.monitoring = False
        self.latest_metrics = {}

        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.reset_parameters()

    def _precompute_freqs_cis(self):
        # Trigger resetting the rope-cache
        dim = self.config.intermediate_size if self.transformer.h[0].expanded else self.config.n_embd
        if self.config.randomize_positions_from is not None:
            max_length = self.config.randomize_positions_from
        else:
            max_length = self.config.block_size
        freqs_cis = precompute_freqs_cis(
            dim // self.config.num_attention_heads,
            max_length,
            self.config.rope_settings.rope_base,  # 50k in the newer models
            self.config.rope_settings.rope_condense_ratio,
        )  # can actually be a buffer now, and remains in fp32! (at least in the settings I tested)
        return freqs_cis

    def reset_parameters(self) -> None:
        self.config.init.apply(self.transformer.wte, "embedding")
        self.config.init.apply(self.transformer.ln_f, "normalization")
        # lm_head init already defined above

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> dict[str, Optional[torch.Tensor]]:
        if self.config.randomize_positions_from is not None and self.training:
            position_ids = torch.sort(
                torch.randint(0, self.config.randomize_positions_from, (input_ids.shape[1],), device=input_ids.device)
            )[0]

        if position_ids is None:
            freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
        else:
            freqs_cis = self.freqs_cis.index_select(1, position_ids)

        x = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)

        if self.emb_scale != 1:
            x = x * self.emb_scale

        for block in self.transformer.h:
            if not self.gradient_checkpointing:
                x = block(x, freqs_cis, attention_mask)
            else:
                x = self.config.checkpoint(block, x, freqs_cis, attention_mask)
        x = self.transformer.ln_f(x)
        if self.monitoring:
            self.monitor_module(x)

        if labels is not None:
            if self.config.use_fused_head:
                loss = self.lm_head(x, labels)
            else:
                logits = torch.matmul(x, self.lm_head.weight)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        else:
            outputs = torch.matmul(x, self.lm_head.weight).float()
            loss = torch.as_tensor(0.0)
        return {
            "loss": loss,
            "logits": outputs if return_logits else None,
            "log_ppl": loss.detach(),
        }

    @torch.no_grad()
    def monitor_module(self, x: torch.Tensor):
        x_c = x - x.mean(dim=-1, keepdim=True)
        normed_x = x_c / x_c.norm(dim=-1, keepdim=True)
        token_corr = (normed_x @ normed_x.transpose(1, 2)).mean() - 1 / x.shape[1]
        metrics = {"last_hidden_token_corr": token_corr, "last_hidden_norm": x.norm(dim=-1).mean()}
        self.latest_metrics = metrics  # will be picked up from monitoring caller


############################ Individual Blocks ########################################################################


class TransformerScaledPreNormBlock(torch.nn.Module):
    expanded = False

    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.norm_1 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_id=layer_id)
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.layer_id = layer_id

        branch_scales = config.init.get_scales(layer_id)
        # self.res_scale, self.skip_scale = float(branch_scales[0]), float(branch_scales[1])
        self.register_buffer("res_scale", torch.as_tensor(branch_scales[0]))
        self.register_buffer("skip_scale", torch.as_tensor(branch_scales[1]))

        # These should better be python floats to trigger ops from
        # https://github.com/pytorch/pytorch/blob/ef19824db8fa698499b705137952c7ba355c473b/aten/src/ATen/native/cuda/CUDALoops.cuh#L8-L28
        # where the float scalar is given as kernel param
        # but that breaks meta tensors in inexplicable ways that I can't fix. So buffers it is for now

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.attn(self.norm_1(x), freqs_cis, mask) * self.res_scale + x * self.skip_scale
        x = self.mlp(self.norm_2(x)) * self.res_scale + x * self.skip_scale
        return x

    def reset_parameters(self) -> None:
        self.config.init.apply(self.norm_1, "normalization")
        self.config.init.apply(self.norm_2, "normalization")


class TransformerPreNormBlock(TransformerScaledPreNormBlock):
    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.attn(self.norm_1(x), freqs_cis, mask) + x
        x = self.mlp(self.norm_2(x)) + x
        return x


class TransformerPostNormBlock(TransformerPreNormBlock):
    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm_1(self.attn(x, freqs_cis, mask) + x)
        x = self.norm_2(self.mlp(x) + x)
        return x


class ScaledTransformerPostNormBlock(TransformerPreNormBlock):
    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm_1(self.attn(x, freqs_cis, mask) * self.res_scale + x * self.skip_scale)
        x = self.norm_2(self.mlp(x) * self.res_scale + x * self.skip_scale)
        return x


class OPTransformerBlock(torch.nn.Module):
    expanded = False

    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.attn = OPSelfAttention(config, layer_id=layer_id)
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.layer_id = layer_id

        branch_scales = config.init.get_scales(layer_id)
        self.register_buffer("res_scale", torch.as_tensor(branch_scales[0]))
        self.register_buffer("skip_scale", torch.as_tensor(branch_scales[1]))

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x = self.attn(x, freqs_cis, mask) * self.res_scale + x * self.skip_scale
        # x = self.mlp(self.norm_2(x)) * self.res_scale + x * self.skip_scale
        x = self.attn(x, freqs_cis, mask) + x
        x = self.norm_2(self.mlp(x) + x)
        return x

    def reset_parameters(self) -> None:
        self.config.init.apply(self.norm_1, "normalization")
        self.config.init.apply(self.norm_2, "normalization")


class SwinScaledBlock(torch.nn.Module):
    expanded = False

    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.norm_1 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_id=layer_id)
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.layer_id = layer_id

        branch_scales = config.init.get_scales(layer_id)
        self.register_buffer("res_scale", torch.as_tensor(branch_scales[0]))
        self.register_buffer("skip_scale", torch.as_tensor(branch_scales[1]))

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm_1(self.attn(x, freqs_cis, mask)) * self.res_scale + x * self.skip_scale
        x = self.norm_2(self.mlp(x)) * self.res_scale + x * self.skip_scale
        return x

    def reset_parameters(self) -> None:
        self.config.init.apply(self.norm_1, "normalization")
        self.config.init.apply(self.norm_2, "normalization")


class SwinBlock(torch.nn.Module):
    expanded = False

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm_1(self.attn(x, freqs_cis, mask)) + x
        x = self.norm_2(self.mlp(x)) + x
        return x


class SandwichBlock(torch.nn.Module):
    expanded = False

    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.norm_1 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_id=layer_id)
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.norm_3 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.norm_4 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.layer_id = layer_id

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm_2(self.attn(self.norm_1(x), freqs_cis, mask) + x)
        x = self.norm_4(self.mlp(self.norm_3(x)) + x)
        return x

    def reset_parameters(self) -> None:
        self.config.init.apply(self.norm_1, "normalization")
        self.config.init.apply(self.norm_2, "normalization")
        self.config.init.apply(self.norm_3, "normalization")
        self.config.init.apply(self.norm_4, "normalization")


class SandwichBlockSwin(SandwichBlock):
    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm_2(self.attn(self.norm_1(x), freqs_cis, mask)) + x
        x = self.norm_4(self.mlp(self.norm_3(x))) + x
        return x


class RevTransformerPreNormBlock(torch.nn.Module):
    """Just using this for performance checks, no backprop implemented"""

    expanded = False

    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config

        self.norm_1 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_id=layer_id)
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.layer_id = layer_id

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)

        y1 = self.attn(self.norm_1(x2), freqs_cis, mask) + x1
        y2 = self.mlp(self.norm_2(y1)) + x2

        return torch.cat([y1, y2], dim=-1)

    def reset_parameters(self) -> None:
        self.config.init.apply(self.norm_1, "normalization")
        self.config.init.apply(self.norm_2, "normalization")


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.n_embd // self.n_head
        self.n_rep = self.n_head // self.n_kv_heads
        shape = (self.n_head + 2 * self.n_kv_heads) * self.head_dim
        self.chunks = [config.n_embd, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim]
        self.Wqkv = config.Linear(config.n_embd, shape, bias=config.bias, init_method=config.init.fn("qkv", layer_id))
        if config.qk_bias:
            self.qk_bias = torch.nn.Parameter(torch.zeros(2, 1, self.n_head, self.head_dim))
        # output projection
        self.proj = config.Linear(
            config.n_embd, config.n_embd, bias=config.bias, init_method=config.init.fn("out_attn", layer_id)
        )
        self.attention_impl = config.attn_nonlin_fn()

        self.layer_id = layer_id
        self.monitoring = False
        self.latest_metrics = {}

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, S, E = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.Wqkv(x)
        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        q, k, v = qkv.split(self.chunks, dim=2)
        # apply rotary
        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_kv_heads, self.head_dim)
        # repeat k/v heads if n_kv_heads < n_heads
        k = self.repeat_kv(k, self.n_rep)  # (B, S, nh, hs)
        v = self.repeat_kv(v.view(B, S, self.n_kv_heads, self.head_dim), self.n_rep)  # (B, S, nh, hs)
        # bias?
        if self.config.qk_bias:
            q_bias, k_bias = self.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
        if self.config.rope_settings.use_rope:
            q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)
        if self.config.qk_norm:
            q = torch.nn.functional.rms_norm(q, (q.shape[-1],))
            k = torch.nn.functional.rms_norm(k, (k.shape[-1],))

        y = self.attention_impl(q, k, v, mask)
        y = y.reshape(B, S, E).contiguous()  # reshape is a view if possible (it mostly is)

        return self.proj(y)

    @staticmethod
    def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            torch.unsqueeze(x, dim=3)
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    @torch.no_grad()
    # @torch.compile(mode="reduce-overhead")
    def monitor_layer(self, q, k, mask):
        """Casting metric computations into low precision because fusion op can be unreliable"""
        S = q.shape[1]
        if mask is None:
            attn_mask_tril = torch.ones([S, S], dtype=torch.bool, device=q.device).tril()
            attn_mask = torch.zeros_like(attn_mask_tril).to(q)
            attn_mask = attn_mask_tril.masked_fill(~attn_mask_tril, -10000)
        else:
            attn_mask = mask
        q = q.half().transpose(1, 2)  # (B, nh, S, hs)
        k = k.half().transpose(1, 2)
        A = ((q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)) + attn_mask).half()
        max_attn_logit = A.max()
        A = torch.softmax(A, dim=-1)  # overwrite A immediately
        if self.config.center_attention:
            A = A + torch.eye(S, device=A.device, dtype=A.dtype)[None, None, :, :]
        if self.config.debias_attention:
            mask_matrix = torch.ones([S, S], dtype=torch.bool, device=q.device).tril()
            A = A - mask_matrix / mask_matrix.sum(dim=1, keepdim=True)
        attn_entropy = 1 / S * torch.where(A > 0, -A.float() * A.float().log(), 0).sum(dim=-1).sum(dim=-1).mean()
        metrics = {f"attn_entropy_{self.layer_id}": attn_entropy, f"max_attn_logit_{self.layer_id}": max_attn_logit}
        self.latest_metrics = metrics  # will be picked up from monitoring caller


class DiffSelfAttention(CausalSelfAttention):
    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__(config, layer_id)
        self.subln = torch.nn.RMSNorm(self.head_dim, eps=config.norm_eps, elementwise_affine=False)
        self.register_buffer("lambda_init", torch.as_tensor(0.8 - 0.6 * math.exp(-0.3 * layer_id), dtype=torch.float))
        self.diff_lmb = torch.nn.Parameter(0.1 * torch.randn(self.head_dim, 4, dtype=torch.float))

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, S, E = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.Wqkv(x)
        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        q, k, v = qkv.split(self.chunks, dim=2)
        # apply rotary
        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_kv_heads, self.head_dim)
        if self.config.rope_settings.use_rope:
            q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)
        # repeat k/v heads if n_kv_heads < n_heads
        k = self.repeat_kv(k, self.n_rep)  # (B, S, nh, hs)
        v = self.repeat_kv(v.view(B, S, self.n_kv_heads, self.head_dim), self.n_rep)  # (B, S, nh, hs)

        q1, k1 = q[:, :, : self.n_head // 2, :], k[:, :, : self.n_head // 2, :]
        q2, k2 = q[:, :, self.n_head // 2 :, :], k[:, :, self.n_head // 2 :, :]

        y1, y2 = self.attention_impl(
            torch.cat([q1, q1, q2, q2], dim=2),
            torch.cat([k1, k1, k2, k2], dim=2),
            v.repeat(1, 1, 2, 1),
            mask,
        ).chunk(2, dim=2)

        lq1, lq2, lk1, lk2 = self.diff_lmb.chunk(4, dim=-1)
        # this whole lambda stack feels overcomplicated??
        lambda_1 = torch.exp(torch.sum(lq1 * lk1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(lq2 * lk2, dim=-1).float()).type_as(q)  # why?
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        y = self.subln(y1 - lambda_full * y2) * (1 - self.lambda_init)
        y = y.reshape(B, S, E).contiguous()

        return self.proj(y)

    def reset_parameters(self) -> None:
        pass  # use custom init for diff_lambda


class SimplifiedDiffSelfAttention(CausalSelfAttention):
    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__(config, layer_id)
        self.subln = torch.nn.RMSNorm(self.head_dim, eps=config.norm_eps, elementwise_affine=False)
        self.register_buffer("lmb_init", torch.as_tensor(0.8 - 0.6 * math.exp(-0.3 * layer_id), dtype=torch.float))
        self.diff_lmb = torch.nn.Parameter(0.1 * torch.randn(self.head_dim, dtype=torch.float))

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, S, E = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.Wqkv(x)
        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        q, k, v = qkv.split(self.chunks, dim=2)
        # apply rotary
        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_kv_heads, self.head_dim)
        if self.config.rope_settings.use_rope:
            q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)
        # repeat k/v heads if n_kv_heads < n_heads
        k = self.repeat_kv(k, self.n_rep)  # (B, S, nh, hs)
        v = self.repeat_kv(v.view(B, S, self.n_kv_heads, self.head_dim), self.n_rep)  # (B, S, nh, hs)

        q1, q2 = q[:, :, : self.n_head // 2, :], q[:, :, self.n_head // 2 :, :]
        k1, k2 = k[:, :, : self.n_head // 2, :], k[:, :, self.n_head // 2 :, :]

        y1, y2 = self.attention_impl(
            torch.cat([q1, q1, q2, q2], dim=2),
            torch.cat([k1, k1, k2, k2], dim=2),
            v.repeat(1, 1, 2, 1),
            mask,
        ).chunk(2, dim=2)
        y = self.subln(y1 - (self.diff_lmb + self.lmb_init) * y2) * (1 - self.lmb_init)
        y = y.reshape(B, S, E).contiguous()

        return self.proj(y)

    def reset_parameters(self) -> None:
        pass  # use custom init for diff_lambda


class TransformerPostNormBlockDiff(TransformerPostNormBlock):
    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        torch.nn.Module.__init__(self)
        self.config = config
        self.norm_1 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = DiffSelfAttention(config, layer_id=layer_id)
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.layer_id = layer_id


class TransformerPostNormBlockDiffSimplified(TransformerPostNormBlock):
    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        torch.nn.Module.__init__(self)
        self.config = config
        self.norm_1 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = SimplifiedDiffSelfAttention(config, layer_id=layer_id)
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.layer_id = layer_id


class PositionalAttention(CausalSelfAttention):
    """Notes:
    Original implementation is overcomplicated by generating I_S positional one-hot positional encodings
    and learnable matrices of shape Q, K: S x (S * num_heads), V: hxh
    Q and K are then reshaped to [num_heads, S, S] vs V: [B, num_heads, S, h]
    (so S doubles as hidden dimension E, which seems somewhat excessive, and Q and K and the pos_encoding are all large)
    and Attn is QKT = [..., num_heads, S, S] x [B, num_heads, S, h] ->  [B, num_heads, S, h]

    Instead, this version is using embedding layers for Q, K (which is equivalent to passing one-hots), as long as
    position_ids=range(block_size). But, the embedding dimension is chosen as the model's embed_dim, to keep things sane
    This can still be excessive for long seq_len problems, so the emb dim can be further reduced via emb_factor

    """

    def __init__(self, config: AnyConfig, layer_id: int, emb_factor: int = 1) -> None:
        torch.nn.Module.__init__(self)
        self.config = config
        self.n_head = config.num_attention_heads
        self.q_embedding = torch.nn.Embedding(config.block_size, config.n_embd // emb_factor)
        self.k_embedding = torch.nn.Embedding(config.block_size, config.n_embd // emb_factor)
        self.poshead_dim = config.n_embd // self.n_head // emb_factor
        self.register_buffer("position_ids", torch.arange(config.block_size).expand((1, -1)))
        self.Wv = config.Linear(
            config.n_embd, config.n_embd, bias=config.bias, init_method=config.init.fn("in_proj", layer_id)
        )
        self.head_dim = config.n_embd // self.n_head
        # output projection
        self.proj = config.Linear(
            config.n_embd, config.n_embd, bias=config.bias, init_method=config.init.fn("out_attn", layer_id)
        )

        self.attention_impl = config.attn_nonlin_fn()
        if emb_factor > 1:
            torch.backends.cuda.enable_math_sdp(True)

        self.layer_id = layer_id
        self.monitoring = False
        self.latest_metrics = {}

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, S, E = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)

        position_ids = self.position_ids[:, : x.shape[1]]
        q = self.q_embedding(position_ids).view(1, S, self.n_head, self.poshead_dim).expand(B, -1, -1, -1)
        k = self.k_embedding(position_ids).view(1, S, self.n_head, self.poshead_dim).expand(B, -1, -1, -1)
        v = self.Wv(x).view(B, S, self.n_head, self.head_dim)

        y = self.attention_impl(q, k, v, mask)
        y = y.reshape(B, S, E).contiguous()
        return self.proj(y)

    def reset_parameters(self) -> None:
        self.config.init.apply(self.q_embedding, "embedding")
        self.config.init.apply(self.k_embedding, "embedding")


class TransformerPostNormPosBlock(TransformerPostNormBlock):
    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        torch.nn.Module.__init__(self)
        self.config = config
        self.norm_1 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = PositionalAttention(config, layer_id=layer_id)
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.layer_id = layer_id


class ParallelGatedMLPSelfAttention(CausalSelfAttention):
    expanded = False

    def __init__(self, config: AnyConfig, layer_id) -> None:
        torch.nn.Module.__init__(self)
        self.config = config

        self.n_head = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.n_embd // self.n_head
        self.n_rep = self.n_head // self.n_kv_heads
        out_dim = config.intermediate_size * 2 + (self.n_head + 2 * self.n_kv_heads) * self.head_dim
        self.chunks = [
            config.intermediate_size,
            config.intermediate_size,
            config.n_embd,
            self.n_kv_heads * self.head_dim,
            self.n_kv_heads * self.head_dim,
        ]

        self.Wqkv_fc = config.Linear(
            config.n_embd, out_dim, bias=config.bias, init_method=config.init.fn("in_proj", layer_id)
        )
        self.proj = config.Linear(
            config.intermediate_size + config.n_embd,
            config.n_embd,
            bias=config.bias,
            init_method=config.init.fn("out_proj", layer_id),
        )
        self.nonlin = config.Nonlin()
        self.attention_impl = config.attn_nonlin_fn()
        self.norm = config.Norm(config.n_embd, eps=config.norm_eps)

    def attention(self, q, k, v, freqs_cis, mask):
        B, S, I = q.shape
        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_kv_heads, self.head_dim)
        if self.config.rope_settings.use_rope:
            q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)
        # repeat k/v heads if n_kv_heads < n_heads
        k = self.repeat_kv(k, self.n_rep)  # (B, S, nh, hs)
        v = self.repeat_kv(v.view(B, S, self.n_kv_heads, self.head_dim), self.n_rep)  # (B, S, nh, hs)

        # Self-attend: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)
        y = self.attention_impl(q, k, v, mask)
        y = y.reshape(B, S, I).contiguous()  # reshape is a view if possible (it mostly is)

        # if self.monitoring:
        #     self.monitor_layer(q, k, mask)
        return y

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        z, gate, q, k, v = self.Wqkv_fc(x).split(self.chunks, dim=-1)
        y = torch.cat([self.nonlin(z) * gate, self.attention(q, k, v, freqs_cis, mask)], dim=2)
        return self.norm(x + self.proj(y))


class GatedBlock(ParallelGatedMLPSelfAttention):
    expanded = True

    def __init__(self, config: AnyConfig, layer_id) -> None:
        super().__init__(config, layer_id)

        self.head_dim = config.intermediate_size // self.n_head
        self.chunks = [
            config.intermediate_size,
            config.intermediate_size,
            self.n_kv_heads * self.head_dim,
            self.n_kv_heads * self.head_dim,
        ]
        self.Wqkv_fc = config.Linear(
            config.n_embd, sum(self.chunks), bias=config.bias, init_method=config.init.fn("in_proj", layer_id)
        )
        self.proj = config.Linear(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            init_method=config.init.fn("out_proj", layer_id),
        )

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        z, q, k, v = self.Wqkv_fc(x).split(self.chunks, dim=-1)
        y = self.nonlin(z) * self.attention(q, k, v, freqs_cis, mask)
        return self.norm(x + self.proj(y))


class OPSelfAttention(torch.nn.Module):
    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.n_embd // self.n_head
        self.n_rep = self.n_head // self.n_kv_heads
        shape = (self.n_head + 2 * self.n_kv_heads) * self.head_dim
        self.chunks = [config.n_embd, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim]
        self.Wqkv = config.Linear(config.n_embd, shape, bias=config.bias, init_method=config.init.fn("qkv", layer_id))
        # output projection
        self.proj = config.Linear(
            config.n_embd, config.n_embd, bias=config.bias, init_method=config.init.fn("out_attn", layer_id)
        )
        self.q_norm = config.Norm(config.n_embd, eps=config.norm_eps)
        self.k_norm = config.Norm(self.n_head * self.n_kv_heads, eps=config.norm_eps)
        self.attention_impl = config.attn_nonlin_fn()

        self.layer_id = layer_id
        self.monitoring = False
        self.latest_metrics = {}

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, S, E = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.Wqkv(x)
        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        q, k, v = qkv.split(self.chunks, dim=2)

        # apply norm and rotary
        q = self.q_norm(q).view(B, S, self.n_head, self.head_dim)
        k = self.k_norm(k).view(B, S, self.n_kv_heads, self.head_dim)
        if self.config.rope_settings.use_rope:
            q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        k = self.repeat_kv(k, self.n_rep)  # (B, S, nh, hs)
        v = self.repeat_kv(v.view(B, S, self.n_kv_heads, self.head_dim), self.n_rep)  # (B, S, nh, hs)

        y = self.attention_impl(q, k, v, mask)
        y = y.reshape(B, S, E).contiguous()  # reshape is a view if possible (it mostly is)
        # if self.monitoring:
        #     self.monitor_layer(q, k, mask)

        # output projection
        return self.proj(y)

    @staticmethod
    def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            torch.unsqueeze(x, dim=3)
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )


class BaseMLP(torch.nn.Module):
    def __init__(self, config: AnyConfig, layer_id: int = 0, in_features: int = 0) -> None:
        super().__init__()
        self.config = config
        in_features = config.n_embd if in_features == 0 else in_features
        self.fc = config.Linear(
            in_features, config.intermediate_size, bias=config.bias, init_method=config.init.fn("in_proj", layer_id)
        )
        self.proj = config.Linear(
            config.intermediate_size, config.n_embd, bias=config.bias, init_method=config.init.fn("out_proj", layer_id)
        )
        self.nonlin = config.Nonlin()
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.nonlin(self.fc(x)))


class GatedMLP(torch.nn.Module):
    def __init__(self, config: AnyConfig, layer_id: int, in_features: int = 0) -> None:
        super().__init__()
        self.config = config
        in_features = config.n_embd if in_features == 0 else in_features
        self.fc = config.Linear(
            in_features, config.intermediate_size * 2, bias=config.bias, init_method=config.init.fn("glu", layer_id)
        )
        self.proj = config.Linear(
            config.intermediate_size, config.n_embd, bias=config.bias, init_method=config.init.fn("out_proj", layer_id)
        )
        self.nonlin = config.Nonlin()

    def forward(self, x: Tensor) -> Tensor:
        # modified to single FC layer to improve parallelism
        x_fc_1, x_fc_2 = self.fc(x).chunk(2, dim=-1)
        x = self.nonlin(x_fc_1) * x_fc_2
        return self.proj(x)


class GatedMLPSeparated(torch.nn.Module):
    def __init__(self, config: AnyConfig, layer_id: int, in_features: int = 0) -> None:
        super().__init__()
        self.config = config
        in_features = config.n_embd if in_features == 0 else in_features
        self.fc1 = config.Linear(
            in_features, config.intermediate_size, bias=config.bias, init_method=config.init.fn("glu", layer_id)
        )
        self.fc2 = config.Linear(
            in_features, config.intermediate_size, bias=config.bias, init_method=config.init.fn("glu", layer_id)
        )
        self.proj = config.Linear(
            config.intermediate_size, config.n_embd, bias=config.bias, init_method=config.init.fn("out_proj", layer_id)
        )
        self.nonlin = config.Nonlin()

    def forward(self, x: Tensor) -> Tensor:
        x_fc_1 = self.fc1(x)
        x = self.nonlin(x_fc_1) * self.fc2(x)
        return self.proj(x)


class GatedMLPXFormers(torch.nn.Module):
    def __init__(self, config: AnyConfig, layer_id: int, in_features: int = 0) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        in_features = config.n_embd if in_features == 0 else in_features
        assert isinstance(config.Nonlin(), torch.nn.SiLU)
        from xformers.ops import SwiGLU  # good luck compiling :> # type: ignore

        self.swiglu = SwiGLU(config.n_embd, config.intermediate_size, bias=False, _pack_weights=True)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.swiglu(x)

    def reset_parameters(self) -> None:
        self.config.init.apply(self.swiglu.w12, "glu", self.layer_id)
        self.config.init.apply(self.swiglu.w3, "w3", self.layer_id)


class GatedMLPFattori(torch.nn.Module):
    def __init__(self, config: AnyConfig, layer_id: int, in_features: int = 0) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        in_features = config.n_embd if in_features == 0 else in_features
        assert isinstance(config.Nonlin(), torch.nn.SiLU)
        from fused_swiglu import FusedSwiGLU  # type: ignore[not a default install]

        self.fc1 = config.Linear(
            in_features, config.intermediate_size, bias=False, init_method=config.init.fn("glu", layer_id)
        )
        self.fc2 = config.Linear(
            in_features, config.intermediate_size, bias=False, init_method=config.init.fn("glu", layer_id)
        )
        self.proj = config.Linear(
            config.intermediate_size, config.n_embd, bias=False, init_method=config.init.fn("w3", layer_id)
        )
        self.swiglu = FusedSwiGLU.apply

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.swiglu(x, self.fc1.weight.T, self.fc2.weight.T))


class HalfGateMLP(torch.nn.Module):
    def __init__(self, config: AnyConfig, layer_id: int, in_features: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        in_features = config.n_embd if in_features is None else in_features
        self.fc = config.Linear(
            in_features, config.n_embd, bias=config.bias, init_method=config.init.fn("w1", layer_id)
        )
        self.proj = config.Linear(
            config.n_embd, config.n_embd, bias=config.bias, init_method=config.init.fn("w3", layer_id)
        )
        self.nonlin = config.Nonlin()

    def forward(self, x: Tensor) -> Tensor:
        # modified to single FC layer to improve parallelism
        x_fc_1 = self.fc(x)
        x = self.nonlin(x_fc_1) * x
        return self.proj(x)


############################################### Embedders ####################################################
# Not used in the final implementation, see remark 3.1


class TimestepEmbedder(torch.nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    Implementation taken from DiT where it is coming from the OAI guided diffusion repo
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class ModulatedTransformerPostNormBlock(torch.nn.Module):
    expanded = False

    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.norm_1 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_id=layer_id)
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.layer_id = layer_id

        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(config.n_embd, 6 * config.n_embd, bias=True)
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Tensor:
        if context is not None:
            ctx = self.adaLN_modulation(context)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ctx.chunk(6, dim=-1)
            x = self.norm_1(self.attn(self.modulate(x, shift_msa, scale_msa), freqs_cis, mask) * gate_msa + x)
            x = self.norm_2(self.mlp(self.modulate(x, shift_mlp, scale_mlp)) * gate_mlp + x)
        else:
            x = self.norm_1(self.attn(x, freqs_cis, mask) + x)
            x = self.norm_2(self.mlp(x) + x)

        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def reset_parameters(self) -> None:
        self.config.init.apply(self.norm_1, "normalization")
        self.config.init.apply(self.norm_2, "normalization")


############################################### Recurrent GPT #################################################


class RecurrentGPT(torch.nn.Module):
    def __init__(
        self,
        config: RecurrentConfig,
        objective,
        gradient_checkpointing=False,
    ) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        # Transformer layers
        prelude = torch.nn.ModuleList(config.Block(config, layer_id=i) for i in range(config.n_layers_in_prelude))

        if config.injection_type == "linear":
            adapter = config.Linear(
                config.n_embd * 2,
                config.n_embd,
                bias=config.bias,
                init_method=config.init.fn("in_proj", config.n_layers_in_prelude),
            )
        elif config.injection_type == "ffn":
            adapter = config.MLP(config, layer_id=0, in_features=config.n_embd * 2)
        else:
            adapter = torch.nn.Identity()

        core_block = torch.nn.ModuleList(
            config.Block(config, layer_id=i + config.n_layers_in_prelude)
            for i in range(config.n_layers_in_recurrent_block)
        )
        o = config.n_layers_in_prelude + config.n_layers_in_recurrent_block * config.mean_recurrence
        coda = torch.nn.ModuleList(config.Block(config, layer_id=i + o) for i in range(config.n_layers_in_coda))

        hidden_state_dim = config.n_embd if config.Block is not RevTransformerPreNormBlock else config.n_embd * 2
        self.transformer = torch.nn.ModuleDict(
            dict(
                wte=torch.nn.Embedding(config.padded_vocab_size, hidden_state_dim),
                prelude=prelude,
                adapter=adapter,
                core_block=core_block,
                coda=coda,
                ln_f=config.Norm(hidden_state_dim, eps=config.norm_eps),
            )
        )
        self.emb_scale = config.init.embedding_scale
        # Head
        if config.use_fused_head == "cce":
            self.lm_head = config.Linear(
                hidden_state_dim, config.padded_vocab_size, bias=False, init_method=config.init.fn("head")
            )
        elif config.use_fused_head == "hhe":
            from recpre.utils import LinearCrossEntropyLoss as LCE

            self.lm_head = LCE(
                hidden_state_dim,
                config.padded_vocab_size,
                ignore_index=objective["ignore_index"],
                init_method=config.init.fn("head"),
            )
        elif self.config.use_fused_head == "full-triton":
            self.lm_head = LinearCrossEntropyLoss(
                hidden_state_dim,
                config.padded_vocab_size,
                ignore_index=objective["ignore_index"],
                z_regularization=objective["z_regularization"],
                logit_scale=config.init.logit_scale,
                init_method=config.init.fn("head"),
                transposed_weight=not self.config.tie_embeddings,
            )
        else:
            self.lm_head = config.Linear(
                hidden_state_dim, config.padded_vocab_size, bias=False, init_method=config.init.fn("head")
            )
        if self.config.tie_embeddings:
            self.lm_head.weight = self.transformer.wte.weight
        self.objective = objective

        # rarely used features:
        if config.embed_step:
            self.step_embedding = TimestepEmbedder(config.n_embd)

        # Misc attributes
        self.max_seq_length = self.config.block_size
        self.gradient_checkpointing = gradient_checkpointing
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        # Externally set:
        self.step = 0
        self.monitoring = False
        self.latest_metrics = {}
        # Remaining inits:
        self.reset_parameters()

    def _precompute_freqs_cis(self):
        # Trigger resetting the rope-cache
        dim = self.config.intermediate_size if self.transformer.core_block[0].expanded else self.config.n_embd
        if self.config.randomize_positions_from is not None:
            max_length = self.config.randomize_positions_from
        else:
            max_length = self.config.block_size
        freqs_cis = precompute_freqs_cis(
            dim // self.config.num_attention_heads,
            max_length,
            self.config.rope_settings.rope_base,  # 50k in the newer models
            self.config.rope_settings.rope_condense_ratio,
        )  # can actually be a buffer now, and remains in fp32! (at least in the settings I tested)
        return freqs_cis

    def reset_parameters(self) -> None:
        self.config.init.apply(self.transformer.wte, "embedding")
        self.config.init.apply(self.transformer.ln_f, "normalization")
        # lm_head init already defined above

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        num_steps_pair: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        if self.config.randomize_positions_from is not None and self.training:
            position_ids = torch.sort(  # need to fork rng for distributed
                torch.randint(0, self.config.randomize_positions_from, (input_ids.shape[1],), device=input_ids.device)
            )[0]

        if position_ids is None:
            freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
        else:
            freqs_cis = self.freqs_cis.index_select(1, position_ids)

        input_embeds = self.transformer.wte(input_ids)
        if self.emb_scale != 1:
            input_embeds = input_embeds * self.emb_scale

        for _, block in enumerate(self.transformer.prelude):
            input_embeds = block(input_embeds, freqs_cis, attention_mask)

        x, num_steps_no_grad, num_steps_with_grad, xk = self.iterate_forward(
            input_embeds,  # type: ignore
            freqs_cis,
            attention_mask,
            num_steps_pair,
        )
        x_rec_output = x

        for _, block in enumerate(self.transformer.coda):
            if self.gradient_checkpointing and "in-coda" in self.config.activation_checkpoint_impl:
                x = self.config.checkpoint(block, x, freqs_cis, attention_mask)
            else:
                x = block(x, freqs_cis, attention_mask)
        if self.gradient_checkpointing and "in-coda" in self.config.activation_checkpoint_impl:
            x = self.config.checkpoint(self.transformer.ln_f, x)
        else:
            x = self.transformer.ln_f(x)

        if self.monitoring:
            self.monitor_module(x, x_rec_output, xk, input_embeds, num_steps_no_grad, num_steps_with_grad)

        if labels is not None:
            logits = None
            if self.config.use_fused_head == "cce":
                from cut_cross_entropy import linear_cross_entropy  # type: ignore[unusal import]

                loss = linear_cross_entropy(
                    x * self.config.init.logit_scale, self.lm_head.weight, labels, filter_eps="auto"
                )
            elif self.config.use_fused_head == "hhe" or self.config.use_fused_head == "full-triton":
                loss = self.lm_head(x * self.config.init.logit_scale, labels)
            else:
                logits = self.lm_head(x).float() * self.config.init.logit_scale
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
            log_ppl = loss.clone().detach()
            if self.config.mcleish_throttle and self.training:
                loss = loss / torch.as_tensor(num_steps_with_grad, device=loss.device)
            if self.config.elbayad_weighing and self.training:
                t = self.config.mean_recurrence
                weights = torch.arange(1, 16 * t, device=loss.device) ** self.config.elbayad_exponent
                weights /= torch.sum(torch.arange(1, t + 1, device=loss.device) ** self.config.elbayad_exponent, dim=0)
                this_weight = weights[num_steps_no_grad + num_steps_with_grad] / weights[t // 2]
                loss = loss * this_weight
        else:
            if self.config.use_fused_head == "cce":
                logits = self.lm_head(x).float() * self.config.init.logit_scale
            elif self.config.use_fused_head == "full-triton":
                logits = (
                    torch.matmul(
                        x, self.lm_head.weight.T if self.config.tie_embeddings else self.lm_head.weight
                    ).float()
                    * self.config.init.logit_scale
                )
            else:
                logits = self.lm_head(x).float() * self.config.init.logit_scale
            loss, log_ppl = torch.as_tensor(0.0), torch.as_tensor(0.0)

        return {
            "loss": loss,
            "logits": logits if return_logits else None,
            "log_ppl": log_ppl,
        }

    @torch._dynamo.disable(recursive=False)  # type: ignore
    def iterate_forward(self, input_embeds, freqs_cis, mask, num_steps_pair: Optional[torch.Tensor] = None):
        x = self.initialize_state(input_embeds)

        if num_steps_pair is None:
            num_steps_no_grad, num_steps_with_grad = self.randomized_iteration_sampler()  # type: ignore
        elif len(num_steps_pair) > 1:
            num_steps_no_grad, num_steps_with_grad = num_steps_pair
        else:
            num_steps_no_grad, num_steps_with_grad = num_steps_pair, torch.tensor(0)

        if self.config.randomize_embed_step:
            offset = torch.randint(0, self.config.mean_recurrence * 8, (1,), device=input_embeds.device)
        else:
            offset = 0

        with torch.no_grad():
            # ultra annoying in ddp due to
            # https://discuss.pytorch.org/t/does-distributeddataparallel-work-with-torch-no-grad-and-find-unused-parameters-false/122594
            # for now running with find_unused_params=True enabled even though the graph structure is (technically) clear
            # and all parameters are always used
            for step in range(num_steps_no_grad):
                xk = x
                x = self.core_block_forward(xk, input_embeds, freqs_cis, mask, step + offset)

        for step in range(num_steps_with_grad):
            xk = x
            if self.gradient_checkpointing and "per-iteration" in self.config.activation_checkpoint_impl:
                x = self.config.checkpoint(
                    self.core_block_forward, xk, input_embeds, freqs_cis, mask, num_steps_no_grad + step + offset
                )
            else:
                x = self.core_block_forward(xk, input_embeds, freqs_cis, mask, num_steps_no_grad + step + offset)
        return self.transformer.ln_f(x), num_steps_no_grad, num_steps_with_grad, xk.detach()

    def core_block_forward(self, x, input_embeds, freqs_cis, mask, step: Union[torch.Tensor, int]):
        if self.config.embed_step:
            context = self.step_embedding(torch.as_tensor([step], device=input_embeds.device))
        else:
            context = None

        if self.config.injection_type == "add":
            x = x + input_embeds
        elif self.config.injection_type == "gate":
            x = x * input_embeds
        elif self.config.injection_type in ["linear", "ffn"]:
            x = self.transformer.adapter(torch.cat([x, input_embeds], dim=-1))
        elif self.config.injection_type == "modulated":  # use in conjunction with Modulated blocks, not with embed_step
            context = x.clone()
        else:
            raise ValueError("Invalid injection type")

        if self.config.intermediate_noise_injection > 0:
            n = self.config.intermediate_noise_injection
            if self.config.geom_noise_injection == "geom":
                step1 = torch.as_tensor(step + 1, device=x.device)  # need to cast for compile
                x = x * (1 - n / step1) + torch.randn_like(x) * n / step1
            elif self.config.geom_noise_injection == "sqrt":
                step1sqrt = torch.as_tensor(step + 1, device=x.device).sqrt()  # need to cast for compile
                x = x * (1 - n / step1sqrt) + torch.randn_like(x) * n / step1sqrt
            elif self.config.geom_noise_injection == "line":
                noise = max(n, (self.config.maximal_recurrence - step) / self.config.maximal_recurrence)  # type: ignore
                x = x * (1 - noise) + torch.randn_like(x) * noise
            elif self.config.geom_noise_injection == "chi":
                noise = 2 * torch.rand(1, device=x.device, dtype=x.dtype) * n
            else:
                x = x * (1 - n) + torch.randn_like(x) * n

        if isinstance(self.transformer.core_block[0], ModulatedTransformerPostNormBlock):
            for _, block in enumerate(self.transformer.core_block):
                if not self.gradient_checkpointing:
                    x = block(x, freqs_cis, mask, context=context)
                else:
                    x = self.config.checkpoint(block, x, freqs_cis, mask, context=context)
        else:
            if context is not None:
                x = x + context

            for _, block in enumerate(self.transformer.core_block):
                if self.gradient_checkpointing and "per-block" in self.config.activation_checkpoint_impl:
                    x = self.config.checkpoint(block, x, freqs_cis, mask)
                else:
                    x = block(x, freqs_cis, mask)
        return x

    @torch._dynamo.disable(recursive=False)  # type: ignore
    def randomized_iteration_sampler(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Outputs are long tensors so that they can be passed through compiled functions"""
        if torch.rand((1,)).is_meta:  # annoying clause to make meta-tensor-based flop counting work
            # these values are only approximate, not all schemes exactly target a mean of n and k
            # they overvalue the compute done when curricula are turned on, but that may be considered
            # a feature, given that it is a valid form of training acceleration
            return self.config.mean_recurrence - self.config.mean_backprop_depth, self.config.mean_backprop_depth  # type: ignore

        seed_n = 514229 + self.step  # easiest way to make the sampler re-runnable in checkpointing
        seed_k = 317811 + self.step
        if not self.config.lockstep_n and torch.distributed.is_initialized():
            seed_n = seed_n * (torch.distributed.get_rank() + 1)
        if not self.config.lockstep_k and torch.distributed.is_initialized():
            seed_k = seed_k * (torch.distributed.get_rank() + 1)

        n_generator = torch.Generator(device="cpu")
        n_generator.manual_seed(seed_n % (2**31 - 1))
        k_generator = torch.Generator(device="cpu")
        k_generator.manual_seed(seed_k % (2**31 - 1))

        if "curriculum-" in self.config.sampling_scheme:
            ramp_length = int(self.config.sampling_scheme.split("curriculum-")[1])
            if self.step > ramp_length:
                t = max(self.config.mean_recurrence - self.config.mean_backprop_depth, 0)
                s = self.config.mean_backprop_depth
            else:
                slope = self.step / ramp_length
                t = max(math.ceil(slope * (self.config.mean_recurrence - self.config.mean_backprop_depth)), 0)
                s = max(math.ceil(slope * self.config.mean_backprop_depth), 1)
        else:
            t = max(self.config.mean_recurrence - self.config.mean_backprop_depth, 0)
            s = self.config.mean_backprop_depth

        if self.training:
            if "bptt" in self.config.sampling_scheme:  # skewed toward n+k ~ max_recurrence
                n = torch.randint(low=0, high=t * 2, size=(1,), generator=n_generator)
                k = torch.randint(low=1, high=1 + min(t * 2 - int(n.item()), s * 2), size=(1,), generator=k_generator)
            elif "non-uniform" in self.config.sampling_scheme:  # n+k ~ uniform, n ~ 1
                n_plus_k = torch.randint(low=0, high=2 * t, size=(1,), generator=n_generator)
                k = torch.randint(low=1, high=2 * min(t, s) + 1, size=(1,), generator=k_generator)
                n = torch.clamp(n_plus_k - k, min=0)
            elif "gupta" in self.config.sampling_scheme:  # skewed toward n+k ~ uniform, k ~ 1
                # https://github.com/aks2203/deep-thinking/issues/10
                n = torch.randint(low=0, high=t * 2, size=(1,), generator=n_generator)
                draw = torch.rand(size=(1,), generator=k_generator)
                skew = torch.randint(low=2 * t, high=t * 8, size=(1,), generator=k_generator)
                k = 1 + (t - n) * draw**skew
            elif "simple" in self.config.sampling_scheme:  # me not make complicate? n + k ~trapezoidal
                n = torch.randint(low=0, high=2 * t, size=(1,), generator=n_generator)
                k = torch.randint(low=1, high=2 * s + 1, size=(1,), generator=k_generator)
            elif "poisson-lognormal-filling" in self.config.sampling_scheme:
                sigma = 0.5
                mu = math.log(t + s) - (sigma**2 / 2)
                rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=n_generator)
                p = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=n_generator) + 1
                n = torch.clamp(p - s, min=0)
                k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))
            elif "poisson-lognormal-fill" in self.config.sampling_scheme:
                sigma = 0.5
                mu = math.log(t) - (sigma**2 / 2)
                rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=n_generator)
                n = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=n_generator)
                k = torch.as_tensor(s)
            elif "poisson-lognormal" in self.config.sampling_scheme:
                sigma = 0.5
                mu = math.log(t) - (sigma**2 / 2)
                rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=n_generator)
                n = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=n_generator)
                k = torch.randint(1, 2 * s + 1, (1,), generator=k_generator)
            elif "poisson-unbounded" in self.config.sampling_scheme:
                n = torch.poisson(torch.tensor([t], dtype=torch.float), generator=n_generator)
                k = torch.randint(low=1, high=2 * s + 1, size=(1,), generator=k_generator)
            elif "poisson-fill" in self.config.sampling_scheme:
                n = torch.poisson(torch.tensor([t], dtype=torch.float), generator=n_generator)
                k = torch.as_tensor(s)
            elif "poisson-bounded" in self.config.sampling_scheme:
                n = torch.minimum(
                    torch.poisson(torch.tensor([t], dtype=torch.float), generator=n_generator),
                    torch.as_tensor(2 * t - 1),
                )
                k = torch.randint(low=1, high=2 * s + 1, size=(1,), generator=k_generator)
            elif "negative-binomial" in self.config.sampling_scheme:
                n = torch.as_tensor(sample_negative_binomial(2 * t, t))
                k = torch.randint(1, 2 * s + 1, (1,))
            elif "sobol" in self.config.sampling_scheme:  # this is sobol+simple
                nk_generator = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed_n)
                n_, k_ = nk_generator.draw(1).flatten()
                n = (n_ * 2 * t).to(torch.long)
                k = (k_ * 2 * s + 1).to(torch.long)
            elif "geometric" in self.config.sampling_scheme:
                n = torch.as_tensor(1.0).geometric_(1 / t, generator=n_generator)
                k = torch.randint(low=1, high=2 * s + 1, size=(1,), generator=k_generator)
            elif "fixed" in self.config.sampling_scheme:
                n, k = torch.as_tensor(t), torch.as_tensor(s)
            elif "non-recurrent" in self.config.sampling_scheme:
                n, k = torch.as_tensor(0), torch.as_tensor(1)
            elif "full" in self.config.sampling_scheme:
                n, k = torch.as_tensor(0), torch.randint(low=1, high=2 * s + 1, size=(1,), generator=k_generator)
        else:
            n, k = torch.as_tensor(self.config.mean_recurrence), torch.as_tensor(0)

        return n.to(dtype=torch.long), k.to(dtype=torch.long)

    def initialize_state(self, input_embeds):
        if self.config.injection_type == "none":
            return input_embeds
        if self.config.state_init == "normal":
            x = torch.randn_like(input_embeds)
        elif self.config.state_init == "embed":  # initialized like a scaled embedding:
            x = torch.randn_like(input_embeds).mul(1 / math.sqrt(input_embeds.shape[-1]))
        elif self.config.state_init == "like-init":
            x = torch.randn_like(input_embeds)
            std = self.config.init.get_std("embedding")
            torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
            if self.emb_scale != 1:
                x = x * self.emb_scale
        elif self.config.state_init == "zero":
            x = torch.zeros_like(input_embeds)
        elif self.config.state_init == "unit":
            x = torch.randn_like(input_embeds)
            std, mean = torch.std_mean(x, dim=-1, keepdim=True)
            x = (x - mean) / std
        return x

    @torch.no_grad()
    def monitor_module(
        self,
        x_out: torch.Tensor,
        x_rec: torch.Tensor,
        xk: torch.Tensor,
        input_embeds: torch.Tensor,
        num_steps_no_grad: torch.Tensor,
        num_steps_with_grad: torch.Tensor,
    ):
        """Should update to track more recurrence metrics"""
        x_out_c = x_out - x_out.mean(dim=-1, keepdim=True)
        normed_x = x_out_c / x_out_c.norm(dim=-1, keepdim=True)
        token_corr = (normed_x @ normed_x.transpose(1, 2)).mean() - 1 / x_out.shape[1]

        x_rec_c = x_rec - x_rec.mean(dim=-1, keepdim=True)
        normed_x = x_rec_c / x_rec_c.norm(dim=-1, keepdim=True)
        token_corr_rec = (normed_x @ normed_x.transpose(1, 2)).mean() - 1 / x_rec.shape[1]
        # k = num_steps_no_grad + num_steps_with_grad
        metrics = {
            "last_hidden_token_corr": token_corr,
            "recurrent_state_token_corr": token_corr_rec,
            "last_hidden_norm": x_out.norm(dim=-1).mean(),
            "recurrent_state_norm": x_rec.norm(dim=-1).mean(),
            "recurrent_diff": (x_rec - input_embeds).norm(dim=-1).mean(),
            "num_steps_no_grad": num_steps_no_grad,
            "num_steps_with_grad": num_steps_with_grad,
            "recurrent_residual": (x_rec - xk).norm(dim=-1).mean(),
            "rel_residual": ((x_rec - xk).norm(dim=-1) / x_rec.norm(dim=-1)).mean(),
            # f"rel_residual_at_{k}": ((x_rec - xk).norm(dim=-1) / x_rec.norm(dim=-1)).mean(),
        }
        self.latest_metrics = metrics  # will be picked up from monitoring caller


########################## Sanity check  Blocks #######################################


class Brick(torch.nn.Module):
    expanded = False

    def __init__(self, config: Config, layer_id: int) -> None:
        super().__init__()
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.layer_id = layer_id

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.mlp(x)


class BrickLP(torch.nn.Module):
    expanded = False

    def __init__(self, config: Config, layer_id: int = 0, in_features: int = 0) -> None:
        super().__init__()
        in_features = config.n_embd if in_features == 0 else in_features
        self.fc = torch.nn.Linear(in_features, in_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


######################### Utility functions ###########################################


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, condense_ratio: int = 1):
    with torch.autocast("cuda", enabled=False):
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(end, dtype=torch.float32, device=inv_freqs.device) / condense_ratio
        freqs = torch.outer(t, inv_freqs).float()
        return torch.stack([torch.cos(freqs)[None, :, None, :], torch.sin(freqs)[None, :, None, :]], dim=4)
        # equivalent to
        # freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        # cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)


# @torch.compile # pick this up from general compile call?
def apply_rotary_emb_complex_like(q: Tensor, k: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    with torch.autocast("cuda", enabled=False):
        # https://github.com/t-vi/lit-llama/blob/9e5eb8b1b376d8ae24e79278008b7190961062e3/lit_llama/model.py
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


def sample_gamma(alpha, beta):
    if alpha < 1:
        return sample_gamma(alpha + 1, beta) * torch.rand(1).pow(1 / alpha)

    d = alpha - 1 / 3
    c = 1 / math.sqrt(9 * d)

    while True:
        x = torch.randn(1)
        v = 1 + c * x
        v = v * v * v
        u = torch.rand(1)

        if u < 1 - 0.0331 * x * x * x * x:
            return d * v / beta

        if torch.log(u) < 0.5 * x * x + d * (1 - v + torch.log(v)):
            return d * v / beta


def sample_negative_binomial(t, target_mean):
    # Calculate p such that the mean of the negative binomial is target_mean
    p = target_mean / (t / 2 + target_mean)
    r = target_mean * p / (1 - p)
    beta = (1 - p) / p
    gamma_sample = sample_gamma(r, beta)
    poisson_sample = torch.poisson(gamma_sample)
    return poisson_sample
