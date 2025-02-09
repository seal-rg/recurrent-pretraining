"""Interface to various attention backends"""

import torch
from functools import partial
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.version

from .pytorch import attention_computation_sdpa
from .flex_attentions import (
    attention_computation_flex,
    attention_computation_flex_softcap,
    attention_computation_flex_docblock,
)
from .amd import attention as aotriton_attention
from .openai import attention as triton_tutorial_attention
from .triton_kernels_seq_par import attention as triton_kernels_attention
from .mosaic import flash_attn_func as mpt_7b_attention
from .cuda_flash_attention import attention_computation_flash
from .binBlk import binBlk_wrapper

"""Notes:

All interfaces standardize on the following shape
 q: (batch_size, seqlen_q, nheads, headdim)
 y: [output] (batch_size, seq_len, n_head, headdim)

but

openai, sdpa and amd will convert internally to BATCH, N_HEAD, N_CTX, HEAD_DIM internally, 
these implementations also handle the conversion back to BATCH N_CTX N_HEAD HEAD_DIM
"""


def _skip_attention(q, k, v, mask=None):
    """For debugging/benchmarking without attention computation"""
    return v.clone()


# @torch._dynamo.disable(recursive=True)
# def sdpa_hip_cannot_compile(q, k, v, mask=None):
#     """This is exiled up here so "provider" is not in <locals> preventing compilation around the attention call."""
#     return attention_computation_sdpa(q, k, v, mask)


def select_attention_implementation(
    provider="sdpa", center=False, debias=False
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    if (provider != "sdpa" or torch.version.hip) and center:
        raise ValueError("Centering not implemented for this provider.")
    if (provider != "sdpa" or torch.version.hip) and debias:
        raise ValueError("debias not implemented for this provider.")

    if provider == "sdpa":
        # if torch.version.hip:
        #     return sdpa_hip_cannot_compile
        # else:
        return partial(attention_computation_sdpa, center=center, debias=debias)
    elif provider == "amd":
        return partial(aotriton_attention, causal=True)
    elif provider == "openai":
        return partial(triton_tutorial_attention, causal=True, sm_scale=None)
    elif provider == "triton-kernels":
        return partial(triton_kernels_attention, causal=True, sm_scale=None)
    elif provider == "mosaic":
        return partial(mpt_7b_attention, bias=None, causal=True, softmax_scale=None)
    elif provider == "tridao":
        return partial(attention_computation_flash, causal=True)
    elif provider == "flex-attention":
        return partial(attention_computation_flex, center=center, debias=debias)
    elif provider == "flex-attention-doc-block":
        return partial(attention_computation_flex_docblock, center=center, debias=debias)
    elif provider == "flex-attention-soft-cap":
        return partial(attention_computation_flex_softcap, center=center, debias=debias)
    elif provider == "binBlk":  # agniv
        return partial(binBlk_wrapper, causal=True, dense=True)
    elif provider == "debug-skip":
        return _skip_attention
    else:
        raise ValueError(f"Attention implementation provider {provider} not registered.")
