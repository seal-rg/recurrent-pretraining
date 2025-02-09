# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""

import triton
import triton.language as tl
import math

import torch
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.version


def is_hip():
    return torch.version.hip


@triton.jit
def max_fn(x, y):
    return tl.max(x, y)


@triton.jit
def dropout_offsets(philox_seed, philox_offset, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, m, n, stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep


@triton.jit
def attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    B_block_ptr,
    start_m,
    seqlen_q,
    q_padded,
    seqlen_k_low,
    seqlen_k_high,
    k_padded,
    dropout_p,
    dropout_seqlen_k,
    philox_seed,
    batch_philox_offset,
    encoded_softmax_block_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    pre_load_v: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    MARGINAL_BLOCK: tl.constexpr,  # MARGINAL_BLOCK = CAUSAL or k_padded
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    lo, hi = seqlen_k_low, seqlen_k_high
    if MARGINAL_BLOCK:
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, lo))
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (0, lo))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        # -- compute qk ----
        # MARGINAL_BLOCK serves as a compile-time switch for first attn_fwd_inner calls to "solid" blocks
        if MARGINAL_BLOCK and k_padded:
            if PADDED_HEAD:
                k = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")
            else:
                k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        else:
            if PADDED_HEAD:
                k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
            else:
                k = tl.load(K_block_ptr)
        if pre_load_v:
            if MARGINAL_BLOCK and k_padded:
                if PADDED_HEAD:
                    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
                else:
                    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            else:
                if PADDED_HEAD:
                    v = tl.load(V_block_ptr, boundary_check=(1,), padding_option="zero")
                else:
                    v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if MARGINAL_BLOCK:
            if CAUSAL:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])  # type: ignore
                qk = tl.where(mask, qk, float("-inf"))
            if k_padded:
                boundary_m = tl.full([BLOCK_M], seqlen_k_high, dtype=tl.int32)
                size_n = start_n + offs_n[None, :]  # type: ignore
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            if q_padded and k_padded:  # CAVEAT: using "or" disables the partial boundary_check branches
                bias = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option="zero")
            elif q_padded:
                bias = tl.load(B_block_ptr, boundary_check=(0,), padding_option="zero")
            elif k_padded:
                bias = tl.load(B_block_ptr, boundary_check=(1,), padding_option="zero")
            else:
                bias = tl.load(B_block_ptr)
            qk += bias * 1.44269504089
        else:
            tl.static_assert(False, f"Unsupported BIAS_TYPE {BIAS_TYPE}")
        qk += tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        # Note about the conflicts of Flash attention algorithm and PyTorch's CUDA implementation
        # PyTorch needs to return softmax(qk) (dropout mask encoded in sign bits)
        # While Flash attention paper computer the dropout AFTER exp2(qk- m_ij)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * dropout_seqlen_k + start_n
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, dropout_seqlen_k)
            if RETURN_ENCODED_SOFTMAX:
                tl.store(
                    encoded_softmax_block_ptr,
                    tl.where(keep, p, -p).to(encoded_softmax_block_ptr.type.element_ty),  # type: ignore
                    boundary_check=(0, 1),
                )
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_softmax_block_ptr, p.to(encoded_softmax_block_ptr.type.element_ty), boundary_check=(0, 1))  # type: ignore
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not pre_load_v:
            if MARGINAL_BLOCK and k_padded:
                if PADDED_HEAD:
                    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
                else:
                    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            else:
                if PADDED_HEAD:
                    v = tl.load(V_block_ptr, boundary_check=(1,), padding_option="zero")
                else:
                    v = tl.load(V_block_ptr)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p.to(V_block_ptr.type.element_ty), v)  # type: ignore
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, BLOCK_N))
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    B,
    sm_scale,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    seqlen_q,
    seqlen_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    encoded_softmax,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)  # head index
    off_z = tl.program_id(2)  # batch index
    num_h = tl.num_programs(1)
    q_padded = start_m * BLOCK_M + BLOCK_M > seqlen_q
    k_padded = True
    if seqlen_k < BLOCK_N:
        seqlen_k_faligned = 0  # floor aligned
    elif seqlen_k % BLOCK_N:
        extra_tokens_n = seqlen_k % BLOCK_N
        seqlen_k_faligned = seqlen_k - extra_tokens_n
    else:
        k_padded = False
        seqlen_k_faligned = seqlen_k

    q_offset = off_h * stride_qh + off_z * stride_qz
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    k_offset = off_h * stride_kh + off_z * stride_kz
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    v_offset = off_h * stride_vh + off_z * stride_vz
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, head_dim),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504089
    # load q: it will stay in SRAM throughout on NV GPUs but in VGPRs on AMD GPUs
    if q_padded:
        if PADDED_HEAD:
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    else:
        q = tl.load(Q_block_ptr, boundary_check=(1,), padding_option="zero") if PADDED_HEAD else tl.load(Q_block_ptr)
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)  # type: ignore
    # stage 1: off-band
    # For causal = True, STAGE = 3 and attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and attn_fwd_inner gets 3 as its STAGE
    off_zh = off_z * num_h + off_h * 1
    batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k if ENABLE_DROPOUT else 0
    if BIAS_TYPE == 0:
        B_block_ptr = 0
    elif BIAS_TYPE == 1:
        B_block_ptr = tl.make_block_ptr(
            base=B + off_h * stride_bh + off_z * stride_bz,
            shape=(seqlen_q, seqlen_k),
            strides=(stride_bm, stride_bn),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        tl.static_assert(False, f"Unsupported BIAS_TYPE {BIAS_TYPE}")

    if RETURN_ENCODED_SOFTMAX:
        encoded_softmax_block_ptr = tl.make_block_ptr(
            base=encoded_softmax + off_zh * seqlen_q * seqlen_k,
            shape=(seqlen_q, seqlen_k),
            strides=(seqlen_k, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        encoded_softmax_block_ptr = 0

    # Stage 1: off-band (for causal) or non-boundary (for irregular seqlen_k) blocks
    if CAUSAL:
        # Causal = True
        seqlen_k_low = 0
        seqlen_k_high = min(seqlen_k_faligned, start_m * BLOCK_M)
    else:
        # Causal = False
        seqlen_k_low = 0
        seqlen_k_high = seqlen_k_faligned
    acc, l_i, m_i = attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,
        B_block_ptr,
        start_m,
        seqlen_q,
        q_padded,
        seqlen_k_low,
        seqlen_k_high,
        False,
        dropout_p,
        seqlen_k,
        philox_seed,
        batch_philox_offset,
        encoded_softmax_block_ptr,
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_N,
        False,
        offs_m,
        offs_n,
        pre_load_v,
        ENABLE_DROPOUT,
        RETURN_ENCODED_SOFTMAX,
        MARGINAL_BLOCK=False,
        PADDED_HEAD=PADDED_HEAD,
        BIAS_TYPE=BIAS_TYPE,
    )  # type: ignore
    # Stage 2: on-band or boundary blocks
    if CAUSAL or k_padded:
        seqlen_k_low = seqlen_k_high
        if CAUSAL:  # noqa: SIM108
            seqlen_k_high = min(seqlen_k, start_m * BLOCK_M + BLOCK_M)  # type: ignore
        else:
            seqlen_k_high = seqlen_k
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            B_block_ptr,
            start_m,
            seqlen_q,
            q_padded,
            seqlen_k_low,
            seqlen_k_high,
            k_padded,
            dropout_p,
            seqlen_k,
            philox_seed,
            batch_philox_offset,
            encoded_softmax_block_ptr,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            CAUSAL,
            offs_m,
            offs_n,
            pre_load_v,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX,
            MARGINAL_BLOCK=True,
            PADDED_HEAD=PADDED_HEAD,
            BIAS_TYPE=BIAS_TYPE,
        )  # type: ignore
    # epilogue
    # write back m
    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    m_ptrs = M + off_zh * seqlen_q + offs_m
    # Check for last block_M
    if q_padded:
        overflow_size = (start_m * BLOCK_M + BLOCK_M) - seqlen_q
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        # This is a > check because mask being 0 blocks the store.
        m_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(m_ptrs, m_i + tl.math.log2(l_i), mask=m_ptrs_mask)
    else:
        tl.store(m_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    o_offset = off_h * stride_oh + off_z * stride_oz
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    if q_padded:
        if PADDED_HEAD:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))
        else:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,))
    else:
        if PADDED_HEAD:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(1,))
        else:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_don,
    seqlen_q,
    head_dim,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
):
    # off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    # off_n = tl.arange(0, D_HEAD)
    off_m = tl.program_id(0) * BLOCK_M
    off_h = tl.program_id(1)  # head index
    off_z = tl.program_id(2)  # batch index
    num_h = tl.num_programs(1)
    o_offset = off_h * stride_oh + off_z * stride_oz
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_om, stride_on),
        offsets=(off_m, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0),
    )
    do_offset = off_h * stride_doh + off_z * stride_doz
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_dom, stride_don),
        offsets=(off_m, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0),
    )
    # load
    # o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    o = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back, shape (q.shape[0] * q.shape[1], q.shape[2])
    off_zh = off_z * num_h + off_h * 1
    # Check for OOB accesses
    delta_ptrs = Delta + off_zh * seqlen_q + off_m + tl.arange(0, BLOCK_M)
    overflow = off_m + BLOCK_M - seqlen_q
    if overflow > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow, dtype=tl.int32)
        mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(delta_ptrs, delta, mask=mask)
    else:
        tl.store(delta_ptrs, delta)


# Helper function, but not always usable due to compiler bugs (esp. used with tl.trans)
@triton.jit
def dot(BLOCK_M: tl.constexpr, QDIM: tl.constexpr, KDIM: tl.constexpr, q, k):
    if BLOCK_M == 1:
        return tl.sum(tl.view(q, [QDIM]) * tl.view(k, [KDIM]))
    else:
        return tl.dot(q, k)


# TODO: Remove Unused 'Out' Argument from kernels below
@triton.jit
def bwd_kernel_dk_dv(
    Q,
    K,
    V,
    B,
    sm_scale,
    Out,
    DO,
    DK,
    DV,
    L,
    D,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_dkz,
    stride_dkh,
    stride_dkn,
    stride_dkk,
    stride_dvz,
    stride_dvh,
    stride_dvk,
    stride_dvn,
    max_seqlens_q,
    max_seqlens_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_N
    off_h = tl.program_id(1)  # head index
    off_z = tl.program_id(2)  # batch index
    num_h = tl.num_programs(1)

    # TODO: Support varlen here
    seqlen_q = max_seqlens_q
    seqlen_k = max_seqlens_k
    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_M)
    # Initialize pointers to Q, K, V
    # Q is consumed depending on block ID. Every block uses
    # previous block offset by BLOCK_M x D_HEAD.
    q_offset = off_h * stride_qh + off_z * stride_qz
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    k_offset = off_h * stride_kh + off_z * stride_kz
    KT_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )

    v_offset = off_h * stride_vh + off_z * stride_vz
    VT_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    do_offset = off_h * stride_oh + off_z * stride_oz
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_om, stride_ok),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    off_zh = off_z * num_h + off_h * 1
    if BIAS_TYPE == 0:
        B_block_ptr = 0
    elif BIAS_TYPE == 1:
        B_block_ptr = tl.make_block_ptr(
            base=B + off_h * stride_bh + off_z * stride_bz,
            shape=(seqlen_q, seqlen_k),
            strides=(stride_bm, stride_bn),
            offsets=(0, start_m),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        tl.static_assert(False, f"Unsupported BIAS_TYPE {BIAS_TYPE}")
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_zh * seqlen_q
    l_ptrs = L + off_zh * seqlen_q
    qk_scale = sm_scale * 1.44269504089
    # load k and v: they will stay in SRAM throughout
    # (BLOCK_DMODEL, BLOCK_N)
    if PADDED_HEAD:
        kt = tl.load(KT_block_ptr, boundary_check=(1, 0), padding_option="zero")
    else:
        kt = tl.load(KT_block_ptr, boundary_check=(1,), padding_option="zero")
    kt = (kt * qk_scale).to(KT_block_ptr.type.element_ty)  # type: ignore
    # (BLOCK_DMODEL, BLOCK_N)
    if PADDED_HEAD:
        vt = tl.load(VT_block_ptr, boundary_check=(1, 0), padding_option="zero")
    else:
        vt = tl.load(VT_block_ptr, boundary_check=(1,), padding_option="zero")
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # This lower loop bound is because of the causal mask. We create a lower triangular
    # result. The upper triangular is -inf (becomes 0 when we do e^x). As such, it can
    # be ignored in the GEMM.
    lo = (start_m // BLOCK_M) * BLOCK_M if CAUSAL else 0
    hi = seqlen_q
    Q_block_ptr = tl.advance(Q_block_ptr, (lo, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo, 0))
    batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k
    if BIAS_TYPE == 1:
        B_block_ptr = tl.advance(B_block_ptr, (lo, 0))
    """
           K1   K2      (d)V      dO
    Q1    qk11 qk12     (d)v1     dO1
    Q2    qk21 qk22     (d)v2     dO2

    QK: (seqlen_q, seqlen_k)
    dO: (seqlen_q, hdim)
    dV: (seqlen_k, hdim)

    dV = (QK)^T dO

    dV1 = qk11 dO1 + qk21 dO2 = q1 k1 dO1 + q2 k1 dO2
    dV2 = qk12 dO1 + qk22 dO2 = q1 k2 dO1 + q2 k2 dO2
                                ~~~~~ = 0
    start_m: select k and dV
    start_n: select q and dO
    """
    # loop over q (seqlen_q, dhead), do (seqlen_q, d_head)
    for start_n in range(lo, hi, BLOCK_M):  # type: ignore
        offs_m_curr = offs_n[:, None] + start_n  # (BLOCK_M, 1)
        # -- load q, do --
        # TODO: It is more optimal to do OOB check only in the last iter.
        # (BLOCK_M, BLOCK_DMODEL), offs = (BLOCK_M * iter, 0) = (start_n, 0)
        if PADDED_HEAD:
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        # do: (BLOCK_M, BLOCK_DMODEL)
        if PADDED_HEAD:
            do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        else:
            do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option="zero")
        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # TODO: These two checks can be optimized to occur on the last iter.
        overflow_size = start_n + BLOCK_M - seqlen_q
        if overflow_size > 0:
            boundary_n = tl.full((BLOCK_N,), seqlen_q, dtype=tl.int32)
            mask = offs_m_curr < boundary_n[None, :]
            qk = tl.where(mask, qk, float("-inf"))
        if CAUSAL:
            qk = tl.where(offs_m_curr >= offs_m[None, :], qk, float("-inf"))
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            # FIXME: do boundary_check correctly
            """
            if q_padded and k_padded:  # CAVEAT: using "or" disables the partial boundary_check branches
                bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            elif q_padded:
                bias = tl.load(B_block_ptr, boundary_check=(0,), padding_option="zero")
            elif k_padded:
                bias = tl.load(B_block_ptr, boundary_check=(1,), padding_option="zero")
            else:
                bias = tl.load(B_block_ptr)
            """
            bias = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option="zero")
            qk += bias * 1.44269504089
        else:
            tl.static_assert(False, f"Unsupported BIAS_TYPE {BIAS_TYPE}")
        # q.offs = (start_n, 0), k.offs = (0, start_m)
        qk += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, q, kt)  # (BLOCK_M, BLOCK_N)
        # Check for OOB accesses on D and LSE
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        d_lse_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        d_lse_padding = tl.full((BLOCK_M,), 0, dtype=tl.float32)
        Di = tl.load(D_ptrs + offs_m_curr, mask=d_lse_ptrs_mask[:, None], other=d_lse_padding[:, None])
        l_i = tl.load(l_ptrs + offs_m_curr, mask=d_lse_ptrs_mask[:, None], other=d_lse_padding[:, None])
        p = tl.math.exp2(qk - l_i)  # (BLOCK_M, BLOCK_N)
        # -- compute dv ----
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_n * seqlen_k + start_m
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
            # CAVEAT: do NOT update p, ds needs the original p
            if BLOCK_M == 1:
                dv += tl.where(keep, p / (1 - dropout_p), 0.0).to(Q.dtype.element_ty) * do
            else:
                dv += tl.dot(tl.trans(tl.where(keep, p / (1 - dropout_p), 0.0)).to(Q.dtype.element_ty), do)
        else:
            if BLOCK_M == 1:
                dv += p.to(Q.dtype.element_ty) * do
            else:
                # dv += tl.dot(tl.trans(p.to(do.dtype)), do)
                dv += tl.dot(tl.trans(p).to(do.dtype), do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # compute dp = dot(do, vt)
        # dp += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, do, vt)
        # do.shape = (BLOCK_M, BLOCK_DMODEL) vt.shape = (BLOCK_DMODEL, BLOCK_N)
        dp += tl.dot(do, vt)
        if ENABLE_DROPOUT:
            dp = tl.where(keep, dp / (1 - dropout_p), 0)  # type: ignore
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di)  # (BLOCK_M, BLOCK_N)
        # compute dk
        if BLOCK_M == 1:
            dk += ds.to(Q.dtype.element_ty) * q
        else:
            # ds.shape = (BLOCK_M, BLOCK_N), q.shape = (BLOCK_M, BLOCK_DMODEL)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)  # (BLOCK_N, BLOCK_DMODEL)
        # update pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))  # Debug DO accessing problems
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (BLOCK_M, 0))
    # initialize pointers to output
    dk_offset = off_h * stride_dkh + off_z * stride_dkz
    DK_block_ptr = tl.make_block_ptr(
        base=DK + dk_offset,
        shape=(seqlen_k, head_dim),
        strides=(stride_dkn, stride_dkk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    dv_offset = off_h * stride_dvh + off_z * stride_dvz
    DV_block_ptr = tl.make_block_ptr(
        base=DV + dv_offset,
        shape=(seqlen_k, head_dim),
        strides=(stride_dvk, stride_dvn),
        offsets=(start_m, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(DK_block_ptr, (dk * sm_scale).to(DK.type.element_ty), boundary_check=(0, 1))
    tl.store(DV_block_ptr, dv.to(DV.type.element_ty), boundary_check=(0, 1))


@triton.jit
def bwd_kernel_dq(
    Q,
    K,
    V,
    B,
    sm_scale,
    Out,
    DO,
    DQ,
    DB,
    L,
    D,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    stride_dqk,
    stride_dbz,
    stride_dbh,
    stride_dbm,
    stride_dbn,
    max_seqlens_q,
    max_seqlens_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_h = tl.program_id(1)  # head index
    off_z = tl.program_id(2)  # batch index
    num_h = tl.num_programs(1)

    # TODO: Support varlen here
    seqlen_q = max_seqlens_q
    seqlen_k = max_seqlens_k
    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Initialize pointers to Q, K, V
    q_offset = off_h * stride_qh + off_z * stride_qz
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    k_offset = off_h * stride_kh + off_z * stride_kz
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    v_offset = off_h * stride_vh + off_z * stride_vz
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    do_offset = off_h * stride_oh + off_z * stride_oz
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_om, stride_ok),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    off_zh = off_z * num_h + off_h * 1
    if BIAS_TYPE == 0:
        B_block_ptr = 0
        DB_block_ptr = 0
    elif BIAS_TYPE == 1:
        B_block_ptr = tl.make_block_ptr(
            base=B + off_h * stride_bh + off_z * stride_bz,
            shape=(seqlen_q, seqlen_k),
            strides=(stride_bm, stride_bn),
            offsets=(start_m, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        store_db = not ((stride_dbz == 0 and stride_dbh == 0) and stride_dbm == 0)
        # Still have to make one even if no_db = False
        # due to a limit of Triton: runtime branches must have identical data types.
        DB_block_ptr = tl.make_block_ptr(
            base=DB + off_h * stride_dbh + off_z * stride_dbz,
            shape=(seqlen_q, seqlen_k),
            strides=(stride_dbm, stride_dbn),
            offsets=(start_m, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        tl.static_assert(False, f"Unsupported BIAS_TYPE {BIAS_TYPE}")
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_zh * seqlen_q
    l_ptrs = L + off_zh * seqlen_q
    qk_scale = sm_scale * 1.44269504089
    # load q and do: they will stay in SRAM throughout
    if PADDED_HEAD:
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)  # type: ignore
    if PADDED_HEAD:
        do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    else:
        do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option="zero")
    # Check for OOB accesses on D and LSE
    overflow_size_q = start_m + BLOCK_M - seqlen_q
    boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size_q, dtype=tl.int32)
    d_lse_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
    d_lse_padding = tl.full((BLOCK_M,), 0, dtype=tl.float32)
    Di = tl.load(D_ptrs + offs_m, mask=d_lse_ptrs_mask, other=d_lse_padding)
    l_i = tl.load(l_ptrs + offs_m, mask=d_lse_ptrs_mask, other=d_lse_padding)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # loop over k, v
    lo = 0
    hi = min(start_m + BLOCK_M, seqlen_k) if CAUSAL else seqlen_k  # type: ignore
    batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k
    """
           K1   K2      (d)V      dO
    Q1    qk11 qk12     (d)v1     dO1
    Q2    qk21 qk22     (d)v2     dO2

    QK: (seqlen_q, seqlen_k)
    dO: (seqlen_q, hdim)
    dV: (seqlen_k, hdim)
    """
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        # shape = (BLOCK_DMODEL, BLOCK_N), offs = (0, BLOCK_N * iter) = (0, start_n)
        if PADDED_HEAD:
            kt = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")
            vt = tl.load(V_block_ptr, boundary_check=(1, 0), padding_option="zero")
        else:
            kt = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
            vt = tl.load(V_block_ptr, boundary_check=(1,), padding_option="zero")
        # -- compute qk ----
        # q.offs = (start_m, 0), k.offs = (0, start_n)
        qk = dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, q, kt)
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_n), qk, float("-inf"))
        boundary_n = tl.full((BLOCK_M,), seqlen_k, dtype=tl.int32)
        size_n = start_n + tl.arange(0, BLOCK_N)
        mask = size_n[None, :] < boundary_n[:, None]
        qk = tl.where(mask, qk, float("-inf"))
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            """
            if q_padded and k_padded:  # CAVEAT: using "or" disables the partial boundary_check branches
                bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            elif q_padded:
                bias = tl.load(B_block_ptr, boundary_check=(0,), padding_option="zero")
            elif k_padded:
                bias = tl.load(B_block_ptr, boundary_check=(1,), padding_option="zero")
            else:
                bias = tl.load(B_block_ptr)
            """
            # FIXME: Must use boundary_check uncondtionally.
            # The optimized tl.load above causes nan for some reason
            bias = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option="zero")
            qk += bias * 1.44269504089
        else:
            tl.static_assert(False, f"Unsupported BIAS_TYPE {BIAS_TYPE}")
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, do, vt)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * seqlen_k + start_n
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
            dp = tl.where(keep, dp / (1 - dropout_p), 0)
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di[:, None])
        # compute dq. Unfortunately we cannot avoid transpose here as this loop
        # uses k both normal and transpose.
        if BLOCK_M == 1:
            dq += tl.view(kt, [BLOCK_DMODEL]) * ds.to(Q.type.element_ty)
        else:
            # ds.shape = (BLOCK_M, BLOCK_N), kt.shape = (BLOCK_DMODEL, BLOCK_N)
            dq += tl.dot(ds.to(Q.type.element_ty), tl.trans(kt))  # (BLOCK_M, BLOCK_DMODEL)
        if BIAS_TYPE == 1:  # noqa: SIM102
            if store_db:
                tl.store(DB_block_ptr, ds.to(DB.type.element_ty), boundary_check=(0, 1))
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_N))
            DB_block_ptr = tl.advance(DB_block_ptr, (0, BLOCK_N))
    # initialize pointers to output
    dq_offset = off_h * stride_dqh + off_z * stride_dqz
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + dq_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_dqm, stride_dqk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(DQ_block_ptr, (dq * sm_scale).to(DQ_block_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def debug_fill_dropout_rng(
    R,
    stride_rz,
    stride_rh,
    stride_rm,
    stride_rn,
    seqlen_q,
    seqlen_k,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)  # head index
    off_z = tl.program_id(2)  # batch index
    d_offset = off_h * stride_rh + off_z * stride_rz
    num_h = tl.num_programs(1)
    off_zh = off_z * num_h + off_h * 1
    batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k
    R_block_ptr = tl.make_block_ptr(
        base=R + d_offset,
        shape=(seqlen_q, seqlen_k),
        strides=(stride_rm, stride_rn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    for start_n in range(0, seqlen_k, BLOCK_N):
        philox_offset = batch_philox_offset + start_m * BLOCK_M * seqlen_k + start_n
        rng = dropout_rng(philox_seed, philox_offset, BLOCK_M, BLOCK_N, seqlen_k)
        tl.store(R_block_ptr, rng.to(R_block_ptr.type.element_ty), boundary_check=(0, 1))  # type: ignore
        R_block_ptr = tl.advance(R_block_ptr, (0, BLOCK_N))


VERBOSE = False
DEFAULT_PHILOX_SEED = 0x1BF52
DEFAULT_PHILOX_OFFSET = 0x1D4B42


def is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0


def is_supported_by_tl_dot(n: int) -> bool:
    return is_power_of_two(n) and n >= 16


if is_hip():
    TRITON_CONFIG_LIST_FWD = [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 0, "pre_load_v": True}, num_stages=1, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 1, "pre_load_v": True}, num_stages=1, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 2, "pre_load_v": True}, num_stages=1, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 3, "pre_load_v": True}, num_stages=1, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 4, "pre_load_v": True}, num_stages=1, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 0, "pre_load_v": False}, num_stages=1, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 1, "pre_load_v": False}, num_stages=1, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 2, "pre_load_v": False}, num_stages=1, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 3, "pre_load_v": False}, num_stages=1, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 4, "pre_load_v": False}, num_stages=1, num_warps=4
        ),
    ]
else:
    TRITON_CONFIG_LIST_FWD = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "pre_load_v": True}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "pre_load_v": False}, num_stages=1, num_warps=4),
    ]

"""
# For faster debugging of backward autotune
TRITON_CONFIG_LIST_FWD = [
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'pre_load_v': True}, num_stages=1, num_warps=4),
   ]
"""


@triton.autotune(
    configs=TRITON_CONFIG_LIST_FWD,
    key=["seqlen_q", "seqlen_k", "CAUSAL"],
)
@triton.jit
def tuned_attn_fwd(
    Q,
    K,
    V,
    B,
    sm_scale,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    seqlen_q,
    seqlen_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    encoded_softmax,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    attn_fwd(
        Q,
        K,
        V,
        B,
        sm_scale,
        M,
        Out,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_bz,
        stride_bh,
        stride_bm,
        stride_bn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        seqlen_q,
        seqlen_k,
        head_dim,
        dropout_p,
        philox_seed,
        philox_offset_base,
        encoded_softmax,
        CAUSAL,
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_N,
        pre_load_v,
        ENABLE_DROPOUT,
        RETURN_ENCODED_SOFTMAX,
        PADDED_HEAD,
        BIAS_TYPE=BIAS_TYPE,
    )


if is_hip():
    TRITON_CONFIG_LIST_BWD_SIZED = [
        triton.Config({"waves_per_eu": 0}, num_stages=1, num_warps=4),
        triton.Config({"waves_per_eu": 1}, num_stages=1, num_warps=4),
        triton.Config({"waves_per_eu": 2}, num_stages=1, num_warps=4),
        triton.Config({"waves_per_eu": 3}, num_stages=1, num_warps=4),
        triton.Config({"waves_per_eu": 4}, num_stages=1, num_warps=4),
    ]
else:
    TRITON_CONFIG_LIST_BWD_SIZED = [
        triton.Config({}, num_stages=1, num_warps=4),
    ]


@triton.autotune(
    configs=TRITON_CONFIG_LIST_BWD_SIZED,
    key=["max_seqlens_q", "max_seqlens_k"],
)
@triton.jit
def sized_tuned_bwd_kernel_dk_dv(
    Q,
    K,
    V,
    B,
    sm_scale,
    Out,
    DO,
    DK,
    DV,
    L,
    D,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_dkz,
    stride_dkh,
    stride_dkn,
    stride_dkk,
    stride_dvz,
    stride_dvh,
    stride_dvk,
    stride_dvn,
    max_seqlens_q,
    max_seqlens_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    bwd_kernel_dk_dv(
        Q,
        K,
        V,
        B,
        sm_scale,
        Out,
        DO,
        DK,
        DV,
        L,
        D,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_bz,
        stride_bh,
        stride_bm,
        stride_bn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_dkz,
        stride_dkh,
        stride_dkn,
        stride_dkk,
        stride_dvz,
        stride_dvh,
        stride_dvk,
        stride_dvn,
        max_seqlens_q,
        max_seqlens_k,
        head_dim,
        dropout_p,
        philox_seed,
        philox_offset_base,
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_N,
        CAUSAL,
        ENABLE_DROPOUT,
        PADDED_HEAD=PADDED_HEAD,
        BIAS_TYPE=BIAS_TYPE,
    )


@triton.autotune(
    configs=TRITON_CONFIG_LIST_BWD_SIZED,
    key=["max_seqlens_q", "max_seqlens_k"],
)
@triton.jit
def sized_tuned_bwd_kernel_dq(
    Q,
    K,
    V,
    B,
    sm_scale,
    Out,
    DO,
    DQ,
    DB,
    L,
    D,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    stride_dqk,
    stride_dbz,
    stride_dbh,
    stride_dbm,
    stride_dbn,
    max_seqlens_q,
    max_seqlens_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    bwd_kernel_dq(
        Q,
        K,
        V,
        B,
        sm_scale,
        Out,
        DO,
        DQ,
        DB,
        L,
        D,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_bz,
        stride_bh,
        stride_bm,
        stride_bn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_dqz,
        stride_dqh,
        stride_dqm,
        stride_dqk,
        stride_dbz,
        stride_dbh,
        stride_dbm,
        stride_dbn,
        max_seqlens_q,
        max_seqlens_k,
        head_dim,
        dropout_p,
        philox_seed,
        philox_offset_base,
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_N,
        CAUSAL,
        ENABLE_DROPOUT,
        PADDED_HEAD=PADDED_HEAD,
        BIAS_TYPE=BIAS_TYPE,
    )


class _attention(torch.autograd.Function):
    # DEBUG_MASK_DTYPE = torch.int32
    # DEBUG_MASK_DTYPE = torch.float32

    @staticmethod
    def forward(ctx, q, k, v, mask=None, causal=True):
        sm_scale = 1 / math.sqrt(q.shape[-1])
        dropout_p = 0.0
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        head_dim_rounded = 2 ** (Lk - 1).bit_length()
        head_dim_rounded = max(16, head_dim_rounded)
        padded_head = head_dim_rounded != Lk
        o = torch.zeros_like(q)

        grid = lambda META: (
            triton.cdiv(q.shape[2], META["BLOCK_M"]),
            q.shape[1],
            q.shape[0],
        )
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        encoded_softmax = None

        philox_seed = DEFAULT_PHILOX_SEED
        philox_offset = DEFAULT_PHILOX_OFFSET

        b = torch.empty((0, 0, 0, 0), device=q.device, dtype=q.dtype)
        BIAS_TYPE = 0

        # assert False, "No time to test autotune for now"
        tuned_attn_fwd[grid](
            q,
            k,
            v,
            b,
            sm_scale,
            M,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            b.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            seqlen_q=q.shape[2],
            seqlen_k=k.shape[2],
            head_dim=Lk,
            dropout_p=dropout_p,
            philox_seed=philox_seed,
            philox_offset_base=philox_offset,
            encoded_softmax=encoded_softmax,
            CAUSAL=causal,
            BLOCK_DMODEL=head_dim_rounded,
            ENABLE_DROPOUT=dropout_p > 0.0,
            RETURN_ENCODED_SOFTMAX=encoded_softmax is not None,
            PADDED_HEAD=padded_head,
            BIAS_TYPE=BIAS_TYPE,
        )

        ## restore the grid for bwd kernel
        try:
            best_config = tuned_attn_fwd.best_config  # type: ignore
            block_m = int(best_config.kwargs["BLOCK_M"])
        except Exception:
            block_m = min(128, q.shape[2], k.shape[2])
        grid = (triton.cdiv(q.shape[2], block_m), q.shape[1], q.shape[0])
        ctx.save_for_backward(q, k, v, b, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.head_dim = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.bias_type = BIAS_TYPE

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, b, o, L = ctx.saved_tensors
        # if q.shape[-1] <= 32:
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv and Lk == ctx.head_dim
        head_dim_rounded = 2 ** (ctx.head_dim - 1).bit_length()
        head_dim_rounded = max(16, head_dim_rounded)
        padded_head = head_dim_rounded != ctx.head_dim

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b)
        delta = torch.empty_like(L)
        max_seqlens_q = q.shape[2]
        max_seqlens_k = k.shape[2]
        # BLOCK = min(max_seqlens_q, max_seqlens_k, q.shape[-1], MAX_BLOCK)
        # BLOCK = BLOCK if is_supported_by_tl_dot(max_seqlens_q) and is_supported_by_tl_dot(max_seqlens_k) else 1
        if not ctx.autotune:  # noqa: SIM108
            BLOCK = 16  # FIXME: Variable block size
        else:
            BLOCK = 128

        grid_prep = (triton.cdiv(do.shape[2], BLOCK), do.shape[1], do.shape[0])
        bwd_preprocess[grid_prep](
            o,
            do,
            delta,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            max_seqlens_q,
            Lk,
            BLOCK_M=BLOCK,  # type: ignore
            D_HEAD=head_dim_rounded,  # type: ignore
            PADDED_HEAD=padded_head,  # FIXME: irregular head dimension
        )

        use_small_block = ctx.dropout_p > 0.0
        use_medium_block = ctx.bias_type != 0
        if use_small_block:
            # DQ_BLOCK_M = min(max_seqlens_q, BLOCK)
            BLOCK_M = 32
            BLOCK_N = 16
        elif use_medium_block:
            BLOCK_M = 64
            BLOCK_N = 32
        else:
            BLOCK_M = 128
            BLOCK_N = 64
        if q.dtype == torch.float32:
            BLOCK_M = max(16, BLOCK_M // 2)
            BLOCK_N = max(16, BLOCK_N // 2)
        # debug_mask = torch.zeros((q.shape[0], q.shape[1], max_seqlens_q, max_seqlens_k), device=q.device, dtype=ctx.encoded_softmax.dtype)
        grid_dk_dv = lambda META: (
            triton.cdiv(max_seqlens_k, META["BLOCK_N"]),
            q.shape[1],
            q.shape[0],
        )
        stride_dbz, stride_dbh, stride_dbm, stride_dbn = db.stride()
        if db.numel() == 0 or not b.requires_grad:
            # Passing all zeros to indicate no elements
            stride_dbz, stride_dbh, stride_dbm, stride_dbn = 0, 0, 0, 0
        else:
            db.fill_(float("nan"))

        if k.requires_grad and v.requires_grad:
            sized_tuned_bwd_kernel_dk_dv[grid_dk_dv](
                q,
                k,
                v,
                b,
                ctx.sm_scale,
                o,
                do,
                dk,
                dv,
                L,
                delta,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                b.stride(0),
                b.stride(1),
                b.stride(2),
                b.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                dk.stride(0),
                dk.stride(1),
                dk.stride(2),
                dk.stride(3),
                dv.stride(0),
                dv.stride(1),
                dv.stride(2),
                dv.stride(3),
                max_seqlens_q=max_seqlens_q,
                max_seqlens_k=max_seqlens_k,
                head_dim=Lk,
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset_base=ctx.philox_offset,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=head_dim_rounded,
                CAUSAL=ctx.causal,
                ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                PADDED_HEAD=padded_head,
                BIAS_TYPE=ctx.bias_type,
            )

        grid_dq = lambda META: (
            triton.cdiv(max_seqlens_q, META["BLOCK_M"]),
            q.shape[1],
            q.shape[0],
        )
        if q.requires_grad:
            sized_tuned_bwd_kernel_dq[grid_dq](
                q,
                k,
                v,
                b,
                ctx.sm_scale,
                o,
                do,
                dq,
                db,
                L,
                delta,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                b.stride(0),
                b.stride(1),
                b.stride(2),
                b.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                dq.stride(0),
                dq.stride(1),
                dq.stride(2),
                dq.stride(3),
                stride_dbz,
                stride_dbh,
                stride_dbm,
                stride_dbn,
                max_seqlens_q=max_seqlens_q,
                max_seqlens_k=max_seqlens_k,
                head_dim=Lk,
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset_base=ctx.philox_offset,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=head_dim_rounded,
                CAUSAL=ctx.causal,
                ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                PADDED_HEAD=padded_head,
                BIAS_TYPE=ctx.bias_type,
            )

        # print(h.asm["ttgir"])
        return dq, dk, dv, None if db.numel() == 0 else db, None, None, None, None, None, None, None, None


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None, causal: bool = True
) -> torch.Tensor:
    q = q.transpose(1, 2)  # (B, nh, S, hs)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    return _attention.apply(q, k, v, mask, causal).transpose(1, 2)  # type: ignore
