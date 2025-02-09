import torch
import triton

import torch.nn.functional as F


import math

try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False


from amd import attention as aotriton_attention
from openai import attention as triton_tutorial_attention
from mosaic import flash_attn_func as mpt_7b_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

method_list = [
    "baseline",
    "sdpa-math",
    "sdpa-mem-efficient",
    "sdpa-flash",
    "amd",
    "mosaic",
    "openai",
]

if HAS_FLASH:
    method_list.append("flash-attn-repo")

BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=method_list,
            line_names=method_list,
            ylabel="ms",
            plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal=True",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "mode": mode,
                "causal": True,
            },
        )
    )


def baseline_attn(q, k, v, mask=None):
    _, _, seqlen, head_dim = q.shape
    # q = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    # k = k.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    # v = v.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
    else:
        attn_mask_tril = torch.ones([seqlen, seqlen], dtype=torch.bool, device=q.device).tril()
        attn_mask = torch.zeros_like(attn_mask_tril).to(q)
        attn_mask = attn_mask_tril.masked_fill(~attn_mask_tril, -10000)
        scores = scores + attn_mask[None, None, :, :]
    scores = F.softmax(scores.float(), dim=-1).type_as(q)
    output = torch.matmul(scores, v)  # (bs, n_local_heads, seqlen, head_dim)
    # output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return output


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device="cuda"):
    # H means "number of heads" here
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    dtype = torch.float16

    if provider == "baseline":
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        fn = lambda: baseline_attn(q, k, v, None)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)  # type: ignore
            fn = lambda: o.backward(do, retain_graph=True)  # type: ignore
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    if provider == "openai":
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        fn = lambda: triton_tutorial_attention(q, k, v, None, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)  # type: ignore
            fn = lambda: o.backward(do, retain_graph=True)  # type: ignore
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    if provider == "amd":
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        fn = lambda: aotriton_attention(q, k, v, mask=None, b=None, causal=causal, sm_scale=None)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)  # type: ignore
            fn = lambda: o.backward(do, retain_graph=True)  # type: ignore
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    if provider == "mosaic":
        q = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        fn = lambda: mpt_7b_attention(q, k, v, mask=None, bias=None, causal=causal, softmax_scale=None)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)  # type: ignore
            fn = lambda: o.backward(do, retain_graph=True)  # type: ignore
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    if provider == "flash-attn-repo":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)  # type: ignore
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    if "sdpa" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

        if "math" in provider:
            backend = SDPBackend.MATH
        elif "flash" in provider:
            backend = SDPBackend.FLASH_ATTENTION
        elif "mem-efficient":
            backend = SDPBackend.EFFICIENT_ATTENTION
        with sdpa_kernel(backend):
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )

        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)  # type: ignore
            fn = lambda: o.backward(do, retain_graph=True)  # type: ignore
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
