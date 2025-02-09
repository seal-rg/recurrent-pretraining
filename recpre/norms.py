# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# -----------------------------
# FusedRMSNorm in Triton
# Credit
# Tri Dao's Triton LayerNorm: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py
# Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html


import torch
import torch.nn.functional as F
from torch.nn import Identity  # noqa # can be loaded as "none" option

import math
import triton
import triton.language as tl


class TorchRMSNorm(torch.nn.RMSNorm):
    # higher mem footprint than compiled RMS
    def __init__(self, size: int, eps=1e-6) -> None:
        super().__init__(normalized_shape=size, eps=eps)


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        x_normed = x_normed.to(dtype=dtype)
        if self.add_unit_offset:
            # Gemma model requires a unit offset
            # https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L176
            return x_normed * (1 + self.weight)
        return x_normed * self.weight

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


class RMSNorm_llama(torch.nn.Module):
    """Saner dtype handling and slightly better for fusion"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        with torch.autocast(enabled=False, device_type=x.device.type):
            return self._norm(x.float()).type_as(x) * self.weight

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


class GemmalikeRMS(torch.nn.Module):
    """Gemma-like handling of the cast (but not the +1 offset)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=torch.float))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self._norm(x.float()) * self.weight).type_as(x)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


class UnparametrizedRMSNorm(torch.nn.Module):
    """Just a norm"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        with torch.autocast(enabled=False, device_type=x.device.type):
            return self._norm(x.float()).type_as(x)


class OlmoLayerNorm(torch.nn.Module):
    """LayerNorm but with no learnable weight or bias."""

    def __init__(self, hidden_size: int, eps=1e-8) -> None:
        super().__init__()
        self.normalized_shape = (hidden_size,)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        return F.layer_norm(hidden_states.to(dtype=torch.float32), self.normalized_shape, None, None, eps=self.eps).to(
            orig_dtype
        )


class FusedRMSNorm(torch.nn.Module):
    """Fused RMS Norm, wraps a fused Triton Kernel"""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.fused_rms_norm_fn = fused_rms_norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """leverages Triton Fused RMS Norm kernel"""
        return self.fused_rms_norm_fn(x, self.weight, eps=self.eps)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N"],
)
@triton.jit
def _rms_norm_fwd_kernel(
    X,
    stride_x,
    Y,
    stride_y,
    W,
    Rstd,
    eps,
    M,  # num rows
    N,  # num cols
    block_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_N)

    # Load input data and weights
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute mean and variance
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Store the reciprocal standard deviation
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    x_hat = x * rstd
    y = x_hat * w

    # Write output
    tl.store(Y + row * stride_y + cols, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N"],
)
@triton.jit
def _rms_norm_bwd_kernel_sm(
    X,
    stride_x,
    W,
    DY,
    stride_dy,
    DX,
    stride_dx,
    Rstd,
    DW,
    eps,
    M,  # num rows
    N,  # num cols
    rows_per_program,
    block_N: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, block_N)
    mask = cols < N

    # Load weights
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # Accumulate gradients for weights
    dw = tl.zeros((block_N,), dtype=tl.float32)

    row_end = min(row_start + rows_per_program, M)
    for row in range(row_start, row_end):
        # Load input, output gradient, and reciprocal standard deviation
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row)

        # Compute normalized input and gradients
        x_hat = x * rstd
        wdy = w * dy
        dw += dy * x_hat
        c1 = tl.sum(x_hat * wdy, axis=0) / N
        dx = (wdy - x_hat * c1) * rstd

        # Store input gradient
        tl.store(DX + row * stride_dx + cols, dx, mask=mask)

    # Store weight gradients
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)


class TritonFusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        x_shape_start = x.shape

        # Flatten input
        x = x.view(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if weight.stride(-1) != 1:
            weight = weight.contiguous()

        M, N = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        max_size = 65536 // x.element_size()
        block_N = min(max_size, triton.next_power_of_2(N))

        if block_N < N:
            raise ValueError(f"N {N} must be <= {block_N=}")

        grid = lambda meta: (M,)
        _rms_norm_fwd_kernel[grid](
            x,
            x.stride(0),
            y,
            y.stride(0),
            weight,
            rstd,
            eps,
            M,
            N,
            block_N,
        )

        ctx.eps = eps
        ctx.save_for_backward(x, weight, rstd)
        ctx.x_shape_start = x_shape_start

        y = y.reshape(x_shape_start)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        eps = ctx.eps
        x_shape_start = ctx.x_shape_start

        # Flatten input and output gradients
        dy = dy.view(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()

        M, N = dy.shape
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)

        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)

        max_size = 65536 // x.element_size()
        block_N = min(max_size, triton.next_power_of_2(N))
        rows_per_sm = math.ceil(M / sm_count)

        if block_N < N:
            raise ValueError(f"N {N} must be <= {block_N=}")

        grid = lambda meta: (sm_count,)
        _rms_norm_bwd_kernel_sm[grid](
            x,
            x.stride(0),
            weight,
            dy,
            dy.stride(0),
            dx,
            dx.stride(0),
            rstd,
            _dw,
            eps,
            M,
            N,
            rows_per_sm,
            block_N,
        )
        dw = _dw.sum(0).to(weight.dtype)
        dx = dx.view(x_shape_start)
        return dx, dw, None


# expose fusedRMSNorm as a function
def fused_rms_norm_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return TritonFusedRMSNorm.apply(x, weight, eps)  # type: ignore
