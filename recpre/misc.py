import torch
from typing import Tuple


############### RoPE reference implementations ############################


# float equivalent to the gpt-neox-like v2 implementation but not to the llama complex implementation
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    qk = torch.cat([xq, xk], dim=2).float()
    freqs_cis = freqs_cis.expand(-1, -1, -1, 2, -1)
    rotated = (freqs_cis[..., 0] * qk) + (freqs_cis[..., 1] * rotate_half(qk))
    return torch.split(rotated.type_as(xq), xq.shape[2], dim=2)


# float equivalent to the v1 implementation but not to the complex implementation
def apply_rotary_emb_v2(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis.expand(-1, -1, -1, 2, -1)
    cos, sin = freqs_cis.unbind(dim=-1)
    return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)


def rotate_half(x: torch.Tensor):
    x = x.unflatten(dim=-1, sizes=(-1, 2))
    x1, x2 = x.unbind(dim=-1)
    rotated_x = torch.stack((-x2, x1), dim=-1)
    return rotated_x.flatten(start_dim=-2)


def precompute_freqs_cis_complex(dim: int, end: int, theta: float = 10000.0):
    # this maps into the real implementation via
    # (torch.view_as_complex(freqs_cis_real)[0,:,0,:] - freqs_cis_c).norm()
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# float equivalent to the complex_like implementation in the model file:
# Would have liked to use this, but complex numbers are not well supported
def apply_rotary_emb_complex(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = torch.view_as_complex(freqs_cis)[0, :, 0, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
