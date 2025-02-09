import torch
import contextlib
# from torch.nn.attention import SDPBackend, sdpa_kernel


with contextlib.suppress(ModuleNotFoundError):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)  # flag is not yet implemented on earlier pytorch versions


def attention_computation_sdpa(q, k, v, mask=None, center=False, debias=False):
    B, S, nh, hs = q.shape
    # Transpose to have batch dim and n_head in the leading spots of the tensor
    q = q.transpose(1, 2)  # (B, nh, S, hs)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Self-attend: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)
    # with sdpa_kernel(SDPBackend.FLASH_ATTENTION): # cannot compile without graph break
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=True)
    # returns (B, nh, S, hs)
    if center:
        y = y + v
    if debias:
        # y = y - 1 / S # cheapo version
        y = y - v.cumsum(dim=2) / torch.arange(1, S + 1, device=q.device, dtype=q.dtype)[None, None, :, None]
        # Note: the previous line needs to be dtype=q.dtype, otherwise the sdpa does not trigger FA2
    return y.transpose(1, 2)
