import torch


# Notes:
# This could also accomodate FIRE, with the right setup?


def attention_computation_flex(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None, center=False, debias=False
):
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    B, S, nh, hs = q.shape
    q = q.transpose(1, 2)  # (B, nh, S, hs)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    ############# this block needs to be moved outside later ######### but it does work on amd
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them)
    block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S)
    ##################################################################

    y = flex_attention(q, k, v, block_mask=block_mask)
    # returns (B, nh, S, hs)
    if center:
        y = y + v
    if debias:
        # y = y - 1 / S # cheapo version
        y = y - v.cumsum(dim=2) / torch.arange(1, S + 1, device=q.device, dtype=torch.float)[None, None, :, None]
    return y.transpose(1, 2)  # type: ignore


def attention_computation_flex_docblock(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None, center=False, debias=False
):
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    B, S, nh, hs = q.shape
    q = q.transpose(1, 2)  # (B, nh, S, hs)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them)
    block_mask = create_block_mask(
        causal, B=None, H=None, Q_LEN=S, KV_LEN=S
    )  # this is bad, mask gen should be frontloaded
    # In this case, we don't need a score_mod, so we won't pass any in.

    y = flex_attention(q, k, v, block_mask=block_mask)
    # returns (B, nh, S, hs)
    if center:
        y = y + v
    if debias:
        # y = y - 1 / S # cheapo version
        y = y - v.cumsum(dim=2) / torch.arange(1, S + 1, device=q.device, dtype=torch.float)[None, None, :, None]
    return y.transpose(1, 2)  # type: ignore


def attention_computation_flex_softcap(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None, center=False, debias=False
):
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    B, S, nh, hs = q.shape
    q = q.transpose(1, 2)  # (B, nh, S, hs)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    softcap = 20

    def soft_cap(score, b, h, q_idx, kv_idx):
        score = score / softcap
        score = torch.tanh(score)
        score = score * softcap
        return score

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them)
    block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S)
    # In this case, we don't need a score_mod, so we won't pass any in.

    y = flex_attention(q, k, v, score_mod=soft_cap, block_mask=block_mask)
    # returns (B, nh, S, hs)
    if center:
        y = y + v
    if debias:
        # y = y - 1 / S # cheapo version
        y = y - v.cumsum(dim=2) / torch.arange(1, S + 1, device=q.device, dtype=torch.float)[None, None, :, None]
    return y.transpose(1, 2)  # type: ignore
