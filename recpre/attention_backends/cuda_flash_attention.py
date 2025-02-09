# Trying to load Tri Dao's implementation.


def attention_computation_flash(q, k, v, mask=None, causal=True):
    try:
        from flash_attn import flash_attn_func  # type: ignore
    except ImportError:
        raise ImportError(
            "This backend can only be used if you managed to compile the flash_attn repo on your machine."
        )
    B, S, nh, hs = q.shape
    # Transpose to have batch dim and n_head in the leading spots of the tensor
    # q = q.transpose(1, 2)  # (B, nh, S, hs)
    # k = k.transpose(1, 2)
    # v = v.transpose(1, 2)

    # Self-attend: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)

    # from documentation:
    # Arguments:
    # q: (batch_size, seqlen, nheads, headdim)
    # k: (batch_size, seqlen, nheads_k, headdim)
    # v: (batch_size, seqlen, nheads_k, headdim)
    # dropout_p: float. Dropout probability.
    # Return:
    # out: (batch_size, seqlen, nheads, headdim).
    y = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
    )
    return y
