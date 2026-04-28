"""Attention sub-references.

Two regimes:

* ``attention_seq_len_1`` — at sequence length 1 the prover takes one
  of two shortcuts depending on whether GQA full-attention is enabled:

    - **GDN-style** (``num_q_heads == num_kv_heads``): ``attn_out = v``.
      The shape of ``attn_out`` is ``v_num_heads * v_d_head`` — the V
      dim may be asymmetric in Qwen3.5-4B/9B (P10-3).
    - **GQA full-attn** (``num_q_heads != num_kv_heads``): each query
      head replicates from its KV group.
      ``attn_out[h, d] = v[h // heads_per_group, d]`` where
      ``heads_per_group = num_q_heads / num_kv_heads``.

  The MLE relation that the v↔attn_out P10-3 sub-proof relies on is:

      MLE(attn_out, r_attn) = MLE(v, r_v)

  where for GDN ``r_v = r_attn`` and for GQA ``r_v`` is built from the
  group-prefix of ``r_attn`` joined with the d_head suffix
  (``r_attn = [r_q || r_d]`` where ``r_q`` has ``log2(num_q_heads)``
  bits and the high ``log2(heads_per_group)`` of those are replication
  bits that drop out — i.e. ``r_v = [r_q[heads_per_group_log:] || r_d]``).
  See commit 86f3ab6 for the bit-ordering math.

* ``attention_full`` — sequence length ≥ 2 row-decomposed attention,
  matching ``rust/zk_ml_prover/src/proving/attention.rs::prove_row_attention``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from m31 import M31, add, mul, inv, from_signed


# ---------------------------------------------------------------------------
# seq_len = 1 shortcuts
# ---------------------------------------------------------------------------


def attention_seq_len_1(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    num_q_heads: int,
    num_kv_heads: int,
    d_head: int,
    is_gqa_full_attn: bool,
) -> np.ndarray:
    """Return ``attn_out`` for the seq_len=1 fast paths.

    ``q``, ``k``, ``v`` are flat M31-encoded numpy arrays.

    For the GDN-style branch the V dim may be asymmetric (``v.size`` is
    ``v_num_heads * v_d_head``, NOT necessarily ``num_kv_heads * d_head``),
    so we simply pass ``v`` through and return a copy. The output dim
    matches ``v.size``.

    For the GQA full-attention branch the V dim must be
    ``num_kv_heads * d_head`` (symmetric) and we replicate each KV
    head ``heads_per_group`` times:
        ``attn_out[h, d] = v[h // heads_per_group, d]``.
    Output dim is ``num_q_heads * d_head``.
    """
    v = np.asarray(v, dtype=np.uint32)

    if not is_gqa_full_attn:
        # GDN identity: attn_out = v (size may differ from num_kv_heads*d_head
        # in asymmetric V layouts; just preserve whatever the caller gave us).
        return v.copy()

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"GQA full-attn requires num_q_heads % num_kv_heads == 0, "
            f"got num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}"
        )
    heads_per_group = num_q_heads // num_kv_heads
    expected_v = num_kv_heads * d_head
    if v.size != expected_v:
        raise ValueError(
            f"GQA full-attn requires |v| == num_kv_heads*d_head ({expected_v}), "
            f"got |v|={v.size}"
        )

    out = np.empty(num_q_heads * d_head, dtype=np.uint32)
    for h in range(num_q_heads):
        kv_idx = h // heads_per_group
        out[h * d_head : (h + 1) * d_head] = v[kv_idx * d_head : (kv_idx + 1) * d_head]
    return out


# ---------------------------------------------------------------------------
# seq_len ≥ 2 row-decomposed attention
# ---------------------------------------------------------------------------


def _softmax_row_via_exp_table(scores: np.ndarray, exp_lookup: dict) -> np.ndarray:
    """Compute softmax of one row using the (input → exp(x)) lookup table.

    ``scores`` is a uint32 array of M31 elements; ``exp_lookup`` maps
    ``input_field_u32 -> output_field_u32`` (built from
    :func:`reference.exp_table`).

    Returns the attention-weight row as a uint32 M31 array.
    """
    scores = np.asarray(scores, dtype=np.uint32)
    e = np.empty_like(scores, dtype=np.uint32)
    for i, s in enumerate(scores.tolist()):
        if s not in exp_lookup:
            raise KeyError(
                f"score value {s} not found in exp_table; the score must "
                f"already be requantized to int16-encoded field elements"
            )
        e[i] = exp_lookup[s]
    sum_e = int(e.astype(np.int64).sum() % M31)
    if sum_e == 0:
        raise ValueError("softmax denominator is zero in M31; can't invert")
    inv_sum = int(inv(np.array([sum_e], dtype=np.uint32))[0])
    y = mul(e, np.full_like(e, inv_sum))
    return y


def attention_full(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    num_heads: int,
    seq_len: int,
    d_head: int,
    exp_scale: int,
) -> np.ndarray:
    """Reference for ``prove_row_attention`` with ``seq_len >= 2``.

    Q, K, V are flat numpy arrays of shape ``(num_heads, seq_len, d_head)``
    in row-major order. Returns the attention output as a flat array
    of the same shape (``num_heads * seq_len * d_head`` field elements).

    The score / softmax / output composition matches the Rust prover:
        scores_i = Q[h, i] @ K[h]^T              (matmul)
        attn_w_i = softmax_via_exp_table(scores_i)
        out_i    = attn_w_i @ V[h]               (matmul)

    ``exp_scale`` selects the exp lookup table; only score values that
    happen to fall in the int16 quantized domain will resolve. The
    proptest harness is responsible for picking inputs whose scores
    land in-table (or for catching out-of-range as a separate failure
    mode).
    """
    Q = np.asarray(Q, dtype=np.uint32)
    K = np.asarray(K, dtype=np.uint32)
    V = np.asarray(V, dtype=np.uint32)

    head_size = seq_len * d_head
    expected = num_heads * head_size
    if Q.size != expected or K.size != expected or V.size != expected:
        raise ValueError(
            f"attention_full: each of Q,K,V must be num_heads*seq_len*d_head"
            f" = {expected}, got Q={Q.size} K={K.size} V={V.size}"
        )

    # Lazy import to avoid attention.py ↔ reference.py cycle.
    from reference import exp_table

    table = exp_table(exp_scale)
    exp_lookup = {int(table[i, 0]): int(table[i, 1]) for i in range(table.shape[0])}

    out = np.zeros_like(V, dtype=np.uint32)
    for h in range(num_heads):
        q_h = Q[h * head_size : (h + 1) * head_size].reshape(seq_len, d_head)
        k_h = K[h * head_size : (h + 1) * head_size].reshape(seq_len, d_head)
        v_h = V[h * head_size : (h + 1) * head_size].reshape(seq_len, d_head)

        # Row-decomposed: each row independently.
        for i in range(seq_len):
            q_row = q_h[i]  # (d_head,)
            # scores[j] = sum_l k_h[j, l] * q_row[l]   for j in 0..seq_len
            # Reduce per-element products mod M31 BEFORE summing to keep
            # the running sum under int64.
            scores = np.zeros(seq_len, dtype=np.uint32)
            for j in range(seq_len):
                products_mod = np.mod(
                    k_h[j].astype(np.int64) * q_row.astype(np.int64),
                    M31,
                )
                acc = int(products_mod.sum() % M31)
                scores[j] = np.uint32(acc)

            attn_w = _softmax_row_via_exp_table(scores, exp_lookup)

            # out_row[d] = sum_l attn_w[l] * v_h[l, d]
            out_row = np.zeros(d_head, dtype=np.uint32)
            for d in range(d_head):
                products_mod = np.mod(
                    attn_w.astype(np.int64) * v_h[:, d].astype(np.int64),
                    M31,
                )
                acc = int(products_mod.sum() % M31)
                out_row[d] = np.uint32(acc)

            out[h * head_size + i * d_head : h * head_size + (i + 1) * d_head] = out_row

    return out
