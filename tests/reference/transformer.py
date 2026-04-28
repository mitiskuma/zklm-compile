"""Layer-level reference forward for a Qwen3-style transformer block.

Mirrors ``qwen_forward_indexed`` in
``rust/zk_ml_prover/src/transformer/qwen.rs``. The output dict matches
the field set in ``QwenForwardTrace`` so the proptest harness can
compare every intermediate.

Imports are intentionally local (not eager at module load) so callers
that only need the per-op references in :mod:`reference` don't pay
the SwiGLU / attention setup cost.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from m31 import M31, HALF, add, mul, from_signed
from reference import (
    matmul,
    rmsnorm_forward,
    silu_table,
    sigmoid_table,
)
from attention import attention_seq_len_1


# ---------------------------------------------------------------------------
# Helpers used by the layer
# ---------------------------------------------------------------------------


def residual_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Field-add two equal-shaped vectors: y = a + b (mod M31)."""
    a = np.asarray(a, dtype=np.uint32)
    b = np.asarray(b, dtype=np.uint32)
    if a.shape != b.shape:
        raise ValueError(f"residual_add: shape mismatch {a.shape} vs {b.shape}")
    return add(a, b)


def swiglu(gate_silu: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU activation: silu(gate) ⊙ up.

    Both inputs are expected to already be M31-encoded; we just do the
    elementwise multiply.
    """
    gate_silu = np.asarray(gate_silu, dtype=np.uint32)
    up = np.asarray(up, dtype=np.uint32)
    if gate_silu.shape != up.shape:
        raise ValueError(
            f"swiglu: shape mismatch {gate_silu.shape} vs {up.shape}"
        )
    return mul(gate_silu, up)


def sigmoid_gate(attn_out: np.ndarray, sigmoid_of_g: np.ndarray) -> np.ndarray:
    """Output gate: gated_attn = attn_out ⊙ sigmoid(g_proj)."""
    attn_out = np.asarray(attn_out, dtype=np.uint32)
    sigmoid_of_g = np.asarray(sigmoid_of_g, dtype=np.uint32)
    if attn_out.shape != sigmoid_of_g.shape:
        raise ValueError(
            f"sigmoid_gate: shape mismatch {attn_out.shape} vs {sigmoid_of_g.shape}"
        )
    return mul(attn_out, sigmoid_of_g)


# ---------------------------------------------------------------------------
# Requantize M31 → int16 → M31 (matches `requantize_to_i16_field` in Rust)
# ---------------------------------------------------------------------------


def _requantize_to_i16_field(values: np.ndarray) -> np.ndarray:
    """Map M31 values into the int16-encoded field domain.

    Steps (mirrors the Rust impl):
      1. Centered (signed) view of each input.
      2. Find max absolute value across the vector.
      3. Scale down so |max| → 32767 using integer division with rounding.
      4. Saturate to [-32768, 32767], re-encode as field.

    Returns a uint32 array.
    """
    values = np.asarray(values, dtype=np.uint32)
    signed = values.astype(np.int64)
    signed = np.where(signed > HALF, signed - M31, signed)

    max_abs = int(np.max(np.abs(signed))) if signed.size > 0 else 1
    if max_abs < 1:
        max_abs = 1

    if max_abs <= 32767:
        scaled = signed
    else:
        # Rounded integer division, mirroring Rust:
        # if v >= 0: (v * 32767 + half) / max_abs
        # else:      (v * 32767 - half) / max_abs   (truncation toward zero)
        half = max_abs // 2
        positive_part = (signed * 32767 + half) // max_abs
        negative_part = -((-signed * 32767 + half) // max_abs)
        # `//` in numpy floors. Rust integer division truncates toward zero,
        # which for non-negative dividends agrees with floor. For negatives
        # we do `-((-v*32767 + half) // max_abs)` to emulate truncation.
        scaled = np.where(signed >= 0, positive_part, negative_part)

    clamped = np.clip(scaled, -32768, 32767).astype(np.int64)
    return np.mod(clamped, M31).astype(np.uint32)


def _build_lookup_index(table: np.ndarray) -> Dict[int, int]:
    """Convert (65536, 2) lookup table to dict[input_field -> output_field]."""
    return {int(table[i, 0]): int(table[i, 1]) for i in range(table.shape[0])}


def _apply_lookup(values: np.ndarray, lookup: Dict[int, int]) -> np.ndarray:
    """Vectorized table lookup over a uint32 M31 array."""
    out = np.empty_like(values, dtype=np.uint32)
    for i, v in enumerate(values.tolist()):
        if v not in lookup:
            raise KeyError(
                f"lookup miss: M31 value {v} not in table — input was not "
                f"requantized into the int16 domain first"
            )
        out[i] = lookup[v]
    return out


# ---------------------------------------------------------------------------
# Headline: full Qwen layer forward
# ---------------------------------------------------------------------------


def qwen_layer_forward(x: np.ndarray, weights: dict, config: dict) -> Dict[str, np.ndarray]:
    """Full Qwen layer forward — mirrors ``qwen_forward_indexed`` in Rust.

    ``weights`` keys: norm1_gamma, w_q, w_k, w_v, w_o, w_g_proj,
                      norm2_gamma, w_gate, w_up, w_down (all uint32 / M31).
    ``config`` keys: d_model, d_ff, num_q_heads, num_kv_heads, d_head,
                     v_num_heads, v_d_head, silu_scale, sigmoid_scale.

    Returns a dict with every intermediate the prover trace exposes:
        x, norm1_x, norm1_delta, norm1_out, q, k, v, attn_out,
        g_proj_out_raw, g_proj_out, g_proj_sigmoid, gated_attn,
        o_proj_out, h, norm2_x, norm2_delta, norm2_out, gate_out_raw,
        gate_out, gate_silu, up_out, swiglu_out, down_out, output.
    """
    x = np.asarray(x, dtype=np.uint32)

    d_model = config["d_model"]
    d_ff = config["d_ff"]
    num_q_heads = config["num_q_heads"]
    num_kv_heads = config["num_kv_heads"]
    d_head = config["d_head"]
    v_num_heads = config.get("v_num_heads", 0) or num_kv_heads
    v_d_head = config.get("v_d_head", 0) or d_head
    silu_scale = config["silu_scale"]
    sigmoid_scale = config["sigmoid_scale"]

    q_dim = num_q_heads * d_head
    k_dim = num_kv_heads * d_head
    v_dim = v_num_heads * v_d_head
    is_gqa_full_attn = num_q_heads != num_kv_heads
    attn_out_dim = q_dim if is_gqa_full_attn else v_dim

    # ----- RMSNorm 1 -----
    norm1_out, norm1_delta = rmsnorm_forward(x, weights["norm1_gamma"], d_model)
    # The Rust trace stores the (possibly perturbed) input it actually
    # normalized. Reconstruct it from x + delta to match exactly.
    norm1_x = x.copy()
    if norm1_delta != 0:
        from m31 import to_field_scalar  # local import keeps top tidy
        norm1_x[0] = int(
            add(
                np.array([norm1_x[0]], dtype=np.uint32),
                np.array([to_field_scalar(int(norm1_delta))], dtype=np.uint32),
            )[0]
        )

    # ----- QKV + g_proj projections (all from norm1_out) -----
    q = matmul(weights["w_q"], norm1_out, q_dim, d_model)
    k = matmul(weights["w_k"], norm1_out, k_dim, d_model)
    v = matmul(weights["w_v"], norm1_out, v_dim, d_model)
    g_proj_out_raw = matmul(weights["w_g_proj"], norm1_out, attn_out_dim, d_model)

    # ----- Attention (seq_len = 1 fast paths) -----
    attn_out = attention_seq_len_1(
        q=q,
        k=k,
        v=v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        d_head=d_head,
        is_gqa_full_attn=is_gqa_full_attn,
    )

    # ----- Output gate: sigmoid(g_proj) ⊙ attn_out -----
    g_proj_out = _requantize_to_i16_field(g_proj_out_raw)
    sigmoid_lookup = _build_lookup_index(sigmoid_table(sigmoid_scale))
    g_proj_sigmoid = _apply_lookup(g_proj_out, sigmoid_lookup)
    gated_attn = sigmoid_gate(attn_out, g_proj_sigmoid)

    # ----- O projection on gated attention -----
    o_proj_out = matmul(weights["w_o"], gated_attn, d_model, attn_out_dim)

    # ----- Residual 1 -----
    h = residual_add(x, o_proj_out)

    # ----- RMSNorm 2 -----
    norm2_out, norm2_delta = rmsnorm_forward(h, weights["norm2_gamma"], d_model)
    norm2_x = h.copy()
    if norm2_delta != 0:
        from m31 import to_field_scalar
        norm2_x[0] = int(
            add(
                np.array([norm2_x[0]], dtype=np.uint32),
                np.array([to_field_scalar(int(norm2_delta))], dtype=np.uint32),
            )[0]
        )

    # ----- Gate + Up projections (both from norm2_out) -----
    gate_out_raw = matmul(weights["w_gate"], norm2_out, d_ff, d_model)
    up_out = matmul(weights["w_up"], norm2_out, d_ff, d_model)

    # ----- SiLU lookup on gate -----
    gate_out = _requantize_to_i16_field(gate_out_raw)
    silu_lookup = _build_lookup_index(silu_table(silu_scale))
    gate_silu = _apply_lookup(gate_out, silu_lookup)
    swiglu_out = swiglu(gate_silu, up_out)

    # ----- Down projection -----
    down_out = matmul(weights["w_down"], swiglu_out, d_model, d_ff)

    # ----- Residual 2 -----
    output = residual_add(h, down_out)

    return {
        "x": x,
        "norm1_x": norm1_x,
        "norm1_delta": int(norm1_delta),
        "norm1_out": norm1_out,
        "q": q,
        "k": k,
        "v": v,
        "attn_out": attn_out,
        "g_proj_out_raw": g_proj_out_raw,
        "g_proj_out": g_proj_out,
        "g_proj_sigmoid": g_proj_sigmoid,
        "gated_attn": gated_attn,
        "o_proj_out": o_proj_out,
        "h": h,
        "norm2_x": norm2_x,
        "norm2_delta": int(norm2_delta),
        "norm2_out": norm2_out,
        "gate_out_raw": gate_out_raw,
        "gate_out": gate_out,
        "gate_silu": gate_silu,
        "up_out": up_out,
        "swiglu_out": swiglu_out,
        "down_out": down_out,
        "output": output,
    }
