"""Top-level reference forwards.

This module is the headline API for Phase 11 P11-1. The functions here
match the contract in ``SPEC.md``; the Rust prover/verifier and the
proptest harness call AGAINST these signatures.

Quantization tolerance:
    * Lookup tables (silu, sigmoid, exp): rounding is **half-to-even**
      (numpy default), per the SPEC. The Rust prover currently calls
      ``f64::round`` (half-away-from-zero), so individual entries may
      differ by ±1 LSB at the int16 level. SPEC.md line 170 explicitly
      allows this.
    * RMSNorm: QR-perturbation may flip ``x[0]`` by ±delta; the
      reference returns the applied delta so the prover can absorb it.
    * Attention: softmax goes exp lookup → sum → divide. End-to-end
      tolerance is ±4 LSBs at int16 (per SPEC).

Field canonicalization: all M31 outputs are in ``[0, M31)`` (uint32).
Tensor flattening: row-major. ``W[i, j]`` lives at flat index ``i*n + j``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from m31 import (
    M31,
    HALF,
    add,
    sub,
    mul,
    inv,
    is_qr,
    mod_sqrt,
    from_signed,
    to_signed,
    pow_field,
    to_field_scalar,
)
from mle import mle_evaluate, eq_evals  # re-export for callers
from attention import attention_seq_len_1, attention_full  # re-export


# ---------------------------------------------------------------------------
# matmul: y = W @ x + bias  (mod M31)
# ---------------------------------------------------------------------------


def matmul(
    W: np.ndarray,
    x: np.ndarray,
    m: int,
    n: int,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Row-major matmul over M31.

    ``W`` is shape ``(m, n)`` flat-encoded (``W[i*n + j]``). ``x`` is
    ``(n,)``. Returns ``y`` of shape ``(m,)``.

    All values are interpreted as M31 elements; outputs are canonical
    ``uint32`` in ``[0, M31)``.
    """
    W = np.asarray(W).reshape(m, n)
    x = np.asarray(x).reshape(n)

    # Each summand W[i,j]*x[j] is < M31^2 < 2^62. Adding ~16384 of those
    # would overflow int64 (max 2^63 - 1). To stay safe for typical Qwen
    # widths (n ≤ ~16k) and beyond, we accumulate by chunks: each chunk
    # of size <= 2 reductions per add stays within int64.
    #
    # Strategy: accumulate the full row in int64 by reducing mod M31
    # after every CHUNK columns. CHUNK = 2 keeps the running sum under
    # 2^63 even for adversarial inputs (2^62 * 2 = 2^63 — borderline,
    # so we use CHUNK = 1 for the partial-product step and reduce after
    # summing into a running M31 accumulator).
    #
    # Simpler & vectorized: compute W*x as int64 elementwise, reduce
    # mod M31 row-wise into uint32, then sum those reduced products in
    # int64 (each is < 2^31; 2^31 * n stays below 2^63 for n < 2^32).
    W64 = W.astype(np.int64, copy=False)
    x64 = x.astype(np.int64, copy=False)
    products_mod = np.mod(W64 * x64[None, :], M31)  # int64, each < 2^31
    y = np.mod(products_mod.sum(axis=1), M31).astype(np.uint32)
    if bias is not None:
        b = np.asarray(bias, dtype=np.uint32).reshape(m)
        y = add(y, b)
    return y


# ---------------------------------------------------------------------------
# RMSNorm with QR perturbation
# ---------------------------------------------------------------------------


def rmsnorm_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    d: int,
    perturbation_delta: int = 0,
) -> tuple[np.ndarray, int]:
    """RMSNorm with QR-perturbation loop, mirroring the Rust prover.

    Returns ``(output, applied_delta)``. ``applied_delta`` is the signed
    perturbation actually applied to ``x[0]`` (0 when no perturbation
    was needed). The prover must absorb this value onto the transcript
    (see SOUNDNESS S5 in the Rust source).

    The optional ``perturbation_delta`` argument is reserved for tests
    that want to force a specific delta; when 0 we run the same search
    loop the Rust forward does.

    Math: y = gamma * x' * r where ``r = sqrt(d / sum(x'**2))`` and
    ``x'`` is ``x`` with ``x'[0] = x[0] + delta``.
    """
    x = np.asarray(x, dtype=np.uint32).copy()
    gamma = np.asarray(gamma, dtype=np.uint32).reshape(d)
    if x.size != d:
        raise ValueError(f"rmsnorm_forward: |x|={x.size} != d={d}")

    d_field = np.uint32(d % M31)

    def try_compute(x_p: np.ndarray) -> Optional[np.ndarray]:
        # Reduce squares mod M31 element-wise before summing to avoid
        # int64 overflow for large d (each square is < 2^62; ~16k of
        # them would overflow int64 if not reduced first).
        sq_mod = np.mod(x_p.astype(np.int64) ** 2, M31)
        sum_sq = int(sq_mod.sum() % M31)
        if sum_sq == 0:
            return None
        sum_sq_arr = np.array([sum_sq], dtype=np.uint32)
        target = int(mul(np.array([d_field], dtype=np.uint32), inv(sum_sq_arr))[0])
        if not bool(is_qr(np.array([target], dtype=np.uint32))[0]):
            return None
        r = int(mod_sqrt(np.array([target], dtype=np.uint32))[0])
        # output = gamma * x_p * r
        r_vec = np.full(d, r, dtype=np.uint32)
        return mul(mul(gamma, x_p), r_vec)

    # Forced delta path (used by tests that already know what the prover applied).
    if perturbation_delta != 0:
        adj = to_field_scalar(int(perturbation_delta))
        x[0] = int(add(np.array([x[0]], dtype=np.uint32), np.array([adj], dtype=np.uint32))[0])
        out = try_compute(x)
        if out is None:
            raise ValueError(
                f"rmsnorm_forward: forced perturbation delta={perturbation_delta} "
                f"did not yield a quadratic residue"
            )
        return out, int(perturbation_delta)

    # Default: search like the Rust prover.
    out = try_compute(x)
    if out is not None:
        return out, 0

    x0_orig = int(x[0])
    for delta in range(1, 100):
        for sign in (1, -1):
            perturbation = delta * sign
            adj = to_field_scalar(perturbation)
            new_x0 = int(
                add(
                    np.array([x0_orig], dtype=np.uint32),
                    np.array([adj], dtype=np.uint32),
                )[0]
            )
            x_p = x.copy()
            x_p[0] = new_x0
            out = try_compute(x_p)
            if out is not None:
                return out, perturbation
    raise RuntimeError("rmsnorm_forward: could not find QR perturbation in 200 attempts")


# ---------------------------------------------------------------------------
# Lookup tables (silu, sigmoid, exp)
#
# Each table has 65536 rows of (input_field, output_field) where input
# enumerates all i16 values and output is the quantized result. Match
# the Rust ``proving::lookup::build_*_table`` byte-for-byte after M31
# canonicalization (subject to the ±1 LSB rounding tolerance noted at
# the top of this module).
# ---------------------------------------------------------------------------


def _i16_to_field_arr(values: np.ndarray) -> np.ndarray:
    """Map an int16 array to canonical M31 representatives (uint32)."""
    return from_signed(values.astype(np.int64))


def _quantize_i16(values_f64: np.ndarray, scale: float) -> np.ndarray:
    """Round-half-to-even quantization with saturation to int16."""
    raw = np.round(values_f64 * scale)  # numpy default: half-to-even
    raw = np.clip(raw, -32768.0, 32767.0)
    return raw.astype(np.int16)


def _build_table(values_f: np.ndarray, scale: int) -> np.ndarray:
    """Return a (65536, 2) int32 array of (input_field, output_field)
    where ``values_f`` is the float64 output for each int16 input.
    """
    s = float(scale)
    inputs_i16 = np.arange(65536, dtype=np.uint16).astype(np.int16)  # 0..32767, then -32768..-1
    outputs_i16 = _quantize_i16(values_f, s)
    in_field = _i16_to_field_arr(inputs_i16)
    out_field = _i16_to_field_arr(outputs_i16)
    table = np.empty((65536, 2), dtype=np.int32)
    table[:, 0] = in_field.astype(np.int64)
    table[:, 1] = out_field.astype(np.int64)
    return table


def silu_table(scale: int) -> np.ndarray:
    """SiLU table: y = x * sigmoid(x), quantized to int16.

    Returns ``(65536, 2)`` int32: ``[input_field, output_field]``.
    """
    s = float(scale)
    inputs_i16 = np.arange(65536, dtype=np.uint16).astype(np.int16)
    x = inputs_i16.astype(np.float64) / s
    sig = 1.0 / (1.0 + np.exp(-x))
    y = x * sig
    return _build_table(y, scale)


def sigmoid_table(scale: int) -> np.ndarray:
    """Sigmoid table: y = 1/(1+exp(-x/scale)), quantized to int16."""
    inputs_i16 = np.arange(65536, dtype=np.uint16).astype(np.int16)
    x = inputs_i16.astype(np.float64) / float(scale)
    y = 1.0 / (1.0 + np.exp(-x))
    return _build_table(y, scale)


def exp_table(scale: int) -> np.ndarray:
    """Exp table: y = exp(x/scale), quantized to int16."""
    inputs_i16 = np.arange(65536, dtype=np.uint16).astype(np.int16)
    x = inputs_i16.astype(np.float64) / float(scale)
    # exp can grow huge; the quantizer saturates to ±32767 at the i16 level.
    with np.errstate(over="ignore"):
        y = np.exp(x)
    return _build_table(y, scale)


# ---------------------------------------------------------------------------
# Headline: full Qwen layer forward.
#
# Mirrors `qwen_forward_indexed` in the Rust source. Returns a dict of
# every intermediate the prover trace exposes so the proptest harness
# can compare value-for-value at any stage.
# ---------------------------------------------------------------------------


def _v_dim(config: dict) -> int:
    """Effective V dim — falls back to symmetric K=V if v_* unset."""
    v_num_heads = config.get("v_num_heads", 0) or config["num_kv_heads"]
    v_d_head = config.get("v_d_head", 0) or config["d_head"]
    return v_num_heads * v_d_head


def qwen_layer_forward(x: np.ndarray, weights: dict, config: dict) -> dict:
    """Full Qwen layer forward.

    See SPEC.md for the contract. Imported lazily to avoid pulling
    transformer helpers at module load time (so callers only paying
    for matmul/lookups don't need the SwiGLU stack).
    """
    from transformer import qwen_layer_forward as _impl

    return _impl(x, weights, config)
