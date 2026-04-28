"""M31 field arithmetic helpers.

The Mersenne-31 prime field is M31 = 2^31 - 1 = 2_147_483_647.

All public functions here return numpy arrays whose dtype is the
canonical positive representation `np.uint32`, with values in
``[0, M31)``. Where signed (centered) views are useful, the helpers
``to_signed`` / ``from_signed`` round-trip between the two views.

Conventions match the Rust prover (`rust/zk_ml_prover/src/field/m31_ops.rs`):
    * `to_field(x)` for signed Python ints == `from_signed(np.array([x]))`.
    * `from_field(F)` (centered view) == `to_signed(np.array([F]))`.
"""

from __future__ import annotations

import numpy as np

# Mersenne-31 prime: 2^31 - 1.
M31 = (1 << 31) - 1
HALF = M31 // 2  # threshold separating positive / negative under the centered view


# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------


def _as_array(x):
    """Promote scalars to 0-d arrays without copying when possible."""
    return np.asarray(x)


def to_signed(x) -> np.ndarray:
    """Return the canonical signed (centered) view of M31 elements.

    Values strictly greater than ``M31 // 2`` are mapped to negative integers
    in ``(-M31 // 2, 0)``; values in ``[0, M31 // 2]`` are kept positive.

    Output dtype: ``int64`` (always wide enough to hold ``±M31``).
    """
    arr = _as_array(x).astype(np.int64, copy=False)
    # Reduce first so callers can pass raw signed integers.
    reduced = np.mod(arr, M31).astype(np.int64)
    return np.where(reduced > HALF, reduced - M31, reduced)


def from_signed(x) -> np.ndarray:
    """Inverse of :func:`to_signed`: reduce ``x`` mod M31 to ``[0, M31)``."""
    arr = _as_array(x).astype(np.int64, copy=False)
    reduced = np.mod(arr, M31).astype(np.uint32)
    return reduced


# ---------------------------------------------------------------------------
# Field arithmetic on np.uint32 arrays in [0, M31)
# ---------------------------------------------------------------------------


def add(a, b) -> np.ndarray:
    """Field addition. Inputs are interpreted in [0, M31)."""
    a64 = _as_array(a).astype(np.int64, copy=False)
    b64 = _as_array(b).astype(np.int64, copy=False)
    return np.mod(a64 + b64, M31).astype(np.uint32)


def sub(a, b) -> np.ndarray:
    """Field subtraction."""
    a64 = _as_array(a).astype(np.int64, copy=False)
    b64 = _as_array(b).astype(np.int64, copy=False)
    return np.mod(a64 - b64, M31).astype(np.uint32)


def neg(a) -> np.ndarray:
    """Field negation."""
    a64 = _as_array(a).astype(np.int64, copy=False)
    return np.mod(-a64, M31).astype(np.uint32)


def mul(a, b) -> np.ndarray:
    """Field multiplication.

    Uses int64 intermediates (a, b < 2^31, product < 2^62 → fits int64).
    """
    a64 = _as_array(a).astype(np.int64, copy=False)
    b64 = _as_array(b).astype(np.int64, copy=False)
    return np.mod(a64 * b64, M31).astype(np.uint32)


def pow_field(a, exponent: int) -> np.ndarray:
    """Square-and-multiply exponentiation in M31."""
    if exponent < 0:
        raise ValueError("negative exponents not supported (use inv first)")
    base = from_signed(_as_array(a))
    result = np.full(base.shape, 1, dtype=np.uint32)
    e = exponent
    while e > 0:
        if e & 1:
            result = mul(result, base)
        base = mul(base, base)
        e >>= 1
    return result


def inv(a) -> np.ndarray:
    """Multiplicative inverse via Fermat: a^(p-2)."""
    arr = from_signed(_as_array(a))
    if np.any(arr == 0):
        raise ZeroDivisionError("inv: zero element has no inverse in M31")
    return pow_field(arr, M31 - 2)


def is_qr(a) -> np.ndarray:
    """Quadratic residue test in M31 (p ≡ 3 mod 4).

    Uses Euler's criterion: a^((p-1)/2) == 1 (with 0 treated as a QR
    to match the Rust convention in `is_qr_m31`).
    """
    arr = from_signed(_as_array(a))
    pow_half = pow_field(arr, (M31 - 1) // 2)
    return (arr == 0) | (pow_half == 1)


def mod_sqrt(a) -> np.ndarray:
    """Modular square root in M31 (p ≡ 3 mod 4): a^((p+1)/4) = a^(2^29).

    Caller is responsible for ensuring ``a`` is a quadratic residue.
    """
    return pow_field(_as_array(a), (M31 + 1) // 4)


# ---------------------------------------------------------------------------
# Convenience: scalar ↔ field
# ---------------------------------------------------------------------------


def to_field_scalar(x: int) -> int:
    """Map a Python signed int to the canonical M31 representative (int)."""
    return int(from_signed(np.array([x], dtype=np.int64))[0])


def from_field_scalar(x: int) -> int:
    """Map a canonical M31 representative to its centered (signed) view."""
    return int(to_signed(np.array([x], dtype=np.int64))[0])
