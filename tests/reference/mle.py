"""Multilinear extension primitives over M31.

These match the Rust prover semantics in
``rust/zk_ml_prover/src/field/m31_ops.rs``:

* ``mle_evaluate`` folds **MSB-first** — ``point[0]`` is the high-bit
  fold variable. After ``k`` rounds the table collapses to a single
  field element.
* ``eq_evals`` builds the equality polynomial as a butterfly such that
  ``mle_evaluate(table, point) == sum_i table[i] * eq_evals(point)[i]``.

The bug class explicitly called out in SPEC.md (commit 86f3ab6, P10-3 GQA
fix) was a slicing error against this convention. A small sanity check
at the bottom of this file pins the layout against the Rust unit test.
"""

from __future__ import annotations

import numpy as np

from m31 import M31, mul, sub, add, from_signed


# ---------------------------------------------------------------------------
# Multilinear extension evaluation (MSB-first folding)
# ---------------------------------------------------------------------------


def mle_evaluate(evals: np.ndarray, point: np.ndarray) -> int:
    """Evaluate the multilinear extension of ``evals`` at ``point``.

    ``evals`` has length ``2**k``; ``point`` has length ``k``.
    ``point[0]`` is folded first and corresponds to the high-order
    bit of the index (MSB-first).

    Returns a Python ``int`` in ``[0, M31)``.
    """
    evals = from_signed(np.asarray(evals)).astype(np.uint32)
    point = from_signed(np.asarray(point)).astype(np.uint32)

    n = evals.size
    k = point.size
    if n != (1 << k):
        raise ValueError(
            f"mle_evaluate: |evals|=2^k expected, got |evals|={n}, k={k}"
        )

    table = evals.copy()
    size = n
    one = np.uint32(1)
    for i in range(k):
        half = size // 2
        r = np.uint32(point[i])
        one_minus_r = (one + M31 - r) % np.uint32(M31)  # field 1 - r
        # Fold: table[j] = (1 - r) * table[j] + r * table[j + half]
        lhs = mul(np.full(half, one_minus_r, dtype=np.uint32), table[:half])
        rhs = mul(np.full(half, r, dtype=np.uint32), table[half:size])
        table[:half] = add(lhs, rhs)
        size = half
    return int(table[0])


# ---------------------------------------------------------------------------
# eq(r, x) butterfly
# ---------------------------------------------------------------------------


def eq_evals(point: np.ndarray) -> np.ndarray:
    """Compute ``[eq(point, x) for x in {0,1}^k]`` as a uint32 array of length 2^k.

    Layout matches the Rust ``eq_evals`` butterfly so that, given the same
    point, ``mle_evaluate(table, point) == sum_i table[i] * eq_evals(point)[i]``.
    """
    point = from_signed(np.asarray(point)).astype(np.uint32)
    k = point.size
    n = 1 << k
    evals = np.zeros(n, dtype=np.uint32)
    evals[0] = 1
    populated = 1
    one = np.uint32(1)
    for i in range(k):
        ri = np.uint32(point[i])
        one_minus_ri = (one + np.uint32(M31) - ri) % np.uint32(M31)
        # Process in reverse to avoid overwriting yet-unread entries.
        for j in range(populated - 1, -1, -1):
            evals[2 * j + 1] = mul(evals[j], ri)
            evals[2 * j] = mul(evals[j], one_minus_ri)
        populated *= 2
    return evals


# ---------------------------------------------------------------------------
# Sanity check (executed only when this file is run as a script).
#
# Pins the MSB-first folding direction against the Rust unit test in
# `field/m31_ops.rs::tests::test_mle_evaluate` and `test_eq_evals`.
# ---------------------------------------------------------------------------


def _sanity():  # pragma: no cover - executed manually
    # MLE on f(x0, x1) with table = [3, 5, 7, 11].
    table = np.array([3, 5, 7, 11], dtype=np.uint32)
    assert mle_evaluate(table, np.array([0, 0])) == 3
    assert mle_evaluate(table, np.array([0, 1])) == 5
    assert mle_evaluate(table, np.array([1, 0])) == 7
    assert mle_evaluate(table, np.array([1, 1])) == 11

    # eq(r, x) layout. With r = (2, 3) the Rust test expects:
    # idx 0 = (0,0): (1-2)(1-3) =  2
    # idx 1 = (0,1): (1-2)( 3 ) = -3
    # idx 2 = (1,0): ( 2 )(1-3) = -4
    # idx 3 = (1,1): ( 2 )( 3 ) =  6
    eqs = eq_evals(np.array([2, 3]))
    expected = np.array([2, -3, -4, 6], dtype=np.int64)
    expected_field = from_signed(expected)
    assert np.array_equal(eqs, expected_field), (eqs, expected_field)

    # Bilinearity: mle_evaluate(t, p) == sum t[i] * eq_evals(p)[i]
    rng = np.random.default_rng(0)
    for _ in range(10):
        k = rng.integers(1, 7)
        t = rng.integers(0, M31, size=1 << k, dtype=np.uint64).astype(np.uint32)
        p = rng.integers(0, M31, size=k, dtype=np.uint64).astype(np.uint32)
        e = eq_evals(p)
        prod = mul(t, e).astype(np.int64)
        s = int(prod.sum() % M31)
        assert s == mle_evaluate(t, p), (s, mle_evaluate(t, p))

    print("mle.py sanity OK")


if __name__ == "__main__":
    _sanity()
