"""Mersenne-31 field arithmetic.

The Mersenne prime p = 2^31 - 1 = 2147483647.
INT8 × INT8 accumulated over 768 terms = 24 bits max → fits with 7 bits headroom.

All operations return values in [0, p-1].
"""

from __future__ import annotations

from .constants import M31


def add(a: int, b: int) -> int:
    """Field addition: (a + b) mod p."""
    s = a + b
    return s - M31 if s >= M31 else s


def sub(a: int, b: int) -> int:
    """Field subtraction: (a - b) mod p."""
    s = a - b
    return s + M31 if s < 0 else s


def mul(a: int, b: int) -> int:
    """Field multiplication: (a * b) mod p.

    Uses the Mersenne prime reduction: x mod (2^31 - 1) = (x & M31) + (x >> 31),
    repeated once since the sum can exceed M31.
    """
    prod = a * b
    lo = prod & M31
    hi = prod >> 31
    r = lo + hi
    return r - M31 if r >= M31 else r


def neg(a: int) -> int:
    """Field negation: -a mod p."""
    return 0 if a == 0 else M31 - a


def inv(a: int) -> int:
    """Field inverse via Fermat's little theorem: a^(p-2) mod p."""
    if a == 0:
        raise ZeroDivisionError("Cannot invert zero in M31")
    return pow(a, M31 - 2, M31)


def to_field(x: int) -> int:
    """Convert a (possibly negative) integer to field element."""
    return x % M31


def from_field(x: int) -> int:
    """Convert a field element to signed integer (centered representation).

    Values > p//2 are interpreted as negative.
    """
    return x - M31 if x > M31 // 2 else x


def batch_mul(a: list[int], b: list[int]) -> list[int]:
    """Element-wise field multiplication."""
    return [mul(ai, bi) for ai, bi in zip(a, b)]


def inner_product(a: list[int], b: list[int]) -> int:
    """Field inner product: sum(a_i * b_i)."""
    s = 0
    for ai, bi in zip(a, b):
        s += ai * bi
    # Reduce at the end (Python ints don't overflow)
    return s % M31
