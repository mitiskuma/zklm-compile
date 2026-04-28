"""P10-7 (): GDN recurrent-state forward — numpy reference.

Mirrors ``rust/zk_ml_prover/src/proving/gdn_recurrence.rs`` exactly. The
prover's audit-mode binding is "verifier and prover compute the same
trajectory and trajectory digest"; this module is the ground-truth
implementation the proptest harness compares the Rust path against.

Bit-equality requirements:
  * All arithmetic in M31 (uint32 modular).
  * Per-step state shape ``(H, d_k, d_v)``, output shape ``(H, d_v)``.
  * Trajectory digest layout matches ``run_recurrence_with_digest`` in
    Rust: domain tag + (T, H, d_k, d_v) header + ``S_0`` + (t, o_t, S_t)
    per step + footer ``b"end"``.
  * Field-element bytes use the canonical positive uint32 little-endian
    representation (Mersenne31 is `#[repr(transparent)]` over u32 in
    Rust, so the byte stream is just ``arr.astype("<u4").tobytes()``).
"""

from __future__ import annotations

import struct
from typing import Dict, List

import numpy as np

import blake3

from m31 import M31, add, mul, sub


DIGEST_DST = b"gdn_recurrence|v1"


def _absorb_field_slice(hasher: "blake3.blake3", tag: bytes, xs: np.ndarray) -> None:
    """Mirror ``absorb_field_slice`` in Rust: tag + u64-LE length + raw u32-LE bytes."""
    hasher.update(tag)
    hasher.update(struct.pack("<Q", xs.size))
    # Mersenne31 is `#[repr(transparent)] over u32`; the canonical bytes
    # are just the little-endian uint32 representation. ``xs`` is already
    # uint32 in [0, M31).
    hasher.update(xs.astype("<u4", copy=False).tobytes())


def gdn_step_forward(
    s_prev: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    gate: np.ndarray,
    beta: np.ndarray,
    num_heads: int,
    d_k: int,
    d_v: int,
) -> Dict[str, np.ndarray]:
    """One step of the M31-integer GDN recurrence.

    Implements (per head h):
        kt_s[j]      = Σ_{i'} k[h, i'] * s_prev[h, i', j]
        m_kk[i, j]   = k[h, i] * kt_s[j]
        m_kv[i, j]   = k[h, i] * v[h, j]
        s_t[h, i, j] = gate[h] * (s_prev[h, i, j] - beta[h] * m_kk) + beta[h] * m_kv
        o_t[h, j]    = Σ_i q[h, i] * s_t[h, i, j]

    All ops are uint32 mod M31 (via ``m31.add``, ``m31.mul``, ``m31.sub``).
    """
    s_prev = s_prev.reshape(num_heads, d_k, d_v).astype(np.uint32, copy=False)
    q = q.reshape(num_heads, d_k).astype(np.uint32, copy=False)
    k = k.reshape(num_heads, d_k).astype(np.uint32, copy=False)
    v = v.reshape(num_heads, d_v).astype(np.uint32, copy=False)
    gate = gate.reshape(num_heads).astype(np.uint32, copy=False)
    beta = beta.reshape(num_heads).astype(np.uint32, copy=False)

    s_next = np.zeros_like(s_prev)
    o_t = np.zeros((num_heads, d_v), dtype=np.uint32)

    for h in range(num_heads):
        # kt_s[j] = Σ_{i'} k[h, i'] * s_prev[h, i', j]
        kt_s = np.zeros(d_v, dtype=np.uint32)
        for ip in range(d_k):
            kp = int(k[h, ip])
            for j in range(d_v):
                kt_s[j] = int(
                    add(
                        np.array([kt_s[j]], dtype=np.uint32),
                        mul(np.array([kp], dtype=np.uint32), np.array([s_prev[h, ip, j]], dtype=np.uint32)),
                    )[0]
                )

        for i in range(d_k):
            ki = int(k[h, i])
            qi = int(q[h, i])
            for j in range(d_v):
                m_kk = int(mul(np.array([ki], dtype=np.uint32), np.array([kt_s[j]], dtype=np.uint32))[0])
                m_kv = int(mul(np.array([ki], dtype=np.uint32), np.array([v[h, j]], dtype=np.uint32))[0])
                inner = int(
                    sub(
                        np.array([s_prev[h, i, j]], dtype=np.uint32),
                        mul(np.array([beta[h]], dtype=np.uint32), np.array([m_kk], dtype=np.uint32)),
                    )[0]
                )
                s_new = int(
                    add(
                        mul(np.array([gate[h]], dtype=np.uint32), np.array([inner], dtype=np.uint32)),
                        mul(np.array([beta[h]], dtype=np.uint32), np.array([m_kv], dtype=np.uint32)),
                    )[0]
                )
                s_next[h, i, j] = s_new
                o_t[h, j] = int(
                    add(
                        np.array([o_t[h, j]], dtype=np.uint32),
                        mul(np.array([qi], dtype=np.uint32), np.array([s_new], dtype=np.uint32)),
                    )[0]
                )

    return {"s_next": s_next.reshape(-1), "o_t": o_t.reshape(-1)}


def gdn_recurrence_forward(
    initial_state: np.ndarray,
    steps: List[Dict[str, np.ndarray]],
    config: Dict[str, int],
) -> Dict[str, object]:
    """Run the audit-mode GDN recurrence for T = ``len(steps)`` tokens.

    Args:
        initial_state: 1-D uint32 array of length ``H * d_k * d_v``.
        steps: list of dicts; each must contain keys
            ``q`` (H*d_k), ``k`` (H*d_k), ``v`` (H*d_v),
            ``gate`` (H), ``beta`` (H). All uint32 in [0, M31).
        config: dict with ``num_heads``, ``d_k``, ``d_v``.

    Returns:
        A dict with keys:
          * ``o_seq``      — list of T uint32 arrays of length H*d_v
          * ``states``     — list of T+1 uint32 arrays of length H*d_k*d_v
          * ``s_final``    — uint32 array (last state, length H*d_k*d_v)
          * ``trajectory_digest`` — 32-byte hex string (matches Rust hex)
    """
    num_heads = int(config["num_heads"])
    d_k = int(config["d_k"])
    d_v = int(config["d_v"])
    state_size = num_heads * d_k * d_v
    output_size = num_heads * d_v

    initial_state = np.asarray(initial_state, dtype=np.uint32)
    if initial_state.size != state_size:
        raise ValueError(
            f"initial_state length {initial_state.size} != H*d_k*d_v {state_size}"
        )

    hasher = blake3.blake3()
    hasher.update(DIGEST_DST)
    hasher.update(struct.pack("<Q", len(steps)))
    hasher.update(struct.pack("<Q", num_heads))
    hasher.update(struct.pack("<Q", d_k))
    hasher.update(struct.pack("<Q", d_v))
    _absorb_field_slice(hasher, b"S_0", initial_state)

    states: List[np.ndarray] = [initial_state.copy()]
    o_seq: List[np.ndarray] = []
    s_prev = initial_state.copy()

    for t, step in enumerate(steps):
        q = np.asarray(step["q"], dtype=np.uint32)
        k = np.asarray(step["k"], dtype=np.uint32)
        v = np.asarray(step["v"], dtype=np.uint32)
        gate = np.asarray(step["gate"], dtype=np.uint32)
        beta = np.asarray(step["beta"], dtype=np.uint32)

        out = gdn_step_forward(s_prev, q, k, v, gate, beta, num_heads, d_k, d_v)
        s_next = out["s_next"]
        o_t = out["o_t"]
        if s_next.size != state_size or o_t.size != output_size:
            raise AssertionError("step shape mismatch")

        hasher.update(struct.pack("<Q", t))
        _absorb_field_slice(hasher, b"o_t", o_t)
        _absorb_field_slice(hasher, b"S_t", s_next)

        states.append(s_next.copy())
        o_seq.append(o_t)
        s_prev = s_next

    hasher.update(b"end")
    hasher.update(struct.pack("<Q", len(steps)))
    hasher.update(struct.pack("<Q", num_heads))
    hasher.update(struct.pack("<Q", d_k))
    hasher.update(struct.pack("<Q", d_v))

    digest = hasher.digest()
    return {
        "o_seq": o_seq,
        "states": states,
        "s_final": s_prev,
        "trajectory_digest": digest.hex(),
    }
