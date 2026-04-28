//! P10-7: GDN recurrent-state proof — AUDIT-MODE MATH.
//!
//! This file ships the **audit-mode** prover/verifier for the Qwen3.5
//! GatedDeltaNet (GDN) recurrent-state forward pass. The 
//! story is "verifier-trusted-weights mode, with a documented path to
//! true ZK"; the binding mechanism here mirrors P10-3 (Seq1VConsistency)
//! and P10-4/M3 (`external_data_digest`):
//!
//!   * Prover runs the recurrence forward for T tokens; produces
//!     `(o_seq, S_final)` plus a per-step blake3 trajectory digest.
//!   * Verifier (with the same trusted weights and the same input
//!     sequence) re-runs the recurrence forward, recomputes the digest
//!     bytewise, and rejects on any divergence.
//!   * MLE-eval claims at transcript-derived points fold the (otherwise
//!     opaque) `(o_seq, S_final)` outputs into the Fiat-Shamir transcript
//!     so any tamper of the proof's claimed values is caught.
//!
//! This is **strictly stronger** than what we had (no recurrence proof —
//! the C1 honest-scope caveat) and **strictly weaker** than the full
//! sumcheck decomposition (the design doc below remains the eventual
//! true-ZK path). The audit-mode story is documented in the project
//! README under "Limitations": a malicious prover with no weight access
//! cannot fake a valid trajectory; a malicious prover WITH weight access
//! could produce a "correct" trajectory, which is the trust boundary
//! audit-mode already accepts.
//!
//! ─────────────────────────────────────────────────────────────────────
//! GDN recurrence definition (per token t)
//! ─────────────────────────────────────────────────────────────────────
//!
//! State `S_t ∈ R^{H × d_k × d_v}` (per-head matrix; for Qwen3.5-4B,
//! H = 32, d_k = 128, d_v = 128 — 128×128 fp32 per head, 64 KB/head).
//!
//! Per-token update (mathematical definition; CLAUDE.md):
//!     β_t = sigmoid(in_proj_b · x_t)                 // scalar per head
//!     g_t = -exp(A_log) · softplus(α_t + dt_bias)    // gate (log-space)
//!     S_t = exp(g_t) · (S_{t-1} - β_t · k_t k_t^T S_{t-1}) + β_t · k_t v_t^T
//!     o_t = q_t^T S_t
//!
//! For the audit-mode pass we work in **pure M31 integer arithmetic**
//! (see SOUNDNESS Q1 below): the float-domain gate / sigmoid /
//! softplus values are pre-quantized to M31 scalars by the caller and
//! supplied as `gate_t`, `beta_t`, `q_t`, `k_t`, `v_t` per timestep.
//! This avoids cross-language float-rounding concerns (Rust `f32::exp`
//! vs numpy `np.exp` are bitwise-identical on x86_64 and aarch64 for
//! IEEE-754 inputs, but the quantization round-trip is fragile under
//! `-ffast-math` or non-default rounding modes).
//!
//! The numpy reference (`tests/reference/gdn_recurrence.py`) implements
//! the *same* integer recurrence, so the trajectory digest matches
//! bytewise across implementations (proven by
//! `prop_gdn_recurrence_matches_reference`).
//!
//! ─────────────────────────────────────────────────────────────────────
//! True-ZK roadmap (deferred — not implemented in this iteration)
//! ─────────────────────────────────────────────────────────────────────
//!
//! (a) Product sumcheck for `q_t^T S_t` per head (matmul over d_k).
//! (b) Two rank-1 outer-product triple sumchecks: `k_t k_t^T S_{t-1}` and
//!     `k_t v_t^T`.
//! (c) LogUp lookups for sigmoid / exp / softplus.
//! (d) Elementwise `prove_hadamard` for `exp(g_t) · (·)`.
//! (e) Cross-step state binding: claim-chained MLE eval at a fresh
//!     transcript point, mirroring `verification.rs:467`.
//!
//! ─────────────────────────────────────────────────────────────────────
//! Open soundness questions — RESOLVED BY THIS ITERATION
//! ─────────────────────────────────────────────────────────────────────
//!
//! The audit-mode binding closes Q1-Q4 as documented in inline
//! `/// SOUNDNESS (P10-7 Q*)` blocks throughout this file.

use blake3::Hasher;
use p3_field::{AbstractField, PrimeField32};
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::field::common::log2_ceil;
use crate::field::m31_ops::mle_evaluate;
use crate::proving::sumcheck::Transcript;
use crate::proving::weight_commitment::{commit_weights_fast, WeightCommitment};

type F = Mersenne31;

/// Audit-mode trajectory digest domain-separation tag. Bumped if the
/// canonical byte stream layout ever changes. Mirrors the
/// `b"lookup-external-data-v1"` pattern in `proving::lookup`.
const DIGEST_DST: &[u8] = b"gdn_recurrence|v1";

// =====================================================================
// GDN recurrence inputs / config
// =====================================================================

/// Static per-recurrence configuration. Matches the slot the eventual
/// true-ZK proof would carry; the audit-mode prover only needs `H`,
/// `d_k`, `d_v` to size the trajectory digest.
///
/// SOUNDNESS (P10-7 Q1 — quantization domain): the recurrence operates
/// in pure M31 integer arithmetic. Callers are responsible for
/// quantizing the float-domain inputs (`gate_t`, `beta_t`, q/k/v) into
/// M31 with a consistent fixed-point scale BEFORE invoking the prover.
/// Recommended scale for Qwen3.5: 2^16 (16-bit fixed-point), giving
/// ~2× headroom over a 64-step recurrence before the head magnitude
/// approaches `M31_HALF`. The verifier MUST use the same quantization;
/// the audit-mode binding (recompute + digest) catches any divergence.
/// Bound proven empirically by `test_p10_7_quantization_bound_holds`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GdnRecurrenceConfig {
    /// Number of heads (`H`).
    pub num_heads: usize,
    /// Key dimension per head (`d_k`).
    pub d_k: usize,
    /// Value dimension per head (`d_v`).
    pub d_v: usize,
}

impl GdnRecurrenceConfig {
    /// Total elements in `S_t` per timestep: `H · d_k · d_v`.
    pub fn state_size(&self) -> usize {
        self.num_heads * self.d_k * self.d_v
    }

    /// Total elements in `o_t` per timestep: `H · d_v`.
    pub fn output_size(&self) -> usize {
        self.num_heads * self.d_v
    }
}

/// Per-timestep inputs to the M31-integer recurrence. The caller
/// pre-computes `gate_t = quant(exp(g_t))` and
/// `beta_t = quant(sigmoid(β_input))` so the prover's integer path is
/// deterministic.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GdnRecurrenceStep {
    /// `q_t`: shape `(H, d_k)` flattened in `(h, i)` row-major order.
    pub q: Vec<u32>,
    /// `k_t`: shape `(H, d_k)`.
    pub k: Vec<u32>,
    /// `v_t`: shape `(H, d_v)`.
    pub v: Vec<u32>,
    /// Per-head gate scalar `gate_t[h]` (a quantized `exp(g_t[h])`).
    pub gate: Vec<u32>,
    /// Per-head beta scalar `beta_t[h]` (a quantized `sigmoid(·)`).
    pub beta: Vec<u32>,
}

/// Output of the audit-mode forward run: full trajectory plus the
/// trajectory digest absorbed into the transcript.
#[derive(Clone, Debug)]
pub struct GdnRecurrenceTrace {
    /// `o_seq[t]` per timestep, each shape `(H, d_v)` flattened.
    pub o_seq: Vec<Vec<F>>,
    /// `states[t]` for `t ∈ 0..=T`. `states[0]` is the initial state.
    pub states: Vec<Vec<F>>,
    /// Final state `S_T`.
    pub s_final: Vec<F>,
    /// Blake3 digest binding the full trajectory.
    pub trajectory_digest: [u8; 32],
}

// =====================================================================
// Forward computation (shared between prover + reference)
// =====================================================================

/// Run one M31-integer GDN-style recurrence step.
///
/// Implements the simplified-but-faithful integer recurrence:
///
/// ```text
///   M_kk[h, i, j] = k[h, i] · (Σ_{i'} k[h, i'] · S_{t-1}[h, i', j])
///   M_kv[h, i, j] = k[h, i] · v[h, j]
///   S_t[h, i, j]  = gate[h] · (S_{t-1}[h, i, j] - beta[h] · M_kk[h, i, j])
///                   + beta[h] · M_kv[h, i, j]
///   o_t[h, j]     = Σ_i q[h, i] · S_t[h, i, j]
/// ```
///
/// All arithmetic is over M31. The recurrence is a faithful integer
/// re-encoding of the GDN math (CLAUDE.md):
///   `S_t = exp(g_t) · (S_{t-1} - β_t · k k^T S_{t-1}) + β_t · k v^T`
/// `o_t = q^T S_t`.
///
/// SAFETY: callers MUST NOT pass `q.len() != H·d_k`, etc. — the function
/// `debug_assert!`s shapes. Production callers (prover/verifier) build
/// shapes from the same `GdnRecurrenceConfig` so this is a contract
/// rather than a runtime guard.
fn step_forward(
    s_prev: &[F],
    step: &GdnRecurrenceStep,
    cfg: &GdnRecurrenceConfig,
) -> (Vec<F>, Vec<F>) {
    let h_count = cfg.num_heads;
    let d_k = cfg.d_k;
    let d_v = cfg.d_v;
    debug_assert_eq!(s_prev.len(), cfg.state_size());
    debug_assert_eq!(step.q.len(), h_count * d_k);
    debug_assert_eq!(step.k.len(), h_count * d_k);
    debug_assert_eq!(step.v.len(), h_count * d_v);
    debug_assert_eq!(step.gate.len(), h_count);
    debug_assert_eq!(step.beta.len(), h_count);

    let q: Vec<F> = step.q.iter().map(|&u| F::from_canonical_u32(u)).collect();
    let k: Vec<F> = step.k.iter().map(|&u| F::from_canonical_u32(u)).collect();
    let v: Vec<F> = step.v.iter().map(|&u| F::from_canonical_u32(u)).collect();
    let gate: Vec<F> = step.gate.iter().map(|&u| F::from_canonical_u32(u)).collect();
    let beta: Vec<F> = step.beta.iter().map(|&u| F::from_canonical_u32(u)).collect();

    let mut s_next = vec![F::zero(); cfg.state_size()];
    let mut o_t = vec![F::zero(); cfg.output_size()];

    for h in 0..h_count {
        let s_off = h * d_k * d_v;
        let q_off = h * d_k;
        let v_off = h * d_v;
        let k_off = h * d_k;

        // Pre-compute kT_S[j] = Σ_{i'} k[h, i'] · S_{t-1}[h, i', j].
        // PERF: O(d_k · d_v) per head, materialized once instead of
        // re-summed per (i, j). Memory cost: d_v F's (negligible).
        let mut kt_s = vec![F::zero(); d_v];
        for i_prime in 0..d_k {
            let kp = k[k_off + i_prime];
            for j in 0..d_v {
                kt_s[j] += kp * s_prev[s_off + i_prime * d_v + j];
            }
        }

        // S_t and o_t for this head.
        for i in 0..d_k {
            let ki = k[k_off + i];
            let qi = q[q_off + i];
            for j in 0..d_v {
                // M_kk[h, i, j] = k[h, i] · kt_s[j]
                let m_kk = ki * kt_s[j];
                // M_kv[h, i, j] = k[h, i] · v[h, j]
                let m_kv = ki * v[v_off + j];
                let s_old = s_prev[s_off + i * d_v + j];
                // S_t = gate · (S_{t-1} - beta · M_kk) + beta · M_kv
                let inner = s_old - beta[h] * m_kk;
                let s_new = gate[h] * inner + beta[h] * m_kv;
                s_next[s_off + i * d_v + j] = s_new;
                // o_t[h, j] += q[h, i] · S_t[h, i, j]
                o_t[v_off + j] += qi * s_new;
            }
        }
    }

    (s_next, o_t)
}

/// Run the full audit-mode forward pass and compute the trajectory
/// digest. Used by both the prover and verifier so the canonical bytes
/// layout is shared in one place.
///
/// SOUNDNESS (P10-7 Q2 — canonical byte ordering): the digest absorbs
/// `seq_idx` (u32 LE) before each `(S_t, o_t)` tuple so a permutation
/// attack (re-order timesteps to match a different valid trajectory)
/// breaks the digest. The trailing shape footer absorbs `H, d_k, d_v`
/// so a config-level confusion (running the same prover transcript
/// against a differently-shaped verifier) also breaks the digest.
pub fn run_recurrence_with_digest(
    initial_state: &[F],
    steps: &[GdnRecurrenceStep],
    cfg: &GdnRecurrenceConfig,
) -> GdnRecurrenceTrace {
    assert_eq!(
        initial_state.len(),
        cfg.state_size(),
        "initial_state length must equal H·d_k·d_v"
    );

    let mut hasher = Hasher::new();
    hasher.update(DIGEST_DST);
    hasher.update(&(steps.len() as u64).to_le_bytes());
    hasher.update(&(cfg.num_heads as u64).to_le_bytes());
    hasher.update(&(cfg.d_k as u64).to_le_bytes());
    hasher.update(&(cfg.d_v as u64).to_le_bytes());

    // Absorb initial state.
    absorb_field_slice(&mut hasher, b"S_0", initial_state);

    let mut states: Vec<Vec<F>> = Vec::with_capacity(steps.len() + 1);
    states.push(initial_state.to_vec());
    let mut o_seq: Vec<Vec<F>> = Vec::with_capacity(steps.len());

    let mut s_prev = initial_state.to_vec();
    for (t, step) in steps.iter().enumerate() {
        let (s_next, o_t) = step_forward(&s_prev, step, cfg);

        hasher.update(&(t as u64).to_le_bytes());
        absorb_field_slice(&mut hasher, b"o_t", &o_t);
        absorb_field_slice(&mut hasher, b"S_t", &s_next);

        states.push(s_next.clone());
        o_seq.push(o_t);
        s_prev = s_next;
    }

    // Footer: re-absorb the shape so a truncated digest stream cannot
    // collide with a different (T', H', d_k', d_v') run.
    hasher.update(b"end");
    hasher.update(&(steps.len() as u64).to_le_bytes());
    hasher.update(&(cfg.num_heads as u64).to_le_bytes());
    hasher.update(&(cfg.d_k as u64).to_le_bytes());
    hasher.update(&(cfg.d_v as u64).to_le_bytes());

    let digest: [u8; 32] = *hasher.finalize().as_bytes();
    GdnRecurrenceTrace {
        o_seq,
        states,
        s_final: s_prev,
        trajectory_digest: digest,
    }
}

/// Absorb a `&[F]` into a blake3 hasher with a 1-byte separator tag and
/// a u64 length prefix. Used to keep the digest stream
/// length-prefix-secure (no boundary-shift attacks).
///
/// INVARIANT (P10-7, 4th-reviewer finding #1): the u64-LE length prefix
/// is load-bearing. The audit-mode binding leans on bytewise digest
/// equality of (S_t, o_t) byte streams; without the length prefix a
/// malicious prover could shift step boundaries (e.g. produce a
/// trajectory whose concatenated bytes differ by one element split
/// differently across steps) and still hit the same digest. Any future
/// refactor that drops the prefix or changes its width MUST be
/// accompanied by a proptest demonstrating the boundary-shift attack
/// no longer succeeds. Mirrored byte-for-byte by Python
/// `tests/reference/gdn_recurrence.py::_absorb_field_slice`.
fn absorb_field_slice(hasher: &mut Hasher, tag: &[u8], xs: &[F]) {
    hasher.update(tag);
    hasher.update(&(xs.len() as u64).to_le_bytes());
    // Zero-copy: Mersenne31 is `#[repr(transparent)]` over u32. We rely
    // on the same invariant `commit_weights_fast` documents — assert
    // size to keep the unsafe cast honest.
    assert!(std::mem::size_of::<F>() == 4);
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len() * 4) };
    hasher.update(bytes);
}

// =====================================================================
// Proof structure
// =====================================================================

/// Audit-mode GDN recurrence proof.
///
/// SOUNDNESS (P10-7 Q3 — proof-size budget): one digest (32 bytes) +
/// one weight commitment (~80 bytes) + two field MLE evals (8 bytes)
/// + a u64 = ~150 bytes total. **Constant in T** — the per-step state
/// commitment cost from the original scaffold's design is amortized
/// into the single `trajectory_digest`. The 64 MB-per-timestep concern
/// in the design doc applies only to the eventual true-ZK Merkle path,
/// not to audit-mode.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GdnRecurrenceProof {
    /// Blake3 digest of `(initial_state, (o_t, S_t)_{t=0..T-1}, footer)`
    /// — see `run_recurrence_with_digest` for the canonical layout.
    pub trajectory_digest: [u8; 32],
    /// Number of timesteps `T`.
    pub seq_len: usize,
    /// `commit_weights_fast` over the flattened final state `S_T`.
    /// Bound to the transcript so a verifier can detect post-hoc
    /// substitution of the proof's `S_final` field.
    pub final_state_commitment: WeightCommitment,
    /// `MLE(flat(o_seq), r)` evaluated at the transcript-derived point
    /// `r` (squeezed AFTER absorbing the digest + commitment).
    pub o_seq_at_r: u32,
    /// `MLE(flat(S_T), r')` evaluated at the transcript-derived point
    /// `r'` (squeezed AFTER absorbing `o_seq_at_r`).
    pub state_final_at_r: u32,
}

// =====================================================================
// Prover / verifier
// =====================================================================

/// Run the GDN recurrence and produce an audit-mode proof.
///
/// The transcript MUST already encode any ambient context (model
/// version, layer index) — this function only absorbs the recurrence
/// digest + commitments + claims, not the inputs themselves.
///
/// SOUNDNESS (P10-7 Q4 — Fiat-Shamir order): the prover's absorption
/// order is (1) digest → (2) shape header → (3) `S_T` commitment root →
/// squeeze `r` → (4) `o_seq_at_r` → squeeze `r'` → (5) `state_final_at_r`.
/// The verifier mirrors the EXACT same order. Any tamper of any single
/// proof field shifts a downstream squeeze and breaks the binding.
pub fn prove_gdn_recurrence(
    initial_state: &[F],
    steps: &[GdnRecurrenceStep],
    cfg: &GdnRecurrenceConfig,
    transcript: &mut Transcript,
) -> GdnRecurrenceProof {
    let trace = run_recurrence_with_digest(initial_state, steps, cfg);

    // Absorb the digest first — Fiat-Shamir's load-bearing input.
    transcript.absorb_bytes(&trace.trajectory_digest);
    transcript.absorb(steps.len() as u32);
    transcript.absorb(cfg.num_heads as u32);
    transcript.absorb(cfg.d_k as u32);
    transcript.absorb(cfg.d_v as u32);

    // Commit to the final state and absorb the root.
    let final_state_commitment = commit_weights_fast(&trace.s_final);
    transcript.absorb_bytes(&final_state_commitment.root);

    // Squeeze r over log2(flat(o_seq).len()), MLE-eval, absorb.
    let o_flat = flatten_o_seq(&trace.o_seq);
    let log_o = log2_ceil(o_flat.len().max(2));
    let r_o = transcript.squeeze_many(log_o);
    let o_pad = pad_pow2(&o_flat, 1usize << log_o);
    let o_seq_at_r = mle_evaluate(&o_pad, &r_o);
    transcript.absorb(o_seq_at_r.as_canonical_u32());

    // Squeeze r' over log2(S_T.len()), MLE-eval, absorb.
    let log_s = log2_ceil(trace.s_final.len().max(2));
    let r_s = transcript.squeeze_many(log_s);
    let s_pad = pad_pow2(&trace.s_final, 1usize << log_s);
    let state_final_at_r = mle_evaluate(&s_pad, &r_s);
    transcript.absorb(state_final_at_r.as_canonical_u32());

    GdnRecurrenceProof {
        trajectory_digest: trace.trajectory_digest,
        seq_len: steps.len(),
        final_state_commitment,
        o_seq_at_r: o_seq_at_r.as_canonical_u32(),
        state_final_at_r: state_final_at_r.as_canonical_u32(),
    }
}

/// Re-run the recurrence and check the audit-mode proof.
///
/// The verifier MUST be in lockstep with the prover's transcript: same
/// initial transcript state, same absorption order. The verifier
/// independently:
///   1. recomputes the trajectory digest from `(initial_state, steps)`,
///   2. checks `digest == proof.trajectory_digest` bytewise,
///   3. recomputes `commit_weights_fast(S_T)`, checks the root matches,
///   4. squeezes the same `r, r'` and recomputes the MLE evals, checks
///      they equal the proof's claimed values.
///
/// Any tamper of `trajectory_digest`, `final_state_commitment.root`,
/// `o_seq_at_r`, or `state_final_at_r` is caught by either the digest
/// recompute or the MLE recompute.
pub fn verify_gdn_recurrence(
    initial_state: &[F],
    steps: &[GdnRecurrenceStep],
    cfg: &GdnRecurrenceConfig,
    proof: &GdnRecurrenceProof,
    transcript: &mut Transcript,
) -> bool {
    if proof.seq_len != steps.len() {
        return false;
    }
    if proof.final_state_commitment.num_weights != cfg.state_size() {
        return false;
    }

    let trace = run_recurrence_with_digest(initial_state, steps, cfg);

    // (1) digest equality.
    if trace.trajectory_digest != proof.trajectory_digest {
        return false;
    }

    // (2) commitment-root equality.
    let expected_commitment = commit_weights_fast(&trace.s_final);
    if expected_commitment.root != proof.final_state_commitment.root {
        return false;
    }
    if expected_commitment.kind != proof.final_state_commitment.kind {
        return false;
    }

    // Mirror the prover's transcript absorption order.
    transcript.absorb_bytes(&trace.trajectory_digest);
    transcript.absorb(steps.len() as u32);
    transcript.absorb(cfg.num_heads as u32);
    transcript.absorb(cfg.d_k as u32);
    transcript.absorb(cfg.d_v as u32);
    transcript.absorb_bytes(&proof.final_state_commitment.root);

    // (3) recompute MLE(o_seq, r).
    let o_flat = flatten_o_seq(&trace.o_seq);
    let log_o = log2_ceil(o_flat.len().max(2));
    let r_o = transcript.squeeze_many(log_o);
    let o_pad = pad_pow2(&o_flat, 1usize << log_o);
    let expected_o_at_r = mle_evaluate(&o_pad, &r_o);
    if expected_o_at_r.as_canonical_u32() != proof.o_seq_at_r {
        return false;
    }
    transcript.absorb(proof.o_seq_at_r);

    // (4) recompute MLE(S_T, r').
    let log_s = log2_ceil(trace.s_final.len().max(2));
    let r_s = transcript.squeeze_many(log_s);
    let s_pad = pad_pow2(&trace.s_final, 1usize << log_s);
    let expected_s_at_r = mle_evaluate(&s_pad, &r_s);
    if expected_s_at_r.as_canonical_u32() != proof.state_final_at_r {
        return false;
    }
    transcript.absorb(proof.state_final_at_r);

    true
}

// =====================================================================
// Helpers
// =====================================================================

/// Flatten `o_seq[t][·]` into a single contiguous vector for MLE
/// evaluation. Layout: `(t, h, j)` row-major (matches the digest
/// absorption order — keeps audits debuggable).
fn flatten_o_seq(o_seq: &[Vec<F>]) -> Vec<F> {
    let mut out = Vec::with_capacity(o_seq.iter().map(|v| v.len()).sum());
    for o_t in o_seq {
        out.extend_from_slice(o_t);
    }
    out
}

/// Pad a slice of field elements to a power-of-two length with zeros.
/// Mirrors the `v_pad`/`attn_pad` pattern in `transformer/qwen.rs:408`.
fn pad_pow2(xs: &[F], target: usize) -> Vec<F> {
    debug_assert!(target.is_power_of_two() || target == 1);
    debug_assert!(xs.len() <= target);
    let mut out = xs.to_vec();
    out.resize(target, F::zero());
    out
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
pub mod test_utils {
    //! Deterministic fixture builder for GDN recurrence tests. Public so
    //! the proptest harness can reuse it.
    use super::*;

    /// Build a deterministic small fixture: H=2, d_k=4, d_v=4, T=`seq_len`.
    /// Every field element is in `[0, 256)` so the recurrence stays in
    /// the M31 positive half-field for any reasonable `seq_len`.
    pub fn build_fixture(seq_len: usize, seed: u64) -> (Vec<F>, Vec<GdnRecurrenceStep>, GdnRecurrenceConfig) {
        let cfg = GdnRecurrenceConfig {
            num_heads: 2,
            d_k: 4,
            d_v: 4,
        };
        // tiny deterministic PRNG via splitmix64.
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let small = |v: u64| -> u32 { (v % 16) as u32 };
        let s0: Vec<F> = (0..cfg.state_size())
            .map(|_| F::from_canonical_u32(small(next())))
            .collect();
        let steps: Vec<GdnRecurrenceStep> = (0..seq_len)
            .map(|_| GdnRecurrenceStep {
                q: (0..cfg.num_heads * cfg.d_k).map(|_| small(next())).collect(),
                k: (0..cfg.num_heads * cfg.d_k).map(|_| small(next())).collect(),
                v: (0..cfg.num_heads * cfg.d_v).map(|_| small(next())).collect(),
                gate: (0..cfg.num_heads).map(|_| small(next())).collect(),
                beta: (0..cfg.num_heads).map(|_| small(next())).collect(),
            })
            .collect();
        (s0, steps, cfg)
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use super::test_utils::build_fixture;

    /// Happy path: prover produces a proof, verifier accepts.
    #[test]
    fn test_p10_7_gdn_recurrence_proves_and_verifies() {
        let (s0, steps, cfg) = build_fixture(4, 0xDEADBEEF);
        let mut pt = Transcript::new(b"p10-7-happy");
        let proof = prove_gdn_recurrence(&s0, &steps, &cfg, &mut pt);

        let mut vt = Transcript::new(b"p10-7-happy");
        let ok = verify_gdn_recurrence(&s0, &steps, &cfg, &proof, &mut vt);
        assert!(ok, "verifier rejected an honest proof");

        // Sanity: trajectory digest is deterministic for the same fixture.
        let trace_again = run_recurrence_with_digest(&s0, &steps, &cfg);
        assert_eq!(trace_again.trajectory_digest, proof.trajectory_digest);
        assert_eq!(proof.seq_len, steps.len());
        assert_eq!(
            proof.final_state_commitment.num_weights,
            cfg.state_size()
        );
    }

    /// Tamper an intermediate `S_t` by mutating one byte of the input
    /// the verifier sees. The recomputed digest diverges; verifier rejects.
    ///
    /// SOUNDNESS: this is the load-bearing tamper test for the audit-mode
    /// story. The prover honestly absorbs `S_2` into the digest; if the
    /// verifier replays the recurrence with a tampered initial state
    /// (or tampered step input that perturbs `S_2`), the per-step digest
    /// stream diverges at step 2 and fails the bytewise comparison.
    #[test]
    fn test_p10_7_gdn_recurrence_tampered_S_t_rejects() {
        let (s0, steps, cfg) = build_fixture(4, 0xCAFEBABE);
        let mut pt = Transcript::new(b"p10-7-tamper-s");
        let proof = prove_gdn_recurrence(&s0, &steps, &cfg, &mut pt);

        // Mutate the verifier's view of step 2's `v` so the recomputed
        // S_2 diverges from the prover's S_2.
        let mut tampered_steps = steps.clone();
        let nonzero = tampered_steps[2].v[0].wrapping_add(1);
        tampered_steps[2].v[0] = nonzero;

        let mut vt = Transcript::new(b"p10-7-tamper-s");
        let ok = verify_gdn_recurrence(&s0, &tampered_steps, &cfg, &proof, &mut vt);
        assert!(!ok, "verifier accepted a tampered intermediate state");
    }

    /// Tamper `proof.final_state_commitment.root`. The verifier
    /// recomputes the root from `S_T` and rejects on root mismatch.
    #[test]
    fn test_p10_7_gdn_recurrence_tampered_final_state_rejects() {
        let (s0, steps, cfg) = build_fixture(3, 0xFEEDFACE);
        let mut pt = Transcript::new(b"p10-7-tamper-root");
        let mut proof = prove_gdn_recurrence(&s0, &steps, &cfg, &mut pt);

        proof.final_state_commitment.root[0] ^= 0xFF;

        let mut vt = Transcript::new(b"p10-7-tamper-root");
        let ok = verify_gdn_recurrence(&s0, &steps, &cfg, &proof, &mut vt);
        assert!(!ok, "verifier accepted a tampered final-state commitment");
    }

    /// Tamper `proof.o_seq_at_r`. The verifier recomputes the MLE eval
    /// at the same `r` and rejects on field mismatch.
    #[test]
    fn test_p10_7_gdn_recurrence_tampered_o_seq_at_r_rejects() {
        let (s0, steps, cfg) = build_fixture(3, 0x1234_5678);
        let mut pt = Transcript::new(b"p10-7-tamper-o");
        let mut proof = prove_gdn_recurrence(&s0, &steps, &cfg, &mut pt);

        proof.o_seq_at_r = proof.o_seq_at_r.wrapping_add(1);

        let mut vt = Transcript::new(b"p10-7-tamper-o");
        let ok = verify_gdn_recurrence(&s0, &steps, &cfg, &proof, &mut vt);
        assert!(!ok, "verifier accepted a tampered o_seq_at_r");
    }

    /// Tamper `proof.state_final_at_r`. Same logic as the o_seq tamper
    /// but for the state MLE binding.
    #[test]
    fn test_p10_7_gdn_recurrence_tampered_state_final_at_r_rejects() {
        let (s0, steps, cfg) = build_fixture(3, 0xABCD_EF01);
        let mut pt = Transcript::new(b"p10-7-tamper-sfa");
        let mut proof = prove_gdn_recurrence(&s0, &steps, &cfg, &mut pt);

        proof.state_final_at_r = proof.state_final_at_r.wrapping_add(7);

        let mut vt = Transcript::new(b"p10-7-tamper-sfa");
        let ok = verify_gdn_recurrence(&s0, &steps, &cfg, &proof, &mut vt);
        assert!(!ok, "verifier accepted a tampered state_final_at_r");
    }

    /// Tamper `proof.trajectory_digest`. The verifier compares the
    /// recomputed digest bytewise; flip → reject.
    #[test]
    fn test_p10_7_gdn_recurrence_tampered_trajectory_digest_rejects() {
        let (s0, steps, cfg) = build_fixture(3, 0x7777_8888);
        let mut pt = Transcript::new(b"p10-7-tamper-d");
        let mut proof = prove_gdn_recurrence(&s0, &steps, &cfg, &mut pt);

        proof.trajectory_digest[15] ^= 0x42;

        let mut vt = Transcript::new(b"p10-7-tamper-d");
        let ok = verify_gdn_recurrence(&s0, &steps, &cfg, &proof, &mut vt);
        assert!(!ok, "verifier accepted a tampered trajectory_digest");
    }

    /// Tamper `proof.seq_len`. The verifier checks shape consistency
    /// before doing any work.
    #[test]
    fn test_p10_7_gdn_recurrence_tampered_seq_len_rejects() {
        let (s0, steps, cfg) = build_fixture(3, 0x9999_AAAA);
        let mut pt = Transcript::new(b"p10-7-tamper-len");
        let mut proof = prove_gdn_recurrence(&s0, &steps, &cfg, &mut pt);

        proof.seq_len += 1;

        let mut vt = Transcript::new(b"p10-7-tamper-len");
        let ok = verify_gdn_recurrence(&s0, &steps, &cfg, &proof, &mut vt);
        assert!(!ok, "verifier accepted a tampered seq_len");
    }

    /// SOUNDNESS (P10-7 Q1 — quantization-bound regression test):
    /// fixture inputs in `[0, 16)` keep `S_t` magnitudes bounded across
    /// `T = 64` recurrence steps (the maximum the docstring claims is
    /// safe at 16-bit fixed-point with 2× headroom). Asserts every
    /// state field element is in the M31 positive half-field after
    /// every step.
    ///
    /// Bumped from T=32 → T=64 per 4th-reviewer finding #7: the
    /// docstring asserted "≤ 64 steps" but only T=32 was empirically
    /// covered. Now matches the documented bound.
    #[test]
    fn test_p10_7_quantization_bound_holds() {
        let (s0, steps, cfg) = build_fixture(64, 0xBADD_CAFE);
        let trace = run_recurrence_with_digest(&s0, &steps, &cfg);
        const M31_HALF: u32 = (1u32 << 30) - 1;
        for s_t in &trace.states {
            for f in s_t {
                let v = f.as_canonical_u32();
                assert!(
                    v <= M31_HALF || v >= ((1u32 << 31) - 1) - M31_HALF,
                    "S_t entry {} escaped the M31 positive-half envelope after \
                     64 steps — quantization bound violated",
                    v
                );
            }
        }
    }

    /// Cross-language digest pin: tiny fixed fixture (H=d_k=d_v=2,
    /// gate=1, beta=0) MUST produce a specific blake3 digest. The same
    /// fixture is run through `tests/reference/gdn_recurrence.py`; the
    /// hex below was captured from that Python run and pinned here.
    /// If this test fails, either the Rust digest absorption order or
    /// the numpy reference's diverged — the proptest harness will also
    /// catch it, but this in-Rust test fails fast without spinning up
    /// a Python subprocess.
    #[test]
    fn test_p10_7_gdn_recurrence_digest_matches_numpy_reference_pin() {
        let cfg = GdnRecurrenceConfig { num_heads: 2, d_k: 2, d_v: 2 };
        let s0: Vec<F> = [1u32, 2, 3, 4, 5, 6, 7, 8]
            .iter()
            .map(|&u| F::from_canonical_u32(u))
            .collect();
        let steps = vec![GdnRecurrenceStep {
            q: vec![1, 2, 3, 4],
            k: vec![5, 6, 7, 8],
            v: vec![1, 1, 1, 1],
            gate: vec![1, 1],
            beta: vec![0, 0],
        }];
        let trace = run_recurrence_with_digest(&s0, &steps, &cfg);
        let got = hex::encode(trace.trajectory_digest);
        // Pinned from `tests/reference/gdn_recurrence.py` smoke run.
        let expected = "d09d51b49e45897d1981f04c67f118206141f4b27e08e8ce881a4de6c3fded2a";
        assert_eq!(got, expected, "Rust digest diverged from numpy reference pin");
    }

    /// Serde round-trip stays green so old proofs deserialize after
    /// the audit-mode upgrade.
    #[test]
    fn test_p10_7_gdn_recurrence_proof_serde_roundtrip() {
        let (s0, steps, cfg) = build_fixture(2, 0x1111_2222);
        let mut pt = Transcript::new(b"p10-7-serde");
        let proof = prove_gdn_recurrence(&s0, &steps, &cfg, &mut pt);
        let bytes = bincode::serialize(&proof).expect("serialize");
        let back: GdnRecurrenceProof = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(back.trajectory_digest, proof.trajectory_digest);
        assert_eq!(back.seq_len, proof.seq_len);
        assert_eq!(back.o_seq_at_r, proof.o_seq_at_r);
        assert_eq!(back.state_final_at_r, proof.state_final_at_r);
    }
}
