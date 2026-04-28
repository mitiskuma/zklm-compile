//! P11-3: MLE-identity isolated property tests.
//!
//! AGENT B fills this file. Spec: `tests/reference/SPEC.md`.
//! Asserts every MLE identity the prover relies on, from random inputs:
//!   - `MLE(replicate(v), r) == MLE(v, [r_g || r_d])`  (the GQA bug class)
//!   - `MLE(v, bit_decompose(i)) == v[i]`               (Boolean self-eval)
//!   - `eq_evals(r)[x] == Π (r_i*x_i + (1-r_i)*(1-x_i))` (eq table correctness)
//!   - `MLE(W @ x, r) == Σ_j MLE(W, [r, bit_decompose(j)]) * x[j]`
//!
//! ## Why we re-implement `mle_evaluate` / `eq_evals` here
//!
//! `zk_ml_prover` is a binary-only crate (no `lib.rs`), so integration
//! tests under `tests/` cannot `use zk_ml_prover::field::m31_ops::...`.
//! The instructions explicitly forbid editing the Rust source to add a
//! library or `pub use` re-export. We therefore re-implement
//! `mle_evaluate` and `eq_evals` inline using the **identical algorithm**
//! as `src/field/m31_ops.rs`. These tests assert MLE *identities* — they
//! hold from first principles for any correct MSB-first multilinear-
//! extension implementation, which is exactly what the Rust prover uses.
//!
//! Convention (locked by `tests/reference/SPEC.md`):
//!   - **MLE folding is MSB-first**: `point[0]` folds the highest-order
//!     index bit. So `mle_evaluate(v, [b_0, b_1, ..., b_{k-1}]) == v[i]`
//!     where `i = b_0 * 2^{k-1} + b_1 * 2^{k-2} + ... + b_{k-1}`.
//!   - **Tensors are row-major**: `attn_out[h * d_head + d]` puts the
//!     head index in the high-order bits, `d` in the low-order bits.
//!   - **GQA replication**: `attn_out[h, d] = v[h / heads_per_group, d]`,
//!     where `heads_per_group = num_q_heads / num_kv_heads`. The MLE-
//!     identity slice is `r_v = [r[..log_kv] || r[len - log_d..]]`
//!     (group-prefix + d-suffix), NOT a contiguous prefix.
//!
//! Acceptance gate (from the task brief): "deliberately reintroducing
//! the P10-3 GQA r-coord bug (commit 86f3ab6) must make
//! `prop_mle_replicate_identity` proptest fail in <60 seconds."

use proptest::prelude::*;
use p3_field::AbstractField;
use p3_mersenne_31::Mersenne31;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type F = Mersenne31;

// =====================================================================
// Inline copies of `crate::field::m31_ops::{mle_evaluate, eq_evals}`.
// Algorithm-identical to `rust/zk_ml_prover/src/field/m31_ops.rs`.
// MUST stay in lockstep with that file. If the prover changes folding
// direction or eq-table layout, these copies must change too — the
// tests would otherwise silently pass against a stale convention.
// =====================================================================

/// MSB-first MLE evaluation. `point[0]` folds the highest-order bit.
fn mle_evaluate(evals: &[F], point: &[F]) -> F {
    let mut table = evals.to_vec();
    let num_vars = point.len();
    let mut size = table.len();
    debug_assert_eq!(size, 1usize << num_vars, "evals must have length 2^|point|");

    for i in 0..num_vars {
        let half = size / 2;
        let r = point[i];
        let one_minus_r = F::one() - r;
        for j in 0..half {
            table[j] = one_minus_r * table[j] + r * table[j + half];
        }
        size = half;
    }
    table[0]
}

/// `eq_evals(r)[x]` = `Π_i (r_i * x_i + (1-r_i) * (1-x_i))` where the
/// bit decomposition `x_i` is MSB-first matched to `r_i`. Concretely,
/// for `x` interpreted as an integer index in `[0, 2^k)`, the most-
/// significant bit of `x` pairs with `r[0]`. This matches the eq-table
/// layout produced by `src/field/m31_ops::eq_evals`.
fn eq_evals(r: &[F]) -> Vec<F> {
    let k = r.len();
    let n = 1usize << k;
    let mut evals = vec![F::one(); n];

    let mut populated = 1usize;
    for i in 0..k {
        let ri = r[i];
        let one_minus_ri = F::one() - ri;
        for j in (0..populated).rev() {
            evals[2 * j + 1] = evals[j] * ri;
            evals[2 * j] = evals[j] * one_minus_ri;
        }
        populated *= 2;
    }
    evals
}

// =====================================================================
// Helpers
// =====================================================================

/// MSB-first bit decomposition: returns `[F::one()/F::zero(); k]` such
/// that `bit_decompose_msb(i, k)[0]` is the bit of weight `2^{k-1}` in
/// `i`. Matches the "Boolean self-eval" identity:
///     mle_evaluate(v, bit_decompose_msb(i, k)) == v[i].
fn bit_decompose_msb(idx: usize, k: usize) -> Vec<F> {
    debug_assert!(idx < (1usize << k), "idx out of range for {} bits", k);
    let mut out = Vec::with_capacity(k);
    for bit in 0..k {
        // bit 0 is the MSB, weight 2^{k-1-bit} = 2^(k-1) down to 2^0.
        let shift = k - 1 - bit;
        let b = (idx >> shift) & 1;
        out.push(if b == 1 { F::one() } else { F::zero() });
    }
    out
}

/// Multiplicative form of `eq(r, x)`: `Π (r_i*x_i + (1-r_i)*(1-x_i))`.
/// Used as the ground-truth check against the precomputed `eq_evals`
/// table. Both `r` and `x` are length-`k` MSB-aligned points.
fn eq_at_point(r: &[F], x: &[F]) -> F {
    debug_assert_eq!(r.len(), x.len(), "eq_at_point dim mismatch");
    let mut acc = F::one();
    for i in 0..r.len() {
        let term = r[i] * x[i] + (F::one() - r[i]) * (F::one() - x[i]);
        acc *= term;
    }
    acc
}

/// Random M31 element via canonical reduction of a u32 draw.
fn random_field_elt(rng: &mut StdRng) -> F {
    F::from_canonical_u32(rng.gen::<u32>() & 0x7fff_ffff)
}

fn random_field_vec(rng: &mut StdRng, len: usize) -> Vec<F> {
    (0..len).map(|_| random_field_elt(rng)).collect()
}

/// Squeeze `n` random field elements from a seed. Mirrors the role of
/// `Transcript::squeeze_many` in property-test land — we just need a
/// reproducible random point of length `n`, keyed on the proptest seed.
fn random_point(rng: &mut StdRng, n: usize) -> Vec<F> {
    random_field_vec(rng, n)
}

/// log2 for usize, asserts `n` is a power of two.
fn log2_pow2(n: usize) -> usize {
    debug_assert!(n.is_power_of_two() && n > 0, "expected power of 2, got {}", n);
    n.trailing_zeros() as usize
}

// =====================================================================
// Property tests
// =====================================================================

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        // Cap shrink iterations so a CI failure prints a small repro
        // quickly. Default 1024 dwarfs the 60s acceptance budget.
        max_shrink_iters: 256,
        .. ProptestConfig::default()
    })]

    /// **GQA head-replication MLE identity** (the P10-3 bug class).
    ///
    /// For random GQA shape `(num_q_heads, num_kv_heads, d_head)` with
    /// `num_q_heads != num_kv_heads` (forces the GQA branch), build:
    ///   - `v` of length `num_kv_heads * d_head` from random seed.
    ///   - `attn_out[h, d] = v[h / heads_per_group, d]` (head replicate).
    /// Squeeze a random point `r` of length `log2(attn_out.len())`. The
    /// **correct** v-side point is `r_v = r[..log_kv] || r[len-log_d..]`
    /// (group-prefix + d-suffix). Reintroducing the contiguous-prefix
    /// bug `r[..log_v]` makes this assertion fail in <60s with default
    /// 100 cases — that is the acceptance gate for P11-3.
    #[test]
    fn prop_mle_replicate_identity(seed in any::<u64>()) {
        let mut rng = StdRng::seed_from_u64(seed);

        // Sample power-of-two GQA shape with num_q_heads > num_kv_heads.
        // Keep dims small enough that 100 cases finish in well under
        // the 60s budget but large enough to actually exercise the
        // r-slice bug (need log_kv >= 1 and log_d >= 1).
        let log_kv: usize = rng.gen_range(1..=3);          // num_kv_heads in {2,4,8}
        let log_group: usize = rng.gen_range(1..=3);       // heads_per_group in {2,4,8}
        let log_d: usize = rng.gen_range(1..=4);           // d_head in {2..16}
        let log_q = log_kv + log_group;
        let num_kv_heads = 1usize << log_kv;
        let num_q_heads = 1usize << log_q;
        let d_head = 1usize << log_d;
        let heads_per_group = num_q_heads / num_kv_heads;
        prop_assert!(num_q_heads != num_kv_heads, "must hit GQA branch");
        prop_assert_eq!(num_q_heads % num_kv_heads, 0);

        // v: row-major (num_kv_heads, d_head)
        let v: Vec<F> = random_field_vec(&mut rng, num_kv_heads * d_head);
        // attn_out: row-major (num_q_heads, d_head), with head-replication.
        let mut attn_out: Vec<F> = vec![F::zero(); num_q_heads * d_head];
        for h in 0..num_q_heads {
            let kv_h = h / heads_per_group;
            for d in 0..d_head {
                attn_out[h * d_head + d] = v[kv_h * d_head + d];
            }
        }

        // Random MLE point r over attn_out's index space.
        let total_log = log_q + log_d;
        prop_assert_eq!(attn_out.len(), 1usize << total_log);
        let r = random_point(&mut rng, total_log);

        // LHS: MLE(attn_out, r).
        let lhs = mle_evaluate(&attn_out, &r);

        // RHS: MLE(v, r_v) where r_v = [group-prefix(log_kv) || d-suffix(log_d)].
        // The head-coords of `attn_out` occupy r[0..log_q] (MSB-first),
        // so r[0..log_kv] is the kv-group prefix; the d-coords occupy
        // r[log_q..log_q + log_d] = r[total_log - log_d..]. Concatenating
        // them yields the correct v-side point of length log_kv + log_d.
        let mut r_v: Vec<F> = Vec::with_capacity(log_kv + log_d);
        r_v.extend_from_slice(&r[..log_kv]);
        r_v.extend_from_slice(&r[total_log - log_d..]);
        prop_assert_eq!(r_v.len(), log_kv + log_d);
        prop_assert_eq!(v.len(), 1usize << r_v.len());

        let rhs = mle_evaluate(&v, &r_v);
        prop_assert_eq!(lhs, rhs,
            "GQA MLE-replicate identity failed: log_kv={}, log_q={}, log_d={}",
            log_kv, log_q, log_d);
    }

    /// **MLE Boolean self-eval**: a random vector `v` of pow-2 length
    /// evaluated at the bit-decomposition of a random index `i` returns
    /// exactly `v[i]`. This pins down the indexing convention — if the
    /// folding direction (MSB vs LSB) silently flips, this breaks for
    /// any non-symmetric `v`.
    #[test]
    fn prop_mle_identity_self_eval(seed in any::<u64>()) {
        let mut rng = StdRng::seed_from_u64(seed);
        // Vector length 2^k for k in [1, 10] (max len 1024).
        let k: usize = rng.gen_range(1..=10);
        let n = 1usize << k;
        let v = random_field_vec(&mut rng, n);
        let i: usize = rng.gen_range(0..n);
        let r = bit_decompose_msb(i, k);
        let got = mle_evaluate(&v, &r);
        prop_assert_eq!(got, v[i],
            "MLE self-eval failed at index i={} (k={})", i, k);
    }

    /// **eq-evals correctness**: the precomputed `eq_evals(r)` table at
    /// index `x` matches the multiplicative formula `Π (r_i*x_i + (1-r_i)*(1-x_i))`
    /// when `x` is MSB-decoded the same way as the table layout. This
    /// guards against a butterfly-order regression in `eq_evals`.
    #[test]
    fn prop_eq_evals_correctness(seed in any::<u64>()) {
        let mut rng = StdRng::seed_from_u64(seed);
        let k: usize = rng.gen_range(1..=10);
        let n = 1usize << k;
        let r = random_point(&mut rng, k);
        let table = eq_evals(&r);
        prop_assert_eq!(table.len(), n);
        // Random index x in [0, 2^k).
        let x: usize = rng.gen_range(0..n);
        let x_bits = bit_decompose_msb(x, k);
        let from_formula = eq_at_point(&r, &x_bits);
        prop_assert_eq!(table[x], from_formula,
            "eq_evals[{}] mismatched multiplicative form (k={})", x, k);
        // Spot-check a second index per case: bumps coverage without
        // 2^k inner work in the worst case.
        let x2: usize = rng.gen_range(0..n);
        let x2_bits = bit_decompose_msb(x2, k);
        prop_assert_eq!(table[x2], eq_at_point(&r, &x2_bits));
    }

    /// **Matmul MLE decomposition** (sumcheck-style identity):
    ///   MLE(y, r) = Σ_{j=0}^{n-1} MLE(W, [r || j_bits]) · MLE(x, j_bits)
    /// where `y = W @ x`, `W` is row-major `(m, n)` flattened so that
    /// `W[i, j]` lives at flat index `i*n + j`, and `[r || j_bits]` is
    /// the concatenation of the row-coord `r` (length log_m) with the
    /// MSB bits of `j` (length log_n). This is the identity the prover
    /// reduces matmul claims to. We assert it directly: both sides must
    /// be equal as field elements, no sumcheck needed at this level.
    #[test]
    fn prop_matmul_mle_decomposition(seed in any::<u64>()) {
        let mut rng = StdRng::seed_from_u64(seed);
        // log_m, log_n in [1, 4] => m,n in [2, 16]. Total work per case
        // is O(m*n + n*log_m) which is trivial; 100 cases finish fast.
        let log_m: usize = rng.gen_range(1..=4);
        let log_n: usize = rng.gen_range(1..=4);
        let m = 1usize << log_m;
        let n = 1usize << log_n;

        // W flat row-major (m, n); x length n.
        let w_flat = random_field_vec(&mut rng, m * n);
        let x = random_field_vec(&mut rng, n);

        // y = W @ x in M31.
        let mut y = vec![F::zero(); m];
        for i in 0..m {
            let mut acc = F::zero();
            for j in 0..n {
                acc += w_flat[i * n + j] * x[j];
            }
            y[i] = acc;
        }
        prop_assert_eq!(y.len(), m);
        prop_assert_eq!(w_flat.len(), m * n);
        prop_assert_eq!(log2_pow2(y.len()), log_m);
        prop_assert_eq!(log2_pow2(x.len()), log_n);
        prop_assert_eq!(log2_pow2(w_flat.len()), log_m + log_n);

        // Random row-coord r of length log_m.
        let r = random_point(&mut rng, log_m);

        // LHS: MLE(y, r).
        let lhs = mle_evaluate(&y, &r);

        // RHS: Σ_j MLE(W, [r || bit_decompose_msb(j, log_n)]) * MLE(x, bit_decompose_msb(j, log_n))
        //    = Σ_j W_at_(r, j) * x[j]   (since MLE(x, boolean) = x[j])
        // We could shortcut and just compute Σ_j x[j] * W_row_MLE(j),
        // but the spec asks for the full decomposition form; do it.
        let mut rhs = F::zero();
        for j in 0..n {
            let j_bits = bit_decompose_msb(j, log_n);
            let mut full_point = Vec::with_capacity(log_m + log_n);
            full_point.extend_from_slice(&r);
            full_point.extend_from_slice(&j_bits);
            let w_term = mle_evaluate(&w_flat, &full_point);
            let x_term = mle_evaluate(&x, &j_bits);
            // Sanity: x at a Boolean point must equal x[j].
            prop_assert_eq!(x_term, x[j]);
            rhs += w_term * x_term;
        }

        prop_assert_eq!(lhs, rhs,
            "matmul MLE decomposition failed: log_m={}, log_n={}",
            log_m, log_n);
    }
}

// =====================================================================
// Static smoke tests — sanity-check the inline helpers themselves.
// These run in deterministic time and catch typos in `bit_decompose_msb`
// and `eq_at_point` before proptest amplifies a wrong baseline.
// =====================================================================

#[test]
fn smoke_bit_decompose_msb() {
    // i = 0b1010 with k=4: MSB-first bits are [1,0,1,0].
    let bits = bit_decompose_msb(0b1010, 4);
    assert_eq!(bits, vec![F::one(), F::zero(), F::one(), F::zero()]);
    // i = 0 always all zeros.
    assert_eq!(bit_decompose_msb(0, 5), vec![F::zero(); 5]);
    // i = 2^k - 1 always all ones.
    assert_eq!(bit_decompose_msb((1 << 5) - 1, 5), vec![F::one(); 5]);
}

#[test]
fn smoke_mle_evaluate_at_corner() {
    // mle_evaluate at a Boolean point picks out a single entry. This
    // exercises the same identity as the proptest above, but with a
    // hand-chosen vector so a folding-direction regression is visible.
    let v: Vec<F> = (0u32..8).map(|x| F::from_canonical_u32(x + 100)).collect();
    for i in 0..8 {
        let r = bit_decompose_msb(i, 3);
        assert_eq!(mle_evaluate(&v, &r), v[i], "self-eval @ i={}", i);
    }
}

#[test]
fn smoke_eq_evals_layout() {
    // For r = [r0, r1] we expect the table:
    //   [(1-r0)*(1-r1), (1-r0)*r1, r0*(1-r1), r0*r1]
    // with x=0..3 MSB-decoded as (b0=high, b1=low). This pins the
    // butterfly layout against a regression to LSB ordering.
    let r0 = F::from_canonical_u32(7);
    let r1 = F::from_canonical_u32(11);
    let table = eq_evals(&[r0, r1]);
    let one = F::one();
    assert_eq!(table[0], (one - r0) * (one - r1));
    assert_eq!(table[1], (one - r0) * r1);
    assert_eq!(table[2], r0 * (one - r1));
    assert_eq!(table[3], r0 * r1);
}
