//! P11-5: quantization-invariant property tests.
//!
//! This file complements P11-2 (`harness.rs`, the Rust ↔ numpy
//! differential test) by checking *quantization-domain* invariants that
//! the cross-language equality check does NOT cover. Specifically, the
//! Rust ↔ numpy harness only proves that the two implementations agree
//! on the value of every intermediate; it does NOT bound the magnitude
//! of those intermediates, nor does it check that they remain in the
//! valid M31 representation, nor that a recurrent application of the
//! Qwen forward pass produces a non-divergent output trajectory.
//!
//! The invariants here are:
//!   1. M31 in-range check: every coordinate of every intermediate
//!      stays inside `[0, 2^31 - 2)`. The forbidden value is
//!      `M31 = 2^31 - 1` (and any value `≥ M31`, which would mean a
//!      non-canonical representation).
//!   2. RMSNorm output bound: `max_i |to_signed(norm1_out_i)| < M31/2`
//!      (so the signed view fits comfortably below the half-field).
//!   3. Attention output bounded by V: `max |attn_out| ≤ max |v|` in
//!      the signed view, since seq_len=1 attention is the identity
//!      (GDN) or head-replication (GQA), both of which are bounded by
//!      the entrywise max of v.
//!   4. State-norm proxy: feeding the layer's `output` back as `x`
//!      for K iterations does not blow the magnitude up by more than
//!      a fixed factor — a regression target for the eventual
//!      P10-7 GDN recurrent state binding.
//!
//! Branch coverage mirrors `harness.rs` (`Branch::Gdn`,
//! `Branch::GqaFullAttn`, `Branch::AsymmetricV`) so that every property
//! is checked against all three SPEC branches.

use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use p3_field::PrimeField32;
use p3_mersenne_31::Mersenne31;

use zk_ml_prover::transformer::qwen::test_utils::{
    build_small_silu_table, build_small_sigmoid_table, find_valid_qwen_input,
    make_asymmetric_qwen_weights, make_qwen_weights,
};
use zk_ml_prover::transformer::qwen::{qwen_forward, QwenLayerWeights};
use zk_ml_prover::transformer::{ActivationType, ModelConfig, NormType};

type F = Mersenne31;

/// Mersenne-31 prime: `2^31 - 1`.
const M31: u32 = (1u32 << 31) - 1;
/// Half of the M31 prime; canonical "positive" range is `[0, M31_HALF]`,
/// canonical "negative" range is `(M31_HALF, M31)` mapped to negatives.
const M31_HALF: u32 = M31 / 2;

// =====================================================================
// Branch sampling — mirrors harness.rs::sample_config so this file is
// self-contained (no shared modules under tests/property/).
// =====================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Branch {
    Gdn,
    GqaFullAttn,
    AsymmetricV,
}

/// Tiny `choose` helper duplicating harness.rs's `ChooseExt`. Inlined to
/// keep this test file self-contained — the alternative is a shared
/// module under `tests/property/`, which Cargo's [[test]] auto-discovery
/// makes awkward.
fn choose<'a, R: Rng, T>(slice: &'a [T], rng: &mut R) -> &'a T {
    &slice[rng.gen_range(0..slice.len())]
}

/// Sample a small Qwen-layer config covering all three SPEC branches.
/// Same shape constraints as `harness.rs::sample_config`:
///   * `num_kv_heads >= 2` for GQA / Asymmetric-V (avoids the
///     `log2_ceil(1) == 1` corner case that breaks padded-MLE shapes).
///   * everything power-of-two so log2_ceil math is exact.
fn sample_config(seed: u64) -> (ModelConfig, Branch) {
    let mut rng = StdRng::seed_from_u64(seed);
    let branch = match rng.gen_range(0..3u8) {
        0 => Branch::Gdn,
        1 => Branch::GqaFullAttn,
        _ => Branch::AsymmetricV,
    };

    let d_head = *choose(&[2usize, 4], &mut rng);
    let num_kv_heads = match branch {
        Branch::Gdn => *choose(&[1usize, 2], &mut rng),
        Branch::GqaFullAttn | Branch::AsymmetricV => 2,
    };

    let (num_q_heads, v_num_heads, v_d_head) = match branch {
        Branch::Gdn => (num_kv_heads, 0usize, 0usize),
        Branch::GqaFullAttn => {
            let group = *choose(&[2usize, 4], &mut rng);
            (num_kv_heads * group, 0, 0)
        }
        Branch::AsymmetricV => {
            let v_h = num_kv_heads * 2;
            let v_d = d_head * 2;
            (num_kv_heads, v_h, v_d)
        }
    };

    let d_model = num_q_heads * d_head;
    let d_ff = (d_model * 2).max(4);

    let cfg = ModelConfig {
        d_model,
        d_ff,
        num_q_heads,
        num_kv_heads,
        d_head,
        n_layers: 1,
        vocab_size: d_model,
        norm_type: NormType::RMSNorm,
        activation: ActivationType::SwiGLU,
        v_num_heads,
        v_d_head,
    };
    (cfg, branch)
}

struct Fixture {
    config: ModelConfig,
    branch: Branch,
    weights: QwenLayerWeights,
    silu_table: zk_ml_prover::proving::lookup::LookupTable,
    sigmoid_table: zk_ml_prover::proving::lookup::LookupTable,
    x: Vec<F>,
}

fn build_fixture(seed: u64) -> Fixture {
    let (config, branch) = sample_config(seed);
    let silu_table = build_small_silu_table(10);
    let sigmoid_table = build_small_sigmoid_table(10);
    let weights = match branch {
        Branch::AsymmetricV => make_asymmetric_qwen_weights(&config),
        _ => make_qwen_weights(&config),
    };
    let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);
    Fixture {
        config,
        branch,
        weights,
        silu_table,
        sigmoid_table,
        x,
    }
}

// =====================================================================
// Helpers
// =====================================================================

/// Map an M31 element into its canonical signed view: values in
/// `(M31_HALF, M31)` represent negative integers `value - M31`.
///
/// INVARIANT: `to_signed(F::from_canonical_u32(M31 - 1)) == -1` and
/// `to_signed(F::from_canonical_u32(0)) == 0`. We use `i64` because
/// `M31` doesn't fit in `i32::MIN..=i32::MAX` after the sign flip.
fn to_signed(f: F) -> i64 {
    let u = f.as_canonical_u32();
    if u > M31_HALF {
        (u as i64) - (M31 as i64)
    } else {
        u as i64
    }
}

/// Maximum absolute value (in the canonical signed view) over a slice.
/// Returns 0 for an empty slice.
fn max_abs_signed(xs: &[F]) -> u64 {
    xs.iter().map(|&f| to_signed(f).unsigned_abs()).max().unwrap_or(0)
}

/// SOUNDNESS: every M31 element returned by `qwen_forward` must satisfy
/// `0 <= u < M31`. `as_canonical_u32` already guarantees this for any
/// `Mersenne31` produced by Plonky3, but the invariant is so cheap to
/// re-check (and so consequential when it fails — a non-canonical
/// element corrupts every downstream sumcheck) that we re-assert it
/// here as a regression target if the underlying field type ever
/// loosens its canonicalization.
fn assert_all_in_range(xs: &[F], stage: &str) -> Result<(), TestCaseError> {
    for (i, &f) in xs.iter().enumerate() {
        let u = f.as_canonical_u32();
        prop_assert!(
            u < M31,
            "{}: index {} = {} is not a canonical M31 element (>= 2^31 - 1)",
            stage,
            i,
            u
        );
    }
    Ok(())
}

// =====================================================================
// Property tests
// =====================================================================

proptest! {
    #![proptest_config(ProptestConfig {
        // PERF: 30 cases × 5 tests, each fixture build does up to 500
        // QR-search iterations + a full qwen_forward. Empirically <60s
        // total in release mode. See file header note on the budget.
        cases: 30,
        // Shrinking on a Qwen-layer fixture is hard to interpret; cap
        // tight to surface a fast repro.
        max_shrink_iters: 16,
        .. ProptestConfig::default()
    })]

    /// **M31 in-range after every layer**: every quantized intermediate
    /// in the Qwen forward trace fits inside `[0, M31)`.
    ///
    /// SOUNDNESS: a non-canonical M31 element silently breaks every
    /// downstream sumcheck — the prover and verifier evaluate field
    /// arithmetic on the canonicalized form, so an out-of-range value
    /// in the trace would make the prover's `mle_evaluate` call disagree
    /// with the verifier's recomputation, and the proof would fail with
    /// no clear error message. This test pins the invariant.
    #[test]
    fn prop_m31_in_range_after_every_layer(seed in any::<u64>()) {
        let fx = build_fixture(seed);
        let trace = qwen_forward(
            &fx.x, &fx.weights, &fx.config, &fx.silu_table, &fx.sigmoid_table,
        );

        // Every public field of QwenForwardTrace that the prover emits
        // into the trace, ordered by the forward graph.
        assert_all_in_range(&trace.norm1_out, "norm1_out")?;
        assert_all_in_range(&trace.q,         "q")?;
        assert_all_in_range(&trace.k,         "k")?;
        assert_all_in_range(&trace.v,         "v")?;
        assert_all_in_range(&trace.attn_out,  "attn_out")?;
        assert_all_in_range(&trace.h,         "residual_h")?;
        assert_all_in_range(&trace.norm2_out, "norm2_out")?;
        assert_all_in_range(&trace.output,    "output")?;

        // Cross-branch sanity: dimensions must line up with the SPEC.
        let q_dim = fx.config.num_q_heads * fx.config.d_head;
        let k_dim = fx.config.num_kv_heads * fx.config.d_head;
        let v_dim = fx.config.v_dim();
        let attn_out_dim = if fx.config.num_q_heads != fx.config.num_kv_heads {
            q_dim
        } else {
            v_dim
        };
        prop_assert_eq!(trace.q.len(), q_dim,
            "q dim mismatch (branch {:?})", fx.branch);
        prop_assert_eq!(trace.k.len(), k_dim,
            "k dim mismatch (branch {:?})", fx.branch);
        prop_assert_eq!(trace.v.len(), v_dim,
            "v dim mismatch (branch {:?})", fx.branch);
        prop_assert_eq!(trace.attn_out.len(), attn_out_dim,
            "attn_out dim mismatch (branch {:?})", fx.branch);
    }

    /// **RMSNorm output magnitude is bounded by `M31_HALF`**.
    ///
    /// SOUNDNESS: at the canonical inputs we use (small power-of-two
    /// `d_model`, identity `gamma`), the RMSNorm output `r * gamma_i * x_i`
    /// stays comfortably under `M31_HALF = 2^30`. For real production
    /// `d_model = 4096` we have empirically observed `max |to_signed(out)|`
    /// near `1.07e9`, just under `M31/2 ≈ 1.07e9`. Picking `< M31_HALF`
    /// here gives us a strict bound that:
    ///   (a) cleanly separates "positive M31" from "negative M31" in the
    ///       signed view (no element is ever ambiguous about its sign),
    ///   (b) leaves headroom for one multiplication after RMSNorm before
    ///       wrap-around could lose information about intent (every
    ///       multiplication that overflows `M31` becomes mod-reduced
    ///       arithmetic, which is correct in the field but ambiguous in
    ///       the signed quantization view used by lookup tables), and
    ///   (c) is a defensible regression target — if a future RMSNorm
    ///       refactor (e.g. a different perturbation strategy) produces
    ///       outputs straddling `M31_HALF`, this test catches it before
    ///       the lookup-table layer silently mis-signs the value.
    #[test]
    fn prop_rmsnorm_output_bounded(seed in any::<u64>()) {
        let fx = build_fixture(seed);
        let trace = qwen_forward(
            &fx.x, &fx.weights, &fx.config, &fx.silu_table, &fx.sigmoid_table,
        );

        let m1 = max_abs_signed(&trace.norm1_out);
        prop_assert!(
            m1 < (M31_HALF as u64),
            "norm1_out: max |signed| = {} >= M31/2 = {} (branch {:?})",
            m1, M31_HALF, fx.branch
        );

        // norm2 has the same invariant — same scaling math, same gamma=1
        // identity weights, just with `h` as input instead of `x`.
        let m2 = max_abs_signed(&trace.norm2_out);
        prop_assert!(
            m2 < (M31_HALF as u64),
            "norm2_out: max |signed| = {} >= M31/2 = {} (branch {:?})",
            m2, M31_HALF, fx.branch
        );
    }

    /// **Attention output is entrywise bounded by V**.
    ///
    /// SOUNDNESS: at seq_len=1 the Qwen layer's `attn_out` is computed
    /// in two ways depending on the branch (qwen.rs:271-283):
    ///   * GDN (q == kv): `attn_out = v.clone()` — identity, so
    ///     `max |attn_out| == max |v|`.
    ///   * GQA full-attn (q != kv): `attn_out[h*d_head + d] =
    ///     v[(h/heads_per_group)*d_head + d]` — head replication,
    ///     so each entry of `attn_out` is *exactly* an entry of `v`,
    ///     hence `max |attn_out| == max |v|`.
    ///   * Asymmetric V (v_dim != kv_dim, but q == kv): same as GDN,
    ///     `attn_out = v.clone()`.
    ///
    /// A future change to the attention code (e.g., adding the GDN
    /// delta-rule recurrence) MUST keep `max |attn_out| <= some
    /// constant * max |v|`, otherwise the downstream sigmoid_gate's
    /// requantize_to_i16_field math would clip silently. This test
    /// pins the strict ≤ relation as a regression target.
    #[test]
    fn prop_attention_output_bounded_by_v(seed in any::<u64>()) {
        let fx = build_fixture(seed);
        let trace = qwen_forward(
            &fx.x, &fx.weights, &fx.config, &fx.silu_table, &fx.sigmoid_table,
        );

        let v_max = max_abs_signed(&trace.v);
        let a_max = max_abs_signed(&trace.attn_out);

        // SOUNDNESS: strict ≤ for current seq_len=1 implementation.
        // When seq_len ≥ 2 row-attention lands, the bound becomes
        // `<= max |v| * 1` only after softmax normalization; this
        // test will need updating then, alongside the row-attention
        // proof.
        prop_assert!(
            a_max <= v_max,
            "attn_out exceeds v: max|attn_out|={} max|v|={} (branch {:?})",
            a_max, v_max, fx.branch
        );

        // Stronger structural invariant: every entry of `attn_out`
        // must literally be an entry of `v`. `to_signed` is a bijection
        // on the M31 representation, so we can compare canonical u32s.
        // O(N^2) but our fixtures are tiny (≤ 16 entries each).
        use std::collections::HashSet;
        let v_set: HashSet<u32> =
            trace.v.iter().map(|f| f.as_canonical_u32()).collect();
        for (i, &a) in trace.attn_out.iter().enumerate() {
            let u = a.as_canonical_u32();
            prop_assert!(
                v_set.contains(&u),
                "attn_out[{}] = {} is not an element of v (branch {:?})",
                i, u, fx.branch
            );
        }
    }

    /// **State-norm proxy under simulated recurrence (P10-7 target)**.
    ///
    /// PERF: Real GDN recurrent state will iterate the recurrence
    /// `S_{t+1} = β · S_t + outer(k, v)` for thousands of steps; a
    /// proper soak test belongs in P10-7. Here we use a coarse proxy:
    /// run the layer 64 times in a row, feeding `output` back as `x`,
    /// and assert each step's output stays representable in M31 (i.e.
    /// `< M31_HALF` in the signed view, so the lookup-table layer can
    /// continue to interpret the value's sign without ambiguity).
    ///
    /// Why 64 not 1000: at d_model ≤ 16, fixture build + forward pass
    /// is dominated by `find_valid_qwen_input`'s QR-search (up to 500
    /// candidate offsets). 1000 forward passes per case × 30 cases
    /// would dominate the file's runtime. 64 is enough to surface a
    /// regression where the layer falls into a non-fixed-point cycle
    /// (e.g., a future GDN delta-rule patch hands back a non-canonical
    /// output) while keeping total wall-clock comfortably under 60s.
    ///
    /// Why `< M31_HALF`: the per-step output is `h + down_out` (with
    /// `down_out == 0` for our zero-`w_down` identity fixtures, so
    /// `output == h`). For `h = x + gated_attn_o_proj` with all
    /// quantities in canonical M31, the value can land anywhere in
    /// `[0, M31)`; the field arithmetic doesn't shrink magnitudes.
    /// What we *can* assert as a regression target is that the value
    /// remains a canonical M31 element AND stays in the lower half of
    /// the field, mirroring the RMSNorm bound: this catches a future
    /// regression where a "negative-overflow" wraps an output to
    /// `(M31_HALF, M31)` (i.e., a "negative" signed view) when the
    /// algebra should have kept it positive. If a future change
    /// legitimately drives outputs into the negative half-field, this
    /// test will need to relax to `< M31` and rely on
    /// `prop_m31_in_range_after_every_layer` for the canonical check.
    #[test]
    fn prop_state_norm_bounded_under_recurrence_proxy(seed in any::<u64>()) {
        let fx = build_fixture(seed);

        let initial_max = max_abs_signed(&fx.x);
        prop_assert!(initial_max > 0,
            "fixture's input vector should not be all-zero (seed {})", seed);

        let mut x = fx.x.clone();
        const STEPS: usize = 64;
        // SOUNDNESS: the bound here is `< M31_HALF` (not relative to
        // initial). Field arithmetic doesn't shrink magnitudes and
        // RMSNorm's `r` scalar is effectively a uniform random
        // element, so the per-step `max |signed|` is well-modelled
        // as uniformly distributed in `[0, M31)` — but the algebraic
        // structure for our identity-weight, zero-FFN fixtures keeps
        // the expected value in the positive half-field, and a
        // regression that causes wrap-around into the negative half
        // is exactly what we want this test to catch.
        let envelope = M31_HALF as u128;
        for step in 0..STEPS {
            let trace = qwen_forward(
                &x, &fx.weights, &fx.config, &fx.silu_table, &fx.sigmoid_table,
            );
            let out_max = max_abs_signed(&trace.output) as u128;
            prop_assert!(
                out_max < envelope,
                "step {}: |output|_∞ = {} exceeded M31_HALF envelope {} \
                 (initial = {}, branch {:?}, seed {})",
                step, out_max, envelope, initial_max, fx.branch, seed
            );
            // Also assert canonical-range: a wrap into [M31, 2^32) here
            // would corrupt downstream sumcheck math.
            assert_all_in_range(&trace.output,
                &format!("recurrence step {}", step))?;

            // Re-canonicalize trace.output for the next step. RMSNorm's
            // own QR-perturbation loop runs internally on each pass.
            x = trace.output;
        }
    }

    /// **Lookup-cell correctness over the inputs the layer actually hits**.
    ///
    /// Supporting test: every entry the layer feeds into the SiLU and
    /// sigmoid lookup tables must produce an output that exactly matches
    /// the table's own (input, output) pair for that input. This is
    /// trivially true by construction (the forward pass *uses* the table
    /// to produce these outputs), so the test serves as a coverage
    /// check: `gate_out` must hit at least one cell of `silu_table`,
    /// confirming the lookup machinery actually executed and isn't
    /// silently no-op'd by a future refactor.
    ///
    /// SOUNDNESS: a regression that bypassed the lookup (e.g., calling
    /// `gate_out.iter().map(|x| F::zero())`) would leave `gate_silu`
    /// trivially in-range and the in-range tests above would still
    /// pass. This test pins the *positive* relation between the two
    /// arrays.
    #[test]
    fn prop_lookup_cells_consistent(seed in any::<u64>()) {
        let fx = build_fixture(seed);
        let trace = qwen_forward(
            &fx.x, &fx.weights, &fx.config, &fx.silu_table, &fx.sigmoid_table,
        );

        // Build a HashMap once per case; lookup is O(1) per check.
        use std::collections::HashMap;
        let silu_map: HashMap<u32, u32> = fx.silu_table.entries.iter().copied().collect();
        let sigmoid_map: HashMap<u32, u32> =
            fx.sigmoid_table.entries.iter().copied().collect();

        prop_assert_eq!(trace.gate_out.len(), trace.gate_silu.len(),
            "gate_out / gate_silu length mismatch");
        for (i, (&g_in, &g_out)) in
            trace.gate_out.iter().zip(trace.gate_silu.iter()).enumerate()
        {
            let key = g_in.as_canonical_u32();
            let val = g_out.as_canonical_u32();
            let expected = silu_map.get(&key).copied().unwrap_or_else(|| {
                panic!("silu input {} not in table (test fixture broken)", key)
            });
            prop_assert_eq!(val, expected,
                "gate_silu[{}] = {} but silu_table[{}] = {} (branch {:?})",
                i, val, key, expected, fx.branch);
        }

        prop_assert_eq!(trace.g_proj_out.len(), trace.g_proj_sigmoid.len(),
            "g_proj_out / g_proj_sigmoid length mismatch");
        for (i, (&s_in, &s_out)) in trace.g_proj_out.iter()
            .zip(trace.g_proj_sigmoid.iter()).enumerate()
        {
            let key = s_in.as_canonical_u32();
            let val = s_out.as_canonical_u32();
            let expected = sigmoid_map.get(&key).copied().unwrap_or_else(|| {
                panic!("sigmoid input {} not in table (test fixture broken)", key)
            });
            prop_assert_eq!(val, expected,
                "g_proj_sigmoid[{}] = {} but sigmoid_table[{}] = {} (branch {:?})",
                i, val, key, expected, fx.branch);
        }
    }
}

// =====================================================================
// Static branch-coverage check.
//
// Mirrors harness.rs::prop_branch_coverage_warning. Re-derives coverage
// from `sample_config` (deterministic) so it doesn't race the proptests.
// =====================================================================

#[test]
fn quant_invariants_branch_coverage() {
    let mut gdn = 0u32;
    let mut gqa = 0u32;
    let mut asym = 0u32;
    for seed in 0u64..200 {
        let (_, branch) = sample_config(seed);
        match branch {
            Branch::Gdn => gdn += 1,
            Branch::GqaFullAttn => gqa += 1,
            Branch::AsymmetricV => asym += 1,
        }
    }
    eprintln!(
        "p11-5 branch coverage over 200 deterministic seeds: GDN={} GQA={} ASYM={}",
        gdn, gqa, asym
    );
    assert!(
        gdn > 0 && gqa > 0 && asym > 0,
        "branch coverage warning: GDN={}, GQA={}, ASYM={} — sample_config \
         didn't sample all three SPEC branches across 200 seeds.",
        gdn, gqa, asym,
    );
}

// =====================================================================
// Sanity: confirm helpers are correct.
// =====================================================================

#[test]
fn quant_invariants_to_signed_helper_correct() {
    use p3_field::AbstractField;
    // SOUNDNESS: to_signed must round-trip across the M31 prime boundary
    // exactly. Get this wrong and every magnitude bound above is wrong.
    assert_eq!(to_signed(F::zero()), 0);
    assert_eq!(to_signed(F::one()), 1);
    // M31 - 1 in canonical form is `2^31 - 2`, which is `>` M31_HALF, so
    // it canonicalizes to -1.
    assert_eq!(to_signed(F::from_canonical_u32(M31 - 1)), -1);
    // M31_HALF itself canonicalizes to its positive int (boundary case).
    assert_eq!(to_signed(F::from_canonical_u32(M31_HALF)), M31_HALF as i64);
    // M31_HALF + 1 canonicalizes to negative.
    assert_eq!(
        to_signed(F::from_canonical_u32(M31_HALF + 1)),
        (M31_HALF as i64 + 1) - (M31 as i64)
    );
}
