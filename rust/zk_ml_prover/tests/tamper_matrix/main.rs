//! P11-4: tamper-coverage matrix.
//!
//! For every (proof type × top-level field × config branch) triple, build a
//! valid Qwen-layer proof, mutate one byte (or one bit / one u32 slot) of the
//! named field, and assert the verifier rejects. The matrix multiplies:
//!
//!   Proof types:   QwenLayerProof (base-field), QwenLayerProofEF
//!   Branches:      GDN (q == kv == v), GQA-full-attn (q != kv, v == kv),
//!                  asymmetric-V (q == kv, v != kv)
//!   Field paths:   ~16 top-level fields per proof type
//!     ⇒ ~96 generated tests
//!
//! Spec: `tests/reference/SPEC.md` § "Config branch coverage". Helpers
//! (`make_qwen_weights`, `make_asymmetric_qwen_weights`, `find_valid_qwen_input`,
//! `build_small_*_table`) live in `transformer::qwen::test_utils`.
//!
//! Acceptance gate: deliberately reintroduce the S5 perturbation_delta-not-
//! absorbed bug in `verify_rmsnorm` and confirm at least one tamper test
//! rejects within 60 s. We don't run that here — it's verified by hand
//! (see report at the end of P11-4 commit).
//!
//! Out of scope (follow-up):
//!   - Recursive tamper across sub-proofs (e.g. inner sumcheck round_polys[k]
//!     for k > 0). The current matrix tampers only top-level field heads.
//!   - GdnRecurrenceProof × branches (waiting on P10-7).
//!   - Random-byte mutation across the full bincode-serialized proof.

use zk_ml_prover::proving::sumcheck::Transcript;
use zk_ml_prover::transformer::ModelConfig;
use zk_ml_prover::transformer::{NormType, ActivationType};
use zk_ml_prover::transformer::qwen::{
    QwenLayerProof, QwenLayerProofEF,
    prove_qwen_layer_with_trace, verify_qwen_layer,
    prove_qwen_layer_precommitted_ef, verify_qwen_layer_ef,
    qwen_forward,
};
use zk_ml_prover::transformer::qwen::test_utils::{
    build_small_silu_table, build_small_sigmoid_table,
    make_qwen_weights, make_asymmetric_qwen_weights, find_valid_qwen_input,
    placeholder_qwen_commitments_for_test,
};

// ============================================================================
// Config branch builders
// ============================================================================

fn cfg_gdn() -> ModelConfig {
    ModelConfig {
        d_model: 8, d_ff: 16,
        num_q_heads: 2, num_kv_heads: 2, d_head: 4,
        n_layers: 1, vocab_size: 8,
        norm_type: NormType::RMSNorm,
        activation: ActivationType::SwiGLU,
        v_num_heads: 0, v_d_head: 0,
    }
}

fn cfg_gqa() -> ModelConfig {
    ModelConfig {
        d_model: 8, d_ff: 16,
        num_q_heads: 4, num_kv_heads: 2, d_head: 4,
        n_layers: 1, vocab_size: 8,
        norm_type: NormType::RMSNorm,
        activation: ActivationType::SwiGLU,
        v_num_heads: 0, v_d_head: 0,
    }
}

fn cfg_asym() -> ModelConfig {
    ModelConfig {
        d_model: 8, d_ff: 16,
        num_q_heads: 2, num_kv_heads: 2, d_head: 4,
        n_layers: 1, vocab_size: 8,
        norm_type: NormType::RMSNorm,
        activation: ActivationType::SwiGLU,
        v_num_heads: 4, v_d_head: 4,
    }
}

// ============================================================================
// Setup + verify helpers
// ============================================================================

/// Build everything needed to prove + verify a QwenLayerProof for the given
/// config. Asymmetric branch uses `make_asymmetric_qwen_weights`; the other
/// two use `make_qwen_weights`.
struct BaseFieldFixture {
    config: ModelConfig,
    weights: zk_ml_prover::transformer::qwen::QwenLayerWeights,
    silu_table: zk_ml_prover::proving::lookup::LookupTable,
    sigmoid_table: zk_ml_prover::proving::lookup::LookupTable,
    x: Vec<p3_mersenne_31::Mersenne31>,
    y: Vec<p3_mersenne_31::Mersenne31>,
}

fn build_base_fixture(config: ModelConfig) -> (BaseFieldFixture, QwenLayerProof) {
    let silu_table = build_small_silu_table(10);
    let sigmoid_table = build_small_sigmoid_table(10);
    let weights = if config.v_num_heads != 0 || config.v_d_head != 0 {
        make_asymmetric_qwen_weights(&config)
    } else {
        make_qwen_weights(&config)
    };
    let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);
    let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);
    let y = trace.output.clone();

    let mut pt = Transcript::new(b"p11-4-base");
    let proof = prove_qwen_layer_with_trace(
        &trace, &weights, &config, &silu_table, &sigmoid_table, &mut pt,
    );

    let fx = BaseFieldFixture { config, weights, silu_table, sigmoid_table, x, y };
    (fx, proof)
}

fn verify_base(fx: &BaseFieldFixture, proof: &QwenLayerProof) -> bool {
    let mut vt = Transcript::new(b"p11-4-base");
    verify_qwen_layer(
        proof, &fx.x, &fx.y, &fx.weights, &fx.config,
        &fx.silu_table, &fx.sigmoid_table, &mut vt,
    )
}

struct EFFixture {
    config: ModelConfig,
    weights: zk_ml_prover::transformer::qwen::QwenLayerWeights,
    silu_table: zk_ml_prover::proving::lookup::LookupTable,
    sigmoid_table: zk_ml_prover::proving::lookup::LookupTable,
    x: Vec<p3_mersenne_31::Mersenne31>,
    y: Vec<p3_mersenne_31::Mersenne31>,
}

fn build_ef_fixture(config: ModelConfig) -> (EFFixture, QwenLayerProofEF) {
    let silu_table = build_small_silu_table(10);
    let sigmoid_table = build_small_sigmoid_table(10);
    let weights = if config.v_num_heads != 0 || config.v_d_head != 0 {
        make_asymmetric_qwen_weights(&config)
    } else {
        make_qwen_weights(&config)
    };
    let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);
    let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);
    let y = trace.output.clone();
    let commitments = placeholder_qwen_commitments_for_test(&weights);

    let mut pt = Transcript::new(b"p11-4-ef");
    let proof = prove_qwen_layer_precommitted_ef(
        &trace, &weights, &config, commitments,
        &silu_table, &sigmoid_table, &mut pt,
    );

    let fx = EFFixture { config, weights, silu_table, sigmoid_table, x, y };
    (fx, proof)
}

fn verify_ef(fx: &EFFixture, proof: &QwenLayerProofEF) -> bool {
    let mut vt = Transcript::new(b"p11-4-ef");
    verify_qwen_layer_ef(
        proof, &fx.x, &fx.y, &fx.weights, &fx.config,
        &fx.silu_table, &fx.sigmoid_table, &mut vt,
    )
}

/// Assert the verifier rejects (returns false) OR panics (out-of-bounds shape
/// check, unwrap inside verify, etc.). Both outcomes count as "tamper detected"
/// for our soundness claim — the verifier did not silently accept.
fn assert_rejects_or_panics<F>(label: &str, f: F)
where
    F: FnOnce() -> bool + std::panic::UnwindSafe,
{
    let prev_hook = std::panic::take_hook();
    // Suppress backtrace spam from the captured panic; we expect it.
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(prev_hook);
    match result {
        Ok(false) | Err(_) => {} // expected: rejected or panicked
        Ok(true) => panic!(
            "P11-4: tamper of {} was silently ACCEPTED by verifier — soundness regression",
            label
        ),
    }
}

// ============================================================================
// Macro: emit one #[test] per (branch × tamper) for QwenLayerProof.
// ============================================================================

macro_rules! tamper_test_qwen_base {
    ($name:ident, $branch_fn:expr, $label:expr, $tamper:expr) => {
        #[test]
        fn $name() {
            let (fx, mut proof) = build_base_fixture($branch_fn());
            // Pre-tamper sanity: untampered proof verifies.
            assert!(verify_base(&fx, &proof),
                "P11-4: untampered proof for {} must verify (otherwise tamper test is vacuous)",
                $label);
            // Apply tamper.
            ($tamper)(&mut proof);
            // Post-tamper: must reject (or panic).
            let label = $label.to_string();
            assert_rejects_or_panics(&label, move || verify_base(&fx, &proof));
        }
    };
}

macro_rules! tamper_test_qwen_ef {
    ($name:ident, $branch_fn:expr, $label:expr, $tamper:expr) => {
        #[test]
        fn $name() {
            let (fx, mut proof) = build_ef_fixture($branch_fn());
            assert!(verify_ef(&fx, &proof),
                "P11-4: untampered EF proof for {} must verify",
                $label);
            ($tamper)(&mut proof);
            let label = $label.to_string();
            assert_rejects_or_panics(&label, move || verify_ef(&fx, &proof));
        }
    };
}

// ============================================================================
// Tamper closures — base-field path (u32 round_polys, u32 finals).
// ============================================================================
//
// Each closure mutates one byte/bit of one named field. We use XOR-1 on the
// least-significant bit so the mutated value is guaranteed to differ from the
// original (XOR is involutive — and since field values are stored as u32 with
// value space [0, M31), flipping bit 0 always lands on a different element).

fn t_base_norm1_var_round_poly(p: &mut QwenLayerProof) {
    p.norm1_proof.var_proof.round_polys[0][0] ^= 1;
}
fn t_base_qkv0_round_poly(p: &mut QwenLayerProof) {
    p.qkv_proofs.0.matmul_proof.sumcheck_proof.round_polys[0][0] ^= 1;
}
fn t_base_qkv1_round_poly(p: &mut QwenLayerProof) {
    p.qkv_proofs.1.matmul_proof.sumcheck_proof.round_polys[0][0] ^= 1;
}
fn t_base_qkv2_round_poly(p: &mut QwenLayerProof) {
    p.qkv_proofs.2.matmul_proof.sumcheck_proof.round_polys[0][0] ^= 1;
}
fn t_base_attn_out_at_r(p: &mut QwenLayerProof) {
    if let Some(ref mut s) = p.attn_proof.seq1_consistency {
        s.attn_out_at_r ^= 1;
    } else {
        panic!("P11-4 setup: seq1_consistency must be Some at seq_len=1");
    }
}
fn t_base_v_at_r(p: &mut QwenLayerProof) {
    if let Some(ref mut s) = p.attn_proof.seq1_consistency {
        s.v_at_r ^= 1;
    } else {
        panic!("P11-4 setup: seq1_consistency must be Some at seq_len=1");
    }
}
fn t_base_o_proj_round_poly(p: &mut QwenLayerProof) {
    p.o_proj_proof.matmul_proof.sumcheck_proof.round_polys[0][0] ^= 1;
}
fn t_base_g_proj_round_poly(p: &mut QwenLayerProof) {
    p.g_proj_proof.matmul_proof.sumcheck_proof.round_polys[0][0] ^= 1;
}
fn t_base_sigmoid_gate_lookup_round_poly(p: &mut QwenLayerProof) {
    p.sigmoid_gate_proof.sigmoid_proof.lookup_proof.sumcheck_proof.round_polys[0][0] ^= 1;
}
fn t_base_residual1_a_finals(p: &mut QwenLayerProof) {
    // Spec calls this `add_finals.0`; the actual struct stores it as
    // `a_finals.0` (the eq_at_s u32). Same meaning, different name.
    p.residual1_proof.a_finals.0 ^= 1;
}
fn t_base_norm2_var_round_poly(p: &mut QwenLayerProof) {
    p.norm2_proof.var_proof.round_polys[0][0] ^= 1;
}
fn t_base_gate_proj_round_poly(p: &mut QwenLayerProof) {
    p.gate_proj_proof.matmul_proof.sumcheck_proof.round_polys[0][0] ^= 1;
}
fn t_base_up_proj_round_poly(p: &mut QwenLayerProof) {
    p.up_proj_proof.matmul_proof.sumcheck_proof.round_polys[0][0] ^= 1;
}
fn t_base_swiglu_silu_lookup_round_poly(p: &mut QwenLayerProof) {
    p.swiglu_proof.silu_proof.lookup_proof.sumcheck_proof.round_polys[0][0] ^= 1;
}
fn t_base_down_proj_round_poly(p: &mut QwenLayerProof) {
    p.down_proj_proof.matmul_proof.sumcheck_proof.round_polys[0][0] ^= 1;
}
fn t_base_residual2_a_finals(p: &mut QwenLayerProof) {
    p.residual2_proof.a_finals.0 ^= 1;
}
fn t_base_w_q_root(p: &mut QwenLayerProof) { p.w_q_commitment.root[0] ^= 1; }
fn t_base_w_k_root(p: &mut QwenLayerProof) { p.w_k_commitment.root[0] ^= 1; }
fn t_base_w_v_root(p: &mut QwenLayerProof) { p.w_v_commitment.root[0] ^= 1; }
fn t_base_w_o_root(p: &mut QwenLayerProof) { p.w_o_commitment.root[0] ^= 1; }
fn t_base_w_g_proj_root(p: &mut QwenLayerProof) { p.w_g_proj_commitment.root[0] ^= 1; }
fn t_base_w_gate_root(p: &mut QwenLayerProof) { p.w_gate_commitment.root[0] ^= 1; }
fn t_base_w_up_root(p: &mut QwenLayerProof) { p.w_up_commitment.root[0] ^= 1; }
fn t_base_w_down_root(p: &mut QwenLayerProof) { p.w_down_commitment.root[0] ^= 1; }

// ============================================================================
// Tamper closures — EF path. round_polys are Vec<EFElement>, EFElement([u32;4]).
// We tamper the first u32 limb of the first round-poly evaluation.
// ============================================================================

fn t_ef_norm1_var_round_poly(p: &mut QwenLayerProofEF) {
    p.norm1_proof.var_proof.round_polys[0][0].0[0] ^= 1;
}
fn t_ef_qkv0_round_poly(p: &mut QwenLayerProofEF) {
    p.qkv_proofs.0.matmul_proof.sumcheck_proof.round_polys[0][0].0[0] ^= 1;
}
fn t_ef_qkv1_round_poly(p: &mut QwenLayerProofEF) {
    p.qkv_proofs.1.matmul_proof.sumcheck_proof.round_polys[0][0].0[0] ^= 1;
}
fn t_ef_qkv2_round_poly(p: &mut QwenLayerProofEF) {
    p.qkv_proofs.2.matmul_proof.sumcheck_proof.round_polys[0][0].0[0] ^= 1;
}
fn t_ef_attn_out_at_r(p: &mut QwenLayerProofEF) {
    if let Some(ref mut s) = p.attn_proof.seq1_consistency {
        s.attn_out_at_r ^= 1;
    } else {
        panic!("P11-4 setup: seq1_consistency must be Some at seq_len=1");
    }
}
fn t_ef_v_at_r(p: &mut QwenLayerProofEF) {
    if let Some(ref mut s) = p.attn_proof.seq1_consistency {
        s.v_at_r ^= 1;
    } else {
        panic!("P11-4 setup: seq1_consistency must be Some at seq_len=1");
    }
}
fn t_ef_o_proj_round_poly(p: &mut QwenLayerProofEF) {
    p.o_proj_proof.matmul_proof.sumcheck_proof.round_polys[0][0].0[0] ^= 1;
}
fn t_ef_g_proj_round_poly(p: &mut QwenLayerProofEF) {
    p.g_proj_proof.matmul_proof.sumcheck_proof.round_polys[0][0].0[0] ^= 1;
}
fn t_ef_sigmoid_gate_lookup_round_poly(p: &mut QwenLayerProofEF) {
    p.sigmoid_gate_proof.sigmoid_proof.lookup_proof.sumcheck_proof.round_polys[0][0].0[0] ^= 1;
}
fn t_ef_residual1_a_finals(p: &mut QwenLayerProofEF) {
    p.residual1_proof.a_finals.0.0[0] ^= 1;
}
fn t_ef_norm2_var_round_poly(p: &mut QwenLayerProofEF) {
    p.norm2_proof.var_proof.round_polys[0][0].0[0] ^= 1;
}
fn t_ef_gate_proj_round_poly(p: &mut QwenLayerProofEF) {
    p.gate_proj_proof.matmul_proof.sumcheck_proof.round_polys[0][0].0[0] ^= 1;
}
fn t_ef_up_proj_round_poly(p: &mut QwenLayerProofEF) {
    p.up_proj_proof.matmul_proof.sumcheck_proof.round_polys[0][0].0[0] ^= 1;
}
fn t_ef_swiglu_silu_lookup_round_poly(p: &mut QwenLayerProofEF) {
    p.swiglu_proof.silu_proof.lookup_proof.sumcheck_proof.round_polys[0][0].0[0] ^= 1;
}
fn t_ef_down_proj_round_poly(p: &mut QwenLayerProofEF) {
    p.down_proj_proof.matmul_proof.sumcheck_proof.round_polys[0][0].0[0] ^= 1;
}
fn t_ef_residual2_a_finals(p: &mut QwenLayerProofEF) {
    p.residual2_proof.a_finals.0.0[0] ^= 1;
}
fn t_ef_w_q_root(p: &mut QwenLayerProofEF) { p.w_q_commitment.root[0] ^= 1; }
fn t_ef_w_k_root(p: &mut QwenLayerProofEF) { p.w_k_commitment.root[0] ^= 1; }
fn t_ef_w_v_root(p: &mut QwenLayerProofEF) { p.w_v_commitment.root[0] ^= 1; }
fn t_ef_w_o_root(p: &mut QwenLayerProofEF) { p.w_o_commitment.root[0] ^= 1; }
fn t_ef_w_g_proj_root(p: &mut QwenLayerProofEF) { p.w_g_proj_commitment.root[0] ^= 1; }
fn t_ef_w_gate_root(p: &mut QwenLayerProofEF) { p.w_gate_commitment.root[0] ^= 1; }
fn t_ef_w_up_root(p: &mut QwenLayerProofEF) { p.w_up_commitment.root[0] ^= 1; }
fn t_ef_w_down_root(p: &mut QwenLayerProofEF) { p.w_down_commitment.root[0] ^= 1; }

// ============================================================================
// Generated tests: QwenLayerProof × {GDN, GQA, asymmetric-V}
// ============================================================================

// --- GDN branch (24 fields) ---
tamper_test_qwen_base!(p11_4_base_gdn_norm1_var_round_poly, cfg_gdn, "base/gdn/norm1.var.rp[0][0]", t_base_norm1_var_round_poly);
tamper_test_qwen_base!(p11_4_base_gdn_qkv0_round_poly,      cfg_gdn, "base/gdn/qkv.0.rp[0][0]",   t_base_qkv0_round_poly);
tamper_test_qwen_base!(p11_4_base_gdn_qkv1_round_poly,      cfg_gdn, "base/gdn/qkv.1.rp[0][0]",   t_base_qkv1_round_poly);
tamper_test_qwen_base!(p11_4_base_gdn_qkv2_round_poly,      cfg_gdn, "base/gdn/qkv.2.rp[0][0]",   t_base_qkv2_round_poly);
tamper_test_qwen_base!(p11_4_base_gdn_attn_out_at_r,        cfg_gdn, "base/gdn/attn_out_at_r",    t_base_attn_out_at_r);
tamper_test_qwen_base!(p11_4_base_gdn_v_at_r,               cfg_gdn, "base/gdn/v_at_r",           t_base_v_at_r);
tamper_test_qwen_base!(p11_4_base_gdn_o_proj_round_poly,    cfg_gdn, "base/gdn/o_proj.rp[0][0]",  t_base_o_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_gdn_g_proj_round_poly,    cfg_gdn, "base/gdn/g_proj.rp[0][0]",  t_base_g_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_gdn_sigmoid_gate_lookup_rp, cfg_gdn, "base/gdn/sigmoid_gate.lookup.rp[0][0]", t_base_sigmoid_gate_lookup_round_poly);
tamper_test_qwen_base!(p11_4_base_gdn_residual1_a_finals,   cfg_gdn, "base/gdn/residual1.a_finals.0", t_base_residual1_a_finals);
tamper_test_qwen_base!(p11_4_base_gdn_norm2_var_round_poly, cfg_gdn, "base/gdn/norm2.var.rp[0][0]", t_base_norm2_var_round_poly);
tamper_test_qwen_base!(p11_4_base_gdn_gate_proj_round_poly, cfg_gdn, "base/gdn/gate_proj.rp[0][0]", t_base_gate_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_gdn_up_proj_round_poly,   cfg_gdn, "base/gdn/up_proj.rp[0][0]",   t_base_up_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_gdn_swiglu_silu_lookup_rp, cfg_gdn, "base/gdn/swiglu.silu.lookup.rp[0][0]", t_base_swiglu_silu_lookup_round_poly);
tamper_test_qwen_base!(p11_4_base_gdn_down_proj_round_poly, cfg_gdn, "base/gdn/down_proj.rp[0][0]", t_base_down_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_gdn_residual2_a_finals,   cfg_gdn, "base/gdn/residual2.a_finals.0", t_base_residual2_a_finals);
tamper_test_qwen_base!(p11_4_base_gdn_w_q_root,             cfg_gdn, "base/gdn/w_q.root[0]",         t_base_w_q_root);
tamper_test_qwen_base!(p11_4_base_gdn_w_k_root,             cfg_gdn, "base/gdn/w_k.root[0]",         t_base_w_k_root);
tamper_test_qwen_base!(p11_4_base_gdn_w_v_root,             cfg_gdn, "base/gdn/w_v.root[0]",         t_base_w_v_root);
tamper_test_qwen_base!(p11_4_base_gdn_w_o_root,             cfg_gdn, "base/gdn/w_o.root[0]",         t_base_w_o_root);
tamper_test_qwen_base!(p11_4_base_gdn_w_g_proj_root,        cfg_gdn, "base/gdn/w_g_proj.root[0]",    t_base_w_g_proj_root);
tamper_test_qwen_base!(p11_4_base_gdn_w_gate_root,          cfg_gdn, "base/gdn/w_gate.root[0]",      t_base_w_gate_root);
tamper_test_qwen_base!(p11_4_base_gdn_w_up_root,            cfg_gdn, "base/gdn/w_up.root[0]",        t_base_w_up_root);
tamper_test_qwen_base!(p11_4_base_gdn_w_down_root,          cfg_gdn, "base/gdn/w_down.root[0]",      t_base_w_down_root);

// --- GQA branch ---
tamper_test_qwen_base!(p11_4_base_gqa_norm1_var_round_poly, cfg_gqa, "base/gqa/norm1.var.rp[0][0]", t_base_norm1_var_round_poly);
tamper_test_qwen_base!(p11_4_base_gqa_qkv0_round_poly,      cfg_gqa, "base/gqa/qkv.0.rp[0][0]",   t_base_qkv0_round_poly);
tamper_test_qwen_base!(p11_4_base_gqa_qkv1_round_poly,      cfg_gqa, "base/gqa/qkv.1.rp[0][0]",   t_base_qkv1_round_poly);
tamper_test_qwen_base!(p11_4_base_gqa_qkv2_round_poly,      cfg_gqa, "base/gqa/qkv.2.rp[0][0]",   t_base_qkv2_round_poly);
tamper_test_qwen_base!(p11_4_base_gqa_attn_out_at_r,        cfg_gqa, "base/gqa/attn_out_at_r",    t_base_attn_out_at_r);
tamper_test_qwen_base!(p11_4_base_gqa_v_at_r,               cfg_gqa, "base/gqa/v_at_r",           t_base_v_at_r);
tamper_test_qwen_base!(p11_4_base_gqa_o_proj_round_poly,    cfg_gqa, "base/gqa/o_proj.rp[0][0]",  t_base_o_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_gqa_g_proj_round_poly,    cfg_gqa, "base/gqa/g_proj.rp[0][0]",  t_base_g_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_gqa_sigmoid_gate_lookup_rp, cfg_gqa, "base/gqa/sigmoid_gate.lookup.rp[0][0]", t_base_sigmoid_gate_lookup_round_poly);
tamper_test_qwen_base!(p11_4_base_gqa_residual1_a_finals,   cfg_gqa, "base/gqa/residual1.a_finals.0", t_base_residual1_a_finals);
tamper_test_qwen_base!(p11_4_base_gqa_norm2_var_round_poly, cfg_gqa, "base/gqa/norm2.var.rp[0][0]", t_base_norm2_var_round_poly);
tamper_test_qwen_base!(p11_4_base_gqa_gate_proj_round_poly, cfg_gqa, "base/gqa/gate_proj.rp[0][0]", t_base_gate_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_gqa_up_proj_round_poly,   cfg_gqa, "base/gqa/up_proj.rp[0][0]",   t_base_up_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_gqa_swiglu_silu_lookup_rp, cfg_gqa, "base/gqa/swiglu.silu.lookup.rp[0][0]", t_base_swiglu_silu_lookup_round_poly);
tamper_test_qwen_base!(p11_4_base_gqa_down_proj_round_poly, cfg_gqa, "base/gqa/down_proj.rp[0][0]", t_base_down_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_gqa_residual2_a_finals,   cfg_gqa, "base/gqa/residual2.a_finals.0", t_base_residual2_a_finals);
tamper_test_qwen_base!(p11_4_base_gqa_w_q_root,             cfg_gqa, "base/gqa/w_q.root[0]",         t_base_w_q_root);
tamper_test_qwen_base!(p11_4_base_gqa_w_k_root,             cfg_gqa, "base/gqa/w_k.root[0]",         t_base_w_k_root);
tamper_test_qwen_base!(p11_4_base_gqa_w_v_root,             cfg_gqa, "base/gqa/w_v.root[0]",         t_base_w_v_root);
tamper_test_qwen_base!(p11_4_base_gqa_w_o_root,             cfg_gqa, "base/gqa/w_o.root[0]",         t_base_w_o_root);
tamper_test_qwen_base!(p11_4_base_gqa_w_g_proj_root,        cfg_gqa, "base/gqa/w_g_proj.root[0]",    t_base_w_g_proj_root);
tamper_test_qwen_base!(p11_4_base_gqa_w_gate_root,          cfg_gqa, "base/gqa/w_gate.root[0]",      t_base_w_gate_root);
tamper_test_qwen_base!(p11_4_base_gqa_w_up_root,            cfg_gqa, "base/gqa/w_up.root[0]",        t_base_w_up_root);
tamper_test_qwen_base!(p11_4_base_gqa_w_down_root,          cfg_gqa, "base/gqa/w_down.root[0]",      t_base_w_down_root);

// --- Asymmetric-V branch ---
tamper_test_qwen_base!(p11_4_base_asym_norm1_var_round_poly, cfg_asym, "base/asym/norm1.var.rp[0][0]", t_base_norm1_var_round_poly);
tamper_test_qwen_base!(p11_4_base_asym_qkv0_round_poly,      cfg_asym, "base/asym/qkv.0.rp[0][0]",   t_base_qkv0_round_poly);
tamper_test_qwen_base!(p11_4_base_asym_qkv1_round_poly,      cfg_asym, "base/asym/qkv.1.rp[0][0]",   t_base_qkv1_round_poly);
tamper_test_qwen_base!(p11_4_base_asym_qkv2_round_poly,      cfg_asym, "base/asym/qkv.2.rp[0][0]",   t_base_qkv2_round_poly);
tamper_test_qwen_base!(p11_4_base_asym_attn_out_at_r,        cfg_asym, "base/asym/attn_out_at_r",    t_base_attn_out_at_r);
tamper_test_qwen_base!(p11_4_base_asym_v_at_r,               cfg_asym, "base/asym/v_at_r",           t_base_v_at_r);
tamper_test_qwen_base!(p11_4_base_asym_o_proj_round_poly,    cfg_asym, "base/asym/o_proj.rp[0][0]",  t_base_o_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_asym_g_proj_round_poly,    cfg_asym, "base/asym/g_proj.rp[0][0]",  t_base_g_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_asym_sigmoid_gate_lookup_rp, cfg_asym, "base/asym/sigmoid_gate.lookup.rp[0][0]", t_base_sigmoid_gate_lookup_round_poly);
tamper_test_qwen_base!(p11_4_base_asym_residual1_a_finals,   cfg_asym, "base/asym/residual1.a_finals.0", t_base_residual1_a_finals);
tamper_test_qwen_base!(p11_4_base_asym_norm2_var_round_poly, cfg_asym, "base/asym/norm2.var.rp[0][0]", t_base_norm2_var_round_poly);
tamper_test_qwen_base!(p11_4_base_asym_gate_proj_round_poly, cfg_asym, "base/asym/gate_proj.rp[0][0]", t_base_gate_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_asym_up_proj_round_poly,   cfg_asym, "base/asym/up_proj.rp[0][0]",   t_base_up_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_asym_swiglu_silu_lookup_rp, cfg_asym, "base/asym/swiglu.silu.lookup.rp[0][0]", t_base_swiglu_silu_lookup_round_poly);
tamper_test_qwen_base!(p11_4_base_asym_down_proj_round_poly, cfg_asym, "base/asym/down_proj.rp[0][0]", t_base_down_proj_round_poly);
tamper_test_qwen_base!(p11_4_base_asym_residual2_a_finals,   cfg_asym, "base/asym/residual2.a_finals.0", t_base_residual2_a_finals);
tamper_test_qwen_base!(p11_4_base_asym_w_q_root,             cfg_asym, "base/asym/w_q.root[0]",         t_base_w_q_root);
tamper_test_qwen_base!(p11_4_base_asym_w_k_root,             cfg_asym, "base/asym/w_k.root[0]",         t_base_w_k_root);
tamper_test_qwen_base!(p11_4_base_asym_w_v_root,             cfg_asym, "base/asym/w_v.root[0]",         t_base_w_v_root);
tamper_test_qwen_base!(p11_4_base_asym_w_o_root,             cfg_asym, "base/asym/w_o.root[0]",         t_base_w_o_root);
tamper_test_qwen_base!(p11_4_base_asym_w_g_proj_root,        cfg_asym, "base/asym/w_g_proj.root[0]",    t_base_w_g_proj_root);
tamper_test_qwen_base!(p11_4_base_asym_w_gate_root,          cfg_asym, "base/asym/w_gate.root[0]",      t_base_w_gate_root);
tamper_test_qwen_base!(p11_4_base_asym_w_up_root,            cfg_asym, "base/asym/w_up.root[0]",        t_base_w_up_root);
tamper_test_qwen_base!(p11_4_base_asym_w_down_root,          cfg_asym, "base/asym/w_down.root[0]",      t_base_w_down_root);

// ============================================================================
// Generated tests: QwenLayerProofEF × {GDN, GQA, asymmetric-V}
// ============================================================================

// --- GDN branch ---
tamper_test_qwen_ef!(p11_4_ef_gdn_norm1_var_round_poly, cfg_gdn, "ef/gdn/norm1.var.rp[0][0]", t_ef_norm1_var_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gdn_qkv0_round_poly,      cfg_gdn, "ef/gdn/qkv.0.rp[0][0]",   t_ef_qkv0_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gdn_qkv1_round_poly,      cfg_gdn, "ef/gdn/qkv.1.rp[0][0]",   t_ef_qkv1_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gdn_qkv2_round_poly,      cfg_gdn, "ef/gdn/qkv.2.rp[0][0]",   t_ef_qkv2_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gdn_attn_out_at_r,        cfg_gdn, "ef/gdn/attn_out_at_r",    t_ef_attn_out_at_r);
tamper_test_qwen_ef!(p11_4_ef_gdn_v_at_r,               cfg_gdn, "ef/gdn/v_at_r",           t_ef_v_at_r);
tamper_test_qwen_ef!(p11_4_ef_gdn_o_proj_round_poly,    cfg_gdn, "ef/gdn/o_proj.rp[0][0]",  t_ef_o_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gdn_g_proj_round_poly,    cfg_gdn, "ef/gdn/g_proj.rp[0][0]",  t_ef_g_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gdn_sigmoid_gate_lookup_rp, cfg_gdn, "ef/gdn/sigmoid_gate.lookup.rp[0][0]", t_ef_sigmoid_gate_lookup_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gdn_residual1_a_finals,   cfg_gdn, "ef/gdn/residual1.a_finals.0", t_ef_residual1_a_finals);
tamper_test_qwen_ef!(p11_4_ef_gdn_norm2_var_round_poly, cfg_gdn, "ef/gdn/norm2.var.rp[0][0]", t_ef_norm2_var_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gdn_gate_proj_round_poly, cfg_gdn, "ef/gdn/gate_proj.rp[0][0]", t_ef_gate_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gdn_up_proj_round_poly,   cfg_gdn, "ef/gdn/up_proj.rp[0][0]",   t_ef_up_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gdn_swiglu_silu_lookup_rp, cfg_gdn, "ef/gdn/swiglu.silu.lookup.rp[0][0]", t_ef_swiglu_silu_lookup_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gdn_down_proj_round_poly, cfg_gdn, "ef/gdn/down_proj.rp[0][0]", t_ef_down_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gdn_residual2_a_finals,   cfg_gdn, "ef/gdn/residual2.a_finals.0", t_ef_residual2_a_finals);
tamper_test_qwen_ef!(p11_4_ef_gdn_w_q_root,             cfg_gdn, "ef/gdn/w_q.root[0]",         t_ef_w_q_root);
tamper_test_qwen_ef!(p11_4_ef_gdn_w_k_root,             cfg_gdn, "ef/gdn/w_k.root[0]",         t_ef_w_k_root);
tamper_test_qwen_ef!(p11_4_ef_gdn_w_v_root,             cfg_gdn, "ef/gdn/w_v.root[0]",         t_ef_w_v_root);
tamper_test_qwen_ef!(p11_4_ef_gdn_w_o_root,             cfg_gdn, "ef/gdn/w_o.root[0]",         t_ef_w_o_root);
tamper_test_qwen_ef!(p11_4_ef_gdn_w_g_proj_root,        cfg_gdn, "ef/gdn/w_g_proj.root[0]",    t_ef_w_g_proj_root);
tamper_test_qwen_ef!(p11_4_ef_gdn_w_gate_root,          cfg_gdn, "ef/gdn/w_gate.root[0]",      t_ef_w_gate_root);
tamper_test_qwen_ef!(p11_4_ef_gdn_w_up_root,            cfg_gdn, "ef/gdn/w_up.root[0]",        t_ef_w_up_root);
tamper_test_qwen_ef!(p11_4_ef_gdn_w_down_root,          cfg_gdn, "ef/gdn/w_down.root[0]",      t_ef_w_down_root);

// --- GQA branch ---
tamper_test_qwen_ef!(p11_4_ef_gqa_norm1_var_round_poly, cfg_gqa, "ef/gqa/norm1.var.rp[0][0]", t_ef_norm1_var_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gqa_qkv0_round_poly,      cfg_gqa, "ef/gqa/qkv.0.rp[0][0]",   t_ef_qkv0_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gqa_qkv1_round_poly,      cfg_gqa, "ef/gqa/qkv.1.rp[0][0]",   t_ef_qkv1_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gqa_qkv2_round_poly,      cfg_gqa, "ef/gqa/qkv.2.rp[0][0]",   t_ef_qkv2_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gqa_attn_out_at_r,        cfg_gqa, "ef/gqa/attn_out_at_r",    t_ef_attn_out_at_r);
tamper_test_qwen_ef!(p11_4_ef_gqa_v_at_r,               cfg_gqa, "ef/gqa/v_at_r",           t_ef_v_at_r);
tamper_test_qwen_ef!(p11_4_ef_gqa_o_proj_round_poly,    cfg_gqa, "ef/gqa/o_proj.rp[0][0]",  t_ef_o_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gqa_g_proj_round_poly,    cfg_gqa, "ef/gqa/g_proj.rp[0][0]",  t_ef_g_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gqa_sigmoid_gate_lookup_rp, cfg_gqa, "ef/gqa/sigmoid_gate.lookup.rp[0][0]", t_ef_sigmoid_gate_lookup_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gqa_residual1_a_finals,   cfg_gqa, "ef/gqa/residual1.a_finals.0", t_ef_residual1_a_finals);
tamper_test_qwen_ef!(p11_4_ef_gqa_norm2_var_round_poly, cfg_gqa, "ef/gqa/norm2.var.rp[0][0]", t_ef_norm2_var_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gqa_gate_proj_round_poly, cfg_gqa, "ef/gqa/gate_proj.rp[0][0]", t_ef_gate_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gqa_up_proj_round_poly,   cfg_gqa, "ef/gqa/up_proj.rp[0][0]",   t_ef_up_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gqa_swiglu_silu_lookup_rp, cfg_gqa, "ef/gqa/swiglu.silu.lookup.rp[0][0]", t_ef_swiglu_silu_lookup_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gqa_down_proj_round_poly, cfg_gqa, "ef/gqa/down_proj.rp[0][0]", t_ef_down_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_gqa_residual2_a_finals,   cfg_gqa, "ef/gqa/residual2.a_finals.0", t_ef_residual2_a_finals);
tamper_test_qwen_ef!(p11_4_ef_gqa_w_q_root,             cfg_gqa, "ef/gqa/w_q.root[0]",         t_ef_w_q_root);
tamper_test_qwen_ef!(p11_4_ef_gqa_w_k_root,             cfg_gqa, "ef/gqa/w_k.root[0]",         t_ef_w_k_root);
tamper_test_qwen_ef!(p11_4_ef_gqa_w_v_root,             cfg_gqa, "ef/gqa/w_v.root[0]",         t_ef_w_v_root);
tamper_test_qwen_ef!(p11_4_ef_gqa_w_o_root,             cfg_gqa, "ef/gqa/w_o.root[0]",         t_ef_w_o_root);
tamper_test_qwen_ef!(p11_4_ef_gqa_w_g_proj_root,        cfg_gqa, "ef/gqa/w_g_proj.root[0]",    t_ef_w_g_proj_root);
tamper_test_qwen_ef!(p11_4_ef_gqa_w_gate_root,          cfg_gqa, "ef/gqa/w_gate.root[0]",      t_ef_w_gate_root);
tamper_test_qwen_ef!(p11_4_ef_gqa_w_up_root,            cfg_gqa, "ef/gqa/w_up.root[0]",        t_ef_w_up_root);
tamper_test_qwen_ef!(p11_4_ef_gqa_w_down_root,          cfg_gqa, "ef/gqa/w_down.root[0]",      t_ef_w_down_root);

// --- Asymmetric-V branch ---
tamper_test_qwen_ef!(p11_4_ef_asym_norm1_var_round_poly, cfg_asym, "ef/asym/norm1.var.rp[0][0]", t_ef_norm1_var_round_poly);
tamper_test_qwen_ef!(p11_4_ef_asym_qkv0_round_poly,      cfg_asym, "ef/asym/qkv.0.rp[0][0]",   t_ef_qkv0_round_poly);
tamper_test_qwen_ef!(p11_4_ef_asym_qkv1_round_poly,      cfg_asym, "ef/asym/qkv.1.rp[0][0]",   t_ef_qkv1_round_poly);
tamper_test_qwen_ef!(p11_4_ef_asym_qkv2_round_poly,      cfg_asym, "ef/asym/qkv.2.rp[0][0]",   t_ef_qkv2_round_poly);
tamper_test_qwen_ef!(p11_4_ef_asym_attn_out_at_r,        cfg_asym, "ef/asym/attn_out_at_r",    t_ef_attn_out_at_r);
tamper_test_qwen_ef!(p11_4_ef_asym_v_at_r,               cfg_asym, "ef/asym/v_at_r",           t_ef_v_at_r);
tamper_test_qwen_ef!(p11_4_ef_asym_o_proj_round_poly,    cfg_asym, "ef/asym/o_proj.rp[0][0]",  t_ef_o_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_asym_g_proj_round_poly,    cfg_asym, "ef/asym/g_proj.rp[0][0]",  t_ef_g_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_asym_sigmoid_gate_lookup_rp, cfg_asym, "ef/asym/sigmoid_gate.lookup.rp[0][0]", t_ef_sigmoid_gate_lookup_round_poly);
tamper_test_qwen_ef!(p11_4_ef_asym_residual1_a_finals,   cfg_asym, "ef/asym/residual1.a_finals.0", t_ef_residual1_a_finals);
tamper_test_qwen_ef!(p11_4_ef_asym_norm2_var_round_poly, cfg_asym, "ef/asym/norm2.var.rp[0][0]", t_ef_norm2_var_round_poly);
tamper_test_qwen_ef!(p11_4_ef_asym_gate_proj_round_poly, cfg_asym, "ef/asym/gate_proj.rp[0][0]", t_ef_gate_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_asym_up_proj_round_poly,   cfg_asym, "ef/asym/up_proj.rp[0][0]",   t_ef_up_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_asym_swiglu_silu_lookup_rp, cfg_asym, "ef/asym/swiglu.silu.lookup.rp[0][0]", t_ef_swiglu_silu_lookup_round_poly);
tamper_test_qwen_ef!(p11_4_ef_asym_down_proj_round_poly, cfg_asym, "ef/asym/down_proj.rp[0][0]", t_ef_down_proj_round_poly);
tamper_test_qwen_ef!(p11_4_ef_asym_residual2_a_finals,   cfg_asym, "ef/asym/residual2.a_finals.0", t_ef_residual2_a_finals);
tamper_test_qwen_ef!(p11_4_ef_asym_w_q_root,             cfg_asym, "ef/asym/w_q.root[0]",         t_ef_w_q_root);
tamper_test_qwen_ef!(p11_4_ef_asym_w_k_root,             cfg_asym, "ef/asym/w_k.root[0]",         t_ef_w_k_root);
tamper_test_qwen_ef!(p11_4_ef_asym_w_v_root,             cfg_asym, "ef/asym/w_v.root[0]",         t_ef_w_v_root);
tamper_test_qwen_ef!(p11_4_ef_asym_w_o_root,             cfg_asym, "ef/asym/w_o.root[0]",         t_ef_w_o_root);
tamper_test_qwen_ef!(p11_4_ef_asym_w_g_proj_root,        cfg_asym, "ef/asym/w_g_proj.root[0]",    t_ef_w_g_proj_root);
tamper_test_qwen_ef!(p11_4_ef_asym_w_gate_root,          cfg_asym, "ef/asym/w_gate.root[0]",      t_ef_w_gate_root);
tamper_test_qwen_ef!(p11_4_ef_asym_w_up_root,            cfg_asym, "ef/asym/w_up.root[0]",        t_ef_w_up_root);
tamper_test_qwen_ef!(p11_4_ef_asym_w_down_root,          cfg_asym, "ef/asym/w_down.root[0]",      t_ef_w_down_root);
