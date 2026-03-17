//! Algebraic RMSNorm proof over M31.
//!
//! RMSNorm(x) = γ * x / sqrt(mean(x²) + ε)
//!
//! Simpler than LayerNorm: no mean subtraction, no beta.
//! Used by Llama, Mistral, Qwen, and most modern transformers.
//!
//! Algebraic approach (squared, no sqrt in-circuit):
//! - Prover claims sum_sq = Σ x[i]² as witness
//! - Define g[i] = γ[i] * x[i]  (in M31, exact)
//! - Define h[i] = y[i]  (the claimed output)
//! - Check 1: Σ x[i]² = sum_sq via product sumcheck on (x, x)
//! - Check 2: sum_sq * Σ eq(ρ,i) * h[i]² = d * Σ eq(ρ,i) * g[i]²
//!   This holds because h[i] = g[i] * r where r² = d/sum_sq.
//!
//! The squared approach avoids needing sqrt to exist in M31 (QR issues).

use p3_field::{AbstractField, PrimeField32};
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::field::common::{log2_ceil, compute_eq_at_point, compute_eq_at_point_ef};
use crate::field::m31_ops::*;
use crate::field::m31_ops::{eq_evals_ef, mle_evaluate_ef};
use crate::proving::sumcheck::{self, EF, EFElement, SumcheckProof, SumcheckProofEF, Transcript};
use crate::proving::weight_commitment::{
    commit_weights_fast, prove_mle_eval_no_merkle, prove_mle_eval_no_merkle_ef_base,
    verify_mle_eval, verify_mle_eval_ef, MleEvalProof, MleEvalProofEF, WeightCommitment,
};

type F = Mersenne31;

/// Proof that y = RMSNorm(x, γ).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RmsNormProof {
    /// Variance check: Σ x[i]² = sum_sq
    pub var_proof: SumcheckProof,
    pub var_finals: (u32, u32), // (x_at_s_a, x_at_s_b)
    pub sum_sq_value: u32,

    /// h-squared: S_h = Σ eq(ρ,i) · h[i]²
    pub h_sq_proof: SumcheckProof,
    pub h_sq_finals: (u32, u32, u32), // (eq_at_s, h_at_s_a, h_at_s_b)
    pub s_h: u32,

    /// g-squared: S_g = Σ eq(ρ,i) · g[i]²
    pub g_sq_proof: SumcheckProof,
    pub g_sq_finals: (u32, u32, u32), // (eq_at_s, g_at_s_a, g_at_s_b)
    pub s_g: u32,

    /// g-consistency: prove g[i] = γ[i] * x[i]
    /// Product sumcheck: Σ eq(ρ2,i) · g[i] = t_g
    pub g_prod_proof: SumcheckProof,
    pub g_prod_finals: (u32, u32),
    /// Triple sumcheck: Σ eq(ρ2,i) · γ[i] · x[i] = t_g
    pub g_triple_proof: SumcheckProof,
    pub g_triple_finals: (u32, u32, u32),
    pub t_g_claimed: u32,

    /// Commitments
    pub x_commitment: WeightCommitment,
    pub gamma_commitment: WeightCommitment,
    pub g_commitment: WeightCommitment,
    pub h_commitment: WeightCommitment,

    /// MLE eval proofs
    pub x_var_eval_proof: MleEvalProof,
    pub x_at_var_s: u32,
    pub g_prod_eval_proof: MleEvalProof,
    pub g_at_prod_s: u32,
    pub x_triple_eval_proof: MleEvalProof,
    pub x_at_triple_s: u32,
    pub gamma_triple_eval_proof: MleEvalProof,
    pub gamma_at_triple_s: u32,
    pub h_sq_eval_proof: MleEvalProof,
    pub h_at_sq_s: u32,
    pub g_sq_eval_proof: MleEvalProof,
    pub g_at_gsq_s: u32,
}

/// Prove RMSNorm: y = γ * x / sqrt(mean(x²) + ε).
///
/// Uses squared constraint (no sqrt needed in M31):
/// sum_sq · Σ eq·h² = d · Σ eq·g² where h=y, g=γ·x.
pub fn prove_rmsnorm(
    x: &[F],
    gamma: &[F],
    y: &[F],
    transcript: &mut Transcript,
) -> RmsNormProof {
    let d = x.len();
    let log_n = log2_ceil(d);
    let n_pad = 1 << log_n;

    // Pad vectors
    let mut x_pad = x.to_vec();
    x_pad.resize(n_pad, F::zero());
    let mut gamma_pad = gamma.to_vec();
    gamma_pad.resize(n_pad, F::zero());
    let mut y_pad = y.to_vec();
    y_pad.resize(n_pad, F::zero());

    // g = gamma * x (in M31, exact)
    let mut g_pad = vec![F::zero(); n_pad];
    for i in 0..d {
        g_pad[i] = gamma_pad[i] * x_pad[i];
    }

    // h = y (the claimed output)
    let h_pad = y_pad.clone();

    let sum_sq: F = x_pad.iter().map(|&v| v * v).sum();

    // Commitments (fast — verifier recomputes independently, no Merkle needed)
    let x_commitment = commit_weights_fast(&x_pad);
    let gamma_commitment = commit_weights_fast(&gamma_pad);
    let g_commitment = commit_weights_fast(&g_pad);
    let h_commitment = commit_weights_fast(&h_pad);

    // Absorb
    transcript.absorb_bytes(&x_commitment.root);
    transcript.absorb_bytes(&gamma_commitment.root);
    transcript.absorb_bytes(&g_commitment.root);
    transcript.absorb_bytes(&h_commitment.root);
    transcript.absorb(sum_sq.as_canonical_u32());

    // --- Step 1: Variance check (Σ x[i]² = sum_sq) ---
    let (var_proof, x_var_a, x_var_b) =
        sumcheck::prove_product_best(&x_pad, &x_pad, log_n, transcript);

    let s_var: Vec<F> = var_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let x_at_var_s = mle_evaluate(&x_pad, &s_var);
    let mut eval_t1 = Transcript::new(b"rmsnorm-x-var");
    eval_t1.absorb_bytes(&x_commitment.root);
    let (_, x_var_eval_proof) = prove_mle_eval_no_merkle(&x_pad, &s_var, &mut eval_t1);

    // --- Step 2: g-consistency check ---
    // Prove Σ eq(ρ,i) · g[i] == Σ eq(ρ,i) · γ[i] · x[i]
    let gc_point = transcript.squeeze_many(log_n);
    let eq_gc = eq_evals(&gc_point);

    let t_g: F = eq_gc.iter().zip(g_pad.iter()).map(|(&e, &g)| e * g).sum();

    // Product sumcheck: Σ eq · g = t_g
    let (g_prod_proof, _eq_at_gp, g_at_gp) =
        sumcheck::prove_product_best(&eq_gc, &g_pad, log_n, transcript);

    let s_gp: Vec<F> = g_prod_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let g_at_prod_s = mle_evaluate(&g_pad, &s_gp);
    let mut eval_t2 = Transcript::new(b"rmsnorm-g-prod");
    eval_t2.absorb_bytes(&g_commitment.root);
    let (_, g_prod_eval_proof) = prove_mle_eval_no_merkle(&g_pad, &s_gp, &mut eval_t2);

    // Triple sumcheck: Σ eq · γ · x = t_g
    let (g_triple_proof, eq_at_gt, gamma_at_gt, x_at_gt) =
        sumcheck::prove_triple_best(&eq_gc, &gamma_pad, &x_pad, log_n, transcript);

    let s_gt: Vec<F> = g_triple_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let x_at_triple_s = mle_evaluate(&x_pad, &s_gt);
    let mut eval_t3 = Transcript::new(b"rmsnorm-x-triple");
    eval_t3.absorb_bytes(&x_commitment.root);
    let (_, x_triple_eval_proof) = prove_mle_eval_no_merkle(&x_pad, &s_gt, &mut eval_t3);
    let gamma_at_triple_s = mle_evaluate(&gamma_pad, &s_gt);
    let mut eval_t4 = Transcript::new(b"rmsnorm-gamma-triple");
    eval_t4.absorb_bytes(&gamma_commitment.root);
    let (_, gamma_triple_eval_proof) = prove_mle_eval_no_merkle(&gamma_pad, &s_gt, &mut eval_t4);

    // --- Step 3: Squared checks ---
    let sq_point = transcript.squeeze_many(log_n);
    let eq_sq = eq_evals(&sq_point);

    // h-squared: S_h = Σ eq · h · h
    let s_h_val: F = eq_sq.iter().zip(h_pad.iter()).map(|(&e, &h)| e * h * h).sum();
    let (h_sq_proof, eq_at_hs, h_at_hs_a, h_at_hs_b) =
        sumcheck::prove_triple_best(&eq_sq, &h_pad, &h_pad, log_n, transcript);

    let s_hs: Vec<F> = h_sq_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let h_at_sq_s = mle_evaluate(&h_pad, &s_hs);
    let mut eval_t5 = Transcript::new(b"rmsnorm-h-sq");
    eval_t5.absorb_bytes(&h_commitment.root);
    let (_, h_sq_eval_proof) = prove_mle_eval_no_merkle(&h_pad, &s_hs, &mut eval_t5);

    // g-squared: S_g = Σ eq · g · g
    let s_g_val: F = eq_sq.iter().zip(g_pad.iter()).map(|(&e, &g)| e * g * g).sum();
    let (g_sq_proof, _eq_at_gs, g_at_gs_a, g_at_gs_b) =
        sumcheck::prove_triple_best(&eq_sq, &g_pad, &g_pad, log_n, transcript);

    let s_gs: Vec<F> = g_sq_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let g_at_gsq_s = mle_evaluate(&g_pad, &s_gs);
    let mut eval_t6 = Transcript::new(b"rmsnorm-g-sq");
    eval_t6.absorb_bytes(&g_commitment.root);
    let (_, g_sq_eval_proof) = prove_mle_eval_no_merkle(&g_pad, &s_gs, &mut eval_t6);

    RmsNormProof {
        var_proof,
        var_finals: (x_var_a.as_canonical_u32(), x_var_b.as_canonical_u32()),
        sum_sq_value: sum_sq.as_canonical_u32(),
        h_sq_proof,
        h_sq_finals: (eq_at_hs.as_canonical_u32(), h_at_hs_a.as_canonical_u32(), h_at_hs_b.as_canonical_u32()),
        s_h: s_h_val.as_canonical_u32(),
        g_sq_proof,
        g_sq_finals: (_eq_at_gs.as_canonical_u32(), g_at_gs_a.as_canonical_u32(), g_at_gs_b.as_canonical_u32()),
        s_g: s_g_val.as_canonical_u32(),
        g_prod_proof,
        g_prod_finals: (_eq_at_gp.as_canonical_u32(), g_at_gp.as_canonical_u32()),
        g_triple_proof,
        g_triple_finals: (eq_at_gt.as_canonical_u32(), gamma_at_gt.as_canonical_u32(), x_at_gt.as_canonical_u32()),
        t_g_claimed: t_g.as_canonical_u32(),
        x_commitment,
        gamma_commitment,
        g_commitment,
        h_commitment,
        x_var_eval_proof,
        x_at_var_s: x_at_var_s.as_canonical_u32(),
        g_prod_eval_proof,
        g_at_prod_s: g_at_prod_s.as_canonical_u32(),
        x_triple_eval_proof,
        x_at_triple_s: x_at_triple_s.as_canonical_u32(),
        gamma_triple_eval_proof,
        gamma_at_triple_s: gamma_at_triple_s.as_canonical_u32(),
        h_sq_eval_proof,
        h_at_sq_s: h_at_sq_s.as_canonical_u32(),
        g_sq_eval_proof,
        g_at_gsq_s: g_at_gsq_s.as_canonical_u32(),
    }
}

/// Verify an RMSNorm proof.
///
/// Checks:
/// 1. Σ x[i]² = sum_sq (variance via product sumcheck)
/// 2. g[i] = γ[i] · x[i] (g-consistency)
/// 3. sum_sq · Σ eq·h² = d · Σ eq·g² (squared output constraint)
/// 4. h commitment matches y
/// 5. All MLE eval proofs
pub fn verify_rmsnorm(
    proof: &RmsNormProof,
    y: &[F],
    d: usize,
    transcript: &mut Transcript,
) -> bool {
    let log_n = log2_ceil(d);
    let n_pad = 1 << log_n;

    let sum_sq = F::from_canonical_u32(proof.sum_sq_value);
    let d_field = F::from_canonical_u32(d as u32);

    // Absorb (must match prover)
    transcript.absorb_bytes(&proof.x_commitment.root);
    transcript.absorb_bytes(&proof.gamma_commitment.root);
    transcript.absorb_bytes(&proof.g_commitment.root);
    transcript.absorb_bytes(&proof.h_commitment.root);
    transcript.absorb(proof.sum_sq_value);

    // --- Variance check ---
    let x_var_a = F::from_canonical_u32(proof.var_finals.0);
    let x_var_b = F::from_canonical_u32(proof.var_finals.1);
    if !sumcheck::verify_product(sum_sq, &proof.var_proof, log_n, x_var_a, x_var_b, transcript) {
        eprintln!("RmsNorm: variance check failed");
        return false;
    }
    if x_var_a != x_var_b {
        eprintln!("RmsNorm: var finals mismatch (x·x but a != b)");
        return false;
    }
    let x_at_var_s = F::from_canonical_u32(proof.x_at_var_s);
    if x_at_var_s != x_var_a { return false; }
    let s_var: Vec<F> = proof.var_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let mut eval_t1 = Transcript::new(b"rmsnorm-x-var");
    eval_t1.absorb_bytes(&proof.x_commitment.root);
    if !verify_mle_eval(&proof.x_commitment, x_at_var_s, &s_var, &proof.x_var_eval_proof, &mut eval_t1) {
        eprintln!("RmsNorm: x var MLE eval failed");
        return false;
    }

    // --- g-consistency check ---
    let gc_point = transcript.squeeze_many(log_n);
    let t_g = F::from_canonical_u32(proof.t_g_claimed);

    // Product sumcheck: Σ eq · g = t_g
    let (eq_at_gp, g_at_gp) = (
        F::from_canonical_u32(proof.g_prod_finals.0),
        F::from_canonical_u32(proof.g_prod_finals.1),
    );
    if !sumcheck::verify_product(t_g, &proof.g_prod_proof, log_n, eq_at_gp, g_at_gp, transcript) {
        eprintln!("RmsNorm: g product sumcheck failed");
        return false;
    }
    let s_gp: Vec<F> = proof.g_prod_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let eq_gp_expected = compute_eq_at_point(&gc_point, &s_gp);
    if eq_gp_expected != eq_at_gp {
        eprintln!("RmsNorm: eq mismatch in g product");
        return false;
    }
    let g_at_prod_s = F::from_canonical_u32(proof.g_at_prod_s);
    if g_at_prod_s != g_at_gp { return false; }
    let mut eval_t2 = Transcript::new(b"rmsnorm-g-prod");
    eval_t2.absorb_bytes(&proof.g_commitment.root);
    if !verify_mle_eval(&proof.g_commitment, g_at_prod_s, &s_gp, &proof.g_prod_eval_proof, &mut eval_t2) {
        eprintln!("RmsNorm: g prod MLE eval failed");
        return false;
    }

    // Triple sumcheck: Σ eq · γ · x = t_g
    let (eq_at_gt, gamma_at_gt, x_at_gt) = (
        F::from_canonical_u32(proof.g_triple_finals.0),
        F::from_canonical_u32(proof.g_triple_finals.1),
        F::from_canonical_u32(proof.g_triple_finals.2),
    );
    if !sumcheck::verify_triple(t_g, &proof.g_triple_proof, log_n, eq_at_gt, gamma_at_gt, x_at_gt, transcript) {
        eprintln!("RmsNorm: g triple sumcheck failed");
        return false;
    }
    let s_gt: Vec<F> = proof.g_triple_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let eq_gt_expected = compute_eq_at_point(&gc_point, &s_gt);
    if eq_gt_expected != eq_at_gt {
        eprintln!("RmsNorm: eq mismatch in g triple");
        return false;
    }
    let x_at_triple_s = F::from_canonical_u32(proof.x_at_triple_s);
    if x_at_triple_s != x_at_gt { return false; }
    let mut eval_t3 = Transcript::new(b"rmsnorm-x-triple");
    eval_t3.absorb_bytes(&proof.x_commitment.root);
    if !verify_mle_eval(&proof.x_commitment, x_at_triple_s, &s_gt, &proof.x_triple_eval_proof, &mut eval_t3) {
        eprintln!("RmsNorm: x triple MLE eval failed");
        return false;
    }
    let gamma_at_triple_s = F::from_canonical_u32(proof.gamma_at_triple_s);
    if gamma_at_triple_s != gamma_at_gt { return false; }
    let mut eval_t4 = Transcript::new(b"rmsnorm-gamma-triple");
    eval_t4.absorb_bytes(&proof.gamma_commitment.root);
    if !verify_mle_eval(&proof.gamma_commitment, gamma_at_triple_s, &s_gt, &proof.gamma_triple_eval_proof, &mut eval_t4) {
        eprintln!("RmsNorm: gamma triple MLE eval failed");
        return false;
    }

    // --- Squared checks ---
    let sq_point = transcript.squeeze_many(log_n);
    let s_h_val = F::from_canonical_u32(proof.s_h);
    let s_g_val = F::from_canonical_u32(proof.s_g);

    // h-squared: S_h = Σ eq · h · h
    let (eq_at_hs, h_at_hs_a, h_at_hs_b) = (
        F::from_canonical_u32(proof.h_sq_finals.0),
        F::from_canonical_u32(proof.h_sq_finals.1),
        F::from_canonical_u32(proof.h_sq_finals.2),
    );
    if !sumcheck::verify_triple(s_h_val, &proof.h_sq_proof, log_n, eq_at_hs, h_at_hs_a, h_at_hs_b, transcript) {
        eprintln!("RmsNorm: h-squared sumcheck failed");
        return false;
    }
    let s_hs: Vec<F> = proof.h_sq_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let eq_hs_expected = compute_eq_at_point(&sq_point, &s_hs);
    if eq_hs_expected != eq_at_hs {
        eprintln!("RmsNorm: eq mismatch in h-squared");
        return false;
    }
    if h_at_hs_a != h_at_hs_b {
        eprintln!("RmsNorm: h-squared finals mismatch");
        return false;
    }
    let h_at_sq_s = F::from_canonical_u32(proof.h_at_sq_s);
    if h_at_sq_s != h_at_hs_a { return false; }
    let mut eval_t5 = Transcript::new(b"rmsnorm-h-sq");
    eval_t5.absorb_bytes(&proof.h_commitment.root);
    if !verify_mle_eval(&proof.h_commitment, h_at_sq_s, &s_hs, &proof.h_sq_eval_proof, &mut eval_t5) {
        eprintln!("RmsNorm: h sq MLE eval failed");
        return false;
    }

    // Verify h commitment matches y
    let mut y_pad = y.to_vec();
    y_pad.resize(n_pad, F::zero());
    let h_commitment_expected = commit_weights_fast(&y_pad);
    if h_commitment_expected.root != proof.h_commitment.root {
        eprintln!("RmsNorm: h commitment mismatch (y)");
        return false;
    }

    // g-squared: S_g = Σ eq · g · g
    let (eq_at_gs, g_at_gs_a, g_at_gs_b) = (
        F::from_canonical_u32(proof.g_sq_finals.0),
        F::from_canonical_u32(proof.g_sq_finals.1),
        F::from_canonical_u32(proof.g_sq_finals.2),
    );
    if !sumcheck::verify_triple(s_g_val, &proof.g_sq_proof, log_n, eq_at_gs, g_at_gs_a, g_at_gs_b, transcript) {
        eprintln!("RmsNorm: g-squared sumcheck failed");
        return false;
    }
    let s_gs: Vec<F> = proof.g_sq_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let eq_gs_expected = compute_eq_at_point(&sq_point, &s_gs);
    if eq_gs_expected != eq_at_gs {
        eprintln!("RmsNorm: eq mismatch in g-squared");
        return false;
    }
    if g_at_gs_a != g_at_gs_b {
        eprintln!("RmsNorm: g-squared finals mismatch");
        return false;
    }
    let g_at_gsq_s = F::from_canonical_u32(proof.g_at_gsq_s);
    if g_at_gsq_s != g_at_gs_a { return false; }
    let mut eval_t6 = Transcript::new(b"rmsnorm-g-sq");
    eval_t6.absorb_bytes(&proof.g_commitment.root);
    if !verify_mle_eval(&proof.g_commitment, g_at_gsq_s, &s_gs, &proof.g_sq_eval_proof, &mut eval_t6) {
        eprintln!("RmsNorm: g sq MLE eval failed");
        return false;
    }

    // --- Final check: sum_sq · S_h == d · S_g ---
    if sum_sq * s_h_val != d_field * s_g_val {
        eprintln!("RmsNorm: sum_sq·S_h != d·S_g (squared output check failed)");
        return false;
    }

    true
}

// =============================================================================
// Extension-field (EF) versions — 124-bit Fiat-Shamir soundness
// =============================================================================

/// Proof that y = RMSNorm(x, γ) with extension-field challenges.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RmsNormProofEF {
    /// Variance check: Σ x[i]² = sum_sq
    pub var_proof: SumcheckProofEF,
    pub var_finals: (EFElement, EFElement), // (x_at_s_a, x_at_s_b)
    pub sum_sq_value: u32, // base field — Σ x² is computed over F

    /// h-squared: S_h = Σ eq(ρ,i) · h[i]²
    pub h_sq_proof: SumcheckProofEF,
    pub h_sq_finals: (EFElement, EFElement, EFElement),
    pub s_h: EFElement,

    /// g-squared: S_g = Σ eq(ρ,i) · g[i]²
    pub g_sq_proof: SumcheckProofEF,
    pub g_sq_finals: (EFElement, EFElement, EFElement),
    pub s_g: EFElement,

    /// g-consistency: prove g[i] = γ[i] * x[i]
    /// Product sumcheck: Σ eq(ρ2,i) · g[i] = t_g
    pub g_prod_proof: SumcheckProofEF,
    pub g_prod_finals: (EFElement, EFElement),
    /// Triple sumcheck: Σ eq(ρ2,i) · γ[i] · x[i] = t_g
    pub g_triple_proof: SumcheckProofEF,
    pub g_triple_finals: (EFElement, EFElement, EFElement),
    pub t_g_claimed: EFElement,

    /// Commitments (base field — same as non-EF)
    pub x_commitment: WeightCommitment,
    pub gamma_commitment: WeightCommitment,
    pub g_commitment: WeightCommitment,
    pub h_commitment: WeightCommitment,

    /// MLE eval proofs (EF)
    pub x_var_eval_proof: MleEvalProofEF,
    pub x_at_var_s: EFElement,
    pub g_prod_eval_proof: MleEvalProofEF,
    pub g_at_prod_s: EFElement,
    pub x_triple_eval_proof: MleEvalProofEF,
    pub x_at_triple_s: EFElement,
    pub gamma_triple_eval_proof: MleEvalProofEF,
    pub gamma_at_triple_s: EFElement,
    pub h_sq_eval_proof: MleEvalProofEF,
    pub h_at_sq_s: EFElement,
    pub g_sq_eval_proof: MleEvalProofEF,
    pub g_at_gsq_s: EFElement,
}

/// Prove RMSNorm with extension-field challenges (124-bit soundness).
///
/// Same algebraic structure as `prove_rmsnorm` but all Fiat-Shamir
/// challenges are drawn from EF, providing 124-bit soundness.
pub fn prove_rmsnorm_ef(
    x: &[F],
    gamma: &[F],
    y: &[F],
    transcript: &mut Transcript,
) -> RmsNormProofEF {
    let d = x.len();
    let log_n = log2_ceil(d);
    let n_pad = 1 << log_n;

    // Pad vectors
    let mut x_pad = x.to_vec();
    x_pad.resize(n_pad, F::zero());
    let mut gamma_pad = gamma.to_vec();
    gamma_pad.resize(n_pad, F::zero());
    let mut y_pad = y.to_vec();
    y_pad.resize(n_pad, F::zero());

    // g = gamma * x (in M31, exact)
    let mut g_pad = vec![F::zero(); n_pad];
    for i in 0..d {
        g_pad[i] = gamma_pad[i] * x_pad[i];
    }

    // h = y (the claimed output)
    let h_pad = y_pad.clone();

    let sum_sq: F = x_pad.iter().map(|&v| v * v).sum();

    // Commitments
    let x_commitment = commit_weights_fast(&x_pad);
    let gamma_commitment = commit_weights_fast(&gamma_pad);
    let g_commitment = commit_weights_fast(&g_pad);
    let h_commitment = commit_weights_fast(&h_pad);

    // Absorb
    transcript.absorb_bytes(&x_commitment.root);
    transcript.absorb_bytes(&gamma_commitment.root);
    transcript.absorb_bytes(&g_commitment.root);
    transcript.absorb_bytes(&h_commitment.root);
    transcript.absorb(sum_sq.as_canonical_u32());

    // --- Step 1: Variance check (Σ x[i]² = sum_sq) ---
    // Both inputs are base field → use prove_product_ef
    let (var_proof, x_var_a, x_var_b) =
        sumcheck::prove_product_ef(&x_pad, &x_pad, log_n, transcript);

    let s_var: Vec<EF> = var_proof.challenges.iter().map(|v| v.to_ef()).collect();
    let x_at_var_s = mle_evaluate_ef(&x_pad, &s_var);
    let mut eval_t1 = Transcript::new(b"rmsnorm-x-var");
    eval_t1.absorb_bytes(&x_commitment.root);
    let (_, x_var_eval_proof) = prove_mle_eval_no_merkle_ef_base(&x_pad, &s_var, &mut eval_t1);

    // --- Step 2: g-consistency check ---
    let gc_point = transcript.squeeze_ef_many(log_n);
    let eq_gc = eq_evals_ef(&gc_point);

    // t_g = Σ eq_gc[i] * g_pad[i]  (EF since eq_gc is EF)
    let t_g: EF = eq_gc.iter().zip(g_pad.iter()).map(|(&e, &g)| e * f_to_ef(g)).sum();

    // Product sumcheck: Σ eq · g = t_g
    // eq_gc is EF, g_pad is F → convert g_pad to EF, use full-EF
    let g_ef: Vec<EF> = g_pad.iter().map(|&v| f_to_ef(v)).collect();
    let (g_prod_proof, _eq_at_gp, g_at_gp) =
        sumcheck::prove_product_ef_full(&eq_gc, &g_ef, log_n, transcript);

    let s_gp: Vec<EF> = g_prod_proof.challenges.iter().map(|v| v.to_ef()).collect();
    let g_at_prod_s = mle_evaluate_ef(&g_pad, &s_gp);
    let mut eval_t2 = Transcript::new(b"rmsnorm-g-prod");
    eval_t2.absorb_bytes(&g_commitment.root);
    let (_, g_prod_eval_proof) = prove_mle_eval_no_merkle_ef_base(&g_pad, &s_gp, &mut eval_t2);

    // Triple sumcheck: Σ eq · γ · x = t_g
    // eq_gc is EF, gamma_pad and x_pad are F → convert all to EF, use full-EF
    let gamma_ef: Vec<EF> = gamma_pad.iter().map(|&v| f_to_ef(v)).collect();
    let x_ef: Vec<EF> = x_pad.iter().map(|&v| f_to_ef(v)).collect();
    let (g_triple_proof, eq_at_gt, gamma_at_gt, x_at_gt) =
        sumcheck::prove_triple_ef_full(&eq_gc, &gamma_ef, &x_ef, log_n, transcript);

    let s_gt: Vec<EF> = g_triple_proof.challenges.iter().map(|v| v.to_ef()).collect();
    let x_at_triple_s = mle_evaluate_ef(&x_pad, &s_gt);
    let mut eval_t3 = Transcript::new(b"rmsnorm-x-triple");
    eval_t3.absorb_bytes(&x_commitment.root);
    let (_, x_triple_eval_proof) = prove_mle_eval_no_merkle_ef_base(&x_pad, &s_gt, &mut eval_t3);
    let gamma_at_triple_s = mle_evaluate_ef(&gamma_pad, &s_gt);
    let mut eval_t4 = Transcript::new(b"rmsnorm-gamma-triple");
    eval_t4.absorb_bytes(&gamma_commitment.root);
    let (_, gamma_triple_eval_proof) = prove_mle_eval_no_merkle_ef_base(&gamma_pad, &s_gt, &mut eval_t4);

    // --- Step 3: Squared checks ---
    let sq_point = transcript.squeeze_ef_many(log_n);
    let eq_sq = eq_evals_ef(&sq_point);

    // h-squared: S_h = Σ eq · h · h
    let s_h_val: EF = eq_sq.iter().zip(h_pad.iter()).map(|(&e, &h)| e * f_to_ef(h) * f_to_ef(h)).sum();
    // eq_sq is EF, h_pad is F → convert to EF, use full-EF
    let h_ef: Vec<EF> = h_pad.iter().map(|&v| f_to_ef(v)).collect();
    let (h_sq_proof, eq_at_hs, h_at_hs_a, h_at_hs_b) =
        sumcheck::prove_triple_ef_full(&eq_sq, &h_ef, &h_ef, log_n, transcript);

    let s_hs: Vec<EF> = h_sq_proof.challenges.iter().map(|v| v.to_ef()).collect();
    let h_at_sq_s = mle_evaluate_ef(&h_pad, &s_hs);
    let mut eval_t5 = Transcript::new(b"rmsnorm-h-sq");
    eval_t5.absorb_bytes(&h_commitment.root);
    let (_, h_sq_eval_proof) = prove_mle_eval_no_merkle_ef_base(&h_pad, &s_hs, &mut eval_t5);

    // g-squared: S_g = Σ eq · g · g
    let s_g_val: EF = eq_sq.iter().zip(g_pad.iter()).map(|(&e, &g)| e * f_to_ef(g) * f_to_ef(g)).sum();
    let (g_sq_proof, _eq_at_gs, g_at_gs_a, g_at_gs_b) =
        sumcheck::prove_triple_ef_full(&eq_sq, &g_ef, &g_ef, log_n, transcript);

    let s_gs: Vec<EF> = g_sq_proof.challenges.iter().map(|v| v.to_ef()).collect();
    let g_at_gsq_s = mle_evaluate_ef(&g_pad, &s_gs);
    let mut eval_t6 = Transcript::new(b"rmsnorm-g-sq");
    eval_t6.absorb_bytes(&g_commitment.root);
    let (_, g_sq_eval_proof) = prove_mle_eval_no_merkle_ef_base(&g_pad, &s_gs, &mut eval_t6);

    RmsNormProofEF {
        var_proof,
        var_finals: (EFElement::from_ef(x_var_a), EFElement::from_ef(x_var_b)),
        sum_sq_value: sum_sq.as_canonical_u32(),
        h_sq_proof,
        h_sq_finals: (EFElement::from_ef(eq_at_hs), EFElement::from_ef(h_at_hs_a), EFElement::from_ef(h_at_hs_b)),
        s_h: EFElement::from_ef(s_h_val),
        g_sq_proof,
        g_sq_finals: (EFElement::from_ef(_eq_at_gs), EFElement::from_ef(g_at_gs_a), EFElement::from_ef(g_at_gs_b)),
        s_g: EFElement::from_ef(s_g_val),
        g_prod_proof,
        g_prod_finals: (EFElement::from_ef(_eq_at_gp), EFElement::from_ef(g_at_gp)),
        g_triple_proof,
        g_triple_finals: (EFElement::from_ef(eq_at_gt), EFElement::from_ef(gamma_at_gt), EFElement::from_ef(x_at_gt)),
        t_g_claimed: EFElement::from_ef(t_g),
        x_commitment,
        gamma_commitment,
        g_commitment,
        h_commitment,
        x_var_eval_proof,
        x_at_var_s: EFElement::from_ef(x_at_var_s),
        g_prod_eval_proof,
        g_at_prod_s: EFElement::from_ef(g_at_prod_s),
        x_triple_eval_proof,
        x_at_triple_s: EFElement::from_ef(x_at_triple_s),
        gamma_triple_eval_proof,
        gamma_at_triple_s: EFElement::from_ef(gamma_at_triple_s),
        h_sq_eval_proof,
        h_at_sq_s: EFElement::from_ef(h_at_sq_s),
        g_sq_eval_proof,
        g_at_gsq_s: EFElement::from_ef(g_at_gsq_s),
    }
}

/// Verify an RMSNorm proof with extension-field challenges.
pub fn verify_rmsnorm_ef(
    proof: &RmsNormProofEF,
    y: &[F],
    d: usize,
    transcript: &mut Transcript,
) -> bool {
    let log_n = log2_ceil(d);
    let n_pad = 1 << log_n;

    let sum_sq = F::from_canonical_u32(proof.sum_sq_value);
    let sum_sq_ef = f_to_ef(sum_sq);
    let d_field_ef = f_to_ef(F::from_canonical_u32(d as u32));

    // Absorb (must match prover)
    transcript.absorb_bytes(&proof.x_commitment.root);
    transcript.absorb_bytes(&proof.gamma_commitment.root);
    transcript.absorb_bytes(&proof.g_commitment.root);
    transcript.absorb_bytes(&proof.h_commitment.root);
    transcript.absorb(proof.sum_sq_value);

    // --- Variance check ---
    let x_var_a = proof.var_finals.0.to_ef();
    let x_var_b = proof.var_finals.1.to_ef();
    let (var_ok, s_var) = sumcheck::verify_product_ef_with_challenges(
        sum_sq_ef, &proof.var_proof, x_var_a, x_var_b, transcript);
    if !var_ok {
        eprintln!("RmsNormEF: variance check failed");
        return false;
    }
    if x_var_a != x_var_b {
        eprintln!("RmsNormEF: var finals mismatch (x·x but a != b)");
        return false;
    }
    let x_at_var_s = proof.x_at_var_s.to_ef();
    if x_at_var_s != x_var_a { return false; }
    let mut eval_t1 = Transcript::new(b"rmsnorm-x-var");
    eval_t1.absorb_bytes(&proof.x_commitment.root);
    if !verify_mle_eval_ef(&proof.x_commitment, x_at_var_s, &s_var, &proof.x_var_eval_proof, &mut eval_t1) {
        eprintln!("RmsNormEF: x var MLE eval failed");
        return false;
    }

    // --- g-consistency check ---
    let gc_point = transcript.squeeze_ef_many(log_n);
    let t_g = proof.t_g_claimed.to_ef();

    // Product sumcheck: Σ eq · g = t_g
    let (eq_at_gp, g_at_gp) = (
        proof.g_prod_finals.0.to_ef(),
        proof.g_prod_finals.1.to_ef(),
    );
    let (gp_ok, s_gp) = sumcheck::verify_product_ef_with_challenges(
        t_g, &proof.g_prod_proof, eq_at_gp, g_at_gp, transcript);
    if !gp_ok {
        eprintln!("RmsNormEF: g product sumcheck failed");
        return false;
    }
    let eq_gp_expected = compute_eq_at_point_ef(&gc_point, &s_gp);
    if eq_gp_expected != eq_at_gp {
        eprintln!("RmsNormEF: eq mismatch in g product");
        return false;
    }
    let g_at_prod_s = proof.g_at_prod_s.to_ef();
    if g_at_prod_s != g_at_gp { return false; }
    let mut eval_t2 = Transcript::new(b"rmsnorm-g-prod");
    eval_t2.absorb_bytes(&proof.g_commitment.root);
    if !verify_mle_eval_ef(&proof.g_commitment, g_at_prod_s, &s_gp, &proof.g_prod_eval_proof, &mut eval_t2) {
        eprintln!("RmsNormEF: g prod MLE eval failed");
        return false;
    }

    // Triple sumcheck: Σ eq · γ · x = t_g
    let (eq_at_gt, gamma_at_gt, x_at_gt) = (
        proof.g_triple_finals.0.to_ef(),
        proof.g_triple_finals.1.to_ef(),
        proof.g_triple_finals.2.to_ef(),
    );
    let (gt_ok, s_gt) = sumcheck::verify_triple_ef_with_challenges(
        t_g, &proof.g_triple_proof, eq_at_gt, gamma_at_gt, x_at_gt, transcript);
    if !gt_ok {
        eprintln!("RmsNormEF: g triple sumcheck failed");
        return false;
    }
    let eq_gt_expected = compute_eq_at_point_ef(&gc_point, &s_gt);
    if eq_gt_expected != eq_at_gt {
        eprintln!("RmsNormEF: eq mismatch in g triple");
        return false;
    }
    let x_at_triple_s = proof.x_at_triple_s.to_ef();
    if x_at_triple_s != x_at_gt { return false; }
    let mut eval_t3 = Transcript::new(b"rmsnorm-x-triple");
    eval_t3.absorb_bytes(&proof.x_commitment.root);
    if !verify_mle_eval_ef(&proof.x_commitment, x_at_triple_s, &s_gt, &proof.x_triple_eval_proof, &mut eval_t3) {
        eprintln!("RmsNormEF: x triple MLE eval failed");
        return false;
    }
    let gamma_at_triple_s = proof.gamma_at_triple_s.to_ef();
    if gamma_at_triple_s != gamma_at_gt { return false; }
    let mut eval_t4 = Transcript::new(b"rmsnorm-gamma-triple");
    eval_t4.absorb_bytes(&proof.gamma_commitment.root);
    if !verify_mle_eval_ef(&proof.gamma_commitment, gamma_at_triple_s, &s_gt, &proof.gamma_triple_eval_proof, &mut eval_t4) {
        eprintln!("RmsNormEF: gamma triple MLE eval failed");
        return false;
    }

    // --- Squared checks ---
    let sq_point = transcript.squeeze_ef_many(log_n);
    let s_h_val = proof.s_h.to_ef();
    let s_g_val = proof.s_g.to_ef();

    // h-squared: S_h = Σ eq · h · h
    let (eq_at_hs, h_at_hs_a, h_at_hs_b) = (
        proof.h_sq_finals.0.to_ef(),
        proof.h_sq_finals.1.to_ef(),
        proof.h_sq_finals.2.to_ef(),
    );
    let (hs_ok, s_hs) = sumcheck::verify_triple_ef_with_challenges(
        s_h_val, &proof.h_sq_proof, eq_at_hs, h_at_hs_a, h_at_hs_b, transcript);
    if !hs_ok {
        eprintln!("RmsNormEF: h-squared sumcheck failed");
        return false;
    }
    let eq_hs_expected = compute_eq_at_point_ef(&sq_point, &s_hs);
    if eq_hs_expected != eq_at_hs {
        eprintln!("RmsNormEF: eq mismatch in h-squared");
        return false;
    }
    if h_at_hs_a != h_at_hs_b {
        eprintln!("RmsNormEF: h-squared finals mismatch");
        return false;
    }
    let h_at_sq_s = proof.h_at_sq_s.to_ef();
    if h_at_sq_s != h_at_hs_a { return false; }
    let mut eval_t5 = Transcript::new(b"rmsnorm-h-sq");
    eval_t5.absorb_bytes(&proof.h_commitment.root);
    if !verify_mle_eval_ef(&proof.h_commitment, h_at_sq_s, &s_hs, &proof.h_sq_eval_proof, &mut eval_t5) {
        eprintln!("RmsNormEF: h sq MLE eval failed");
        return false;
    }

    // Verify h commitment matches y
    let mut y_pad = y.to_vec();
    y_pad.resize(n_pad, F::zero());
    let h_commitment_expected = commit_weights_fast(&y_pad);
    if h_commitment_expected.root != proof.h_commitment.root {
        eprintln!("RmsNormEF: h commitment mismatch (y)");
        return false;
    }

    // g-squared: S_g = Σ eq · g · g
    let (eq_at_gs, g_at_gs_a, g_at_gs_b) = (
        proof.g_sq_finals.0.to_ef(),
        proof.g_sq_finals.1.to_ef(),
        proof.g_sq_finals.2.to_ef(),
    );
    let (gs_ok, s_gs) = sumcheck::verify_triple_ef_with_challenges(
        s_g_val, &proof.g_sq_proof, eq_at_gs, g_at_gs_a, g_at_gs_b, transcript);
    if !gs_ok {
        eprintln!("RmsNormEF: g-squared sumcheck failed");
        return false;
    }
    let eq_gs_expected = compute_eq_at_point_ef(&sq_point, &s_gs);
    if eq_gs_expected != eq_at_gs {
        eprintln!("RmsNormEF: eq mismatch in g-squared");
        return false;
    }
    if g_at_gs_a != g_at_gs_b {
        eprintln!("RmsNormEF: g-squared finals mismatch");
        return false;
    }
    let g_at_gsq_s = proof.g_at_gsq_s.to_ef();
    if g_at_gsq_s != g_at_gs_a { return false; }
    let mut eval_t6 = Transcript::new(b"rmsnorm-g-sq");
    eval_t6.absorb_bytes(&proof.g_commitment.root);
    if !verify_mle_eval_ef(&proof.g_commitment, g_at_gsq_s, &s_gs, &proof.g_sq_eval_proof, &mut eval_t6) {
        eprintln!("RmsNormEF: g sq MLE eval failed");
        return false;
    }

    // --- Final check: sum_sq · S_h == d · S_g (in EF) ---
    if sum_sq_ef * s_h_val != d_field_ef * s_g_val {
        eprintln!("RmsNormEF: sum_sq·S_h != d·S_g (squared output check failed)");
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::Field;
    use crate::field::common::mod_sqrt_m31;

    fn is_qr_m31(a: F) -> bool {
        if a == F::zero() { return true; }
        let mut r = a;
        for _ in 0..30 { r = r * r; }
        r * a.inverse() == F::one()
    }

    /// Compute RMSNorm output in M31 when d/sum_sq is a QR.
    fn rmsnorm_output(x: &[F], gamma: &[F], d: usize) -> Option<Vec<F>> {
        let sum_sq: F = x.iter().map(|&v| v * v).sum();
        if sum_sq == F::zero() { return None; }
        let target = F::from_canonical_u32(d as u32) * sum_sq.inverse();
        if !is_qr_m31(target) { return None; }
        let r = mod_sqrt_m31(target);
        Some(x.iter().zip(gamma.iter()).map(|(&xi, &gi)| gi * xi * r).collect())
    }

    #[test]
    fn test_rmsnorm_simple() {
        let d = 4;
        // Try inputs until we find one where d/sum_sq is QR
        for offset in 0u32..200 {
            let x: Vec<F> = (0..d).map(|i| F::from_canonical_u32(i as u32 + 1 + offset)).collect();
            let gamma = vec![F::one(); d];
            if let Some(y) = rmsnorm_output(&x, &gamma, d) {
                let mut pt = Transcript::new(b"rmsnorm-test");
                let proof = prove_rmsnorm(&x, &gamma, &y, &mut pt);
                let mut vt = Transcript::new(b"rmsnorm-test");
                assert!(verify_rmsnorm(&proof, &y, d, &mut vt), "RMSNorm verification failed");
                return;
            }
        }
        panic!("Could not find valid input");
    }

    #[test]
    fn test_rmsnorm_with_gamma() {
        let d = 4;
        let gamma: Vec<F> = vec![2, 3, 4, 5].iter().map(|&v| F::from_canonical_u32(v)).collect();
        for offset in 0u32..200 {
            let x: Vec<F> = (0..d).map(|i| F::from_canonical_u32(i as u32 + 1 + offset)).collect();
            if let Some(y) = rmsnorm_output(&x, &gamma, d) {
                let mut pt = Transcript::new(b"rmsnorm-gamma-test");
                let proof = prove_rmsnorm(&x, &gamma, &y, &mut pt);
                let mut vt = Transcript::new(b"rmsnorm-gamma-test");
                assert!(verify_rmsnorm(&proof, &y, d, &mut vt));
                return;
            }
        }
        panic!("Could not find valid input");
    }

    #[test]
    fn test_rmsnorm_wrong_output() {
        let d = 4;
        let gamma = vec![F::one(); d];
        for offset in 0u32..200 {
            let x: Vec<F> = (0..d).map(|i| F::from_canonical_u32(i as u32 + 1 + offset)).collect();
            if let Some(mut y) = rmsnorm_output(&x, &gamma, d) {
                y[0] = y[0] + F::one(); // tamper
                let mut pt = Transcript::new(b"rmsnorm-wrong");
                let proof = prove_rmsnorm(&x, &gamma, &y, &mut pt);
                let mut vt = Transcript::new(b"rmsnorm-wrong");
                assert!(!verify_rmsnorm(&proof, &y, d, &mut vt), "Should reject tampered output");
                return;
            }
        }
        panic!("Could not find valid input");
    }

    #[test]
    fn test_rmsnorm_larger() {
        let d = 16;
        let gamma: Vec<F> = (0..d).map(|i| F::from_canonical_u32(i as u32 + 1)).collect();
        for offset in 0u32..500 {
            let x: Vec<F> = (0..d).map(|i| F::from_canonical_u32(i as u32 * 3 + 1 + offset)).collect();
            if let Some(y) = rmsnorm_output(&x, &gamma, d) {
                let mut pt = Transcript::new(b"rmsnorm-large");
                let proof = prove_rmsnorm(&x, &gamma, &y, &mut pt);
                let mut vt = Transcript::new(b"rmsnorm-large");
                assert!(verify_rmsnorm(&proof, &y, d, &mut vt));
                return;
            }
        }
        panic!("Could not find valid input");
    }

    #[test]
    fn test_rmsnorm_ef_basic() {
        let d = 4;
        let gamma = vec![F::one(); d];
        for offset in 0u32..200 {
            let x: Vec<F> = (0..d).map(|i| F::from_canonical_u32(i as u32 + 1 + offset)).collect();
            if let Some(y) = rmsnorm_output(&x, &gamma, d) {
                let mut pt = Transcript::new(b"rmsnorm-ef-test");
                let proof = prove_rmsnorm_ef(&x, &gamma, &y, &mut pt);
                let mut vt = Transcript::new(b"rmsnorm-ef-test");
                assert!(verify_rmsnorm_ef(&proof, &y, d, &mut vt), "RMSNorm EF verification should pass");
                return;
            }
        }
        panic!("Could not find valid input");
    }
}
