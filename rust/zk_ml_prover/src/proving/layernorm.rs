//! Algebraic LayerNorm proof over M31.
//!
//! LayerNorm(x) = γ * (x - μ) / sqrt(var(x) + ε) + β
//!
//! where μ = mean(x), var(x) = mean((x - μ)²).
//!
//! Algebraic trick (no sqrt in-circuit):
//! - Prover claims `μ` (mean) and `r = 1/sqrt(var+ε)` as witnesses
//! - Define x_centered[i] = x[i] - μ
//! - Check 1: Σ x_centered[i] = 0 (mean subtraction correct)
//!   via product sumcheck on (eq, x_centered) with claim = 0
//! - Check 2: r² × Σ x_centered[i]² = d (variance constraint, like RMSNorm)
//!   via product sumcheck on (x_centered, x_centered)
//! - Check 3: y[i] = γ[i] * x_centered[i] * r + β[i]
//!   Rearrange: (y[i] - β[i]) = γ[i] * x_centered[i] * r
//!   Triple sumcheck on (eq, γ, x_centered) with claim = Σ eq · (y - β) / r

use p3_field::{AbstractField, Field, PrimeField32};
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::field::common::{log2_ceil, compute_eq_at_point, compute_eq_at_point_ef};
use crate::field::m31_ops::*;
use crate::proving::sumcheck::{self, SumcheckProof, SumcheckProofEF, EFElement, EF, Transcript};
use crate::proving::weight_commitment::{
    commit_weights_fast, prove_mle_eval_no_merkle, verify_mle_eval, MleEvalProof,
    MleEvalProofEF, prove_mle_eval_no_merkle_ef_base, verify_mle_eval_ef,
    WeightCommitment,
};

type F = Mersenne31;

/// Proof that y = LayerNorm(x, γ, β).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerNormProof {
    /// Prover's claimed mean μ in M31.
    pub mu: u32,
    /// Prover's claimed 1/sqrt(var+ε) in M31.
    pub r_inv_sqrt: u32,

    // --- Mean check: Σ (x[i] - μ) = 0 via product sumcheck on (eq, x_centered) ---
    pub mean_check_proof: SumcheckProof,
    pub mean_check_finals: (u32, u32), // (eq_at_s, x_centered_at_s)

    // --- Variance check: Σ x_centered[i]² via product sumcheck ---
    pub var_proof: SumcheckProof,
    pub var_finals: (u32, u32), // (xc_at_s_a, xc_at_s_b)
    pub sum_sq_value: u32,

    // --- Output check: triple sumcheck on (eq, γ, x_centered) ---
    pub output_proof: SumcheckProof,
    pub output_finals: (u32, u32, u32), // (eq_at_s, gamma_at_s, xc_at_s)

    // --- Commitments + MLE eval proofs ---
    pub x_commitment: WeightCommitment,
    pub gamma_commitment: WeightCommitment,
    pub beta_commitment: WeightCommitment,

    // MLE eval proofs for x_centered at mean_check challenge point
    // (We commit to x, then verifier can derive x_centered MLE from x MLE and μ)
    // Actually, we commit to x_centered directly since it's used in all three checks.
    pub xc_commitment: WeightCommitment,

    // x_centered eval at mean_check point
    pub xc_mean_eval_proof: MleEvalProof,
    pub xc_at_mean_s: u32,

    // x_centered eval at var_check point
    pub xc_var_eval_proof: MleEvalProof,
    pub xc_at_var_s: u32,

    // x_centered eval at output_check point
    pub xc_output_eval_proof: MleEvalProof,
    pub xc_at_output_s: u32,

    // gamma eval at output_check point
    pub gamma_output_eval_proof: MleEvalProof,
    pub gamma_at_output_s: u32,
}

/// Prove LayerNorm: y = γ * (x - μ) * r + β where r² * Σ(x-μ)² = d.
pub fn prove_layernorm(
    x: &[F],
    gamma: &[F],
    beta: &[F],
    y: &[F],
    mu: F,
    r_inv_sqrt: F,
    transcript: &mut Transcript,
) -> LayerNormProof {
    let d = x.len();
    let log_n = log2_ceil(d);
    let n_pad = 1 << log_n;

    // Pad all vectors to power of 2
    let mut x_pad = x.to_vec();
    x_pad.resize(n_pad, F::zero());
    let mut gamma_pad = gamma.to_vec();
    gamma_pad.resize(n_pad, F::zero());
    let mut beta_pad = beta.to_vec();
    beta_pad.resize(n_pad, F::zero());
    let mut y_pad = y.to_vec();
    y_pad.resize(n_pad, F::zero());

    // Compute x_centered = x - mu (padded entries stay 0 since x_pad[i>=d] = 0
    // but we must subtract mu from ALL entries to keep the polynomial consistent...
    // Actually, for padded entries x=0, x_centered = 0 - mu = -mu which would
    // break sum(x_centered) = 0. So we only subtract mu from the first d entries.
    let mut xc_pad = vec![F::zero(); n_pad];
    for i in 0..d {
        xc_pad[i] = x_pad[i] - mu;
    }
    // Padded entries remain 0 (they don't contribute to sum since they're 0 in
    // the original, and we define the mean over d elements not n_pad).

    // Commit to x_centered, gamma, beta, x
    let x_commitment = commit_weights_fast(&x_pad);
    let xc_commitment = commit_weights_fast(&xc_pad);
    let gamma_commitment = commit_weights_fast(&gamma_pad);
    let beta_commitment = commit_weights_fast(&beta_pad);

    // Absorb commitments, mu, and r into transcript
    transcript.absorb_bytes(&x_commitment.root);
    transcript.absorb_bytes(&xc_commitment.root);
    transcript.absorb_bytes(&gamma_commitment.root);
    transcript.absorb_bytes(&beta_commitment.root);
    transcript.absorb(mu.as_canonical_u32());
    transcript.absorb(r_inv_sqrt.as_canonical_u32());

    // --- Step 1: Mean check ---
    // Prove Σ x_centered[i] = 0 via product sumcheck: Σ eq(point, b) · xc[b] = 0
    // where point is a random evaluation point.
    let mean_point = transcript.squeeze_many(log_n);
    let _eq_mean = eq_evals(&mean_point);

    // The claim is: x̃c(mean_point) * (sum of eq) ... no.
    // Actually: product sumcheck proves Σ_b eq(point, b) * xc[b] = x̃c(point).
    // We need to show Σ xc[i] = 0.
    // Σ_b eq(point, b) * xc[b] = x̃c(point). This doesn't directly give us Σ xc[i] = 0.
    //
    // Better approach: just compute sum_xc = Σ xc[i] and prove it equals 0.
    // Use the "sum via MLE" trick: Σ xc[i] = Σ_b xc[b] = x̃c evaluated at all-halves point?
    // No. Σ_b xc[b] is just the sum of the table values.
    //
    // Simplest: product sumcheck with f=ones, g=xc. Σ 1·xc[i] = sum_xc.
    // Then verifier checks sum_xc = 0.
    let ones = vec![F::one(); n_pad];
    let _sum_xc: F = xc_pad.iter().copied().sum();

    let (mean_check_proof, ones_at_s, xc_at_s_mean) =
        sumcheck::prove_product_best(&ones, &xc_pad, log_n, transcript);

    // MLE eval proof for xc at mean_check challenge point
    let s_mean: Vec<F> = mean_check_proof
        .challenges
        .iter()
        .map(|&v| F::from_canonical_u32(v))
        .collect();
    let xc_at_mean_s = mle_evaluate(&xc_pad, &s_mean);
    let mut eval_t1 = Transcript::new(b"layernorm-xc-mean-eval");
    eval_t1.absorb_bytes(&xc_commitment.root);
    let (_, xc_mean_eval_proof) = prove_mle_eval_no_merkle(&xc_pad, &s_mean, &mut eval_t1);

    // --- Step 2: Variance check ---
    // Prove Σ x_centered[i]² via product sumcheck (xc · xc)
    let sum_sq: F = xc_pad.iter().map(|&v| v * v).sum();

    let (var_proof, xc_at_s_var_a, xc_at_s_var_b) =
        sumcheck::prove_product_best(&xc_pad, &xc_pad, log_n, transcript);

    // MLE eval proof for xc at var challenge point
    let s_var: Vec<F> = var_proof
        .challenges
        .iter()
        .map(|&v| F::from_canonical_u32(v))
        .collect();
    let xc_at_var_s = mle_evaluate(&xc_pad, &s_var);
    let mut eval_t2 = Transcript::new(b"layernorm-xc-var-eval");
    eval_t2.absorb_bytes(&xc_commitment.root);
    let (_, xc_var_eval_proof) = prove_mle_eval_no_merkle(&xc_pad, &s_var, &mut eval_t2);

    // --- Step 3: Output check ---
    // y[i] = gamma[i] * xc[i] * r + beta[i]
    // => (y[i] - beta[i]) = gamma[i] * xc[i] * r
    // => Σ eq(pt, b) * gamma[b] * xc[b] = Σ eq(pt, b) * (y[b] - beta[b]) / r
    let eval_point = transcript.squeeze_many(log_n);
    let eq_out = eq_evals(&eval_point);

    let (output_proof, eq_at_s, gamma_at_s_out, xc_at_s_out) =
        sumcheck::prove_triple_best(&eq_out, &gamma_pad, &xc_pad, log_n, transcript);

    // MLE eval proofs for xc and gamma at output challenge point
    let s_out: Vec<F> = output_proof
        .challenges
        .iter()
        .map(|&v| F::from_canonical_u32(v))
        .collect();

    let xc_at_output_s = mle_evaluate(&xc_pad, &s_out);
    let mut eval_t3 = Transcript::new(b"layernorm-xc-output-eval");
    eval_t3.absorb_bytes(&xc_commitment.root);
    let (_, xc_output_eval_proof) = prove_mle_eval_no_merkle(&xc_pad, &s_out, &mut eval_t3);

    let gamma_at_output_s = mle_evaluate(&gamma_pad, &s_out);
    let mut eval_t4 = Transcript::new(b"layernorm-gamma-output-eval");
    eval_t4.absorb_bytes(&gamma_commitment.root);
    let (_, gamma_output_eval_proof) = prove_mle_eval_no_merkle(&gamma_pad, &s_out, &mut eval_t4);

    LayerNormProof {
        mu: mu.as_canonical_u32(),
        r_inv_sqrt: r_inv_sqrt.as_canonical_u32(),
        mean_check_proof,
        mean_check_finals: (
            ones_at_s.as_canonical_u32(),
            xc_at_s_mean.as_canonical_u32(),
        ),
        var_proof,
        var_finals: (
            xc_at_s_var_a.as_canonical_u32(),
            xc_at_s_var_b.as_canonical_u32(),
        ),
        sum_sq_value: sum_sq.as_canonical_u32(),
        output_proof,
        output_finals: (
            eq_at_s.as_canonical_u32(),
            gamma_at_s_out.as_canonical_u32(),
            xc_at_s_out.as_canonical_u32(),
        ),
        x_commitment,
        xc_commitment,
        gamma_commitment,
        beta_commitment,
        xc_mean_eval_proof,
        xc_at_mean_s: xc_at_mean_s.as_canonical_u32(),
        xc_var_eval_proof,
        xc_at_var_s: xc_at_var_s.as_canonical_u32(),
        xc_output_eval_proof,
        xc_at_output_s: xc_at_output_s.as_canonical_u32(),
        gamma_output_eval_proof,
        gamma_at_output_s: gamma_at_output_s.as_canonical_u32(),
    }
}

/// Verify a LayerNorm proof.
///
/// Checks:
/// 1. Σ (x[i] - μ) = 0 (mean check via product sumcheck with ones)
/// 2. r² × Σ (x[i] - μ)² = d (variance constraint)
/// 3. y = γ ⊙ (x - μ) × r + β (output check via triple sumcheck)
/// 4. MLE eval proofs against commitments
pub fn verify_layernorm(
    proof: &LayerNormProof,
    y: &[F],
    beta: &[F],
    d: usize,
    transcript: &mut Transcript,
) -> bool {
    let log_n = log2_ceil(d);
    let n_pad = 1 << log_n;

    let _mu = F::from_canonical_u32(proof.mu);
    let r = F::from_canonical_u32(proof.r_inv_sqrt);
    let sum_sq = F::from_canonical_u32(proof.sum_sq_value);
    let d_field = F::from_canonical_u32(d as u32);

    // --- Algebraic check: r² × Σ(x-μ)² = d ---
    if r * r * sum_sq != d_field {
        eprintln!("LayerNorm: r² * sum_sq != d");
        return false;
    }

    // Absorb commitments, mu, r into transcript (must match prover)
    transcript.absorb_bytes(&proof.x_commitment.root);
    transcript.absorb_bytes(&proof.xc_commitment.root);
    transcript.absorb_bytes(&proof.gamma_commitment.root);
    transcript.absorb_bytes(&proof.beta_commitment.root);
    transcript.absorb(proof.mu);
    transcript.absorb(proof.r_inv_sqrt);

    // --- Check 1: Mean check ---
    // Product sumcheck: Σ 1 · xc[i] = 0 (claimed sum must be 0)
    // The "ones" polynomial: 1̃(s) = 1 for all s (MLE of all-ones is 1 everywhere).
    // Actually MLE of [1,1,...,1] at any point is 1.
    let _mean_point = transcript.squeeze_many(log_n);

    let ones_at_s = F::from_canonical_u32(proof.mean_check_finals.0);
    let xc_at_s_mean = F::from_canonical_u32(proof.mean_check_finals.1);

    let (mean_ok, s_mean) = sumcheck::verify_product_with_challenges(
        F::zero(), // claimed sum = 0
        &proof.mean_check_proof,
        ones_at_s,
        xc_at_s_mean,
        transcript,
    );
    if !mean_ok {
        eprintln!("LayerNorm: mean check sumcheck failed");
        return false;
    }

    // Verify ones_at_s = 1 (MLE of all-ones vector is 1 everywhere)
    if ones_at_s != F::one() {
        eprintln!("LayerNorm: ones_at_s != 1");
        return false;
    }

    // Verify xc MLE eval proof at mean_check point
    let xc_at_mean_s = F::from_canonical_u32(proof.xc_at_mean_s);
    if xc_at_mean_s != xc_at_s_mean {
        eprintln!("LayerNorm: xc_at_mean_s mismatch");
        return false;
    }
    let mut eval_t1 = Transcript::new(b"layernorm-xc-mean-eval");
    eval_t1.absorb_bytes(&proof.xc_commitment.root);
    if !verify_mle_eval(
        &proof.xc_commitment,
        xc_at_mean_s,
        &s_mean,
        &proof.xc_mean_eval_proof,
        &mut eval_t1,
    ) {
        eprintln!("LayerNorm: xc mean MLE eval proof failed");
        return false;
    }

    // --- Check 2: Variance check ---
    let xc_var_a = F::from_canonical_u32(proof.var_finals.0);
    let xc_var_b = F::from_canonical_u32(proof.var_finals.1);

    let (var_ok, s_var) = sumcheck::verify_product_with_challenges(
        sum_sq,
        &proof.var_proof,
        xc_var_a,
        xc_var_b,
        transcript,
    );
    if !var_ok {
        eprintln!("LayerNorm: variance sumcheck failed");
        return false;
    }

    // Both finals must be equal (xc · xc)
    if xc_var_a != xc_var_b {
        eprintln!("LayerNorm: var finals mismatch (xc·xc but a != b)");
        return false;
    }

    // Verify xc MLE eval proof at var point
    let xc_at_var_s = F::from_canonical_u32(proof.xc_at_var_s);
    if xc_at_var_s != xc_var_a {
        eprintln!("LayerNorm: xc_at_var_s mismatch");
        return false;
    }
    let mut eval_t2 = Transcript::new(b"layernorm-xc-var-eval");
    eval_t2.absorb_bytes(&proof.xc_commitment.root);
    if !verify_mle_eval(
        &proof.xc_commitment,
        xc_at_var_s,
        &s_var,
        &proof.xc_var_eval_proof,
        &mut eval_t2,
    ) {
        eprintln!("LayerNorm: xc var MLE eval proof failed");
        return false;
    }

    // --- Check 3: Output check ---
    let eval_point = transcript.squeeze_many(log_n);

    // Compute (ỹ - β̃)(eval_point)
    let mut y_pad = y.to_vec();
    y_pad.resize(n_pad, F::zero());
    let mut beta_pad = beta.to_vec();
    beta_pad.resize(n_pad, F::zero());
    let y_at_point = mle_evaluate(&y_pad, &eval_point);
    let beta_at_point = mle_evaluate(&beta_pad, &eval_point);

    // claimed_xg = (ỹ(pt) - β̃(pt)) / r
    let claimed_xg = (y_at_point - beta_at_point) * r.inverse();

    let (ef, gf, hf) = (
        F::from_canonical_u32(proof.output_finals.0),
        F::from_canonical_u32(proof.output_finals.1),
        F::from_canonical_u32(proof.output_finals.2),
    );

    let (out_ok, s_out) = sumcheck::verify_triple_with_challenges(
        claimed_xg,
        &proof.output_proof,
        ef, gf, hf,
        transcript,
    );
    if !out_ok {
        eprintln!("LayerNorm: output sumcheck failed");
        return false;
    }

    // Verify eq(eval_point, s_out) independently using derived challenges
    let eq_expected = compute_eq_at_point(&eval_point, &s_out);
    if eq_expected != ef {
        eprintln!("LayerNorm: eq_at_s mismatch in output sumcheck");
        return false;
    }

    // Verify xc MLE eval at output point
    let xc_at_output_s = F::from_canonical_u32(proof.xc_at_output_s);
    if xc_at_output_s != hf {
        eprintln!("LayerNorm: xc_at_output_s mismatch");
        return false;
    }
    let mut eval_t3 = Transcript::new(b"layernorm-xc-output-eval");
    eval_t3.absorb_bytes(&proof.xc_commitment.root);
    if !verify_mle_eval(
        &proof.xc_commitment,
        xc_at_output_s,
        &s_out,
        &proof.xc_output_eval_proof,
        &mut eval_t3,
    ) {
        eprintln!("LayerNorm: xc output MLE eval proof failed");
        return false;
    }

    // Verify gamma MLE eval at output point
    let gamma_at_output_s = F::from_canonical_u32(proof.gamma_at_output_s);
    if gamma_at_output_s != gf {
        eprintln!("LayerNorm: gamma_at_output_s mismatch");
        return false;
    }
    let mut eval_t4 = Transcript::new(b"layernorm-gamma-output-eval");
    eval_t4.absorb_bytes(&proof.gamma_commitment.root);
    if !verify_mle_eval(
        &proof.gamma_commitment,
        gamma_at_output_s,
        &s_out,
        &proof.gamma_output_eval_proof,
        &mut eval_t4,
    ) {
        eprintln!("LayerNorm: gamma output MLE eval proof failed");
        return false;
    }

    true
}

/// Squared LayerNorm proof — works even when d/sum_sq is NOT a quadratic residue.
///
/// Instead of needing r = sqrt(d/sum_sq) (which may not exist in M31),
/// we prove the squared constraint: sum_sq · Σ h² = d · Σ g² where:
///   h[i] = y[i] - β[i]
///   g[i] = γ[i] · xc[i]
///
/// This holds because h[i] = γ[i]·xc[i]·r + β[i] - β[i] = g[i]·r,
/// so h² = g²·r² = g²·d/sum_sq, giving sum_sq·h² = d·g².
///
/// Concretely (all at same random point ρ):
///   S_h = Σ eq(ρ,i) · h(i)²    (triple sumcheck on eq, h, h)
///   S_g = Σ eq(ρ,i) · g(i)²    (triple sumcheck on eq, g, g)
///   g-consistency: Σ eq(ρ,i) · g(i) == Σ eq(ρ,i) · γ(i) · xc(i)
///   Verifier checks: sum_sq · S_h == d · S_g
///
/// The prover provides pre-computed y from float LayerNorm (Python).
/// g[i] = γ[i] · xc[i] is computed in M31 by the prover.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerNormSqrProof {
    pub mu: u32,

    // Mean check
    pub mean_check_proof: SumcheckProof,
    pub mean_check_finals: (u32, u32),

    // Variance check
    pub var_proof: SumcheckProof,
    pub var_finals: (u32, u32),
    pub sum_sq_value: u32,

    // g-consistency: Σ eq · g == Σ eq · γ · xc
    // Product sumcheck for Σ eq · g, triple sumcheck for Σ eq · γ · xc
    pub g_prod_proof: SumcheckProof,
    pub g_prod_finals: (u32, u32), // (eq_at_s, g_at_s)
    pub g_triple_proof: SumcheckProof,
    pub g_triple_finals: (u32, u32, u32), // (eq_at_s, gamma_at_s, xc_at_s)
    pub t_g_claimed: u32, // the shared claimed sum

    // h-squared: S_h = Σ eq · h · h
    pub h_sq_proof: SumcheckProof,
    pub h_sq_finals: (u32, u32, u32), // (eq_at_s, h_at_s_a, h_at_s_b)
    pub s_h: u32,

    // g-squared: S_g = Σ eq · g · g
    pub g_sq_proof: SumcheckProof,
    pub g_sq_finals: (u32, u32, u32), // (eq_at_s, g_at_s_a, g_at_s_b)
    pub s_g: u32,

    // Commitments
    pub xc_commitment: WeightCommitment,
    pub gamma_commitment: WeightCommitment,
    pub g_commitment: WeightCommitment,
    pub h_commitment: WeightCommitment,

    // MLE eval proofs
    pub xc_mean_eval_proof: MleEvalProof,
    pub xc_at_mean_s: u32,
    pub xc_var_eval_proof: MleEvalProof,
    pub xc_at_var_s: u32,
    // g-consistency evals
    pub g_prod_eval_proof: MleEvalProof,
    pub g_at_prod_s: u32,
    pub xc_triple_eval_proof: MleEvalProof,
    pub xc_at_triple_s: u32,
    pub gamma_triple_eval_proof: MleEvalProof,
    pub gamma_at_triple_s: u32,
    // h-squared evals
    pub h_sq_eval_proof: MleEvalProof,
    pub h_at_sq_s: u32,
    // g-squared evals
    pub g_sq_eval_proof: MleEvalProof,
    pub g_at_gsq_s: u32,
}

/// Prove LayerNorm using squared constraint (no sqrt needed).
pub fn prove_layernorm_sqr(
    x: &[F],
    gamma: &[F],
    beta: &[F],
    y: &[F],
    mu: F,
    transcript: &mut Transcript,
) -> LayerNormSqrProof {
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
    let mut beta_pad = beta.to_vec();
    beta_pad.resize(n_pad, F::zero());

    // x_centered (only subtract mu from real entries)
    let mut xc_pad = vec![F::zero(); n_pad];
    for i in 0..d {
        xc_pad[i] = x_pad[i] - mu;
    }

    // g = gamma * xc (in M31, exact)
    let mut g_pad = vec![F::zero(); n_pad];
    for i in 0..d {
        g_pad[i] = gamma_pad[i] * xc_pad[i];
    }

    // h = y - beta
    let mut h_pad = vec![F::zero(); n_pad];
    for i in 0..d {
        h_pad[i] = y_pad[i] - beta_pad[i];
    }

    let sum_sq: F = xc_pad.iter().map(|&v| v * v).sum();

    // Commitments
    let xc_commitment = commit_weights_fast(&xc_pad);
    let gamma_commitment = commit_weights_fast(&gamma_pad);
    let g_commitment = commit_weights_fast(&g_pad);
    let h_commitment = commit_weights_fast(&h_pad);

    // Absorb
    transcript.absorb_bytes(&xc_commitment.root);
    transcript.absorb_bytes(&gamma_commitment.root);
    transcript.absorb_bytes(&g_commitment.root);
    transcript.absorb_bytes(&h_commitment.root);
    transcript.absorb(mu.as_canonical_u32());
    transcript.absorb(sum_sq.as_canonical_u32());

    // --- Step 1: Mean check (Σ xc[i] = 0) ---
    let _mean_point = transcript.squeeze_many(log_n);
    let ones = vec![F::one(); n_pad];
    let (mean_check_proof, ones_at_s, xc_at_s_mean) =
        sumcheck::prove_product_best(&ones, &xc_pad, log_n, transcript);

    let s_mean: Vec<F> = mean_check_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let xc_at_mean_s = mle_evaluate(&xc_pad, &s_mean);
    let mut eval_t1 = Transcript::new(b"lnsqr-xc-mean");
    eval_t1.absorb_bytes(&xc_commitment.root);
    let (_, xc_mean_eval_proof) = prove_mle_eval_no_merkle(&xc_pad, &s_mean, &mut eval_t1);

    // --- Step 2: Variance check (Σ xc[i]² = sum_sq) ---
    let (var_proof, xc_var_a, xc_var_b) =
        sumcheck::prove_product_best(&xc_pad, &xc_pad, log_n, transcript);

    let s_var: Vec<F> = var_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let xc_at_var_s = mle_evaluate(&xc_pad, &s_var);
    let mut eval_t2 = Transcript::new(b"lnsqr-xc-var");
    eval_t2.absorb_bytes(&xc_commitment.root);
    let (_, xc_var_eval_proof) = prove_mle_eval_no_merkle(&xc_pad, &s_var, &mut eval_t2);

    // --- Step 3: g-consistency check ---
    // Prove Σ eq(ρ,i) · g[i] == Σ eq(ρ,i) · γ[i] · xc[i] at same point
    let gc_point = transcript.squeeze_many(log_n);
    let eq_gc = eq_evals(&gc_point);

    let t_g: F = eq_gc.iter().zip(g_pad.iter()).map(|(&e, &g)| e * g).sum();

    // Product sumcheck: Σ eq · g = t_g
    let (g_prod_proof, eq_at_gp, g_at_gp) =
        sumcheck::prove_product_best(&eq_gc, &g_pad, log_n, transcript);

    let s_gp: Vec<F> = g_prod_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let g_at_prod_s = mle_evaluate(&g_pad, &s_gp);
    let mut eval_t3 = Transcript::new(b"lnsqr-g-prod");
    eval_t3.absorb_bytes(&g_commitment.root);
    let (_, g_prod_eval_proof) = prove_mle_eval_no_merkle(&g_pad, &s_gp, &mut eval_t3);

    // Triple sumcheck: Σ eq · γ · xc = t_g (same claimed value)
    let (g_triple_proof, eq_at_gt, gamma_at_gt, xc_at_gt) =
        sumcheck::prove_triple_best(&eq_gc, &gamma_pad, &xc_pad, log_n, transcript);

    let s_gt: Vec<F> = g_triple_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let xc_at_triple_s = mle_evaluate(&xc_pad, &s_gt);
    let mut eval_t4 = Transcript::new(b"lnsqr-xc-triple");
    eval_t4.absorb_bytes(&xc_commitment.root);
    let (_, xc_triple_eval_proof) = prove_mle_eval_no_merkle(&xc_pad, &s_gt, &mut eval_t4);
    let gamma_at_triple_s = mle_evaluate(&gamma_pad, &s_gt);
    let mut eval_t5 = Transcript::new(b"lnsqr-gamma-triple");
    eval_t5.absorb_bytes(&gamma_commitment.root);
    let (_, gamma_triple_eval_proof) = prove_mle_eval_no_merkle(&gamma_pad, &s_gt, &mut eval_t5);

    // --- Step 4: Squared checks ---
    let sq_point = transcript.squeeze_many(log_n);
    let eq_sq = eq_evals(&sq_point);

    // h-squared: S_h = Σ eq · h · h
    let s_h_val: F = eq_sq.iter().zip(h_pad.iter()).map(|(&e, &h)| e * h * h).sum();
    let (h_sq_proof, eq_at_hs, h_at_hs_a, h_at_hs_b) =
        sumcheck::prove_triple_best(&eq_sq, &h_pad, &h_pad, log_n, transcript);

    let s_hs: Vec<F> = h_sq_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let h_at_sq_s = mle_evaluate(&h_pad, &s_hs);
    let mut eval_t6 = Transcript::new(b"lnsqr-h-sq");
    eval_t6.absorb_bytes(&h_commitment.root);
    let (_, h_sq_eval_proof) = prove_mle_eval_no_merkle(&h_pad, &s_hs, &mut eval_t6);

    // g-squared: S_g = Σ eq · g · g
    let s_g_val: F = eq_sq.iter().zip(g_pad.iter()).map(|(&e, &g)| e * g * g).sum();
    let (g_sq_proof, eq_at_gs, g_at_gs_a, g_at_gs_b) =
        sumcheck::prove_triple_best(&eq_sq, &g_pad, &g_pad, log_n, transcript);

    let s_gs: Vec<F> = g_sq_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let g_at_gsq_s = mle_evaluate(&g_pad, &s_gs);
    let mut eval_t7 = Transcript::new(b"lnsqr-g-sq");
    eval_t7.absorb_bytes(&g_commitment.root);
    let (_, g_sq_eval_proof) = prove_mle_eval_no_merkle(&g_pad, &s_gs, &mut eval_t7);

    LayerNormSqrProof {
        mu: mu.as_canonical_u32(),
        mean_check_proof,
        mean_check_finals: (ones_at_s.as_canonical_u32(), xc_at_s_mean.as_canonical_u32()),
        var_proof,
        var_finals: (xc_var_a.as_canonical_u32(), xc_var_b.as_canonical_u32()),
        sum_sq_value: sum_sq.as_canonical_u32(),
        g_prod_proof,
        g_prod_finals: (eq_at_gp.as_canonical_u32(), g_at_gp.as_canonical_u32()),
        g_triple_proof,
        g_triple_finals: (eq_at_gt.as_canonical_u32(), gamma_at_gt.as_canonical_u32(), xc_at_gt.as_canonical_u32()),
        t_g_claimed: t_g.as_canonical_u32(),
        h_sq_proof,
        h_sq_finals: (eq_at_hs.as_canonical_u32(), h_at_hs_a.as_canonical_u32(), h_at_hs_b.as_canonical_u32()),
        s_h: s_h_val.as_canonical_u32(),
        g_sq_proof,
        g_sq_finals: (eq_at_gs.as_canonical_u32(), g_at_gs_a.as_canonical_u32(), g_at_gs_b.as_canonical_u32()),
        s_g: s_g_val.as_canonical_u32(),
        xc_commitment,
        gamma_commitment,
        g_commitment,
        h_commitment,
        xc_mean_eval_proof,
        xc_at_mean_s: xc_at_mean_s.as_canonical_u32(),
        xc_var_eval_proof,
        xc_at_var_s: xc_at_var_s.as_canonical_u32(),
        g_prod_eval_proof,
        g_at_prod_s: g_at_prod_s.as_canonical_u32(),
        xc_triple_eval_proof,
        xc_at_triple_s: xc_at_triple_s.as_canonical_u32(),
        gamma_triple_eval_proof,
        gamma_at_triple_s: gamma_at_triple_s.as_canonical_u32(),
        h_sq_eval_proof,
        h_at_sq_s: h_at_sq_s.as_canonical_u32(),
        g_sq_eval_proof,
        g_at_gsq_s: g_at_gsq_s.as_canonical_u32(),
    }
}

/// Verify a squared LayerNorm proof.
pub fn verify_layernorm_sqr(
    proof: &LayerNormSqrProof,
    y: &[F],
    beta: &[F],
    d: usize,
    transcript: &mut Transcript,
) -> bool {
    let log_n = log2_ceil(d);
    let n_pad = 1 << log_n;

    let sum_sq = F::from_canonical_u32(proof.sum_sq_value);
    let d_field = F::from_canonical_u32(d as u32);

    // Absorb (must match prover)
    transcript.absorb_bytes(&proof.xc_commitment.root);
    transcript.absorb_bytes(&proof.gamma_commitment.root);
    transcript.absorb_bytes(&proof.g_commitment.root);
    transcript.absorb_bytes(&proof.h_commitment.root);
    transcript.absorb(proof.mu);
    transcript.absorb(proof.sum_sq_value);

    // --- Mean check ---
    let _mean_point = transcript.squeeze_many(log_n);
    let ones_at_s = F::from_canonical_u32(proof.mean_check_finals.0);
    let xc_at_s_mean = F::from_canonical_u32(proof.mean_check_finals.1);

    if !sumcheck::verify_product(F::zero(), &proof.mean_check_proof, log_n, ones_at_s, xc_at_s_mean, transcript) {
        eprintln!("LayerNormSqr: mean check failed");
        return false;
    }
    if ones_at_s != F::one() {
        eprintln!("LayerNormSqr: ones_at_s != 1");
        return false;
    }
    let xc_at_mean_s = F::from_canonical_u32(proof.xc_at_mean_s);
    if xc_at_mean_s != xc_at_s_mean { return false; }
    let s_mean: Vec<F> = proof.mean_check_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let mut eval_t1 = Transcript::new(b"lnsqr-xc-mean");
    eval_t1.absorb_bytes(&proof.xc_commitment.root);
    if !verify_mle_eval(&proof.xc_commitment, xc_at_mean_s, &s_mean, &proof.xc_mean_eval_proof, &mut eval_t1) {
        eprintln!("LayerNormSqr: xc mean MLE eval failed");
        return false;
    }

    // --- Variance check ---
    let xc_var_a = F::from_canonical_u32(proof.var_finals.0);
    let xc_var_b = F::from_canonical_u32(proof.var_finals.1);
    if !sumcheck::verify_product(sum_sq, &proof.var_proof, log_n, xc_var_a, xc_var_b, transcript) {
        eprintln!("LayerNormSqr: variance check failed");
        return false;
    }
    if xc_var_a != xc_var_b {
        eprintln!("LayerNormSqr: var finals mismatch");
        return false;
    }
    let xc_at_var_s = F::from_canonical_u32(proof.xc_at_var_s);
    if xc_at_var_s != xc_var_a { return false; }
    let s_var: Vec<F> = proof.var_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let mut eval_t2 = Transcript::new(b"lnsqr-xc-var");
    eval_t2.absorb_bytes(&proof.xc_commitment.root);
    if !verify_mle_eval(&proof.xc_commitment, xc_at_var_s, &s_var, &proof.xc_var_eval_proof, &mut eval_t2) {
        eprintln!("LayerNormSqr: xc var MLE eval failed");
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
        eprintln!("LayerNormSqr: g product sumcheck failed");
        return false;
    }
    let s_gp: Vec<F> = proof.g_prod_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let eq_gp_expected = compute_eq_at_point(&gc_point, &s_gp);
    if eq_gp_expected != eq_at_gp {
        eprintln!("LayerNormSqr: eq mismatch in g product");
        return false;
    }
    let g_at_prod_s = F::from_canonical_u32(proof.g_at_prod_s);
    if g_at_prod_s != g_at_gp { return false; }
    let mut eval_t3 = Transcript::new(b"lnsqr-g-prod");
    eval_t3.absorb_bytes(&proof.g_commitment.root);
    if !verify_mle_eval(&proof.g_commitment, g_at_prod_s, &s_gp, &proof.g_prod_eval_proof, &mut eval_t3) {
        eprintln!("LayerNormSqr: g prod MLE eval failed");
        return false;
    }

    // Triple sumcheck: Σ eq · γ · xc = t_g (same claimed value!)
    let (eq_at_gt, gamma_at_gt, xc_at_gt) = (
        F::from_canonical_u32(proof.g_triple_finals.0),
        F::from_canonical_u32(proof.g_triple_finals.1),
        F::from_canonical_u32(proof.g_triple_finals.2),
    );
    if !sumcheck::verify_triple(t_g, &proof.g_triple_proof, log_n, eq_at_gt, gamma_at_gt, xc_at_gt, transcript) {
        eprintln!("LayerNormSqr: g triple sumcheck failed");
        return false;
    }
    let s_gt: Vec<F> = proof.g_triple_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let eq_gt_expected = compute_eq_at_point(&gc_point, &s_gt);
    if eq_gt_expected != eq_at_gt {
        eprintln!("LayerNormSqr: eq mismatch in g triple");
        return false;
    }
    let xc_at_triple_s = F::from_canonical_u32(proof.xc_at_triple_s);
    if xc_at_triple_s != xc_at_gt { return false; }
    let mut eval_t4 = Transcript::new(b"lnsqr-xc-triple");
    eval_t4.absorb_bytes(&proof.xc_commitment.root);
    if !verify_mle_eval(&proof.xc_commitment, xc_at_triple_s, &s_gt, &proof.xc_triple_eval_proof, &mut eval_t4) {
        eprintln!("LayerNormSqr: xc triple MLE eval failed");
        return false;
    }
    let gamma_at_triple_s = F::from_canonical_u32(proof.gamma_at_triple_s);
    if gamma_at_triple_s != gamma_at_gt { return false; }
    let mut eval_t5 = Transcript::new(b"lnsqr-gamma-triple");
    eval_t5.absorb_bytes(&proof.gamma_commitment.root);
    if !verify_mle_eval(&proof.gamma_commitment, gamma_at_triple_s, &s_gt, &proof.gamma_triple_eval_proof, &mut eval_t5) {
        eprintln!("LayerNormSqr: gamma triple MLE eval failed");
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
        eprintln!("LayerNormSqr: h-squared sumcheck failed");
        return false;
    }
    let s_hs: Vec<F> = proof.h_sq_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let eq_hs_expected = compute_eq_at_point(&sq_point, &s_hs);
    if eq_hs_expected != eq_at_hs {
        eprintln!("LayerNormSqr: eq mismatch in h-squared");
        return false;
    }
    if h_at_hs_a != h_at_hs_b {
        eprintln!("LayerNormSqr: h-squared finals mismatch (h·h but a != b)");
        return false;
    }
    let h_at_sq_s = F::from_canonical_u32(proof.h_at_sq_s);
    if h_at_sq_s != h_at_hs_a { return false; }
    let mut eval_t6 = Transcript::new(b"lnsqr-h-sq");
    eval_t6.absorb_bytes(&proof.h_commitment.root);
    if !verify_mle_eval(&proof.h_commitment, h_at_sq_s, &s_hs, &proof.h_sq_eval_proof, &mut eval_t6) {
        eprintln!("LayerNormSqr: h sq MLE eval failed");
        return false;
    }

    // Verify h commitment matches y - beta
    let mut y_pad = y.to_vec();
    y_pad.resize(n_pad, F::zero());
    let mut beta_pad = beta.to_vec();
    beta_pad.resize(n_pad, F::zero());
    let mut h_expected = vec![F::zero(); n_pad];
    for i in 0..d {
        h_expected[i] = y_pad[i] - beta_pad[i];
    }
    let h_commitment_expected = commit_weights_fast(&h_expected);
    if h_commitment_expected.root != proof.h_commitment.root {
        eprintln!("LayerNormSqr: h commitment mismatch (y - beta)");
        return false;
    }

    // g-squared: S_g = Σ eq · g · g
    let (eq_at_gs, g_at_gs_a, g_at_gs_b) = (
        F::from_canonical_u32(proof.g_sq_finals.0),
        F::from_canonical_u32(proof.g_sq_finals.1),
        F::from_canonical_u32(proof.g_sq_finals.2),
    );
    if !sumcheck::verify_triple(s_g_val, &proof.g_sq_proof, log_n, eq_at_gs, g_at_gs_a, g_at_gs_b, transcript) {
        eprintln!("LayerNormSqr: g-squared sumcheck failed");
        return false;
    }
    let s_gs: Vec<F> = proof.g_sq_proof.challenges.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let eq_gs_expected = compute_eq_at_point(&sq_point, &s_gs);
    if eq_gs_expected != eq_at_gs {
        eprintln!("LayerNormSqr: eq mismatch in g-squared");
        return false;
    }
    if g_at_gs_a != g_at_gs_b {
        eprintln!("LayerNormSqr: g-squared finals mismatch (g·g but a != b)");
        return false;
    }
    let g_at_gsq_s = F::from_canonical_u32(proof.g_at_gsq_s);
    if g_at_gsq_s != g_at_gs_a { return false; }
    let mut eval_t7 = Transcript::new(b"lnsqr-g-sq");
    eval_t7.absorb_bytes(&proof.g_commitment.root);
    if !verify_mle_eval(&proof.g_commitment, g_at_gsq_s, &s_gs, &proof.g_sq_eval_proof, &mut eval_t7) {
        eprintln!("LayerNormSqr: g sq MLE eval failed");
        return false;
    }

    // --- Final check: sum_sq · S_h == d · S_g ---
    if sum_sq * s_h_val != d_field * s_g_val {
        eprintln!("LayerNormSqr: sum_sq·S_h != d·S_g (squared output check failed)");
        return false;
    }

    true
}

// =============================================================================
// ROADMAP: Eliminate Python dependency for LayerNorm forward pass
//
// Current state (technical debt):
//   Python computes LN output in float, quantizes, sends via `ln_output_q`.
//   Rust uses this pre-computed output for the squared proof.
//   This is wrong — Python overhead dominates and the prover should be self-contained.
//
// Target:
//   Rust computes LN forward pass internally (fixed-point or float via f64).
//   No `ln_output` field in the binary protocol. Python sends only weights + input.
//   The squared proof (prove_layernorm_sqr) remains — it doesn't need sqrt, which
//   was the whole point. But y is computed Rust-side, not received from Python.
//
// Broader context:
//   - GatedDeltaNet (softmax-free) is the flagship: 100% provable, zero hacks.
//   - GPT-2 attention gap will be closed via lookup-proved exp() in Rust,
//     proving softmax over the (small) seq_len-sized score vector.
//   - Both models fully in Rust — Python becomes just weight loader + tokenizer.
// =============================================================================

// =============================================================================
// Extension Field (EF) versions of LayerNormSqr proof
// =============================================================================

/// LayerNorm proof with extension-field challenges (124-bit soundness).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerNormProofEF {
    pub mu: u32,
    pub r_inv_sqrt: u32,
    pub sum_sq_value: u32,
    // Mean check: Σ 1·xc = 0 (EF product-ones sumcheck)
    pub mean_check_proof: SumcheckProofEF,
    pub xc_at_mean_s: EFElement,
    // Variance check: Σ xc·xc = sum_sq (EF product sumcheck)
    pub var_proof: SumcheckProofEF,
    pub xc_at_var_s: EFElement,
    // Output check: Σ eq·γ·xc = claimed (EF triple sumcheck)
    pub output_proof: SumcheckProofEF,
    pub eq_at_out_s: EFElement,
    pub gamma_at_out_s: EFElement,
    pub xc_at_out_s: EFElement,
    // Commitments
    pub x_commitment: WeightCommitment,
    pub xc_commitment: WeightCommitment,
    pub gamma_commitment: WeightCommitment,
    pub beta_commitment: WeightCommitment,
    // MLE eval proofs (EF)
    pub xc_mean_eval_proof: MleEvalProofEF,
    pub xc_var_eval_proof: MleEvalProofEF,
    pub xc_out_eval_proof: MleEvalProofEF,
    pub gamma_out_eval_proof: MleEvalProofEF,
}

/// Prove LayerNorm with 124-bit EF challenges.
#[allow(dead_code)]
pub fn prove_layernorm_ef(
    x: &[F],
    gamma: &[F],
    beta: &[F],
    y: &[F],
    mu: F,
    r_inv_sqrt: F,
    transcript: &mut Transcript,
) -> LayerNormProofEF {
    let d = x.len();
    let log_n = log2_ceil(d);
    let n_pad = 1 << log_n;

    let mut x_pad = x.to_vec(); x_pad.resize(n_pad, F::zero());
    let mut gamma_pad = gamma.to_vec(); gamma_pad.resize(n_pad, F::zero());
    let mut y_pad = y.to_vec(); y_pad.resize(n_pad, F::zero());

    // x_centered = x - mu (only first d entries)
    let mut xc_pad = vec![F::zero(); n_pad];
    for i in 0..d { xc_pad[i] = x_pad[i] - mu; }

    let sum_sq: F = xc_pad.iter().map(|&v| v * v).sum();

    // Commitments
    let x_commitment = commit_weights_fast(&x_pad);
    let xc_commitment = commit_weights_fast(&xc_pad);
    let gamma_commitment = commit_weights_fast(&gamma_pad);
    let beta_pad = {
        let mut bp = beta.to_vec(); bp.resize(n_pad, F::zero()); bp
    };
    let beta_commitment = commit_weights_fast(&beta_pad);

    // Absorb
    transcript.absorb_bytes(&x_commitment.root);
    transcript.absorb_bytes(&xc_commitment.root);
    transcript.absorb_bytes(&gamma_commitment.root);
    transcript.absorb_bytes(&beta_commitment.root);
    transcript.absorb(mu.as_canonical_u32());
    transcript.absorb(r_inv_sqrt.as_canonical_u32());

    // --- Mean check: Σ 1·xc = 0 (EF product-ones sumcheck) ---
    let _mean_point = transcript.squeeze_ef_many(log_n);
    let xc_ef: Vec<EF> = xc_pad.iter().map(|&v| f_to_ef(v)).collect();
    let (mean_check_proof, _ones_at_s, _xc_at_mean) =
        sumcheck::prove_product_ones_ef_full(&xc_ef, log_n, transcript);

    let s_mean: Vec<EF> = mean_check_proof.challenges.iter().map(|v| v.to_ef()).collect();
    let xc_at_mean_s = mle_evaluate_ef(&xc_pad, &s_mean);
    let mut eval_t1 = Transcript::new(b"layernorm-xc-mean-eval");
    eval_t1.absorb_bytes(&xc_commitment.root);
    let (_, xc_mean_eval_proof) = prove_mle_eval_no_merkle_ef_base(&xc_pad, &s_mean, &mut eval_t1);

    // --- Variance check: Σ xc·xc = sum_sq (EF product sumcheck) ---
    let (var_proof, _xc_var_a, _xc_var_b) =
        sumcheck::prove_product_ef(&xc_pad, &xc_pad, log_n, transcript);

    let s_var: Vec<EF> = var_proof.challenges.iter().map(|v| v.to_ef()).collect();
    let xc_at_var_s = mle_evaluate_ef(&xc_pad, &s_var);
    let mut eval_t2 = Transcript::new(b"layernorm-xc-var-eval");
    eval_t2.absorb_bytes(&xc_commitment.root);
    let (_, xc_var_eval_proof) = prove_mle_eval_no_merkle_ef_base(&xc_pad, &s_var, &mut eval_t2);

    // --- Output check: Σ eq·γ·xc = claimed (EF triple sumcheck) ---
    let eval_point = transcript.squeeze_ef_many(log_n);
    let eq_out = eq_evals_ef(&eval_point);

    let gamma_ef: Vec<EF> = gamma_pad.iter().map(|&v| f_to_ef(v)).collect();
    let xc_ef2: Vec<EF> = xc_pad.iter().map(|&v| f_to_ef(v)).collect();
    let (output_proof, _eq_at_s, _gamma_at_s, _xc_at_s) =
        sumcheck::prove_triple_ef_full(&eq_out, &gamma_ef, &xc_ef2, log_n, transcript);

    let s_out: Vec<EF> = output_proof.challenges.iter().map(|v| v.to_ef()).collect();
    let xc_at_out_s = mle_evaluate_ef(&xc_pad, &s_out);
    let mut eval_t3 = Transcript::new(b"layernorm-xc-output-eval");
    eval_t3.absorb_bytes(&xc_commitment.root);
    let (_, xc_out_eval_proof) = prove_mle_eval_no_merkle_ef_base(&xc_pad, &s_out, &mut eval_t3);

    let gamma_at_out_s = mle_evaluate_ef(&gamma_pad, &s_out);
    let mut eval_t4 = Transcript::new(b"layernorm-gamma-output-eval");
    eval_t4.absorb_bytes(&gamma_commitment.root);
    let (_, gamma_out_eval_proof) = prove_mle_eval_no_merkle_ef_base(&gamma_pad, &s_out, &mut eval_t4);

    LayerNormProofEF {
        mu: mu.as_canonical_u32(),
        r_inv_sqrt: r_inv_sqrt.as_canonical_u32(),
        sum_sq_value: sum_sq.as_canonical_u32(),
        mean_check_proof,
        xc_at_mean_s: EFElement::from_ef(xc_at_mean_s),
        var_proof,
        xc_at_var_s: EFElement::from_ef(xc_at_var_s),
        output_proof,
        eq_at_out_s: EFElement::from_ef(_eq_at_s),
        gamma_at_out_s: EFElement::from_ef(gamma_at_out_s),
        xc_at_out_s: EFElement::from_ef(xc_at_out_s),
        x_commitment, xc_commitment, gamma_commitment, beta_commitment,
        xc_mean_eval_proof, xc_var_eval_proof,
        xc_out_eval_proof, gamma_out_eval_proof,
    }
}

/// Verify a LayerNorm proof with EF challenges.
#[allow(dead_code)]
pub fn verify_layernorm_ef(
    proof: &LayerNormProofEF,
    y: &[F],
    beta: &[F],
    d: usize,
    transcript: &mut Transcript,
) -> bool {
    let log_n = log2_ceil(d);
    let n_pad = 1 << log_n;

    let r = F::from_canonical_u32(proof.r_inv_sqrt);
    let sum_sq = F::from_canonical_u32(proof.sum_sq_value);
    let d_field = F::from_canonical_u32(d as u32);

    // Algebraic check: r² × sum_sq = d
    if r * r * sum_sq != d_field {
        eprintln!("LayerNorm EF: r² * sum_sq != d");
        return false;
    }

    // Absorb (must match prover)
    transcript.absorb_bytes(&proof.x_commitment.root);
    transcript.absorb_bytes(&proof.xc_commitment.root);
    transcript.absorb_bytes(&proof.gamma_commitment.root);
    transcript.absorb_bytes(&proof.beta_commitment.root);
    transcript.absorb(proof.mu);
    transcript.absorb(proof.r_inv_sqrt);

    // --- Mean check: Σ 1·xc = 0 ---
    let _mean_point = transcript.squeeze_ef_many(log_n);

    let ones_at_s = EF::one(); // constant-1 MLE
    let xc_at_mean = proof.xc_at_mean_s.to_ef();

    let (mean_ok, mean_challenges) = sumcheck::verify_product_ef_with_challenges(
        EF::zero(), // claimed sum = 0
        &proof.mean_check_proof,
        ones_at_s,
        xc_at_mean,
        transcript,
    );
    if !mean_ok { eprintln!("LayerNorm EF: mean check failed"); return false; }

    // Verify xc MLE eval at mean challenge point
    let s_mean = mean_challenges;
    let mut eval_t1 = Transcript::new(b"layernorm-xc-mean-eval");
    eval_t1.absorb_bytes(&proof.xc_commitment.root);
    if !verify_mle_eval_ef(&proof.xc_commitment, xc_at_mean, &s_mean, &proof.xc_mean_eval_proof, &mut eval_t1) {
        eprintln!("LayerNorm EF: xc mean eval failed"); return false;
    }

    // --- Variance check: Σ xc·xc = sum_sq ---
    let xc_var = proof.xc_at_var_s.to_ef();

    let (var_ok, var_challenges) = sumcheck::verify_product_ef_with_challenges(
        f_to_ef(sum_sq),
        &proof.var_proof,
        xc_var,
        xc_var, // both finals are xc (same polynomial)
        transcript,
    );
    if !var_ok { eprintln!("LayerNorm EF: var check failed"); return false; }

    let s_var = var_challenges;
    let mut eval_t2 = Transcript::new(b"layernorm-xc-var-eval");
    eval_t2.absorb_bytes(&proof.xc_commitment.root);
    if !verify_mle_eval_ef(&proof.xc_commitment, xc_var, &s_var, &proof.xc_var_eval_proof, &mut eval_t2) {
        eprintln!("LayerNorm EF: xc var eval failed"); return false;
    }

    // --- Output check: Σ eq·γ·xc = (ỹ(pt) - β̃(pt)) / r ---
    let eval_point = transcript.squeeze_ef_many(log_n);

    let mut y_pad = y.to_vec(); y_pad.resize(n_pad, F::zero());
    let mut beta_pad = beta.to_vec(); beta_pad.resize(n_pad, F::zero());
    let y_at_point = mle_evaluate_ef(&y_pad, &eval_point);
    let beta_at_point = mle_evaluate_ef(&beta_pad, &eval_point);
    let r_ef = f_to_ef(r);
    let claimed_xg = (y_at_point - beta_at_point) * r_ef.inverse();

    let eq_at_s = proof.eq_at_out_s.to_ef();
    let gamma_at_s = proof.gamma_at_out_s.to_ef();
    let xc_at_s = proof.xc_at_out_s.to_ef();

    let (out_ok, out_challenges) = sumcheck::verify_triple_ef_with_challenges(
        claimed_xg,
        &proof.output_proof,
        eq_at_s,
        gamma_at_s,
        xc_at_s,
        transcript,
    );
    if !out_ok { eprintln!("LayerNorm EF: output check failed"); return false; }

    // Verify eq(eval_point, s*) independently
    let s_out = &out_challenges;
    let eq_expected = compute_eq_at_point_ef(&eval_point, s_out);
    if eq_expected != eq_at_s {
        eprintln!("LayerNorm EF: eq mismatch at output point");
        return false;
    }

    // Verify xc + gamma MLE eval proofs at output challenge point
    let mut eval_t3 = Transcript::new(b"layernorm-xc-output-eval");
    eval_t3.absorb_bytes(&proof.xc_commitment.root);
    if !verify_mle_eval_ef(&proof.xc_commitment, xc_at_s, s_out, &proof.xc_out_eval_proof, &mut eval_t3) {
        eprintln!("LayerNorm EF: xc output eval failed"); return false;
    }

    let mut eval_t4 = Transcript::new(b"layernorm-gamma-output-eval");
    eval_t4.absorb_bytes(&proof.gamma_commitment.root);
    if !verify_mle_eval_ef(&proof.gamma_commitment, gamma_at_s, s_out, &proof.gamma_out_eval_proof, &mut eval_t4) {
        eprintln!("LayerNorm EF: gamma output eval failed"); return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Modular square root in M31 (p = 2^31 - 1 ≡ 3 mod 4).
    /// sqrt(a) = a^((p+1)/4) = a^(2^29).
    fn mod_sqrt_m31(a: F) -> F {
        let mut result = a;
        for _ in 0..29 {
            result = result * result;
        }
        result
    }

    /// Check if `a` is a quadratic residue mod M31.
    fn is_qr_m31(a: F) -> bool {
        if a == F::zero() {
            return true;
        }
        let mut r = a;
        for _ in 0..30 {
            r = r * r;
        }
        r * a.inverse() == F::one()
    }

    /// Compute r such that r² * sum_sq = d (mod p).
    fn compute_r_for_test(sum_sq: F, d: usize) -> F {
        let d_field = F::from_canonical_u32(d as u32);
        let target = d_field * sum_sq.inverse();
        mod_sqrt_m31(target)
    }

    #[test]
    fn test_layernorm_simple() {
        // Pick x values where d/sum_sq(x_centered) is a QR in M31.
        let d = 4usize;
        let x: Vec<F> = vec![1, 2, 3, 5]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let gamma: Vec<F> = vec![1, 1, 1, 1]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let beta: Vec<F> = vec![0, 0, 0, 0]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();

        let sum_x: F = x.iter().copied().sum();
        let d_inv = F::from_canonical_u32(d as u32).inverse();
        let mu = sum_x * d_inv;

        // x_centered
        let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
        // Verify sum = 0
        let sum_xc: F = xc.iter().copied().sum();
        assert_eq!(sum_xc, F::zero());

        let sum_sq: F = xc.iter().map(|&v| v * v).sum();
        let target = F::from_canonical_u32(d as u32) * sum_sq.inverse();
        assert!(is_qr_m31(target), "d/sum_sq must be QR");
        let r = compute_r_for_test(sum_sq, d);
        assert_eq!(r * r * sum_sq, F::from_canonical_u32(d as u32));

        // y[i] = gamma[i] * xc[i] * r + beta[i]
        let y: Vec<F> = xc
            .iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .map(|((&xci, &gi), &bi)| gi * xci * r + bi)
            .collect();

        let mut pt = Transcript::new(b"layernorm-test");
        let proof = prove_layernorm(&x, &gamma, &beta, &y, mu, r, &mut pt);

        let mut vt = Transcript::new(b"layernorm-test");
        assert!(verify_layernorm(&proof, &y, &beta, d, &mut vt));
    }

    #[test]
    fn test_layernorm_with_gamma_beta() {
        let d = 4usize;
        let x: Vec<F> = vec![1, 2, 3, 7]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let gamma: Vec<F> = vec![2, 3, 4, 5]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let beta: Vec<F> = vec![100, 200, 300, 400]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();

        let sum_x: F = x.iter().copied().sum();
        let mu = sum_x * F::from_canonical_u32(d as u32).inverse();
        let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
        let sum_xc: F = xc.iter().copied().sum();
        assert_eq!(sum_xc, F::zero());

        let sum_sq: F = xc.iter().map(|&v| v * v).sum();
        let target = F::from_canonical_u32(d as u32) * sum_sq.inverse();
        assert!(is_qr_m31(target), "d/sum_sq must be QR");
        let r = compute_r_for_test(sum_sq, d);

        let y: Vec<F> = xc
            .iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .map(|((&xci, &gi), &bi)| gi * xci * r + bi)
            .collect();

        let mut pt = Transcript::new(b"layernorm-test2");
        let proof = prove_layernorm(&x, &gamma, &beta, &y, mu, r, &mut pt);

        let mut vt = Transcript::new(b"layernorm-test2");
        assert!(verify_layernorm(&proof, &y, &beta, d, &mut vt));
    }

    #[test]
    fn test_layernorm_wrong_mu() {
        let d = 4usize;
        let x: Vec<F> = vec![1, 2, 3, 5]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let gamma: Vec<F> = vec![1, 1, 1, 1]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let beta: Vec<F> = vec![0, 0, 0, 0]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();

        // Correct mu
        let sum_x: F = x.iter().copied().sum();
        let mu = sum_x * F::from_canonical_u32(d as u32).inverse();
        let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
        let sum_sq: F = xc.iter().map(|&v| v * v).sum();
        let r = compute_r_for_test(sum_sq, d);
        let y: Vec<F> = xc
            .iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .map(|((&xci, &gi), &bi)| gi * xci * r + bi)
            .collect();

        // Use WRONG mu
        let wrong_mu = mu + F::one();
        let mut pt = Transcript::new(b"layernorm-wrong-mu");
        let proof = prove_layernorm(&x, &gamma, &beta, &y, wrong_mu, r, &mut pt);

        let mut vt = Transcript::new(b"layernorm-wrong-mu");
        // Should fail: mean check will fail (sum of x_centered != 0)
        assert!(!verify_layernorm(&proof, &y, &beta, d, &mut vt));
    }

    #[test]
    fn test_layernorm_wrong_r() {
        let d = 4usize;
        let x: Vec<F> = vec![1, 2, 3, 5]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let gamma: Vec<F> = vec![1, 1, 1, 1]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let beta: Vec<F> = vec![0, 0, 0, 0]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();

        let sum_x: F = x.iter().copied().sum();
        let mu = sum_x * F::from_canonical_u32(d as u32).inverse();
        let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
        let sum_sq: F = xc.iter().map(|&v| v * v).sum();
        let r = compute_r_for_test(sum_sq, d);
        let y: Vec<F> = xc
            .iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .map(|((&xci, &gi), &bi)| gi * xci * r + bi)
            .collect();

        // Use WRONG r
        let wrong_r = r + F::one();
        let mut pt = Transcript::new(b"layernorm-wrong-r");
        let proof = prove_layernorm(&x, &gamma, &beta, &y, mu, wrong_r, &mut pt);

        let mut vt = Transcript::new(b"layernorm-wrong-r");
        // Should fail: r² * sum_sq != d
        assert!(!verify_layernorm(&proof, &y, &beta, d, &mut vt));
    }

    // --- Squared variant tests ---

    fn compute_ln_output_with_r(xc: &[F], gamma: &[F], beta: &[F], r: F) -> Vec<F> {
        xc.iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .map(|((&xci, &gi), &bi)| gi * xci * r + bi)
            .collect()
    }

    #[test]
    fn test_layernorm_sqr_simple() {
        let d = 4usize;
        let x: Vec<F> = vec![1, 2, 3, 5].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let gamma: Vec<F> = vec![1, 1, 1, 1].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let beta: Vec<F> = vec![0, 0, 0, 0].iter().map(|&v| F::from_canonical_u32(v)).collect();

        let sum_x: F = x.iter().copied().sum();
        let mu = sum_x * F::from_canonical_u32(d as u32).inverse();
        let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
        let sum_sq: F = xc.iter().map(|&v| v * v).sum();
        let target = F::from_canonical_u32(d as u32) * sum_sq.inverse();
        assert!(is_qr_m31(target));
        let r = compute_r_for_test(sum_sq, d);
        let y = compute_ln_output_with_r(&xc, &gamma, &beta, r);

        let mut pt = Transcript::new(b"lnsqr-test");
        let proof = prove_layernorm_sqr(&x, &gamma, &beta, &y, mu, &mut pt);
        let mut vt = Transcript::new(b"lnsqr-test");
        assert!(verify_layernorm_sqr(&proof, &y, &beta, d, &mut vt));
    }

    #[test]
    fn test_layernorm_sqr_with_gamma_beta() {
        let d = 4usize;
        let x: Vec<F> = vec![1, 2, 3, 7].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let gamma: Vec<F> = vec![2, 3, 4, 5].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let beta: Vec<F> = vec![100, 200, 300, 400].iter().map(|&v| F::from_canonical_u32(v)).collect();

        let sum_x: F = x.iter().copied().sum();
        let mu = sum_x * F::from_canonical_u32(d as u32).inverse();
        let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
        let sum_sq: F = xc.iter().map(|&v| v * v).sum();
        let target = F::from_canonical_u32(d as u32) * sum_sq.inverse();
        assert!(is_qr_m31(target));
        let r = compute_r_for_test(sum_sq, d);
        let y = compute_ln_output_with_r(&xc, &gamma, &beta, r);

        let mut pt = Transcript::new(b"lnsqr-test2");
        let proof = prove_layernorm_sqr(&x, &gamma, &beta, &y, mu, &mut pt);
        let mut vt = Transcript::new(b"lnsqr-test2");
        assert!(verify_layernorm_sqr(&proof, &y, &beta, d, &mut vt));
    }

    #[test]
    fn test_layernorm_sqr_non_qr() {
        // Find x where d/sum_sq is NOT a QR — the squared proof handles this.
        // The trick: for non-QR, no y in M31 satisfies y = γ·xc·r + β (r doesn't exist).
        // But we can construct y such that (y-β)²·sum_sq = (γ·xc)²·d coordinate-wise.
        // This has a solution: h[i]² = g[i]² · (d/sum_sq), so h[i] = ±g[i] · sqrt(d/sum_sq).
        // Since d/sum_sq is non-QR, we can't find h in M31.
        //
        // HOWEVER, the proof checks sum_sq · Σ eq·h² == d · Σ eq·g² at a random point.
        // If we set y such that h[i]² · sum_sq == g[i]² · d for ALL i,
        // then the sumcheck passes trivially.
        //
        // For non-QR case: if g[i]² · d / sum_sq is a QR (which it is — it's (g[i])²·(d/sum_sq),
        // and (d/sum_sq) is non-QR, and g[i]² is always QR (it's a square), so the product
        // is QR iff d/sum_sq is QR... which it isn't. So h[i]² can't equal a non-QR.
        //
        // This means: for non-QR, NO y exists in M31 that satisfies the per-coordinate constraint.
        // The proof checks the SUMMED version (over a random eval point), which is weaker.
        // The sum could equal zero on both sides even if individual terms don't match.
        //
        // In the real system: Python computes y in FLOAT, quantizes to M31. The quantized y
        // won't exactly satisfy h²·sum_sq = g²·d, but may pass the polynomial check at
        // random points (the "error" is at quantization noise level).
        //
        // For this test: just verify the code doesn't crash on non-QR inputs.
        let d = 4usize;
        let gamma: Vec<F> = vec![1, 1, 1, 1].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let beta: Vec<F> = vec![0, 0, 0, 0].iter().map(|&v| F::from_canonical_u32(v)).collect();

        let mut found = false;
        for seed in 1u32..1000 {
            let x: Vec<F> = (0..d).map(|i| F::from_canonical_u32(seed + i as u32 * 7)).collect();
            let sum_x: F = x.iter().copied().sum();
            let mu = sum_x * F::from_canonical_u32(d as u32).inverse();
            let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
            let sum_sq: F = xc.iter().map(|&v| v * v).sum();
            if sum_sq == F::zero() { continue; }
            let target = F::from_canonical_u32(d as u32) * sum_sq.inverse();

            if !is_qr_m31(target) {
                // Simulate Python: compute LN in f64, quantize
                let x_f64: Vec<f64> = x.iter().map(|v| v.as_canonical_u32() as f64).collect();
                let mu_f64: f64 = x_f64.iter().sum::<f64>() / d as f64;
                let xc_f64: Vec<f64> = x_f64.iter().map(|v| v - mu_f64).collect();
                let var: f64 = xc_f64.iter().map(|v| v * v).sum::<f64>() / d as f64;
                let std = var.sqrt();
                let y_f64: Vec<f64> = (0..d).map(|i| xc_f64[i] / std).collect();
                let p = (1u64 << 31) - 1;
                let y: Vec<F> = y_f64.iter().map(|&v| {
                    let v_round = v.round() as i64;
                    let v_mod = ((v_round % p as i64) + p as i64) as u64 % p;
                    F::from_canonical_u32(v_mod as u32)
                }).collect();

                let mut pt = Transcript::new(b"lnsqr-nonqr");
                let proof = prove_layernorm_sqr(&x, &gamma, &beta, &y, mu, &mut pt);
                let mut vt = Transcript::new(b"lnsqr-nonqr");
                let result = verify_layernorm_sqr(&proof, &y, &beta, d, &mut vt);
                // For non-QR, the f64-quantized y cannot exactly satisfy the M31
                // algebraic identity h²·sum_sq = g²·d, so verification should reject.
                assert!(!result,
                    "Non-QR with quantized y should fail verification (seed={})", seed);
                found = true;
                break;
            }
        }
        assert!(found, "Could not find non-QR case in first 1000 seeds");
    }

    #[test]
    fn test_layernorm_sqr_wrong_output() {
        let d = 4usize;
        let x: Vec<F> = vec![1, 2, 3, 5].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let gamma: Vec<F> = vec![1, 1, 1, 1].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let beta: Vec<F> = vec![0, 0, 0, 0].iter().map(|&v| F::from_canonical_u32(v)).collect();

        let sum_x: F = x.iter().copied().sum();
        let mu = sum_x * F::from_canonical_u32(d as u32).inverse();
        let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
        let sum_sq: F = xc.iter().map(|&v| v * v).sum();
        let r = compute_r_for_test(sum_sq, d);
        let mut y = compute_ln_output_with_r(&xc, &gamma, &beta, r);
        y[0] = y[0] + F::one(); // tamper

        let mut pt = Transcript::new(b"lnsqr-wrong");
        let proof = prove_layernorm_sqr(&x, &gamma, &beta, &y, mu, &mut pt);
        let mut vt = Transcript::new(b"lnsqr-wrong");
        assert!(!verify_layernorm_sqr(&proof, &y, &beta, d, &mut vt));
    }

    // ===== Extension field LayerNorm tests =====

    #[test]
    fn test_layernorm_ef_simple() {
        let d = 4usize;
        let x: Vec<F> = vec![1, 2, 3, 5].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let gamma: Vec<F> = vec![1, 1, 1, 1].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let beta: Vec<F> = vec![0, 0, 0, 0].iter().map(|&v| F::from_canonical_u32(v)).collect();

        let sum_x: F = x.iter().copied().sum();
        let mu = sum_x * F::from_canonical_u32(d as u32).inverse();
        let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
        let sum_sq: F = xc.iter().map(|&v| v * v).sum();
        assert!(is_qr_m31(F::from_canonical_u32(d as u32) * sum_sq.inverse()));
        let r = compute_r_for_test(sum_sq, d);

        let y: Vec<F> = xc.iter().zip(gamma.iter()).zip(beta.iter())
            .map(|((&xci, &gi), &bi)| gi * xci * r + bi).collect();

        let mut pt = Transcript::new(b"layernorm-ef-test");
        let proof = prove_layernorm_ef(&x, &gamma, &beta, &y, mu, r, &mut pt);

        let mut vt = Transcript::new(b"layernorm-ef-test");
        assert!(verify_layernorm_ef(&proof, &y, &beta, d, &mut vt));
    }

    #[test]
    fn test_layernorm_ef_with_gamma_beta() {
        let d = 4usize;
        let x: Vec<F> = vec![1, 2, 3, 7].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let gamma: Vec<F> = vec![2, 3, 4, 5].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let beta: Vec<F> = vec![100, 200, 300, 400].iter().map(|&v| F::from_canonical_u32(v)).collect();

        let sum_x: F = x.iter().copied().sum();
        let mu = sum_x * F::from_canonical_u32(d as u32).inverse();
        let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
        let sum_sq: F = xc.iter().map(|&v| v * v).sum();
        assert!(is_qr_m31(F::from_canonical_u32(d as u32) * sum_sq.inverse()));
        let r = compute_r_for_test(sum_sq, d);

        let y: Vec<F> = xc.iter().zip(gamma.iter()).zip(beta.iter())
            .map(|((&xci, &gi), &bi)| gi * xci * r + bi).collect();

        let mut pt = Transcript::new(b"layernorm-ef-test2");
        let proof = prove_layernorm_ef(&x, &gamma, &beta, &y, mu, r, &mut pt);

        let mut vt = Transcript::new(b"layernorm-ef-test2");
        assert!(verify_layernorm_ef(&proof, &y, &beta, d, &mut vt));
    }

    #[test]
    fn test_layernorm_ef_tampered() {
        let d = 4usize;
        let x: Vec<F> = vec![1, 2, 3, 5].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let gamma: Vec<F> = vec![1, 1, 1, 1].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let beta: Vec<F> = vec![0, 0, 0, 0].iter().map(|&v| F::from_canonical_u32(v)).collect();

        let sum_x: F = x.iter().copied().sum();
        let mu = sum_x * F::from_canonical_u32(d as u32).inverse();
        let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
        let sum_sq: F = xc.iter().map(|&v| v * v).sum();
        let r = compute_r_for_test(sum_sq, d);

        let mut y: Vec<F> = xc.iter().zip(gamma.iter()).zip(beta.iter())
            .map(|((&xci, &gi), &bi)| gi * xci * r + bi).collect();
        y[0] = y[0] + F::one(); // tamper

        let mut pt = Transcript::new(b"layernorm-ef-bad");
        let proof = prove_layernorm_ef(&x, &gamma, &beta, &y, mu, r, &mut pt);

        let mut vt = Transcript::new(b"layernorm-ef-bad");
        assert!(!verify_layernorm_ef(&proof, &y, &beta, d, &mut vt));
    }
}
