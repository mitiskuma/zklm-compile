//! Proofs for element-wise operations over M31 via sumcheck.
//!
//! - **Hadamard product** (c = a ⊙ b): SiLU gating, attention output gating
//! - **Element-wise add** (c = a + b): residual connections
//! - **Scalar multiply** (c = s · a): RMSNorm scaling, attention score scaling
//!
//! Each operation reduces to sumcheck over the eq polynomial, producing
//! MLE evaluation claims that chain into the GKR-style proof composition.

use p3_field::{AbstractField, PrimeField32};
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::field::common::{compute_eq_at_point, compute_eq_at_point_ef};
use crate::field::m31_ops::*;
use crate::proving::sumcheck::{self, EF, EFElement, SumcheckProof, SumcheckProofEF, Transcript};
use crate::proving::weight_commitment::{
    commit_weights_fast, prove_mle_eval_no_merkle, prove_mle_eval_no_merkle_ef_base,
    verify_mle_eval, verify_mle_eval_ef, MleEvalProof, MleEvalProofEF, WeightCommitment,
};

type F = Mersenne31;

// ===== Hadamard product: c = a ⊙ b =====
//
// Prove via triple sumcheck: Σ eq(r,x) · a[x] · b[x] = v
// Then verify c̃(r) = v via MLE eval proof against c's commitment.

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HadamardProof {
    /// Triple sumcheck proving Σ eq(r,x)·a[x]·b[x] = v
    pub product_sumcheck: SumcheckProof,
    /// Final evaluations (eq_at_s, a_at_s, b_at_s) at sumcheck challenge point
    pub product_finals: (u32, u32, u32),
    /// c̃(r) claimed value
    pub c_eval: u32,
    /// MLE eval proof: c̃(r) = c_eval against commitment
    pub c_eval_proof: MleEvalProof,
    /// Commitment to c
    pub c_commitment: WeightCommitment,
}

/// Prove c = a ⊙ b (element-wise product).
///
/// Strategy: pick random r, prove Σ eq(r,x)·a[x]·b[x] = v via triple sumcheck,
/// then prove c̃(r) = v via MLE eval proof. If both hold, c = a ⊙ b with
/// high probability.
pub fn prove_hadamard(
    a: &[F],
    b: &[F],
    c: &[F],
    point: &[F],
    transcript: &mut Transcript,
) -> HadamardProof {
    let log_n = point.len();
    let n_pad = 1 << log_n;

    let mut a_pad = a.to_vec();
    a_pad.resize(n_pad, F::zero());
    let mut b_pad = b.to_vec();
    b_pad.resize(n_pad, F::zero());
    let mut c_pad = c.to_vec();
    c_pad.resize(n_pad, F::zero());

    // Commit to c
    let c_commitment = commit_weights_fast(&c_pad);
    transcript.absorb_bytes(&c_commitment.root);

    // eq(r, x) for all x in {0,1}^k
    let eq_r = eq_evals(point);

    // Triple sumcheck: Σ eq(r,x) · a[x] · b[x] = c̃(r)
    // Because c = a ⊙ b, this sum equals Σ eq(r,x) · c[x] = c̃(r).
    let (product_sumcheck, eq_at_s, a_at_s, b_at_s) =
        sumcheck::prove_triple_best(&eq_r, &a_pad, &b_pad, log_n, transcript);

    // The sumcheck claimed sum is c̃(r)
    let c_eval = mle_evaluate(&c_pad, point);

    // MLE eval proof: prove c̃(r) = c_eval against c_commitment
    let mut eval_transcript = Transcript::new(b"hadamard-c-eval");
    eval_transcript.absorb_bytes(&c_commitment.root);
    let (_, c_eval_proof) = prove_mle_eval_no_merkle(&c_pad, point, &mut eval_transcript);

    HadamardProof {
        product_sumcheck,
        product_finals: (
            eq_at_s.as_canonical_u32(),
            a_at_s.as_canonical_u32(),
            b_at_s.as_canonical_u32(),
        ),
        c_eval: c_eval.as_canonical_u32(),
        c_eval_proof,
        c_commitment,
    }
}

/// Verify a Hadamard product proof.
///
/// Checks:
/// 1. Triple sumcheck: Σ eq(r,x)·a[x]·b[x] = c_eval is valid
/// 2. eq(r, s*) matches prover's claim (computed independently)
/// 3. c̃(r) = c_eval via MLE eval proof against c_commitment
pub fn verify_hadamard(
    proof: &HadamardProof,
    point: &[F],
    transcript: &mut Transcript,
) -> bool {
    let log_n = point.len();

    // Absorb c commitment (must match prover)
    transcript.absorb_bytes(&proof.c_commitment.root);

    let c_eval = F::from_canonical_u32(proof.c_eval);
    let eq_at_s = F::from_canonical_u32(proof.product_finals.0);
    let a_at_s = F::from_canonical_u32(proof.product_finals.1);
    let b_at_s = F::from_canonical_u32(proof.product_finals.2);

    // Verify triple sumcheck: claimed_sum = c_eval
    if !sumcheck::verify_triple(
        c_eval,
        &proof.product_sumcheck,
        log_n,
        eq_at_s,
        a_at_s,
        b_at_s,
        transcript,
    ) {
        return false;
    }

    // Verify eq(r, s*) independently
    let s_point: Vec<F> = proof
        .product_sumcheck
        .challenges
        .iter()
        .map(|&v| F::from_canonical_u32(v))
        .collect();
    let eq_expected = compute_eq_at_point(point, &s_point);
    if eq_expected != eq_at_s {
        return false;
    }

    // Verify c̃(r) = c_eval via MLE eval proof
    let mut eval_transcript = Transcript::new(b"hadamard-c-eval");
    eval_transcript.absorb_bytes(&proof.c_commitment.root);
    if !verify_mle_eval(
        &proof.c_commitment,
        c_eval,
        point,
        &proof.c_eval_proof,
        &mut eval_transcript,
    ) {
        return false;
    }

    true
}

// ===== Element-wise addition: c = a + b =====
//
// MLE is linear: c̃(r) = ã(r) + b̃(r).
// Prove ã(r) and b̃(r) separately via product sumcheck with eq,
// then verify c̃(r) = ã(r) + b̃(r).

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AddProof {
    /// Product sumcheck: Σ eq(r,x)·a[x] = ã(r)
    pub a_sumcheck: SumcheckProof,
    /// Final evaluations (eq_at_s, a_at_s)
    pub a_finals: (u32, u32),
    /// Product sumcheck: Σ eq(r,x)·b[x] = b̃(r)
    pub b_sumcheck: SumcheckProof,
    /// Final evaluations (eq_at_s, b_at_s)
    pub b_finals: (u32, u32),
    /// c̃(r) claimed value (should equal a_eval + b_eval)
    pub c_eval: u32,
    /// MLE eval proof: c̃(r) = c_eval against commitment
    pub c_eval_proof: MleEvalProof,
    /// Commitment to c
    pub c_commitment: WeightCommitment,
}

/// Prove c = a + b (element-wise addition).
///
/// Strategy: prove ã(r) and b̃(r) via product sumchecks with eq(r, ·),
/// then prove c̃(r) = ã(r) + b̃(r) via MLE eval proof.
pub fn prove_add(
    a: &[F],
    b: &[F],
    c: &[F],
    point: &[F],
    transcript: &mut Transcript,
) -> AddProof {
    let log_n = point.len();
    let n_pad = 1 << log_n;

    let mut a_pad = a.to_vec();
    a_pad.resize(n_pad, F::zero());
    let mut b_pad = b.to_vec();
    b_pad.resize(n_pad, F::zero());
    let mut c_pad = c.to_vec();
    c_pad.resize(n_pad, F::zero());

    let c_commitment = commit_weights_fast(&c_pad);
    transcript.absorb_bytes(&c_commitment.root);

    let eq_r = eq_evals(point);

    // Product sumcheck for a: Σ eq(r,x)·a[x] = ã(r)
    let (a_sumcheck, a_eq_at_s, a_at_s) =
        sumcheck::prove_product_best(&eq_r, &a_pad, log_n, transcript);

    // Product sumcheck for b: Σ eq(r,x)·b[x] = b̃(r)
    let (b_sumcheck, b_eq_at_s, b_at_s) =
        sumcheck::prove_product_best(&eq_r, &b_pad, log_n, transcript);

    let c_eval = mle_evaluate(&c_pad, point);

    // MLE eval proof for c̃(r)
    let mut eval_transcript = Transcript::new(b"add-c-eval");
    eval_transcript.absorb_bytes(&c_commitment.root);
    let (_, c_eval_proof) = prove_mle_eval_no_merkle(&c_pad, point, &mut eval_transcript);

    AddProof {
        a_sumcheck,
        a_finals: (a_eq_at_s.as_canonical_u32(), a_at_s.as_canonical_u32()),
        b_sumcheck,
        b_finals: (b_eq_at_s.as_canonical_u32(), b_at_s.as_canonical_u32()),
        c_eval: c_eval.as_canonical_u32(),
        c_eval_proof,
        c_commitment,
    }
}

/// Verify an element-wise addition proof.
///
/// Checks:
/// 1. Product sumcheck for ã(r) is valid
/// 2. Product sumcheck for b̃(r) is valid
/// 3. ã(r) + b̃(r) = c̃(r)
/// 4. c̃(r) via MLE eval proof against c_commitment
pub fn verify_add(
    proof: &AddProof,
    point: &[F],
    transcript: &mut Transcript,
) -> bool {
    transcript.absorb_bytes(&proof.c_commitment.root);

    let a_eq_at_s = F::from_canonical_u32(proof.a_finals.0);
    let a_at_s = F::from_canonical_u32(proof.a_finals.1);

    // Verify a sumcheck (derive challenge point from transcript)
    let a_eval = mle_evaluate_from_sumcheck_claim(&proof.a_sumcheck, point);
    let (a_ok, s_a) = sumcheck::verify_product_with_challenges(
        a_eval,
        &proof.a_sumcheck,
        a_eq_at_s,
        a_at_s,
        transcript,
    );
    if !a_ok { return false; }
    // Verify eq independently using derived challenges
    if compute_eq_at_point(point, &s_a) != a_eq_at_s {
        return false;
    }

    let b_eq_at_s = F::from_canonical_u32(proof.b_finals.0);
    let b_at_s = F::from_canonical_u32(proof.b_finals.1);

    let b_eval = mle_evaluate_from_sumcheck_claim(&proof.b_sumcheck, point);
    let (b_ok, s_b) = sumcheck::verify_product_with_challenges(
        b_eval,
        &proof.b_sumcheck,
        b_eq_at_s,
        b_at_s,
        transcript,
    );
    if !b_ok { return false; }
    if compute_eq_at_point(point, &s_b) != b_eq_at_s {
        return false;
    }

    // Check a_eval + b_eval = c_eval
    let c_eval = F::from_canonical_u32(proof.c_eval);
    if a_eval + b_eval != c_eval {
        return false;
    }

    // Verify c MLE eval proof
    let mut eval_transcript = Transcript::new(b"add-c-eval");
    eval_transcript.absorb_bytes(&proof.c_commitment.root);
    if !verify_mle_eval(
        &proof.c_commitment,
        c_eval,
        point,
        &proof.c_eval_proof,
        &mut eval_transcript,
    ) {
        return false;
    }

    true
}

// ===== Scalar multiply: c = s · a =====
//
// c̃(r) = s · ã(r). Prove ã(r) via product sumcheck, verifier checks c̃(r) = s · ã(r).

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalarMulProof {
    /// The public scalar
    pub scalar: u32,
    /// Product sumcheck: Σ eq(r,x)·a[x] = ã(r)
    pub a_sumcheck: SumcheckProof,
    /// Final evaluations (eq_at_s, a_at_s)
    pub a_finals: (u32, u32),
}

/// Prove c = s · a (scalar multiplication).
///
/// Since s is public, we only need to prove ã(r) = v. The verifier
/// checks c̃(r) = s · v independently.
#[allow(dead_code)]
pub fn prove_scalar_mul(
    a: &[F],
    scalar: F,
    _c: &[F],
    point: &[F],
    transcript: &mut Transcript,
) -> ScalarMulProof {
    let log_n = point.len();
    let n_pad = 1 << log_n;

    let mut a_pad = a.to_vec();
    a_pad.resize(n_pad, F::zero());

    let eq_r = eq_evals(point);

    // Product sumcheck: Σ eq(r,x)·a[x] = ã(r)
    let (a_sumcheck, eq_at_s, a_at_s) =
        sumcheck::prove_product_best(&eq_r, &a_pad, log_n, transcript);

    ScalarMulProof {
        scalar: scalar.as_canonical_u32(),
        a_sumcheck,
        a_finals: (eq_at_s.as_canonical_u32(), a_at_s.as_canonical_u32()),
    }
}

/// Verify a scalar multiplication proof.
///
/// Returns (valid, c_at_point) where c_at_point = scalar · ã(r).
/// The caller can use c_at_point to chain into the next layer.
#[allow(dead_code)]
pub fn verify_scalar_mul(
    proof: &ScalarMulProof,
    point: &[F],
    transcript: &mut Transcript,
) -> (bool, F) {
    let eq_at_s = F::from_canonical_u32(proof.a_finals.0);
    let a_at_s = F::from_canonical_u32(proof.a_finals.1);
    let scalar = F::from_canonical_u32(proof.scalar);

    // Recover a_eval from sumcheck
    let a_eval = mle_evaluate_from_sumcheck_claim(&proof.a_sumcheck, point);

    let (ok, s_point) = sumcheck::verify_product_with_challenges(
        a_eval,
        &proof.a_sumcheck,
        eq_at_s,
        a_at_s,
        transcript,
    );
    if !ok { return (false, F::zero()); }

    // Verify eq independently using derived challenges
    if compute_eq_at_point(point, &s_point) != eq_at_s {
        return (false, F::zero());
    }

    let c_at_point = scalar * a_eval;
    (true, c_at_point)
}

// ===== Helpers =====

/// Recover the claimed sum from a product sumcheck proof.
/// The claimed sum = round_polys[0][0] + round_polys[0][1] (first round: p(0) + p(1)).
fn mle_evaluate_from_sumcheck_claim(proof: &SumcheckProof, _point: &[F]) -> F {
    let s0 = F::from_canonical_u32(proof.round_polys[0][0]);
    let s1 = F::from_canonical_u32(proof.round_polys[0][1]);
    s0 + s1
}

// ===== Extension-field (EF) versions =====
//
// Same operations as above, but challenge points are in the degree-4 extension
// field EF = BinomialExtensionField<Complex<M31>, 2> for 124-bit soundness.
// Polynomial coefficients remain in the base field F; only evaluation points
// and Fiat-Shamir challenges live in EF.

/// Recover the claimed sum from an EF sumcheck proof.
/// claimed_sum = round_polys[0][0] + round_polys[0][1].
fn mle_evaluate_from_sumcheck_claim_ef(proof: &SumcheckProofEF) -> EF {
    let s0 = proof.round_polys[0][0].to_ef();
    let s1 = proof.round_polys[0][1].to_ef();
    s0 + s1
}

// ===== Hadamard product EF: c = a ⊙ b =====

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HadamardProofEF {
    /// Triple sumcheck proving Σ eq(r,x)·a[x]·b[x] = v (EF challenges)
    pub product_sumcheck: SumcheckProofEF,
    /// Final evaluations (eq_at_s, a_at_s, b_at_s) in EF
    pub product_finals: (EFElement, EFElement, EFElement),
    /// c̃(r) claimed value in EF
    pub c_eval: EFElement,
    /// MLE eval proof: c̃(r) = c_eval against commitment (EF)
    pub c_eval_proof: MleEvalProofEF,
    /// Commitment to c
    pub c_commitment: WeightCommitment,
}

/// Prove c = a ⊙ b with extension-field challenge point.
pub fn prove_hadamard_ef(
    a: &[F],
    b: &[F],
    c: &[F],
    point: &[EF],
    transcript: &mut Transcript,
) -> HadamardProofEF {
    let log_n = point.len();
    let n_pad = 1 << log_n;

    let mut a_pad = a.to_vec();
    a_pad.resize(n_pad, F::zero());
    let mut b_pad = b.to_vec();
    b_pad.resize(n_pad, F::zero());
    let mut c_pad = c.to_vec();
    c_pad.resize(n_pad, F::zero());

    // Commit to c
    let c_commitment = commit_weights_fast(&c_pad);
    transcript.absorb_bytes(&c_commitment.root);

    // eq(r, x) for all x in {0,1}^k — in EF
    let eq_r = eq_evals_ef(point);

    // Convert a_pad, b_pad to EF for triple sumcheck
    let a_ef: Vec<EF> = a_pad.iter().map(|&v| f_to_ef(v)).collect();
    let b_ef: Vec<EF> = b_pad.iter().map(|&v| f_to_ef(v)).collect();

    // Triple sumcheck: Σ eq(r,x) · a[x] · b[x] = c̃(r)
    let (product_sumcheck, eq_at_s, a_at_s, b_at_s) =
        sumcheck::prove_triple_ef_full(&eq_r, &a_ef, &b_ef, log_n, transcript);

    // c̃(r) in EF: base-field c evaluated at EF point
    let c_eval = mle_evaluate_ef(&c_pad, point);

    // MLE eval proof: prove c̃(r) = c_eval against c_commitment
    let mut eval_transcript = Transcript::new(b"hadamard-c-eval-ef");
    eval_transcript.absorb_bytes(&c_commitment.root);
    let (_, c_eval_proof) = prove_mle_eval_no_merkle_ef_base(&c_pad, point, &mut eval_transcript);

    HadamardProofEF {
        product_sumcheck,
        product_finals: (
            EFElement::from_ef(eq_at_s),
            EFElement::from_ef(a_at_s),
            EFElement::from_ef(b_at_s),
        ),
        c_eval: EFElement::from_ef(c_eval),
        c_eval_proof,
        c_commitment,
    }
}

/// Verify a Hadamard product proof with EF challenges.
pub fn verify_hadamard_ef(
    proof: &HadamardProofEF,
    point: &[EF],
    transcript: &mut Transcript,
) -> bool {
    let _log_n = point.len();

    transcript.absorb_bytes(&proof.c_commitment.root);

    let c_eval = proof.c_eval.to_ef();
    let eq_at_s = proof.product_finals.0.to_ef();
    let a_at_s = proof.product_finals.1.to_ef();
    let b_at_s = proof.product_finals.2.to_ef();

    // Verify triple sumcheck — derive challenges for s*
    let (triple_ok, s_point) = sumcheck::verify_triple_ef_with_challenges(
        c_eval, &proof.product_sumcheck, eq_at_s, a_at_s, b_at_s, transcript,
    );
    if !triple_ok {
        return false;
    }
    let eq_expected = compute_eq_at_point_ef(point, &s_point);
    if eq_expected != eq_at_s {
        return false;
    }

    // Verify c̃(r) = c_eval via MLE eval proof
    let mut eval_transcript = Transcript::new(b"hadamard-c-eval-ef");
    eval_transcript.absorb_bytes(&proof.c_commitment.root);
    if !verify_mle_eval_ef(
        &proof.c_commitment,
        c_eval,
        point,
        &proof.c_eval_proof,
        &mut eval_transcript,
    ) {
        return false;
    }

    true
}

// ===== Element-wise addition EF: c = a + b =====

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AddProofEF {
    /// Product sumcheck: Σ eq(r,x)·a[x] = ã(r) (EF)
    pub a_sumcheck: SumcheckProofEF,
    /// Final evaluations (eq_at_s, a_at_s) in EF
    pub a_finals: (EFElement, EFElement),
    /// Product sumcheck: Σ eq(r,x)·b[x] = b̃(r) (EF)
    pub b_sumcheck: SumcheckProofEF,
    /// Final evaluations (eq_at_s, b_at_s) in EF
    pub b_finals: (EFElement, EFElement),
    /// c̃(r) claimed value in EF
    pub c_eval: EFElement,
    /// MLE eval proof: c̃(r) = c_eval against commitment (EF)
    pub c_eval_proof: MleEvalProofEF,
    /// Commitment to c
    pub c_commitment: WeightCommitment,
}

/// Prove c = a + b with extension-field challenge point.
pub fn prove_add_ef(
    a: &[F],
    b: &[F],
    c: &[F],
    point: &[EF],
    transcript: &mut Transcript,
) -> AddProofEF {
    let log_n = point.len();
    let n_pad = 1 << log_n;

    let mut a_pad = a.to_vec();
    a_pad.resize(n_pad, F::zero());
    let mut b_pad = b.to_vec();
    b_pad.resize(n_pad, F::zero());
    let mut c_pad = c.to_vec();
    c_pad.resize(n_pad, F::zero());

    let c_commitment = commit_weights_fast(&c_pad);
    transcript.absorb_bytes(&c_commitment.root);

    let eq_r = eq_evals_ef(point);

    // Convert a_pad, b_pad to EF for product sumcheck
    let a_ef: Vec<EF> = a_pad.iter().map(|&v| f_to_ef(v)).collect();
    let b_ef: Vec<EF> = b_pad.iter().map(|&v| f_to_ef(v)).collect();

    // Product sumcheck for a: Σ eq(r,x)·a[x] = ã(r)
    let (a_sumcheck, a_eq_at_s, a_at_s) =
        sumcheck::prove_product_ef_full(&eq_r, &a_ef, log_n, transcript);

    // Product sumcheck for b: Σ eq(r,x)·b[x] = b̃(r)
    let (b_sumcheck, b_eq_at_s, b_at_s) =
        sumcheck::prove_product_ef_full(&eq_r, &b_ef, log_n, transcript);

    let c_eval = mle_evaluate_ef(&c_pad, point);

    // MLE eval proof for c̃(r)
    let mut eval_transcript = Transcript::new(b"add-c-eval-ef");
    eval_transcript.absorb_bytes(&c_commitment.root);
    let (_, c_eval_proof) = prove_mle_eval_no_merkle_ef_base(&c_pad, point, &mut eval_transcript);

    AddProofEF {
        a_sumcheck,
        a_finals: (EFElement::from_ef(a_eq_at_s), EFElement::from_ef(a_at_s)),
        b_sumcheck,
        b_finals: (EFElement::from_ef(b_eq_at_s), EFElement::from_ef(b_at_s)),
        c_eval: EFElement::from_ef(c_eval),
        c_eval_proof,
        c_commitment,
    }
}

/// Verify an element-wise addition proof with EF challenges.
pub fn verify_add_ef(
    proof: &AddProofEF,
    point: &[EF],
    transcript: &mut Transcript,
) -> bool {
    let _log_n = point.len();

    transcript.absorb_bytes(&proof.c_commitment.root);

    let a_eq_at_s = proof.a_finals.0.to_ef();
    let a_at_s = proof.a_finals.1.to_ef();

    // Verify a sumcheck
    let a_eval = mle_evaluate_from_sumcheck_claim_ef(&proof.a_sumcheck);
    let (a_ok, s_a) = sumcheck::verify_product_ef_with_challenges(
        a_eval, &proof.a_sumcheck, a_eq_at_s, a_at_s, transcript);
    if !a_ok { return false; }
    if compute_eq_at_point_ef(point, &s_a) != a_eq_at_s { return false; }

    let b_eq_at_s = proof.b_finals.0.to_ef();
    let b_at_s = proof.b_finals.1.to_ef();
    let b_eval = mle_evaluate_from_sumcheck_claim_ef(&proof.b_sumcheck);
    let (b_ok, s_b) = sumcheck::verify_product_ef_with_challenges(
        b_eval, &proof.b_sumcheck, b_eq_at_s, b_at_s, transcript);
    if !b_ok { return false; }
    if compute_eq_at_point_ef(point, &s_b) != b_eq_at_s { return false; }

    // Check a_eval + b_eval = c_eval
    let c_eval = proof.c_eval.to_ef();
    if a_eval + b_eval != c_eval {
        return false;
    }

    // Verify c MLE eval proof
    let mut eval_transcript = Transcript::new(b"add-c-eval-ef");
    eval_transcript.absorb_bytes(&proof.c_commitment.root);
    if !verify_mle_eval_ef(
        &proof.c_commitment,
        c_eval,
        point,
        &proof.c_eval_proof,
        &mut eval_transcript,
    ) {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_field_vec(vals: &[u32]) -> Vec<F> {
        vals.iter().map(|&v| F::from_canonical_u32(v)).collect()
    }

    #[test]
    fn test_hadamard_basic() {
        // a = [2, 3, 5, 7], b = [3, 4, 2, 1], c = [6, 12, 10, 7]
        let a = make_field_vec(&[2, 3, 5, 7]);
        let b = make_field_vec(&[3, 4, 2, 1]);
        let c: Vec<F> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();

        let point = vec![F::from_canonical_u32(13), F::from_canonical_u32(37)];

        let mut pt = Transcript::new(b"hadamard-test");
        let proof = prove_hadamard(&a, &b, &c, &point, &mut pt);

        let mut vt = Transcript::new(b"hadamard-test");
        assert!(verify_hadamard(&proof, &point, &mut vt));
    }

    #[test]
    fn test_hadamard_wrong_c() {
        let a = make_field_vec(&[2, 3, 5, 7]);
        let b = make_field_vec(&[3, 4, 2, 1]);
        // Wrong c — not a ⊙ b
        let c = make_field_vec(&[6, 12, 10, 999]);

        let point = vec![F::from_canonical_u32(13), F::from_canonical_u32(37)];

        let mut pt = Transcript::new(b"hadamard-test");
        let proof = prove_hadamard(&a, &b, &c, &point, &mut pt);

        let mut vt = Transcript::new(b"hadamard-test");
        // Should fail because c̃(r) != Σ eq(r,x)·a[x]·b[x]
        assert!(!verify_hadamard(&proof, &point, &mut vt));
    }

    #[test]
    fn test_hadamard_larger() {
        let n = 16;
        let a: Vec<F> = (0..n).map(|i| F::from_canonical_u32(i + 1)).collect();
        let b: Vec<F> = (0..n).map(|i| F::from_canonical_u32(i * 2 + 3)).collect();
        let c: Vec<F> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();

        let point: Vec<F> = (0..4).map(|i| F::from_canonical_u32(i * 7 + 5)).collect();

        let mut pt = Transcript::new(b"hadamard-large");
        let proof = prove_hadamard(&a, &b, &c, &point, &mut pt);

        let mut vt = Transcript::new(b"hadamard-large");
        assert!(verify_hadamard(&proof, &point, &mut vt));
    }

    #[test]
    fn test_add_basic() {
        let a = make_field_vec(&[10, 20, 30, 40]);
        let b = make_field_vec(&[1, 2, 3, 4]);
        let c: Vec<F> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

        let point = vec![F::from_canonical_u32(5), F::from_canonical_u32(11)];

        let mut pt = Transcript::new(b"add-test");
        let proof = prove_add(&a, &b, &c, &point, &mut pt);

        let mut vt = Transcript::new(b"add-test");
        assert!(verify_add(&proof, &point, &mut vt));
    }

    #[test]
    fn test_add_wrong_c() {
        let a = make_field_vec(&[10, 20, 30, 40]);
        let b = make_field_vec(&[1, 2, 3, 4]);
        let c = make_field_vec(&[11, 22, 33, 999]); // wrong last element

        let point = vec![F::from_canonical_u32(5), F::from_canonical_u32(11)];

        let mut pt = Transcript::new(b"add-test");
        let proof = prove_add(&a, &b, &c, &point, &mut pt);

        let mut vt = Transcript::new(b"add-test");
        assert!(!verify_add(&proof, &point, &mut vt));
    }

    #[test]
    fn test_add_larger() {
        let n = 8;
        let a: Vec<F> = (0..n).map(|i| F::from_canonical_u32(i * 100)).collect();
        let b: Vec<F> = (0..n).map(|i| F::from_canonical_u32(i + 7)).collect();
        let c: Vec<F> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

        let point: Vec<F> = (0..3).map(|i| F::from_canonical_u32(i * 13 + 2)).collect();

        let mut pt = Transcript::new(b"add-large");
        let proof = prove_add(&a, &b, &c, &point, &mut pt);

        let mut vt = Transcript::new(b"add-large");
        assert!(verify_add(&proof, &point, &mut vt));
    }

    #[test]
    fn test_scalar_mul_basic() {
        let a = make_field_vec(&[3, 7, 11, 13]);
        let scalar = F::from_canonical_u32(5);
        let c: Vec<F> = a.iter().map(|&x| scalar * x).collect();

        let point = vec![F::from_canonical_u32(9), F::from_canonical_u32(23)];

        let mut pt = Transcript::new(b"smul-test");
        let proof = prove_scalar_mul(&a, scalar, &c, &point, &mut pt);

        let mut vt = Transcript::new(b"smul-test");
        let (valid, c_at_point) = verify_scalar_mul(&proof, &point, &mut vt);
        assert!(valid);

        // Verify c_at_point matches c̃(r)
        let expected = mle_evaluate(&c, &point);
        assert_eq!(c_at_point, expected);
    }

    #[test]
    fn test_scalar_mul_tampered_scalar_returns_wrong_value() {
        // The scalar is a public parameter — the sumcheck proves ã(r), not s.
        // Tampering with s in the proof doesn't break the sumcheck, but the
        // caller gets c̃(r) = s_fake * ã(r) ≠ s_real * ã(r). The caller must
        // independently verify the scalar matches the expected value.
        let a = make_field_vec(&[3, 7, 11, 13]);
        let scalar = F::from_canonical_u32(5);
        let c: Vec<F> = a.iter().map(|&x| scalar * x).collect();

        let point = vec![F::from_canonical_u32(9), F::from_canonical_u32(23)];

        let mut pt = Transcript::new(b"smul-test");
        let mut proof = prove_scalar_mul(&a, scalar, &c, &point, &mut pt);

        proof.scalar = 6; // tamper

        let mut vt = Transcript::new(b"smul-test");
        let (valid, c_at_point) = verify_scalar_mul(&proof, &point, &mut vt);
        assert!(valid, "Sumcheck passes — it proves a_eval, not the scalar");
        let expected = mle_evaluate(&c, &point);
        assert_ne!(c_at_point, expected, "But derived c̃(r) is wrong (6*a ≠ 5*a)");
    }

    #[test]
    fn test_scalar_mul_one() {
        // s=1 should be identity
        let a = make_field_vec(&[42, 100, 7, 3]);
        let scalar = F::one();
        let c = a.clone();

        let point = vec![F::from_canonical_u32(17), F::from_canonical_u32(31)];

        let mut pt = Transcript::new(b"smul-id");
        let proof = prove_scalar_mul(&a, scalar, &c, &point, &mut pt);

        let mut vt = Transcript::new(b"smul-id");
        let (valid, c_at_point) = verify_scalar_mul(&proof, &point, &mut vt);
        assert!(valid);
        assert_eq!(c_at_point, mle_evaluate(&a, &point));
    }

    #[test]
    fn test_scalar_mul_zero() {
        let a = make_field_vec(&[42, 100, 7, 3]);
        let scalar = F::zero();
        let c = vec![F::zero(); 4];

        let point = vec![F::from_canonical_u32(17), F::from_canonical_u32(31)];

        let mut pt = Transcript::new(b"smul-zero");
        let proof = prove_scalar_mul(&a, scalar, &c, &point, &mut pt);

        let mut vt = Transcript::new(b"smul-zero");
        let (valid, c_at_point) = verify_scalar_mul(&proof, &point, &mut vt);
        assert!(valid);
        assert_eq!(c_at_point, F::zero());
    }

    // ===== Extension-field (EF) tests =====

    fn make_ef_point(vals: &[u32]) -> Vec<EF> {
        vals.iter().map(|&v| f_to_ef(F::from_canonical_u32(v))).collect()
    }

    #[test]
    fn test_hadamard_ef_basic() {
        let a = make_field_vec(&[3, 7, 11, 13]);
        let b = make_field_vec(&[2, 4, 6, 8]);
        let c: Vec<F> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();

        let point = make_ef_point(&[5, 19]);

        let mut pt = Transcript::new(b"hadamard-ef-test");
        let proof = prove_hadamard_ef(&a, &b, &c, &point, &mut pt);

        let mut vt = Transcript::new(b"hadamard-ef-test");
        assert!(verify_hadamard_ef(&proof, &point, &mut vt), "Hadamard EF verification should pass");
    }

    #[test]
    fn test_add_ef_basic() {
        let a = make_field_vec(&[10, 20, 30, 40]);
        let b = make_field_vec(&[1, 2, 3, 4]);
        let c: Vec<F> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

        let point = make_ef_point(&[5, 11]);

        let mut pt = Transcript::new(b"add-ef-test");
        let proof = prove_add_ef(&a, &b, &c, &point, &mut pt);

        let mut vt = Transcript::new(b"add-ef-test");
        assert!(verify_add_ef(&proof, &point, &mut vt), "Add EF verification should pass");
    }
}
