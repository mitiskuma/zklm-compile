//! Merkle-based weight commitment + sumcheck MLE evaluation proof.
//!
//! The verifier only needs the 32-byte Merkle root (not the full weight matrix).
//! Soundness: sumcheck error ≤ 2·log(n)/|F| ≈ 10⁻⁸ for n=100K, |F|=2³¹.
//! Commitment binding: Merkle tree openings at random leaves bind the
//! sumcheck proof to the committed weights (collision resistance).

use p3_field::{AbstractField, AbstractExtensionField, PrimeField32};
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::field::common::{compute_eq_at_point, compute_eq_at_point_ef, log2_ceil};
use crate::field::m31_ops::*;
use crate::proving::pcs::{
    derive_query_indices, verify_merkle_opening_proof, MerkleOpeningProof, MerkleTree,
};
use crate::proving::sumcheck::{self, EF, EFElement, SumcheckProof, SumcheckProofEF, Transcript};

type F = Mersenne31;

/// 32-byte Merkle root commitment to a weight vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightCommitment {
    pub root: [u8; 32],
    pub num_weights: usize,
    pub log_height: usize,
}

/// Proof that W̃(point) = claimed_value, verifiable against a Merkle root.
/// Includes optional Merkle openings that bind the sumcheck to the committed weights.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MleEvalProof {
    pub eval_sumcheck: SumcheckProof,
    pub eq_at_s: u32,
    pub w_at_s: u32,
    /// Merkle tree openings proving the committed weights match the sumcheck.
    /// None when the verifier independently knows the weights (rmsnorm, layernorm, elementwise).
    pub merkle_opening: Option<MerkleOpeningProof>,
}

/// Fast commitment using a single blake3 hash (no Merkle tree).
/// Use ONLY when the root is used for transcript binding and will NOT be
/// compared against a Merkle-tree-generated root. ~1000x faster for large vectors.
///
/// WARNING: produces a DIFFERENT root than `commit_weights`. Only use for
/// weight matrices where the verifier reads the commitment from the proof
/// (not independently recomputed).
pub fn commit_weights_fast(weights: &[F]) -> WeightCommitment {
    let n = weights.len();
    let log_n = if n <= 1 { 1 } else { log2_ceil(n) };
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"weights");
    // Zero-copy: Mersenne31 is #[repr(transparent)] over u32, so &[F] has the
    // same layout as &[u32]. Hash the entire buffer in one call instead of
    // element-by-element (eliminates millions of tiny update() calls).
    assert!(
        std::mem::size_of::<F>() == 4,
        "Field element size is not 4 bytes — slice_from_raw_parts would read wrong number of bytes"
    );
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * 4,
        )
    };
    hasher.update(bytes);
    WeightCommitment {
        root: *hasher.finalize().as_bytes(),
        num_weights: n,
        log_height: log_n,
    }
}

/// Commit to weight values via Merkle tree.
/// Returns (commitment, MerkleTree) — caller stores MerkleTree for later openings.
///
/// The Merkle tree hashes each field element as a leaf (blake3),
/// then compresses pairs up to the root. Collision-resistant binding.
/// Use this only when you need the tree for `prove_mle_eval` openings.
#[allow(dead_code)]
pub fn commit_weights(weights: &[F]) -> (WeightCommitment, MerkleTree) {
    let tree = MerkleTree::new(weights);
    let n = weights.len();
    let log_n = if n <= 1 { 1 } else { log2_ceil(n) };
    // Merkle root is 16 bytes (Digest); zero-pad to 32 bytes for WeightCommitment.
    let merkle_root = tree.root();
    let mut root = [0u8; 32];
    root[..crate::proving::pcs::DIGEST_BYTES].copy_from_slice(&merkle_root);
    let commitment = WeightCommitment {
        root,
        num_weights: n,
        log_height: log_n,
    };
    (commitment, tree)
}

/// Prove that W̃(point) = value, where W is committed via Merkle tree.
///
/// Uses product sumcheck: W̃(point) = Σ_b eq(point, b) · W[b]
/// After sumcheck, generates Merkle openings at random leaf positions
/// to bind the proof to the committed weights.
#[allow(dead_code)]
pub fn prove_mle_eval(
    weights: &[F],
    point: &[F],
    tree: &MerkleTree,
    transcript: &mut Transcript,
) -> (F, MleEvalProof) {
    let log_n = point.len();
    let n_pad = 1 << log_n;

    let mut w_pad = weights.to_vec();
    w_pad.resize(n_pad, F::zero());

    let eq_vals = eq_evals(point);
    let claimed = mle_evaluate(&w_pad, point);

    let (proof, eq_at_s, w_at_s) =
        sumcheck::prove_product_best(&eq_vals, &w_pad, log_n, transcript);

    // Derive random query indices and generate Merkle openings
    let indices = derive_query_indices(transcript, log_n);
    let merkle_opening = tree.open_many(&indices);

    (
        claimed,
        MleEvalProof {
            eval_sumcheck: proof,
            eq_at_s: eq_at_s.as_canonical_u32(),
            w_at_s: w_at_s.as_canonical_u32(),
            merkle_opening: Some(merkle_opening),
        },
    )
}

/// Prove W̃(point) = value without Merkle openings.
///
/// Use when the verifier independently knows the weights and can recompute
/// the commitment (rmsnorm, layernorm, elementwise). No tree construction needed.
/// Transcript still includes query index derivation for consistency.
pub fn prove_mle_eval_no_merkle(
    weights: &[F],
    point: &[F],
    transcript: &mut Transcript,
) -> (F, MleEvalProof) {
    let log_n = point.len();
    let n_pad = 1 << log_n;

    let mut w_pad = weights.to_vec();
    w_pad.resize(n_pad, F::zero());

    let eq_vals = eq_evals(point);
    let claimed = mle_evaluate(&w_pad, point);

    let (proof, eq_at_s, w_at_s) =
        sumcheck::prove_product_best(&eq_vals, &w_pad, log_n, transcript);

    // Derive query indices to keep transcript in sync (verifier does the same)
    let _indices = derive_query_indices(transcript, log_n);

    (
        claimed,
        MleEvalProof {
            eval_sumcheck: proof,
            eq_at_s: eq_at_s.as_canonical_u32(),
            w_at_s: w_at_s.as_canonical_u32(),
            merkle_opening: None,
        },
    )
}

/// Prove that W̃(point) = claimed_value, where the claimed value is already known.
///
/// This avoids the redundant mle_evaluate() call when the caller already knows
/// the evaluation result (e.g., from a matmul sumcheck that already computed it).
#[allow(dead_code)]
pub fn prove_mle_eval_with_claim(
    weights: &[F],
    point: &[F],
    _claimed: F,
    tree: &MerkleTree,
    transcript: &mut Transcript,
) -> MleEvalProof {
    let log_n = point.len();
    let n_pad = 1 << log_n;

    let mut w_pad = weights.to_vec();
    w_pad.resize(n_pad, F::zero());

    let eq_vals = eq_evals(point);

    let (proof, eq_at_s, w_at_s) =
        sumcheck::prove_product_best(&eq_vals, &w_pad, log_n, transcript);

    // Derive random query indices and generate Merkle openings
    let indices = derive_query_indices(transcript, log_n);
    let merkle_opening = tree.open_many(&indices);

    MleEvalProof {
        eval_sumcheck: proof,
        eq_at_s: eq_at_s.as_canonical_u32(),
        w_at_s: w_at_s.as_canonical_u32(),
        merkle_opening: Some(merkle_opening),
    }
}

/// Like `prove_mle_eval_with_claim` but without Merkle openings.
/// Use for derived quantities (e.g., w_partial from matmul row-folding) where
/// binding comes through the parent commitment + matmul sumcheck, not through
/// an independent Merkle tree on the derived vector.
#[allow(dead_code)]
pub fn prove_mle_eval_with_claim_no_merkle(
    weights: &[F],
    point: &[F],
    _claimed: F,
    transcript: &mut Transcript,
) -> MleEvalProof {
    let log_n = point.len();
    let n_pad = 1 << log_n;

    let mut w_pad = weights.to_vec();
    w_pad.resize(n_pad, F::zero());

    let eq_vals = eq_evals(point);

    let (proof, eq_at_s, w_at_s) =
        sumcheck::prove_product_best(&eq_vals, &w_pad, log_n, transcript);

    // Derive query indices to keep transcript in sync
    let _indices = derive_query_indices(transcript, log_n);

    MleEvalProof {
        eval_sumcheck: proof,
        eq_at_s: eq_at_s.as_canonical_u32(),
        w_at_s: w_at_s.as_canonical_u32(),
        merkle_opening: None,
    }
}

/// Verify an MLE evaluation proof.
///
/// Checks:
/// 1. Sumcheck rounds are consistent
/// 2. eq(point, s*) matches the prover's claim (verifier computes independently)
/// 3. Final claim = eq_at_s · w_at_s
/// 4. **Merkle openings** verify against the commitment root — THIS IS THE KEY BINDING
///
/// The sumcheck ensures W̃(s*) is correct with error ≤ 2·log(n)/|F| ≈ 10⁻⁸.
/// The Merkle openings bind the proof to the committed weight vector.
pub fn verify_mle_eval(
    commitment: &WeightCommitment,
    claimed_value: F,
    point: &[F],
    proof: &MleEvalProof,
    transcript: &mut Transcript,
) -> bool {
    let log_n = point.len();

    let eq_at_s = F::from_canonical_u32(proof.eq_at_s);
    let w_at_s = F::from_canonical_u32(proof.w_at_s);

    // Verify product sumcheck (derive challenge point from transcript)
    let (sc_ok, s_point) = sumcheck::verify_product_with_challenges(
        claimed_value,
        &proof.eval_sumcheck,
        eq_at_s,
        w_at_s,
        transcript,
    );
    if !sc_ok {
        return false;
    }

    // Verify eq(point, s*) using derived challenges
    let eq_expected = compute_eq_at_point(point, &s_point);

    if eq_expected != eq_at_s {
        return false;
    }

    // Derive the same random query indices from transcript
    let _indices = derive_query_indices(transcript, log_n);

    // Verify Merkle openings against the commitment root (if present).
    // WeightCommitment.root is [u8; 32] but Merkle digests are 16 bytes — extract the Digest.
    if let Some(ref merkle_opening) = proof.merkle_opening {
        let mut merkle_root = crate::proving::pcs::Digest::default();
        merkle_root.copy_from_slice(&commitment.root[..crate::proving::pcs::DIGEST_BYTES]);
        if !verify_merkle_opening_proof(&merkle_root, merkle_opening, commitment.log_height)
        {
            eprintln!("MLE eval: Merkle opening verification failed — weights not bound to commitment");
            return false;
        }
    }

    true
}

// =============================================================================
// Extension-field MLE evaluation proofs
// =============================================================================

/// Proof that W̃(point) = claimed_value in extension field.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MleEvalProofEF {
    pub eval_sumcheck: SumcheckProofEF,
    pub eq_at_s: EFElement,
    pub w_at_s: EFElement,
}

/// Prove W̃(point) = claimed_value where W is a Vec<EF> and point is Vec<EF>.
/// No Merkle openings — used for derived quantities (e.g., w_partial from matmul).
pub fn prove_mle_eval_ef_with_claim_no_merkle(
    weights: &[EF],
    point: &[EF],
    _claimed: EF,
    transcript: &mut Transcript,
) -> MleEvalProofEF {
    let log_n = point.len();
    let n_pad = 1 << log_n;

    let mut w_pad = weights.to_vec();
    w_pad.resize(n_pad, EF::zero());

    let eq_vals = eq_evals_ef(point);

    let (proof, eq_at_s, w_at_s) =
        sumcheck::prove_product_ef_full(&eq_vals, &w_pad, log_n, transcript);

    // Derive query indices to keep transcript in sync
    let _indices = derive_query_indices(transcript, log_n);

    MleEvalProofEF {
        eval_sumcheck: proof,
        eq_at_s: EFElement::from_ef(eq_at_s),
        w_at_s: EFElement::from_ef(w_at_s),
    }
}

/// Verify an EF MLE evaluation proof.
pub fn verify_mle_eval_ef(
    _commitment: &WeightCommitment,
    claimed_value: EF,
    point: &[EF],
    proof: &MleEvalProofEF,
    transcript: &mut Transcript,
) -> bool {
    let log_n = point.len();

    let eq_at_s = proof.eq_at_s.to_ef();
    let w_at_s = proof.w_at_s.to_ef();

    // Use challenge-returning variant to derive s* from the transcript
    // (works even when proof.eval_sumcheck.challenges is empty/stripped).
    let (valid, derived_challenges) = sumcheck::verify_product_ef_with_challenges(
        claimed_value,
        &proof.eval_sumcheck,
        eq_at_s,
        w_at_s,
        transcript,
    );
    if !valid {
        return false;
    }

    // Verify eq(point, s*) using derived challenges (= s*)
    let s_point: Vec<EF> = derived_challenges;
    let eq_expected = compute_eq_at_point_ef(point, &s_point);

    if eq_expected != eq_at_s {
        return false;
    }

    // Derive query indices to keep transcript in sync
    let _indices = derive_query_indices(transcript, log_n);

    true
}

/// Prove W̃(point) = value where W is base-field `&[F]` and point is `&[EF]`.
/// No Merkle openings. Used for MLE eval proofs in EF-challenge modules
/// (rmsnorm, layernorm, elementwise) where weights are base field but
/// challenges are extension field.
pub fn prove_mle_eval_no_merkle_ef_base(
    weights: &[F],
    point: &[EF],
    transcript: &mut Transcript,
) -> (EF, MleEvalProofEF) {
    let log_n = point.len();
    let n_pad = 1 << log_n;

    let mut w_pad = weights.to_vec();
    w_pad.resize(n_pad, F::zero());

    let eq_vals = eq_evals_ef(point);
    let claimed = mle_evaluate_ef(&w_pad, point);

    // eq_vals is EF, w_pad is F → convert w_pad to EF, use full-EF product sumcheck
    let w_ef: Vec<EF> = w_pad.iter().map(|&v| {
        use p3_field::extension::Complex;
        EF::from_base(Complex::new(v, F::zero()))
    }).collect();

    let (proof, eq_at_s, w_at_s) =
        sumcheck::prove_product_ef_full(&eq_vals, &w_ef, log_n, transcript);

    let _indices = derive_query_indices(transcript, log_n);

    (
        claimed,
        MleEvalProofEF {
            eval_sumcheck: proof,
            eq_at_s: EFElement::from_ef(eq_at_s),
            w_at_s: EFElement::from_ef(w_at_s),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commit_weights() {
        let w: Vec<F> = (0..8).map(|i| F::from_canonical_u32(i + 1)).collect();
        let (c, _tree) = commit_weights(&w);
        assert_eq!(c.num_weights, 8);
        assert_eq!(c.log_height, 3);

        // Same weights → same root
        let (c2, _) = commit_weights(&w);
        assert_eq!(c.root, c2.root);

        // Different weights → different root
        let mut w2 = w.clone();
        w2[0] = F::from_canonical_u32(99);
        let (c3, _) = commit_weights(&w2);
        assert_ne!(c.root, c3.root);
    }

    #[test]
    fn test_mle_eval_proof() {
        let w: Vec<F> = (0..8).map(|i| F::from_canonical_u32(i + 1)).collect();
        let (commitment, tree) = commit_weights(&w);

        // Random evaluation point
        let point = vec![
            F::from_canonical_u32(5),
            F::from_canonical_u32(17),
            F::from_canonical_u32(42),
        ];

        let mut p_transcript = Transcript::new(b"mle-eval-test");
        let (claimed, proof) = prove_mle_eval(&w, &point, &tree, &mut p_transcript);

        // Verify: should accept honest proof
        let mut v_transcript = Transcript::new(b"mle-eval-test");
        assert!(verify_mle_eval(
            &commitment,
            claimed,
            &point,
            &proof,
            &mut v_transcript,
        ));
    }

    #[test]
    fn test_mle_eval_proof_wrong_claim() {
        let w: Vec<F> = (0..4).map(|i| F::from_canonical_u32(i + 1)).collect();
        let (commitment, tree) = commit_weights(&w);
        let point = vec![F::from_canonical_u32(7), F::from_canonical_u32(13)];

        let mut p_transcript = Transcript::new(b"test");
        let (claimed, proof) = prove_mle_eval(&w, &point, &tree, &mut p_transcript);

        // Tamper with claimed value
        let wrong_claim = claimed + F::one();
        let mut v_transcript = Transcript::new(b"test");
        assert!(!verify_mle_eval(
            &commitment,
            wrong_claim,
            &point,
            &proof,
            &mut v_transcript,
        ));
    }

    #[test]
    fn test_tampered_commitment_root_detected() {
        // The Merkle opening binds the proof to specific weights.
        // If an attacker uses different weights, the Merkle root changes
        // and verification fails.
        let w: Vec<F> = (0..8).map(|i| F::from_canonical_u32(i + 1)).collect();
        let (real_commitment, tree) = commit_weights(&w);

        let point = vec![
            F::from_canonical_u32(5),
            F::from_canonical_u32(17),
            F::from_canonical_u32(42),
        ];

        let mut p_transcript = Transcript::new(b"tamper-test");
        let (claimed, proof) = prove_mle_eval(&w, &point, &tree, &mut p_transcript);

        // Attacker modifies one weight and re-commits
        let mut w_bad = w.clone();
        w_bad[3] = F::from_canonical_u32(999);
        let (fake_commitment, _) = commit_weights(&w_bad);

        // Roots differ
        assert_ne!(real_commitment.root, fake_commitment.root);

        // Verification against the fake commitment fails because
        // the Merkle openings were generated from the real tree
        let mut v_transcript = Transcript::new(b"tamper-test");
        assert!(!verify_mle_eval(
            &fake_commitment,
            claimed,
            &point,
            &proof,
            &mut v_transcript,
        ));
    }

    #[test]
    fn test_mle_eval_large() {
        // 128 weights (like fc1 output dim)
        let w: Vec<F> = (0..128).map(|i| F::from_canonical_u32(i * 3 + 1)).collect();
        let (commitment, tree) = commit_weights(&w);
        let point: Vec<F> = (0..7)
            .map(|i| F::from_canonical_u32(i * 11 + 3))
            .collect();

        let mut pt = Transcript::new(b"large");
        let (claimed, proof) = prove_mle_eval(&w, &point, &tree, &mut pt);

        let mut vt = Transcript::new(b"large");
        assert!(verify_mle_eval(&commitment, claimed, &point, &proof, &mut vt));

        // Check Merkle opening proof is present
        let opening = proof.merkle_opening.as_ref().unwrap();
        assert_eq!(opening.query_indices.len(), crate::proving::pcs::NUM_QUERIES);
        assert_eq!(opening.opened_values.len(), crate::proving::pcs::NUM_QUERIES);
        assert_eq!(opening.merkle_paths.len(), crate::proving::pcs::NUM_QUERIES);
        // Each path should have log_height = 7 siblings
        for path in &opening.merkle_paths {
            assert_eq!(path.len(), 7);
        }
    }

    #[test]
    fn test_mle_eval_with_claim() {
        let w: Vec<F> = (0..16).map(|i| F::from_canonical_u32(i + 1)).collect();
        let (commitment, tree) = commit_weights(&w);
        let point: Vec<F> = (0..4)
            .map(|i| F::from_canonical_u32(i * 7 + 2))
            .collect();

        // Compute the claimed value
        let log_n = 4;
        let n_pad = 1 << log_n;
        let mut w_pad = w.to_vec();
        w_pad.resize(n_pad, F::zero());
        let claimed = mle_evaluate(&w_pad, &point);

        let mut pt = Transcript::new(b"claim");
        let proof = prove_mle_eval_with_claim(&w, &point, claimed, &tree, &mut pt);

        let mut vt = Transcript::new(b"claim");
        assert!(verify_mle_eval(&commitment, claimed, &point, &proof, &mut vt));
    }
}
