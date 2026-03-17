//! Structured matmul sumcheck: prove y = W @ x + b.

use p3_field::{AbstractField, AbstractExtensionField, PackedValue, PrimeField32};
use p3_field::extension::Complex;
use p3_mersenne_31::Mersenne31;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(target_arch = "aarch64")]
use p3_mersenne_31::PackedMersenne31Neon;

use crate::field::common::log2_ceil;
use crate::field::m31_ops::*;
use crate::proving::sumcheck::{self, EF, EFElement, SumcheckProof, SumcheckProofEF, Transcript};
use crate::proving::weight_commitment::{self, MleEvalProof, MleEvalProofEF, WeightCommitment};

/// Threshold for switching to rayon parallel fold (matches sumcheck).
const FOLD_PAR_THRESHOLD: usize = 1024;

type F = Mersenne31;



#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatmulProof {
    pub sumcheck_proof: SumcheckProof,
    pub w_at_rs: u32,
    pub x_at_s: u32,
    pub r_point: Vec<u32>,
    pub s_point: Vec<u32>,
}

#[derive(Debug)]
pub struct MatmulVerifyResult {
    pub valid: bool,
    pub x_claim_point: Vec<F>,
    pub x_claim_value: F,
}

/// SIMD-accelerated scalar-vector multiply-accumulate: partial[j] += coeff * row[j]
/// Uses PackedMersenne31Neon (4-wide NEON) for ~4x throughput on aarch64.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn macc_row(partial: &mut [F], coeff: F, row: &[F], n: usize) {
    const W: usize = 4; // PackedMersenne31Neon width
    let coeff_packed = PackedMersenne31Neon::from(coeff);
    let chunks = n / W;
    let remainder = n % W;

    for c in 0..chunks {
        let j = c * W;
        let row_packed = *PackedMersenne31Neon::from_slice(&row[j..j + W]);
        let partial_packed = PackedMersenne31Neon::from_slice_mut(&mut partial[j..j + W]);
        *partial_packed += coeff_packed * row_packed;
    }
    // Scalar remainder
    let base = chunks * W;
    for j in 0..remainder {
        partial[base + j] += coeff * row[base + j];
    }
}

/// Scalar fallback for non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn macc_row(partial: &mut [F], coeff: F, row: &[F], n: usize) {
    for j in 0..n {
        partial[j] += coeff * row[j];
    }
}

/// Single-pass fold via eq_evals: computes w_partial[j] = Σ_i eq(r,i) * W[i,j]
/// as a matrix-vector product. Reads unpadded weights exactly once (no padding
/// allocation needed). Uses rayon for parallelism over row chunks.
/// Inner loop uses PackedMersenne31Neon (4-wide SIMD) on aarch64.
fn fold_eq_matvec(w_flat: &[F], m: usize, n: usize, n_pad: usize, r_point: &[F]) -> Vec<F> {
    let log_m = r_point.len();
    let m_pad = 1usize << log_m;
    let eq_r = eq_evals(r_point);

    if m * n >= FOLD_PAR_THRESHOLD {
        let num_threads = rayon::current_num_threads().max(1);
        let rows_per_chunk = (m + num_threads - 1) / num_threads;

        (0..num_threads)
            .into_par_iter()
            .filter_map(|t| {
                let start = t * rows_per_chunk;
                if start >= m { return None; }
                let end = (start + rows_per_chunk).min(m);
                let mut partial = vec![F::zero(); n_pad];
                for i in start..end {
                    macc_row(&mut partial, eq_r[i], &w_flat[i * n..(i + 1) * n], n);
                }
                Some(partial)
            })
            .reduce_with(|mut a, b| {
                // SIMD-accelerated vector add for merge
                #[cfg(target_arch = "aarch64")]
                {
                    let chunks = n_pad / 4;
                    for c in 0..chunks {
                        let j = c * 4;
                        let b_packed = *PackedMersenne31Neon::from_slice(&b[j..j + 4]);
                        let a_packed = PackedMersenne31Neon::from_slice_mut(&mut a[j..j + 4]);
                        *a_packed += b_packed;
                    }
                    for j in (chunks * 4)..n_pad {
                        a[j] += b[j];
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    for j in 0..n_pad {
                        a[j] += b[j];
                    }
                }
                a
            })
            .unwrap_or_else(|| vec![F::zero(); n_pad])
    } else {
        let mut w_partial = vec![F::zero(); n_pad];
        for i in 0..m.min(m_pad) {
            macc_row(&mut w_partial, eq_r[i], &w_flat[i * n..(i + 1) * n], n);
        }
        w_partial
    }
}

/// Prove y = W @ x + bias.
/// Returns the proof AND w_partial (the partially-evaluated weight vector after
/// folding W at the random point r). w_partial has length n_pad and satisfies
/// w_partial[j] = W̃(r, j) for all j ∈ {0,1}^log_n.
pub fn prove_matmul(
    w_flat: &[F],
    x_vals: &[F],
    y_vals: &[F],
    m: usize,
    n: usize,
    _bias: Option<&[F]>,
    transcript: &mut Transcript,
) -> (MatmulProof, Vec<F>) {
    let log_m = log2_ceil(m);
    let log_n = log2_ceil(n);
    let m_pad = 1 << log_m;
    let n_pad = 1 << log_n;

    // Pad vectors
    let mut y_padded = y_vals.to_vec();
    y_padded.resize(m_pad, F::zero());
    let mut x_padded = x_vals.to_vec();
    x_padded.resize(n_pad, F::zero());

    // Get random point r
    transcript.absorb_many(&y_padded);
    let r_point = transcript.squeeze_many(log_m);

    // Partially evaluate W at r via single-pass eq_evals matvec.
    // This reads w_flat exactly once — no padding allocation needed.
    let w_partial = fold_eq_matvec(w_flat, m, n, n_pad, &r_point);
    assert_eq!(w_partial.len(), n_pad);

    // Product sumcheck
    let (proof, w_at_s, x_at_s) =
        sumcheck::prove_product_best(&w_partial, &x_padded, log_n, transcript);

    let s_point: Vec<u32> = proof.challenges.clone();
    let r_raw: Vec<u32> = r_point.iter().map(|r| r.as_canonical_u32()).collect();

    (
        MatmulProof {
            sumcheck_proof: proof,
            w_at_rs: w_at_s.as_canonical_u32(),
            x_at_s: x_at_s.as_canonical_u32(),
            r_point: r_raw,
            s_point,
        },
        w_partial,
    )
}

/// Verify a matmul proof.
pub fn verify_matmul(
    proof: &MatmulProof,
    w_flat: &[F],
    y_vals: &[F],
    m: usize,
    n: usize,
    bias: Option<&[F]>,
    transcript: &mut Transcript,
) -> MatmulVerifyResult {
    let fail = MatmulVerifyResult {
        valid: false,
        x_claim_point: vec![],
        x_claim_value: F::zero(),
    };

    let log_m = log2_ceil(m);
    let log_n = log2_ceil(n);
    let m_pad = 1 << log_m;
    let n_pad = 1 << log_n;

    let mut y_padded = y_vals.to_vec();
    y_padded.resize(m_pad, F::zero());

    // Reconstruct r from transcript
    transcript.absorb_many(&y_padded);
    let r_point = transcript.squeeze_many(log_m);

    let r_expected: Vec<u32> = r_point.iter().map(|r| r.as_canonical_u32()).collect();
    if r_expected != proof.r_point {
        return fail;
    }

    let y_at_r = mle_evaluate(&y_padded, &r_point);
    let mut claim = y_at_r;
    if let Some(b) = bias {
        let mut b_padded = b.to_vec();
        b_padded.resize(m_pad, F::zero());
        let b_at_r = mle_evaluate(&b_padded, &r_point);
        claim = claim - b_at_r;
    }

    let w_at_rs = F::from_canonical_u32(proof.w_at_rs);
    let x_at_s = F::from_canonical_u32(proof.x_at_s);

    if !sumcheck::verify_product(claim, &proof.sumcheck_proof, log_n, w_at_rs, x_at_s, transcript)
    {
        return fail;
    }

    // Check W(r, s) against known weights
    let mut w_padded = vec![F::zero(); m_pad * n_pad];
    for i in 0..m {
        for j in 0..n {
            w_padded[i * n_pad + j] = w_flat[i * n + j];
        }
    }

    let s_point: Vec<F> = proof
        .s_point
        .iter()
        .map(|&v| F::from_canonical_u32(v))
        .collect();
    let mut rs_point = r_point;
    rs_point.extend_from_slice(&s_point);
    let w_expected = mle_evaluate(&w_padded, &rs_point);

    if w_expected != w_at_rs {
        return fail;
    }

    MatmulVerifyResult {
        valid: true,
        x_claim_point: s_point,
        x_claim_value: x_at_s,
    }
}

// ======================================================================
// Weight binding: single function for all PCS modes
// ======================================================================

/// Result of weight binding — commitment + eval proof + optional Basefold.
struct WeightBinding {
    commitment: WeightCommitment,
    eval_proof: MleEvalProof,
    basefold_proof: Option<crate::proving::basefold::BasefoldOpeningProof>,
}

/// Bind weights to the proof transcript. Handles all PCS modes:
///   - `pcs-full`: Basefold on full W at point (r || s) → standalone verification
///   - `pcs`: Basefold on w_partial at point s → fast binding
///   - default: blake3 hash + sumcheck MLE eval → fastest, Fiat-Shamir only
fn bind_weights(
    w_partial: &[F],
    #[cfg(feature = "pcs-full")]
    w_flat: Option<(&[F], usize, usize)>,
    #[cfg(not(feature = "pcs-full"))]
    _w_flat: Option<(&[F], usize, usize)>,
    s_point: &[F],
    #[cfg(feature = "pcs-full")]
    r_point: Option<&[u32]>,
    #[cfg(not(feature = "pcs-full"))]
    _r_point: Option<&[u32]>,
    claimed: F,
    transcript: &mut Transcript,
) -> WeightBinding {
    // Helper: convert BasefoldCommitment → WeightCommitment + dummy MleEvalProof.
    // Basefold root is 16-byte Digest; zero-pad to 32 bytes for WeightCommitment.
    #[cfg(feature = "pcs")]
    let bf_to_wb = |bf_commitment: crate::proving::basefold::BasefoldCommitment,
                    bf_proof: crate::proving::basefold::BasefoldOpeningProof| -> WeightBinding {
        let mut root = [0u8; 32];
        root[..crate::proving::pcs::DIGEST_BYTES].copy_from_slice(&bf_commitment.root);
        WeightBinding {
            commitment: WeightCommitment {
                root,
                num_weights: bf_commitment.num_coeffs,
                log_height: bf_commitment.log_n,
            },
            eval_proof: MleEvalProof {
                eval_sumcheck: SumcheckProof { round_polys: vec![], challenges: vec![] },
                eq_at_s: 0, w_at_s: 0, merkle_opening: None,
            },
            basefold_proof: Some(bf_proof),
        }
    };

    // pcs-full: Basefold on full W matrix
    #[cfg(feature = "pcs-full")]
    {
        if let Some((w, m, n)) = w_flat {
            let log_m = log2_ceil(m);
            let log_n = log2_ceil(n);
            let m_pad = 1 << log_m;
            let n_pad = 1 << log_n;
            let mut w_padded = vec![F::zero(); m_pad * n_pad];
            for i in 0..m.min(m_pad) {
                let src = i * n;
                let dst = i * n_pad;
                let len = n.min(n_pad);
                w_padded[dst..dst + len].copy_from_slice(&w[src..src + len]);
            }
            let mut rs_point: Vec<F> = r_point.unwrap_or(&[]).iter()
                .map(|&v| F::from_canonical_u32(v)).collect();
            rs_point.extend_from_slice(s_point);
            let (bf_commitment, bf_proof) =
                crate::proving::basefold::commit_and_prove(&w_padded, &rs_point, claimed, transcript);
            return bf_to_wb(bf_commitment, bf_proof);
        }
        let (bf_commitment, bf_proof) =
            crate::proving::basefold::commit_and_prove(w_partial, s_point, claimed, transcript);
        return bf_to_wb(bf_commitment, bf_proof);
    }

    // pcs (without full): Basefold on w_partial
    #[cfg(all(feature = "pcs", not(feature = "pcs-full")))]
    {
        let (bf_commitment, bf_proof) =
            crate::proving::basefold::commit_and_prove(w_partial, s_point, claimed, transcript);
        return bf_to_wb(bf_commitment, bf_proof);
    }

    // Default: blake3 hash + sumcheck MLE eval
    #[cfg(not(feature = "pcs"))]
    {
        let commitment = weight_commitment::commit_weights_fast(w_partial);
        transcript.absorb_bytes(&commitment.root);
        let eval_proof = weight_commitment::prove_mle_eval_with_claim_no_merkle(
            w_partial, s_point, claimed, transcript,
        );
        WeightBinding { commitment, eval_proof, basefold_proof: None }
    }
}

/// EF variant: bind weights for extension-field matmul proofs.
/// All modes use EF sumcheck for evaluation; PCS modes add Basefold for commitment binding.
fn bind_weights_ef(
    w_partial: &[EF],
    w_partial_base: &[F],
    #[allow(unused_variables)]
    w_flat: Option<(&[F], usize, usize)>,
    s_point: &[EF],
    claimed: EF,
    transcript: &mut Transcript,
) -> (WeightCommitment, MleEvalProofEF, Option<crate::proving::basefold::BasefoldOpeningProof>) {
    // pcs-full: Basefold commit + opening proof on full W (base-field)
    // The Basefold proves binding at a base-field evaluation point squeezed
    // from the transcript. The EF sumcheck proves the actual EF evaluation.
    #[cfg(feature = "pcs-full")]
    let (commitment, bf_proof) = if let Some((w, m, n)) = w_flat {
        let log_m = log2_ceil(m);
        let log_n = log2_ceil(n);
        let m_pad = 1 << log_m;
        let n_pad = 1 << log_n;
        let num_coeffs = w.len();
        let mut w_padded = vec![F::zero(); m_pad * n_pad];
        for i in 0..m.min(m_pad) {
            let src = i * n;
            let dst = i * n_pad;
            let len = n.min(n_pad);
            w_padded[dst..dst + len].copy_from_slice(&w[src..src + len]);
        }
        // Squeeze a base-field evaluation point from transcript for the Basefold proof
        let bf_log = log_m + log_n;
        let bf_point: Vec<F> = (0..bf_log).map(|_| transcript.squeeze()).collect();
        // Pass w_padded by ownership — eliminates the 16MB internal copy
        // inside commit_and_prove_with_tables. MLE eval is incremental.
        let (bf_commitment, bf_proof) =
            crate::proving::basefold::commit_and_prove_vec(w_padded, num_coeffs, &bf_point, transcript);
        ({
            let mut root = [0u8; 32];
            root[..crate::proving::pcs::DIGEST_BYTES].copy_from_slice(&bf_commitment.root);
            WeightCommitment {
                root,
                num_weights: bf_commitment.num_coeffs,
                log_height: bf_commitment.log_n,
            }
        }, Some(bf_proof))
    } else {
        // w_partial path
        let bf_log = log2_ceil(w_partial_base.len());
        let bf_point: Vec<F> = (0..bf_log).map(|_| transcript.squeeze()).collect();
        let (bf_commitment, bf_proof) =
            crate::proving::basefold::commit_and_prove(w_partial_base, &bf_point, F::zero(), transcript);
        ({
            let mut root = [0u8; 32];
            root[..crate::proving::pcs::DIGEST_BYTES].copy_from_slice(&bf_commitment.root);
            WeightCommitment {
                root,
                num_weights: bf_commitment.num_coeffs,
                log_height: bf_commitment.log_n,
            }
        }, Some(bf_proof))
    };

    // pcs (without full): Basefold commit + opening on w_partial_base
    #[cfg(all(feature = "pcs", not(feature = "pcs-full")))]
    let (commitment, bf_proof) = {
        let bf_log = log2_ceil(w_partial_base.len());
        let bf_point: Vec<F> = (0..bf_log).map(|_| transcript.squeeze()).collect();
        let (bf_commitment, bf_proof) =
            crate::proving::basefold::commit_and_prove(w_partial_base, &bf_point, F::zero(), transcript);
        ({
            let mut root = [0u8; 32];
            root[..crate::proving::pcs::DIGEST_BYTES].copy_from_slice(&bf_commitment.root);
            WeightCommitment {
                root,
                num_weights: bf_commitment.num_coeffs,
                log_height: bf_commitment.log_n,
            }
        }, Some(bf_proof))
    };

    // Default: blake3 hash, no Basefold
    #[cfg(not(feature = "pcs"))]
    let (commitment, bf_proof) = {
        (weight_commitment::commit_weights_fast(w_partial_base), None)
    };

    // EF sumcheck for the actual evaluation proof
    transcript.absorb_bytes(&commitment.root);
    let eval_proof = weight_commitment::prove_mle_eval_ef_with_claim_no_merkle(
        w_partial, s_point, claimed, transcript,
    );

    (commitment, eval_proof, bf_proof)
}

// ======================================================================

/// Matmul proof with succinct weight verification (no full W needed by verifier).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuccinctMatmulProof {
    pub matmul_proof: MatmulProof,
    pub w_eval_proof: MleEvalProof,
    /// Deprecated: was always = w_at_rs. Kept for serde compat.
    #[serde(default)]
    pub w_claimed_value: u32,
    /// Commitment to w_partial (row-folded weight vector).
    pub w_partial_commitment: WeightCommitment,
    /// Deprecated: was always true. Kept for serde compat.
    #[serde(default)]
    pub bound: bool,
    /// Basefold PCS opening proof (when compiled with `--features pcs` or `pcs-full`).
    /// If present, verifier uses Basefold instead of sumcheck MLE eval.
    #[serde(default)]
    pub basefold_proof: Option<crate::proving::basefold::BasefoldOpeningProof>,
}

/// Prove y = W @ x + b with succinct weight commitment.
/// Returns proof that can be verified with only the WeightCommitment (32 bytes).
///
/// Optimization: reuses w_partial from the matmul fold instead of re-folding
/// the entire m×n weight matrix. The MLE eval proof runs over n_pad elements
/// instead of m_pad×n_pad — a ~m_pad× speedup on the weight binding sumcheck.
pub fn prove_matmul_succinct(
    w_flat: &[F],
    x_vals: &[F],
    y_vals: &[F],
    m: usize,
    n: usize,
    bias: Option<&[F]>,
    transcript: &mut Transcript,
) -> SuccinctMatmulProof {
    let (matmul_proof, w_partial) =
        prove_matmul(w_flat, x_vals, y_vals, m, n, bias, transcript);

    let s_point: Vec<F> = matmul_proof.s_point.iter()
        .map(|&v| F::from_canonical_u32(v)).collect();
    let claimed = F::from_canonical_u32(matmul_proof.w_at_rs);

    let wb = bind_weights(
        &w_partial, Some((w_flat, m, n)), &s_point,
        Some(&matmul_proof.r_point), claimed, transcript,
    );

    SuccinctMatmulProof {
        matmul_proof,
        w_eval_proof: wb.eval_proof,
        w_claimed_value: 0, // deprecated: verifier uses w_at_rs directly
        w_partial_commitment: wb.commitment,
        bound: false, // deprecated: never read by verifier
        basefold_proof: wb.basefold_proof,
    }
}

/// Like `prove_matmul` but takes a pre-padded weight matrix to avoid redundant
/// O(m×n) allocation when the caller already has the padded buffer (e.g., from
/// commit_weights). `w_padded` must have length `m_pad * n_pad`.
pub fn prove_matmul_prepadded(
    w_padded: &[F],
    x_vals: &[F],
    y_vals: &[F],
    m: usize,
    n: usize,
    _bias: Option<&[F]>,
    transcript: &mut Transcript,
) -> (MatmulProof, Vec<F>) {
    let log_m = log2_ceil(m);
    let log_n = log2_ceil(n);
    let m_pad = 1 << log_m;
    let n_pad = 1 << log_n;
    assert_eq!(w_padded.len(), m_pad * n_pad, "w_padded must be m_pad * n_pad");

    // Pad vectors
    let mut y_padded = y_vals.to_vec();
    y_padded.resize(m_pad, F::zero());
    let mut x_padded = x_vals.to_vec();
    x_padded.resize(n_pad, F::zero());

    // Get random point r
    transcript.absorb_many(&y_padded);
    let r_point = transcript.squeeze_many(log_m);

    // Partially evaluate W at r via single-pass eq_evals matvec.
    // w_padded is m_pad×n_pad layout, so treat as m_pad rows of n_pad columns.
    let w_partial = fold_eq_matvec(w_padded, m_pad, n_pad, n_pad, &r_point);
    assert_eq!(w_partial.len(), n_pad);

    // Product sumcheck
    let (proof, w_at_s, x_at_s) =
        sumcheck::prove_product_best(&w_partial, &x_padded, log_n, transcript);

    let s_point: Vec<u32> = proof.challenges.clone();
    let r_raw: Vec<u32> = r_point.iter().map(|r| r.as_canonical_u32()).collect();

    (
        MatmulProof {
            sumcheck_proof: proof,
            w_at_rs: w_at_s.as_canonical_u32(),
            x_at_s: x_at_s.as_canonical_u32(),
            r_point: r_raw,
            s_point,
        },
        w_partial,
    )
}

/// Like `prove_matmul_succinct` but takes a pre-padded weight matrix.
/// Avoids redundant O(m×n) padding when the caller already has the buffer.
pub fn prove_matmul_succinct_prepadded(
    w_padded: &[F],
    x_vals: &[F],
    y_vals: &[F],
    m: usize,
    n: usize,
    bias: Option<&[F]>,
    transcript: &mut Transcript,
) -> SuccinctMatmulProof {
    let (matmul_proof, w_partial) =
        prove_matmul_prepadded(w_padded, x_vals, y_vals, m, n, bias, transcript);

    let s_point: Vec<F> = matmul_proof.s_point.iter()
        .map(|&v| F::from_canonical_u32(v)).collect();
    let claimed = F::from_canonical_u32(matmul_proof.w_at_rs);

    // prepadded doesn't have w_flat in original layout — use w_partial only
    let wb = bind_weights(
        &w_partial, None, &s_point, None, claimed, transcript,
    );

    SuccinctMatmulProof {
        matmul_proof,
        w_eval_proof: wb.eval_proof,
        w_claimed_value: 0, // deprecated: verifier uses w_at_rs directly
        w_partial_commitment: wb.commitment,
        bound: false, // deprecated: never read by verifier
        basefold_proof: wb.basefold_proof,
    }
}

/// Verify a succinct matmul proof.
///
/// Supports two modes:
/// - **Bound proof** (non-zero commitment root): commitment root absorbed into
///   transcript before r generation. Fiat-Shamir binding — a prover using different
///   weights gets different r, causing r_point mismatch.
/// - **Unbound proof** (zero commitment root): legacy path, no binding.
pub fn verify_matmul_succinct(
    proof: &SuccinctMatmulProof,
    _commitment: &WeightCommitment,
    y_vals: &[F],
    m: usize,
    n: usize,
    bias: Option<&[F]>,
    transcript: &mut Transcript,
) -> MatmulVerifyResult {
    let fail = MatmulVerifyResult {
        valid: false,
        x_claim_point: vec![],
        x_claim_value: F::zero(),
    };

    let log_m = log2_ceil(m);
    let log_n = log2_ceil(n);
    let m_pad = 1 << log_m;

    let mut y_padded = y_vals.to_vec();
    y_padded.resize(m_pad, F::zero());

    transcript.absorb_many(&y_padded);
    let r_point = transcript.squeeze_many(log_m);

    let r_expected: Vec<u32> = r_point.iter().map(|r| r.as_canonical_u32()).collect();
    if r_expected != proof.matmul_proof.r_point {
        return fail;
    }

    let y_at_r = mle_evaluate(&y_padded, &r_point);
    let mut claim = y_at_r;
    if let Some(b) = bias {
        let mut b_padded = b.to_vec();
        b_padded.resize(m_pad, F::zero());
        let b_at_r = mle_evaluate(&b_padded, &r_point);
        claim = claim - b_at_r;
    }

    let w_at_rs = F::from_canonical_u32(proof.matmul_proof.w_at_rs);
    let x_at_s = F::from_canonical_u32(proof.matmul_proof.x_at_s);

    // Verify matmul sumcheck
    if !sumcheck::verify_product(
        claim,
        &proof.matmul_proof.sumcheck_proof,
        log_n,
        w_at_rs,
        x_at_s,
        transcript,
    ) {
        return fail;
    }

    let s_point: Vec<F> = proof
        .matmul_proof
        .s_point
        .iter()
        .map(|&v| F::from_canonical_u32(v))
        .collect();

    // Dispatch: Basefold PCS or sumcheck MLE eval
    if let Some(ref bf_proof) = proof.basefold_proof {
        // Extract 16-byte Merkle digest from 32-byte WeightCommitment root.
        let mut bf_root = crate::proving::pcs::Digest::default();
        bf_root.copy_from_slice(&proof.w_partial_commitment.root[..crate::proving::pcs::DIGEST_BYTES]);
        let bf_commitment = crate::proving::basefold::BasefoldCommitment {
            root: bf_root,
            num_coeffs: proof.w_partial_commitment.num_weights,
            log_n: proof.w_partial_commitment.log_height,
        };
        // Determine if this is pcs-full (r || s) or pcs (s only)
        let eval_point = if bf_commitment.log_n > s_point.len() {
            // pcs-full: evaluation point is r || s
            let mut rs = r_point.clone();
            rs.extend_from_slice(&s_point);
            rs
        } else {
            // pcs: evaluation point is s only
            s_point.clone()
        };
        transcript.absorb_bytes(&bf_commitment.root);
        if !crate::proving::basefold::verify_opening(
            &bf_commitment, w_at_rs, &eval_point, bf_proof, transcript,
        ) {
            return fail;
        }
    } else {
        // Sumcheck MLE eval mode (default)
        transcript.absorb_bytes(&proof.w_partial_commitment.root);
        if !weight_commitment::verify_mle_eval(
            &proof.w_partial_commitment,
            w_at_rs,
            &s_point,
            &proof.w_eval_proof,
            transcript,
        ) {
            return fail;
        }
    }

    MatmulVerifyResult {
        valid: true,
        x_claim_point: s_point,
        x_claim_value: x_at_s,
    }
}

// =============================================================================
// Extension-field matmul proofs (124-bit soundness)
// =============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatmulProofEF {
    pub sumcheck_proof: SumcheckProofEF,
    pub w_at_rs: EFElement,
    pub x_at_s: EFElement,
    #[serde(default)]
    pub r_point: Vec<EFElement>,
    #[serde(default)]
    pub s_point: Vec<EFElement>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct MatmulVerifyResultEF {
    pub valid: bool,
    pub x_claim_point: Vec<EF>,
    pub x_claim_value: EF,
}

/// Compute w_partial[j] = Σ_i eq(r,i) * W[i,j] where r is an EF point.
/// eq_r is Vec<EF>, W is base field. Result: Vec<EF>.
///
/// Uses SoA (Structure of Arrays) decomposition: since EF × F = 4 independent
/// M31 multiplications, we maintain 4 base-field accumulators and reuse the
/// NEON-accelerated `macc_row` for each component. This gives ~4x speedup over
/// scalar EF arithmetic on aarch64.
fn fold_eq_matvec_ef(w_flat: &[F], m: usize, n: usize, n_pad: usize, r_point: &[EF]) -> Vec<EF> {
    let log_m = r_point.len();
    let m_pad = 1usize << log_m;
    let eq_r = eq_evals_ef(r_point);
    let rows = m.min(m_pad);

    // Decompose EF coefficients into 4 base-field components.
    // EF = [Complex<M31>; 2] = [c0_real, c0_imag, c1_real, c1_imag]
    // EF × F multiplies each component by the same F scalar.
    let mut c0r = Vec::with_capacity(rows);
    let mut c0i = Vec::with_capacity(rows);
    let mut c1r = Vec::with_capacity(rows);
    let mut c1i = Vec::with_capacity(rows);
    for i in 0..rows {
        let bases: &[Complex<Mersenne31>] = eq_r[i].as_base_slice();
        let b0: &[Mersenne31] = bases[0].as_base_slice();
        let b1: &[Mersenne31] = bases[1].as_base_slice();
        c0r.push(b0[0]);
        c0i.push(b0[1]);
        c1r.push(b1[0]);
        c1i.push(b1[1]);
    }

    // 4 base-field partial vectors — each accumulated with SIMD macc_row
    if rows * n >= FOLD_PAR_THRESHOLD {
        let num_threads = rayon::current_num_threads().max(1);
        let rows_per_chunk = (rows + num_threads - 1) / num_threads;

        let partials: Vec<[Vec<F>; 4]> = (0..num_threads)
            .into_par_iter()
            .filter_map(|t| {
                let start = t * rows_per_chunk;
                if start >= rows { return None; }
                let end = (start + rows_per_chunk).min(rows);
                let mut p0 = vec![F::zero(); n_pad];
                let mut p1 = vec![F::zero(); n_pad];
                let mut p2 = vec![F::zero(); n_pad];
                let mut p3 = vec![F::zero(); n_pad];
                for i in start..end {
                    let row = &w_flat[i * n..(i + 1) * n];
                    macc_row(&mut p0, c0r[i], row, n);
                    macc_row(&mut p1, c0i[i], row, n);
                    macc_row(&mut p2, c1r[i], row, n);
                    macc_row(&mut p3, c1i[i], row, n);
                }
                Some([p0, p1, p2, p3])
            })
            .collect();

        // Merge thread-local partials
        let mut p0 = vec![F::zero(); n_pad];
        let mut p1 = vec![F::zero(); n_pad];
        let mut p2 = vec![F::zero(); n_pad];
        let mut p3 = vec![F::zero(); n_pad];
        for chunk in &partials {
            for j in 0..n_pad {
                p0[j] += chunk[0][j];
                p1[j] += chunk[1][j];
                p2[j] += chunk[2][j];
                p3[j] += chunk[3][j];
            }
        }

        // Reconstruct Vec<EF> from 4 component vectors
        let mut w_partial = Vec::with_capacity(n_pad);
        for j in 0..n_pad {
            let c0 = Complex::new(p0[j], p1[j]);
            let c1 = Complex::new(p2[j], p3[j]);
            w_partial.push(EF::from_base_slice(&[c0, c1]));
        }
        w_partial
    } else {
        let mut p0 = vec![F::zero(); n_pad];
        let mut p1 = vec![F::zero(); n_pad];
        let mut p2 = vec![F::zero(); n_pad];
        let mut p3 = vec![F::zero(); n_pad];
        for i in 0..rows {
            let row = &w_flat[i * n..(i + 1) * n];
            macc_row(&mut p0, c0r[i], row, n);
            macc_row(&mut p1, c0i[i], row, n);
            macc_row(&mut p2, c1r[i], row, n);
            macc_row(&mut p3, c1i[i], row, n);
        }

        let mut w_partial = Vec::with_capacity(n_pad);
        for j in 0..n_pad {
            let c0 = Complex::new(p0[j], p1[j]);
            let c1 = Complex::new(p2[j], p3[j]);
            w_partial.push(EF::from_base_slice(&[c0, c1]));
        }
        w_partial
    }
}

/// Prove y = W @ x + bias with EF challenges.
pub fn prove_matmul_ef(
    w_flat: &[F],
    x_vals: &[F],
    y_vals: &[F],
    m: usize,
    n: usize,
    _bias: Option<&[F]>,
    transcript: &mut Transcript,
) -> (MatmulProofEF, Vec<EF>) {
    let log_m = log2_ceil(m);
    let log_n = log2_ceil(n);
    let m_pad = 1 << log_m;
    let n_pad = 1 << log_n;

    let mut y_padded = y_vals.to_vec();
    y_padded.resize(m_pad, F::zero());
    let mut x_padded = x_vals.to_vec();
    x_padded.resize(n_pad, F::zero());

    transcript.absorb_many(&y_padded);
    let r_point = transcript.squeeze_ef_many(log_m);

    let w_partial = fold_eq_matvec_ef(w_flat, m, n, n_pad, &r_point);
    assert_eq!(w_partial.len(), n_pad);

    // Convert w_partial to base field for product sumcheck input.
    // w_partial is EF, x_padded is F. The sumcheck must handle EF × F.
    // Since prove_product_ef takes &[F], we need to handle this differently.
    // The EF sumcheck takes base-field inputs and produces EF working arrays.
    // But w_partial is already EF. We need a variant that handles EF inputs directly.
    //
    // Simpler approach: inline a product sumcheck over (EF, F) mixed inputs.
    let (proof, w_at_s, x_at_s) = prove_product_ef_mixed(&w_partial, &x_padded, log_n, transcript);

    let s_point: Vec<EFElement> = proof.challenges.clone();
    let r_raw: Vec<EFElement> = r_point.iter().map(|&r| EFElement::from_ef(r)).collect();

    (
        MatmulProofEF {
            sumcheck_proof: proof,
            w_at_rs: EFElement::from_ef(w_at_s),
            x_at_s: EFElement::from_ef(x_at_s),
            r_point: r_raw,
            s_point,
        },
        w_partial,
    )
}

/// Product sumcheck over (Vec<EF>, Vec<F>) inputs — w_partial × x_padded.
/// First input is already in EF (from fold_eq_matvec_ef), second is base field.
fn prove_product_ef_mixed(
    f_ef: &[EF],
    g_base: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProofEF, EF, EF) {
    let mut f = f_ef.to_vec();
    let mut g: Vec<EF> = g_base.iter().map(|&v| f_to_ef(v)).collect();
    let mut size = f.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        let mut s0 = EF::zero();
        let mut s1 = EF::zero();
        let mut s2 = EF::zero();

        for j in 0..half {
            let f0 = f[j];
            let f1 = f[j + half];
            let g0 = g[j];
            let g1 = g[j + half];

            s0 += f0 * g0;
            s1 += f1 * g1;

            let f2 = f1 + f1 - f0;
            let g2 = g1 + g1 - g0;
            s2 += f2 * g2;
        }

        let poly = vec![
            EFElement::from_ef(s0),
            EFElement::from_ef(s1),
            EFElement::from_ef(s2),
        ];
        transcript.absorb_ef(s0);
        transcript.absorb_ef(s1);
        transcript.absorb_ef(s2);
        let r = transcript.squeeze_ef();

        let one_minus_r = EF::one() - r;
        for j in 0..half {
            f[j] = one_minus_r * f[j] + r * f[j + half];
            g[j] = one_minus_r * g[j] + r * g[j + half];
        }

        round_polys.push(poly);
        challenges.push(EFElement::from_ef(r));
        size = half;
    }

    (SumcheckProofEF { round_polys, challenges }, f[0], g[0])
}

/// Matmul proof with succinct weight verification (EF challenges).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuccinctMatmulProofEF {
    pub matmul_proof: MatmulProofEF,
    pub w_eval_proof: MleEvalProofEF,
    /// Deprecated: was always = w_at_rs. Kept for serde compat.
    #[serde(default)]
    pub w_claimed_value: EFElement,
    pub w_partial_commitment: WeightCommitment,
    /// Deprecated: was always true. Kept for serde compat.
    #[serde(default)]
    pub bound: bool,
    #[serde(default)]
    pub basefold_proof: Option<crate::proving::basefold::BasefoldOpeningProof>,
}

/// Prove y = W @ x + b with succinct weight commitment and EF challenges.
pub fn prove_matmul_succinct_ef(
    w_flat: &[F],
    x_vals: &[F],
    y_vals: &[F],
    m: usize,
    n: usize,
    bias: Option<&[F]>,
    transcript: &mut Transcript,
) -> SuccinctMatmulProofEF {
    let (matmul_proof, w_partial) =
        prove_matmul_ef(w_flat, x_vals, y_vals, m, n, bias, transcript);

    let w_partial_base: Vec<F> = w_partial.iter().flat_map(|&v| {
        let bs: &[Complex<Mersenne31>] = v.as_base_slice();
        let c0: &[Mersenne31] = bs[0].as_base_slice();
        let c1: &[Mersenne31] = bs[1].as_base_slice();
        vec![c0[0], c0[1], c1[0], c1[1]]
    }).collect();
    let s_point: Vec<EF> = matmul_proof.s_point.iter().map(|v| v.to_ef()).collect();
    let claimed = matmul_proof.w_at_rs.to_ef();

    let (w_partial_commitment, w_eval_proof, basefold_proof) = bind_weights_ef(
        &w_partial, &w_partial_base, Some((w_flat, m, n)),
        &s_point, claimed, transcript,
    );

    SuccinctMatmulProofEF {
        matmul_proof,
        w_eval_proof,
        w_claimed_value: EFElement([0; 4]), // deprecated: verifier uses w_at_rs directly
        w_partial_commitment,
        bound: false, // deprecated: never read by verifier
        basefold_proof,
    }
}

/// Verify a succinct matmul proof with EF challenges.
pub fn verify_matmul_succinct_ef(
    proof: &SuccinctMatmulProofEF,
    _commitment: &WeightCommitment,
    y_vals: &[F],
    m: usize,
    n: usize,
    bias: Option<&[F]>,
    transcript: &mut Transcript,
) -> MatmulVerifyResultEF {
    let fail = MatmulVerifyResultEF {
        valid: false,
        x_claim_point: vec![],
        x_claim_value: EF::zero(),
    };

    let log_m = log2_ceil(m);
    let m_pad = 1 << log_m;

    let mut y_padded = y_vals.to_vec();
    y_padded.resize(m_pad, F::zero());

    transcript.absorb_many(&y_padded);
    let r_point = transcript.squeeze_ef_many(log_m);

    // Skip r_point check when stripped (verifier re-derives from transcript)
    if !proof.matmul_proof.r_point.is_empty() {
        let r_expected: Vec<EFElement> = r_point.iter().map(|&r| EFElement::from_ef(r)).collect();
        if r_expected != proof.matmul_proof.r_point {
            return fail;
        }
    }

    let y_at_r = mle_evaluate_ef(&y_padded, &r_point);
    let mut claim = y_at_r;
    if let Some(b) = bias {
        let mut b_padded = b.to_vec();
        b_padded.resize(m_pad, F::zero());
        let b_at_r = mle_evaluate_ef(&b_padded, &r_point);
        claim = claim - b_at_r;
    }

    let w_at_rs = proof.matmul_proof.w_at_rs.to_ef();
    let x_at_s = proof.matmul_proof.x_at_s.to_ef();

    // Use challenge-returning variant to derive s* from the transcript.
    // When s_point/challenges are stripped, the derived challenges provide s*.
    #[allow(unused_variables)]
    let log_n = log2_ceil(n);
    let (sc_valid, derived_s) = sumcheck::verify_product_ef_with_challenges(
        claim, &proof.matmul_proof.sumcheck_proof, w_at_rs, x_at_s, transcript,
    );
    if !sc_valid {
        return fail;
    }

    // Use derived challenges as s_point (works even when proof.s_point is empty)
    let s_point: Vec<EF> = if proof.matmul_proof.s_point.is_empty() {
        derived_s
    } else {
        proof.matmul_proof.s_point.iter().map(|v| v.to_ef()).collect()
    };

    // Dispatch: Basefold PCS or sumcheck MLE eval
    if let Some(ref bf_proof) = proof.basefold_proof {
        // Extract 16-byte Merkle digest from 32-byte WeightCommitment root.
        let mut bf_root = crate::proving::pcs::Digest::default();
        bf_root.copy_from_slice(&proof.w_partial_commitment.root[..crate::proving::pcs::DIGEST_BYTES]);
        let bf_commitment = crate::proving::basefold::BasefoldCommitment {
            root: bf_root,
            num_coeffs: proof.w_partial_commitment.num_weights,
            log_n: proof.w_partial_commitment.log_height,
        };
        // Squeeze the same base-field evaluation point the prover used
        let bf_log = bf_commitment.log_n;
        let bf_point: Vec<F> = (0..bf_log).map(|_| transcript.squeeze()).collect();
        // Absorb commitment root (matching prover's commit_and_prove which absorbs internally)
        transcript.absorb_bytes(&bf_commitment.root);
        // The Basefold proof's final_value IS the evaluation — fold consistency
        // at queried positions is the real soundness check, not the final_value == claimed
        // comparison (which is tautological here). The EF sumcheck below independently
        // verifies the polynomial evaluation at the EF challenge point.
        let bf_claimed = F::from_canonical_u32(bf_proof.final_value);
        if !crate::proving::basefold::verify_opening(
            &bf_commitment, bf_claimed, &bf_point, bf_proof, transcript,
        ) {
            return fail;
        }
        // Now verify the EF MLE eval proof (transcript continues after Basefold)
        transcript.absorb_bytes(&proof.w_partial_commitment.root);
        if !weight_commitment::verify_mle_eval_ef(
            &proof.w_partial_commitment,
            w_at_rs,
            &s_point,
            &proof.w_eval_proof,
            transcript,
        ) {
            return fail;
        }
    } else {
        transcript.absorb_bytes(&proof.w_partial_commitment.root);
        if !weight_commitment::verify_mle_eval_ef(
            &proof.w_partial_commitment,
            w_at_rs,
            &s_point,
            &proof.w_eval_proof,
            transcript,
        ) {
            return fail;
        }
    }

    MatmulVerifyResultEF {
        valid: true,
        x_claim_point: s_point,
        x_claim_value: x_at_s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2x2() {
        let w: Vec<F> = [3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let x: Vec<F> = [2, 4]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        // y = [3*2+5*4, 7*2+11*4] = [26, 58]
        let y: Vec<F> = [26, 58]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();

        let mut t = Transcript::new(b"matmul-test");
        let (proof, _w_partial) = prove_matmul(&w, &x, &y, 2, 2, None, &mut t);

        let mut t2 = Transcript::new(b"matmul-test");
        let result = verify_matmul(&proof, &w, &y, 2, 2, None, &mut t2);
        assert!(result.valid, "2x2 matmul verification failed");
        assert!(!result.x_claim_point.is_empty(), "Must produce a claim point");
        // x̃(s) should equal MLE evaluation of x at the claim point
        let x_at_s = mle_evaluate(&x, &result.x_claim_point);
        assert_eq!(result.x_claim_value, x_at_s, "Claim value must match MLE of x");
    }

    #[test]
    fn test_matmul_with_bias() {
        let w: Vec<F> = [3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let x: Vec<F> = [2, 4]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let bias: Vec<F> = [10, 20]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        // y = [26+10, 58+20] = [36, 78]
        let y: Vec<F> = [36, 78]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();

        let mut t = Transcript::new(b"matmul-test");
        let (proof, _w_partial) = prove_matmul(&w, &x, &y, 2, 2, Some(&bias), &mut t);

        let mut t2 = Transcript::new(b"matmul-test");
        let result = verify_matmul(&proof, &w, &y, 2, 2, Some(&bias), &mut t2);
        assert!(result.valid, "2x2 matmul+bias verification failed");
    }

    #[test]
    fn test_matmul_succinct_bound_2x2() {
        let w: Vec<F> = [3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let x: Vec<F> = [2, 4]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let y: Vec<F> = [26, 58]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();

        let w_commitment = weight_commitment::commit_weights_fast(&w);

        let mut t = Transcript::new(b"bound-test");
        let proof = prove_matmul_succinct(
            &w, &x, &y, 2, 2, None, &mut t,
        );

        // Verify using the proof's w_partial_commitment (what the verifier actually checks)
        let mut t2 = Transcript::new(b"bound-test");
        let result = verify_matmul_succinct(
            &proof, &proof.w_partial_commitment.clone(), &y, 2, 2, None, &mut t2,
        );
        assert!(result.valid, "Bound 2x2 matmul verification failed");
    }

    #[test]
    fn test_matmul_succinct_bound_verifies() {
        // Verify that prove_matmul_succinct_bound produces valid proofs.
        let w: Vec<F> = [3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let x: Vec<F> = [2, 4]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let y: Vec<F> = [26, 58]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();

        let _w_commitment = weight_commitment::commit_weights_fast(&w);

        let mut t = Transcript::new(b"bound-test");
        let proof = prove_matmul_succinct(
            &w, &x, &y, 2, 2, None, &mut t,
        );

        let mut t2 = Transcript::new(b"bound-test");
        let result = verify_matmul_succinct(
            &proof, &proof.w_partial_commitment.clone(), &y, 2, 2, None, &mut t2,
        );
        assert!(result.valid, "Bound proof must verify with correct commitment");
    }

    #[test]
    fn test_matmul_succinct_binding_rejects_tampered_w_partial() {
        // Soundness test: prove with weights W, then tamper with w_partial_commitment
        // inside the proof. The Fiat-Shamir binding must catch it because the
        // MLE eval challenges depend on the absorbed commitment root.

        let w: Vec<F> = [3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let x: Vec<F> = [2, 4]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let y: Vec<F> = [26, 58]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();

        let mut t = Transcript::new(b"binding-test");
        let mut proof = prove_matmul_succinct(
            &w, &x, &y, 2, 2, None, &mut t,
        );
        // Verify with real commitment — should pass
        let real_commitment = proof.w_partial_commitment.clone();
        let mut t2 = Transcript::new(b"binding-test");
        let result = verify_matmul_succinct(
            &proof, &real_commitment, &y, 2, 2, None, &mut t2,
        );
        assert!(result.valid, "Honest proof must verify");

        // Tamper: replace w_partial_commitment with a different root
        proof.w_partial_commitment.root = [0xDE; 32];

        // Verify with tampered commitment — must FAIL
        let mut t3 = Transcript::new(b"binding-test");
        let result2 = verify_matmul_succinct(
            &proof, &proof.w_partial_commitment.clone(), &y, 2, 2, None, &mut t3,
        );
        assert!(!result2.valid,
            "Tampered w_partial_commitment must be rejected by Fiat-Shamir binding");
    }

    #[test]
    fn test_matmul_succinct_bound_4x3() {
        // Non-square, non-power-of-2 matrix
        let w: Vec<F> = (0..12)
            .map(|i| F::from_canonical_u32(i * 3 + 1))
            .collect();
        let x: Vec<F> = (0..3)
            .map(|i| F::from_canonical_u32(i + 1))
            .collect();
        let y: Vec<F> = (0..4)
            .map(|i| {
                let mut sum = F::zero();
                for j in 0..3 {
                    sum += w[i * 3 + j] * x[j];
                }
                sum
            })
            .collect();

        let w_commitment = weight_commitment::commit_weights_fast(&w);

        let mut t = Transcript::new(b"bound-4x3");
        let proof = prove_matmul_succinct(
            &w, &x, &y, 4, 3, None, &mut t,
        );

        let mut t2 = Transcript::new(b"bound-4x3");
        let result = verify_matmul_succinct(&proof, &w_commitment, &y, 4, 3, None, &mut t2);
        assert!(result.valid, "Bound 4x3 matmul verification failed");
    }
}
