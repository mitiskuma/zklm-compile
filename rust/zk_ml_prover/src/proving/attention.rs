//! Row-decomposed attention proving over M31.
//!
//! Decomposes attention into independent per-row proofs:
//!   For each row i:
//!     1. scores_i = Q[i] @ K^T  (1×d_k · d_k×seq_len = 1×seq_len)
//!     2. attn_i = softmax(scores_i)
//!     3. out_i = attn_i @ V      (1×seq_len · seq_len×d_head = 1×d_head)
//!
//! This avoids materializing the full seq_len × seq_len attention matrix,
//! and each row proof is independent, enabling parallelism via rayon.

use p3_field::{AbstractField, Field, PrimeField32};
use p3_mersenne_31::Mersenne31;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::field::common::log2_ceil;
use crate::proving::lookup::LookupTable;
use crate::field::m31_ops::mle_evaluate;
use crate::proving::matmul::{prove_matmul, verify_matmul, MatmulProof};
use crate::proving::softmax::{prove_softmax, verify_softmax, SoftmaxProof};
use crate::proving::sumcheck::Transcript;

type F = Mersenne31;

/// Proof for a single row of attention.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowProof {
    /// Proves scores_i = Q[i] @ K^T (1×d_head matmul producing 1×seq_len)
    pub score_proof: MatmulProof,
    /// Proves attn_i = softmax(scores_i)
    pub softmax_proof: SoftmaxProof,
    /// Proves out_i = attn_i @ V (1×seq_len matmul producing 1×d_head)
    pub output_proof: MatmulProof,
    /// Score values (seq_len elements)
    pub scores: Vec<u32>,
    /// Softmax output values (seq_len elements)
    pub attn_weights: Vec<u32>,
    /// Output values (d_head elements)
    pub output: Vec<u32>,
}

/// Full row-decomposed multi-head attention proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowAttentionProof {
    /// Per-head, per-row proofs: row_proofs[h][i]
    pub row_proofs: Vec<Vec<RowProof>>,
    pub num_heads: usize,
    pub seq_len: usize,
    pub d_head: usize,
}

/// Compute softmax from field elements using the exp lookup table.
fn compute_softmax_row(
    scores: &[F],
    index_map: &HashMap<u32, u32>,
) -> (Vec<F>, Vec<F>) {
    let e: Vec<F> = scores
        .iter()
        .map(|&s| F::from_canonical_u32(index_map[&s.as_canonical_u32()]))
        .collect();
    let sum: F = e.iter().copied().sum();
    let inv_s = sum.inverse();
    let y: Vec<F> = e.iter().map(|&ei| ei * inv_s).collect();
    (e, y)
}

/// Prove row-decomposed multi-head attention.
///
/// Q, K, V are flat arrays of shape (num_heads, seq_len, d_head).
/// The proof decomposes into independent per-row sub-proofs, parallelized with rayon.
pub fn prove_row_attention(
    q: &[F],
    k: &[F],
    v: &[F],
    num_heads: usize,
    seq_len: usize,
    d_head: usize,
    exp_table: &LookupTable,
    transcript: &mut Transcript,
) -> RowAttentionProof {
    let head_size = seq_len * d_head;
    assert_eq!(q.len(), num_heads * head_size);
    assert_eq!(k.len(), num_heads * head_size);
    assert_eq!(v.len(), num_heads * head_size);

    // Build exp index map once
    let mut index_map = HashMap::new();
    for &(inp, out) in &exp_table.entries {
        index_map.insert(inp, out);
    }

    // Absorb Q, K, V dimensions into transcript for domain separation
    transcript.absorb(num_heads as u32);
    transcript.absorb(seq_len as u32);
    transcript.absorb(d_head as u32);

    let mut all_row_proofs = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        let q_head = &q[h * head_size..(h + 1) * head_size];
        let k_head = &k[h * head_size..(h + 1) * head_size];
        let v_head = &v[h * head_size..(h + 1) * head_size];

        // Transpose K_head: (seq_len × d_head) -> (d_head × seq_len)
        let mut k_t = vec![F::zero(); d_head * seq_len];
        for r in 0..seq_len {
            for c in 0..d_head {
                k_t[c * seq_len + r] = k_head[r * d_head + c];
            }
        }

        // Parallelize across rows
        let row_proofs: Vec<RowProof> = (0..seq_len)
            .into_par_iter()
            .map(|i| {
                // Fork transcript: label with head and row index
                let mut row_transcript =
                    Transcript::new(format!("row-attn-h{}-r{}", h, i).as_bytes());

                // Extract Q[h][i]: 1 × d_head
                let q_row = &q_head[i * d_head..(i + 1) * d_head];

                // 1. Prove scores_i = Q[h][i] @ K[h]^T
                //    W = K^T (d_head × seq_len), x = Q[h][i] (d_head), y = scores (seq_len)
                //    Actually matmul is y = W @ x where W is (m × n), x is (n), y is (m)
                //    We want scores (seq_len) = K^T (d_head × seq_len)^T ... no.
                //    prove_matmul proves y = W @ x, W is (m×n), x is (n,), y is (m,).
                //    We want: scores_i (seq_len,) = K^T^T @ q_row ... that's K @ q_row
                //    No: Q[i] @ K^T means (1×d_head) · (d_head×seq_len) = 1×seq_len
                //    As a matmul y = W @ x: W = K^T transposed = K (seq_len×d_head),
                //    x = q_row (d_head), y = scores (seq_len).
                //    So W = K_head (seq_len × d_head), m = seq_len, n = d_head.

                // Compute scores
                let mut scores = vec![F::zero(); seq_len];
                for j in 0..seq_len {
                    let mut acc = F::zero();
                    for l in 0..d_head {
                        acc += k_head[j * d_head + l] * q_row[l];
                    }
                    scores[j] = acc;
                }

                let (score_proof, _) = prove_matmul(
                    k_head, // W = K_head (seq_len × d_head)
                    q_row,  // x = Q[h][i] (d_head)
                    &scores, // y = scores (seq_len)
                    seq_len, // m
                    d_head,  // n
                    None,
                    &mut row_transcript,
                );

                // 2. Softmax
                let (e, y) = compute_softmax_row(&scores, &index_map);
                let softmax_proof =
                    prove_softmax(&scores, &e, &y, exp_table, &mut row_transcript);

                // 3. Prove out_i = attn_weights @ V[h]
                //    y = W @ x: W = V_head (seq_len × d_head), but we want
                //    out_i (d_head) = V^T (d_head × seq_len) @ attn_weights (seq_len)
                //    So W = V^T (d_head × seq_len), m = d_head, n = seq_len.
                let mut v_t = vec![F::zero(); d_head * seq_len];
                for r in 0..seq_len {
                    for c in 0..d_head {
                        v_t[c * seq_len + r] = v_head[r * d_head + c];
                    }
                }

                let mut out_row = vec![F::zero(); d_head];
                for j in 0..d_head {
                    let mut acc = F::zero();
                    for l in 0..seq_len {
                        acc += y[l] * v_head[l * d_head + j];
                    }
                    out_row[j] = acc;
                }

                let (output_proof, _) = prove_matmul(
                    &v_t,  // W = V^T (d_head × seq_len)
                    &y,    // x = attn_weights (seq_len)
                    &out_row, // y = output (d_head)
                    d_head, // m
                    seq_len, // n
                    None,
                    &mut row_transcript,
                );

                RowProof {
                    score_proof,
                    softmax_proof,
                    output_proof,
                    scores: scores.iter().map(|v| v.as_canonical_u32()).collect(),
                    attn_weights: y.iter().map(|v| v.as_canonical_u32()).collect(),
                    output: out_row.iter().map(|v| v.as_canonical_u32()).collect(),
                }
            })
            .collect();

        all_row_proofs.push(row_proofs);
    }

    // Absorb all row proof commitments into the main transcript for binding
    for h in 0..num_heads {
        for i in 0..seq_len {
            let rp = &all_row_proofs[h][i];
            for &v in &rp.scores {
                transcript.absorb(v);
            }
            for &v in &rp.output {
                transcript.absorb(v);
            }
        }
    }

    RowAttentionProof {
        row_proofs: all_row_proofs,
        num_heads,
        seq_len,
        d_head,
    }
}

/// Verify a row-decomposed multi-head attention proof.
pub fn verify_row_attention(
    proof: &RowAttentionProof,
    q: &[F],
    k: &[F],
    v: &[F],
    exp_table: &LookupTable,
    transcript: &mut Transcript,
) -> bool {
    let num_heads = proof.num_heads;
    let seq_len = proof.seq_len;
    let d_head = proof.d_head;
    let head_size = seq_len * d_head;

    transcript.absorb(num_heads as u32);
    transcript.absorb(seq_len as u32);
    transcript.absorb(d_head as u32);

    for h in 0..num_heads {
        let q_head = &q[h * head_size..(h + 1) * head_size];
        let k_head = &k[h * head_size..(h + 1) * head_size];
        let v_head = &v[h * head_size..(h + 1) * head_size];

        let mut v_t = vec![F::zero(); d_head * seq_len];
        for r in 0..seq_len {
            for c in 0..d_head {
                v_t[c * seq_len + r] = v_head[r * d_head + c];
            }
        }

        // Parallelize verification across rows
        let results: Vec<bool> = (0..seq_len)
            .into_par_iter()
            .map(|i| {
                let rp = &proof.row_proofs[h][i];
                let mut row_transcript =
                    Transcript::new(format!("row-attn-h{}-r{}", h, i).as_bytes());

                let _q_row = &q_head[i * d_head..(i + 1) * d_head];
                let scores: Vec<F> = rp.scores.iter().map(|&v| F::from_canonical_u32(v)).collect();
                let _attn_weights: Vec<F> =
                    rp.attn_weights.iter().map(|&v| F::from_canonical_u32(v)).collect();
                let output: Vec<F> = rp.output.iter().map(|&v| F::from_canonical_u32(v)).collect();

                // 1. Verify score matmul
                let score_result = verify_matmul(
                    &rp.score_proof,
                    k_head,
                    &scores,
                    seq_len,
                    d_head,
                    None,
                    &mut row_transcript,
                );
                if !score_result.valid {
                    return false;
                }

                // 2. Verify softmax
                if !verify_softmax(&rp.softmax_proof, exp_table, seq_len, &mut row_transcript) {
                    return false;
                }

                // 3. Verify output matmul
                let output_result = verify_matmul(
                    &rp.output_proof,
                    &v_t,
                    &output,
                    d_head,
                    seq_len,
                    None,
                    &mut row_transcript,
                );
                if !output_result.valid {
                    return false;
                }

                true
            })
            .collect();

        if results.iter().any(|&r| !r) {
            return false;
        }
    }

    // Absorb into main transcript (must match prover)
    for h in 0..num_heads {
        for i in 0..seq_len {
            let rp = &proof.row_proofs[h][i];
            for &v in &rp.scores {
                transcript.absorb(v);
            }
            for &v in &rp.output {
                transcript.absorb(v);
            }
        }
    }

    true
}

/// Prove row-decomposed Grouped Query Attention (GQA).
///
/// Q has `num_q_heads` heads, K/V have `num_kv_heads` heads.
/// Each KV head serves `num_q_heads / num_kv_heads` Q heads.
/// When num_q_heads == num_kv_heads, this is standard MHA.
/// When num_kv_heads == 1, this is MQA.
///
/// Q: flat array of shape (num_q_heads, seq_len, d_head)
/// K: flat array of shape (num_kv_heads, seq_len, d_head)
/// V: flat array of shape (num_kv_heads, seq_len, d_head)
pub fn prove_row_attention_gqa(
    q: &[F],
    k: &[F],
    v: &[F],
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    d_head: usize,
    exp_table: &LookupTable,
    transcript: &mut Transcript,
) -> RowAttentionProof {
    let head_size = seq_len * d_head;
    assert_eq!(q.len(), num_q_heads * head_size);
    assert_eq!(k.len(), num_kv_heads * head_size);
    assert_eq!(v.len(), num_kv_heads * head_size);
    assert_eq!(num_q_heads % num_kv_heads, 0, "num_q_heads must be divisible by num_kv_heads");

    let heads_per_group = num_q_heads / num_kv_heads;

    let mut index_map = HashMap::new();
    for &(inp, out) in &exp_table.entries {
        index_map.insert(inp, out);
    }

    // Absorb dimensions (use num_q_heads for transcript, plus num_kv_heads)
    transcript.absorb(num_q_heads as u32);
    transcript.absorb(num_kv_heads as u32);
    transcript.absorb(seq_len as u32);
    transcript.absorb(d_head as u32);

    let mut all_row_proofs = Vec::with_capacity(num_q_heads);

    for h in 0..num_q_heads {
        let q_head = &q[h * head_size..(h + 1) * head_size];
        // Map Q head to its KV head group
        let kv_idx = h / heads_per_group;
        let k_head = &k[kv_idx * head_size..(kv_idx + 1) * head_size];
        let v_head = &v[kv_idx * head_size..(kv_idx + 1) * head_size];

        let row_proofs: Vec<RowProof> = (0..seq_len)
            .into_par_iter()
            .map(|i| {
                let mut row_transcript =
                    Transcript::new(format!("gqa-h{}-r{}", h, i).as_bytes());

                let q_row = &q_head[i * d_head..(i + 1) * d_head];

                // 1. scores_i = K_head @ q_row
                let mut scores = vec![F::zero(); seq_len];
                for j in 0..seq_len {
                    let mut acc = F::zero();
                    for l in 0..d_head {
                        acc += k_head[j * d_head + l] * q_row[l];
                    }
                    scores[j] = acc;
                }

                let (score_proof, _) = prove_matmul(
                    k_head, q_row, &scores,
                    seq_len, d_head, None, &mut row_transcript,
                );

                // 2. softmax
                let (e, y) = compute_softmax_row(&scores, &index_map);
                let softmax_proof = prove_softmax(&scores, &e, &y, exp_table, &mut row_transcript);

                // 3. out_i = V^T @ attn_weights
                let mut v_t = vec![F::zero(); d_head * seq_len];
                for r in 0..seq_len {
                    for c in 0..d_head {
                        v_t[c * seq_len + r] = v_head[r * d_head + c];
                    }
                }

                let mut out_row = vec![F::zero(); d_head];
                for j in 0..d_head {
                    let mut acc = F::zero();
                    for l in 0..seq_len {
                        acc += y[l] * v_head[l * d_head + j];
                    }
                    out_row[j] = acc;
                }

                let (output_proof, _) = prove_matmul(
                    &v_t, &y, &out_row,
                    d_head, seq_len, None, &mut row_transcript,
                );

                RowProof {
                    score_proof,
                    softmax_proof,
                    output_proof,
                    scores: scores.iter().map(|v| v.as_canonical_u32()).collect(),
                    attn_weights: y.iter().map(|v| v.as_canonical_u32()).collect(),
                    output: out_row.iter().map(|v| v.as_canonical_u32()).collect(),
                }
            })
            .collect();

        all_row_proofs.push(row_proofs);
    }

    // Absorb into main transcript
    for h in 0..num_q_heads {
        for i in 0..seq_len {
            let rp = &all_row_proofs[h][i];
            for &v in &rp.scores { transcript.absorb(v); }
            for &v in &rp.output { transcript.absorb(v); }
        }
    }

    RowAttentionProof {
        row_proofs: all_row_proofs,
        num_heads: num_q_heads,
        seq_len,
        d_head,
    }
}

/// Verify a GQA proof.
#[allow(dead_code)]
pub fn verify_row_attention_gqa(
    proof: &RowAttentionProof,
    q: &[F],
    k: &[F],
    v: &[F],
    num_kv_heads: usize,
    exp_table: &LookupTable,
    transcript: &mut Transcript,
) -> bool {
    let num_q_heads = proof.num_heads;
    let seq_len = proof.seq_len;
    let d_head = proof.d_head;
    let head_size = seq_len * d_head;
    let heads_per_group = num_q_heads / num_kv_heads;

    transcript.absorb(num_q_heads as u32);
    transcript.absorb(num_kv_heads as u32);
    transcript.absorb(seq_len as u32);
    transcript.absorb(d_head as u32);

    for h in 0..num_q_heads {
        let kv_idx = h / heads_per_group;
        let k_head = &k[kv_idx * head_size..(kv_idx + 1) * head_size];
        let v_head = &v[kv_idx * head_size..(kv_idx + 1) * head_size];

        let mut v_t = vec![F::zero(); d_head * seq_len];
        for r in 0..seq_len {
            for c in 0..d_head {
                v_t[c * seq_len + r] = v_head[r * d_head + c];
            }
        }

        let results: Vec<bool> = (0..seq_len)
            .into_par_iter()
            .map(|i| {
                let rp = &proof.row_proofs[h][i];
                let mut row_transcript =
                    Transcript::new(format!("gqa-h{}-r{}", h, i).as_bytes());

                let scores: Vec<F> = rp.scores.iter().map(|&v| F::from_canonical_u32(v)).collect();
                let output: Vec<F> = rp.output.iter().map(|&v| F::from_canonical_u32(v)).collect();

                let score_result = verify_matmul(
                    &rp.score_proof, k_head, &scores,
                    seq_len, d_head, None, &mut row_transcript,
                );
                if !score_result.valid { return false; }

                // Verify x_claim: the matmul verifier claims x (= Q row) evaluates to
                // x_claim_value at x_claim_point. Check by computing Q row's MLE.
                if !score_result.x_claim_point.is_empty() {
                    let q_head = &q[h * head_size..(h + 1) * head_size];
                    let q_row = &q_head[i * d_head..(i + 1) * d_head];
                    let log_n = log2_ceil(d_head);
                    let n_pad = 1 << log_n;
                    let mut q_padded = q_row.to_vec();
                    q_padded.resize(n_pad, F::zero());
                    let q_at_point = mle_evaluate(&q_padded, &score_result.x_claim_point);
                    if q_at_point != score_result.x_claim_value {
                        return false;
                    }
                }

                if !verify_softmax(&rp.softmax_proof, exp_table, seq_len, &mut row_transcript) {
                    return false;
                }

                let output_result = verify_matmul(
                    &rp.output_proof, &v_t, &output,
                    d_head, seq_len, None, &mut row_transcript,
                );
                if !output_result.valid { return false; }

                true
            })
            .collect();

        if results.iter().any(|&r| !r) { return false; }
    }

    for h in 0..num_q_heads {
        for i in 0..seq_len {
            let rp = &proof.row_proofs[h][i];
            for &v in &rp.scores { transcript.absorb(v); }
            for &v in &rp.output { transcript.absorb(v); }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::common::quantize_i16;
    use crate::proving::lookup::LookupTable;
    use crate::field::m31_ops::to_field;
    use crate::proving::weight_commitment::WeightCommitment;
    use std::time::Instant;

    fn build_small_exp_table(scale: i32) -> LookupTable {
        let s = scale as f64;
        let mut entries = Vec::with_capacity(256);
        for raw in 0u32..256 {
            let input_i16 = raw as i8 as i16;
            let x = input_i16 as f64 / s;
            let y = x.exp();
            let output_i16 = quantize_i16(y, s);
            entries.push((
                to_field(input_i16 as i64).as_canonical_u32(),
                to_field(output_i16 as i64).as_canonical_u32(),
            ));
        }
        LookupTable {
            name: "exp_small".to_string(),
            entries,
            commitment: WeightCommitment {
                root: [0u8; 32],
                num_weights: 256,
                log_height: 8,
            },
        }
    }

    /// Generate deterministic Q, K, V with small values that fit in exp table range.
    fn make_qkv(
        num_heads: usize,
        seq_len: usize,
        d_head: usize,
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        let total = num_heads * seq_len * d_head;
        // Use small values so Q@K^T scores stay in [-128, 127] for the exp table
        let q: Vec<F> = (0..total)
            .map(|i| F::from_canonical_u32(((i % 3) + 1) as u32))
            .collect();
        let k: Vec<F> = (0..total)
            .map(|i| F::from_canonical_u32(((i % 2) + 1) as u32))
            .collect();
        let v: Vec<F> = (0..total)
            .map(|i| F::from_canonical_u32(((i % 4) + 1) as u32))
            .collect();
        (q, k, v)
    }

    #[test]
    fn test_row_attention_small() {
        let num_heads = 2;
        let seq_len = 4;
        let d_head = 4;
        let exp_table = build_small_exp_table(100);

        let (q, k, v) = make_qkv(num_heads, seq_len, d_head);

        let mut pt = Transcript::new(b"row-attn-test");
        let proof = prove_row_attention(&q, &k, &v, num_heads, seq_len, d_head, &exp_table, &mut pt);

        assert_eq!(proof.row_proofs.len(), num_heads);
        assert_eq!(proof.row_proofs[0].len(), seq_len);

        let mut vt = Transcript::new(b"row-attn-test");
        assert!(
            verify_row_attention(&proof, &q, &k, &v, &exp_table, &mut vt),
            "Small row attention verification failed"
        );
    }

    #[test]
    fn test_row_attention_medium() {
        let num_heads = 4;
        let seq_len = 16;
        let d_head = 8;
        let exp_table = build_small_exp_table(100);

        // Use even smaller values to keep scores in range with d_head=8
        let total = num_heads * seq_len * d_head;
        let q: Vec<F> = (0..total)
            .map(|i| F::from_canonical_u32((i % 2 + 1) as u32))
            .collect();
        let k: Vec<F> = (0..total)
            .map(|i| F::from_canonical_u32((i % 2 + 1) as u32))
            .collect();
        let v: Vec<F> = (0..total)
            .map(|i| F::from_canonical_u32((i % 3 + 1) as u32))
            .collect();

        let mut pt = Transcript::new(b"row-attn-medium");
        let proof = prove_row_attention(&q, &k, &v, num_heads, seq_len, d_head, &exp_table, &mut pt);

        let mut vt = Transcript::new(b"row-attn-medium");
        assert!(
            verify_row_attention(&proof, &q, &k, &v, &exp_table, &mut vt),
            "Medium row attention verification failed"
        );
    }

    #[test]
    fn test_row_attention_scaling() {
        let num_heads = 1;
        let seq_len = 64;
        let d_head = 16;
        let exp_table = build_small_exp_table(100);

        // Very small values: scores = sum of d_head products of values in {1},
        // so scores will be at most d_head = 16, well within table range.
        let total = num_heads * seq_len * d_head;
        let q: Vec<F> = (0..total).map(|_| F::one()).collect();
        let k: Vec<F> = (0..total).map(|_| F::one()).collect();
        let v: Vec<F> = (0..total)
            .map(|i| F::from_canonical_u32((i % 3 + 1) as u32))
            .collect();

        let t0 = Instant::now();
        let mut pt = Transcript::new(b"row-attn-scale");
        let proof = prove_row_attention(&q, &k, &v, num_heads, seq_len, d_head, &exp_table, &mut pt);
        let prove_time = t0.elapsed();

        let t1 = Instant::now();
        let mut vt = Transcript::new(b"row-attn-scale");
        let valid = verify_row_attention(&proof, &q, &k, &v, &exp_table, &mut vt);
        let verify_time = t1.elapsed();

        assert!(valid, "Scaling row attention verification failed");
        eprintln!(
            "Row attention (1 head, seq_len=64, d_head=16): prove={:.1}ms, verify={:.1}ms",
            prove_time.as_secs_f64() * 1000.0,
            verify_time.as_secs_f64() * 1000.0
        );
    }

    /// GQA: 4 Q heads, 2 KV heads (each KV head serves 2 Q heads)
    #[test]
    fn test_gqa_basic() {
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let seq_len = 4;
        let d_head = 4;
        let exp_table = build_small_exp_table(100);

        let (q, _, _) = make_qkv(num_q_heads, seq_len, d_head);
        // K,V only have num_kv_heads heads
        let kv_total = num_kv_heads * seq_len * d_head;
        let k: Vec<F> = (0..kv_total)
            .map(|i| F::from_canonical_u32(((i % 2) + 1) as u32))
            .collect();
        let v: Vec<F> = (0..kv_total)
            .map(|i| F::from_canonical_u32(((i % 4) + 1) as u32))
            .collect();

        let mut pt = Transcript::new(b"gqa-test");
        let proof = prove_row_attention_gqa(
            &q, &k, &v, num_q_heads, num_kv_heads, seq_len, d_head, &exp_table, &mut pt,
        );

        assert_eq!(proof.row_proofs.len(), num_q_heads);
        assert_eq!(proof.num_heads, num_q_heads);

        let mut vt = Transcript::new(b"gqa-test");
        assert!(verify_row_attention_gqa(
            &proof, &q, &k, &v, num_kv_heads, &exp_table, &mut vt
        ));
    }

    /// MQA: 4 Q heads, 1 KV head
    #[test]
    fn test_gqa_mqa() {
        let num_q_heads = 4;
        let num_kv_heads = 1;
        let seq_len = 4;
        let d_head = 4;
        let exp_table = build_small_exp_table(100);

        let (q, _, _) = make_qkv(num_q_heads, seq_len, d_head);
        let kv_total = num_kv_heads * seq_len * d_head;
        let k: Vec<F> = (0..kv_total)
            .map(|i| F::from_canonical_u32(((i % 2) + 1) as u32))
            .collect();
        let v: Vec<F> = (0..kv_total)
            .map(|i| F::from_canonical_u32(((i % 3) + 1) as u32))
            .collect();

        let mut pt = Transcript::new(b"mqa-test");
        let proof = prove_row_attention_gqa(
            &q, &k, &v, num_q_heads, num_kv_heads, seq_len, d_head, &exp_table, &mut pt,
        );

        let mut vt = Transcript::new(b"mqa-test");
        assert!(verify_row_attention_gqa(
            &proof, &q, &k, &v, num_kv_heads, &exp_table, &mut vt
        ));
    }

    /// GQA degenerates to MHA when num_q_heads == num_kv_heads
    #[test]
    fn test_gqa_equals_mha() {
        let num_heads = 2;
        let seq_len = 4;
        let d_head = 4;
        let exp_table = build_small_exp_table(100);

        let (q, k, v) = make_qkv(num_heads, seq_len, d_head);

        // GQA with equal heads should produce a valid proof
        let mut pt = Transcript::new(b"gqa-mha");
        let proof = prove_row_attention_gqa(
            &q, &k, &v, num_heads, num_heads, seq_len, d_head, &exp_table, &mut pt,
        );

        let mut vt = Transcript::new(b"gqa-mha");
        assert!(verify_row_attention_gqa(
            &proof, &q, &k, &v, num_heads, &exp_table, &mut vt
        ));
    }

    /// Soundness: tampering with a sumcheck round polynomial in the score proof must cause rejection.
    #[test]
    fn test_row_attention_tampered_proof() {
        let num_heads = 2;
        let seq_len = 4;
        let d_head = 4;
        let exp_table = build_small_exp_table(100);

        let (q, k, v) = make_qkv(num_heads, seq_len, d_head);

        let mut pt = Transcript::new(b"row-attn-test");
        let proof = prove_row_attention(&q, &k, &v, num_heads, seq_len, d_head, &exp_table, &mut pt);

        // Sanity: the untampered proof verifies
        let mut vt = Transcript::new(b"row-attn-test");
        assert!(verify_row_attention(&proof, &q, &k, &v, &exp_table, &mut vt));

        // Tamper: flip a value in the first row's score sumcheck round polynomial
        let mut tampered = proof.clone();
        let poly = &mut tampered.row_proofs[0][0].score_proof.sumcheck_proof.round_polys[0];
        poly[0] = poly[0].wrapping_add(1) % (Mersenne31::ORDER_U32);

        let mut vt2 = Transcript::new(b"row-attn-test");
        let result = verify_row_attention(&tampered, &q, &k, &v, &exp_table, &mut vt2);
        assert!(!result, "Tampered score proof must be rejected");
    }

    /// Soundness: passing a different output to the verifier must cause rejection.
    #[test]
    fn test_row_attention_wrong_output() {
        let num_heads = 2;
        let seq_len = 4;
        let d_head = 4;
        let exp_table = build_small_exp_table(100);

        let (q, k, v) = make_qkv(num_heads, seq_len, d_head);

        let mut pt = Transcript::new(b"row-attn-test");
        let proof = prove_row_attention(&q, &k, &v, num_heads, seq_len, d_head, &exp_table, &mut pt);

        // Sanity: the correct proof verifies
        let mut vt = Transcript::new(b"row-attn-test");
        assert!(verify_row_attention(&proof, &q, &k, &v, &exp_table, &mut vt));

        // Tamper: change the claimed output of the first row
        let mut tampered = proof.clone();
        let out = &mut tampered.row_proofs[0][0].output;
        out[0] = (out[0] + 1) % Mersenne31::ORDER_U32;

        let mut vt2 = Transcript::new(b"row-attn-test");
        let result = verify_row_attention(&tampered, &q, &k, &v, &exp_table, &mut vt2);
        assert!(!result, "Wrong output must be rejected");
    }
}
