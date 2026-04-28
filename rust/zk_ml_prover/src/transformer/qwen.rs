//! Qwen3.5-style transformer layer proving (RMSNorm + GQA/GDN + output gating + SwiGLU).

use p3_field::{AbstractField, PrimeField32};
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::proving::elementwise::{prove_add, verify_add, AddProof, prove_add_ef, verify_add_ef, AddProofEF};
use crate::proving::lookup::LookupTable;
use crate::proving::matmul::{prove_matmul_succinct, verify_matmul_succinct, SuccinctMatmulProof, verify_matmul_succinct_ef, SuccinctMatmulProofEF};
use crate::proving::matmul::prove_matmul_succinct_ef;
use crate::proving::rmsnorm::{prove_rmsnorm, verify_rmsnorm, RmsNormProof, prove_rmsnorm_ef, verify_rmsnorm_ef, RmsNormProofEF};
use crate::proving::attention::RowAttentionProof;
use crate::proving::sigmoid_gate::{prove_sigmoid_gate, verify_sigmoid_gate, SigmoidGateProof, prove_sigmoid_gate_ef, SigmoidGateProofEF};
use crate::proving::sumcheck::Transcript;
use crate::proving::swiglu::{prove_swiglu, verify_swiglu, SwiGluProof, prove_swiglu_ef, SwiGluProofEF};
use crate::field::common::{log2_ceil};
use crate::proving::weight_commitment::WeightCommitment;
use super::{matmul_forward, rmsnorm_forward, requantize_to_i16_field, lookup_fast, commit_weight_matrix, ModelConfig, build_table_index, TableIndex};

type F = Mersenne31;

// ============================================================================
// Qwen3.5-style transformer (RMSNorm + GQA/GDN + output gating + SwiGLU)
// ============================================================================

/// SOUNDNESS (S3): absorb the model dimensions onto the transcript at the
/// start of every Qwen prove/verify pair. Without this, an adversary who can
/// influence `ModelConfig` could swap dimensions (e.g., flip the heuristic
/// `is_gqa_full_attn = num_q_heads != num_kv_heads` that picks attn_out_dim)
/// and have the proof appear valid against a different model. With it, any
/// dim mismatch between prover and verifier diverges the challenge stream
/// at the very first squeeze, causing verification to fail.
///
/// Order is fixed; both prover and verifier MUST call this with identical
/// values BEFORE the first weight-commitment absorption.
fn absorb_qwen_dims(config: &ModelConfig, transcript: &mut Transcript) {
    transcript.absorb_bytes(b"qwen-dims-v1");
    transcript.absorb(config.d_model as u32);
    transcript.absorb(config.d_ff as u32);
    transcript.absorb(config.num_q_heads as u32);
    transcript.absorb(config.num_kv_heads as u32);
    transcript.absorb(config.d_head as u32);
    // V dims (asymmetric GDN: Qwen3.5-4B/9B). Use v_num_heads/v_d_head as stored
    // (may be 0 = fall-back symmetric); the helper v_dim() is what's actually
    // used downstream so absorbing it explicitly closes the heuristic.
    transcript.absorb(config.v_num_heads as u32);
    transcript.absorb(config.v_d_head as u32);
    transcript.absorb(config.v_dim() as u32);
    let q_dim = config.num_q_heads * config.d_head;
    let k_dim = config.num_kv_heads * config.d_head;
    let v_dim = config.v_dim();
    let is_gqa_full_attn = config.num_q_heads != config.num_kv_heads;
    let attn_out_dim = if is_gqa_full_attn { q_dim } else { v_dim };
    transcript.absorb(q_dim as u32);
    transcript.absorb(k_dim as u32);
    transcript.absorb(attn_out_dim as u32);
    transcript.absorb(if is_gqa_full_attn { 1 } else { 0 });
}

/// Weights for a Qwen3.5-style transformer layer.
/// Same as Llama except: has w_g_proj for output gating (sigmoid gate).
pub struct QwenLayerWeights {
    pub norm1_gamma: Vec<F>,
    pub w_q: Vec<F>,        // (q_dim × d_model)
    pub w_k: Vec<F>,        // (kv_dim × d_model)
    pub w_v: Vec<F>,        // (kv_dim × d_model)
    pub w_o: Vec<F>,        // (d_model × q_dim)
    pub w_g_proj: Vec<F>,   // (q_dim × d_model) — output gate projection
    pub norm2_gamma: Vec<F>,
    pub w_gate: Vec<F>,     // (d_ff × d_model)
    pub w_up: Vec<F>,       // (d_ff × d_model)
    pub w_down: Vec<F>,     // (d_model × d_ff)
}

/// Proof for a Qwen3.5-style transformer layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QwenLayerProof {
    pub norm1_proof: RmsNormProof,
    pub qkv_proofs: (SuccinctMatmulProof, SuccinctMatmulProof, SuccinctMatmulProof),
    pub attn_proof: RowAttentionProof,
    pub o_proj_proof: SuccinctMatmulProof,
    pub g_proj_proof: SuccinctMatmulProof,
    pub sigmoid_gate_proof: SigmoidGateProof,
    pub residual1_proof: AddProof,
    pub norm2_proof: RmsNormProof,
    pub gate_proj_proof: SuccinctMatmulProof,
    pub up_proj_proof: SuccinctMatmulProof,
    pub swiglu_proof: SwiGluProof,
    pub down_proj_proof: SuccinctMatmulProof,
    pub residual2_proof: AddProof,
    // Weight commitments
    pub w_q_commitment: WeightCommitment,
    pub w_k_commitment: WeightCommitment,
    pub w_v_commitment: WeightCommitment,
    pub w_o_commitment: WeightCommitment,
    pub w_g_proj_commitment: WeightCommitment,
    pub w_gate_commitment: WeightCommitment,
    pub w_up_commitment: WeightCommitment,
    pub w_down_commitment: WeightCommitment,
}

/// Pre-computed weight commitments for a Qwen layer (deterministic from weights).
#[derive(Clone, Debug)]
pub struct QwenLayerCommitments {
    pub w_q: WeightCommitment,
    pub w_k: WeightCommitment,
    pub w_v: WeightCommitment,
    pub w_o: WeightCommitment,
    pub w_g_proj: WeightCommitment,
    pub w_gate: WeightCommitment,
    pub w_up: WeightCommitment,
    pub w_down: WeightCommitment,
}

/// Create placeholder commitments with zero roots.
///
/// SOUNDNESS (E6): zero-root placeholders break weight binding entirely — the
/// transcript would absorb a constant value regardless of the underlying weight
/// witness, so two distinct weight matrices would produce identical challenges.
/// This is ONLY safe inside tests where the verifier is colluding (i.e. is
/// the test harness itself). Reviewer flagged that the `pub` export leaked
/// this footgun to library consumers, so it's now `cfg(test)`-gated and
/// `pub(crate)`-scoped. Production proving must use `commit_qwen_layer`.
#[cfg(test)]
pub(crate) fn placeholder_qwen_commitments(weights: &QwenLayerWeights) -> QwenLayerCommitments {
    let placeholder = |w: &[F]| -> WeightCommitment {
        let n = w.len();
        let log_n = if n <= 1 { 1 } else { log2_ceil(n) };
        WeightCommitment {
            root: [0u8; 32],
            num_weights: n,
            log_height: log_n,
            kind: crate::proving::weight_commitment::WeightDigestKind::Blake3Fast,
        }
    };
    QwenLayerCommitments {
        w_q: placeholder(&weights.w_q),
        w_k: placeholder(&weights.w_k),
        w_v: placeholder(&weights.w_v),
        w_o: placeholder(&weights.w_o),
        w_g_proj: placeholder(&weights.w_g_proj),
        w_gate: placeholder(&weights.w_gate),
        w_up: placeholder(&weights.w_up),
        w_down: placeholder(&weights.w_down),
    }
}

/// Compute all 8 weight commitments for a Qwen layer.
pub fn commit_qwen_layer(weights: &QwenLayerWeights, config: &ModelConfig) -> QwenLayerCommitments {
    let d_model = config.d_model;
    let d_ff = config.d_ff;
    let q_dim = config.num_q_heads * config.d_head;
    let kv_dim = config.num_kv_heads * config.d_head;
    QwenLayerCommitments {
        w_q: commit_weight_matrix(&weights.w_q, q_dim, d_model),
        w_k: commit_weight_matrix(&weights.w_k, kv_dim, d_model),
        w_v: commit_weight_matrix(&weights.w_v, kv_dim, d_model),
        w_o: commit_weight_matrix(&weights.w_o, d_model, q_dim),
        w_g_proj: commit_weight_matrix(&weights.w_g_proj, q_dim, d_model),
        w_gate: commit_weight_matrix(&weights.w_gate, d_ff, d_model),
        w_up: commit_weight_matrix(&weights.w_up, d_ff, d_model),
        w_down: commit_weight_matrix(&weights.w_down, d_model, d_ff),
    }
}

/// All intermediate values from Qwen forward pass.
pub struct QwenForwardTrace {
    pub x: Vec<F>,
    pub norm1_x: Vec<F>,
    /// S5: signed perturbation applied to norm1 input (0 if no QR perturbation).
    pub norm1_delta: i32,
    pub norm1_out: Vec<F>,
    pub q: Vec<F>,
    pub k: Vec<F>,
    pub v: Vec<F>,
    pub attn_out: Vec<F>,
    #[allow(dead_code)]
    pub o_proj_out: Vec<F>,
    pub g_proj_out_raw: Vec<F>,   // raw matmul output
    pub g_proj_out: Vec<F>,       // requantized for sigmoid lookup
    pub g_proj_sigmoid: Vec<F>,   // sigmoid(g_proj_out)
    pub gated_out: Vec<F>,        // o_proj_out ⊙ sigmoid(g_proj_out)
    pub h: Vec<F>,                // x + gated_out
    pub norm2_x: Vec<F>,
    /// S5: signed perturbation applied to norm2 input.
    pub norm2_delta: i32,
    pub norm2_out: Vec<F>,
    pub gate_out_raw: Vec<F>,
    pub gate_out: Vec<F>,
    pub gate_silu: Vec<F>,
    pub up_out: Vec<F>,
    pub swiglu_out: Vec<F>,
    pub down_out: Vec<F>,
    pub output: Vec<F>,           // h + down_out
}

/// Compute Qwen forward pass for seq_len=1.
pub fn qwen_forward(
    x: &[F],
    weights: &QwenLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
    sigmoid_table: &LookupTable,
) -> QwenForwardTrace {
    let silu_index = build_table_index(silu_table);
    let sigmoid_index = build_table_index(sigmoid_table);
    qwen_forward_indexed(x, weights, config, silu_table, sigmoid_table, &silu_index, &sigmoid_index)
}

/// Compute Qwen forward pass with pre-built lookup indices (avoids 65536-entry HashMap rebuild per layer).
pub fn qwen_forward_indexed(
    x: &[F],
    weights: &QwenLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
    sigmoid_table: &LookupTable,
    silu_index: &TableIndex,
    sigmoid_index: &TableIndex,
) -> QwenForwardTrace {
    let d_model = config.d_model;
    let d_ff = config.d_ff;
    let num_q_heads = config.num_q_heads;
    let num_kv_heads = config.num_kv_heads;
    let d_head = config.d_head;

    // RMSNorm 1
    let (norm1_out, norm1_x, norm1_delta) = rmsnorm_forward(x, &weights.norm1_gamma);

    // QKV + g_proj projections (all from norm1_out, independent → parallel).
    // GDN may have asymmetric V (Qwen3.5-4B/9B): v_dim != kv_dim. Heuristic:
    // GQA full-attention has num_q_heads != num_kv_heads → output replicates V to q_dim.
    // GDN-style has num_q_heads == num_kv_heads → output is v_dim, gate/o_proj sized v_dim.
    let q_dim = num_q_heads * d_head;
    let k_dim = num_kv_heads * d_head;
    let v_dim = config.v_dim();
    let is_gqa_full_attn = num_q_heads != num_kv_heads;
    let attn_out_dim = if is_gqa_full_attn { q_dim } else { v_dim };
    let (q, (k, (v, g_proj_out_raw))) = rayon::join(
        || matmul_forward(&weights.w_q, &norm1_out, q_dim, d_model, None),
        || rayon::join(
            || matmul_forward(&weights.w_k, &norm1_out, k_dim, d_model, None),
            || rayon::join(
                || matmul_forward(&weights.w_v, &norm1_out, v_dim, d_model, None),
                || matmul_forward(&weights.w_g_proj, &norm1_out, attn_out_dim, d_model, None),
            ),
        ),
    );

    // Attention output at seq_len=1.
    // Full GQA: replicate V across query head groups → q_dim.
    // GDN: passthrough V (v_dim already matches gate / o_proj input).
    //
    // SOUNDNESS (S4 — known gap): at seq_len=1 with
    // empty history, the row-attention sumcheck is skipped (proof.row_proofs
    // is empty) and `attn_out` is computed as a deterministic function of `v`
    // — identity for GDN, head-replication for GQA. The DOWNSTREAM chain
    // (sigmoid_gate → o_proj → residual) constrains the value the prover
    // declares for attn_out, but the relationship `attn_out = f(v)` is NOT
    // proven inside the layer. A malicious prover can declare an arbitrary
    // attn_out and rebuild a self-consistent downstream chain; the verifier
    // would still accept because it never sees v independently — the v
    // matmul proof binds `v = W_v @ norm1_out`, but no sub-proof binds
    // `attn_out = identity(v)` (GDN) or `attn_out = replicate(v)` (GQA).
    //
    // For full GDN math, the seq_len=1 case should also enforce
    // `attn_out = β · (q · k) · v` per the delta-rule degenerate form;
    // that is queued as future work alongside the recurrent-state proof
    // (see Limitations in root README). Today the seq_len=1 step is
    // sound up to the layer's matmul + RMSNorm + gating + residual
    // structure; the missing sub-proof is the v→attn_out wiring.
    let attn_out = if is_gqa_full_attn {
        let heads_per_group = num_q_heads / num_kv_heads;
        let mut out = Vec::with_capacity(q_dim);
        for h in 0..num_q_heads {
            let kv_idx = h / heads_per_group;
            for d in 0..d_head {
                out.push(v[kv_idx * d_head + d]);
            }
        }
        out
    } else {
        v.clone()
    };

    // Output gate: sigmoid → gated = attn_out ⊙ sigmoid(g_proj_out)
    let g_proj_out = requantize_to_i16_field(&g_proj_out_raw, sigmoid_table);
    let g_proj_sigmoid: Vec<F> = g_proj_out.iter().map(|&v| lookup_fast(v.as_canonical_u32(), &sigmoid_index)).collect();

    // Gate applied to attn_out (attn_out_dim), BEFORE o_proj.
    let gated_attn: Vec<F> = attn_out.iter().zip(g_proj_sigmoid.iter())
        .map(|(&a, &b)| a * b).collect();

    // O projection on gated attention.
    let gated_out = matmul_forward(&weights.w_o, &gated_attn, d_model, attn_out_dim, None);

    // Residual 1
    let h: Vec<F> = x.iter().zip(gated_out.iter()).map(|(&a, &b)| a + b).collect();

    // RMSNorm 2
    let (norm2_out, norm2_x, norm2_delta) = rmsnorm_forward(&h, &weights.norm2_gamma);

    // Gate + Up projections (both from norm2_out, independent → parallel)
    let (gate_out_raw, up_out) = rayon::join(
        || matmul_forward(&weights.w_gate, &norm2_out, d_ff, d_model, None),
        || matmul_forward(&weights.w_up, &norm2_out, d_ff, d_model, None),
    );

    // Requantize gate for SiLU lookup
    let gate_out = requantize_to_i16_field(&gate_out_raw, silu_table);
    let gate_silu: Vec<F> = gate_out.iter().map(|&v| lookup_fast(v.as_canonical_u32(), &silu_index)).collect();
    let swiglu_out: Vec<F> = gate_silu.iter().zip(up_out.iter()).map(|(&a, &b)| a * b).collect();

    // Down projection
    let down_out = matmul_forward(&weights.w_down, &swiglu_out, d_model, d_ff, None);

    // Residual 2
    let output: Vec<F> = h.iter().zip(down_out.iter()).map(|(&a, &b)| a + b).collect();

    QwenForwardTrace {
        x: x.to_vec(), norm1_x, norm1_delta, norm1_out, q, k, v, attn_out,
        o_proj_out: vec![], g_proj_out_raw, g_proj_out, g_proj_sigmoid, gated_out,
        h, norm2_x, norm2_delta, norm2_out, gate_out_raw, gate_out, gate_silu,
        up_out, swiglu_out, down_out, output,
    }
}

/// Prove a Qwen3.5-style transformer layer with a pre-computed forward trace.
/// Avoids redundant forward pass when the trace is already available.
pub fn prove_qwen_layer_with_trace(
    trace: &QwenForwardTrace,
    weights: &QwenLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
    sigmoid_table: &LookupTable,
    transcript: &mut Transcript,
) -> QwenLayerProof {
    let d_model = config.d_model;
    let d_ff = config.d_ff;
    let q_dim = config.num_q_heads * config.d_head;
    let k_dim = config.num_kv_heads * config.d_head;
    let v_dim = config.v_dim();
    let is_gqa_full_attn = config.num_q_heads != config.num_kv_heads;
    let attn_out_dim = if is_gqa_full_attn { q_dim } else { v_dim };

    // S3: absorb dimensions FIRST (before any commitment or challenge), so
    // any prover/verifier disagreement on config diverges the transcript.
    absorb_qwen_dims(config, transcript);

    // Compute commitments first — needed for Fiat-Shamir binding (transcript
    // must absorb commitment roots before generating matmul challenges).
    let w_q_commitment = commit_weight_matrix(&weights.w_q, q_dim, d_model);
    let w_k_commitment = commit_weight_matrix(&weights.w_k, k_dim, d_model);
    let w_v_commitment = commit_weight_matrix(&weights.w_v, v_dim, d_model);
    let w_o_commitment = commit_weight_matrix(&weights.w_o, d_model, attn_out_dim);
    let w_g_proj_commitment = commit_weight_matrix(&weights.w_g_proj, attn_out_dim, d_model);
    let w_gate_commitment = commit_weight_matrix(&weights.w_gate, d_ff, d_model);
    let w_up_commitment = commit_weight_matrix(&weights.w_up, d_ff, d_model);
    let w_down_commitment = commit_weight_matrix(&weights.w_down, d_model, d_ff);

    // Absorb all weight commitments — Fiat-Shamir binding.
    transcript.absorb_bytes(&w_q_commitment.root);
    transcript.absorb_bytes(&w_k_commitment.root);
    transcript.absorb_bytes(&w_v_commitment.root);
    transcript.absorb_bytes(&w_o_commitment.root);
    transcript.absorb_bytes(&w_g_proj_commitment.root);
    transcript.absorb_bytes(&w_gate_commitment.root);
    transcript.absorb_bytes(&w_up_commitment.root);
    transcript.absorb_bytes(&w_down_commitment.root);

    {
        // 1. RMSNorm 1
        let norm1_proof = prove_rmsnorm(&trace.norm1_x, &weights.norm1_gamma, &trace.norm1_out, trace.norm1_delta, transcript);

        // 2. QKV matmuls (V uses v_dim; may differ from K_dim for asymmetric GDN)
        //
        // PERF (G3): at seq_len=1 the GDN path uses only
        // `trace.v` downstream (attn_out = v passthrough → sigmoid_gate →
        // o_proj). `trace.q` and `trace.k` are computed for consistency
        // with the trace structure but their values are not consumed by
        // any subsequent sub-proof, so the q_proof and k_proof here are
        // soundness-neutral: a malicious prover could declare any q/k
        // values without affecting the output proof. They cost ~2 matmul
        // proofs of overhead per layer. Two future paths:
        //   (a) Drop q_proof/k_proof at seq_len=1 (cheaper proofs).
        //   (b) Wire q/k into a recurrent-state-precomputation proof when
        //       the GDN delta-rule recurrence is added.
        // Both are tracked alongside the recurrent-state work.
        let q_proof = prove_matmul_succinct(&weights.w_q, &trace.norm1_out, &trace.q, q_dim, d_model, None, transcript);
        let k_proof = prove_matmul_succinct(&weights.w_k, &trace.norm1_out, &trace.k, k_dim, d_model, None, transcript);
        let v_proof = prove_matmul_succinct(&weights.w_v, &trace.norm1_out, &trace.v, v_dim, d_model, None, transcript);

        // 3. Attention (seq_len=1).
        //
        // SOUNDNESS (S4 closure / P10-3): row sumcheck
        // is empty at seq_len=1, but we now bind `attn_out ↔ v` via a
        // sub-proof: squeeze a fresh point `r_attn` over log2(attn_out_dim)
        // and `r_v` over log2(v_dim), commit `MLE(attn_out, r_attn)` and
        // `MLE(v, r_v)`. Verifier independently recomputes both from
        // canonical trace (audit mode) and asserts they match the prover's
        // claims. For GDN (q_heads == kv_heads, attn_out_dim == v_dim
        // and attn_out is identity of v) the verifier additionally
        // enforces `attn_out_at_r == v_at_r` at the same point — that's
        // the load-bearing Schwartz-Zippel binding that survives the
        // future true-ZK migration. See `proving/attention.rs`
        // `Seq1VConsistency` for the format.
        let log_attn = crate::field::common::log2_ceil(attn_out_dim);
        let r_attn = transcript.squeeze_many(log_attn);
        let attn_pad_size = 1usize << log_attn;
        let mut attn_pad = trace.attn_out.clone();
        attn_pad.resize(attn_pad_size, F::zero());
        let attn_out_at_r = crate::field::m31_ops::mle_evaluate(&attn_pad, &r_attn);

        // SOUNDNESS (P10-3 GQA fix — 3rd reviewer follow-up): for GDN
        // (attn_out_dim == v_dim, attn_out is identity of v) evaluate v
        // at the same `r_attn`. For GQA full-attn we need the correct
        // bit-coordinate slice. `mle_evaluate` folds MSB-first, so the
        // attn_out flat index `i = group * (heads_per_group * d_head) +
        // within_group * d_head + d` decomposes as r_attn = [r_g | r_w | r_d]
        // with len(r_g)=log_kv, len(r_w)=log(heads_per_group), len(r_d)=log_d.
        // Because attn_out[g, w, d] = v[g, d] doesn't depend on `w`, the
        // within-group bits sum out via Σ eq(r_w,·) = 1, giving
        // `MLE(attn_out, r_attn) = MLE(v, [r_g || r_d])`. So r_v is the
        // **concatenation of the group prefix and the d-head suffix** —
        // NOT the contiguous prefix of length log_v as a previous version
        // of this code mistakenly took.
        let r_v: Vec<F> = if !is_gqa_full_attn {
            r_attn.clone()
        } else {
            let log_d = crate::field::common::log2_ceil(config.d_head);
            let log_kv = crate::field::common::log2_ceil(config.num_kv_heads);
            let mut rv = Vec::with_capacity(log_kv + log_d);
            rv.extend_from_slice(&r_attn[..log_kv]);
            rv.extend_from_slice(&r_attn[r_attn.len() - log_d..]);
            rv
        };
        let v_pad_size = 1usize << crate::field::common::log2_ceil(v_dim);
        let mut v_pad = trace.v.clone();
        v_pad.resize(v_pad_size, F::zero());
        let v_at_r = crate::field::m31_ops::mle_evaluate(&v_pad, &r_v);

        let seq1_consistency = Some(crate::proving::attention::Seq1VConsistency {
            attn_out_at_r: attn_out_at_r.as_canonical_u32(),
            v_at_r: v_at_r.as_canonical_u32(),
            is_gqa_full_attn,
        });

        // Absorb the sub-proof claims so subsequent challenges depend on
        // the v↔attn_out binding (any tamper diverges the transcript).
        transcript.absorb(attn_out_at_r.as_canonical_u32());
        transcript.absorb(v_at_r.as_canonical_u32());

        let attn_proof = RowAttentionProof {
            row_proofs: vec![],
            num_heads: config.num_q_heads,
            seq_len: 1,
            d_head: config.d_head,
            seq1_consistency,
        };

        // 4. O projection: gated_out = W_o @ gated_attn (input dim = attn_out_dim)
        let gated_attn: Vec<F> = trace.attn_out.iter().zip(trace.g_proj_sigmoid.iter())
            .map(|(&a, &b)| a * b).collect();
        let o_proj_proof = prove_matmul_succinct(
            &weights.w_o, &gated_attn, &trace.gated_out,
            d_model, attn_out_dim, None, transcript,
        );

        // 5. g_proj matmul: g_proj_out_raw = W_g @ norm1_out (output dim = attn_out_dim)
        let g_proj_proof = prove_matmul_succinct(
            &weights.w_g_proj, &trace.norm1_out, &trace.g_proj_out_raw,
            attn_out_dim, d_model, None, transcript,
        );

        // 6. Sigmoid gate: gated_attn = attn_out ⊙ sigmoid(g_proj_out)
        let sigmoid_gate_proof = prove_sigmoid_gate(
            &trace.g_proj_out, &trace.g_proj_sigmoid, &trace.attn_out, &gated_attn,
            sigmoid_table, transcript,
        );

        // 7. Residual 1
        let log_d = log2_ceil(d_model);
        let res1_point = transcript.squeeze_many(log_d);
        let residual1_proof = prove_add(&trace.x, &trace.gated_out, &trace.h, &res1_point, transcript);

        // 8. RMSNorm 2
        let norm2_proof = prove_rmsnorm(&trace.norm2_x, &weights.norm2_gamma, &trace.norm2_out, trace.norm2_delta, transcript);

        // 9. Gate + Up projections
        let gate_proj_proof = prove_matmul_succinct(
            &weights.w_gate, &trace.norm2_out, &trace.gate_out_raw,
            d_ff, d_model, None, transcript,
        );
        let up_proj_proof = prove_matmul_succinct(
            &weights.w_up, &trace.norm2_out, &trace.up_out,
            d_ff, d_model, None, transcript,
        );

        // 10. SwiGLU
        let swiglu_proof = prove_swiglu(
            &trace.gate_out, &trace.gate_silu, &trace.up_out, &trace.swiglu_out,
            silu_table, transcript,
        );

        // 11. Down projection
        let down_proj_proof = prove_matmul_succinct(
            &weights.w_down, &trace.swiglu_out, &trace.down_out,
            d_model, d_ff, None, transcript,
        );

        // 12. Residual 2
        let res2_point = transcript.squeeze_many(log_d);
        let residual2_proof = prove_add(&trace.h, &trace.down_out, &trace.output, &res2_point, transcript);

        QwenLayerProof {
            norm1_proof, qkv_proofs: (q_proof, k_proof, v_proof),
            attn_proof, o_proj_proof, g_proj_proof, sigmoid_gate_proof,
            residual1_proof, norm2_proof, gate_proj_proof, up_proj_proof,
            swiglu_proof, down_proj_proof, residual2_proof,
            w_q_commitment, w_k_commitment, w_v_commitment, w_o_commitment,
            w_g_proj_commitment, w_gate_commitment, w_up_commitment, w_down_commitment,
        }
    }
}

/// Verify a Qwen3.5-style transformer layer proof.
pub fn verify_qwen_layer(
    proof: &QwenLayerProof,
    x: &[F],
    y: &[F],
    weights: &QwenLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
    sigmoid_table: &LookupTable,
    transcript: &mut Transcript,
) -> bool {
    let d_model = config.d_model;
    let d_ff = config.d_ff;
    let q_dim = config.num_q_heads * config.d_head;
    let k_dim = config.num_kv_heads * config.d_head;
    let v_dim = config.v_dim();
    let is_gqa_full_attn = config.num_q_heads != config.num_kv_heads;
    let attn_out_dim = if is_gqa_full_attn { q_dim } else { v_dim };

    let trace = qwen_forward(x, weights, config, silu_table, sigmoid_table);
    if trace.output != y {
        eprintln!("Qwen: output mismatch");
        return false;
    }

    // S3: absorb dimensions FIRST (mirror prover), so config-injection diverges
    // the transcript and verification fails.
    absorb_qwen_dims(config, transcript);

    // Absorb all weight commitments — must match prover absorption order.
    transcript.absorb_bytes(&proof.w_q_commitment.root);
    transcript.absorb_bytes(&proof.w_k_commitment.root);
    transcript.absorb_bytes(&proof.w_v_commitment.root);
    transcript.absorb_bytes(&proof.w_o_commitment.root);
    transcript.absorb_bytes(&proof.w_g_proj_commitment.root);
    transcript.absorb_bytes(&proof.w_gate_commitment.root);
    transcript.absorb_bytes(&proof.w_up_commitment.root);
    transcript.absorb_bytes(&proof.w_down_commitment.root);

    // 1. RMSNorm 1
    if !verify_rmsnorm(&proof.norm1_proof, &trace.norm1_out, d_model, transcript) {
        eprintln!("Qwen: RMSNorm 1 failed"); return false;
    }

    // 2. QKV matmuls (V uses v_dim for asymmetric GDN)
    let q_r = verify_matmul_succinct(&proof.qkv_proofs.0, &proof.w_q_commitment, &trace.q, q_dim, d_model, None, transcript);
    if !q_r.valid { eprintln!("Qwen: Q matmul failed"); return false; }
    let k_r = verify_matmul_succinct(&proof.qkv_proofs.1, &proof.w_k_commitment, &trace.k, k_dim, d_model, None, transcript);
    if !k_r.valid { eprintln!("Qwen: K matmul failed"); return false; }
    let v_r = verify_matmul_succinct(&proof.qkv_proofs.2, &proof.w_v_commitment, &trace.v, v_dim, d_model, None, transcript);
    if !v_r.valid { eprintln!("Qwen: V matmul failed"); return false; }

    // 3. Attention (seq_len=1: trivial — but P10-3 v↔attn_out binding).
    //
    // SOUNDNESS (S4 closure / P10-3): mirror the prover
    // at qwen.rs:393. Squeeze the same fresh point r, recompute MLE evals
    // from the verifier's canonical trace, assert match against the
    // prover's claims. For GDN-style (q == kv) additionally enforce the
    // identity check `attn_out_at_r == v_at_r` at the same r — that's
    // the proof-level binding that survives the future true-ZK migration.
    {
        let log_attn = log2_ceil(attn_out_dim);
        let r_attn = transcript.squeeze_many(log_attn);
        let attn_pad_size = 1usize << log_attn;
        let mut attn_pad = trace.attn_out.clone();
        attn_pad.resize(attn_pad_size, F::zero());
        let canonical_attn_at_r = crate::field::m31_ops::mle_evaluate(&attn_pad, &r_attn);

        let r_v: Vec<F> = if !is_gqa_full_attn {
            r_attn.clone()
        } else {
            // P10-3 GQA fix: group-prefix + d-head-suffix slice.
            // See prover-side comment in `prove_qwen_layer_*` for the full
            // derivation. Bug history: a previous version took the
            // contiguous prefix `r_attn[..log_v]`, which mixes group bits
            // with within-group bits and is NOT the correct MLE relation.
            let log_d = log2_ceil(config.d_head);
            let log_kv = log2_ceil(config.num_kv_heads);
            let mut rv = Vec::with_capacity(log_kv + log_d);
            rv.extend_from_slice(&r_attn[..log_kv]);
            rv.extend_from_slice(&r_attn[r_attn.len() - log_d..]);
            rv
        };
        let v_pad_size = 1usize << log2_ceil(v_dim);
        let mut v_pad = trace.v.clone();
        v_pad.resize(v_pad_size, F::zero());
        let canonical_v_at_r = crate::field::m31_ops::mle_evaluate(&v_pad, &r_v);

        match &proof.attn_proof.seq1_consistency {
            Some(s) => {
                if s.attn_out_at_r != canonical_attn_at_r.as_canonical_u32() {
                    eprintln!("Qwen: seq1 attn_out_at_r mismatch (P10-3 binding broken)");
                    return false;
                }
                if s.v_at_r != canonical_v_at_r.as_canonical_u32() {
                    eprintln!("Qwen: seq1 v_at_r mismatch (P10-3 binding broken)");
                    return false;
                }
                // GDN identity: attn_out is a copy of v, so MLE evals at
                // the same r MUST be equal. The Schwartz-Zippel binding
                // that survives true-ZK migration.
                if !s.is_gqa_full_attn && s.attn_out_at_r != s.v_at_r {
                    eprintln!("Qwen: seq1 GDN identity check failed (attn_out_at_r != v_at_r)");
                    return false;
                }
                // Mirror the prover's transcript absorption of the claims
                // so subsequent challenges depend on the binding.
                transcript.absorb(s.attn_out_at_r);
                transcript.absorb(s.v_at_r);
            }
            None => {
                eprintln!("Qwen: missing seq1_consistency at seq_len=1 (P10-3 required for )");
                return false;
            }
        }
    }

    // 4. O projection (input dim = attn_out_dim)
    let o_r = verify_matmul_succinct(&proof.o_proj_proof, &proof.w_o_commitment, &trace.gated_out, d_model, attn_out_dim, None, transcript);
    if !o_r.valid { eprintln!("Qwen: O proj failed"); return false; }

    // 5. g_proj matmul (output dim = attn_out_dim)
    let g_r = verify_matmul_succinct(&proof.g_proj_proof, &proof.w_g_proj_commitment, &trace.g_proj_out_raw, attn_out_dim, d_model, None, transcript);
    if !g_r.valid { eprintln!("Qwen: g_proj failed"); return false; }

    // 6. Sigmoid gate
    //
    // SOUNDNESS (S4 chain, reviewer follow-up):
    // `verify_sigmoid_gate` does NOT take a `&trace.*` argument — it only
    // checks the proof's internal commitments + sumcheck. By itself this
    // would NOT bind the prover's `attn_out` to the verifier's canonical
    // `trace.attn_out`. The S4 closure (audit-mode trace recomputation)
    // works only because the IMMEDIATELY-FOLLOWING o_proj verifier (call
    // site below at the matmul step) takes `&trace.gated_out` and runs
    // a sumcheck claim built from `mle(canonical_gated_out, r)`, which
    // forces the prover's gated_attn (= attn_out * sigmoid_gate) to
    // MLE-agree with `canonical_gated_attn` with overwhelming probability.
    //
    // INVARIANT: any future refactor that moves o_proj away from
    // immediately following sigmoid_gate, OR that drops the o_proj proof
    // entirely, MUST replace this implicit chain with an explicit
    // `&trace.attn_out` binding inside `verify_sigmoid_gate`. The
    // `test_qwen_seq_len_1_attn_out_tamper_rejected_via_canonical_trace`
    // regression catches accidental breakage by tampering attn_out and
    // asserting rejection — it fails the moment the chain breaks.
    if !verify_sigmoid_gate(&proof.sigmoid_gate_proof, attn_out_dim, transcript) {
        eprintln!("Qwen: sigmoid gate failed"); return false;
    }

    // 7. Residual 1
    let log_d = log2_ceil(d_model);
    let res1_point = transcript.squeeze_many(log_d);
    if !verify_add(&proof.residual1_proof, &res1_point, transcript) {
        eprintln!("Qwen: residual 1 failed"); return false;
    }

    // 8. RMSNorm 2
    if !verify_rmsnorm(&proof.norm2_proof, &trace.norm2_out, d_model, transcript) {
        eprintln!("Qwen: RMSNorm 2 failed"); return false;
    }

    // 9. Gate + Up
    let gate_r = verify_matmul_succinct(&proof.gate_proj_proof, &proof.w_gate_commitment, &trace.gate_out_raw, d_ff, d_model, None, transcript);
    if !gate_r.valid { eprintln!("Qwen: gate proj failed"); return false; }
    let up_r = verify_matmul_succinct(&proof.up_proj_proof, &proof.w_up_commitment, &trace.up_out, d_ff, d_model, None, transcript);
    if !up_r.valid { eprintln!("Qwen: up proj failed"); return false; }

    // 10. SwiGLU
    if !verify_swiglu(&proof.swiglu_proof, d_ff, transcript) {
        eprintln!("Qwen: SwiGLU failed"); return false;
    }

    // 11. Down projection
    let down_r = verify_matmul_succinct(&proof.down_proj_proof, &proof.w_down_commitment, &trace.down_out, d_model, d_ff, None, transcript);
    if !down_r.valid { eprintln!("Qwen: down proj failed"); return false; }

    // 12. Residual 2
    let res2_point = transcript.squeeze_many(log_d);
    if !verify_add(&proof.residual2_proof, &res2_point, transcript) {
        eprintln!("Qwen: residual 2 failed"); return false;
    }

    true
}

/// Proof for a Qwen3.5 layer with extension-field challenges (124-bit soundness).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QwenLayerProofEF {
    pub norm1_proof: RmsNormProofEF,
    pub qkv_proofs: (SuccinctMatmulProofEF, SuccinctMatmulProofEF, SuccinctMatmulProofEF),
    pub attn_proof: RowAttentionProof,
    pub o_proj_proof: SuccinctMatmulProofEF,
    pub g_proj_proof: SuccinctMatmulProofEF,
    pub sigmoid_gate_proof: SigmoidGateProofEF,
    pub residual1_proof: AddProofEF,
    pub norm2_proof: RmsNormProofEF,
    pub gate_proj_proof: SuccinctMatmulProofEF,
    pub up_proj_proof: SuccinctMatmulProofEF,
    pub swiglu_proof: SwiGluProofEF,
    pub down_proj_proof: SuccinctMatmulProofEF,
    pub residual2_proof: AddProofEF,
    pub w_q_commitment: WeightCommitment,
    pub w_k_commitment: WeightCommitment,
    pub w_v_commitment: WeightCommitment,
    pub w_o_commitment: WeightCommitment,
    pub w_g_proj_commitment: WeightCommitment,
    pub w_gate_commitment: WeightCommitment,
    pub w_up_commitment: WeightCommitment,
    pub w_down_commitment: WeightCommitment,
    // Note: batch_pcs_proof was removed — the verifier never checked it.
    // Individual matmul proofs use EF MLE eval (124-bit) or individual Basefold
    // for PCS binding. The batch was dead data adding ~100KB to pcs proofs.
}

impl QwenLayerProofEF {
    /// Strip all redundant Fiat-Shamir data from the proof.
    /// The verifier derives these from the transcript via `_with_challenges` verifiers.
    pub fn strip_redundant(&mut self) {
        // Matmul proofs: r_point, s_point, all sumcheck challenges
        let strip_matmul = |p: &mut SuccinctMatmulProofEF| {
            p.matmul_proof.r_point.clear();
            p.matmul_proof.s_point.clear();
            p.matmul_proof.sumcheck_proof.challenges.clear();
            p.w_eval_proof.eval_sumcheck.challenges.clear();
        };
        strip_matmul(&mut self.qkv_proofs.0);
        strip_matmul(&mut self.qkv_proofs.1);
        strip_matmul(&mut self.qkv_proofs.2);
        strip_matmul(&mut self.o_proj_proof);
        strip_matmul(&mut self.g_proj_proof);
        strip_matmul(&mut self.gate_proj_proof);
        strip_matmul(&mut self.up_proj_proof);
        strip_matmul(&mut self.down_proj_proof);

        // RmsNorm proofs: all sumcheck challenges + MLE eval challenges
        let strip_norm = |n: &mut RmsNormProofEF| {
            n.var_proof.challenges.clear();
            n.g_prod_proof.challenges.clear();
            n.g_triple_proof.challenges.clear();
            n.h_sq_proof.challenges.clear();
            n.g_sq_proof.challenges.clear();
            n.x_var_eval_proof.eval_sumcheck.challenges.clear();
            n.g_prod_eval_proof.eval_sumcheck.challenges.clear();
            n.gamma_triple_eval_proof.eval_sumcheck.challenges.clear();
            n.x_triple_eval_proof.eval_sumcheck.challenges.clear();
            n.h_sq_eval_proof.eval_sumcheck.challenges.clear();
            n.g_sq_eval_proof.eval_sumcheck.challenges.clear();
        };
        strip_norm(&mut self.norm1_proof);
        strip_norm(&mut self.norm2_proof);

        // SwiGlu/SigmoidGate hadamard + Add sumcheck challenges
        self.swiglu_proof.hadamard_proof.product_sumcheck.challenges.clear();
        self.sigmoid_gate_proof.hadamard_proof.product_sumcheck.challenges.clear();
        self.residual1_proof.a_sumcheck.challenges.clear();
        self.residual1_proof.b_sumcheck.challenges.clear();
        self.residual2_proof.a_sumcheck.challenges.clear();
        self.residual2_proof.b_sumcheck.challenges.clear();

        // EF LogUp: strip α/β (verifier re-derives), sumcheck challenges, inputs/outputs
        self.swiglu_proof.silu_proof.lookup_proof.sumcheck_proof.challenges.clear();
        self.swiglu_proof.silu_proof.lookup_proof.inputs.clear();
        self.swiglu_proof.silu_proof.lookup_proof.outputs.clear();
        self.sigmoid_gate_proof.sigmoid_proof.lookup_proof.sumcheck_proof.challenges.clear();
        self.sigmoid_gate_proof.sigmoid_proof.lookup_proof.inputs.clear();
        self.sigmoid_gate_proof.sigmoid_proof.lookup_proof.outputs.clear();
    }
}

/// Prove a Qwen3.5 layer with pre-computed commitments and extension-field challenges.
pub fn prove_qwen_layer_precommitted_ef(
    trace: &QwenForwardTrace,
    weights: &QwenLayerWeights,
    config: &ModelConfig,
    commitments: QwenLayerCommitments,
    silu_table: &LookupTable,
    sigmoid_table: &LookupTable,
    transcript: &mut Transcript,
) -> QwenLayerProofEF {
    let d_model = config.d_model;
    let d_ff = config.d_ff;
    let q_dim = config.num_q_heads * config.d_head;
    let k_dim = config.num_kv_heads * config.d_head;
    let v_dim = config.v_dim();
    let is_gqa_full_attn = config.num_q_heads != config.num_kv_heads;
    let attn_out_dim = if is_gqa_full_attn { q_dim } else { v_dim };

    // S3: absorb dimensions FIRST (config-injection defense).
    absorb_qwen_dims(config, transcript);

    // Absorb all weight commitments before proving — Fiat-Shamir binding.
    transcript.absorb_bytes(&commitments.w_q.root);
    transcript.absorb_bytes(&commitments.w_k.root);
    transcript.absorb_bytes(&commitments.w_v.root);
    transcript.absorb_bytes(&commitments.w_o.root);
    transcript.absorb_bytes(&commitments.w_g_proj.root);
    transcript.absorb_bytes(&commitments.w_gate.root);
    transcript.absorb_bytes(&commitments.w_up.root);
    transcript.absorb_bytes(&commitments.w_down.root);

    // ── Prove all sub-proofs with EF challenges ──
    // prove_matmul_succinct_ef calls bind_weights_ef internally, which dispatches
    // to individual Basefold (pcs/pcs-full) or MLE eval (default) per matmul.
    let norm1_proof = prove_rmsnorm_ef(&trace.norm1_x, &weights.norm1_gamma, &trace.norm1_out, trace.norm1_delta, transcript);
    let q_proof = prove_matmul_succinct_ef(&weights.w_q, &trace.norm1_out, &trace.q, q_dim, d_model, None, transcript);
    let k_proof = prove_matmul_succinct_ef(&weights.w_k, &trace.norm1_out, &trace.k, k_dim, d_model, None, transcript);
    let v_proof = prove_matmul_succinct_ef(&weights.w_v, &trace.norm1_out, &trace.v, v_dim, d_model, None, transcript);

    // P10-3 (S4 closure, EF path): seq_len=1 v↔attn_out consistency.
    // Mirrors the base path above; uses base-field squeeze for r so the
    // verifier can recompute MLE evals deterministically. The base-field
    // log_attn-coordinate r is sufficient — the EF challenge stream is
    // separately squeezed for the matmul sumchecks.
    let attn_proof = {
        let log_attn = crate::field::common::log2_ceil(attn_out_dim);
        let r_attn = transcript.squeeze_many(log_attn);
        let attn_pad_size = 1usize << log_attn;
        let mut attn_pad = trace.attn_out.clone();
        attn_pad.resize(attn_pad_size, F::zero());
        let attn_out_at_r = crate::field::m31_ops::mle_evaluate(&attn_pad, &r_attn);

        let r_v: Vec<F> = if !is_gqa_full_attn {
            r_attn.clone()
        } else {
            // Same group-prefix + d-suffix slice as the base path above.
            let log_d = crate::field::common::log2_ceil(config.d_head);
            let log_kv = crate::field::common::log2_ceil(config.num_kv_heads);
            let mut rv = Vec::with_capacity(log_kv + log_d);
            rv.extend_from_slice(&r_attn[..log_kv]);
            rv.extend_from_slice(&r_attn[r_attn.len() - log_d..]);
            rv
        };
        let v_pad_size = 1usize << crate::field::common::log2_ceil(v_dim);
        let mut v_pad = trace.v.clone();
        v_pad.resize(v_pad_size, F::zero());
        let v_at_r = crate::field::m31_ops::mle_evaluate(&v_pad, &r_v);

        let seq1_consistency = Some(crate::proving::attention::Seq1VConsistency {
            attn_out_at_r: attn_out_at_r.as_canonical_u32(),
            v_at_r: v_at_r.as_canonical_u32(),
            is_gqa_full_attn,
        });
        transcript.absorb(attn_out_at_r.as_canonical_u32());
        transcript.absorb(v_at_r.as_canonical_u32());
        RowAttentionProof {
            row_proofs: vec![],
            num_heads: config.num_q_heads,
            seq_len: 1,
            d_head: config.d_head,
            seq1_consistency,
        }
    };
    let gated_attn: Vec<F> = trace.attn_out.iter().zip(trace.g_proj_sigmoid.iter()).map(|(&a, &b)| a * b).collect();
    let o_proj_proof = prove_matmul_succinct_ef(&weights.w_o, &gated_attn, &trace.gated_out, d_model, attn_out_dim, None, transcript);
    let g_proj_proof = prove_matmul_succinct_ef(&weights.w_g_proj, &trace.norm1_out, &trace.g_proj_out_raw, attn_out_dim, d_model, None, transcript);
    let sigmoid_gate_proof = prove_sigmoid_gate_ef(&trace.g_proj_out, &trace.g_proj_sigmoid, &trace.attn_out, &gated_attn, sigmoid_table, transcript);
    let log_d = log2_ceil(d_model);
    let res1_point = transcript.squeeze_ef_many(log_d);
    let residual1_proof = prove_add_ef(&trace.x, &trace.gated_out, &trace.h, &res1_point, transcript);
    let norm2_proof = prove_rmsnorm_ef(&trace.norm2_x, &weights.norm2_gamma, &trace.norm2_out, trace.norm2_delta, transcript);
    let gate_proj_proof = prove_matmul_succinct_ef(&weights.w_gate, &trace.norm2_out, &trace.gate_out_raw, d_ff, d_model, None, transcript);
    let up_proj_proof = prove_matmul_succinct_ef(&weights.w_up, &trace.norm2_out, &trace.up_out, d_ff, d_model, None, transcript);
    let swiglu_proof = prove_swiglu_ef(&trace.gate_out, &trace.gate_silu, &trace.up_out, &trace.swiglu_out, silu_table, transcript);
    let down_proj_proof = prove_matmul_succinct_ef(&weights.w_down, &trace.swiglu_out, &trace.down_out, d_model, d_ff, None, transcript);
    let res2_point = transcript.squeeze_ef_many(log_d);
    let residual2_proof = prove_add_ef(&trace.h, &trace.down_out, &trace.output, &res2_point, transcript);

    // Strip lookup data (verifier has trace data via _with_data verifiers)
    let mut swiglu_proof = swiglu_proof;
    swiglu_proof.silu_proof.lookup_proof.inputs.clear();
    swiglu_proof.silu_proof.lookup_proof.outputs.clear();
    let mut sigmoid_gate_proof = sigmoid_gate_proof;
    sigmoid_gate_proof.sigmoid_proof.lookup_proof.inputs.clear();
    sigmoid_gate_proof.sigmoid_proof.lookup_proof.outputs.clear();

    let mut proof = QwenLayerProofEF {
        norm1_proof, qkv_proofs: (q_proof, k_proof, v_proof),
        attn_proof, o_proj_proof, g_proj_proof, sigmoid_gate_proof,
        residual1_proof, norm2_proof, gate_proj_proof, up_proj_proof,
        swiglu_proof, down_proj_proof, residual2_proof,
        w_q_commitment: commitments.w_q, w_k_commitment: commitments.w_k,
        w_v_commitment: commitments.w_v, w_o_commitment: commitments.w_o,
        w_g_proj_commitment: commitments.w_g_proj, w_gate_commitment: commitments.w_gate,
        w_up_commitment: commitments.w_up, w_down_commitment: commitments.w_down,
    };
    proof.strip_redundant();
    proof
}

/// Verify a Qwen3.5-style transformer layer proof with extension-field challenges.
pub fn verify_qwen_layer_ef(
    proof: &QwenLayerProofEF,
    x: &[F],
    y: &[F],
    weights: &QwenLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
    sigmoid_table: &LookupTable,
    transcript: &mut Transcript,
) -> bool {
    let silu_index = build_table_index(silu_table);
    let sigmoid_index = build_table_index(sigmoid_table);
    verify_qwen_layer_ef_indexed(proof, x, y, weights, config, silu_table, sigmoid_table,
        &silu_index, &sigmoid_index, transcript)
}

/// Verify with pre-built table indexes (avoids 65K HashMap rebuild per layer).
pub fn verify_qwen_layer_ef_indexed(
    proof: &QwenLayerProofEF,
    x: &[F],
    y: &[F],
    weights: &QwenLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
    sigmoid_table: &LookupTable,
    silu_index: &TableIndex,
    sigmoid_index: &TableIndex,
    transcript: &mut Transcript,
) -> bool {
    let d_model = config.d_model;
    let d_ff = config.d_ff;
    let q_dim = config.num_q_heads * config.d_head;
    let k_dim = config.num_kv_heads * config.d_head;
    let v_dim = config.v_dim();
    let is_gqa_full_attn = config.num_q_heads != config.num_kv_heads;
    let attn_out_dim = if is_gqa_full_attn { q_dim } else { v_dim };

    let trace = qwen_forward_indexed(x, weights, config, silu_table, sigmoid_table, silu_index, sigmoid_index);
    if trace.output != y {
        eprintln!("Qwen EF: output mismatch");
        return false;
    }

    // S3: absorb dimensions FIRST (mirror prover, config-injection defense).
    absorb_qwen_dims(config, transcript);

    // Absorb all weight commitments — must match prover absorption order.
    transcript.absorb_bytes(&proof.w_q_commitment.root);
    transcript.absorb_bytes(&proof.w_k_commitment.root);
    transcript.absorb_bytes(&proof.w_v_commitment.root);
    transcript.absorb_bytes(&proof.w_o_commitment.root);
    transcript.absorb_bytes(&proof.w_g_proj_commitment.root);
    transcript.absorb_bytes(&proof.w_gate_commitment.root);
    transcript.absorb_bytes(&proof.w_up_commitment.root);
    transcript.absorb_bytes(&proof.w_down_commitment.root);

    // 1. RMSNorm 1
    if !verify_rmsnorm_ef(&proof.norm1_proof, &trace.norm1_out, d_model, transcript) {
        eprintln!("Qwen EF: RMSNorm 1 failed"); return false;
    }

    // 2. QKV matmuls (V uses v_dim for asymmetric GDN)
    let q_r = verify_matmul_succinct_ef(&proof.qkv_proofs.0, &proof.w_q_commitment, &trace.q, q_dim, d_model, None, transcript);
    if !q_r.valid { eprintln!("Qwen EF: Q matmul failed"); return false; }
    let k_r = verify_matmul_succinct_ef(&proof.qkv_proofs.1, &proof.w_k_commitment, &trace.k, k_dim, d_model, None, transcript);
    if !k_r.valid { eprintln!("Qwen EF: K matmul failed"); return false; }
    let v_r = verify_matmul_succinct_ef(&proof.qkv_proofs.2, &proof.w_v_commitment, &trace.v, v_dim, d_model, None, transcript);
    if !v_r.valid { eprintln!("Qwen EF: V matmul failed"); return false; }

    // 3. Attention (seq_len=1: trivial — but P10-3 v↔attn_out binding).
    // Mirror prover's seq1 consistency block. Same logic as base path.
    {
        let log_attn = log2_ceil(attn_out_dim);
        let r_attn = transcript.squeeze_many(log_attn);
        let attn_pad_size = 1usize << log_attn;
        let mut attn_pad = trace.attn_out.clone();
        attn_pad.resize(attn_pad_size, F::zero());
        let canonical_attn_at_r = crate::field::m31_ops::mle_evaluate(&attn_pad, &r_attn);

        let r_v: Vec<F> = if !is_gqa_full_attn {
            r_attn.clone()
        } else {
            // P10-3 GQA fix: group-prefix + d-head-suffix slice.
            // See prover-side comment in `prove_qwen_layer_*` for the full
            // derivation. Bug history: a previous version took the
            // contiguous prefix `r_attn[..log_v]`, which mixes group bits
            // with within-group bits and is NOT the correct MLE relation.
            let log_d = log2_ceil(config.d_head);
            let log_kv = log2_ceil(config.num_kv_heads);
            let mut rv = Vec::with_capacity(log_kv + log_d);
            rv.extend_from_slice(&r_attn[..log_kv]);
            rv.extend_from_slice(&r_attn[r_attn.len() - log_d..]);
            rv
        };
        let v_pad_size = 1usize << log2_ceil(v_dim);
        let mut v_pad = trace.v.clone();
        v_pad.resize(v_pad_size, F::zero());
        let canonical_v_at_r = crate::field::m31_ops::mle_evaluate(&v_pad, &r_v);

        match &proof.attn_proof.seq1_consistency {
            Some(s) => {
                if s.attn_out_at_r != canonical_attn_at_r.as_canonical_u32() {
                    eprintln!("Qwen EF: seq1 attn_out_at_r mismatch (P10-3 binding broken)");
                    return false;
                }
                if s.v_at_r != canonical_v_at_r.as_canonical_u32() {
                    eprintln!("Qwen EF: seq1 v_at_r mismatch (P10-3 binding broken)");
                    return false;
                }
                if !s.is_gqa_full_attn && s.attn_out_at_r != s.v_at_r {
                    eprintln!("Qwen EF: seq1 GDN identity check failed");
                    return false;
                }
                transcript.absorb(s.attn_out_at_r);
                transcript.absorb(s.v_at_r);
            }
            None => {
                eprintln!("Qwen EF: missing seq1_consistency at seq_len=1");
                return false;
            }
        }
    }

    // 4. O projection (input dim = attn_out_dim)
    let o_r = verify_matmul_succinct_ef(&proof.o_proj_proof, &proof.w_o_commitment, &trace.gated_out, d_model, attn_out_dim, None, transcript);
    if !o_r.valid { eprintln!("Qwen EF: O proj failed"); return false; }

    // 5. g_proj matmul (output dim = attn_out_dim)
    let g_r = verify_matmul_succinct_ef(&proof.g_proj_proof, &proof.w_g_proj_commitment, &trace.g_proj_out_raw, attn_out_dim, d_model, None, transcript);
    if !g_r.valid { eprintln!("Qwen EF: g_proj failed"); return false; }

    // 6. Sigmoid gate — pass trace data for transcript binding
    {
        use crate::proving::sigmoid_gate::verify_sigmoid_gate_ef_with_data;
        let sig_inputs: Vec<u32> = trace.g_proj_out.iter().map(|v| v.as_canonical_u32()).collect();
        let sig_outputs: Vec<u32> = trace.g_proj_sigmoid.iter().map(|v| v.as_canonical_u32()).collect();
        if !verify_sigmoid_gate_ef_with_data(&proof.sigmoid_gate_proof, attn_out_dim,
            Some((&sig_inputs, &sig_outputs)), transcript) {
            eprintln!("Qwen EF: sigmoid gate failed"); return false;
        }
    }

    // 7. Residual 1
    let log_d = log2_ceil(d_model);
    let res1_point = transcript.squeeze_ef_many(log_d);
    if !verify_add_ef(&proof.residual1_proof, &res1_point, transcript) {
        eprintln!("Qwen EF: residual 1 failed"); return false;
    }

    // 8. RMSNorm 2
    if !verify_rmsnorm_ef(&proof.norm2_proof, &trace.norm2_out, d_model, transcript) {
        eprintln!("Qwen EF: RMSNorm 2 failed"); return false;
    }

    // 9. Gate + Up
    let gate_r = verify_matmul_succinct_ef(&proof.gate_proj_proof, &proof.w_gate_commitment, &trace.gate_out_raw, d_ff, d_model, None, transcript);
    if !gate_r.valid { eprintln!("Qwen EF: gate proj failed"); return false; }
    let up_r = verify_matmul_succinct_ef(&proof.up_proj_proof, &proof.w_up_commitment, &trace.up_out, d_ff, d_model, None, transcript);
    if !up_r.valid { eprintln!("Qwen EF: up proj failed"); return false; }

    // 10. SwiGLU — pass trace data for SiLU transcript binding
    {
        use crate::proving::swiglu::verify_swiglu_ef_with_data;
        let silu_inputs: Vec<u32> = trace.gate_out.iter().map(|v| v.as_canonical_u32()).collect();
        let silu_outputs: Vec<u32> = trace.gate_silu.iter().map(|v| v.as_canonical_u32()).collect();
        if !verify_swiglu_ef_with_data(&proof.swiglu_proof, d_ff,
            Some((&silu_inputs, &silu_outputs)), transcript) {
            eprintln!("Qwen EF: SwiGLU failed"); return false;
        }
    }

    // 11. Down projection
    let down_r = verify_matmul_succinct_ef(&proof.down_proj_proof, &proof.w_down_commitment, &trace.down_out, d_model, d_ff, None, transcript);
    if !down_r.valid { eprintln!("Qwen EF: down proj failed"); return false; }

    // 12. Residual 2
    let res2_point = transcript.squeeze_ef_many(log_d);
    if !verify_add_ef(&proof.residual2_proof, &res2_point, transcript) {
        eprintln!("Qwen EF: residual 2 failed"); return false;
    }

    true
}

/// Test-only helpers for building Qwen layer fixtures (weights, lookup tables,
/// valid inputs, placeholder commitments). Public so the Phase 11 integration
/// tests under `tests/` can reuse them without duplicating the small-scale
/// setup. NOT for production use — `make_qwen_weights` returns deterministic
/// identity matrices, `build_small_*_table` produce 256-entry tables (vs the
/// production 65536), and `find_valid_qwen_input` is a brute-force 500-offset
/// search.
#[doc(hidden)]
pub mod test_utils {
    use super::*;

    /// Re-export of the (cfg(test)) `placeholder_qwen_commitments` helper for
    /// integration tests. SOUNDNESS (E6): placeholder commitments break weight
    /// binding entirely (zero roots) — only safe when the verifier is the test
    /// harness itself, which is the case for everything in this module.
    pub fn placeholder_qwen_commitments_for_test(
        weights: &QwenLayerWeights,
    ) -> QwenLayerCommitments {
        let log2_ceil = crate::field::common::log2_ceil;
        let placeholder = |w: &[F]| -> WeightCommitment {
            let n = w.len();
            let log_n = if n <= 1 { 1 } else { log2_ceil(n) };
            WeightCommitment {
                root: [0u8; 32],
                num_weights: n,
                log_height: log_n,
                kind: crate::proving::weight_commitment::WeightDigestKind::Blake3Fast,
            }
        };
        QwenLayerCommitments {
            w_q: placeholder(&weights.w_q),
            w_k: placeholder(&weights.w_k),
            w_v: placeholder(&weights.w_v),
            w_o: placeholder(&weights.w_o),
            w_g_proj: placeholder(&weights.w_g_proj),
            w_gate: placeholder(&weights.w_gate),
            w_up: placeholder(&weights.w_up),
            w_down: placeholder(&weights.w_down),
        }
    }
    use crate::field::common::{i16_to_field, quantize_i16, is_qr_m31 as is_qr_m31_common};
    use crate::proving::weight_commitment::WeightCommitment;
    use p3_field::{AbstractField, Field};

    /// Build a 256-entry SiLU lookup table (vs production's 65536-entry full
    /// INT16 domain). Sufficient for d_model ≤ 16 test fixtures.
    pub fn build_small_silu_table(scale: i32) -> LookupTable {
        let s = scale as f64;
        let mut entries = Vec::with_capacity(256);
        for raw in 0u32..256 {
            let input_i16 = raw as i8 as i16;
            let x = input_i16 as f64 / s;
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            let y = x * sigmoid;
            let output_i16 = quantize_i16(y, s);
            entries.push((
                i16_to_field(input_i16).as_canonical_u32(),
                i16_to_field(output_i16).as_canonical_u32(),
            ));
        }
        LookupTable {
            name: "silu_small".to_string(),
            entries,
            commitment: WeightCommitment {
                root: [0u8; 32], num_weights: 256, log_height: 8,
                kind: crate::proving::weight_commitment::WeightDigestKind::Blake3Fast,
            },
        }
    }

    /// Build a 256-entry sigmoid lookup table.
    pub fn build_small_sigmoid_table(scale: i32) -> LookupTable {
        let s = scale as f64;
        let mut entries = Vec::with_capacity(256);
        for raw in 0u32..256 {
            let input_i16 = raw as i8 as i16;
            let x = input_i16 as f64 / s;
            let y = 1.0 / (1.0 + (-x).exp());
            let output_i16 = quantize_i16(y, s);
            entries.push((
                i16_to_field(input_i16).as_canonical_u32(),
                i16_to_field(output_i16).as_canonical_u32(),
            ));
        }
        LookupTable {
            name: "sigmoid_small".to_string(),
            entries,
            commitment: WeightCommitment {
                root: [0u8; 32], num_weights: 256, log_height: 8,
                kind: crate::proving::weight_commitment::WeightDigestKind::Blake3Fast,
            },
        }
    }

    /// Build symmetric Qwen weights (q_dim == kv_dim == v_dim case).
    pub fn make_qwen_weights(config: &ModelConfig) -> QwenLayerWeights {
        let d = config.d_model;
        let d_ff = config.d_ff;
        let q_dim = config.num_q_heads * config.d_head;
        let kv_dim = config.num_kv_heads * config.d_head;

        QwenLayerWeights {
            norm1_gamma: vec![F::one(); d],
            w_q: {
                let mut w = vec![F::zero(); q_dim * d];
                for i in 0..q_dim.min(d) { w[i * d + i] = F::one(); }
                w
            },
            w_k: {
                let mut w = vec![F::zero(); kv_dim * d];
                for i in 0..kv_dim.min(d) { w[i * d + i] = F::one(); }
                w
            },
            w_v: {
                let mut w = vec![F::zero(); kv_dim * d];
                for i in 0..kv_dim.min(d) { w[i * d + i] = F::one(); }
                w
            },
            w_o: {
                let mut w = vec![F::zero(); d * q_dim];
                for i in 0..d.min(q_dim) { w[i * q_dim + i] = F::one(); }
                w
            },
            w_g_proj: vec![F::zero(); q_dim * d], // zero so sigmoid input = 0
            norm2_gamma: vec![F::one(); d],
            w_gate: vec![F::zero(); d_ff * d],
            w_up: vec![F::zero(); d_ff * d],
            w_down: vec![F::zero(); d * d_ff],
        }
    }

    /// Build asymmetric-V Qwen weights: w_v sized v_dim*d_model (NOT
    /// kv_dim*d_model); w_o and w_g_proj sized by attn_out_dim.
    pub fn make_asymmetric_qwen_weights(config: &ModelConfig) -> QwenLayerWeights {
        let d = config.d_model;
        let d_ff = config.d_ff;
        let q_dim = config.num_q_heads * config.d_head;
        let kv_dim = config.num_kv_heads * config.d_head;
        let v_dim = config.v_dim();
        let is_gqa_full_attn = config.num_q_heads != config.num_kv_heads;
        let attn_out_dim = if is_gqa_full_attn { q_dim } else { v_dim };

        let mk_identity = |out: usize, inn: usize| -> Vec<F> {
            let mut w = vec![F::zero(); out * inn];
            for i in 0..out.min(inn) { w[i * inn + i] = F::one(); }
            w
        };
        QwenLayerWeights {
            norm1_gamma: vec![F::one(); d],
            w_q:        mk_identity(q_dim, d),
            w_k:        mk_identity(kv_dim, d),
            w_v:        mk_identity(v_dim, d),
            w_o:        mk_identity(d, attn_out_dim),
            w_g_proj:   vec![F::zero(); attn_out_dim * d],
            norm2_gamma: vec![F::one(); d],
            w_gate:     vec![F::zero(); d_ff * d],
            w_up:       vec![F::zero(); d_ff * d],
            w_down:     vec![F::zero(); d * d_ff],
        }
    }

    /// Brute-force search for an input vector that survives both RMSNorm
    /// QR-perturbation rejections (norm1 and norm2 sum_sq^-1 must both be
    /// quadratic residues mod M31). Panics after 500 offsets.
    pub fn find_valid_qwen_input(
        config: &ModelConfig,
        weights: &QwenLayerWeights,
        silu_table: &LookupTable,
        sigmoid_table: &LookupTable,
    ) -> Vec<F> {
        for offset in 0u32..500 {
            let candidate: Vec<F> = (0..config.d_model)
                .map(|i| F::from_canonical_u32(i as u32 + 1 + offset))
                .collect();

            let sum_sq: F = candidate.iter().map(|&v| v * v).sum();
            if sum_sq == F::zero() { continue; }
            let target = F::from_canonical_u32(config.d_model as u32) * sum_sq.inverse();
            if !is_qr_m31_common(target) { continue; }

            let trace = qwen_forward(&candidate, weights, config, silu_table, sigmoid_table);

            let sum_sq2: F = trace.h.iter().map(|&v| v * v).sum();
            if sum_sq2 == F::zero() { continue; }
            let target2 = F::from_canonical_u32(config.d_model as u32) * sum_sq2.inverse();
            if !is_qr_m31_common(target2) { continue; }

            return candidate;
        }
        panic!("Could not find valid Qwen input");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_utils::{
        build_small_silu_table, build_small_sigmoid_table,
        make_qwen_weights, make_asymmetric_qwen_weights, find_valid_qwen_input,
    };
    use p3_field::AbstractField;

    /// SOUNDNESS regression (S3): absorbing distinct ModelConfig dimensions onto
    /// otherwise-identical transcripts must diverge the squeezed challenge.
    /// Confirms config-injection cannot pass undetected through the dim-bind step.
    #[test]
    fn test_absorb_qwen_dims_diverges_on_config_change() {
        let make_config = |num_q_heads, num_kv_heads, d_head, v_num_heads, v_d_head| ModelConfig {
            d_model: 4096,
            d_ff: 12288,
            num_q_heads,
            num_kv_heads,
            d_head,
            n_layers: 1,
            vocab_size: 0,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads,
            v_d_head,
        };

        // Baseline: Qwen3.5-9B GDN layer config (asymmetric V).
        let cfg_a = make_config(16, 16, 128, 32, 128);

        // Each variant flips ONE dimension that must be transcript-bound.
        let variants = vec![
            ("d_model", ModelConfig { d_model: 4097, ..make_config(16, 16, 128, 32, 128) }),
            ("d_ff", ModelConfig { d_ff: 12289, ..make_config(16, 16, 128, 32, 128) }),
            ("num_q_heads", make_config(17, 16, 128, 32, 128)),
            ("num_kv_heads", make_config(16, 15, 128, 32, 128)),
            ("d_head", make_config(16, 16, 129, 32, 128)),
            ("v_num_heads", make_config(16, 16, 128, 33, 128)),
            ("v_d_head", make_config(16, 16, 128, 32, 129)),
            // Heuristic-flip: GQA full-attn vs GDN. num_q_heads=16, num_kv_heads=8
            // changes is_gqa_full_attn from false → true and attn_out_dim accordingly.
            ("is_gqa_heuristic", make_config(16, 8, 128, 32, 128)),
        ];

        let mut t_a = Transcript::new(b"qwen-dim-test");
        absorb_qwen_dims(&cfg_a, &mut t_a);
        let challenge_a = t_a.squeeze();

        for (label, cfg_b) in variants {
            let mut t_b = Transcript::new(b"qwen-dim-test");
            absorb_qwen_dims(&cfg_b, &mut t_b);
            let challenge_b = t_b.squeeze();
            assert_ne!(
                challenge_a, challenge_b,
                "config-injection on {} did not diverge transcript — S3 binding failed",
                label
            );
        }
    }

    // Test fixture helpers (build_small_silu_table, build_small_sigmoid_table,
    // make_qwen_weights, find_valid_qwen_input, make_asymmetric_qwen_weights)
    // moved to super::test_utils so the Phase 11 integration tests can reuse
    // them. They're imported via `use super::test_utils::{...}` above.

    #[test]
    fn test_qwen_layer_ef_basic() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 2,
            num_kv_heads: 2,
            d_head: 4,
            n_layers: 1,
            vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,
        };
        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_qwen_weights(&config);
        let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);

        let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);
        let commitments = placeholder_qwen_commitments(&weights);

        // Prove with EF
        let mut pt = Transcript::new(b"qwen-ef-test");
        let proof = prove_qwen_layer_precommitted_ef(
            &trace, &weights, &config, commitments, &silu_table, &sigmoid_table, &mut pt,
        );

        // Verify with EF
        let mut vt = Transcript::new(b"qwen-ef-test");
        assert!(
            verify_qwen_layer_ef(
                &proof, &x, &trace.output, &weights, &config,
                &silu_table, &sigmoid_table, &mut vt,
            ),
            "Qwen EF layer verification should pass"
        );
    }

    #[test]
    fn test_qwen_layer_ef_tampered() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 2,
            num_kv_heads: 2,
            d_head: 4,
            n_layers: 1,
            vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,
        };
        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_qwen_weights(&config);
        let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);

        let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);
        let commitments = placeholder_qwen_commitments(&weights);

        // Prove with correct trace
        let mut pt = Transcript::new(b"qwen-ef-tamper");
        let proof = prove_qwen_layer_precommitted_ef(
            &trace, &weights, &config, commitments, &silu_table, &sigmoid_table, &mut pt,
        );

        // Tamper with output
        let mut tampered = trace.output.clone();
        tampered[0] = tampered[0] + F::one();

        // Verify with tampered output should fail
        let mut vt = Transcript::new(b"qwen-ef-tamper");
        assert!(
            !verify_qwen_layer_ef(
                &proof, &x, &tampered, &weights, &config,
                &silu_table, &sigmoid_table, &mut vt,
            ),
            "Qwen EF should reject tampered output"
        );
    }

    #[test]
    fn test_qwen_layer_base_field_basic() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 2,
            num_kv_heads: 2,
            d_head: 4,
            n_layers: 1,
            vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,
        };
        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_qwen_weights(&config);
        let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);

        let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);

        // Prove with base-field challenges
        let mut pt = Transcript::new(b"qwen-base-test");
        let proof = prove_qwen_layer_with_trace(
            &trace, &weights, &config, &silu_table, &sigmoid_table, &mut pt,
        );

        // Verify with base-field challenges
        let mut vt = Transcript::new(b"qwen-base-test");
        assert!(
            verify_qwen_layer(
                &proof, &x, &trace.output, &weights, &config,
                &silu_table, &sigmoid_table, &mut vt,
            ),
            "Qwen base-field layer verification should pass"
        );
    }

    // make_asymmetric_qwen_weights moved to super::test_utils.

    /// SOUNDNESS regression (T1, Phase 5): asymmetric V layer (Qwen3.5-4B/9B
    /// shape). q_dim and kv_dim match (GDN-style); v_dim is independent and
    /// LARGER than kv_dim. Confirms the prove + verify path in qwen.rs
    /// correctly threads the v_dim through to w_v / w_o / w_g_proj sizing
    /// AND through to the transcript (S3 absorbs `v_num_heads`, `v_d_head`,
    /// `v_dim` already; this exercises the live numeric path).
    #[test]
    fn test_qwen_asymmetric_v_dim_proves_and_verifies() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 2,
            num_kv_heads: 2,             // GDN-style: q == kv
            d_head: 4,                    // → q_dim = kv_dim = 8
            n_layers: 1,
            vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 4, v_d_head: 4,  // → v_dim = 16, MISMATCHED with kv_dim
        };
        let kv_dim = config.num_kv_heads * config.d_head;
        let v_dim = config.v_dim();
        assert_ne!(kv_dim, v_dim, "test must exercise asymmetric V (kv != v)");

        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_asymmetric_qwen_weights(&config);
        let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);

        let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);
        let commitments = placeholder_qwen_commitments(&weights);

        // Prove with EF (production default).
        let mut pt = Transcript::new(b"qwen-asym-v-test");
        let proof = prove_qwen_layer_precommitted_ef(
            &trace, &weights, &config, commitments, &silu_table, &sigmoid_table, &mut pt,
        );

        let mut vt = Transcript::new(b"qwen-asym-v-test");
        assert!(
            verify_qwen_layer_ef(
                &proof, &x, &trace.output, &weights, &config,
                &silu_table, &sigmoid_table, &mut vt,
            ),
            "asymmetric-V Qwen layer must prove + verify"
        );
    }

    /// SOUNDNESS regression (T4, Phase 5): a 1-byte tamper of any committed
    /// transcript-input must cause verification to fail. Specifically: take a
    /// valid Qwen base-field proof, flip one round-poly coefficient inside
    /// the rmsnorm var_proof, and re-run verify. Verifier reconstructs the
    /// transcript from the proof; the flipped coefficient diverges the
    /// challenge stream, so the inner equality at the end of sumcheck fails.
    #[test]
    fn test_qwen_proof_round_poly_byte_tamper_rejects() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 2,
            num_kv_heads: 2,
            d_head: 4,
            n_layers: 1,
            vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,
        };
        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_qwen_weights(&config);
        let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);

        let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);

        let mut pt = Transcript::new(b"qwen-base-tamper-rp");
        let mut proof = prove_qwen_layer_with_trace(
            &trace, &weights, &config, &silu_table, &sigmoid_table, &mut pt,
        );

        // Sanity: untampered proof verifies.
        let mut vt0 = Transcript::new(b"qwen-base-tamper-rp");
        assert!(
            verify_qwen_layer(
                &proof, &x, &trace.output, &weights, &config,
                &silu_table, &sigmoid_table, &mut vt0,
            ),
            "untampered proof must verify"
        );

        // Tamper: flip one bit of the first coefficient in the first round
        // poly of rmsnorm1's var_proof. (Any committed value the verifier
        // absorbs would produce an equivalent test; this site is convenient.)
        assert!(!proof.norm1_proof.var_proof.round_polys.is_empty(),
            "test assumes rmsnorm1 var_proof has at least one round poly");
        assert!(!proof.norm1_proof.var_proof.round_polys[0].is_empty(),
            "test assumes round poly is non-empty");
        proof.norm1_proof.var_proof.round_polys[0][0] ^= 1;

        let mut vt = Transcript::new(b"qwen-base-tamper-rp");
        assert!(
            !verify_qwen_layer(
                &proof, &x, &trace.output, &weights, &config,
                &silu_table, &sigmoid_table, &mut vt,
            ),
            "verifier must reject single-bit tamper of round_poly coefficient"
        );
    }

    /// SOUNDNESS regression (T2, Phase 5): swapping a committed weight matrix's
    /// commitment root post-prove must cause verification to fail. The
    /// transcript binds to commitment roots (S2), so a different root → a
    /// different challenge stream → final equality fails.
    #[test]
    fn test_qwen_proof_commitment_swap_rejects() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 2,
            num_kv_heads: 2,
            d_head: 4,
            n_layers: 1,
            vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,
        };
        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_qwen_weights(&config);
        let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);

        let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);
        let commitments = placeholder_qwen_commitments(&weights);

        let mut pt = Transcript::new(b"qwen-ef-commit-swap");
        let mut proof = prove_qwen_layer_precommitted_ef(
            &trace, &weights, &config, commitments, &silu_table, &sigmoid_table, &mut pt,
        );

        // Sanity: untampered proof verifies.
        let mut vt0 = Transcript::new(b"qwen-ef-commit-swap");
        assert!(
            verify_qwen_layer_ef(
                &proof, &x, &trace.output, &weights, &config,
                &silu_table, &sigmoid_table, &mut vt0,
            ),
            "untampered EF proof must verify"
        );

        // Swap w_q commitment root with a clearly different value.
        // (Placeholder roots are zero; tamper to non-zero so the byte stream
        // truly changes — this exercises the S2 transcript binding.)
        proof.w_q_commitment.root[0] ^= 0xAA;

        let mut vt = Transcript::new(b"qwen-ef-commit-swap");
        assert!(
            !verify_qwen_layer_ef(
                &proof, &x, &trace.output, &weights, &config,
                &silu_table, &sigmoid_table, &mut vt,
            ),
            "verifier must reject swapped commitment root"
        );
    }

    /// SOUNDNESS regression (S4 / P10-3):
    /// at seq_len=1 the row-attention sumcheck stays empty
    /// (`row_proofs: vec![]`) BUT the proof now carries a non-empty
    /// `seq1_consistency` sub-proof binding `MLE(attn_out, r)` and
    /// `MLE(v, r)`. For GDN-style configs (q == kv), the verifier
    /// additionally enforces the identity `attn_out_at_r == v_at_r`
    /// at the same `r` — that's the proof-level binding that survives
    /// the future true-ZK migration.
    ///
    /// This test pins the post-P10-3 structure: empty `row_proofs`
    /// AND `Some(Seq1VConsistency)` populated correctly. If the row
    /// sumcheck is ever activated at seq_len=1 (e.g., the full GDN
    /// β·(q·k)·v degenerate form lands), this test becomes the
    /// canonical "structural change" trip-wire and should be
    /// replaced.
    #[test]
    fn test_qwen_seq_len_1_attn_proof_is_empty_pinning() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 2,
            num_kv_heads: 2,                   // GDN branch (q == kv)
            d_head: 4,
            n_layers: 1,
            vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,
        };
        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_qwen_weights(&config);
        let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);
        let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);

        // The trace at seq_len=1 sets attn_out = v.clone() in the GDN branch.
        // If a future change to qwen_forward starts proving the v↔attn_out
        // relationship via a non-empty row_proofs at seq_len=1, this test
        // becomes a positive proof of structural change and should be
        // replaced by the actual S4 tamper test.
        assert_eq!(trace.attn_out.len(), trace.v.len(),
            "GDN seq_len=1: attn_out and v share length");
        for (i, (&a, &b)) in trace.attn_out.iter().zip(trace.v.iter()).enumerate() {
            assert_eq!(a, b,
                "GDN seq_len=1: attn_out[{i}] must equal v[{i}] (current passthrough behavior)");
        }

        let mut pt = Transcript::new(b"qwen-s4-pinning");
        let proof = prove_qwen_layer_with_trace(
            &trace, &weights, &config, &silu_table, &sigmoid_table, &mut pt,
        );

        // Pin: row_proofs is empty at seq_len=1 (current S4 architecture);
        // BUT seq1_consistency is now Some after P10-3 closure.
        assert_eq!(proof.attn_proof.seq_len, 1, "test config is seq_len=1");
        assert!(proof.attn_proof.row_proofs.is_empty(),
            "S4: at seq_len=1 the row-attention sumcheck is empty");
        let seq1 = proof.attn_proof.seq1_consistency.as_ref()
            .expect("P10-3: seq1_consistency must be populated at seq_len=1");
        // For GDN-style (q == kv), the identity check holds: attn_out_at_r == v_at_r.
        assert!(!seq1.is_gqa_full_attn,
            "test config is GDN-style (q_heads == kv_heads)");
        assert_eq!(seq1.attn_out_at_r, seq1.v_at_r,
            "P10-3 GDN identity: attn_out_at_r must equal v_at_r at the same r");
    }

    /// SOUNDNESS regression (S3 end-to-end, reviewer
    /// follow-up): the helper-level `test_absorb_qwen_dims_diverges_on_config_change`
    /// proves that `absorb_qwen_dims` produces divergent transcripts on
    /// dim flips, but it doesn't exercise the full prove→tamper→verify
    /// flow. This test does: produce a valid EF proof with `configA`,
    /// then run the verifier with a `configB` that differs from `configA`
    /// in exactly ONE dim that doesn't affect weight sizing (`d_ff` in
    /// the symmetric-V case). The S3 absorption happens before any
    /// challenge is squeezed, so configB → different transcript →
    /// different challenges → first downstream sumcheck rejects.
    /// Catches the canonical config-injection attack at the proof
    /// level, not just the helper level.
    #[test]
    fn test_qwen_proof_config_injection_rejects_on_d_ff_swap() {
        let config_a = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 2,
            num_kv_heads: 2,                   // GDN branch
            d_head: 4,
            n_layers: 1,
            vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,
        };
        // configB differs ONLY in d_ff. Because we don't actually compile
        // an MLP weight matrix sized by d_ff (the test uses
        // make_qwen_weights which sizes against config_a), this would
        // fail at the matmul-shape check too — but the absorption-level
        // divergence already trips first since dims are absorbed before
        // any downstream verification step.
        let mut config_b = config_a.clone();
        config_b.d_ff = config_a.d_ff + 4; // any non-equal value

        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_qwen_weights(&config_a);
        let x = find_valid_qwen_input(&config_a, &weights, &silu_table, &sigmoid_table);
        let trace = qwen_forward(&x, &weights, &config_a, &silu_table, &sigmoid_table);
        let commitments = placeholder_qwen_commitments(&weights);

        let mut pt = Transcript::new(b"qwen-s3-config-injection");
        let proof = prove_qwen_layer_precommitted_ef(
            &trace, &weights, &config_a, commitments, &silu_table, &sigmoid_table, &mut pt,
        );

        // Sanity: untampered proof verifies under configA.
        let mut vt0 = Transcript::new(b"qwen-s3-config-injection");
        assert!(
            verify_qwen_layer_ef(
                &proof, &x, &trace.output, &weights, &config_a,
                &silu_table, &sigmoid_table, &mut vt0,
            ),
            "untampered EF proof must verify under matching configA"
        );

        // Tamper: verify with configB (only d_ff differs). The S3
        // absorption injects a different d_ff into the verifier's
        // transcript, so squeezed challenges diverge from prover's.
        // Catch.unwind isolates the panics that some shape-checked
        // sub-proofs may emit when challenges drift far from valid.
        let mut vt = Transcript::new(b"qwen-s3-config-injection");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify_qwen_layer_ef(
                &proof, &x, &trace.output, &weights, &config_b,
                &silu_table, &sigmoid_table, &mut vt,
            )
        }));
        match result {
            Ok(false) => { /* expected — verifier returned false */ }
            Err(_) => { /* also acceptable — sub-proof shape check tripped */ }
            Ok(true) => panic!(
                "S3 config-injection: verifier accepted proof under \
                 configB (d_ff differs) — transcript dim binding is broken"
            ),
        }
    }

    /// PERF regression (G3): at seq_len=1 in the GDN
    /// path, q_proof and k_proof are produced but their values are never
    /// consumed by downstream sub-proofs (attn_out = v passthrough →
    /// sigmoid_gate → o_proj). The proofs are soundness-neutral but cost
    /// ~2 matmul proofs of overhead per layer. This test pins the
    /// presence of all three QKV proofs so a future optimization that
    /// drops Q/K at seq_len=1 produces a coordinated roadmap update
    /// rather than a silent change to the proof structure.
    ///
    /// When the GDN delta-rule recurrent-state proof is added, the
    /// expected fix is to wire trace.q + trace.k into a state-update
    /// sub-proof; this test will then need to be replaced with the
    /// corresponding "q,k actually consumed" assertion.
    #[test]
    fn test_qwen_seq_len_1_qk_proofs_present_pinning() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 2,
            num_kv_heads: 2,                   // GDN branch
            d_head: 4,
            n_layers: 1,
            vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,
        };
        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_qwen_weights(&config);
        let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);
        let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);

        let mut pt = Transcript::new(b"qwen-g3-pinning");
        let proof = prove_qwen_layer_with_trace(
            &trace, &weights, &config, &silu_table, &sigmoid_table, &mut pt,
        );

        // The QKV tuple has all three sub-proofs populated. Specifically,
        // each sub-proof has a non-empty round-poly stream (sumcheck on
        // d_model = 8 → log2 = 3 rounds, so at least 3 round_polys).
        assert!(!proof.qkv_proofs.0.matmul_proof.sumcheck_proof.round_polys.is_empty(),
            "G3: q_proof exists at seq_len=1 (currently produced even though q is unused)");
        assert!(!proof.qkv_proofs.1.matmul_proof.sumcheck_proof.round_polys.is_empty(),
            "G3: k_proof exists at seq_len=1 (currently produced even though k is unused)");
        assert!(!proof.qkv_proofs.2.matmul_proof.sumcheck_proof.round_polys.is_empty(),
            "G3: v_proof exists and IS consumed (attn_out = v passthrough)");
    }

    /// SOUNDNESS regression (S4 closure, reviewer
    /// follow-up): demonstrate that the previously-flagged "malicious
    /// prover substitutes attn_out at seq_len=1" attack does NOT actually
    /// succeed in the audit-mode architecture, because the verifier
    /// re-runs `qwen_forward(x, weights, config, ...)` and uses ITS OWN
    /// canonical trace as the binding context for every sub-proof. A
    /// proof generated against a tampered trace (where attn_out has been
    /// substituted) carries MLE evaluations of the FAKE attn_out at
    /// downstream challenge points, but the verifier checks those
    /// evaluations against the CANONICAL trace's attn_out — they differ,
    /// and the verifier rejects.
    ///
    /// This test runs `qwen_forward` once to get the canonical trace,
    /// constructs a tampered trace by mutating `attn_out[0]`, generates
    /// a proof against the tampered trace, then asks the verifier to
    /// validate against the canonical (x, y). The verifier recomputes
    /// the canonical trace internally and compares sub-proof bindings
    /// to it; the tampered proof must reject.
    ///
    /// If a future refactor changes the verifier to NOT recompute the
    /// trace (i.e. moves toward true ZK with hidden weights), this test
    /// must be replaced with a proof-level v↔attn_out consistency
    /// sub-proof — and S4 will need a fresh fix. Until that day the
    /// verifier's canonical-trace check is the binding mechanism.
    #[test]
    fn test_qwen_seq_len_1_attn_out_tamper_rejected_via_canonical_trace() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 2,
            num_kv_heads: 2,                   // GDN branch
            d_head: 4,
            n_layers: 1,
            vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,
        };
        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_qwen_weights(&config);
        let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);

        // Canonical trace + canonical y (what the verifier will compute).
        let canonical_trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);
        let canonical_y = canonical_trace.output;

        // Tampered trace: re-run qwen_forward and flip attn_out[0]. The
        // tampered trace's downstream values (gated_attn, gated_out, h,
        // ...) are NOT recomputed — the prover proves the inconsistent
        // tuple as-is. (A more sophisticated tamper would propagate the
        // flip; this simpler form already exercises the audit-mode
        // mitigation since the verifier's canonical sub-proof bindings
        // mismatch the prover's at the very first matmul/sigmoid_gate
        // step that touches attn_out.)
        let mut tampered_trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);
        tampered_trace.attn_out[0] = tampered_trace.attn_out[0] + F::one();

        let mut pt = Transcript::new(b"qwen-s4-attn-tamper");
        let tampered_proof = prove_qwen_layer_with_trace(
            &tampered_trace, &weights, &config, &silu_table, &sigmoid_table, &mut pt,
        );

        // Verify against the canonical y (what an honest user would supply).
        // The verifier will:
        //   1. Recompute canonical_trace via qwen_forward → has canonical
        //      attn_out, gated_out, output.
        //   2. Check `trace.output == canonical_y` — passes.
        //   3. Run sub-proofs against canonical_trace's intermediates.
        //      The tampered_proof's MLE evaluations were generated against
        //      tampered_trace's attn_out (and the propagated downstream
        //      values), so the verifier's canonical-trace bindings will
        //      mismatch, causing rejection.
        let mut vt = Transcript::new(b"qwen-s4-attn-tamper");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify_qwen_layer(
                &tampered_proof, &x, &canonical_y, &weights, &config,
                &silu_table, &sigmoid_table, &mut vt,
            )
        }));
        match result {
            Ok(false) => { /* expected — verifier returned false */ }
            Err(_) => { /* also acceptable — sub-proof shape check tripped */ }
            Ok(true) => panic!(
                "S4 closure: verifier accepted tampered-attn_out proof \
                 against canonical y — the canonical-trace recomputation \
                 is NOT acting as a binding mechanism, audit-mode S4 \
                 mitigation is broken"
            ),
        }
    }

    /// SOUNDNESS regression (P10-3, post-S4-closure):
    /// the seq1_consistency sub-proof itself rejects tampering, even
    /// before the canonical-trace audit-mode check kicks in. Generate a
    /// valid proof, surgically flip the `attn_out_at_r` claim in the
    /// proof bytes (so the prover's claim no longer matches what the
    /// canonical trace would produce), then verify with canonical y
    /// and assert rejection. This is the proof-level binding check —
    /// load-bearing for the future true-ZK migration where
    /// canonical-trace recomputation goes away.
    #[test]
    fn test_qwen_seq1_consistency_attn_out_at_r_tamper_rejects() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 2,
            num_kv_heads: 2,                   // GDN branch
            d_head: 4,
            n_layers: 1,
            vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,
        };
        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_qwen_weights(&config);
        let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);
        let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);

        let mut pt = Transcript::new(b"qwen-p10-3-tamper");
        let mut proof = prove_qwen_layer_with_trace(
            &trace, &weights, &config, &silu_table, &sigmoid_table, &mut pt,
        );

        // Sanity: untampered proof verifies.
        let mut vt0 = Transcript::new(b"qwen-p10-3-tamper");
        assert!(verify_qwen_layer(&proof, &x, &trace.output, &weights, &config,
            &silu_table, &sigmoid_table, &mut vt0,
        ), "untampered proof must verify");

        // Tamper: flip the seq1_consistency.attn_out_at_r claim.
        let seq1 = proof.attn_proof.seq1_consistency.as_mut()
            .expect("P10-3: seq1_consistency present at seq_len=1");
        seq1.attn_out_at_r = seq1.attn_out_at_r.wrapping_add(1);

        let mut vt = Transcript::new(b"qwen-p10-3-tamper");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify_qwen_layer(&proof, &x, &trace.output, &weights, &config,
                &silu_table, &sigmoid_table, &mut vt)
        }));
        match result {
            Ok(false) => { /* expected: seq1 attn_out_at_r mismatch fires */ }
            Err(_) => { /* also acceptable: downstream sub-proof shape check */ }
            Ok(true) => panic!(
                "P10-3 binding broken: verifier accepted tampered seq1_consistency.attn_out_at_r"
            ),
        }
    }

    /// SOUNDNESS regression (P10-3 GDN-identity): for GDN-style configs
    /// the verifier additionally enforces `attn_out_at_r == v_at_r`.
    /// Tamper just `v_at_r` (without touching `attn_out_at_r`) so the
    /// equality check fires but the canonical-trace check on attn_out
    /// still passes — verifies the GDN identity check is the
    /// load-bearing binding, not just the canonical-trace check.
    #[test]
    fn test_qwen_seq1_gdn_identity_v_at_r_tamper_rejects() {
        let config = ModelConfig {
            d_model: 8, d_ff: 16, num_q_heads: 2, num_kv_heads: 2,
            d_head: 4, n_layers: 1, vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,
        };
        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_qwen_weights(&config);
        let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);
        let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);

        let mut pt = Transcript::new(b"qwen-p10-3-gdn");
        let mut proof = prove_qwen_layer_with_trace(
            &trace, &weights, &config, &silu_table, &sigmoid_table, &mut pt,
        );

        // Tamper v_at_r only.
        let seq1 = proof.attn_proof.seq1_consistency.as_mut()
            .expect("P10-3: seq1_consistency present");
        seq1.v_at_r = seq1.v_at_r.wrapping_add(1);

        let mut vt = Transcript::new(b"qwen-p10-3-gdn");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify_qwen_layer(&proof, &x, &trace.output, &weights, &config,
                &silu_table, &sigmoid_table, &mut vt)
        }));
        match result {
            Ok(false) => { /* expected */ }
            Err(_) => { /* acceptable */ }
            Ok(true) => panic!("P10-3 GDN identity check broken — v_at_r tamper accepted"),
        }
    }

    /// SOUNDNESS regression (P10-3 GQA r-coordinate fix — 3rd reviewer
    /// follow-up): exercise the GQA full-attn branch (q_heads != kv_heads)
    /// to ensure the seq1_consistency sub-proof actually binds when
    /// `attn_out` is a head-replicated view of `v` rather than identity.
    /// Specifically: build a config where `num_q_heads != num_kv_heads`
    /// (the GQA full-attn branch), generate a valid proof, tamper
    /// `attn_out_at_r`, and assert verifier rejection. Without the
    /// correct group-prefix + d-head-suffix slicing of `r`, the GQA
    /// path provided zero algebraic binding (the verifier was just
    /// re-checking against canonical-trace recomputation).
    ///
    /// In the current Qwen proof structure, GQA full-attn requires
    /// `make_qwen_weights` plus a non-zero `g_proj_out` distribution
    /// to drive the sigmoid_gate proof; we reuse the existing test
    /// harness but flip `num_q_heads` to 4 vs `num_kv_heads = 2`,
    /// keeping `d_head = 4` so the dims (q_dim=16, kv_dim=8) are
    /// well-formed power-of-2.
    #[test]
    fn test_qwen_seq1_gqa_attn_out_at_r_tamper_rejects() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 4,                    // GQA: q != kv
            num_kv_heads: 2,
            d_head: 4,                         // q_dim=16, kv_dim=8 (powers of 2)
            n_layers: 1,
            vocab_size: 8,
            norm_type: super::super::NormType::RMSNorm,
            activation: super::super::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,       // symmetric V (kv_dim path)
        };
        let silu_table = build_small_silu_table(10);
        let sigmoid_table = build_small_sigmoid_table(10);
        let weights = make_qwen_weights(&config);
        let x = find_valid_qwen_input(&config, &weights, &silu_table, &sigmoid_table);
        let trace = qwen_forward(&x, &weights, &config, &silu_table, &sigmoid_table);

        // Sanity: trace.attn_out is the head-replicated view, not identity.
        // For (num_q_heads=4, num_kv_heads=2, d_head=4): each kv head is
        // replicated to 2 q heads, so attn_out has q_dim=16 entries while
        // v has kv_dim=8.
        assert_eq!(trace.attn_out.len(), 16);
        assert_eq!(trace.v.len(), 8);

        let mut pt = Transcript::new(b"qwen-p10-3-gqa-tamper");
        let mut proof = prove_qwen_layer_with_trace(
            &trace, &weights, &config, &silu_table, &sigmoid_table, &mut pt,
        );

        // Sanity: untampered proof verifies.
        let mut vt0 = Transcript::new(b"qwen-p10-3-gqa-tamper");
        assert!(verify_qwen_layer(&proof, &x, &trace.output, &weights, &config,
            &silu_table, &sigmoid_table, &mut vt0,
        ), "untampered GQA proof must verify");

        // Confirm we're actually exercising the GQA branch.
        let seq1 = proof.attn_proof.seq1_consistency.as_ref()
            .expect("P10-3: seq1_consistency present");
        assert!(seq1.is_gqa_full_attn,
            "test config must trigger is_gqa_full_attn (q_heads != kv_heads)");

        // Tamper: flip attn_out_at_r. With the correct group-prefix +
        // d-suffix slicing, the verifier's canonical recomputation
        // produces a different value than the prover's tampered claim,
        // and the equality check at the verify path fires.
        let seq1_mut = proof.attn_proof.seq1_consistency.as_mut().unwrap();
        seq1_mut.attn_out_at_r = seq1_mut.attn_out_at_r.wrapping_add(1);

        let mut vt = Transcript::new(b"qwen-p10-3-gqa-tamper");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify_qwen_layer(&proof, &x, &trace.output, &weights, &config,
                &silu_table, &sigmoid_table, &mut vt)
        }));
        match result {
            Ok(false) => { /* expected: GQA seq1 attn_out_at_r mismatch fires */ }
            Err(_) => { /* acceptable: downstream sub-proof tripped first */ }
            Ok(true) => panic!(
                "P10-3 GQA binding broken: verifier accepted tampered \
                 seq1_consistency.attn_out_at_r in GQA branch. The r-coord \
                 slicing for v_at_r is wrong — should be group-prefix + \
                 d-head-suffix, not contiguous prefix."
            ),
        }
    }
}
