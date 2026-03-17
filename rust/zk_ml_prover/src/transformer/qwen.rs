//! Qwen3.5-style transformer layer proving (RMSNorm + GQA/GDN + output gating + SwiGLU).

use p3_field::PrimeField32;
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
/// Only for tests — production must use `commit_qwen_layer` for real binding.
#[allow(dead_code)]
pub fn placeholder_qwen_commitments(weights: &QwenLayerWeights) -> QwenLayerCommitments {
    let placeholder = |w: &[F]| -> WeightCommitment {
        let n = w.len();
        let log_n = if n <= 1 { 1 } else { log2_ceil(n) };
        WeightCommitment {
            root: [0u8; 32],
            num_weights: n,
            log_height: log_n,
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
    let (norm1_out, norm1_x) = rmsnorm_forward(x, &weights.norm1_gamma);

    // QKV + g_proj projections (all from norm1_out, independent → parallel)
    let q_dim = num_q_heads * d_head;
    let kv_dim = num_kv_heads * d_head;
    let (q, (k, (v, g_proj_out_raw))) = rayon::join(
        || matmul_forward(&weights.w_q, &norm1_out, q_dim, d_model, None),
        || rayon::join(
            || matmul_forward(&weights.w_k, &norm1_out, kv_dim, d_model, None),
            || rayon::join(
                || matmul_forward(&weights.w_v, &norm1_out, kv_dim, d_model, None),
                || matmul_forward(&weights.w_g_proj, &norm1_out, q_dim, d_model, None),
            ),
        ),
    );

    // Attention (seq_len=1: trivial, same as Llama)
    let heads_per_group = num_q_heads / num_kv_heads;
    let mut attn_out = Vec::with_capacity(q_dim);
    for h in 0..num_q_heads {
        let kv_idx = h / heads_per_group;
        for d in 0..d_head {
            attn_out.push(v[kv_idx * d_head + d]);
        }
    }

    // Output gate: sigmoid → gated = attn_out ⊙ sigmoid(g_proj_out)
    let g_proj_out = requantize_to_i16_field(&g_proj_out_raw, sigmoid_table);
    let g_proj_sigmoid: Vec<F> = g_proj_out.iter().map(|&v| lookup_fast(v.as_canonical_u32(), &sigmoid_index)).collect();

    // Gate is applied to attn_out (q_dim), BEFORE o_proj.
    let gated_attn: Vec<F> = attn_out.iter().zip(g_proj_sigmoid.iter())
        .map(|(&a, &b)| a * b).collect();

    // O projection on gated attention (only compute the gated path)
    let gated_out = matmul_forward(&weights.w_o, &gated_attn, d_model, q_dim, None);

    // Residual 1
    let h: Vec<F> = x.iter().zip(gated_out.iter()).map(|(&a, &b)| a + b).collect();

    // RMSNorm 2
    let (norm2_out, norm2_x) = rmsnorm_forward(&h, &weights.norm2_gamma);

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
        x: x.to_vec(), norm1_x, norm1_out, q, k, v, attn_out,
        o_proj_out: vec![], g_proj_out_raw, g_proj_out, g_proj_sigmoid, gated_out,
        h, norm2_x, norm2_out, gate_out_raw, gate_out, gate_silu,
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
    let kv_dim = config.num_kv_heads * config.d_head;

    // Compute commitments first — needed for Fiat-Shamir binding (transcript
    // must absorb commitment roots before generating matmul challenges).
    let w_q_commitment = commit_weight_matrix(&weights.w_q, q_dim, d_model);
    let w_k_commitment = commit_weight_matrix(&weights.w_k, kv_dim, d_model);
    let w_v_commitment = commit_weight_matrix(&weights.w_v, kv_dim, d_model);
    let w_o_commitment = commit_weight_matrix(&weights.w_o, d_model, q_dim);
    let w_g_proj_commitment = commit_weight_matrix(&weights.w_g_proj, q_dim, d_model);
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
        let norm1_proof = prove_rmsnorm(&trace.norm1_x, &weights.norm1_gamma, &trace.norm1_out, transcript);

        // 2. QKV matmuls
        let q_proof = prove_matmul_succinct(&weights.w_q, &trace.norm1_out, &trace.q, q_dim, d_model, None, transcript);
        let k_proof = prove_matmul_succinct(&weights.w_k, &trace.norm1_out, &trace.k, kv_dim, d_model, None, transcript);
        let v_proof = prove_matmul_succinct(&weights.w_v, &trace.norm1_out, &trace.v, kv_dim, d_model, None, transcript);

        // 3. Attention (seq_len=1: trivial)
        let attn_proof = RowAttentionProof {
            row_proofs: vec![],
            num_heads: config.num_q_heads,
            seq_len: 1,
            d_head: config.d_head,
        };

        // 4. O projection: gated_out = W_o @ gated_attn
        let gated_attn: Vec<F> = trace.attn_out.iter().zip(trace.g_proj_sigmoid.iter())
            .map(|(&a, &b)| a * b).collect();
        let o_proj_proof = prove_matmul_succinct(
            &weights.w_o, &gated_attn, &trace.gated_out,
            d_model, q_dim, None, transcript,
        );

        // 5. g_proj matmul: g_proj_out_raw = W_g @ norm1_out
        let g_proj_proof = prove_matmul_succinct(
            &weights.w_g_proj, &trace.norm1_out, &trace.g_proj_out_raw,
            q_dim, d_model, None, transcript,
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
        let norm2_proof = prove_rmsnorm(&trace.norm2_x, &weights.norm2_gamma, &trace.norm2_out, transcript);

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
    let kv_dim = config.num_kv_heads * config.d_head;

    let trace = qwen_forward(x, weights, config, silu_table, sigmoid_table);
    if trace.output != y {
        eprintln!("Qwen: output mismatch");
        return false;
    }

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

    // 2. QKV matmuls
    let q_r = verify_matmul_succinct(&proof.qkv_proofs.0, &proof.w_q_commitment, &trace.q, q_dim, d_model, None, transcript);
    if !q_r.valid { eprintln!("Qwen: Q matmul failed"); return false; }
    let k_r = verify_matmul_succinct(&proof.qkv_proofs.1, &proof.w_k_commitment, &trace.k, kv_dim, d_model, None, transcript);
    if !k_r.valid { eprintln!("Qwen: K matmul failed"); return false; }
    let v_r = verify_matmul_succinct(&proof.qkv_proofs.2, &proof.w_v_commitment, &trace.v, kv_dim, d_model, None, transcript);
    if !v_r.valid { eprintln!("Qwen: V matmul failed"); return false; }

    // 3. Attention (seq_len=1: trivial)

    // 4. O projection
    let o_r = verify_matmul_succinct(&proof.o_proj_proof, &proof.w_o_commitment, &trace.gated_out, d_model, q_dim, None, transcript);
    if !o_r.valid { eprintln!("Qwen: O proj failed"); return false; }

    // 5. g_proj matmul
    let g_r = verify_matmul_succinct(&proof.g_proj_proof, &proof.w_g_proj_commitment, &trace.g_proj_out_raw, q_dim, d_model, None, transcript);
    if !g_r.valid { eprintln!("Qwen: g_proj failed"); return false; }

    // 6. Sigmoid gate
    if !verify_sigmoid_gate(&proof.sigmoid_gate_proof, q_dim, transcript) {
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
    let kv_dim = config.num_kv_heads * config.d_head;

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
    let norm1_proof = prove_rmsnorm_ef(&trace.norm1_x, &weights.norm1_gamma, &trace.norm1_out, transcript);
    let q_proof = prove_matmul_succinct_ef(&weights.w_q, &trace.norm1_out, &trace.q, q_dim, d_model, None, transcript);
    let k_proof = prove_matmul_succinct_ef(&weights.w_k, &trace.norm1_out, &trace.k, kv_dim, d_model, None, transcript);
    let v_proof = prove_matmul_succinct_ef(&weights.w_v, &trace.norm1_out, &trace.v, kv_dim, d_model, None, transcript);
    let attn_proof = RowAttentionProof { row_proofs: vec![], num_heads: config.num_q_heads, seq_len: 1, d_head: config.d_head };
    let gated_attn: Vec<F> = trace.attn_out.iter().zip(trace.g_proj_sigmoid.iter()).map(|(&a, &b)| a * b).collect();
    let o_proj_proof = prove_matmul_succinct_ef(&weights.w_o, &gated_attn, &trace.gated_out, d_model, q_dim, None, transcript);
    let g_proj_proof = prove_matmul_succinct_ef(&weights.w_g_proj, &trace.norm1_out, &trace.g_proj_out_raw, q_dim, d_model, None, transcript);
    let sigmoid_gate_proof = prove_sigmoid_gate_ef(&trace.g_proj_out, &trace.g_proj_sigmoid, &trace.attn_out, &gated_attn, sigmoid_table, transcript);
    let log_d = log2_ceil(d_model);
    let res1_point = transcript.squeeze_ef_many(log_d);
    let residual1_proof = prove_add_ef(&trace.x, &trace.gated_out, &trace.h, &res1_point, transcript);
    let norm2_proof = prove_rmsnorm_ef(&trace.norm2_x, &weights.norm2_gamma, &trace.norm2_out, transcript);
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
    let kv_dim = config.num_kv_heads * config.d_head;

    let trace = qwen_forward_indexed(x, weights, config, silu_table, sigmoid_table, silu_index, sigmoid_index);
    if trace.output != y {
        eprintln!("Qwen EF: output mismatch");
        return false;
    }

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

    // 2. QKV matmuls
    let q_r = verify_matmul_succinct_ef(&proof.qkv_proofs.0, &proof.w_q_commitment, &trace.q, q_dim, d_model, None, transcript);
    if !q_r.valid { eprintln!("Qwen EF: Q matmul failed"); return false; }
    let k_r = verify_matmul_succinct_ef(&proof.qkv_proofs.1, &proof.w_k_commitment, &trace.k, kv_dim, d_model, None, transcript);
    if !k_r.valid { eprintln!("Qwen EF: K matmul failed"); return false; }
    let v_r = verify_matmul_succinct_ef(&proof.qkv_proofs.2, &proof.w_v_commitment, &trace.v, kv_dim, d_model, None, transcript);
    if !v_r.valid { eprintln!("Qwen EF: V matmul failed"); return false; }

    // 3. Attention (seq_len=1: trivial)

    // 4. O projection
    let o_r = verify_matmul_succinct_ef(&proof.o_proj_proof, &proof.w_o_commitment, &trace.gated_out, d_model, q_dim, None, transcript);
    if !o_r.valid { eprintln!("Qwen EF: O proj failed"); return false; }

    // 5. g_proj matmul
    let g_r = verify_matmul_succinct_ef(&proof.g_proj_proof, &proof.w_g_proj_commitment, &trace.g_proj_out_raw, q_dim, d_model, None, transcript);
    if !g_r.valid { eprintln!("Qwen EF: g_proj failed"); return false; }

    // 6. Sigmoid gate — pass trace data for transcript binding
    {
        use crate::proving::sigmoid_gate::verify_sigmoid_gate_ef_with_data;
        let sig_inputs: Vec<u32> = trace.g_proj_out.iter().map(|v| v.as_canonical_u32()).collect();
        let sig_outputs: Vec<u32> = trace.g_proj_sigmoid.iter().map(|v| v.as_canonical_u32()).collect();
        if !verify_sigmoid_gate_ef_with_data(&proof.sigmoid_gate_proof, q_dim,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::common::{i16_to_field, quantize_i16, is_qr_m31 as is_qr_m31_common};
    use crate::proving::weight_commitment::WeightCommitment;
    use p3_field::{AbstractField, Field};

    fn build_small_silu_table(scale: i32) -> LookupTable {
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
            commitment: WeightCommitment { root: [0u8; 32], num_weights: 256, log_height: 8 },
        }
    }

    fn build_small_sigmoid_table(scale: i32) -> LookupTable {
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
            commitment: WeightCommitment { root: [0u8; 32], num_weights: 256, log_height: 8 },
        }
    }

    fn make_qwen_weights(config: &ModelConfig) -> QwenLayerWeights {
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

    fn find_valid_qwen_input(
        config: &ModelConfig,
        weights: &QwenLayerWeights,
        silu_table: &LookupTable,
        sigmoid_table: &LookupTable,
    ) -> Vec<F> {
        for offset in 0u32..500 {
            let candidate: Vec<F> = (0..config.d_model)
                .map(|i| F::from_canonical_u32(i as u32 + 1 + offset))
                .collect();

            // Check RMSNorm 1
            let sum_sq: F = candidate.iter().map(|&v| v * v).sum();
            if sum_sq == F::zero() { continue; }
            let target = F::from_canonical_u32(config.d_model as u32) * sum_sq.inverse();
            if !is_qr_m31_common(target) { continue; }

            // Run forward to check RMSNorm 2
            let trace = qwen_forward(&candidate, weights, config, silu_table, sigmoid_table);

            // Check RMSNorm 2 (h must have valid QR)
            let sum_sq2: F = trace.h.iter().map(|&v| v * v).sum();
            if sum_sq2 == F::zero() { continue; }
            let target2 = F::from_canonical_u32(config.d_model as u32) * sum_sq2.inverse();
            if !is_qr_m31_common(target2) { continue; }

            return candidate;
        }
        panic!("Could not find valid Qwen input");
    }

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
}
