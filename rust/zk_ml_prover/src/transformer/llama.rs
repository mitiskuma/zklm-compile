//! Llama/Mistral-style transformer layer proving (RMSNorm + GQA + SwiGLU).

use p3_field::{AbstractField, PrimeField32};
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::proving::elementwise::{prove_add, verify_add, AddProof, prove_add_ef, verify_add_ef, AddProofEF};
use crate::proving::lookup::LookupTable;
use crate::proving::matmul::{prove_matmul_succinct, verify_matmul_succinct, SuccinctMatmulProof,
    prove_matmul_succinct_ef, verify_matmul_succinct_ef, SuccinctMatmulProofEF};
use crate::proving::rmsnorm::{prove_rmsnorm, verify_rmsnorm, RmsNormProof,
    prove_rmsnorm_ef, verify_rmsnorm_ef, RmsNormProofEF};
use crate::proving::attention::RowAttentionProof;
use crate::proving::sumcheck::Transcript;
use crate::proving::swiglu::{prove_swiglu, verify_swiglu, SwiGluProof,
    prove_swiglu_ef, verify_swiglu_ef, SwiGluProofEF};
use crate::field::common::log2_ceil;
use crate::proving::weight_commitment::WeightCommitment;
use super::{matmul_forward, rmsnorm_forward, requantize_to_i16_field, commit_weight_matrix, ModelConfig};

type F = Mersenne31;

/// Weights for a Llama-style transformer layer (RMSNorm + GQA + SwiGLU).
pub struct LlamaLayerWeights {
    /// RMSNorm before attention
    pub norm1_gamma: Vec<F>,
    /// QKV projection weights: W_q (num_q_heads*d_head × d_model),
    /// W_k (num_kv_heads*d_head × d_model), W_v (num_kv_heads*d_head × d_model)
    pub w_q: Vec<F>,
    pub w_k: Vec<F>,
    pub w_v: Vec<F>,
    /// Output projection: W_o (d_model × num_q_heads*d_head)
    pub w_o: Vec<F>,
    /// RMSNorm before MLP
    pub norm2_gamma: Vec<F>,
    /// SwiGLU MLP: gate_proj (d_ff × d_model), up_proj (d_ff × d_model), down_proj (d_model × d_ff)
    pub w_gate: Vec<F>,
    pub w_up: Vec<F>,
    pub w_down: Vec<F>,
}

/// Proof for a Llama-style transformer layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlamaLayerProof {
    pub norm1_proof: RmsNormProof,
    pub qkv_proofs: (SuccinctMatmulProof, SuccinctMatmulProof, SuccinctMatmulProof),
    pub attn_proof: RowAttentionProof,
    pub o_proj_proof: SuccinctMatmulProof,
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
    pub w_gate_commitment: WeightCommitment,
    pub w_up_commitment: WeightCommitment,
    pub w_down_commitment: WeightCommitment,
}

/// All intermediate values from Llama-style forward pass.
pub struct LlamaForwardTrace {
    pub x: Vec<F>,          // original input
    pub norm1_x: Vec<F>,    // possibly perturbed input to RMSNorm 1
    pub norm1_out: Vec<F>,
    pub q: Vec<F>,  // (num_q_heads * d_head)
    pub k: Vec<F>,  // (num_kv_heads * d_head)
    pub v: Vec<F>,  // (num_kv_heads * d_head)
    pub attn_out: Vec<F>, // (num_q_heads * d_head) — raw attention output before o_proj
    pub o_proj_out: Vec<F>, // (d_model)
    pub h: Vec<F>,         // x + o_proj_out
    pub norm2_x: Vec<F>,   // possibly perturbed input to RMSNorm 2
    pub norm2_out: Vec<F>,
    pub gate_out_raw: Vec<F>, // raw gate matmul output (d_ff)
    pub gate_out: Vec<F>,  // requantized to i16 for SiLU lookup (d_ff)
    pub gate_silu: Vec<F>, // SiLU(gate_out)
    pub up_out: Vec<F>,    // (d_ff)
    pub swiglu_out: Vec<F>, // gate_silu ⊙ up_out
    pub down_out: Vec<F>,  // (d_model)
    pub output: Vec<F>,    // h + down_out
}

/// Compute Llama forward pass for seq_len=1 (single token proving).
pub fn llama_forward(
    x: &[F],
    weights: &LlamaLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
) -> LlamaForwardTrace {
    let d_model = config.d_model;
    let d_ff = config.d_ff;
    let num_q_heads = config.num_q_heads;
    let num_kv_heads = config.num_kv_heads;
    let d_head = config.d_head;

    // RMSNorm 1
    let (norm1_out, norm1_x) = rmsnorm_forward(x, &weights.norm1_gamma);

    // QKV projections (independent — run in parallel)
    let q_dim = num_q_heads * d_head;
    let kv_dim = num_kv_heads * d_head;
    let (q, (k, v)) = rayon::join(
        || matmul_forward(&weights.w_q, &norm1_out, q_dim, d_model, None),
        || rayon::join(
            || matmul_forward(&weights.w_k, &norm1_out, kv_dim, d_model, None),
            || matmul_forward(&weights.w_v, &norm1_out, kv_dim, d_model, None),
        ),
    );

    // Attention (seq_len=1 for single token — scores are trivial)
    // For seq_len=1, attention is identity: softmax of single score = 1.0, out = V
    // So attn_out = V reshaped per Q head (each Q head picks its KV group)
    let heads_per_group = num_q_heads / num_kv_heads;
    let mut attn_out = Vec::with_capacity(q_dim);
    for h in 0..num_q_heads {
        let kv_idx = h / heads_per_group;
        for d in 0..d_head {
            attn_out.push(v[kv_idx * d_head + d]);
        }
    }

    // Output projection
    let o_proj_out = matmul_forward(&weights.w_o, &attn_out, d_model, q_dim, None);

    // Residual 1
    let h: Vec<F> = x.iter().zip(o_proj_out.iter()).map(|(&a, &b)| a + b).collect();

    // RMSNorm 2
    let (norm2_out, norm2_x) = rmsnorm_forward(&h, &weights.norm2_gamma);

    // Gate + Up projections (independent — run in parallel)
    let (gate_out_raw, up_out) = rayon::join(
        || matmul_forward(&weights.w_gate, &norm2_out, d_ff, d_model, None),
        || matmul_forward(&weights.w_up, &norm2_out, d_ff, d_model, None),
    );

    // Requantize gate output to i16 range for SiLU lookup.
    let gate_out = requantize_to_i16_field(&gate_out_raw, silu_table);

    // SwiGLU: SiLU(gate) ⊙ up — direct index lookup (O(1) arithmetic per element)
    let gate_silu: Vec<F> = gate_out.iter().map(|&v| {
        let idx = crate::proving::lookup::field_to_table_index(v.as_canonical_u32(), silu_table.entries.len());
        let (_, out) = silu_table.entries[idx];
        F::from_canonical_u32(out)
    }).collect();
    let swiglu_out: Vec<F> = gate_silu.iter().zip(up_out.iter()).map(|(&a, &b)| a * b).collect();

    // Down projection
    let down_out = matmul_forward(&weights.w_down, &swiglu_out, d_model, d_ff, None);

    // Residual 2
    let output: Vec<F> = h.iter().zip(down_out.iter()).map(|(&a, &b)| a + b).collect();

    LlamaForwardTrace {
        x: x.to_vec(), norm1_x, norm1_out, q, k, v, attn_out, o_proj_out, h,
        norm2_x, norm2_out, gate_out_raw, gate_out, gate_silu, up_out, swiglu_out, down_out, output,
    }
}

/// Prove a Llama-style transformer layer.
///
/// For seq_len=1 (single token), attention simplifies to identity.
/// The proof covers: RMSNorm → QKV matmuls → O projection → residual →
/// RMSNorm → gate/up matmuls → SwiGLU → down matmul → residual.
#[allow(dead_code)]
pub fn prove_llama_layer(
    x: &[F],
    weights: &LlamaLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
    transcript: &mut Transcript,
) -> LlamaLayerProof {
    let trace = llama_forward(x, weights, config, silu_table);
    prove_llama_layer_with_trace(&trace, weights, config, silu_table, transcript)
}

/// Prove a Llama-style layer using a pre-computed forward trace.
/// Avoids recomputing the forward pass when the trace is already available.
pub fn prove_llama_layer_with_trace(
    trace: &LlamaForwardTrace,
    weights: &LlamaLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
    transcript: &mut Transcript,
) -> LlamaLayerProof {
    let d_model = config.d_model;
    let d_ff = config.d_ff;
    let q_dim = config.num_q_heads * config.d_head;
    let kv_dim = config.num_kv_heads * config.d_head;

    // Compute all weight commitments FIRST (7-way parallel via rayon).
    use std::sync::Mutex;
    let cq = Mutex::new(None);
    let ck = Mutex::new(None);
    let cv = Mutex::new(None);
    let co = Mutex::new(None);
    let cg = Mutex::new(None);
    let cu = Mutex::new(None);
    let cd = Mutex::new(None);
    rayon::scope(|rs| {
        rs.spawn(|_| *cq.lock().unwrap() = Some(commit_weight_matrix(&weights.w_q, q_dim, d_model)));
        rs.spawn(|_| *ck.lock().unwrap() = Some(commit_weight_matrix(&weights.w_k, kv_dim, d_model)));
        rs.spawn(|_| *cv.lock().unwrap() = Some(commit_weight_matrix(&weights.w_v, kv_dim, d_model)));
        rs.spawn(|_| *co.lock().unwrap() = Some(commit_weight_matrix(&weights.w_o, d_model, q_dim)));
        rs.spawn(|_| *cg.lock().unwrap() = Some(commit_weight_matrix(&weights.w_gate, d_ff, d_model)));
        rs.spawn(|_| *cu.lock().unwrap() = Some(commit_weight_matrix(&weights.w_up, d_ff, d_model)));
        rs.spawn(|_| *cd.lock().unwrap() = Some(commit_weight_matrix(&weights.w_down, d_model, d_ff)));
    });
    let w_q_commitment = cq.into_inner().unwrap().unwrap();
    let w_k_commitment = ck.into_inner().unwrap().unwrap();
    let w_v_commitment = cv.into_inner().unwrap().unwrap();
    let w_o_commitment = co.into_inner().unwrap().unwrap();
    let w_gate_commitment = cg.into_inner().unwrap().unwrap();
    let w_up_commitment = cu.into_inner().unwrap().unwrap();
    let w_down_commitment = cd.into_inner().unwrap().unwrap();

    // Absorb all weight commitment roots — Fiat-Shamir binding.
    // All subsequent challenges depend on committed weights.
    transcript.absorb_bytes(&w_q_commitment.root);
    transcript.absorb_bytes(&w_k_commitment.root);
    transcript.absorb_bytes(&w_v_commitment.root);
    transcript.absorb_bytes(&w_o_commitment.root);
    transcript.absorb_bytes(&w_gate_commitment.root);
    transcript.absorb_bytes(&w_up_commitment.root);
    transcript.absorb_bytes(&w_down_commitment.root);

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

    // 4. Output projection
    let o_proj_proof = prove_matmul_succinct(
        &weights.w_o, &trace.attn_out, &trace.o_proj_out,
        d_model, q_dim, None, transcript,
    );

    // 5. Residual 1
    let log_d = log2_ceil(d_model);
    let res1_point = transcript.squeeze_many(log_d);
    let residual1_proof = prove_add(&trace.x, &trace.o_proj_out, &trace.h, &res1_point, transcript);

    // 6. RMSNorm 2
    let norm2_proof = prove_rmsnorm(&trace.norm2_x, &weights.norm2_gamma, &trace.norm2_out, transcript);

    // 7. Gate + Up projections
    let gate_proj_proof = prove_matmul_succinct(
        &weights.w_gate, &trace.norm2_out, &trace.gate_out_raw,
        d_ff, d_model, None, transcript,
    );
    let up_proj_proof = prove_matmul_succinct(
        &weights.w_up, &trace.norm2_out, &trace.up_out,
        d_ff, d_model, None, transcript,
    );

    // 8. SwiGLU
    let swiglu_proof = prove_swiglu(
        &trace.gate_out, &trace.gate_silu, &trace.up_out, &trace.swiglu_out,
        silu_table, transcript,
    );

    // 9. Down projection
    let down_proj_proof = prove_matmul_succinct(
        &weights.w_down, &trace.swiglu_out, &trace.down_out,
        d_model, d_ff, None, transcript,
    );

    // 10. Residual 2
    let res2_point = transcript.squeeze_many(log_d);
    let residual2_proof = prove_add(&trace.h, &trace.down_out, &trace.output, &res2_point, transcript);

    LlamaLayerProof {
        norm1_proof, qkv_proofs: (q_proof, k_proof, v_proof),
        attn_proof, o_proj_proof, residual1_proof,
        norm2_proof, gate_proj_proof, up_proj_proof, swiglu_proof,
        down_proj_proof, residual2_proof,
        w_q_commitment, w_k_commitment, w_v_commitment, w_o_commitment,
        w_gate_commitment, w_up_commitment, w_down_commitment,
    }
}

/// Verify a Llama-style transformer layer proof.
pub fn verify_llama_layer(
    proof: &LlamaLayerProof,
    x: &[F],
    y: &[F],
    weights: &LlamaLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
    transcript: &mut Transcript,
) -> bool {
    let d_model = config.d_model;
    let d_ff = config.d_ff;
    let q_dim = config.num_q_heads * config.d_head;
    let kv_dim = config.num_kv_heads * config.d_head;

    // Recompute forward pass for intermediates
    let trace = llama_forward(x, weights, config, silu_table);
    if trace.output != y {
        eprintln!("Llama: output mismatch");
        return false;
    }

    // Absorb all weight commitment roots — must match prover absorption order.
    transcript.absorb_bytes(&proof.w_q_commitment.root);
    transcript.absorb_bytes(&proof.w_k_commitment.root);
    transcript.absorb_bytes(&proof.w_v_commitment.root);
    transcript.absorb_bytes(&proof.w_o_commitment.root);
    transcript.absorb_bytes(&proof.w_gate_commitment.root);
    transcript.absorb_bytes(&proof.w_up_commitment.root);
    transcript.absorb_bytes(&proof.w_down_commitment.root);

    // 1. RMSNorm 1
    if !verify_rmsnorm(&proof.norm1_proof, &trace.norm1_out, d_model, transcript) {
        eprintln!("Llama: RMSNorm 1 failed"); return false;
    }

    // 2. QKV matmuls
    let q_r = verify_matmul_succinct(&proof.qkv_proofs.0, &proof.w_q_commitment, &trace.q, q_dim, d_model, None, transcript);
    if !q_r.valid { eprintln!("Llama: Q matmul failed"); return false; }
    let k_r = verify_matmul_succinct(&proof.qkv_proofs.1, &proof.w_k_commitment, &trace.k, kv_dim, d_model, None, transcript);
    if !k_r.valid { eprintln!("Llama: K matmul failed"); return false; }
    let v_r = verify_matmul_succinct(&proof.qkv_proofs.2, &proof.w_v_commitment, &trace.v, kv_dim, d_model, None, transcript);
    if !v_r.valid { eprintln!("Llama: V matmul failed"); return false; }

    // 3. Attention (seq_len=1: trivial, skip)

    // 4. O projection
    let o_r = verify_matmul_succinct(&proof.o_proj_proof, &proof.w_o_commitment, &trace.o_proj_out, d_model, q_dim, None, transcript);
    if !o_r.valid { eprintln!("Llama: O proj failed"); return false; }

    // 5. Residual 1
    let log_d = log2_ceil(d_model);
    let res1_point = transcript.squeeze_many(log_d);
    if !verify_add(&proof.residual1_proof, &res1_point, transcript) {
        eprintln!("Llama: residual 1 failed"); return false;
    }

    // 6. RMSNorm 2
    if !verify_rmsnorm(&proof.norm2_proof, &trace.norm2_out, d_model, transcript) {
        eprintln!("Llama: RMSNorm 2 failed"); return false;
    }

    // 7. Gate + Up (gate matmul verified against raw output)
    let gate_r = verify_matmul_succinct(&proof.gate_proj_proof, &proof.w_gate_commitment, &trace.gate_out_raw, d_ff, d_model, None, transcript);
    if !gate_r.valid { eprintln!("Llama: gate proj failed"); return false; }
    let up_r = verify_matmul_succinct(&proof.up_proj_proof, &proof.w_up_commitment, &trace.up_out, d_ff, d_model, None, transcript);
    if !up_r.valid { eprintln!("Llama: up proj failed"); return false; }

    // 8. SwiGLU
    if !verify_swiglu(&proof.swiglu_proof, d_ff, transcript) {
        eprintln!("Llama: SwiGLU failed"); return false;
    }

    // 9. Down projection
    let down_r = verify_matmul_succinct(&proof.down_proj_proof, &proof.w_down_commitment, &trace.down_out, d_model, d_ff, None, transcript);
    if !down_r.valid { eprintln!("Llama: down proj failed"); return false; }

    // 10. Residual 2
    let res2_point = transcript.squeeze_many(log_d);
    if !verify_add(&proof.residual2_proof, &res2_point, transcript) {
        eprintln!("Llama: residual 2 failed"); return false;
    }

    true
}

/// Proof for a Llama-style layer with extension-field challenges (124-bit soundness).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlamaLayerProofEF {
    pub norm1_proof: RmsNormProofEF,
    pub qkv_proofs: (SuccinctMatmulProofEF, SuccinctMatmulProofEF, SuccinctMatmulProofEF),
    pub attn_proof: RowAttentionProof,
    pub o_proj_proof: SuccinctMatmulProofEF,
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
    pub w_gate_commitment: WeightCommitment,
    pub w_up_commitment: WeightCommitment,
    pub w_down_commitment: WeightCommitment,
}

/// Prove a Llama-style layer with extension-field challenges (124-bit soundness).
#[allow(dead_code)]
pub fn prove_llama_layer_ef(
    x: &[F],
    weights: &LlamaLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
    transcript: &mut Transcript,
) -> LlamaLayerProofEF {
    let trace = llama_forward(x, weights, config, silu_table);
    prove_llama_layer_ef_with_trace(&trace, weights, config, silu_table, transcript)
}

/// Prove a Llama-style EF layer using a pre-computed forward trace.
pub fn prove_llama_layer_ef_with_trace(
    trace: &LlamaForwardTrace,
    weights: &LlamaLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
    transcript: &mut Transcript,
) -> LlamaLayerProofEF {
    let d_model = config.d_model;
    let d_ff = config.d_ff;
    let q_dim = config.num_q_heads * config.d_head;
    let kv_dim = config.num_kv_heads * config.d_head;

    // Compute all weight commitments (7-way parallel via rayon).
    use std::sync::Mutex;
    let cq = Mutex::new(None);
    let ck = Mutex::new(None);
    let cv = Mutex::new(None);
    let co = Mutex::new(None);
    let cg = Mutex::new(None);
    let cu = Mutex::new(None);
    let cd = Mutex::new(None);
    rayon::scope(|rs| {
        rs.spawn(|_| *cq.lock().unwrap() = Some(commit_weight_matrix(&weights.w_q, q_dim, d_model)));
        rs.spawn(|_| *ck.lock().unwrap() = Some(commit_weight_matrix(&weights.w_k, kv_dim, d_model)));
        rs.spawn(|_| *cv.lock().unwrap() = Some(commit_weight_matrix(&weights.w_v, kv_dim, d_model)));
        rs.spawn(|_| *co.lock().unwrap() = Some(commit_weight_matrix(&weights.w_o, d_model, q_dim)));
        rs.spawn(|_| *cg.lock().unwrap() = Some(commit_weight_matrix(&weights.w_gate, d_ff, d_model)));
        rs.spawn(|_| *cu.lock().unwrap() = Some(commit_weight_matrix(&weights.w_up, d_ff, d_model)));
        rs.spawn(|_| *cd.lock().unwrap() = Some(commit_weight_matrix(&weights.w_down, d_model, d_ff)));
    });
    let w_q_commitment = cq.into_inner().unwrap().unwrap();
    let w_k_commitment = ck.into_inner().unwrap().unwrap();
    let w_v_commitment = cv.into_inner().unwrap().unwrap();
    let w_o_commitment = co.into_inner().unwrap().unwrap();
    let w_gate_commitment = cg.into_inner().unwrap().unwrap();
    let w_up_commitment = cu.into_inner().unwrap().unwrap();
    let w_down_commitment = cd.into_inner().unwrap().unwrap();

    // Absorb all weight commitment roots — Fiat-Shamir binding.
    transcript.absorb_bytes(&w_q_commitment.root);
    transcript.absorb_bytes(&w_k_commitment.root);
    transcript.absorb_bytes(&w_v_commitment.root);
    transcript.absorb_bytes(&w_o_commitment.root);
    transcript.absorb_bytes(&w_gate_commitment.root);
    transcript.absorb_bytes(&w_up_commitment.root);
    transcript.absorb_bytes(&w_down_commitment.root);

    // 1. RMSNorm 1 (EF)
    let norm1_proof = prove_rmsnorm_ef(&trace.norm1_x, &weights.norm1_gamma, &trace.norm1_out, transcript);

    // 2. QKV matmuls (EF)
    let q_proof = prove_matmul_succinct_ef(&weights.w_q, &trace.norm1_out, &trace.q, q_dim, d_model, None, transcript);
    let k_proof = prove_matmul_succinct_ef(&weights.w_k, &trace.norm1_out, &trace.k, kv_dim, d_model, None, transcript);
    let v_proof = prove_matmul_succinct_ef(&weights.w_v, &trace.norm1_out, &trace.v, kv_dim, d_model, None, transcript);

    // 3. Attention (seq_len=1: trivial, base-field — same as Qwen EF)
    let attn_proof = RowAttentionProof {
        row_proofs: vec![], num_heads: config.num_q_heads, seq_len: 1, d_head: config.d_head,
    };

    // 4. Output projection (EF)
    let o_proj_proof = prove_matmul_succinct_ef(
        &weights.w_o, &trace.attn_out, &trace.o_proj_out, d_model, q_dim, None, transcript,
    );

    // 5. Residual 1 (EF)
    let log_d = log2_ceil(d_model);
    let res1_point = transcript.squeeze_ef_many(log_d);
    let residual1_proof = prove_add_ef(&trace.x, &trace.o_proj_out, &trace.h, &res1_point, transcript);

    // 6. RMSNorm 2 (EF)
    let norm2_proof = prove_rmsnorm_ef(&trace.norm2_x, &weights.norm2_gamma, &trace.norm2_out, transcript);

    // 7. Gate + Up projections (EF)
    let gate_proj_proof = prove_matmul_succinct_ef(
        &weights.w_gate, &trace.norm2_out, &trace.gate_out_raw, d_ff, d_model, None, transcript,
    );
    let up_proj_proof = prove_matmul_succinct_ef(
        &weights.w_up, &trace.norm2_out, &trace.up_out, d_ff, d_model, None, transcript,
    );

    // 8. SwiGLU (EF — with EF SiLU lookup, 124-bit challenges)
    let swiglu_proof = prove_swiglu_ef(
        &trace.gate_out, &trace.gate_silu, &trace.up_out, &trace.swiglu_out,
        silu_table, transcript,
    );

    // 9. Down projection (EF)
    let down_proj_proof = prove_matmul_succinct_ef(
        &weights.w_down, &trace.swiglu_out, &trace.down_out, d_model, d_ff, None, transcript,
    );

    // 10. Residual 2 (EF)
    let res2_point = transcript.squeeze_ef_many(log_d);
    let residual2_proof = prove_add_ef(&trace.h, &trace.down_out, &trace.output, &res2_point, transcript);

    LlamaLayerProofEF {
        norm1_proof, qkv_proofs: (q_proof, k_proof, v_proof),
        attn_proof, o_proj_proof, residual1_proof,
        norm2_proof, gate_proj_proof, up_proj_proof, swiglu_proof,
        down_proj_proof, residual2_proof,
        w_q_commitment, w_k_commitment, w_v_commitment, w_o_commitment,
        w_gate_commitment, w_up_commitment, w_down_commitment,
    }
}

/// Verify a Llama-style EF layer proof.
#[allow(dead_code)]
pub fn verify_llama_layer_ef(
    proof: &LlamaLayerProofEF,
    x: &[F],
    y: &[F],
    weights: &LlamaLayerWeights,
    config: &ModelConfig,
    silu_table: &LookupTable,
    transcript: &mut Transcript,
) -> bool {
    let d_model = config.d_model;
    let d_ff = config.d_ff;
    let q_dim = config.num_q_heads * config.d_head;
    let kv_dim = config.num_kv_heads * config.d_head;

    // Recompute forward pass for intermediates
    let trace = llama_forward(x, weights, config, silu_table);
    if trace.output != y {
        eprintln!("Llama EF: output mismatch");
        return false;
    }

    // Absorb all weight commitment roots — must match prover absorption order.
    transcript.absorb_bytes(&proof.w_q_commitment.root);
    transcript.absorb_bytes(&proof.w_k_commitment.root);
    transcript.absorb_bytes(&proof.w_v_commitment.root);
    transcript.absorb_bytes(&proof.w_o_commitment.root);
    transcript.absorb_bytes(&proof.w_gate_commitment.root);
    transcript.absorb_bytes(&proof.w_up_commitment.root);
    transcript.absorb_bytes(&proof.w_down_commitment.root);

    // 1. RMSNorm 1 (EF)
    if !verify_rmsnorm_ef(&proof.norm1_proof, &trace.norm1_out, d_model, transcript) {
        eprintln!("Llama EF: RMSNorm 1 failed"); return false;
    }

    // 2. QKV matmuls (EF)
    let q_r = verify_matmul_succinct_ef(&proof.qkv_proofs.0, &proof.w_q_commitment, &trace.q, q_dim, d_model, None, transcript);
    if !q_r.valid { eprintln!("Llama EF: Q matmul failed"); return false; }
    let k_r = verify_matmul_succinct_ef(&proof.qkv_proofs.1, &proof.w_k_commitment, &trace.k, kv_dim, d_model, None, transcript);
    if !k_r.valid { eprintln!("Llama EF: K matmul failed"); return false; }
    let v_r = verify_matmul_succinct_ef(&proof.qkv_proofs.2, &proof.w_v_commitment, &trace.v, kv_dim, d_model, None, transcript);
    if !v_r.valid { eprintln!("Llama EF: V matmul failed"); return false; }

    // 3. Attention (seq_len=1: trivial, skip)

    // 4. O projection (EF)
    let o_r = verify_matmul_succinct_ef(&proof.o_proj_proof, &proof.w_o_commitment, &trace.o_proj_out, d_model, q_dim, None, transcript);
    if !o_r.valid { eprintln!("Llama EF: O proj failed"); return false; }

    // 5. Residual 1 (EF)
    let log_d = log2_ceil(d_model);
    let res1_point = transcript.squeeze_ef_many(log_d);
    if !verify_add_ef(&proof.residual1_proof, &res1_point, transcript) {
        eprintln!("Llama EF: residual 1 failed"); return false;
    }

    // 6. RMSNorm 2 (EF)
    if !verify_rmsnorm_ef(&proof.norm2_proof, &trace.norm2_out, d_model, transcript) {
        eprintln!("Llama EF: RMSNorm 2 failed"); return false;
    }

    // 7. Gate + Up (EF)
    let gate_r = verify_matmul_succinct_ef(&proof.gate_proj_proof, &proof.w_gate_commitment, &trace.gate_out_raw, d_ff, d_model, None, transcript);
    if !gate_r.valid { eprintln!("Llama EF: gate proj failed"); return false; }
    let up_r = verify_matmul_succinct_ef(&proof.up_proj_proof, &proof.w_up_commitment, &trace.up_out, d_ff, d_model, None, transcript);
    if !up_r.valid { eprintln!("Llama EF: up proj failed"); return false; }

    // 8. SwiGLU (EF)
    if !verify_swiglu_ef(&proof.swiglu_proof, d_ff, transcript) {
        eprintln!("Llama EF: SwiGLU failed"); return false;
    }

    // 9. Down projection (EF)
    let down_r = verify_matmul_succinct_ef(&proof.down_proj_proof, &proof.w_down_commitment, &trace.down_out, d_model, d_ff, None, transcript);
    if !down_r.valid { eprintln!("Llama EF: down proj failed"); return false; }

    // 10. Residual 2 (EF)
    let res2_point = transcript.squeeze_ef_many(log_d);
    if !verify_add_ef(&proof.residual2_proof, &res2_point, transcript) {
        eprintln!("Llama EF: residual 2 failed"); return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::common::{i16_to_field, quantize_i16, is_qr_m31 as is_qr_m31_common};
    use crate::proving::weight_commitment::WeightCommitment;
    use super::super::{NormType, ActivationType};
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

    fn make_llama_weights(config: &ModelConfig) -> LlamaLayerWeights {
        let d = config.d_model;
        let d_ff = config.d_ff;
        let q_dim = config.num_q_heads * config.d_head;
        let kv_dim = config.num_kv_heads * config.d_head;

        LlamaLayerWeights {
            norm1_gamma: vec![F::one(); d],
            // Identity-like Q projection
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
            norm2_gamma: vec![F::one(); d],
            // Zero gate/up weights, small biases for SiLU table range
            w_gate: vec![F::zero(); d_ff * d],
            w_up: vec![F::zero(); d_ff * d],
            w_down: vec![F::zero(); d * d_ff],
        }
    }

    /// Find input for Llama layer where both RMSNorms have valid QR.
    fn find_valid_llama_input(config: &ModelConfig, weights: &LlamaLayerWeights, silu_table: &LookupTable) -> Vec<F> {
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
            let (norm1_out, _) = rmsnorm_forward(&candidate, &weights.norm1_gamma);
            let q_dim = config.num_q_heads * config.d_head;
            let kv_dim = config.num_kv_heads * config.d_head;
            let _q = matmul_forward(&weights.w_q, &norm1_out, q_dim, config.d_model, None);
            let _k = matmul_forward(&weights.w_k, &norm1_out, kv_dim, config.d_model, None);
            let v = matmul_forward(&weights.w_v, &norm1_out, kv_dim, config.d_model, None);

            // seq_len=1: attn_out = V mapped through GQA
            let heads_per_group = config.num_q_heads / config.num_kv_heads;
            let mut attn_out = Vec::with_capacity(q_dim);
            for h in 0..config.num_q_heads {
                let kv_idx = h / heads_per_group;
                for d in 0..config.d_head {
                    attn_out.push(v[kv_idx * config.d_head + d]);
                }
            }
            let o_proj_out = matmul_forward(&weights.w_o, &attn_out, config.d_model, q_dim, None);
            let h: Vec<F> = candidate.iter().zip(o_proj_out.iter()).map(|(&a, &b)| a + b).collect();

            let sum_sq2: F = h.iter().map(|&v| v * v).sum();
            if sum_sq2 == F::zero() { continue; }
            let target2 = F::from_canonical_u32(config.d_model as u32) * sum_sq2.inverse();
            if !is_qr_m31_common(target2) { continue; }

            // Check gate_out values in silu table (with zero weights, gate_out = 0)
            // Zero weights means gate_out = 0, which should be in table
            let zero_key = F::zero().as_canonical_u32();
            if !silu_table.entries.iter().any(|&(inp, _)| inp == zero_key) { continue; }

            return candidate;
        }
        panic!("Could not find valid Llama input");
    }

    #[test]
    fn test_llama_layer_basic() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 4,
            num_kv_heads: 2,
            d_head: 2,
            n_layers: 1,
            vocab_size: 8,
            norm_type: NormType::RMSNorm,
            activation: ActivationType::SwiGLU,
        };
        let silu_table = build_small_silu_table(10);
        let weights = make_llama_weights(&config);
        let x = find_valid_llama_input(&config, &weights, &silu_table);

        let trace = llama_forward(&x, &weights, &config, &silu_table);

        let mut pt = Transcript::new(b"llama-test");
        let proof = prove_llama_layer(&x, &weights, &config, &silu_table, &mut pt);

        let mut vt = Transcript::new(b"llama-test");
        assert!(
            verify_llama_layer(&proof, &x, &trace.output, &weights, &config, &silu_table, &mut vt),
            "Llama layer verification should pass"
        );
    }

    #[test]
    fn test_llama_layer_tampered() {
        let config = ModelConfig {
            d_model: 8,
            d_ff: 16,
            num_q_heads: 4,
            num_kv_heads: 2,
            d_head: 2,
            n_layers: 1,
            vocab_size: 8,
            norm_type: NormType::RMSNorm,
            activation: ActivationType::SwiGLU,
        };
        let silu_table = build_small_silu_table(10);
        let weights = make_llama_weights(&config);
        let x = find_valid_llama_input(&config, &weights, &silu_table);

        let trace = llama_forward(&x, &weights, &config, &silu_table);
        let mut tampered = trace.output.clone();
        tampered[0] = tampered[0] + F::one();

        let mut pt = Transcript::new(b"llama-tamper");
        let proof = prove_llama_layer(&x, &weights, &config, &silu_table, &mut pt);

        let mut vt = Transcript::new(b"llama-tamper");
        assert!(
            !verify_llama_layer(&proof, &x, &tampered, &weights, &config, &silu_table, &mut vt),
            "Should reject tampered output"
        );
    }

    // ===== Extension field Llama tests =====

    #[test]
    fn test_llama_layer_ef_basic() {
        let config = ModelConfig {
            d_model: 8, d_ff: 16, num_q_heads: 4, num_kv_heads: 2,
            d_head: 2, n_layers: 1, vocab_size: 8,
            norm_type: NormType::RMSNorm, activation: ActivationType::SwiGLU,
        };
        let silu_table = build_small_silu_table(10);
        let weights = make_llama_weights(&config);
        let x = find_valid_llama_input(&config, &weights, &silu_table);

        let trace = llama_forward(&x, &weights, &config, &silu_table);

        let mut pt = Transcript::new(b"llama-ef-test");
        let proof = prove_llama_layer_ef(&x, &weights, &config, &silu_table, &mut pt);

        let mut vt = Transcript::new(b"llama-ef-test");
        assert!(
            verify_llama_layer_ef(&proof, &x, &trace.output, &weights, &config, &silu_table, &mut vt),
            "Llama EF layer verification should pass"
        );
    }

    #[test]
    fn test_llama_layer_ef_tampered() {
        let config = ModelConfig {
            d_model: 8, d_ff: 16, num_q_heads: 4, num_kv_heads: 2,
            d_head: 2, n_layers: 1, vocab_size: 8,
            norm_type: NormType::RMSNorm, activation: ActivationType::SwiGLU,
        };
        let silu_table = build_small_silu_table(10);
        let weights = make_llama_weights(&config);
        let x = find_valid_llama_input(&config, &weights, &silu_table);

        let trace = llama_forward(&x, &weights, &config, &silu_table);
        let mut tampered = trace.output.clone();
        tampered[0] = tampered[0] + F::one();

        let mut pt = Transcript::new(b"llama-ef-tamper");
        let proof = prove_llama_layer_ef(&x, &weights, &config, &silu_table, &mut pt);

        let mut vt = Transcript::new(b"llama-ef-tamper");
        assert!(
            !verify_llama_layer_ef(&proof, &x, &tampered, &weights, &config, &silu_table, &mut vt),
            "Should reject tampered EF output"
        );
    }
}
