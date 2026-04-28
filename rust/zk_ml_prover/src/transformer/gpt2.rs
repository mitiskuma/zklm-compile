//! GPT-2 style transformer layer proving.

use p3_field::{AbstractField, Field, PrimeField32};
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::proving::elementwise::{prove_add, verify_add, AddProof,
    prove_add_ef, verify_add_ef, AddProofEF};
use crate::proving::gelu::{prove_gelu, verify_gelu, GeluProof,
    prove_gelu_ef, verify_gelu_ef, GeluProofEF};
use crate::proving::layernorm::{prove_layernorm, verify_layernorm, LayerNormProof,
    prove_layernorm_ef, verify_layernorm_ef, LayerNormProofEF};
use crate::proving::lookup::LookupTable;
use crate::proving::matmul::{prove_matmul_succinct_prepadded, verify_matmul_succinct, SuccinctMatmulProof,
    prove_matmul_succinct_ef, verify_matmul_succinct_ef, SuccinctMatmulProofEF};
use crate::field::common::{log2_ceil, mod_sqrt_m31, is_qr_m31};
use crate::field::m31_ops::to_field;
use crate::proving::weight_commitment::{commit_weights_fast, WeightCommitment};
use crate::protocol::GPT2Desc;
use super::{matmul_forward, compute_r, build_table_index, lookup_fast};

type F = Mersenne31;

/// Weights for a single transformer layer.
pub struct TransformerLayerWeights {
    /// LayerNorm 1 parameters
    pub ln1_gamma: Vec<F>,
    pub ln1_beta: Vec<F>,
    /// Simplified attention: W_attn (d_model x d_model), no bias
    pub w_attn: Vec<F>,
    /// LayerNorm 2 parameters
    pub ln2_gamma: Vec<F>,
    pub ln2_beta: Vec<F>,
    /// MLP fc1: (d_ff x d_model)
    pub w_fc1: Vec<F>,
    pub b_fc1: Vec<F>,
    /// MLP fc2: (d_model x d_ff)
    pub w_fc2: Vec<F>,
    pub b_fc2: Vec<F>,
}

/// Proof for a single GPT-2-style transformer layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformerLayerProof {
    /// Pre-attention LayerNorm
    pub ln1_proof: LayerNormProof,
    /// Attention (simplified as single matmul W_attn @ ln1_out)
    pub attn_proof: SuccinctMatmulProof,
    /// Residual connection 1: h = x + attn_out
    pub residual1_proof: AddProof,
    /// Pre-MLP LayerNorm
    pub ln2_proof: LayerNormProof,
    /// MLP fc1: fc1_out = W1 @ ln2_out + b1
    pub mlp_fc1_proof: SuccinctMatmulProof,
    /// MLP GELU activation
    pub mlp_gelu_proof: GeluProof,
    /// MLP fc2: fc2_out = W2 @ gelu_out + b2
    pub mlp_fc2_proof: SuccinctMatmulProof,
    /// Residual connection 2: output = h + mlp_out
    pub residual2_proof: AddProof,
    /// Weight commitments for verification
    pub w_attn_commitment: WeightCommitment,
    pub w_fc1_commitment: WeightCommitment,
    pub w_fc2_commitment: WeightCommitment,
}

/// All intermediate values from the forward pass, needed for proving.
struct ForwardTrace {
    x: Vec<F>,
    ln1_out: Vec<F>,
    ln1_mu: F,
    ln1_r: F,
    attn_out: Vec<F>,
    h: Vec<F>,           // x + attn_out
    ln2_out: Vec<F>,
    ln2_mu: F,
    ln2_r: F,
    fc1_out: Vec<F>,
    gelu_out: Vec<F>,
    fc2_out: Vec<F>,
    output: Vec<F>,       // h + fc2_out
}

/// Compute the forward pass and collect all intermediates.
fn forward_pass(
    x: &[F],
    weights: &TransformerLayerWeights,
    d_model: usize,
    d_ff: usize,
    gelu_table: &LookupTable,
) -> ForwardTrace {
    // --- LayerNorm 1 ---
    let d = d_model;
    let sum_x: F = x.iter().copied().sum();
    let d_inv = F::from_canonical_u32(d as u32).inverse();
    let ln1_mu = sum_x * d_inv;
    let xc: Vec<F> = x.iter().map(|&xi| xi - ln1_mu).collect();
    let sum_sq: F = xc.iter().map(|&v| v * v).sum();
    let ln1_r = compute_r(sum_sq, d);
    let ln1_out: Vec<F> = xc
        .iter()
        .zip(weights.ln1_gamma.iter())
        .zip(weights.ln1_beta.iter())
        .map(|((&xci, &gi), &bi)| gi * xci * ln1_r + bi)
        .collect();

    // --- Attention (simplified: W_attn @ ln1_out) ---
    let attn_out = matmul_forward(&weights.w_attn, &ln1_out, d_model, d_model, None);

    // --- Residual 1: h = x + attn_out ---
    let h: Vec<F> = x.iter().zip(attn_out.iter()).map(|(&a, &b)| a + b).collect();

    // --- LayerNorm 2 ---
    let sum_h: F = h.iter().copied().sum();
    let ln2_mu = sum_h * d_inv;
    let hc: Vec<F> = h.iter().map(|&hi| hi - ln2_mu).collect();
    let sum_sq2: F = hc.iter().map(|&v| v * v).sum();
    let ln2_r = compute_r(sum_sq2, d);
    let ln2_out: Vec<F> = hc
        .iter()
        .zip(weights.ln2_gamma.iter())
        .zip(weights.ln2_beta.iter())
        .map(|((&hci, &gi), &bi)| gi * hci * ln2_r + bi)
        .collect();

    // --- MLP fc1 ---
    let fc1_out = matmul_forward(
        &weights.w_fc1,
        &ln2_out,
        d_ff,
        d_model,
        Some(&weights.b_fc1),
    );

    // --- GELU ---
    let gelu_index = build_table_index(gelu_table);
    let gelu_out: Vec<F> = fc1_out
        .iter()
        .map(|&v| lookup_fast(v.as_canonical_u32(), &gelu_index))
        .collect();

    // --- MLP fc2 ---
    let fc2_out = matmul_forward(
        &weights.w_fc2,
        &gelu_out,
        d_model,
        d_ff,
        Some(&weights.b_fc2),
    );

    // --- Residual 2: output = h + fc2_out ---
    let output: Vec<F> = h.iter().zip(fc2_out.iter()).map(|(&a, &b)| a + b).collect();

    ForwardTrace {
        x: x.to_vec(),
        ln1_out,
        ln1_mu,
        ln1_r,
        attn_out,
        h,
        ln2_out,
        ln2_mu,
        ln2_r,
        fc1_out,
        gelu_out,
        fc2_out,
        output,
    }
}

/// Prove a single transformer layer.
///
/// Generates sub-proofs for each operation and chains the transcript.
/// Returns (proof, output) where output is the layer's output vector,
/// avoiding a redundant forward pass when chaining layers.
pub fn prove_transformer_layer(
    x: &[F],
    weights: &TransformerLayerWeights,
    d_model: usize,
    d_ff: usize,
    gelu_table: &LookupTable,
    transcript: &mut crate::proving::sumcheck::Transcript,
) -> (TransformerLayerProof, Vec<F>) {
    let trace = forward_pass(x, weights, d_model, d_ff, gelu_table);

    // Commit weight matrices
    let log_m_attn = log2_ceil(d_model);
    let log_n_attn = log2_ceil(d_model);
    let m_pad_attn = 1 << log_m_attn;
    let n_pad_attn = 1 << log_n_attn;
    let mut w_attn_padded = vec![F::zero(); m_pad_attn * n_pad_attn];
    for i in 0..d_model {
        for j in 0..d_model {
            w_attn_padded[i * n_pad_attn + j] = weights.w_attn[i * d_model + j];
        }
    }
    let w_attn_commitment = commit_weights_fast(&w_attn_padded);

    let log_m_fc1 = log2_ceil(d_ff);
    let log_n_fc1 = log2_ceil(d_model);
    let m_pad_fc1 = 1 << log_m_fc1;
    let n_pad_fc1 = 1 << log_n_fc1;
    let mut w_fc1_padded = vec![F::zero(); m_pad_fc1 * n_pad_fc1];
    for i in 0..d_ff {
        for j in 0..d_model {
            w_fc1_padded[i * n_pad_fc1 + j] = weights.w_fc1[i * d_model + j];
        }
    }
    let w_fc1_commitment = commit_weights_fast(&w_fc1_padded);

    let log_m_fc2 = log2_ceil(d_model);
    let log_n_fc2 = log2_ceil(d_ff);
    let m_pad_fc2 = 1 << log_m_fc2;
    let n_pad_fc2 = 1 << log_n_fc2;
    let mut w_fc2_padded = vec![F::zero(); m_pad_fc2 * n_pad_fc2];
    for i in 0..d_model {
        for j in 0..d_ff {
            w_fc2_padded[i * n_pad_fc2 + j] = weights.w_fc2[i * d_ff + j];
        }
    }
    let w_fc2_commitment = commit_weights_fast(&w_fc2_padded);

    // Absorb all weight commitment roots — Fiat-Shamir binding.
    // All subsequent challenges depend on committed weights.
    transcript.absorb_bytes(&w_attn_commitment.root);
    transcript.absorb_bytes(&w_fc1_commitment.root);
    transcript.absorb_bytes(&w_fc2_commitment.root);

    // 1. LayerNorm 1
    let ln1_proof = prove_layernorm(
        &trace.x,
        &weights.ln1_gamma,
        &weights.ln1_beta,
        &trace.ln1_out,
        trace.ln1_mu,
        trace.ln1_r,
        transcript,
    );

    // 2. Attention matmul (simplified) — reuse pre-padded buffer from commitment
    let attn_proof = prove_matmul_succinct_prepadded(
        &w_attn_padded,
        &trace.ln1_out,
        &trace.attn_out,
        d_model,
        d_model,
        None,
        transcript,
    );

    // 3. Residual 1: h = x + attn_out
    let log_d = log2_ceil(d_model);
    let residual1_point = transcript.squeeze_many(log_d);
    let residual1_proof = prove_add(
        &trace.x,
        &trace.attn_out,
        &trace.h,
        &residual1_point,
        transcript,
    );

    // 4. LayerNorm 2
    let ln2_proof = prove_layernorm(
        &trace.h,
        &weights.ln2_gamma,
        &weights.ln2_beta,
        &trace.ln2_out,
        trace.ln2_mu,
        trace.ln2_r,
        transcript,
    );

    // 5. MLP fc1 — reuse pre-padded buffer from commitment
    let mlp_fc1_proof = prove_matmul_succinct_prepadded(
        &w_fc1_padded,
        &trace.ln2_out,
        &trace.fc1_out,
        d_ff,
        d_model,
        Some(&weights.b_fc1),
        transcript,
    );

    // 6. GELU
    let mlp_gelu_proof = prove_gelu(&trace.fc1_out, &trace.gelu_out, gelu_table, transcript);

    // 7. MLP fc2 — reuse pre-padded buffer from commitment
    let mlp_fc2_proof = prove_matmul_succinct_prepadded(
        &w_fc2_padded,
        &trace.gelu_out,
        &trace.fc2_out,
        d_model,
        d_ff,
        Some(&weights.b_fc2),
        transcript,
    );

    // 8. Residual 2: output = h + fc2_out
    let residual2_point = transcript.squeeze_many(log_d);
    let residual2_proof = prove_add(
        &trace.h,
        &trace.fc2_out,
        &trace.output,
        &residual2_point,
        transcript,
    );

    let output = trace.output;
    (TransformerLayerProof {
        ln1_proof,
        attn_proof,
        residual1_proof,
        ln2_proof,
        mlp_fc1_proof,
        mlp_gelu_proof,
        mlp_fc2_proof,
        residual2_proof,
        w_attn_commitment,
        w_fc1_commitment,
        w_fc2_commitment,
    }, output)
}

/// Verify a transformer layer proof.
///
/// The verifier needs:
/// - The output y (public)
/// - LayerNorm betas (public parameters)
/// - Bias vectors (public parameters)
/// - GELU table (public)
/// - Weight commitments are in the proof
///
/// Returns true if all sub-proofs verify.
pub fn verify_transformer_layer(
    proof: &TransformerLayerProof,
    x: &[F],
    y: &[F],
    weights: &TransformerLayerWeights,
    d_model: usize,
    d_ff: usize,
    gelu_table: &LookupTable,
    transcript: &mut crate::proving::sumcheck::Transcript,
) -> bool {
    // Recompute the forward pass to get intermediates for verification.
    // In a production system, these would be committed or chained via claims.
    // For now, the verifier recomputes to check consistency.
    let trace = forward_pass(x, weights, d_model, d_ff, gelu_table);

    // Check output matches
    if trace.output != y {
        eprintln!("Transformer: output mismatch");
        return false;
    }

    // Absorb all weight commitment roots — must match prover absorption order.
    transcript.absorb_bytes(&proof.w_attn_commitment.root);
    transcript.absorb_bytes(&proof.w_fc1_commitment.root);
    transcript.absorb_bytes(&proof.w_fc2_commitment.root);

    // 1. Verify LayerNorm 1
    if !verify_layernorm(
        &proof.ln1_proof,
        &trace.ln1_out,
        &weights.ln1_beta,
        d_model,
        transcript,
    ) {
        eprintln!("Transformer: LN1 verification failed");
        return false;
    }

    // 2. Verify attention matmul
    let attn_result = verify_matmul_succinct(
        &proof.attn_proof,
        &proof.w_attn_commitment,
        &trace.attn_out,
        d_model,
        d_model,
        None,
        transcript,
    );
    if !attn_result.valid {
        eprintln!("Transformer: attention matmul verification failed");
        return false;
    }

    // 3. Verify residual 1
    let log_d = log2_ceil(d_model);
    let residual1_point = transcript.squeeze_many(log_d);
    if !verify_add(&proof.residual1_proof, &residual1_point, transcript) {
        eprintln!("Transformer: residual1 verification failed");
        return false;
    }

    // 4. Verify LayerNorm 2
    if !verify_layernorm(
        &proof.ln2_proof,
        &trace.ln2_out,
        &weights.ln2_beta,
        d_model,
        transcript,
    ) {
        eprintln!("Transformer: LN2 verification failed");
        return false;
    }

    // 5. Verify MLP fc1
    let fc1_result = verify_matmul_succinct(
        &proof.mlp_fc1_proof,
        &proof.w_fc1_commitment,
        &trace.fc1_out,
        d_ff,
        d_model,
        Some(&weights.b_fc1),
        transcript,
    );
    if !fc1_result.valid {
        eprintln!("Transformer: MLP fc1 verification failed");
        return false;
    }

    // 6. Verify GELU
    if !verify_gelu(&proof.mlp_gelu_proof, trace.fc1_out.len(), transcript) {
        eprintln!("Transformer: GELU verification failed");
        return false;
    }

    // 7. Verify MLP fc2
    let fc2_result = verify_matmul_succinct(
        &proof.mlp_fc2_proof,
        &proof.w_fc2_commitment,
        &trace.fc2_out,
        d_model,
        d_ff,
        Some(&weights.b_fc2),
        transcript,
    );
    if !fc2_result.valid {
        eprintln!("Transformer: MLP fc2 verification failed");
        return false;
    }

    // 8. Verify residual 2
    let residual2_point = transcript.squeeze_many(log_d);
    if !verify_add(&proof.residual2_proof, &residual2_point, transcript) {
        eprintln!("Transformer: residual2 verification failed");
        return false;
    }

    true
}

impl TransformerLayerProof {
    /// Strip redundant data. Verifiers derive challenges from transcript replay.
    pub fn strip_redundant(&mut self) {
        // Matmul proofs: strip sumcheck challenges
        let strip_matmul = |p: &mut SuccinctMatmulProof| {
            p.matmul_proof.sumcheck_proof.challenges.clear();
            p.w_eval_proof.eval_sumcheck.challenges.clear();
        };
        strip_matmul(&mut self.attn_proof);
        strip_matmul(&mut self.mlp_fc1_proof);
        strip_matmul(&mut self.mlp_fc2_proof);

        // LayerNorm proofs: strip all sumcheck + MLE eval challenges
        let strip_ln = |ln: &mut LayerNormProof| {
            ln.mean_check_proof.challenges.clear();
            ln.var_proof.challenges.clear();
            ln.output_proof.challenges.clear();
            ln.xc_mean_eval_proof.eval_sumcheck.challenges.clear();
            ln.xc_var_eval_proof.eval_sumcheck.challenges.clear();
            ln.xc_output_eval_proof.eval_sumcheck.challenges.clear();
            ln.gamma_output_eval_proof.eval_sumcheck.challenges.clear();
        };
        strip_ln(&mut self.ln1_proof);
        strip_ln(&mut self.ln2_proof);

        // GELU lookup: strip sumcheck challenges only
        // (inputs/outputs kept — base-field verify_lookup reads them for transcript)
        self.mlp_gelu_proof.lookup_proof.sumcheck_proof.challenges.clear();

        // Add (residual): strip sumcheck challenges
        self.residual1_proof.a_sumcheck.challenges.clear();
        self.residual1_proof.b_sumcheck.challenges.clear();
        self.residual2_proof.a_sumcheck.challenges.clear();
        self.residual2_proof.b_sumcheck.challenges.clear();
    }
}

/// GPT-2 transformer layer proof with extension-field challenges (124-bit soundness).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformerLayerProofEF {
    pub ln1_proof: LayerNormProofEF,
    pub attn_proof: SuccinctMatmulProofEF,
    pub residual1_proof: AddProofEF,
    pub ln2_proof: LayerNormProofEF,
    pub mlp_fc1_proof: SuccinctMatmulProofEF,
    pub mlp_gelu_proof: GeluProofEF,
    pub mlp_fc2_proof: SuccinctMatmulProofEF,
    pub residual2_proof: AddProofEF,
    pub w_attn_commitment: WeightCommitment,
    pub w_fc1_commitment: WeightCommitment,
    pub w_fc2_commitment: WeightCommitment,
}

/// Prove a GPT-2 transformer layer with 124-bit EF challenges.
#[allow(dead_code)]
pub fn prove_transformer_layer_ef(
    x: &[F],
    weights: &TransformerLayerWeights,
    d_model: usize,
    d_ff: usize,
    gelu_table: &LookupTable,
    transcript: &mut crate::proving::sumcheck::Transcript,
) -> (TransformerLayerProofEF, Vec<F>) {
    let trace = forward_pass(x, weights, d_model, d_ff, gelu_table);

    // Commit weight matrices (same as base-field — commitments are field-agnostic)
    let w_attn_commitment = commit_weights_fast(&{
        let log_m = log2_ceil(d_model); let log_n = log2_ceil(d_model);
        let mp = 1 << log_m; let np = 1 << log_n;
        let mut p = vec![F::zero(); mp * np];
        for i in 0..d_model { for j in 0..d_model { p[i * np + j] = weights.w_attn[i * d_model + j]; } }
        p
    });
    let w_fc1_commitment = commit_weights_fast(&{
        let log_m = log2_ceil(d_ff); let log_n = log2_ceil(d_model);
        let mp = 1 << log_m; let np = 1 << log_n;
        let mut p = vec![F::zero(); mp * np];
        for i in 0..d_ff { for j in 0..d_model { p[i * np + j] = weights.w_fc1[i * d_model + j]; } }
        p
    });
    let w_fc2_commitment = commit_weights_fast(&{
        let log_m = log2_ceil(d_model); let log_n = log2_ceil(d_ff);
        let mp = 1 << log_m; let np = 1 << log_n;
        let mut p = vec![F::zero(); mp * np];
        for i in 0..d_model { for j in 0..d_ff { p[i * np + j] = weights.w_fc2[i * d_ff + j]; } }
        p
    });

    // Absorb weight commitment roots — Fiat-Shamir binding
    transcript.absorb_bytes(&w_attn_commitment.root);
    transcript.absorb_bytes(&w_fc1_commitment.root);
    transcript.absorb_bytes(&w_fc2_commitment.root);

    // 1. LayerNorm 1 (EF)
    let ln1_proof = prove_layernorm_ef(
        &trace.x, &weights.ln1_gamma, &weights.ln1_beta, &trace.ln1_out,
        trace.ln1_mu, trace.ln1_r, transcript,
    );

    // 2. Attention matmul (EF)
    let attn_proof = prove_matmul_succinct_ef(
        &weights.w_attn, &trace.ln1_out, &trace.attn_out,
        d_model, d_model, None, transcript,
    );

    // 3. Residual 1 (EF)
    let log_d = log2_ceil(d_model);
    let res1_point = transcript.squeeze_ef_many(log_d);
    let residual1_proof = prove_add_ef(&trace.x, &trace.attn_out, &trace.h, &res1_point, transcript);

    // 4. LayerNorm 2 (EF)
    let ln2_proof = prove_layernorm_ef(
        &trace.h, &weights.ln2_gamma, &weights.ln2_beta, &trace.ln2_out,
        trace.ln2_mu, trace.ln2_r, transcript,
    );

    // 5. MLP fc1 (EF)
    let mlp_fc1_proof = prove_matmul_succinct_ef(
        &weights.w_fc1, &trace.ln2_out, &trace.fc1_out,
        d_ff, d_model, Some(&weights.b_fc1), transcript,
    );

    // 6. GELU (EF — 124-bit lookup challenges)
    let mlp_gelu_proof = prove_gelu_ef(&trace.fc1_out, &trace.gelu_out, gelu_table, transcript);

    // 7. MLP fc2 (EF)
    let mlp_fc2_proof = prove_matmul_succinct_ef(
        &weights.w_fc2, &trace.gelu_out, &trace.fc2_out,
        d_model, d_ff, Some(&weights.b_fc2), transcript,
    );

    // 8. Residual 2 (EF)
    let res2_point = transcript.squeeze_ef_many(log_d);
    let residual2_proof = prove_add_ef(&trace.h, &trace.fc2_out, &trace.output, &res2_point, transcript);

    let output = trace.output;
    (TransformerLayerProofEF {
        ln1_proof, attn_proof, residual1_proof,
        ln2_proof, mlp_fc1_proof, mlp_gelu_proof, mlp_fc2_proof, residual2_proof,
        w_attn_commitment, w_fc1_commitment, w_fc2_commitment,
    }, output)
}

/// Verify a GPT-2 transformer layer proof with EF challenges.
#[allow(dead_code)]
pub fn verify_transformer_layer_ef(
    proof: &TransformerLayerProofEF,
    x: &[F],
    y: &[F],
    weights: &TransformerLayerWeights,
    d_model: usize,
    d_ff: usize,
    gelu_table: &LookupTable,
    transcript: &mut crate::proving::sumcheck::Transcript,
) -> bool {
    let trace = forward_pass(x, weights, d_model, d_ff, gelu_table);
    if trace.output != y {
        eprintln!("GPT2 EF: output mismatch");
        return false;
    }

    // Absorb weight commitment roots
    transcript.absorb_bytes(&proof.w_attn_commitment.root);
    transcript.absorb_bytes(&proof.w_fc1_commitment.root);
    transcript.absorb_bytes(&proof.w_fc2_commitment.root);

    // 1. LayerNorm 1 (EF)
    if !verify_layernorm_ef(&proof.ln1_proof, &trace.ln1_out, &weights.ln1_beta, d_model, transcript) {
        eprintln!("GPT2 EF: LN1 failed"); return false;
    }

    // 2. Attention matmul (EF)
    let attn_r = verify_matmul_succinct_ef(
        &proof.attn_proof, &proof.w_attn_commitment, &trace.attn_out,
        d_model, d_model, None, transcript,
    );
    if !attn_r.valid { eprintln!("GPT2 EF: attn matmul failed"); return false; }

    // 3. Residual 1 (EF)
    let log_d = log2_ceil(d_model);
    let res1_point = transcript.squeeze_ef_many(log_d);
    if !verify_add_ef(&proof.residual1_proof, &res1_point, transcript) {
        eprintln!("GPT2 EF: residual1 failed"); return false;
    }

    // 4. LayerNorm 2 (EF)
    if !verify_layernorm_ef(&proof.ln2_proof, &trace.ln2_out, &weights.ln2_beta, d_model, transcript) {
        eprintln!("GPT2 EF: LN2 failed"); return false;
    }

    // 5. MLP fc1 (EF)
    let fc1_r = verify_matmul_succinct_ef(
        &proof.mlp_fc1_proof, &proof.w_fc1_commitment, &trace.fc1_out,
        d_ff, d_model, Some(&weights.b_fc1), transcript,
    );
    if !fc1_r.valid { eprintln!("GPT2 EF: fc1 failed"); return false; }

    // 6. GELU (EF)
    if !verify_gelu_ef(&proof.mlp_gelu_proof, trace.fc1_out.len(), transcript) {
        eprintln!("GPT2 EF: GELU failed"); return false;
    }

    // 7. MLP fc2 (EF)
    let fc2_r = verify_matmul_succinct_ef(
        &proof.mlp_fc2_proof, &proof.w_fc2_commitment, &trace.fc2_out,
        d_model, d_ff, Some(&weights.b_fc2), transcript,
    );
    if !fc2_r.valid { eprintln!("GPT2 EF: fc2 failed"); return false; }

    // 8. Residual 2 (EF)
    let res2_point = transcript.squeeze_ef_many(log_d);
    if !verify_add_ef(&proof.residual2_proof, &res2_point, transcript) {
        eprintln!("GPT2 EF: residual2 failed"); return false;
    }

    true
}

/// Weights for a full GPT-2 model.
pub struct GPT2Weights {
    pub layers: Vec<TransformerLayerWeights>,
    pub final_ln_gamma: Vec<F>,
    pub final_ln_beta: Vec<F>,
    pub lm_head: Vec<F>, // vocab_size × d_model (row-major)
}

/// Proof for full GPT-2 inference.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GPT2Proof {
    pub layer_proofs: Vec<TransformerLayerProof>,
    pub final_ln_proof: Option<LayerNormProof>,
    pub lm_head_proof: Option<SuccinctMatmulProof>,
    pub lm_head_commitment: Option<WeightCommitment>,
}

/// Compute LayerNorm forward pass (standalone, for chaining between layers).
fn layernorm_forward(x: &[F], gamma: &[F], beta: &[F], d: usize) -> (Vec<F>, F, F) {
    let sum_x: F = x.iter().copied().sum();
    let d_inv = F::from_canonical_u32(d as u32).inverse();
    let mu = sum_x * d_inv;
    let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
    let sum_sq: F = xc.iter().map(|&v| v * v).sum();
    let r = compute_r(sum_sq, d);
    let out: Vec<F> = xc
        .iter()
        .zip(gamma.iter())
        .zip(beta.iter())
        .map(|((&xci, &gi), &bi)| gi * xci * r + bi)
        .collect();
    (out, mu, r)
}

/// Prove a full GPT-2 model: N transformer layers → final LayerNorm → LM head matmul.
pub fn prove_gpt2(
    x: &[F],
    weights: &GPT2Weights,
    d_model: usize,
    d_ff: usize,
    vocab_size: usize,
    gelu_table: &LookupTable,
    transcript: &mut crate::proving::sumcheck::Transcript,
) -> GPT2Proof {
    // Chain N transformer layers
    let mut current = x.to_vec();
    let mut layer_proofs = Vec::new();
    for layer_weights in &weights.layers {
        let (mut proof, output) = prove_transformer_layer(&current, layer_weights, d_model, d_ff, gelu_table, transcript);
        proof.strip_redundant();
        current = output;
        layer_proofs.push(proof);
    }

    // Final LayerNorm + LM head (skip if vocab_size == 0)
    let (final_ln_proof, lm_head_proof, lm_head_commitment) = if vocab_size > 0 {
        let (final_ln_out, final_mu, final_r) =
            layernorm_forward(&current, &weights.final_ln_gamma, &weights.final_ln_beta, d_model);
        let ln_proof = prove_layernorm(
            &current,
            &weights.final_ln_gamma,
            &weights.final_ln_beta,
            &final_ln_out,
            final_mu,
            final_r,
            transcript,
        );

        let logits = matmul_forward(&weights.lm_head, &final_ln_out, vocab_size, d_model, None);

        let log_m = log2_ceil(vocab_size);
        let log_n = log2_ceil(d_model);
        let m_pad = 1 << log_m;
        let n_pad = 1 << log_n;
        let mut lm_head_padded = vec![F::zero(); m_pad * n_pad];
        for i in 0..vocab_size {
            for j in 0..d_model {
                lm_head_padded[i * n_pad + j] = weights.lm_head[i * d_model + j];
            }
        }
        let commitment = commit_weights_fast(&lm_head_padded);

        let matmul_proof = prove_matmul_succinct_prepadded(
            &lm_head_padded, &final_ln_out, &logits,
            vocab_size, d_model, None, transcript,
        );

        (Some(ln_proof), Some(matmul_proof), Some(commitment))
    } else {
        (None, None, None)
    };

    GPT2Proof {
        layer_proofs,
        final_ln_proof,
        lm_head_proof,
        lm_head_commitment,
    }
}

/// Verify a full GPT-2 proof.
pub fn verify_gpt2(
    proof: &GPT2Proof,
    x: &[F],
    logits: &[F],
    weights: &GPT2Weights,
    d_model: usize,
    d_ff: usize,
    vocab_size: usize,
    gelu_table: &LookupTable,
    transcript: &mut crate::proving::sumcheck::Transcript,
) -> bool {
    // Recompute forward pass through all layers to get intermediates
    let mut current = x.to_vec();

    // Verify each transformer layer
    for (i, (layer_proof, layer_weights)) in proof
        .layer_proofs
        .iter()
        .zip(weights.layers.iter())
        .enumerate()
    {
        let trace = forward_pass(&current, layer_weights, d_model, d_ff, gelu_table);
        if !verify_transformer_layer(
            layer_proof,
            &current,
            &trace.output,
            layer_weights,
            d_model,
            d_ff,
            gelu_table,
            transcript,
        ) {
            eprintln!("GPT2: layer {} verification failed", i);
            return false;
        }
        current = trace.output;
    }

    // Verify final LayerNorm + LM head (skip if vocab_size == 0)
    if vocab_size > 0 {
        let (final_ln_out, _final_mu, _final_r) =
            layernorm_forward(&current, &weights.final_ln_gamma, &weights.final_ln_beta, d_model);
        if let Some(ref ln_proof) = proof.final_ln_proof {
            if !verify_layernorm(ln_proof, &final_ln_out, &weights.final_ln_beta, d_model, transcript) {
                eprintln!("GPT2: final LayerNorm verification failed");
                return false;
            }
        }

        let computed_logits = matmul_forward(&weights.lm_head, &final_ln_out, vocab_size, d_model, None);
        if computed_logits != logits {
            eprintln!("GPT2: logits mismatch");
            return false;
        }

        if let (Some(ref lm_proof), Some(ref lm_commit)) = (&proof.lm_head_proof, &proof.lm_head_commitment) {
            let lm_result = verify_matmul_succinct(lm_proof, lm_commit, logits, vocab_size, d_model, None, transcript);
            if !lm_result.valid {
                eprintln!("GPT2: LM head matmul verification failed");
                return false;
            }
        }
    }

    true
}


// ===== Forward pass helpers (moved from pipeline.rs) =====

pub(crate) fn convert_gpt2_weights(desc: &GPT2Desc) -> GPT2Weights {
    let to_f = |v: &[u32]| -> Vec<F> { v.iter().map(|&x| F::from_canonical_u32(x)).collect() };

    let layers: Vec<TransformerLayerWeights> = desc
        .layers
        .iter()
        .map(|l| TransformerLayerWeights {
            ln1_gamma: to_f(&l.ln1_gamma),
            ln1_beta: to_f(&l.ln1_beta),
            w_attn: to_f(&l.w_attn),
            ln2_gamma: to_f(&l.ln2_gamma),
            ln2_beta: to_f(&l.ln2_beta),
            w_fc1: to_f(&l.w_fc1),
            b_fc1: to_f(&l.b_fc1),
            w_fc2: to_f(&l.w_fc2),
            b_fc2: to_f(&l.b_fc2),
        })
        .collect();

    GPT2Weights {
        layers,
        final_ln_gamma: to_f(&desc.final_ln_gamma),
        final_ln_beta: to_f(&desc.final_ln_beta),
        lm_head: to_f(&desc.lm_head),
    }
}

/// Simple forward pass through a transformer layer (returns output only).
pub(crate) fn transformer_forward_pass(
    x: &[F],
    weights: &TransformerLayerWeights,
    d_model: usize,
    d_ff: usize,
    gelu_table: &LookupTable,
) -> Vec<F> {
    // LN1
    let (ln1_out, _, _, _) =
        layernorm_forward_simple(x, &weights.ln1_gamma, &weights.ln1_beta, d_model);
    // Attention (simplified matmul)
    let attn_out = matmul_forward_simple(&weights.w_attn, &ln1_out, d_model, d_model, None);
    // Residual 1
    let h: Vec<F> = x.iter().zip(attn_out.iter()).map(|(&a, &b)| a + b).collect();
    // LN2
    let (ln2_out, _, _, _) =
        layernorm_forward_simple(&h, &weights.ln2_gamma, &weights.ln2_beta, d_model);
    // MLP fc1
    let fc1_out = matmul_forward_simple(
        &weights.w_fc1,
        &ln2_out,
        d_ff,
        d_model,
        Some(&weights.b_fc1),
    );
    // GELU
    let gelu_out: Vec<F> = fc1_out.iter().map(|&v| lookup_gelu_simple(v, gelu_table)).collect();
    // MLP fc2
    let fc2_out = matmul_forward_simple(
        &weights.w_fc2,
        &gelu_out,
        d_model,
        d_ff,
        Some(&weights.b_fc2),
    );
    // Residual 2
    h.iter().zip(fc2_out.iter()).map(|(&a, &b)| a + b).collect()
}

/// Returns (output, mu, r, perturbation).
/// When d/sum_sq is QNR in M31, perturbs x[0] by a small delta until it becomes QR.
/// This changes the LN output negligibly (1 element out of 768) but makes the proof work.
pub(crate) fn layernorm_forward_simple(x: &[F], gamma: &[F], beta: &[F], d: usize) -> (Vec<F>, F, F, i32) {
    let d_field = F::from_canonical_u32(d as u32);
    let d_inv = d_field.inverse();

    // Try original input first, then small perturbations to x[0]
    for delta in 0i32..100 {
        for sign in &[1i32, -1i32] {
            let perturbation = if delta == 0 { 0 } else { delta * sign };
            let mut x_perturbed = x.to_vec();
            if perturbation != 0 {
                let adj = if perturbation > 0 {
                    F::from_canonical_u32(perturbation as u32)
                } else {
                    F::zero() - F::from_canonical_u32((-perturbation) as u32)
                };
                x_perturbed[0] = x_perturbed[0] + adj;
            }

            let sum_x: F = x_perturbed.iter().copied().sum();
            let mu = sum_x * d_inv;
            let xc: Vec<F> = x_perturbed.iter().map(|&xi| xi - mu).collect();
            let sum_sq: F = xc.iter().map(|&v| v * v).sum();
            if sum_sq == F::zero() { continue; }
            let target = d_field * sum_sq.inverse();

            if is_qr_m31(target) {
                let r = mod_sqrt_m31(target);
                let out: Vec<F> = xc
                    .iter()
                    .zip(gamma.iter())
                    .zip(beta.iter())
                    .map(|((&xci, &gi), &bi)| gi * xci * r + bi)
                    .collect();
                return (out, mu, r, perturbation);
            }

            if delta == 0 { break; } // don't try sign=-1 for delta=0
        }
    }
    // Fallback (should never happen — probability 2^-100)
    panic!("layernorm_forward_simple: could not find QR perturbation in 200 attempts");
}


pub(crate) fn matmul_forward_simple(w: &[F], x: &[F], m: usize, n: usize, bias: Option<&[F]>) -> Vec<F> {
    let mut y = Vec::with_capacity(m);
    for i in 0..m {
        let mut acc = F::zero();
        for j in 0..n {
            acc += w[i * n + j] * x[j];
        }
        if let Some(b) = bias {
            acc += b[i];
        }
        y.push(acc);
    }
    y
}

pub(crate) fn lookup_gelu_simple(v: F, table: &LookupTable) -> F {
    let key = v.as_canonical_u32();
    for &(inp, out) in &table.entries {
        if inp == key {
            return F::from_canonical_u32(out);
        }
    }
    F::zero()
}

// mod_sqrt_m31 and is_qr_m31 imported from crate::field::common

/// Build a small GELU table (256 entries, i8 range) for tests and small models.
pub(crate) fn build_small_gelu_table(scale: i32) -> LookupTable {
    let s = scale as f64;
    let mut entries = Vec::with_capacity(256);
    for raw in 0u32..256 {
        let input_i16 = raw as i8 as i16;
        let x = input_i16 as f64 / s;
        let y = 0.5
            * x
            * (1.0
                + ((2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
        let output_i16 = {
            let v = (y * s).round();
            if v > 32767.0 {
                32767i16
            } else if v < -32768.0 {
                -32768i16
            } else {
                v as i16
            }
        };
        entries.push((
            to_field(input_i16 as i64).as_canonical_u32(),
            to_field(output_i16 as i64).as_canonical_u32(),
        ));
    }
    LookupTable {
        name: "gelu_small".to_string(),
        entries,
        commitment: WeightCommitment {
            root: [0u8; 32],
            num_weights: 256,
            log_height: 8,
            kind: crate::proving::weight_commitment::WeightDigestKind::Blake3Fast,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proving::lookup::LookupTable;
    use crate::proving::sumcheck::Transcript;
    use crate::proving::weight_commitment::WeightCommitment;
    use p3_field::{AbstractField, Field};

    /// Build a small GELU table (256 entries, i8 range) for faster tests.
    fn build_small_gelu_table(scale: i32) -> LookupTable {
        let s = scale as f64;
        let mut entries = Vec::with_capacity(256);
        for raw in 0u32..256 {
            let input_i16 = raw as i8 as i16;
            let x = input_i16 as f64 / s;
            let y = 0.5 * x * (1.0 + ((2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
            let output_i16 = {
                let v = (y * s).round();
                if v > 32767.0 { 32767i16 }
                else if v < -32768.0 { -32768i16 }
                else { v as i16 }
            };
            entries.push((
                to_field(input_i16 as i64).as_canonical_u32(),
                to_field(output_i16 as i64).as_canonical_u32(),
            ));
        }
        LookupTable {
            name: "gelu_small".to_string(),
            entries,
            commitment: WeightCommitment {
                root: [0u8; 32],
                num_weights: 256,
                log_height: 8,
                kind: crate::proving::weight_commitment::WeightDigestKind::Blake3Fast,
            },
        }
    }

    /// Check if a is a quadratic residue in M31.
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

    /// Create weights that keep all intermediates in the i8 GELU table range.
    /// Key insight: fc1_out must be to_field(small_int) for GELU lookup.
    /// Solution: zero fc1 weights, small fc1 biases (directly controls fc1_out).
    fn make_test_weights(d_model: usize, d_ff: usize) -> TransformerLayerWeights {
        TransformerLayerWeights {
            ln1_gamma: vec![F::one(); d_model],
            ln1_beta: vec![F::zero(); d_model],
            // Small attention weights (identity-like)
            w_attn: {
                let mut w = vec![F::zero(); d_model * d_model];
                for i in 0..d_model {
                    w[i * d_model + i] = F::one(); // identity matrix
                }
                w
            },
            ln2_gamma: vec![F::one(); d_model],
            ln2_beta: vec![F::zero(); d_model],
            // Zero fc1 weights so fc1_out = b_fc1 (small, controlled values)
            w_fc1: vec![F::zero(); d_ff * d_model],
            // Small biases in i8 range for GELU table lookup
            b_fc1: (0..d_ff)
                .map(|i| to_field((i as i64 % 20) - 10)) // values in [-10, 9]
                .collect(),
            // Zero fc2 weights so fc2_out = b_fc2
            w_fc2: vec![F::zero(); d_model * d_ff],
            b_fc2: vec![F::zero(); d_model],
        }
    }

    /// Find input values where LayerNorm's d/sum_sq is a QR in M31.
    /// This is needed because not all sum_sq values have a square root in M31.
    #[allow(dead_code)]
    fn find_valid_input(d_model: usize) -> Vec<F> {
        // Try different inputs until both LN1 and LN2 have valid r values.
        // With gamma=1, beta=0, LN output = xc * r, so LN2 input = x + W_attn @ (xc * r).
        // We need to check both LN1 and LN2.
        // For simplicity, use inputs that are symmetric around mean to get nice sum_sq.
        //
        // d=8: x = [1,2,3,4,5,6,7,8], mean=4.5
        // xc = [-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5]
        // sum_sq = 2*(3.5^2+2.5^2+1.5^2+0.5^2) = 2*(12.25+6.25+2.25+0.25) = 42
        // d/sum_sq = 8/42 = 4/21. Need sqrt(4/21) in M31.
        //
        // Just try several options.
        for offset in 0u32..100 {
            let x: Vec<F> = (0..d_model)
                .map(|i| F::from_canonical_u32(i as u32 + 1 + offset))
                .collect();
            let sum_x: F = x.iter().copied().sum();
            let d_inv = F::from_canonical_u32(d_model as u32).inverse();
            let mu = sum_x * d_inv;
            let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
            let sum_sq: F = xc.iter().map(|&v| v * v).sum();
            if sum_sq == F::zero() {
                continue;
            }
            let target = F::from_canonical_u32(d_model as u32) * sum_sq.inverse();
            if is_qr_m31(target) {
                return x;
            }
        }
        panic!("Could not find valid input for LayerNorm");
    }

    #[test]
    fn test_transformer_layer_simple() {
        let d_model = 8;
        let d_ff = 16;
        let gelu_table = build_small_gelu_table(10);
        let weights = make_test_weights(d_model, d_ff);

        // Find input where LN1 has valid r. We also need LN2 to have valid r.
        // Try inputs until the full forward pass works (both LN1 and LN2 have QR).
        let mut x = None;
        for offset in 0u32..200 {
            let candidate: Vec<F> = (0..d_model)
                .map(|i| F::from_canonical_u32(i as u32 + 1 + offset))
                .collect();

            // Check LN1
            let sum_x: F = candidate.iter().copied().sum();
            let d_inv = F::from_canonical_u32(d_model as u32).inverse();
            let mu = sum_x * d_inv;
            let xc: Vec<F> = candidate.iter().map(|&xi| xi - mu).collect();
            let sum_sq: F = xc.iter().map(|&v| v * v).sum();
            if sum_sq == F::zero() {
                continue;
            }
            let target1 = F::from_canonical_u32(d_model as u32) * sum_sq.inverse();
            if !is_qr_m31(target1) {
                continue;
            }

            // Run forward to check LN2
            let r1 = compute_r(sum_sq, d_model);
            let ln1_out: Vec<F> = xc
                .iter()
                .zip(weights.ln1_gamma.iter())
                .zip(weights.ln1_beta.iter())
                .map(|((&xci, &gi), &bi)| gi * xci * r1 + bi)
                .collect();
            let attn_out = matmul_forward(&weights.w_attn, &ln1_out, d_model, d_model, None);
            let h: Vec<F> = candidate
                .iter()
                .zip(attn_out.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let sum_h: F = h.iter().copied().sum();
            let mu2 = sum_h * d_inv;
            let hc: Vec<F> = h.iter().map(|&hi| hi - mu2).collect();
            let sum_sq2: F = hc.iter().map(|&v| v * v).sum();
            if sum_sq2 == F::zero() {
                continue;
            }
            let target2 = F::from_canonical_u32(d_model as u32) * sum_sq2.inverse();
            if !is_qr_m31(target2) {
                continue;
            }

            // Also need fc1_out values to be in gelu table range (-128..127 as i8)
            let r2 = compute_r(sum_sq2, d_model);
            let ln2_out: Vec<F> = hc
                .iter()
                .zip(weights.ln2_gamma.iter())
                .zip(weights.ln2_beta.iter())
                .map(|((&hci, &gi), &bi)| gi * hci * r2 + bi)
                .collect();
            let fc1_out = matmul_forward(
                &weights.w_fc1,
                &ln2_out,
                d_ff,
                d_model,
                Some(&weights.b_fc1),
            );

            // Check all fc1 outputs are in the table
            let all_in_table = fc1_out.iter().all(|&v| {
                let key = v.as_canonical_u32();
                gelu_table.entries.iter().any(|&(inp, _)| inp == key)
            });
            if !all_in_table {
                continue;
            }

            x = Some(candidate);
            break;
        }

        let x = x.expect("Could not find valid input for full transformer layer");

        // Prove (returns output, no separate forward pass needed)
        let mut pt = Transcript::new(b"transformer-test");
        let (proof, output) = prove_transformer_layer(
            &x, &weights, d_model, d_ff, &gelu_table, &mut pt,
        );

        // Verify
        let mut vt = Transcript::new(b"transformer-test");
        assert!(
            verify_transformer_layer(
                &proof,
                &x,
                &output,
                &weights,
                d_model,
                d_ff,
                &gelu_table,
                &mut vt,
            ),
            "Transformer layer verification should pass"
        );
    }

    /// Find a valid input for the full GPT-2 model (all LayerNorms must have QR).
    /// Uses zero fc1/fc2 weights so intermediates stay controlled.
    fn find_valid_gpt2_input(
        d_model: usize,
        d_ff: usize,
        weights: &GPT2Weights,
        gelu_table: &LookupTable,
    ) -> Vec<F> {
        for offset in 0u32..500 {
            let candidate: Vec<F> = (0..d_model)
                .map(|i| F::from_canonical_u32(i as u32 + 1 + offset))
                .collect();

            let valid = (|| {
                let mut current = candidate.clone();
                // Check all transformer layers
                for layer_weights in &weights.layers {
                    // Check LN1
                    let d_inv = F::from_canonical_u32(d_model as u32).inverse();
                    let sum_x: F = current.iter().copied().sum();
                    let mu = sum_x * d_inv;
                    let xc: Vec<F> = current.iter().map(|&xi| xi - mu).collect();
                    let sum_sq: F = xc.iter().map(|&v| v * v).sum();
                    if sum_sq == F::zero() { return false; }
                    let target = F::from_canonical_u32(d_model as u32) * sum_sq.inverse();
                    if !is_qr_m31(target) { return false; }

                    // Run full layer forward
                    let trace = forward_pass(&current, layer_weights, d_model, d_ff, gelu_table);

                    // Check LN2 was valid (check h's variance)
                    let sum_h: F = trace.h.iter().copied().sum();
                    let mu2 = sum_h * d_inv;
                    let hc: Vec<F> = trace.h.iter().map(|&hi| hi - mu2).collect();
                    let sum_sq2: F = hc.iter().map(|&v| v * v).sum();
                    if sum_sq2 == F::zero() { return false; }
                    let target2 = F::from_canonical_u32(d_model as u32) * sum_sq2.inverse();
                    if !is_qr_m31(target2) { return false; }

                    // Check fc1_out in gelu table
                    let all_in_table = trace.fc1_out.iter().all(|&v| {
                        let key = v.as_canonical_u32();
                        gelu_table.entries.iter().any(|&(inp, _)| inp == key)
                    });
                    if !all_in_table { return false; }

                    current = trace.output;
                }

                // Check final LayerNorm
                let d_inv = F::from_canonical_u32(d_model as u32).inverse();
                let sum_x: F = current.iter().copied().sum();
                let mu = sum_x * d_inv;
                let xc: Vec<F> = current.iter().map(|&xi| xi - mu).collect();
                let sum_sq: F = xc.iter().map(|&v| v * v).sum();
                if sum_sq == F::zero() { return false; }
                let target = F::from_canonical_u32(d_model as u32) * sum_sq.inverse();
                if !is_qr_m31(target) { return false; }

                true
            })();

            if valid {
                return candidate;
            }
        }
        panic!("Could not find valid GPT-2 input");
    }

    fn make_gpt2_weights(n_layers: usize, d_model: usize, d_ff: usize, vocab_size: usize) -> GPT2Weights {
        let layers: Vec<TransformerLayerWeights> = (0..n_layers)
            .map(|_| make_test_weights(d_model, d_ff))
            .collect();
        GPT2Weights {
            layers,
            final_ln_gamma: vec![F::one(); d_model],
            final_ln_beta: vec![F::zero(); d_model],
            // Small identity-like lm_head (vocab_size × d_model)
            lm_head: {
                let mut w = vec![F::zero(); vocab_size * d_model];
                for i in 0..vocab_size.min(d_model) {
                    w[i * d_model + i] = F::one();
                }
                w
            },
        }
    }

    #[test]
    fn test_gpt2_2_layers() {
        let d_model = 8;
        let d_ff = 16;
        let vocab_size = 8;
        let n_layers = 2;
        let gelu_table = build_small_gelu_table(10);
        let weights = make_gpt2_weights(n_layers, d_model, d_ff, vocab_size);

        let x = find_valid_gpt2_input(d_model, d_ff, &weights, &gelu_table);

        // Compute expected logits
        let mut current = x.clone();
        for layer_weights in &weights.layers {
            let trace = forward_pass(&current, layer_weights, d_model, d_ff, &gelu_table);
            current = trace.output;
        }
        let (final_ln_out, _, _) =
            layernorm_forward(&current, &weights.final_ln_gamma, &weights.final_ln_beta, d_model);
        let logits = matmul_forward(&weights.lm_head, &final_ln_out, vocab_size, d_model, None);

        // Prove
        let mut pt = Transcript::new(b"gpt2-2layer-test");
        let proof = prove_gpt2(&x, &weights, d_model, d_ff, vocab_size, &gelu_table, &mut pt);

        // Verify
        let mut vt = Transcript::new(b"gpt2-2layer-test");
        assert!(
            verify_gpt2(&proof, &x, &logits, &weights, d_model, d_ff, vocab_size, &gelu_table, &mut vt),
            "GPT-2 2-layer verification should pass"
        );
    }

    #[test]
    fn test_gpt2_12_layers() {
        let d_model = 8;
        let d_ff = 16;
        let vocab_size = 8;
        let n_layers = 12;
        let gelu_table = build_small_gelu_table(10);
        let weights = make_gpt2_weights(n_layers, d_model, d_ff, vocab_size);

        let x = find_valid_gpt2_input(d_model, d_ff, &weights, &gelu_table);

        // Compute expected logits
        let mut current = x.clone();
        for layer_weights in &weights.layers {
            let trace = forward_pass(&current, layer_weights, d_model, d_ff, &gelu_table);
            current = trace.output;
        }
        let (final_ln_out, _, _) =
            layernorm_forward(&current, &weights.final_ln_gamma, &weights.final_ln_beta, d_model);
        let logits = matmul_forward(&weights.lm_head, &final_ln_out, vocab_size, d_model, None);

        // Prove
        let mut pt = Transcript::new(b"gpt2-12layer-test");
        let proof = prove_gpt2(&x, &weights, d_model, d_ff, vocab_size, &gelu_table, &mut pt);

        // Verify
        let mut vt = Transcript::new(b"gpt2-12layer-test");
        assert!(
            verify_gpt2(&proof, &x, &logits, &weights, d_model, d_ff, vocab_size, &gelu_table, &mut vt),
            "GPT-2 12-layer verification should pass"
        );
    }

    #[test]
    fn test_transformer_layer_tampered_output() {
        let d_model = 8;
        let d_ff = 16;
        let gelu_table = build_small_gelu_table(10);
        let weights = make_test_weights(d_model, d_ff);

        // Find valid input (same search as above)
        let mut x = None;
        for offset in 0u32..200 {
            let candidate: Vec<F> = (0..d_model)
                .map(|i| F::from_canonical_u32(i as u32 + 1 + offset))
                .collect();
            let sum_x: F = candidate.iter().copied().sum();
            let d_inv = F::from_canonical_u32(d_model as u32).inverse();
            let mu = sum_x * d_inv;
            let xc: Vec<F> = candidate.iter().map(|&xi| xi - mu).collect();
            let sum_sq: F = xc.iter().map(|&v| v * v).sum();
            if sum_sq == F::zero() {
                continue;
            }
            let target1 = F::from_canonical_u32(d_model as u32) * sum_sq.inverse();
            if !is_qr_m31(target1) {
                continue;
            }
            let r1 = compute_r(sum_sq, d_model);
            let ln1_out: Vec<F> = xc
                .iter()
                .zip(weights.ln1_gamma.iter())
                .zip(weights.ln1_beta.iter())
                .map(|((&xci, &gi), &bi)| gi * xci * r1 + bi)
                .collect();
            let attn_out = matmul_forward(&weights.w_attn, &ln1_out, d_model, d_model, None);
            let h: Vec<F> = candidate
                .iter()
                .zip(attn_out.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let sum_h: F = h.iter().copied().sum();
            let mu2 = sum_h * d_inv;
            let hc: Vec<F> = h.iter().map(|&hi| hi - mu2).collect();
            let sum_sq2: F = hc.iter().map(|&v| v * v).sum();
            if sum_sq2 == F::zero() {
                continue;
            }
            let target2 = F::from_canonical_u32(d_model as u32) * sum_sq2.inverse();
            if !is_qr_m31(target2) {
                continue;
            }
            let r2 = compute_r(sum_sq2, d_model);
            let ln2_out: Vec<F> = hc
                .iter()
                .zip(weights.ln2_gamma.iter())
                .zip(weights.ln2_beta.iter())
                .map(|((&hci, &gi), &bi)| gi * hci * r2 + bi)
                .collect();
            let fc1_out = matmul_forward(
                &weights.w_fc1,
                &ln2_out,
                d_ff,
                d_model,
                Some(&weights.b_fc1),
            );
            let all_in_table = fc1_out.iter().all(|&v| {
                let key = v.as_canonical_u32();
                gelu_table.entries.iter().any(|&(inp, _)| inp == key)
            });
            if !all_in_table {
                continue;
            }
            x = Some(candidate);
            break;
        }

        let x = x.expect("Could not find valid input");

        // Prove with correct values (returns output, no separate forward pass needed)
        let mut pt = Transcript::new(b"transformer-tamper");
        let (proof, output) = prove_transformer_layer(
            &x, &weights, d_model, d_ff, &gelu_table, &mut pt,
        );

        // Tamper with output
        let mut tampered_output = output.clone();
        tampered_output[0] = tampered_output[0] + F::one();

        // Verify with tampered output should fail
        let mut vt = Transcript::new(b"transformer-tamper");
        assert!(
            !verify_transformer_layer(
                &proof,
                &x,
                &tampered_output,
                &weights,
                d_model,
                d_ff,
                &gelu_table,
                &mut vt,
            ),
            "Verification should reject tampered output"
        );
    }

    #[test]
    fn test_transformer_layer_tampered_subproof() {
        let d_model = 8;
        let d_ff = 16;
        let gelu_table = build_small_gelu_table(10);
        let weights = make_test_weights(d_model, d_ff);

        // Find valid input (same search as test_transformer_layer_simple)
        let mut x = None;
        for offset in 0u32..200 {
            let candidate: Vec<F> = (0..d_model)
                .map(|i| F::from_canonical_u32(i as u32 + 1 + offset))
                .collect();
            let sum_x: F = candidate.iter().copied().sum();
            let d_inv = F::from_canonical_u32(d_model as u32).inverse();
            let mu = sum_x * d_inv;
            let xc: Vec<F> = candidate.iter().map(|&xi| xi - mu).collect();
            let sum_sq: F = xc.iter().map(|&v| v * v).sum();
            if sum_sq == F::zero() {
                continue;
            }
            let target1 = F::from_canonical_u32(d_model as u32) * sum_sq.inverse();
            if !is_qr_m31(target1) {
                continue;
            }
            let r1 = compute_r(sum_sq, d_model);
            let ln1_out: Vec<F> = xc
                .iter()
                .zip(weights.ln1_gamma.iter())
                .zip(weights.ln1_beta.iter())
                .map(|((&xci, &gi), &bi)| gi * xci * r1 + bi)
                .collect();
            let attn_out = matmul_forward(&weights.w_attn, &ln1_out, d_model, d_model, None);
            let h: Vec<F> = candidate
                .iter()
                .zip(attn_out.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let sum_h: F = h.iter().copied().sum();
            let mu2 = sum_h * d_inv;
            let hc: Vec<F> = h.iter().map(|&hi| hi - mu2).collect();
            let sum_sq2: F = hc.iter().map(|&v| v * v).sum();
            if sum_sq2 == F::zero() {
                continue;
            }
            let target2 = F::from_canonical_u32(d_model as u32) * sum_sq2.inverse();
            if !is_qr_m31(target2) {
                continue;
            }
            let r2 = compute_r(sum_sq2, d_model);
            let ln2_out: Vec<F> = hc
                .iter()
                .zip(weights.ln2_gamma.iter())
                .zip(weights.ln2_beta.iter())
                .map(|((&hci, &gi), &bi)| gi * hci * r2 + bi)
                .collect();
            let fc1_out = matmul_forward(
                &weights.w_fc1,
                &ln2_out,
                d_ff,
                d_model,
                Some(&weights.b_fc1),
            );
            let all_in_table = fc1_out.iter().all(|&v| {
                let key = v.as_canonical_u32();
                gelu_table.entries.iter().any(|&(inp, _)| inp == key)
            });
            if !all_in_table {
                continue;
            }
            x = Some(candidate);
            break;
        }

        let x = x.expect("Could not find valid input");

        // Prove with correct values
        let mut pt = Transcript::new(b"transformer-subproof-tamper");
        let (mut proof, output) = prove_transformer_layer(
            &x, &weights, d_model, d_ff, &gelu_table, &mut pt,
        );

        // Tamper with an intermediate sub-proof field:
        // Modify w_at_rs inside the attention matmul proof.
        // This is a claimed MLE evaluation — flipping it should break verification
        // at the sumcheck level, not just the output comparison.
        let original = proof.attn_proof.matmul_proof.w_at_rs;
        proof.attn_proof.matmul_proof.w_at_rs = original.wrapping_add(1) % ((1u32 << 31) - 1);

        // Verify with correct output but tampered sub-proof should fail
        let mut vt = Transcript::new(b"transformer-subproof-tamper");
        assert!(
            !verify_transformer_layer(
                &proof,
                &x,
                &output,
                &weights,
                d_model,
                d_ff,
                &gelu_table,
                &mut vt,
            ),
            "Verification should reject tampered sub-proof (attn matmul w_at_rs)"
        );
    }

    // ===== Extension field GPT-2 tests =====

    #[test]
    fn test_transformer_layer_ef_simple() {
        let d_model = 8;
        let d_ff = 16;
        let gelu_table = build_small_gelu_table(10);
        let weights = make_test_weights(d_model, d_ff);

        let mut x = None;
        for offset in 0u32..200 {
            let candidate: Vec<F> = (0..d_model)
                .map(|i| F::from_canonical_u32(i as u32 + 1 + offset))
                .collect();

            let sum_x: F = candidate.iter().copied().sum();
            let d_inv = F::from_canonical_u32(d_model as u32).inverse();
            let mu = sum_x * d_inv;
            let xc: Vec<F> = candidate.iter().map(|&xi| xi - mu).collect();
            let sum_sq: F = xc.iter().map(|&v| v * v).sum();
            if sum_sq == F::zero() || !is_qr_m31(F::from_canonical_u32(d_model as u32) * sum_sq.inverse()) {
                continue;
            }

            // Check LN2 too (after forward pass)
            let trace = forward_pass(&candidate, &weights, d_model, d_ff, &gelu_table);
            let sum_x2: F = trace.h.iter().copied().sum();
            let mu2 = sum_x2 * d_inv;
            let xc2: Vec<F> = trace.h.iter().map(|&xi| xi - mu2).collect();
            let sum_sq2: F = xc2.iter().map(|&v| v * v).sum();
            if sum_sq2 == F::zero() || !is_qr_m31(F::from_canonical_u32(d_model as u32) * sum_sq2.inverse()) {
                continue;
            }

            x = Some(candidate);
            break;
        }
        let x = x.expect("Could not find valid GPT-2 EF input");

        let mut pt = Transcript::new(b"gpt2-ef-test");
        let (proof, output) = prove_transformer_layer_ef(
            &x, &weights, d_model, d_ff, &gelu_table, &mut pt,
        );

        let mut vt = Transcript::new(b"gpt2-ef-test");
        assert!(
            verify_transformer_layer_ef(
                &proof, &x, &output, &weights, d_model, d_ff, &gelu_table, &mut vt,
            ),
            "GPT-2 EF layer verification should pass"
        );
    }

    #[test]
    fn test_transformer_layer_ef_tampered() {
        let d_model = 8;
        let d_ff = 16;
        let gelu_table = build_small_gelu_table(10);
        let weights = make_test_weights(d_model, d_ff);

        let mut x = None;
        for offset in 0u32..200 {
            let candidate: Vec<F> = (0..d_model)
                .map(|i| F::from_canonical_u32(i as u32 + 1 + offset))
                .collect();

            let sum_x: F = candidate.iter().copied().sum();
            let d_inv = F::from_canonical_u32(d_model as u32).inverse();
            let mu = sum_x * d_inv;
            let xc: Vec<F> = candidate.iter().map(|&xi| xi - mu).collect();
            let sum_sq: F = xc.iter().map(|&v| v * v).sum();
            if sum_sq == F::zero() || !is_qr_m31(F::from_canonical_u32(d_model as u32) * sum_sq.inverse()) {
                continue;
            }

            let trace = forward_pass(&candidate, &weights, d_model, d_ff, &gelu_table);
            let sum_x2: F = trace.h.iter().copied().sum();
            let mu2 = sum_x2 * d_inv;
            let xc2: Vec<F> = trace.h.iter().map(|&xi| xi - mu2).collect();
            let sum_sq2: F = xc2.iter().map(|&v| v * v).sum();
            if sum_sq2 == F::zero() || !is_qr_m31(F::from_canonical_u32(d_model as u32) * sum_sq2.inverse()) {
                continue;
            }

            x = Some(candidate);
            break;
        }
        let x = x.expect("Could not find valid GPT-2 EF input");

        let mut pt = Transcript::new(b"gpt2-ef-tamper");
        let (proof, mut output) = prove_transformer_layer_ef(
            &x, &weights, d_model, d_ff, &gelu_table, &mut pt,
        );
        output[0] = output[0] + F::one();

        let mut vt = Transcript::new(b"gpt2-ef-tamper");
        assert!(
            !verify_transformer_layer_ef(
                &proof, &x, &output, &weights, d_model, d_ff, &gelu_table, &mut vt,
            ),
            "Should reject tampered GPT-2 EF output"
        );
    }
}
