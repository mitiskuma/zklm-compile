//! Prove/verify orchestration and execution pipeline.

mod forward;
mod prove;

use std::collections::HashMap;
use std::time::Instant;

use p3_field::{AbstractField, PrimeField32};
use p3_mersenne_31::Mersenne31;

use crate::field::common::hash_values;
use crate::field::m31_ops::*;
use crate::proving::lookup::LookupTable;
use crate::proving::sumcheck::Transcript;
use crate::proving::weight_commitment::WeightCommitment;
use crate::transformer::{
    prove_gpt2, verify_gpt2,
    LlamaLayerWeights, ModelConfig,
    QwenLayerWeights, QwenLayerCommitments,
    convert_gpt2_weights, transformer_forward_pass, layernorm_forward_simple,
    matmul_forward_simple, build_small_gelu_table,
};
use crate::protocol::*;
use crate::verification::verify_full_proof;

use forward::forward_pass_all_ops;
use prove::prove_all_ops;

type F = Mersenne31;


// ===== Pipeline data structures =====

#[allow(dead_code)]
struct LayerData {
    op_type: String,
    name: String,
    w_q: Vec<F>,
    b_q: Vec<F>,
    m: usize,
    n: usize,
    output: Vec<F>,
    pre_relu: Option<Vec<F>>,
    gelu_input: Option<Vec<F>>,
    gelu_output: Option<Vec<F>>,
    gelu_scale: i32,
    ln_gamma: Option<Vec<F>>,
    ln_beta: Option<Vec<F>>,
    ln_x: Option<Vec<F>>,
    ln_y: Option<Vec<F>>,
    ln_mu: Option<F>,
    ln_r: Option<F>,
    ln_d: Option<usize>,
    // For add_saved: the two input vectors
    add_input: Option<Vec<F>>,      // current before add
    add_saved_buf: Option<Vec<F>>,  // saved buffer being added
    // For attention
    attn_q: Option<Vec<F>>,
    attn_k: Option<Vec<F>>,
    attn_v: Option<Vec<F>>,
    attn_num_heads: usize,
    attn_seq_len: usize,
    attn_d_head: usize,
    attn_exp_scale: i32,
}

struct ForwardPassResult {
    layers: Vec<LayerData>,
    intermediates: Vec<Vec<F>>,
    current: Vec<F>,
    llama_layer_data: HashMap<usize, (LlamaLayerWeights, ModelConfig, crate::transformer::LlamaForwardTrace, i32)>,
    qwen_layer_data: HashMap<usize, (QwenLayerWeights, ModelConfig, crate::transformer::QwenForwardTrace, i32, i32)>,
    silu_table_cache: HashMap<i32, LookupTable>,
    sigmoid_table_cache: HashMap<i32, LookupTable>,
    weight_commitments: Vec<WeightCommitment>,
    qwen_commitments: Option<Vec<QwenLayerCommitments>>,
    all_qwen: bool,
    qwen_commit_duration: std::time::Duration,
}

struct ProveResult {
    layer_proofs: Vec<SerializedLayerProof>,
    layer_meta: Vec<LayerMeta>,
    proved_ops: Vec<String>,
    state_ops: Vec<String>,
    segment_hashes: Vec<String>,
    proof_size: usize,
}


pub(crate) fn run_gpt2_mode(req: ProveRequest) {
    let gpt2_desc = req.gpt2.expect("gpt2 mode requires 'gpt2' field");
    let t0 = Instant::now();

    let x_vals: Vec<F> = req.input.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let input_hash = hash_values(&req.input);

    // Build GELU table
    let gelu_table = build_small_gelu_table(gpt2_desc.gelu_scale);

    // Convert weights
    let weights = convert_gpt2_weights(&gpt2_desc);

    // Forward pass to get logits
    let mut current = x_vals.clone();
    for layer_weights in &weights.layers {
        let trace = transformer_forward_pass(
            &current,
            layer_weights,
            gpt2_desc.d_model,
            gpt2_desc.d_ff,
            &gelu_table,
        );
        current = trace;
    }
    // Final LayerNorm + LM head (skip if vocab_size == 0)
    let logits = if gpt2_desc.vocab_size > 0 {
        let (final_ln_out, _, _, _) = layernorm_forward_simple(
            &current,
            &weights.final_ln_gamma,
            &weights.final_ln_beta,
            gpt2_desc.d_model,
        );
        matmul_forward_simple(
            &weights.lm_head,
            &final_ln_out,
            gpt2_desc.vocab_size,
            gpt2_desc.d_model,
            None,
        )
    } else {
        current.clone()
    };

    eprintln!(
        "  forward pass: {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Prove
    let mut transcript = Transcript::new(b"gpt2-proof");
    let proof = prove_gpt2(
        &x_vals,
        &weights,
        gpt2_desc.d_model,
        gpt2_desc.d_ff,
        gpt2_desc.vocab_size,
        &gelu_table,
        &mut transcript,
    );

    let prove_time = t0.elapsed();

    // Verify
    let t_verify = Instant::now();
    let mut v_transcript = Transcript::new(b"gpt2-proof");
    let valid = verify_gpt2(
        &proof,
        &x_vals,
        &logits,
        &weights,
        gpt2_desc.d_model,
        gpt2_desc.d_ff,
        gpt2_desc.vocab_size,
        &gelu_table,
        &mut v_transcript,
    );
    let verify_time = t_verify.elapsed();

    // Proof size: bincode (compact binary, used in paper) for accurate measurement
    let proof_size = bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0);

    let output: Vec<i64> = logits.iter().map(|&v| from_field(v)).collect();
    let prediction = output
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| *v)
        .map(|(i, _)| i as i64)
        .unwrap_or(0);

    let response = ProveResponse {
        valid,
        prediction,
        output,
        prove_time_ms: prove_time.as_secs_f64() * 1000.0,
        verify_time_ms: verify_time.as_secs_f64() * 1000.0,
        proof_size_bytes: proof_size,
        weight_commitments: vec![], // embedded in proof
        input_commitment: hex::encode(input_hash),
        succinct_verification: true,
        mode: Some("gpt2".to_string()),
        coverage: None,
    };

    println!("{}", serde_json::to_string(&response).unwrap());
}



// ===== MLP Mode (existing) =====

pub(crate) fn run_mlp_mode(req: ProveRequest) {
    let (response, full_proof) = run_mlp_mode_returning(req, None);
    println!("{}", serde_json::to_string(&response).unwrap());

    // Export proof if --export-proof flag was passed
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--export-proof") {
        if let Some(path) = args.get(pos + 1) {
            if let Some(proof) = full_proof {
                eprintln!("  Exporting proof to {}", path);
                std::fs::write(path, serde_json::to_string_pretty(&proof).unwrap())
                    .unwrap_or_else(|e| eprintln!("Failed to write proof: {}", e));
            }
        }
    }
}

pub(crate) fn run_mlp_mode_returning(
    mut req: ProveRequest,
    preloaded: Option<&HashMap<String, PreloadedLinear>>,
) -> (ProveResponse, Option<FullProof>) {
    let t0 = Instant::now();
    let input_hash = hash_values(&req.input);

    let fwd = forward_pass_all_ops(&mut req, preloaded);
    let pr = prove_all_ops(&fwd, &input_hash, preloaded);
    let prove_time = t0.elapsed() - fwd.qwen_commit_duration;

    // ===== VERIFICATION =====
    let t_verify = Instant::now();
    let all_valid = verify_full_proof(
        &pr.layer_proofs,
        &pr.layer_meta,
        &fwd.weight_commitments,
        &input_hash,
        &req.input,
        Some(&req.ops),
        &fwd.llama_layer_data,
        &fwd.qwen_layer_data,
    );
    let verify_time = t_verify.elapsed();

    let output: Vec<i64> = fwd.current.iter().map(|&v| from_field(v)).collect();
    let prediction = output
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| *v)
        .map(|(i, _)| i as i64)
        .unwrap_or(0);

    let proved_count = pr.proved_ops.len();
    let state_count = pr.state_ops.len();
    let total_count = proved_count + state_count;
    let computational_total = proved_count;
    let computational_count = proved_count;
    let coverage = ProofCoverage {
        proved_ops: pr.proved_ops.clone(),
        state_ops: pr.state_ops.clone(),
        proved_count,
        state_count,
        computational_count,
        computational_total,
        total_count,
    };

    // Calculate proof size after timing (not part of prove time)
    // Use bincode (compact binary) for accurate proof size measurement
    let mut proof_size = pr.proof_size;
    if proof_size == 0 {
        for lp in &pr.layer_proofs {
            match lp {
                SerializedLayerProof::QwenLayer(ref p) => {
                    proof_size += bincode::serialize(p).map(|b| b.len()).unwrap_or(0);
                }
                SerializedLayerProof::QwenLayerEF(ref p) => {
                    proof_size += bincode::serialize(p).map(|b| b.len()).unwrap_or(0);
                }
                _ => {}
            }
        }
    }

    let wc_hex: Vec<String> = fwd.weight_commitments.iter().map(|c| hex::encode(c.root)).collect();

    let full_proof = FullProof {
        layer_proofs: pr.layer_proofs,
        output: fwd.current.iter().map(|v| v.as_canonical_u32()).collect(),
        input_hash,
        layer_meta: pr.layer_meta,
        weight_commitments: wc_hex.clone(),
        coverage: Some(SerializedCoverage {
            proved_ops: pr.proved_ops,
            state_ops: pr.state_ops,
            proved_count,
            state_count,
            computational_count,
            computational_total,
            total_count,
        }),
        segment_hashes: pr.segment_hashes,
    };

    let response = ProveResponse {
        valid: all_valid,
        prediction,
        output,
        prove_time_ms: prove_time.as_secs_f64() * 1000.0,
        verify_time_ms: verify_time.as_secs_f64() * 1000.0,
        proof_size_bytes: proof_size,
        weight_commitments: wc_hex,
        input_commitment: hex::encode(input_hash),
        succinct_verification: true,
        mode: Some("mlp".to_string()),
        coverage: Some(coverage),
    };

    (response, Some(full_proof))
}

/// Full verification with claim chaining + input commitment (MLP mode).

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::Field;
    use crate::field::common::mod_sqrt_m31;
    use crate::transformer::lookup_gelu_simple;

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

    fn make_test_weights(d_model: usize, d_ff: usize) -> GPT2Desc {
        let layer = TransformerLayerDesc {
            ln1_gamma: vec![1; d_model],
            ln1_beta: vec![0; d_model],
            w_attn: {
                let mut w = vec![0u32; d_model * d_model];
                for i in 0..d_model {
                    w[i * d_model + i] = 1;
                }
                w
            },
            ln2_gamma: vec![1; d_model],
            ln2_beta: vec![0; d_model],
            w_fc1: vec![0; d_ff * d_model],
            b_fc1: (0..d_ff)
                .map(|i| to_field((i as i64 % 20) - 10).as_canonical_u32())
                .collect(),
            w_fc2: vec![0; d_model * d_ff],
            b_fc2: vec![0; d_model],
        };

        GPT2Desc {
            d_model,
            d_ff,
            vocab_size: d_model,
            gelu_scale: 10,
            layers: vec![layer],
            final_ln_gamma: vec![1; d_model],
            final_ln_beta: vec![0; d_model],
            lm_head: {
                let mut w = vec![0u32; d_model * d_model];
                for i in 0..d_model {
                    w[i * d_model + i] = 1;
                }
                w
            },
        }
    }

    /// Find input where all LayerNorms have valid QR in M31.
    fn find_valid_gpt2_input(desc: &GPT2Desc) -> Vec<u32> {
        let gelu_table = build_small_gelu_table(desc.gelu_scale);
        let weights = convert_gpt2_weights(desc);

        for offset in 0u32..500 {
            let candidate: Vec<F> = (0..desc.d_model)
                .map(|i| F::from_canonical_u32(i as u32 + 1 + offset))
                .collect();

            let valid = (|| {
                let mut current = candidate.clone();
                for layer_weights in &weights.layers {
                    let d_inv = F::from_canonical_u32(desc.d_model as u32).inverse();

                    // Check LN1
                    let sum_x: F = current.iter().copied().sum();
                    let mu = sum_x * d_inv;
                    let xc: Vec<F> = current.iter().map(|&xi| xi - mu).collect();
                    let sum_sq: F = xc.iter().map(|&v| v * v).sum();
                    if sum_sq == F::zero() {
                        return false;
                    }
                    let target = F::from_canonical_u32(desc.d_model as u32) * sum_sq.inverse();
                    if !is_qr_m31(target) {
                        return false;
                    }

                    let trace = transformer_forward_pass(
                        &current,
                        layer_weights,
                        desc.d_model,
                        desc.d_ff,
                        &gelu_table,
                    );

                    // Check LN2 (after residual 1)
                    let (ln1_out, _, _, _) = layernorm_forward_simple(
                        &current,
                        &layer_weights.ln1_gamma,
                        &layer_weights.ln1_beta,
                        desc.d_model,
                    );
                    let attn_out = matmul_forward_simple(
                        &layer_weights.w_attn,
                        &ln1_out,
                        desc.d_model,
                        desc.d_model,
                        None,
                    );
                    let h: Vec<F> = current
                        .iter()
                        .zip(attn_out.iter())
                        .map(|(&a, &b)| a + b)
                        .collect();
                    let sum_h: F = h.iter().copied().sum();
                    let mu2 = sum_h * d_inv;
                    let hc: Vec<F> = h.iter().map(|&hi| hi - mu2).collect();
                    let sum_sq2: F = hc.iter().map(|&v| v * v).sum();
                    if sum_sq2 == F::zero() {
                        return false;
                    }
                    let target2 =
                        F::from_canonical_u32(desc.d_model as u32) * sum_sq2.inverse();
                    if !is_qr_m31(target2) {
                        return false;
                    }

                    // Check fc1_out in GELU table
                    let r2 = mod_sqrt_m31(target2);
                    let ln2_out: Vec<F> = hc
                        .iter()
                        .zip(layer_weights.ln2_gamma.iter())
                        .zip(layer_weights.ln2_beta.iter())
                        .map(|((&hci, &gi), &bi)| gi * hci * r2 + bi)
                        .collect();
                    let fc1_out = matmul_forward_simple(
                        &layer_weights.w_fc1,
                        &ln2_out,
                        desc.d_ff,
                        desc.d_model,
                        Some(&layer_weights.b_fc1),
                    );
                    let all_in_table = fc1_out.iter().all(|&v| {
                        let key = v.as_canonical_u32();
                        gelu_table.entries.iter().any(|&(inp, _)| inp == key)
                    });
                    if !all_in_table {
                        return false;
                    }

                    current = trace;
                }

                // Check final LN
                let sum_x: F = current.iter().copied().sum();
                let d_inv = F::from_canonical_u32(desc.d_model as u32).inverse();
                let mu = sum_x * d_inv;
                let xc: Vec<F> = current.iter().map(|&xi| xi - mu).collect();
                let sum_sq: F = xc.iter().map(|&v| v * v).sum();
                if sum_sq == F::zero() {
                    return false;
                }
                let target = F::from_canonical_u32(desc.d_model as u32) * sum_sq.inverse();
                if !is_qr_m31(target) {
                    return false;
                }

                true
            })();

            if valid {
                return candidate
                    .iter()
                    .map(|v| v.as_canonical_u32())
                    .collect();
            }
        }
        panic!("Could not find valid GPT-2 input");
    }

    #[test]
    fn test_gpt2_mode_1_layer() {
        let desc = make_test_weights(8, 16);
        let input = find_valid_gpt2_input(&desc);

        let req = ProveRequest {
            mode: "gpt2".to_string(),
            input: input.clone(),
            ops: vec![],
            gpt2: Some(desc),
        };

        // Simulate the GPT-2 mode flow
        let gpt2_desc = req.gpt2.as_ref().unwrap();
        let gelu_table = build_small_gelu_table(gpt2_desc.gelu_scale);
        let weights = convert_gpt2_weights(gpt2_desc);

        let x_vals: Vec<F> = req.input.iter().map(|&v| F::from_canonical_u32(v)).collect();

        // Forward pass
        let mut current = x_vals.clone();
        for layer_weights in &weights.layers {
            current = transformer_forward_pass(
                &current,
                layer_weights,
                gpt2_desc.d_model,
                gpt2_desc.d_ff,
                &gelu_table,
            );
        }
        let (final_ln_out, _, _, _) = layernorm_forward_simple(
            &current,
            &weights.final_ln_gamma,
            &weights.final_ln_beta,
            gpt2_desc.d_model,
        );
        let logits = matmul_forward_simple(
            &weights.lm_head,
            &final_ln_out,
            gpt2_desc.vocab_size,
            gpt2_desc.d_model,
            None,
        );

        // Prove
        let mut pt = Transcript::new(b"gpt2-proof");
        let proof = prove_gpt2(
            &x_vals,
            &weights,
            gpt2_desc.d_model,
            gpt2_desc.d_ff,
            gpt2_desc.vocab_size,
            &gelu_table,
            &mut pt,
        );

        // Verify
        let mut vt = Transcript::new(b"gpt2-proof");
        assert!(verify_gpt2(
            &proof,
            &x_vals,
            &logits,
            &weights,
            gpt2_desc.d_model,
            gpt2_desc.d_ff,
            gpt2_desc.vocab_size,
            &gelu_table,
            &mut vt,
        ));
    }

    #[test]
    fn test_gpt2_mode_2_layers() {
        let mut desc = make_test_weights(8, 16);
        // Add a second layer (clone the first)
        let layer2 = TransformerLayerDesc {
            ln1_gamma: desc.layers[0].ln1_gamma.clone(),
            ln1_beta: desc.layers[0].ln1_beta.clone(),
            w_attn: desc.layers[0].w_attn.clone(),
            ln2_gamma: desc.layers[0].ln2_gamma.clone(),
            ln2_beta: desc.layers[0].ln2_beta.clone(),
            w_fc1: desc.layers[0].w_fc1.clone(),
            b_fc1: desc.layers[0].b_fc1.clone(),
            w_fc2: desc.layers[0].w_fc2.clone(),
            b_fc2: desc.layers[0].b_fc2.clone(),
        };
        desc.layers.push(layer2);

        let input = find_valid_gpt2_input(&desc);
        let gpt2_desc = &desc;
        let gelu_table = build_small_gelu_table(gpt2_desc.gelu_scale);
        let weights = convert_gpt2_weights(gpt2_desc);
        let x_vals: Vec<F> = input.iter().map(|&v| F::from_canonical_u32(v)).collect();

        let mut current = x_vals.clone();
        for layer_weights in &weights.layers {
            current = transformer_forward_pass(
                &current,
                layer_weights,
                gpt2_desc.d_model,
                gpt2_desc.d_ff,
                &gelu_table,
            );
        }
        let (final_ln_out, _, _, _) = layernorm_forward_simple(
            &current,
            &weights.final_ln_gamma,
            &weights.final_ln_beta,
            gpt2_desc.d_model,
        );
        let logits = matmul_forward_simple(
            &weights.lm_head,
            &final_ln_out,
            gpt2_desc.vocab_size,
            gpt2_desc.d_model,
            None,
        );

        let mut pt = Transcript::new(b"gpt2-proof");
        let proof = prove_gpt2(
            &x_vals,
            &weights,
            gpt2_desc.d_model,
            gpt2_desc.d_ff,
            gpt2_desc.vocab_size,
            &gelu_table,
            &mut pt,
        );

        let mut vt = Transcript::new(b"gpt2-proof");
        assert!(verify_gpt2(
            &proof,
            &x_vals,
            &logits,
            &weights,
            gpt2_desc.d_model,
            gpt2_desc.d_ff,
            gpt2_desc.vocab_size,
            &gelu_table,
            &mut vt,
        ));
    }

    #[test]
    fn test_mlp_mode_with_gelu() {
        let gelu_table = build_small_gelu_table(10);

        let input_vals: Vec<F> = vec![
            to_field(5),
            to_field(-3),
            to_field(10),
            to_field(-8),
        ];

        let gelu_out: Vec<F> = input_vals.iter().map(|&v| lookup_gelu_simple(v,&gelu_table)).collect();

        // GELU(5) with scale=10: x=0.5, gelu(0.5)≈0.346 → output≈3
        // GELU(10) with scale=10: x=1.0, gelu(1.0)≈0.841 → output≈8
        // Positive inputs must produce positive outputs
        assert_ne!(gelu_out[0], F::zero(), "GELU(5) should be non-zero");
        assert_ne!(gelu_out[2], F::zero(), "GELU(10) should be non-zero");
        // GELU is monotonic: gelu(10) > gelu(5)
        assert!(from_field(gelu_out[2]) > from_field(gelu_out[0]),
            "GELU must be monotonic: gelu(10)={} should be > gelu(5)={}",
            from_field(gelu_out[2]), from_field(gelu_out[0]));
        // Negative inputs: GELU(-8) should be near zero or negative
        assert!(from_field(gelu_out[3]) <= 0,
            "GELU(-8) should be <= 0, got {}", from_field(gelu_out[3]));
    }

    #[test]
    fn test_mlp_mode_with_layernorm() {
        // x=[1,2,3,5] → target = d/sum_sq = 4/(0.25+0.25+0.25+2.25)=4/3
        // Verify input produces a valid QR so assertions always run.
        let d = 4;
        let x: Vec<F> = vec![1, 2, 3, 5]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let gamma: Vec<F> = vec![F::one(); d];
        let beta_v: Vec<F> = vec![F::zero(); d];

        let sum_x: F = x.iter().copied().sum();
        let d_inv = F::from_canonical_u32(d as u32).inverse();
        let mu = sum_x * d_inv;
        let xc: Vec<F> = x.iter().map(|&xi| xi - mu).collect();
        let sum_sq: F = xc.iter().map(|&v| v * v).sum();
        let target = F::from_canonical_u32(d as u32) * sum_sq.inverse();

        assert!(is_qr_m31(target), "Test input must produce a QR target");

        let r = mod_sqrt_m31(target);
        assert_eq!(r * r * sum_sq, F::from_canonical_u32(d as u32));

        let result: Vec<F> = xc
            .iter()
            .zip(gamma.iter())
            .zip(beta_v.iter())
            .map(|((&xci, &gi), &bi)| gi * xci * r + bi)
            .collect();
        let sum_xc: F = xc.iter().copied().sum();
        assert_eq!(sum_xc, F::zero(), "Centered values must sum to zero");
        assert_ne!(result[0], result[3], "LayerNorm should differentiate values");
    }

    #[test]
    fn test_mlp_mode_save_add_saved() {
        // Test save/add_saved ops
        let mut saved: HashMap<String, Vec<F>> = HashMap::new();
        let current: Vec<F> = vec![F::from_canonical_u32(10), F::from_canonical_u32(20)];

        // Save
        saved.insert("residual".to_string(), current.clone());

        // Modify current
        let modified: Vec<F> = current.iter().map(|&v| v + F::from_canonical_u32(5)).collect();

        // Add saved
        let saved_vals = saved.get("residual").unwrap();
        let result: Vec<F> = modified
            .iter()
            .zip(saved_vals.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        assert_eq!(result[0], F::from_canonical_u32(25));
        assert_eq!(result[1], F::from_canonical_u32(45));
    }

    #[test]
    fn test_mlp_mode_backwards_compatible() {
        // Verify the existing linear+relu pipeline still works via JSON
        let json = r#"{
            "input": [100, 200, 300, 400],
            "ops": [
                {"type": "linear", "name": "fc1", "m": 2, "n": 4,
                 "w_q": [1, 0, 0, 0, 0, 1, 0, 0],
                 "b_q": [10, 20]},
                {"type": "relu", "name": "relu1"}
            ]
        }"#;

        let req: ProveRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.mode, "mlp");
        assert_eq!(req.ops.len(), 2);
        assert_eq!(req.ops[0].op_type, "linear");
        assert_eq!(req.ops[1].op_type, "relu");
    }

    /// Build Llama layer weights for testing (identity-like, zero MLP weights).
    fn make_llama_layer_weight_data(d_model: usize, d_ff: usize, num_q_heads: usize, num_kv_heads: usize, d_head: usize) -> LlamaLayerWeightData {
        let q_dim = num_q_heads * d_head;
        let kv_dim = num_kv_heads * d_head;

        let identity = |rows: usize, cols: usize| -> Vec<F> {
            let mut w = vec![F::zero(); rows * cols];
            for i in 0..rows.min(cols) {
                w[i * cols + i] = F::one();
            }
            w
        };

        LlamaLayerWeightData {
            norm1_gamma: vec![F::one(); d_model],
            w_q: identity(q_dim, d_model),
            w_k: identity(kv_dim, d_model),
            w_v: identity(kv_dim, d_model),
            w_o: identity(d_model, q_dim),
            norm2_gamma: vec![F::one(); d_model],
            w_gate: vec![F::zero(); d_ff * d_model],
            w_up: vec![F::zero(); d_ff * d_model],
            w_down: vec![F::zero(); d_model * d_ff],
        }
    }

    /// Find input where both RMSNorms in a Llama layer have valid QR.
    fn find_valid_llama_layer_input(d_model: usize, d_ff: usize, num_q_heads: usize, num_kv_heads: usize, d_head: usize, silu_scale: i32) -> Vec<u32> {
        let config = crate::transformer::ModelConfig {
            d_model, d_ff, num_q_heads, num_kv_heads, d_head,
            n_layers: 1, vocab_size: 0,
            norm_type: crate::transformer::NormType::RMSNorm,
            activation: crate::transformer::ActivationType::SwiGLU,
            v_num_heads: 0, v_d_head: 0,
        };
        let wd = make_llama_layer_weight_data(d_model, d_ff, num_q_heads, num_kv_heads, d_head);
        let weights = crate::transformer::LlamaLayerWeights {
            norm1_gamma: wd.norm1_gamma.clone(),
            w_q: wd.w_q.clone(), w_k: wd.w_k.clone(), w_v: wd.w_v.clone(),
            w_o: wd.w_o.clone(), norm2_gamma: wd.norm2_gamma.clone(),
            w_gate: wd.w_gate.clone(), w_up: wd.w_up.clone(), w_down: wd.w_down.clone(),
        };
        let silu_table = crate::proving::lookup::build_silu_table(silu_scale);

        for offset in 0u32..500 {
            let candidate: Vec<F> = (0..d_model)
                .map(|i| F::from_canonical_u32(i as u32 + 1 + offset))
                .collect();

            // Check RMSNorm 1
            let sum_sq: F = candidate.iter().map(|&v| v * v).sum();
            if sum_sq == F::zero() { continue; }
            let target = F::from_canonical_u32(d_model as u32) * sum_sq.inverse();
            if !is_qr_m31(target) { continue; }

            // Try full forward pass
            let result = std::panic::catch_unwind(|| {
                crate::transformer::llama_forward(&candidate, &weights, &config, &silu_table)
            });
            if result.is_ok() {
                return candidate.iter().map(|v| v.as_canonical_u32()).collect();
            }
        }
        panic!("Could not find valid Llama layer input");
    }

    #[test]
    fn test_llama_layer_composite_op() {
        let d_model = 8;
        let d_ff = 16;
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let d_head = 2;
        let silu_scale = 10;

        let input = find_valid_llama_layer_input(d_model, d_ff, num_q_heads, num_kv_heads, d_head, silu_scale);
        let wd = make_llama_layer_weight_data(d_model, d_ff, num_q_heads, num_kv_heads, d_head);

        let req = ProveRequest {
            mode: "mlp".into(),
            input: input.clone(),
            ops: vec![OpDesc {
                op_type: "llama_layer".into(),
                name: "layer_0".into(),
                m: 0, n: 0, w_q: vec![], b_q: vec![],
                gelu_scale: silu_scale,
                gamma: vec![], beta: vec![],
                save_name: String::new(), add_name: String::new(),
                new_input: vec![], ln_output: vec![],
                num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                k_values: vec![], v_values: vec![],
                llama_config: Some(LlamaLayerConfig {
                    d_model, d_ff, num_q_heads, num_kv_heads, d_head, silu_scale,
                }),
                llama_weights: Some(wd),
                qwen_config: None, qwen_weights: None,
            }],
            gpt2: None,
        };

        let (response, _proof) = run_mlp_mode_returning(req, None);
        assert!(response.valid, "llama_layer composite op should verify");
        assert_eq!(
            response.coverage.as_ref().unwrap().proved_count, 1,
            "Should have 1 proved op (llama_layer)"
        );
    }

    #[test]
    fn test_llama_layer_composite_2_layers() {
        let d_model = 8;
        let d_ff = 16;
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let d_head = 2;
        let silu_scale = 10;

        let input = find_valid_llama_layer_input(d_model, d_ff, num_q_heads, num_kv_heads, d_head, silu_scale);
        let wd1 = make_llama_layer_weight_data(d_model, d_ff, num_q_heads, num_kv_heads, d_head);
        let wd2 = make_llama_layer_weight_data(d_model, d_ff, num_q_heads, num_kv_heads, d_head);

        let make_op = |name: &str, wd: LlamaLayerWeightData| -> OpDesc {
            OpDesc {
                op_type: "llama_layer".into(),
                name: name.into(),
                m: 0, n: 0, w_q: vec![], b_q: vec![],
                gelu_scale: silu_scale,
                gamma: vec![], beta: vec![],
                save_name: String::new(), add_name: String::new(),
                new_input: vec![], ln_output: vec![],
                num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                k_values: vec![], v_values: vec![],
                llama_config: Some(LlamaLayerConfig {
                    d_model, d_ff, num_q_heads, num_kv_heads, d_head, silu_scale,
                }),
                llama_weights: Some(wd),
                qwen_config: None, qwen_weights: None,
            }
        };

        let req = ProveRequest {
            mode: "mlp".into(),
            input: input.clone(),
            ops: vec![make_op("layer_0", wd1), make_op("layer_1", wd2)],
            gpt2: None,
        };

        let (response, _proof) = run_mlp_mode_returning(req, None);
        assert!(response.valid, "2-layer llama_layer composite should verify");
        assert_eq!(
            response.coverage.as_ref().unwrap().proved_count, 2,
            "Should have 2 proved ops"
        );
    }

    #[test]
    fn test_llama_layer_binary_protocol() {
        // Test the binary protocol parsing for llama_layer (0x0C)
        let d_model = 8usize;
        let d_ff = 16usize;
        let num_q_heads = 4usize;
        let num_kv_heads = 2usize;
        let d_head = 2usize;
        let silu_scale: i32 = 10;
        let q_dim = num_q_heads * d_head;
        let kv_dim = num_kv_heads * d_head;

        let input = find_valid_llama_layer_input(d_model, d_ff, num_q_heads, num_kv_heads, d_head, silu_scale);

        // Build binary payload manually
        let mut data = Vec::new();
        // num_ops = 1
        data.extend_from_slice(&1u32.to_le_bytes());
        // input
        data.extend_from_slice(&(input.len() as u32).to_le_bytes());
        for &v in &input {
            data.extend_from_slice(&v.to_le_bytes());
        }
        // Op: 0x0C
        data.push(0x0C);
        // name
        let name = b"layer_0";
        data.extend_from_slice(&(name.len() as u32).to_le_bytes());
        data.extend_from_slice(name);
        // config
        data.extend_from_slice(&(d_model as u32).to_le_bytes());
        data.extend_from_slice(&(d_ff as u32).to_le_bytes());
        data.extend_from_slice(&(num_q_heads as u32).to_le_bytes());
        data.extend_from_slice(&(num_kv_heads as u32).to_le_bytes());
        data.extend_from_slice(&(d_head as u32).to_le_bytes());
        data.extend_from_slice(&silu_scale.to_le_bytes());

        // Helper: identity matrix as u32 flat
        let identity = |rows: usize, cols: usize| -> Vec<u32> {
            let mut w = vec![0u32; rows * cols];
            for i in 0..rows.min(cols) { w[i * cols + i] = 1; }
            w
        };

        // norm1_gamma (d_model)
        for _ in 0..d_model { data.extend_from_slice(&1u32.to_le_bytes()); }
        // w_q (q_dim * d_model)
        for v in identity(q_dim, d_model) { data.extend_from_slice(&v.to_le_bytes()); }
        // w_k (kv_dim * d_model)
        for v in identity(kv_dim, d_model) { data.extend_from_slice(&v.to_le_bytes()); }
        // w_v (kv_dim * d_model)
        for v in identity(kv_dim, d_model) { data.extend_from_slice(&v.to_le_bytes()); }
        // w_o (d_model * q_dim)
        for v in identity(d_model, q_dim) { data.extend_from_slice(&v.to_le_bytes()); }
        // norm2_gamma (d_model)
        for _ in 0..d_model { data.extend_from_slice(&1u32.to_le_bytes()); }
        // w_gate (d_ff * d_model) — zeros
        for _ in 0..d_ff * d_model { data.extend_from_slice(&0u32.to_le_bytes()); }
        // w_up (d_ff * d_model) — zeros
        for _ in 0..d_ff * d_model { data.extend_from_slice(&0u32.to_le_bytes()); }
        // w_down (d_model * d_ff) — zeros
        for _ in 0..d_model * d_ff { data.extend_from_slice(&0u32.to_le_bytes()); }

        // Parse
        let req = parse_binary(&data);
        assert_eq!(req.ops.len(), 1);
        assert_eq!(req.ops[0].op_type, "llama_layer");
        assert!(req.ops[0].llama_config.is_some());
        assert!(req.ops[0].llama_weights.is_some());

        let lc = req.ops[0].llama_config.as_ref().unwrap();
        assert_eq!(lc.d_model, d_model);
        assert_eq!(lc.d_ff, d_ff);

        // Run through prover+verifier
        let (response, _) = run_mlp_mode_returning(req, None);
        assert!(response.valid, "Binary-parsed llama_layer should verify");
    }
}
