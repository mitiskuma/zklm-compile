//! Extracted prove_all_ops: proves all forward-pass operations.

use std::collections::HashMap;
use std::time::Instant;

use p3_field::{AbstractField, PrimeField32};

use crate::field::common::{log2_ceil, hash_values};
use crate::field::m31_ops::{eq_evals, from_field, mle_evaluate};
use crate::proving::sumcheck::{self, Transcript};
use crate::proving::matmul::prove_matmul_succinct;
use crate::proving::gelu::prove_gelu;
use crate::proving::layernorm::prove_layernorm_sqr;
use crate::proving::weight_commitment::{commit_weights_fast, prove_mle_eval_no_merkle};
use crate::proving::attention::prove_row_attention;
use crate::protocol::*;
use crate::transformer::{
    QwenLayerProofEF, prove_qwen_layer_precommitted_ef,
    prove_qwen_layer_with_trace, prove_llama_layer_with_trace,
};

use super::{F, ForwardPassResult, ProveResult};

pub(super) fn prove_all_ops(
    fwd: &ForwardPassResult,
    input_hash: &[u8; 32],
    preloaded: Option<&HashMap<String, PreloadedLinear>>,
) -> ProveResult {
    // Local aliases — lets the extracted code reference these without fwd. prefix
    let layers = &fwd.layers;
    let intermediates = &fwd.intermediates;
    let llama_layer_data = &fwd.llama_layer_data;
    let qwen_layer_data = &fwd.qwen_layer_data;
    let all_qwen = fwd.all_qwen;

    let mut transcript = Transcript::new(b"structured-ml-proof");
    let mut proof_size = 0usize;
    let num_layers = layers.len();
    let mut layer_proofs: Vec<SerializedLayerProof> = Vec::new();
    let mut layer_meta: Vec<LayerMeta> = Vec::new();
    let mut proved_ops: Vec<String> = Vec::new();
    let mut state_ops: Vec<String> = Vec::new();
    let mut segment_hashes: Vec<String> = Vec::new();

    // Build layer_meta in forward order
    for layer in layers.iter() {
        layer_meta.push(LayerMeta {
            op_type: layer.op_type.clone(),
            name: layer.name.clone(),
            m: layer.m,
            n: layer.n,
            b_q: if layer.op_type == "layernorm" {
                // Store beta in b_q for verification
                layer.ln_beta.as_ref().map_or(vec![], |b: &Vec<F>| b.iter().map(|v| v.as_canonical_u32()).collect())
            } else {
                layer.b_q.iter().map(|v: &F| v.as_canonical_u32()).collect()
            },
            output: if layer.op_type == "relu" {
                vec![]
            } else {
                layer.output.iter().map(|v: &F| v.as_canonical_u32()).collect()
            },
            attn_q: layer.attn_q.as_ref().map_or(vec![], |v: &Vec<F>| v.iter().map(|f: &F| f.as_canonical_u32()).collect()),
            attn_k: layer.attn_k.as_ref().map_or(vec![], |v: &Vec<F>| v.iter().map(|f: &F| f.as_canonical_u32()).collect()),
            attn_v: layer.attn_v.as_ref().map_or(vec![], |v: &Vec<F>| v.iter().map(|f: &F| f.as_canonical_u32()).collect()),
            attn_exp_scale: layer.attn_exp_scale,
        });
    }

    if all_qwen && num_layers > 1 {
        // ===== PARALLEL PROVE: independent per-layer transcripts =====
        use rayon::prelude::*;
        // Build lookup tables once (all layers share the same scales)
        let first_data = match qwen_layer_data.values().next() {
            Some(d) => d,
            None => {
                eprintln!("  prove: all_qwen=true but qwen_layer_data is empty, falling back to sequential");
                // Fall through to sequential path — this shouldn't happen but don't panic
                return prove_all_ops(fwd, input_hash, preloaded);
            }
        };
        let silu_table = crate::proving::lookup::build_silu_table(first_data.3);
        let sigmoid_table = crate::proving::lookup::build_sigmoid_table(first_data.4);

        let layer_commitments = fwd.qwen_commitments.as_ref()
            .expect("all_qwen=true but qwen_commitments is None — forward pass bug");

        // Prove all layers in parallel with extension-field challenges (124-bit soundness).
        // In pcs-full mode, limit parallelism to avoid memory pressure from batch
        // Basefold encoding (8 polynomials × ~64MB per layer = ~500MB per concurrent layer).
        // pcs-full: limit parallelism (8 large encodings per layer = ~500MB each)
        // pcs/default: full parallelism (w_partial encodings are tiny)
        #[cfg(feature = "pcs-full")]
        let par_chunk_size = 8.max(1);
        #[cfg(not(feature = "pcs-full"))]
        let par_chunk_size = num_layers;

        let parallel_results: Vec<(usize, QwenLayerProofEF, String)> = {
            let indices: Vec<usize> = (0..num_layers).collect();
            let mut results = Vec::with_capacity(num_layers);
            for chunk in indices.chunks(par_chunk_size) {
                let chunk_results: Vec<_> = chunk.par_iter()
                    .map(|&idx| {
                        let (ref weights, ref config, ref trace, _, _) =
                            qwen_layer_data.get(&idx).expect("qwen_layer data missing");
                        let seed = format!("qwen-layer-{}", idx);
                        let mut layer_transcript = Transcript::new(seed.as_bytes());
                        layer_transcript.absorb_bytes(input_hash);
                        let proof = prove_qwen_layer_precommitted_ef(
                            trace, weights, config, layer_commitments[idx].clone(),
                            &silu_table, &sigmoid_table,
                            &mut layer_transcript,
                        );
                        (idx, proof, layers[idx].name.clone())
                    })
                    .collect();
                results.extend(chunk_results);
            }
            results
        };

        // Collect results in reverse order (matching sequential prove order)
        let mut results_by_idx: Vec<(usize, QwenLayerProofEF, String)> = parallel_results;
        results_by_idx.sort_by(|a, b| b.0.cmp(&a.0)); // reverse order

        for (_, proof, name) in results_by_idx {
            layer_proofs.push(SerializedLayerProof::QwenLayerEF(proof));
            proved_ops.push(name);
        }
    } else {
    // ===== SEQUENTIAL PROVE (original path for mixed ops) =====
    // Prove in reverse order
    for idx in (0..num_layers).rev() {
        let layer = &layers[idx];
        let layer_t0 = Instant::now();
        match layer.op_type.as_str() {
            "linear" => {
                let prev_output = &intermediates[idx];
                // Use preloaded weights if available (layer.w_q is empty for preloaded)
                let w_for_proof: &[F] = if layer.w_q.is_empty() {
                    if let Some(ref pl) = preloaded {
                        &pl.get(&layer.name)
                            .unwrap_or_else(|| panic!("Missing preloaded weight: {}", layer.name))
                            .w_f
                    } else {
                        panic!("Empty w_q without preloaded weights for layer {}", layer.name);
                    }
                } else {
                    &layer.w_q
                };
                // Fiat-Shamir binding: absorb weight commitment before matmul
                let w_commitment = commit_weights_fast(w_for_proof);
                transcript.absorb_bytes(&w_commitment.root);
                let proof = prove_matmul_succinct(
                    w_for_proof,
                    prev_output,
                    &layer.output,
                    layer.m,
                    layer.n,
                    Some(&layer.b_q),
                    &mut transcript,
                );
                let matmul_rounds = proof.matmul_proof.sumcheck_proof.round_polys.len();
                let eval_rounds = proof.w_eval_proof.eval_sumcheck.round_polys.len();
                proof_size += matmul_rounds * 3 * 4 + 8;
                proof_size += eval_rounds * 3 * 4 + 8;
                layer_proofs.push(SerializedLayerProof::Matmul(proof));
                proved_ops.push(layer.name.clone());
                eprintln!(
                    "  linear({}) {}x{}: {:.1}ms",
                    layer.name,
                    layer.m,
                    layer.n,
                    layer_t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            "relu" => {
                let z_vals = layer.pre_relu.as_ref()
                    .expect("relu layer missing pre_relu data — forward pass did not populate it");
                let log_n = log2_ceil(z_vals.len());
                let n_pad = 1 << log_n;
                let mut z_pad = z_vals.to_vec();
                z_pad.resize(n_pad, F::zero());

                let a_pad: Vec<F> = z_pad
                    .iter()
                    .map(|&v| if from_field(v) >= 0 { v } else { F::zero() })
                    .collect();
                let b_vals: Vec<F> = z_pad
                    .iter()
                    .map(|&v| {
                        if from_field(v) >= 0 {
                            F::one()
                        } else {
                            F::zero()
                        }
                    })
                    .collect();

                let a_commitment = commit_weights_fast(&a_pad);

                let (chain_eval_proof, chain_value) = if !layer_proofs.is_empty() {
                    if let SerializedLayerProof::Matmul(ref prev_matmul) =
                        layer_proofs.last().unwrap()
                    {
                        let chain_point: Vec<F> = prev_matmul
                            .matmul_proof
                            .s_point
                            .iter()
                            .map(|&v| F::from_canonical_u32(v))
                            .collect();
                        let mut chain_transcript = Transcript::new(b"relu-chain");
                        chain_transcript.absorb_bytes(&a_commitment.root);
                        let (claimed, proof) =
                            prove_mle_eval_no_merkle(&a_pad, &chain_point, &mut chain_transcript);
                        (Some(proof), Some(claimed.as_canonical_u32()))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };

                transcript.absorb_bytes(&a_commitment.root);
                let r_point = transcript.squeeze_many(log_n);

                let a_at_r = mle_evaluate(&a_pad, &r_point);
                let mut eval_transcript = Transcript::new(b"relu-eval");
                eval_transcript.absorb_bytes(&a_commitment.root);
                let (_, a_eval_proof) = prove_mle_eval_no_merkle(&a_pad, &r_point, &mut eval_transcript);

                let eq_r = eq_evals(&r_point);

                let (product_proof, pf, pg, ph) =
                    sumcheck::prove_triple_best(&eq_r, &b_vals, &z_pad, log_n, &mut transcript);

                let r2_point = transcript.squeeze_many(log_n);
                let eq_r2 = eq_evals(&r2_point);
                let one_minus_b: Vec<F> = b_vals.iter().map(|&v| F::one() - v).collect();

                let (boolean_proof, bf, bg, bh) = sumcheck::prove_triple_best(
                    &eq_r2,
                    &b_vals,
                    &one_minus_b,
                    log_n,
                    &mut transcript,
                );

                let sc_size = (product_proof.round_polys.len()
                    + boolean_proof.round_polys.len())
                    * 4
                    * 4
                    + 24;
                let eval_size = a_eval_proof.eval_sumcheck.round_polys.len() * 3 * 4 + 8;
                let chain_size = chain_eval_proof
                    .as_ref()
                    .map(|p| p.eval_sumcheck.round_polys.len() * 3 * 4 + 8)
                    .unwrap_or(0);
                proof_size += sc_size + eval_size + chain_size + 32 + 4;

                layer_proofs.push(SerializedLayerProof::Relu {
                    a_commitment,
                    a_at_r: a_at_r.as_canonical_u32(),
                    a_eval_proof,
                    chain_eval_proof,
                    chain_value,
                    product_proof,
                    product_finals: (
                        pf.as_canonical_u32(),
                        pg.as_canonical_u32(),
                        ph.as_canonical_u32(),
                    ),
                    boolean_proof,
                    boolean_finals: (
                        bf.as_canonical_u32(),
                        bg.as_canonical_u32(),
                        bh.as_canonical_u32(),
                    ),
                });
                proved_ops.push(layer.name.clone());
                eprintln!(
                    "  relu({}): {:.1}ms",
                    layer.name,
                    layer_t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            "layernorm" => {
                let ln_x = layer.ln_x.as_ref()
                    .expect("layernorm layer missing ln_x — forward pass did not populate it");
                let ln_gamma = layer.ln_gamma.as_ref()
                    .expect("layernorm layer missing ln_gamma — forward pass did not populate it");
                let ln_beta = layer.ln_beta.as_ref()
                    .expect("layernorm layer missing ln_beta — forward pass did not populate it");
                let ln_y = layer.ln_y.as_ref()
                    .expect("layernorm layer missing ln_y — forward pass did not populate it");
                let ln_mu = layer.ln_mu
                    .expect("layernorm layer missing ln_mu — forward pass did not populate it");

                let proof = prove_layernorm_sqr(
                    ln_x, ln_gamma, ln_beta, ln_y, ln_mu, &mut transcript,
                );

                let proof_bytes = serde_json::to_vec(&proof).unwrap_or_default();
                proof_size += proof_bytes.len();

                layer_proofs.push(SerializedLayerProof::LayerNormSqr(proof));
                proved_ops.push(layer.name.clone());
                eprintln!(
                    "  layernorm({}): {:.1}ms",
                    layer.name,
                    layer_t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            "gelu" => {
                let gelu_input = layer.gelu_input.as_ref()
                    .expect("gelu layer missing gelu_input — forward pass did not populate it");
                let gelu_output = layer.gelu_output.as_ref()
                    .expect("gelu layer missing gelu_output — forward pass did not populate it");
                use crate::proving::gelu::build_gelu_table;
                let table = build_gelu_table(layer.gelu_scale);

                let proof = prove_gelu(gelu_input, gelu_output, &table, &mut transcript);

                let proof_bytes = serde_json::to_vec(&proof).unwrap_or_default();
                proof_size += proof_bytes.len();

                layer_proofs.push(SerializedLayerProof::Gelu(proof));
                proved_ops.push(layer.name.clone());
                eprintln!(
                    "  gelu({}): {:.1}ms",
                    layer.name,
                    layer_t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            "attention" => {
                let q = layer.attn_q.as_ref()
                    .expect("attention layer missing attn_q — forward pass did not populate it");
                let k = layer.attn_k.as_ref()
                    .expect("attention layer missing attn_k — forward pass did not populate it");
                let v = layer.attn_v.as_ref()
                    .expect("attention layer missing attn_v — forward pass did not populate it");

                if layer.attn_seq_len == 1 {
                    // seq_len=1: softmax of single element is always 1.0, output = V.
                    // Emit a PassThrough proof and absorb output hash for transcript binding.
                    let output_hash = hash_values(
                        &layer.output.iter().map(|v| v.as_canonical_u32()).collect::<Vec<_>>(),
                    );
                    transcript.absorb_bytes(&output_hash);
                    layer_proofs.push(SerializedLayerProof::PassThrough);
                    proved_ops.push(format!("{} (attn-trivial)", layer.name));
                    eprintln!(
                        "  attention({}): {:.1}ms (seq_len=1, trivial)",
                        layer.name,
                        layer_t0.elapsed().as_secs_f64() * 1000.0,
                    );
                } else {
                    let exp_table = crate::proving::lookup::build_exp_table(layer.attn_exp_scale);

                    let proof = prove_row_attention(
                        q, k, v,
                        layer.attn_num_heads, layer.attn_seq_len, layer.attn_d_head,
                        &exp_table, &mut transcript,
                    );

                    let proof_bytes = serde_json::to_vec(&proof).unwrap_or_default();
                    proof_size += proof_bytes.len();

                    layer_proofs.push(SerializedLayerProof::Attention(proof));
                    proved_ops.push(layer.name.clone());
                    eprintln!(
                        "  attention({}): {:.1}ms ({}h×{}s×{}d)",
                        layer.name,
                        layer_t0.elapsed().as_secs_f64() * 1000.0,
                        layer.attn_num_heads, layer.attn_seq_len, layer.attn_d_head,
                    );
                }
            }
            "add_saved" => {
                // Residual add: c = a + b. The verifier chains claims using the
                // saved buffer's output data (available in layer_meta).
                // Find the save_name from the original op (stored in layer.name convention).
                // The add_name was stored during forward pass as part of the op.
                // add_res1_0 → save_res1_0
                let save_layer_name = layer.name.replacen("add_", "save_", 1);
                layer_proofs.push(SerializedLayerProof::Add {
                    save_name: save_layer_name,
                });
                proved_ops.push(format!("{} (add)", layer.name));
                eprintln!(
                    "  add({}): {:.1}ms",
                    layer.name,
                    layer_t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            "save" => {
                // Save commitment: hash the saved buffer for later verification by add_saved.
                let output_vals: Vec<u32> = layer.output.iter().map(|v| v.as_canonical_u32()).collect();
                let content_hash = hash_values(&output_vals);
                transcript.absorb_bytes(&content_hash);
                let save_name = layer.name.clone();
                layer_proofs.push(SerializedLayerProof::SaveCommitment {
                    save_name: save_name.clone(),
                    content_hash,
                });
                state_ops.push(format!("{} (save-committed)", layer.name));
                eprintln!(
                    "  save({}): {:.1}ms (committed)",
                    layer.name,
                    layer_t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            "set_input" => {
                // Cross-segment commitment: link this input to previous segment's output.
                let new_input_vals: Vec<u32> = layer.output.iter().map(|v| v.as_canonical_u32()).collect();
                let new_input_hash = hash_values(&new_input_vals);
                // Get previous layer's output hash for cross-segment link.
                let prev_output_hash = if idx > 0 {
                    let prev = &layers[idx - 1];
                    hash_values(&prev.output.iter().map(|v: &F| v.as_canonical_u32()).collect::<Vec<_>>())
                } else {
                    [0u8; 32]
                };
                transcript.absorb_bytes(&prev_output_hash);
                transcript.absorb_bytes(&new_input_hash);
                segment_hashes.push(hex::encode(new_input_hash));
                layer_proofs.push(SerializedLayerProof::SegmentBoundary {
                    prev_output_hash,
                    new_input_hash,
                });
                state_ops.push(format!("{} (segment-boundary)", layer.name));
                eprintln!(
                    "  set_input({}): {:.1}ms (boundary-committed)",
                    layer.name,
                    layer_t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            "rmsnorm" => {
                let x = layer.ln_x.as_ref()
                    .expect("rmsnorm layer missing ln_x — forward pass did not populate it");
                let gamma = layer.ln_gamma.as_ref()
                    .expect("rmsnorm layer missing ln_gamma — forward pass did not populate it");
                let y = layer.ln_y.as_ref()
                    .expect("rmsnorm layer missing ln_y — forward pass did not populate it");

                let proof = crate::proving::rmsnorm::prove_rmsnorm(x, gamma, y, &mut transcript);
                let proof_bytes = serde_json::to_vec(&proof).unwrap_or_default();
                proof_size += proof_bytes.len();

                layer_proofs.push(SerializedLayerProof::RmsNorm(proof));
                proved_ops.push(layer.name.clone());
                eprintln!("  rmsnorm({}): {:.1}ms", layer.name, layer_t0.elapsed().as_secs_f64() * 1000.0);
            }
            "silu" => {
                let silu_input = layer.gelu_input.as_ref()
                    .expect("silu layer missing gelu_input — forward pass did not populate it");
                let silu_output = layer.gelu_output.as_ref()
                    .expect("silu layer missing gelu_output — forward pass did not populate it");
                let table = crate::proving::lookup::build_silu_table(layer.gelu_scale);

                let proof = crate::proving::silu::prove_silu(silu_input, silu_output, &table, &mut transcript);
                let proof_bytes = serde_json::to_vec(&proof).unwrap_or_default();
                proof_size += proof_bytes.len();

                layer_proofs.push(SerializedLayerProof::Silu(proof));
                proved_ops.push(layer.name.clone());
                eprintln!("  silu({}): {:.1}ms", layer.name, layer_t0.elapsed().as_secs_f64() * 1000.0);
            }
            "swiglu" => {
                let gate = layer.gelu_input.as_ref()
                    .expect("swiglu layer missing gelu_input (gate) — forward pass did not populate it");
                let gate_silu = layer.gelu_output.as_ref()
                    .expect("swiglu layer missing gelu_output (gate_silu) — forward pass did not populate it");
                let up = layer.ln_y.as_ref()
                    .expect("swiglu layer missing ln_y (up) — forward pass did not populate it"); // up stored in ln_y
                let output = &layer.output;
                let silu_table = crate::proving::lookup::build_silu_table(layer.gelu_scale);

                let proof = crate::proving::swiglu::prove_swiglu(gate, gate_silu, up, output, &silu_table, &mut transcript);
                let proof_bytes = serde_json::to_vec(&proof).unwrap_or_default();
                proof_size += proof_bytes.len();

                layer_proofs.push(SerializedLayerProof::SwiGlu(proof));
                proved_ops.push(layer.name.clone());
                eprintln!("  swiglu({}): {:.1}ms", layer.name, layer_t0.elapsed().as_secs_f64() * 1000.0);
            }
            "gqa_attention" => {
                let q = layer.attn_q.as_ref()
                    .expect("gqa_attention layer missing attn_q — forward pass did not populate it");
                let k = layer.attn_k.as_ref()
                    .expect("gqa_attention layer missing attn_k — forward pass did not populate it");
                let v = layer.attn_v.as_ref()
                    .expect("gqa_attention layer missing attn_v — forward pass did not populate it");
                let num_kv_heads = layer.n;

                if layer.attn_seq_len == 1 {
                    let output_hash = hash_values(
                        &layer.output.iter().map(|v| v.as_canonical_u32()).collect::<Vec<_>>(),
                    );
                    transcript.absorb_bytes(&output_hash);
                    layer_proofs.push(SerializedLayerProof::PassThrough);
                    proved_ops.push(format!("{} (gqa-trivial)", layer.name));
                    eprintln!("  gqa_attention({}): {:.1}ms (seq_len=1, trivial)", layer.name, layer_t0.elapsed().as_secs_f64() * 1000.0);
                } else {
                    let exp_table = crate::proving::lookup::build_exp_table(layer.attn_exp_scale);
                    let proof = crate::proving::attention::prove_row_attention_gqa(
                        q, k, v,
                        layer.attn_num_heads, num_kv_heads,
                        layer.attn_seq_len, layer.attn_d_head,
                        &exp_table, &mut transcript,
                    );
                    let proof_bytes = serde_json::to_vec(&proof).unwrap_or_default();
                    proof_size += proof_bytes.len();

                    layer_proofs.push(SerializedLayerProof::GqaAttention(proof));
                    proved_ops.push(layer.name.clone());
                    eprintln!("  gqa_attention({}): {:.1}ms ({}qh×{}kvh×{}s×{}d)",
                        layer.name, layer_t0.elapsed().as_secs_f64() * 1000.0,
                        layer.attn_num_heads, num_kv_heads, layer.attn_seq_len, layer.attn_d_head);
                }
            }
            "llama_layer" => {
                let (ref weights, ref config, ref trace, silu_scale) =
                    llama_layer_data.get(&idx).expect("llama_layer data missing");
                let silu_table = fwd.silu_table_cache.get(silu_scale)
                    .expect("llama_layer: silu_table_cache missing entry for silu_scale — forward pass caching bug");

                let proof = prove_llama_layer_with_trace(
                    trace, weights, config, silu_table, &mut transcript,
                );

                let proof_bytes = serde_json::to_vec(&proof).unwrap_or_default();
                proof_size += proof_bytes.len();

                layer_proofs.push(SerializedLayerProof::LlamaLayer(proof));
                proved_ops.push(layer.name.clone());
                eprintln!(
                    "  llama_layer({}): {:.1}ms",
                    layer.name,
                    layer_t0.elapsed().as_secs_f64() * 1000.0,
                );
            }
            "qwen_layer" => {
                let (ref weights, ref config, ref trace, silu_scale, sigmoid_scale) =
                    qwen_layer_data.get(&idx).expect("qwen_layer data missing");
                let silu_table = fwd.silu_table_cache.get(silu_scale)
                    .expect("qwen_layer: silu_table_cache missing entry for silu_scale — forward pass caching bug");
                let sigmoid_table = fwd.sigmoid_table_cache.get(sigmoid_scale)
                    .expect("qwen_layer: sigmoid_table_cache missing entry for sigmoid_scale — forward pass caching bug");

                let proof = prove_qwen_layer_with_trace(
                    trace, weights, config, silu_table, sigmoid_table, &mut transcript,
                );

                let proof_bytes = serde_json::to_vec(&proof).unwrap_or_default();
                proof_size += proof_bytes.len();

                layer_proofs.push(SerializedLayerProof::QwenLayer(proof));
                proved_ops.push(layer.name.clone());
                eprintln!(
                    "  qwen_layer({}): {:.1}ms",
                    layer.name,
                    layer_t0.elapsed().as_secs_f64() * 1000.0,
                );
            }
            "passthrough" => {
                let output_hash = hash_values(
                    &layer.output.iter().map(|v| v.as_canonical_u32()).collect::<Vec<_>>(),
                );
                transcript.absorb_bytes(&output_hash);
                layer_proofs.push(SerializedLayerProof::PassThrough);
                state_ops.push(format!("{} (passthrough)", layer.name));
                eprintln!(
                    "  passthrough({}): {:.1}ms",
                    layer.name,
                    layer_t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            _ => {}
        }
    }
    } // end else (sequential)


    ProveResult {
        layer_proofs,
        layer_meta,
        proved_ops,
        state_ops,
        segment_hashes,
        proof_size,
    }
}
