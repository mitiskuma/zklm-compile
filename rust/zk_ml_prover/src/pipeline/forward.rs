//! Forward pass execution for all op types.

use std::collections::HashMap;
use std::time::Instant;

use p3_field::{AbstractField, Field, PrimeField32};

use crate::field::common::{mod_sqrt_m31, is_qr_m31};
use crate::field::m31_ops::*;
use crate::proving::lookup::LookupTable;
use crate::proving::weight_commitment::commit_weights_fast;
use crate::protocol::*;
use crate::transformer::{
    LlamaLayerWeights, ModelConfig, NormType, ActivationType,
    llama_forward,
    QwenLayerWeights, qwen_forward_indexed,
    QwenLayerCommitments, commit_qwen_layer,
    build_table_index, TableIndex,
    layernorm_forward_simple,
};

use super::{F, LayerData, ForwardPassResult};

pub(super) fn forward_pass_all_ops(
    req: &mut ProveRequest,
    preloaded: Option<&HashMap<String, PreloadedLinear>>,
) -> ForwardPassResult {
    let t0 = Instant::now();

    let x_vals: Vec<F> = req.input.iter().map(|&v| F::from_canonical_u32(v)).collect();

    // Forward pass in M31
    let mut current = x_vals.clone();
    let mut intermediates: Vec<Vec<F>> = vec![current.clone()];
    // Named buffers for save/add_saved ops
    let mut saved_buffers: HashMap<String, Vec<F>> = HashMap::new();
    // Build GELU table lazily (only if needed)
    let _gelu_table: Option<LookupTable> = None;
    // Cache SiLU tables by scale to avoid rebuilding 65536-entry tables per layer
    let mut silu_table_cache: HashMap<i32, LookupTable> = HashMap::new();
    let mut sigmoid_table_cache: HashMap<i32, LookupTable> = HashMap::new();
    // Cache HashMap indices for O(1) lookup in forward pass
    let mut silu_index_cache: HashMap<i32, TableIndex> = HashMap::new();
    let mut sigmoid_index_cache: HashMap<i32, TableIndex> = HashMap::new();

    let mut layers: Vec<LayerData> = Vec::new();
    // Side-channel for llama_layer composite ops: stores data needed for proving/verifying
    let mut llama_layer_data: HashMap<usize, (LlamaLayerWeights, ModelConfig, crate::transformer::LlamaForwardTrace, i32)> = HashMap::new();
    let mut qwen_layer_data: HashMap<usize, (QwenLayerWeights, ModelConfig, crate::transformer::QwenForwardTrace, i32, i32)> = HashMap::new();

    for op in &mut req.ops {
        match op.op_type.as_str() {
            "linear" => {
                // Use preloaded weights by reference if available, otherwise convert from u32
                let (w_ref, w_owned, b_q, m, n) = if let Some(ref pl) = preloaded {
                    if let Some(pw) = pl.get(&op.name) {
                        let b: Vec<F> = op.b_q.iter().map(|&v| F::from_canonical_u32(v)).collect();
                        // Borrow — no clone!
                        (Some(&pw.w_f[..]), None, b, pw.m, pw.n)
                    } else {
                        let w: Vec<F> = op.w_q.iter().map(|&v| F::from_canonical_u32(v)).collect();
                        let b: Vec<F> = op.b_q.iter().map(|&v| F::from_canonical_u32(v)).collect();
                        (None, Some(w), b, op.m, op.n)
                    }
                } else {
                    let w: Vec<F> = op.w_q.iter().map(|&v| F::from_canonical_u32(v)).collect();
                    let b: Vec<F> = op.b_q.iter().map(|&v| F::from_canonical_u32(v)).collect();
                    (None, Some(w), b, op.m, op.n)
                };
                let w_slice: &[F] = w_ref.unwrap_or_else(|| w_owned.as_ref().unwrap());

                // Forward pass: parallel matvec for large layers
                use rayon::prelude::*;
                let cur_len = current.len();
                let result: Vec<F> = if m * n >= 4096 {
                    (0..m).into_par_iter().map(|i| {
                        let mut acc = F::zero();
                        let row = &w_slice[i * n..(i + 1) * n];
                        for j in 0..n.min(cur_len) {
                            acc += row[j] * current[j];
                        }
                        if i < b_q.len() { acc += b_q[i]; }
                        acc
                    }).collect()
                } else {
                    (0..m).map(|i| {
                        let mut acc = F::zero();
                        let row = &w_slice[i * n..(i + 1) * n];
                        for j in 0..n.min(cur_len) {
                            acc += row[j] * current[j];
                        }
                        if i < b_q.len() { acc += b_q[i]; }
                        acc
                    }).collect()
                };

                // Store owned weights for prove loop (or empty if preloaded)
                let w_for_layer = w_owned.unwrap_or_default();

                layers.push(LayerData {
                    op_type: "linear".into(),
                    name: op.name.clone(),
                    w_q: w_for_layer,
                    b_q,
                    m,
                    n,
                    output: result.clone(),
                    pre_relu: None,
                    gelu_input: None,
                    gelu_output: None,
                    gelu_scale: 0,
                    ln_gamma: None,
                    ln_beta: None,
                    ln_x: None,
                    ln_y: None,
                    ln_mu: None,
                    ln_r: None,
                    ln_d: None,
                    add_input: None,
                    add_saved_buf: None,
                    attn_q: None, attn_k: None, attn_v: None,
                    attn_num_heads: 0, attn_seq_len: 0, attn_d_head: 0, attn_exp_scale: 0,
                });
                current = result;
                intermediates.push(current.clone());
            }
            "relu" => {
                let pre_relu = current.clone();
                let result: Vec<F> = current
                    .iter()
                    .map(|&v| if from_field(v) >= 0 { v } else { F::zero() })
                    .collect();

                layers.push(LayerData {
                    op_type: "relu".into(),
                    name: op.name.clone(),
                    w_q: vec![],
                    b_q: vec![],
                    m: 0,
                    n: 0,
                    output: result.clone(),
                    pre_relu: Some(pre_relu),
                    gelu_input: None,
                    gelu_output: None,
                    gelu_scale: 0,
                    ln_gamma: None,
                    ln_beta: None,
                    ln_x: None,
                    ln_y: None,
                    ln_mu: None,
                    ln_r: None,
                    ln_d: None,
                    add_input: None,
                    add_saved_buf: None,
                    attn_q: None, attn_k: None, attn_v: None,
                    attn_num_heads: 0, attn_seq_len: 0, attn_d_head: 0, attn_exp_scale: 0,
                });
                current = result;
                intermediates.push(current.clone());
            }
            "gelu" => {
                let scale = op.gelu_scale;
                // Build full i16 GELU table (65536 entries) for this scale
                // Build table for this layer's scale (each layer may have different fc1 range)
                use crate::proving::gelu::build_gelu_table;
                let _table = build_gelu_table(scale);

                // Python provides requantized input/output in w_q/b_q
                let gelu_input: Vec<F> = op.w_q.iter().map(|&v| F::from_canonical_u32(v)).collect();
                let gelu_output: Vec<F> = op.b_q.iter().map(|&v| F::from_canonical_u32(v)).collect();

                // GELU output becomes the new current activations
                let result = gelu_output.clone();

                layers.push(LayerData {
                    op_type: "gelu".into(),
                    name: op.name.clone(),
                    w_q: vec![],
                    b_q: vec![],
                    m: 0,
                    n: 0,
                    output: result.clone(),
                    pre_relu: None,
                    gelu_input: Some(gelu_input),
                    gelu_output: Some(gelu_output),
                    gelu_scale: scale,
                    ln_gamma: None,
                    ln_beta: None,
                    ln_x: None,
                    ln_y: None,
                    ln_mu: None,
                    ln_r: None,
                    ln_d: None,
                    add_input: None,
                    add_saved_buf: None,
                    attn_q: None, attn_k: None, attn_v: None,
                    attn_num_heads: 0, attn_seq_len: 0, attn_d_head: 0, attn_exp_scale: 0,
                });
                current = result;
                intermediates.push(current.clone());
            }
            "attention" => {
                // current = c_attn output (3 * num_heads * d_head for seq_len=1)
                // or Q values if K/V provided externally
                let num_heads = op.num_heads;
                let seq_len = op.seq_len;
                let d_head = op.d_head;
                let head_size = seq_len * d_head;
                let total_qkv = num_heads * head_size;

                // Determine Q, K, V sources
                let (q, k, v_vals) = if !op.k_values.is_empty() {
                    // K/V provided externally (e.g. KV cache for seq_len > 1)
                    let q: Vec<F> = current[..total_qkv].to_vec();
                    let k: Vec<F> = op.k_values.iter().map(|&v| F::from_canonical_u32(v)).collect();
                    let v: Vec<F> = op.v_values.iter().map(|&v| F::from_canonical_u32(v)).collect();
                    (q, k, v)
                } else {
                    // Split current (3 * total_qkv) into Q, K, V
                    assert_eq!(current.len(), 3 * total_qkv,
                        "attention: expected {} (3*{}), got {}", 3 * total_qkv, total_qkv, current.len());
                    let q = current[..total_qkv].to_vec();
                    let k = current[total_qkv..2 * total_qkv].to_vec();
                    let v = current[2 * total_qkv..3 * total_qkv].to_vec();
                    (q, k, v)
                };

                assert_eq!(k.len(), total_qkv, "attention: K length mismatch");
                assert_eq!(v_vals.len(), total_qkv, "attention: V length mismatch");

                // Compute attention output: for each head, for each row:
                //   scores = Q[h][i] @ K[h]^T, attn = softmax(scores), out = attn @ V[h]
                let result = if seq_len == 1 {
                    // Special case: softmax of single element is always 1.0
                    // So attention output = V directly (for each head)
                    v_vals.clone()
                } else {
                    let exp_table = crate::proving::lookup::build_exp_table(op.exp_scale);
                    let mut index_map = HashMap::new();
                    for &(inp, out) in &exp_table.entries {
                        index_map.insert(inp, out);
                    }

                    let mut output = vec![F::zero(); total_qkv];
                    for h in 0..num_heads {
                        let q_head = &q[h * head_size..(h + 1) * head_size];
                        let k_head = &k[h * head_size..(h + 1) * head_size];
                        let v_head = &v_vals[h * head_size..(h + 1) * head_size];

                        for i in 0..seq_len {
                            let q_row = &q_head[i * d_head..(i + 1) * d_head];
                            let mut scores = vec![F::zero(); seq_len];
                            for j in 0..seq_len {
                                let mut acc = F::zero();
                                for l in 0..d_head {
                                    acc += k_head[j * d_head + l] * q_row[l];
                                }
                                scores[j] = acc;
                            }
                            let e: Vec<F> = scores.iter()
                                .map(|&s| F::from_canonical_u32(index_map[&s.as_canonical_u32()]))
                                .collect();
                            let sum: F = e.iter().copied().sum();
                            let inv_s = sum.inverse();
                            let y: Vec<F> = e.iter().map(|&ei| ei * inv_s).collect();
                            for j in 0..d_head {
                                let mut acc = F::zero();
                                for l in 0..seq_len {
                                    acc += y[l] * v_head[l * d_head + j];
                                }
                                output[h * head_size + i * d_head + j] = acc;
                            }
                        }
                    }
                    output
                };

                layers.push(LayerData {
                    op_type: "attention".into(),
                    name: op.name.clone(),
                    w_q: vec![],
                    b_q: vec![],
                    m: 0,
                    n: 0,
                    output: result.clone(),
                    pre_relu: None,
                    gelu_input: None,
                    gelu_output: None,
                    gelu_scale: 0,
                    ln_gamma: None,
                    ln_beta: None,
                    ln_x: None,
                    ln_y: None,
                    ln_mu: None,
                    ln_r: None,
                    ln_d: None,
                    add_input: None,
                    add_saved_buf: None,
                    attn_q: Some(q),
                    attn_k: Some(k),
                    attn_v: Some(v_vals),
                    attn_num_heads: num_heads,
                    attn_seq_len: seq_len,
                    attn_d_head: d_head,
                    attn_exp_scale: op.exp_scale,
                });
                current = result;
                intermediates.push(current.clone());
            }
            "layernorm" => {
                let gamma: Vec<F> = op.gamma.iter().map(|&v| F::from_canonical_u32(v)).collect();
                let beta_v: Vec<F> = op.beta.iter().map(|&v| F::from_canonical_u32(v)).collect();
                let d = current.len();

                // Compute LN in M31. For non-QR d/sum_sq, perturbs x[0] by small delta
                // until QR. This makes the squared proof work 100% of the time.
                let (result, mu, r, perturbation) =
                    layernorm_forward_simple(&current, &gamma, &beta_v, d);
                if perturbation != 0 {
                    eprintln!("  LN {}: perturbed x[0] by {} to get QR", op.name, perturbation);
                }

                let ln_x = if perturbation != 0 {
                    let mut x_perturbed = current.clone();
                    let adj = if perturbation > 0 {
                        F::from_canonical_u32(perturbation as u32)
                    } else {
                        F::zero() - F::from_canonical_u32((-perturbation) as u32)
                    };
                    x_perturbed[0] = x_perturbed[0] + adj;
                    x_perturbed
                } else {
                    current.clone()
                };
                let actual_op_type = "layernorm";
                layers.push(LayerData {
                    op_type: actual_op_type.into(),
                    name: op.name.clone(),
                    w_q: vec![],
                    b_q: vec![],
                    m: 0,
                    n: 0,
                    output: result.clone(),
                    pre_relu: None,
                    gelu_input: None,
                    gelu_output: None,
                    gelu_scale: 0,
                    ln_gamma: Some(gamma),
                    ln_beta: Some(beta_v),
                    ln_x: Some(ln_x),
                    ln_y: Some(result.clone()),
                    ln_mu: Some(mu),
                    ln_r: Some(r),
                    ln_d: Some(d),
                    add_input: None,
                    add_saved_buf: None,
                    attn_q: None, attn_k: None, attn_v: None,
                    attn_num_heads: 0, attn_seq_len: 0, attn_d_head: 0, attn_exp_scale: 0,
                });
                current = result;
                intermediates.push(current.clone());
            }
            "save" => {
                // Save current activations under a name. No proof needed.
                saved_buffers.insert(op.save_name.clone(), current.clone());
                layers.push(LayerData {
                    op_type: "save".into(),
                    name: op.name.clone(),
                    w_q: vec![],
                    b_q: vec![],
                    m: 0,
                    n: 0,
                    output: current.clone(),
                    pre_relu: None,
                    gelu_input: None,
                    gelu_output: None,
                    gelu_scale: 0,
                    ln_gamma: None,
                    ln_beta: None,
                    ln_x: None,
                    ln_y: None,
                    ln_mu: None,
                    ln_r: None,
                    ln_d: None,
                    add_input: None,
                    add_saved_buf: None,
                    attn_q: None, attn_k: None, attn_v: None,
                    attn_num_heads: 0, attn_seq_len: 0, attn_d_head: 0, attn_exp_scale: 0,
                });
                intermediates.push(current.clone());
            }
            "add_saved" => {
                // Add a saved buffer to current activations.
                let saved = saved_buffers
                    .get(&op.add_name)
                    .unwrap_or_else(|| panic!("No saved buffer named '{}'", op.add_name));
                assert_eq!(
                    current.len(),
                    saved.len(),
                    "add_saved: dimension mismatch"
                );
                let result: Vec<F> = current
                    .iter()
                    .zip(saved.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();

                layers.push(LayerData {
                    op_type: "add_saved".into(),
                    name: op.name.clone(),
                    w_q: vec![],
                    b_q: vec![],
                    m: 0,
                    n: 0,
                    output: result.clone(),
                    pre_relu: None,
                    gelu_input: None,
                    gelu_output: None,
                    gelu_scale: 0,
                    ln_gamma: None,
                    ln_beta: None,
                    ln_x: None,
                    ln_y: None,
                    ln_mu: None,
                    ln_r: None,
                    ln_d: None,
                    add_input: Some(current.clone()),
                    add_saved_buf: Some(saved.clone()),
                    attn_q: None, attn_k: None, attn_v: None,
                    attn_num_heads: 0, attn_seq_len: 0, attn_d_head: 0, attn_exp_scale: 0,
                });
                current = result;
                intermediates.push(current.clone());
            }
            "set_input" => {
                // Replace current activations with new quantized values.
                // Starts a new independent proof segment.
                let new_vals: Vec<F> = op.new_input.iter().map(|&v| F::from_canonical_u32(v)).collect();
                layers.push(LayerData {
                    op_type: "set_input".into(),
                    name: op.name.clone(),
                    w_q: vec![],
                    b_q: vec![],
                    m: 0,
                    n: 0,
                    output: new_vals.clone(),
                    pre_relu: None,
                    gelu_input: None,
                    gelu_output: None,
                    gelu_scale: 0,
                    ln_gamma: None,
                    ln_beta: None,
                    ln_x: None,
                    ln_y: None,
                    ln_mu: None,
                    ln_r: None,
                    ln_d: None,
                    add_input: None,
                    add_saved_buf: None,
                    attn_q: None, attn_k: None, attn_v: None,
                    attn_num_heads: 0, attn_seq_len: 0, attn_d_head: 0, attn_exp_scale: 0,
                });
                current = new_vals;
                intermediates.push(current.clone());
            }
            "rmsnorm" => {
                let gamma: Vec<F> = op.gamma.iter().map(|&v| F::from_canonical_u32(v)).collect();
                let d = current.len();

                // RMSNorm: y = gamma * x / rms(x). Use Python-provided output directly.
                let result: Vec<F> = if !op.ln_output.is_empty() {
                    op.ln_output.iter().map(|&v| F::from_canonical_u32(v)).collect()
                } else {
                    // Compute in M31
                    let sum_sq: F = current.iter().map(|&v| v * v).sum();
                    let d_f = F::from_canonical_u32(d as u32);
                    let target = d_f * sum_sq.inverse();
                    assert!(is_qr_m31(target), "RMSNorm: target is not a quadratic residue in M31");
                    let r = mod_sqrt_m31(target);
                    current.iter().zip(gamma.iter()).map(|(&xi, &gi)| gi * xi * r).collect()
                };

                layers.push(LayerData {
                    op_type: "rmsnorm".into(),
                    name: op.name.clone(),
                    w_q: vec![], b_q: vec![],
                    m: 0, n: 0,
                    output: result.clone(),
                    pre_relu: None,
                    gelu_input: None, gelu_output: None, gelu_scale: 0,
                    ln_gamma: Some(gamma),
                    ln_beta: None,
                    ln_x: Some(current.clone()),
                    ln_y: Some(result.clone()),
                    ln_mu: None, ln_r: None, ln_d: Some(d),
                    add_input: None, add_saved_buf: None,
                    attn_q: None, attn_k: None, attn_v: None,
                    attn_num_heads: 0, attn_seq_len: 0, attn_d_head: 0, attn_exp_scale: 0,
                });
                current = result;
                intermediates.push(current.clone());
            }
            "silu" => {
                // SiLU lookup: same pattern as GELU
                let silu_input: Vec<F> = op.w_q.iter().map(|&v| F::from_canonical_u32(v)).collect();
                let silu_output: Vec<F> = op.b_q.iter().map(|&v| F::from_canonical_u32(v)).collect();

                layers.push(LayerData {
                    op_type: "silu".into(),
                    name: op.name.clone(),
                    w_q: vec![], b_q: vec![],
                    m: 0, n: 0,
                    output: silu_output.clone(),
                    pre_relu: None,
                    gelu_input: Some(silu_input),
                    gelu_output: Some(silu_output.clone()),
                    gelu_scale: op.gelu_scale,
                    ln_gamma: None, ln_beta: None, ln_x: None, ln_y: None,
                    ln_mu: None, ln_r: None, ln_d: None,
                    add_input: None, add_saved_buf: None,
                    attn_q: None, attn_k: None, attn_v: None,
                    attn_num_heads: 0, attn_seq_len: 0, attn_d_head: 0, attn_exp_scale: 0,
                });
                current = silu_output;
                intermediates.push(current.clone());
            }
            "swiglu" => {
                // SwiGLU: gate (w_q), gate_silu (b_q), up (ln_output)
                let gate: Vec<F> = op.w_q.iter().map(|&v| F::from_canonical_u32(v)).collect();
                let gate_silu: Vec<F> = op.b_q.iter().map(|&v| F::from_canonical_u32(v)).collect();
                let up: Vec<F> = op.ln_output.iter().map(|&v| F::from_canonical_u32(v)).collect();
                let output: Vec<F> = gate_silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();

                layers.push(LayerData {
                    op_type: "swiglu".into(),
                    name: op.name.clone(),
                    w_q: vec![], b_q: vec![],
                    m: 0, n: 0,
                    output: output.clone(),
                    pre_relu: None,
                    gelu_input: Some(gate),
                    gelu_output: Some(gate_silu),
                    gelu_scale: op.gelu_scale,
                    ln_gamma: None, ln_beta: None,
                    ln_x: None,
                    ln_y: Some(up), // reuse ln_y for up_values
                    ln_mu: None, ln_r: None, ln_d: None,
                    add_input: None, add_saved_buf: None,
                    attn_q: None, attn_k: None, attn_v: None,
                    attn_num_heads: 0, attn_seq_len: 0, attn_d_head: 0, attn_exp_scale: 0,
                });
                current = output;
                intermediates.push(current.clone());
            }
            "gqa_attention" => {
                let num_q_heads = op.num_heads;
                let num_kv_heads = op.n; // packed into n field
                let seq_len = op.seq_len;
                let d_head = op.d_head;
                let q_head_size = seq_len * d_head;
                let total_q = num_q_heads * q_head_size;
                let total_kv = num_kv_heads * q_head_size;

                let q: Vec<F> = current[..total_q].to_vec();
                let k: Vec<F> = if !op.k_values.is_empty() {
                    op.k_values.iter().map(|&v| F::from_canonical_u32(v)).collect()
                } else {
                    current[total_q..total_q + total_kv].to_vec()
                };
                let v_vals: Vec<F> = if !op.v_values.is_empty() {
                    op.v_values.iter().map(|&v| F::from_canonical_u32(v)).collect()
                } else {
                    current[total_q + total_kv..total_q + 2 * total_kv].to_vec()
                };

                let result = if seq_len == 1 {
                    // seq_len=1: attention = V mapped through GQA grouping
                    let heads_per_group = num_q_heads / num_kv_heads;
                    let mut out = Vec::with_capacity(total_q);
                    for h in 0..num_q_heads {
                        let kv_idx = h / heads_per_group;
                        for d in 0..d_head {
                            out.push(v_vals[kv_idx * d_head + d]);
                        }
                    }
                    out
                } else {
                    // Full GQA attention
                    let exp_table = crate::proving::lookup::build_exp_table(op.exp_scale);
                    let mut index_map = HashMap::new();
                    for &(inp, out) in &exp_table.entries {
                        index_map.insert(inp, out);
                    }
                    let heads_per_group = num_q_heads / num_kv_heads;
                    let mut output = vec![F::zero(); total_q];
                    for h in 0..num_q_heads {
                        let kv_idx = h / heads_per_group;
                        let q_head = &q[h * q_head_size..(h + 1) * q_head_size];
                        let k_head = &k[kv_idx * q_head_size..(kv_idx + 1) * q_head_size];
                        let v_head = &v_vals[kv_idx * q_head_size..(kv_idx + 1) * q_head_size];
                        for i in 0..seq_len {
                            let q_row = &q_head[i * d_head..(i + 1) * d_head];
                            let mut scores = vec![F::zero(); seq_len];
                            for j in 0..seq_len {
                                let mut acc = F::zero();
                                for l in 0..d_head {
                                    acc += k_head[j * d_head + l] * q_row[l];
                                }
                                scores[j] = acc;
                            }
                            let e: Vec<F> = scores.iter()
                                .map(|&s| F::from_canonical_u32(index_map[&s.as_canonical_u32()]))
                                .collect();
                            let sum: F = e.iter().copied().sum();
                            let inv_s = sum.inverse();
                            let y: Vec<F> = e.iter().map(|&ei| ei * inv_s).collect();
                            for j in 0..d_head {
                                let mut acc = F::zero();
                                for l in 0..seq_len {
                                    acc += y[l] * v_head[l * d_head + j];
                                }
                                output[h * q_head_size + i * d_head + j] = acc;
                            }
                        }
                    }
                    output
                };

                layers.push(LayerData {
                    op_type: "gqa_attention".into(),
                    name: op.name.clone(),
                    w_q: vec![], b_q: vec![],
                    m: num_q_heads, n: num_kv_heads,
                    output: result.clone(),
                    pre_relu: None,
                    gelu_input: None, gelu_output: None, gelu_scale: 0,
                    ln_gamma: None, ln_beta: None, ln_x: None, ln_y: None,
                    ln_mu: None, ln_r: None, ln_d: None,
                    add_input: None, add_saved_buf: None,
                    attn_q: Some(q), attn_k: Some(k), attn_v: Some(v_vals),
                    attn_num_heads: num_q_heads, attn_seq_len: seq_len, attn_d_head: d_head,
                    attn_exp_scale: op.exp_scale,
                });
                current = result;
                intermediates.push(current.clone());
            }
            "llama_layer" => {
                let lc = op.llama_config.as_ref().expect("llama_layer requires llama_config");
                // Move weights out of op — zero-copy, no allocation
                let lw = op.llama_weights.take().expect("llama_layer requires llama_weights");

                let config = ModelConfig {
                    d_model: lc.d_model, d_ff: lc.d_ff,
                    num_q_heads: lc.num_q_heads, num_kv_heads: lc.num_kv_heads,
                    d_head: lc.d_head, n_layers: 1, vocab_size: 0,
                    norm_type: NormType::RMSNorm, activation: ActivationType::SwiGLU,
                    v_num_heads: 0, v_d_head: 0,  // Llama: symmetric → fall back to num_kv_heads/d_head
                };
                // Weights are already Vec<F> — move directly, no conversion or cloning
                let weights = LlamaLayerWeights {
                    norm1_gamma: lw.norm1_gamma, w_q: lw.w_q, w_k: lw.w_k, w_v: lw.w_v,
                    w_o: lw.w_o, norm2_gamma: lw.norm2_gamma,
                    w_gate: lw.w_gate, w_up: lw.w_up, w_down: lw.w_down,
                };
                // Cache SiLU table — avoid rebuilding 65536-entry table for every layer
                silu_table_cache.entry(lc.silu_scale)
                    .or_insert_with(|| crate::proving::lookup::build_silu_table(lc.silu_scale));
                let silu_table = silu_table_cache.get(&lc.silu_scale).unwrap();
                let trace = llama_forward(&current, &weights, &config, silu_table);
                let result = trace.output.clone();

                let layer_idx = layers.len();
                llama_layer_data.insert(layer_idx, (weights, config, trace, lc.silu_scale));

                layers.push(LayerData {
                    op_type: "llama_layer".into(),
                    name: op.name.clone(),
                    w_q: vec![], b_q: vec![],
                    m: 0, n: 0,
                    output: result.clone(),
                    pre_relu: None,
                    gelu_input: None, gelu_output: None, gelu_scale: 0,
                    ln_gamma: None, ln_beta: None, ln_x: None, ln_y: None,
                    ln_mu: None, ln_r: None, ln_d: None,
                    add_input: None, add_saved_buf: None,
                    attn_q: None, attn_k: None, attn_v: None,
                    attn_num_heads: 0, attn_seq_len: 0, attn_d_head: 0, attn_exp_scale: 0,
                });
                current = result;
                intermediates.push(current.clone());
            }
            "qwen_layer" => {
                let qc = op.qwen_config.as_ref().expect("qwen_layer requires qwen_config");
                let qw = op.qwen_weights.take().expect("qwen_layer requires qwen_weights");

                let config = ModelConfig {
                    d_model: qc.d_model, d_ff: qc.d_ff,
                    num_q_heads: qc.num_q_heads, num_kv_heads: qc.num_kv_heads,
                    d_head: qc.d_head, n_layers: 1, vocab_size: 0,
                    norm_type: NormType::RMSNorm, activation: ActivationType::SwiGLU,
                    v_num_heads: qc.v_num_heads, v_d_head: qc.v_d_head,
                };
                let weights = QwenLayerWeights {
                    norm1_gamma: qw.norm1_gamma, w_q: qw.w_q, w_k: qw.w_k, w_v: qw.w_v,
                    w_o: qw.w_o, w_g_proj: qw.w_g_proj, norm2_gamma: qw.norm2_gamma,
                    w_gate: qw.w_gate, w_up: qw.w_up, w_down: qw.w_down,
                };
                silu_table_cache.entry(qc.silu_scale)
                    .or_insert_with(|| crate::proving::lookup::build_silu_table(qc.silu_scale));
                let silu_table = silu_table_cache.get(&qc.silu_scale).unwrap();
                sigmoid_table_cache.entry(qc.sigmoid_scale)
                    .or_insert_with(|| crate::proving::lookup::build_sigmoid_table(qc.sigmoid_scale));
                let sigmoid_table = sigmoid_table_cache.get(&qc.sigmoid_scale).unwrap();
                silu_index_cache.entry(qc.silu_scale)
                    .or_insert_with(|| build_table_index(silu_table));
                let silu_index = silu_index_cache.get(&qc.silu_scale).unwrap();
                sigmoid_index_cache.entry(qc.sigmoid_scale)
                    .or_insert_with(|| build_table_index(sigmoid_table));
                let sigmoid_index = sigmoid_index_cache.get(&qc.sigmoid_scale).unwrap();
                let trace = qwen_forward_indexed(&current, &weights, &config, silu_table, sigmoid_table, silu_index, sigmoid_index);
                let result = trace.output.clone();

                let layer_idx = layers.len();
                qwen_layer_data.insert(layer_idx, (weights, config, trace, qc.silu_scale, qc.sigmoid_scale));

                layers.push(LayerData {
                    op_type: "qwen_layer".into(),
                    name: op.name.clone(),
                    w_q: vec![], b_q: vec![],
                    m: 0, n: 0,
                    output: result.clone(),
                    pre_relu: None,
                    gelu_input: None, gelu_output: None, gelu_scale: 0,
                    ln_gamma: None, ln_beta: None, ln_x: None, ln_y: None,
                    ln_mu: None, ln_r: None, ln_d: None,
                    add_input: None, add_saved_buf: None,
                    attn_q: None, attn_k: None, attn_v: None,
                    attn_num_heads: 0, attn_seq_len: 0, attn_d_head: 0, attn_exp_scale: 0,
                });
                current = result;
                intermediates.push(current.clone());
            }
            _ => panic!("Unknown op type: {}", op.op_type),
        }
    }

    eprintln!(
        "  forward pass: {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Commit weights (use preloaded commitments if available)
    let t_commit = Instant::now();
    let mut weight_commitments = Vec::new();
    let mut linear_idx = 0;
    for op in &req.ops {
        if op.op_type == "linear" {
            if let Some(ref preloaded_map) = preloaded {
                // In server mode, find matching preloaded weight by name
                if let Some(pw) = preloaded_map.get(&op.name) {
                    weight_commitments.push(pw.commitment.clone());
                } else {
                    weight_commitments.push(commit_weights_fast(&layers[linear_idx].w_q));
                }
            } else {
                weight_commitments.push(commit_weights_fast(&layers[linear_idx].w_q));
            }
            linear_idx += 1;
        }
    }
    // Recount for correct linear_idx tracking (layers may include non-linear ops)
    // Actually linear_idx above doesn't track correctly since layers includes all ops.
    // Let's redo this properly.
    weight_commitments.clear();
    for layer in &layers {
        if layer.op_type == "linear" {
            if let Some(ref preloaded_map) = preloaded {
                if let Some(pw) = preloaded_map.get(&layer.name) {
                    weight_commitments.push(pw.commitment.clone());
                } else {
                    weight_commitments.push(commit_weights_fast(&layer.w_q));
                }
            } else {
                weight_commitments.push(commit_weights_fast(&layer.w_q));
            }
        }
    }
    eprintln!(
        "  commit weights: {:.1}ms",
        t_commit.elapsed().as_secs_f64() * 1000.0
    );

    // Pre-compute qwen layer commitments (one-time cost, excluded from prove_time).
    // In production, the model publisher distributes these with the model.
    let all_qwen = layers.iter().all(|l| l.op_type == "qwen_layer");
    let mut qwen_commit_duration = std::time::Duration::ZERO;
    let qwen_commitments: Option<Vec<QwenLayerCommitments>> = if all_qwen && layers.len() > 1 {
        let t_qwen_commit = Instant::now();
        let commitments: Vec<QwenLayerCommitments> = (0..layers.len())
            .map(|idx| {
                let (ref weights, ref config, _, _, _) =
                    qwen_layer_data.get(&idx).expect("qwen_layer data missing");
                commit_qwen_layer(weights, config)
            })
            .collect();
        qwen_commit_duration = t_qwen_commit.elapsed();
        eprintln!("  qwen commit: {:.1}ms ({} layers)", qwen_commit_duration.as_secs_f64() * 1000.0, layers.len());
        Some(commitments)
    } else {
        None
    };

    ForwardPassResult {
        layers,
        intermediates,
        current,
        llama_layer_data,
        qwen_layer_data,
        silu_table_cache,
        sigmoid_table_cache,
        weight_commitments,
        qwen_commitments,
        all_qwen,
        qwen_commit_duration,
    }
}
