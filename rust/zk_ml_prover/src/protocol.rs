use std::io::{self, BufReader, Read};
use p3_field::PrimeField32;
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};
use crate::proving::sumcheck::SumcheckProof;
use crate::proving::weight_commitment::{WeightCommitment, MleEvalProof};
use crate::proving::matmul::SuccinctMatmulProof;
use crate::proving::layernorm::{LayerNormProof, LayerNormSqrProof};
use crate::proving::gelu::GeluProof;
use crate::proving::attention::RowAttentionProof;
use crate::field::common::i16_to_field;

type F = Mersenne31;

// ===== JSON Protocol =====

/// Top-level request. Dispatches based on `mode`.
///   - "mlp" (default): flat linear+relu sequence (existing behavior)
///   - "gpt2": full GPT-2 transformer (N layers + final LN + LM head)
#[derive(Deserialize)]
pub(crate) struct ProveRequest {
    #[serde(default = "default_mode")]
    pub(crate) mode: String,
    /// Input activations (M31 field elements as u32).
    pub(crate) input: Vec<u32>,

    // --- MLP mode fields ---
    #[serde(default)]
    pub(crate) ops: Vec<OpDesc>,

    // --- GPT-2 mode fields ---
    #[serde(default)]
    pub(crate) gpt2: Option<GPT2Desc>,
}

pub(crate) fn default_mode() -> String {
    "mlp".to_string()
}

#[derive(Deserialize)]
// TODO: Replace with type-safe enum Op { Linear{..}, Relu{..}, Gelu{..}, LayerNorm{..}, ... }
// Current flat struct is correct but wastes memory (empty vecs for non-matching types).
// Enum would catch type errors at compile time and reduce allocation.
pub(crate) struct OpDesc {
    #[serde(rename = "type")]
    pub(crate) op_type: String,
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) m: usize,
    #[serde(default)]
    pub(crate) n: usize,
    #[serde(default)]
    pub(crate) w_q: Vec<u32>,
    #[serde(default)]
    pub(crate) b_q: Vec<u32>,
    // --- GELU fields ---
    #[serde(default = "default_gelu_scale")]
    pub(crate) gelu_scale: i32,
    // --- LayerNorm fields ---
    #[serde(default)]
    pub(crate) gamma: Vec<u32>,
    #[serde(default)]
    pub(crate) beta: Vec<u32>,
    // --- save / add_saved fields ---
    #[serde(default)]
    pub(crate) save_name: String,
    #[serde(default)]
    pub(crate) add_name: String,
    // --- set_input fields ---
    #[serde(default)]
    pub(crate) new_input: Vec<u32>,
    // --- Pre-computed LN output (from Python float computation) ---
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) ln_output: Vec<u32>,
    // --- Attention fields ---
    #[serde(default)]
    pub(crate) num_heads: usize,
    #[serde(default)]
    pub(crate) seq_len: usize,
    #[serde(default)]
    pub(crate) d_head: usize,
    #[serde(default)]
    pub(crate) exp_scale: i32,
    #[serde(default)]
    pub(crate) k_values: Vec<u32>,
    #[serde(default)]
    pub(crate) v_values: Vec<u32>,
    // --- Llama layer fields ---
    #[serde(default)]
    pub(crate) llama_config: Option<LlamaLayerConfig>,
    #[serde(default)]
    pub(crate) llama_weights: Option<LlamaLayerWeightData>,
    // --- Qwen layer fields ---
    #[serde(default)]
    pub(crate) qwen_config: Option<QwenLayerConfig>,
    #[serde(default)]
    pub(crate) qwen_weights: Option<QwenLayerWeightData>,
}

#[derive(Deserialize, Clone, Default)]
pub(crate) struct LlamaLayerConfig {
    pub(crate) d_model: usize,
    pub(crate) d_ff: usize,
    pub(crate) num_q_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) d_head: usize,
    pub(crate) silu_scale: i32,
}

#[derive(Deserialize, Clone, Default)]
pub(crate) struct LlamaLayerWeightData {
    pub(crate) norm1_gamma: Vec<F>,
    pub(crate) w_q: Vec<F>,
    pub(crate) w_k: Vec<F>,
    pub(crate) w_v: Vec<F>,
    pub(crate) w_o: Vec<F>,
    pub(crate) norm2_gamma: Vec<F>,
    pub(crate) w_gate: Vec<F>,
    pub(crate) w_up: Vec<F>,
    pub(crate) w_down: Vec<F>,
}

#[derive(Deserialize, Clone, Default)]
pub(crate) struct QwenLayerConfig {
    pub(crate) d_model: usize,
    pub(crate) d_ff: usize,
    pub(crate) num_q_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) d_head: usize,
    pub(crate) silu_scale: i32,
    pub(crate) sigmoid_scale: i32,
}

#[derive(Deserialize, Clone, Default)]
pub(crate) struct QwenLayerWeightData {
    pub(crate) norm1_gamma: Vec<F>,
    pub(crate) w_q: Vec<F>,
    pub(crate) w_k: Vec<F>,
    pub(crate) w_v: Vec<F>,
    pub(crate) w_o: Vec<F>,
    pub(crate) w_g_proj: Vec<F>,
    pub(crate) norm2_gamma: Vec<F>,
    pub(crate) w_gate: Vec<F>,
    pub(crate) w_up: Vec<F>,
    pub(crate) w_down: Vec<F>,
}

/// Zero-copy transmute Vec<u32> → Vec<F> (Mersenne31 is #[repr(transparent)] around u32).
/// All values must be in canonical form (< 2^31 - 1).
pub(crate) fn transmute_u32_to_f(v: Vec<u32>) -> Vec<F> {
    // Safety: Mersenne31 is #[repr(transparent)] over u32.
    // These assertions verify the layout invariants at runtime (cannot be const
    // because F is a concrete type alias, not a generic, but the checks are
    // optimized away after the first call since the sizes are known at compile time).
    assert!(
        std::mem::size_of::<Mersenne31>() == std::mem::size_of::<u32>(),
        "Mersenne31 size does not match u32"
    );
    assert!(
        std::mem::align_of::<Mersenne31>() == std::mem::align_of::<u32>(),
        "Mersenne31 alignment does not match u32"
    );
    let mut v = std::mem::ManuallyDrop::new(v);
    let ptr = v.as_mut_ptr() as *mut F;
    let len = v.len();
    let cap = v.capacity();
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

pub(crate) fn default_gelu_scale() -> i32 {
    1000
}

/// GPT-2 model description for the "gpt2" mode.
#[derive(Deserialize)]
pub(crate) struct GPT2Desc {
    pub(crate) d_model: usize,
    pub(crate) d_ff: usize,
    pub(crate) vocab_size: usize,
    #[serde(default = "default_gelu_scale")]
    pub(crate) gelu_scale: i32,
    pub(crate) layers: Vec<TransformerLayerDesc>,
    pub(crate) final_ln_gamma: Vec<u32>,
    pub(crate) final_ln_beta: Vec<u32>,
    pub(crate) lm_head: Vec<u32>, // vocab_size * d_model
}

#[derive(Deserialize)]
pub(crate) struct TransformerLayerDesc {
    pub(crate) ln1_gamma: Vec<u32>,
    pub(crate) ln1_beta: Vec<u32>,
    pub(crate) w_attn: Vec<u32>, // d_model * d_model
    pub(crate) ln2_gamma: Vec<u32>,
    pub(crate) ln2_beta: Vec<u32>,
    pub(crate) w_fc1: Vec<u32>,  // d_ff * d_model
    pub(crate) b_fc1: Vec<u32>,  // d_ff
    pub(crate) w_fc2: Vec<u32>,  // d_model * d_ff
    pub(crate) b_fc2: Vec<u32>,  // d_model
}

#[derive(Serialize)]
pub(crate) struct ProofCoverage {
    pub(crate) proved_ops: Vec<String>,
    pub(crate) state_ops: Vec<String>,
    pub(crate) proved_count: usize,
    pub(crate) state_count: usize,
    /// Computational ops only (excludes save/set_input state management).
    pub(crate) computational_count: usize,
    pub(crate) computational_total: usize,
    pub(crate) total_count: usize,
}

#[derive(Serialize)]
pub(crate) struct ProveResponse {
    pub(crate) valid: bool,
    pub(crate) prediction: i64,
    pub(crate) output: Vec<i64>,
    pub(crate) prove_time_ms: f64,
    pub(crate) verify_time_ms: f64,
    pub(crate) proof_size_bytes: usize,
    pub(crate) weight_commitments: Vec<String>,
    pub(crate) input_commitment: String,
    pub(crate) succinct_verification: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) coverage: Option<ProofCoverage>,
}

// ===== Proof structures (MLP mode) =====

#[derive(Serialize, Deserialize)]
pub(crate) struct FullProof {
    pub(crate) layer_proofs: Vec<SerializedLayerProof>,
    pub(crate) output: Vec<u32>,
    pub(crate) input_hash: [u8; 32],
    pub(crate) layer_meta: Vec<LayerMeta>,
    #[serde(default)]
    pub(crate) weight_commitments: Vec<String>,
    #[serde(default)]
    pub(crate) coverage: Option<SerializedCoverage>,
    /// Hash of activations at each set_input boundary.
    /// segment_hashes[i] = H(values) at i-th set_input op.
    /// Enables cross-segment audit: verifier can check that set_input values
    /// match the previous segment's output (once all-Rust forward pass eliminates
    /// the Python intermediary, these become provably linked).
    #[serde(default)]
    pub(crate) segment_hashes: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct SerializedCoverage {
    pub(crate) proved_ops: Vec<String>,
    pub(crate) state_ops: Vec<String>,
    pub(crate) proved_count: usize,
    pub(crate) state_count: usize,
    pub(crate) computational_count: usize,
    pub(crate) computational_total: usize,
    pub(crate) total_count: usize,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct LayerMeta {
    pub(crate) op_type: String,
    pub(crate) name: String,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) b_q: Vec<u32>,
    #[serde(default)]
    pub(crate) attn_q: Vec<u32>,
    #[serde(default)]
    pub(crate) attn_k: Vec<u32>,
    #[serde(default)]
    pub(crate) attn_v: Vec<u32>,
    #[serde(default)]
    pub(crate) attn_exp_scale: i32,
    pub(crate) output: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
pub(crate) enum SerializedLayerProof {
    Matmul(SuccinctMatmulProof),
    Relu {
        a_commitment: WeightCommitment,
        a_at_r: u32,
        a_eval_proof: MleEvalProof,
        chain_eval_proof: Option<MleEvalProof>,
        chain_value: Option<u32>,
        product_proof: SumcheckProof,
        product_finals: (u32, u32, u32),
        boolean_proof: SumcheckProof,
        boolean_finals: (u32, u32, u32),
    },
    LayerNorm(LayerNormProof),
    LayerNormSqr(LayerNormSqrProof),
    Gelu(GeluProof),
    Attention(RowAttentionProof),
    /// Residual add: c[i] = a[i] + b[i]. Verifier evaluates saved buffer MLE directly.
    Add {
        /// Name of the save layer whose output is the saved buffer (b).
        save_name: String,
    },
    /// Cross-segment commitment: proves set_input boundary links to previous segment's output.
    SegmentBoundary {
        prev_output_hash: [u8; 32],
        new_input_hash: [u8; 32],
    },
    /// Save commitment: proves saved buffer content is committed for later add_saved verification.
    SaveCommitment {
        save_name: String,
        content_hash: [u8; 32],
    },
    RmsNorm(crate::proving::rmsnorm::RmsNormProof),
    Silu(crate::proving::silu::SiluProof),
    SwiGlu(crate::proving::swiglu::SwiGluProof),
    GqaAttention(RowAttentionProof),
    LlamaLayer(crate::transformer::LlamaLayerProof),
    QwenLayer(crate::transformer::QwenLayerProof),
    QwenLayerEF(crate::transformer::QwenLayerProofEF),
    /// Legacy passthrough — only for truly unproved ops (should be empty after Phase 2).
    PassThrough,
}

// ===== Server Mode =====

pub(crate) struct PreloadedLinear {
    pub(crate) w_f: Vec<F>,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) commitment: WeightCommitment,
}

#[allow(dead_code)]
pub(crate) enum ServerOp {
    LinearRef { name: String, weight_name: String, bias: Vec<F> },
    Relu { name: String },
    Gelu { name: String, gelu_scale: i32, gelu_input_i16: Vec<i16>, gelu_output_i16: Vec<i16> },
    SetInput { name: String, new_input: Vec<u32> },
    LayerNorm { name: String, gamma: Vec<F>, beta: Vec<F>, ln_output: Vec<u32> },
    Save { name: String, save_name: String },
    AddSaved { name: String, add_name: String },
    Attention {
        name: String,
        num_heads: usize,
        seq_len: usize,
        d_head: usize,
        exp_scale: i32,
        k_values: Vec<u32>,
        v_values: Vec<u32>,
    },
    RmsNorm {
        name: String,
        gamma: Vec<F>,
        rmsnorm_output: Vec<u32>,
    },
    Silu {
        name: String,
        silu_scale: i32,
        silu_input_i16: Vec<i16>,
        silu_output_i16: Vec<i16>,
    },
    SwiGlu {
        name: String,
        silu_scale: i32,
        gate_i16: Vec<i16>,
        gate_silu_i16: Vec<i16>,
        up_values: Vec<u32>,
    },
    GqaAttention {
        name: String,
        num_q_heads: usize,
        num_kv_heads: usize,
        seq_len: usize,
        d_head: usize,
        exp_scale: i32,
        k_values: Vec<u32>,
        v_values: Vec<u32>,
    },
    LlamaLayer {
        name: String,
        config: LlamaLayerConfig,
        weights: LlamaLayerWeightData,
    },
    LlamaLayerRef {
        name: String,
        config: LlamaLayerConfig,
        weight_names: [String; 9],  // norm1_gamma, w_q, w_k, w_v, w_o, norm2_gamma, w_gate, w_up, w_down
    },
    QwenLayer {
        name: String,
        config: QwenLayerConfig,
        weights: QwenLayerWeightData,
    },
    QwenLayerRef {
        name: String,
        config: QwenLayerConfig,
        weight_names: [String; 10],  // norm1_gamma, w_q, w_k, w_v, w_o, w_g_proj, norm2_gamma, w_gate, w_up, w_down
    },
}

pub(crate) fn read_u32_from(reader: &mut BufReader<io::Stdin>) -> u32 {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).unwrap();
    u32::from_le_bytes(buf)
}

pub(crate) fn read_u32_from_opt(reader: &mut BufReader<io::Stdin>) -> Option<u32> {
    let mut buf = [0u8; 4];
    match reader.read_exact(&mut buf) {
        Ok(()) => Some(u32::from_le_bytes(buf)),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => None,
        Err(e) => panic!("IO error reading u32: {}", e),
    }
}

/// Parse the binary protocol (after the 0x00 magic byte has been consumed).
///
/// Format:
///   [u32 LE] num_ops
///   [u32 LE] input_len
///   [u32 LE * input_len] input values
///   For each op:
///     [u8] op_type: 0=linear, 1=relu, 2=set_input, 3=layernorm, 4=save, 5=add_saved, 6=gelu
///     [u32 LE] name_len
///     [u8 * name_len] name bytes
///     If linear: [u32 LE] m, [u32 LE] n, [u32 LE * m*n] w_q, [u32 LE * m] b_q
///     If relu: (nothing)
///     If set_input: [u32 LE] new_input_len, [u32 LE * new_input_len] values
pub(crate) fn parse_binary(data: &[u8]) -> ProveRequest {
    let mut pos = 0;

    let read_u32 = |pos: &mut usize| -> u32 {
        let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        v
    };

    let num_ops = read_u32(&mut pos);
    let input_len = read_u32(&mut pos) as usize;

    let mut input = Vec::with_capacity(input_len);
    for _ in 0..input_len {
        input.push(read_u32(&mut pos));
    }

    let mut ops = Vec::with_capacity(num_ops as usize);
    for _ in 0..num_ops {
        let op_type_byte = data[pos];
        pos += 1;

        let name_len = read_u32(&mut pos) as usize;
        let name = String::from_utf8(data[pos..pos + name_len].to_vec())
            .expect("invalid UTF-8 in op name");
        pos += name_len;

        match op_type_byte {
            0 => {
                // linear
                let m = read_u32(&mut pos) as usize;
                let n = read_u32(&mut pos) as usize;
                let mut w_q = Vec::with_capacity(m * n);
                for _ in 0..m * n {
                    w_q.push(read_u32(&mut pos));
                }
                let mut b_q = Vec::with_capacity(m);
                for _ in 0..m {
                    b_q.push(read_u32(&mut pos));
                }
                ops.push(OpDesc {
                    op_type: "linear".into(),
                    name, m, n, w_q, b_q,
                    gelu_scale: default_gelu_scale(),
                    gamma: vec![], beta: vec![],
                    save_name: String::new(), add_name: String::new(),
                    new_input: vec![], ln_output: vec![],
                    num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                    k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                });
            }
            1 => {
                // relu
                ops.push(OpDesc {
                    op_type: "relu".into(), name,
                    m: 0, n: 0, w_q: vec![], b_q: vec![],
                    gelu_scale: default_gelu_scale(),
                    gamma: vec![], beta: vec![],
                    save_name: String::new(), add_name: String::new(),
                    new_input: vec![], ln_output: vec![],
                    num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                    k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                });
            }
            2 => {
                // set_input
                let new_input_len = read_u32(&mut pos) as usize;
                let mut new_input = Vec::with_capacity(new_input_len);
                for _ in 0..new_input_len {
                    new_input.push(read_u32(&mut pos));
                }
                ops.push(OpDesc {
                    op_type: "set_input".into(), name,
                    m: 0, n: 0, w_q: vec![], b_q: vec![],
                    gelu_scale: default_gelu_scale(),
                    gamma: vec![], beta: vec![],
                    save_name: String::new(), add_name: String::new(),
                    new_input, ln_output: vec![],
                    num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                    k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                });
            }
            3 => {
                // layernorm
                let gamma_len = read_u32(&mut pos) as usize;
                let mut gamma = Vec::with_capacity(gamma_len);
                for _ in 0..gamma_len {
                    gamma.push(read_u32(&mut pos));
                }
                let beta_len = read_u32(&mut pos) as usize;
                let mut beta = Vec::with_capacity(beta_len);
                for _ in 0..beta_len {
                    beta.push(read_u32(&mut pos));
                }
                let ln_out_len = read_u32(&mut pos) as usize;
                let mut ln_output = Vec::with_capacity(ln_out_len);
                for _ in 0..ln_out_len {
                    ln_output.push(read_u32(&mut pos));
                }
                ops.push(OpDesc {
                    op_type: "layernorm".into(), name,
                    m: 0, n: 0, w_q: vec![], b_q: vec![],
                    gelu_scale: default_gelu_scale(),
                    gamma, beta,
                    save_name: String::new(), add_name: String::new(),
                    new_input: vec![], ln_output,
                    num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                    k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                });
            }
            4 => {
                // save
                let sname_len = read_u32(&mut pos) as usize;
                let save_name = String::from_utf8(data[pos..pos + sname_len].to_vec())
                    .expect("invalid UTF-8 in save_name");
                pos += sname_len;
                ops.push(OpDesc {
                    op_type: "save".into(), name,
                    m: 0, n: 0, w_q: vec![], b_q: vec![],
                    gelu_scale: default_gelu_scale(),
                    gamma: vec![], beta: vec![],
                    save_name, add_name: String::new(),
                    new_input: vec![], ln_output: vec![],
                    num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                    k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                });
            }
            5 => {
                // add_saved
                let aname_len = read_u32(&mut pos) as usize;
                let add_name = String::from_utf8(data[pos..pos + aname_len].to_vec())
                    .expect("invalid UTF-8 in add_name");
                pos += aname_len;
                ops.push(OpDesc {
                    op_type: "add_saved".into(), name,
                    m: 0, n: 0, w_q: vec![], b_q: vec![],
                    gelu_scale: default_gelu_scale(),
                    gamma: vec![], beta: vec![],
                    save_name: String::new(), add_name,
                    new_input: vec![], ln_output: vec![],
                    num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                    k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                });
            }
            6 => {
                // gelu: scale (i32), n_gelu (u32), then n_gelu (input_i16, output_i16) pairs
                let gelu_scale = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;
                let n_gelu = read_u32(&mut pos) as usize;
                let mut gelu_input_f = Vec::with_capacity(n_gelu);
                let mut gelu_output_f = Vec::with_capacity(n_gelu);
                for _ in 0..n_gelu {
                    let inp = i16::from_le_bytes(data[pos..pos+2].try_into().unwrap());
                    pos += 2;
                    let out = i16::from_le_bytes(data[pos..pos+2].try_into().unwrap());
                    pos += 2;
                    gelu_input_f.push(i16_to_field(inp).as_canonical_u32());
                    gelu_output_f.push(i16_to_field(out).as_canonical_u32());
                }
                ops.push(OpDesc {
                    op_type: "gelu".into(), name,
                    m: 0, n: 0,
                    w_q: gelu_input_f, b_q: gelu_output_f,
                    gelu_scale,
                    gamma: vec![], beta: vec![],
                    save_name: String::new(), add_name: String::new(),
                    new_input: vec![], ln_output: vec![],
                    num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                    k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                });
            }
            7 => {
                // attention: num_heads, seq_len, d_head, exp_scale, n_kv, k_values, v_values
                let num_heads = read_u32(&mut pos) as usize;
                let seq_len = read_u32(&mut pos) as usize;
                let d_head = read_u32(&mut pos) as usize;
                let exp_scale = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;
                let n_kv = read_u32(&mut pos) as usize;
                let mut k_values = Vec::with_capacity(n_kv);
                for _ in 0..n_kv {
                    k_values.push(read_u32(&mut pos));
                }
                let mut v_values = Vec::with_capacity(n_kv);
                for _ in 0..n_kv {
                    v_values.push(read_u32(&mut pos));
                }
                ops.push(OpDesc {
                    op_type: "attention".into(), name,
                    m: 0, n: 0, w_q: vec![], b_q: vec![],
                    gelu_scale: default_gelu_scale(),
                    gamma: vec![], beta: vec![],
                    save_name: String::new(), add_name: String::new(),
                    new_input: vec![], ln_output: vec![],
                    num_heads, seq_len, d_head, exp_scale,
                    k_values, v_values,
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                });
            }
            8 => {
                // rmsnorm: gamma_len, gamma[], output_len, output[]
                let gamma_len = read_u32(&mut pos) as usize;
                let mut gamma = Vec::with_capacity(gamma_len);
                for _ in 0..gamma_len {
                    gamma.push(read_u32(&mut pos));
                }
                let out_len = read_u32(&mut pos) as usize;
                let mut rmsnorm_output = Vec::with_capacity(out_len);
                for _ in 0..out_len {
                    rmsnorm_output.push(read_u32(&mut pos));
                }
                ops.push(OpDesc {
                    op_type: "rmsnorm".into(), name,
                    m: 0, n: 0, w_q: vec![], b_q: vec![],
                    gelu_scale: default_gelu_scale(),
                    gamma, beta: vec![],
                    save_name: String::new(), add_name: String::new(),
                    new_input: vec![], ln_output: rmsnorm_output,
                    num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                    k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                });
            }
            9 => {
                // silu: silu_scale (i32), n_entries (u32), then n_entries (input_i16, output_i16) pairs
                let silu_scale = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;
                let n_silu = read_u32(&mut pos) as usize;
                let mut silu_input_f = Vec::with_capacity(n_silu);
                let mut silu_output_f = Vec::with_capacity(n_silu);
                for _ in 0..n_silu {
                    let inp = i16::from_le_bytes(data[pos..pos+2].try_into().unwrap());
                    pos += 2;
                    let out = i16::from_le_bytes(data[pos..pos+2].try_into().unwrap());
                    pos += 2;
                    silu_input_f.push(i16_to_field(inp).as_canonical_u32());
                    silu_output_f.push(i16_to_field(out).as_canonical_u32());
                }
                ops.push(OpDesc {
                    op_type: "silu".into(), name,
                    m: 0, n: 0,
                    w_q: silu_input_f, b_q: silu_output_f,
                    gelu_scale: silu_scale,
                    gamma: vec![], beta: vec![],
                    save_name: String::new(), add_name: String::new(),
                    new_input: vec![], ln_output: vec![],
                    num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                    k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                });
            }
            10 => {
                // swiglu: silu_scale (i32), n (u32), gate_i16[n], gate_silu_i16[n], up_values[n] (u32)
                let silu_scale = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;
                let n = read_u32(&mut pos) as usize;
                let mut gate_f = Vec::with_capacity(n);
                let mut gate_silu_f = Vec::with_capacity(n);
                for _ in 0..n {
                    gate_f.push(i16_to_field(
                        i16::from_le_bytes(data[pos..pos+2].try_into().unwrap())
                    ).as_canonical_u32());
                    pos += 2;
                }
                for _ in 0..n {
                    gate_silu_f.push(i16_to_field(
                        i16::from_le_bytes(data[pos..pos+2].try_into().unwrap())
                    ).as_canonical_u32());
                    pos += 2;
                }
                let mut up_values = Vec::with_capacity(n);
                for _ in 0..n {
                    up_values.push(read_u32(&mut pos));
                }
                ops.push(OpDesc {
                    op_type: "swiglu".into(), name,
                    m: 0, n: 0,
                    w_q: gate_f, b_q: gate_silu_f,
                    gelu_scale: silu_scale,
                    gamma: vec![], beta: vec![],
                    save_name: String::new(), add_name: String::new(),
                    new_input: vec![], ln_output: up_values,
                    num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                    k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                });
            }
            11 => {
                // gqa_attention: num_q_heads, num_kv_heads, seq_len, d_head, exp_scale, n_kv, k_values, v_values
                let num_q_heads = read_u32(&mut pos) as usize;
                let num_kv_heads = read_u32(&mut pos) as usize;
                let seq_len = read_u32(&mut pos) as usize;
                let d_head = read_u32(&mut pos) as usize;
                let exp_scale = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;
                let n_kv = read_u32(&mut pos) as usize;
                let mut k_values = Vec::with_capacity(n_kv);
                for _ in 0..n_kv { k_values.push(read_u32(&mut pos)); }
                let mut v_values = Vec::with_capacity(n_kv);
                for _ in 0..n_kv { v_values.push(read_u32(&mut pos)); }
                ops.push(OpDesc {
                    op_type: "gqa_attention".into(), name,
                    m: num_q_heads, n: num_kv_heads,
                    w_q: vec![], b_q: vec![],
                    gelu_scale: default_gelu_scale(),
                    gamma: vec![], beta: vec![],
                    save_name: String::new(), add_name: String::new(),
                    new_input: vec![], ln_output: vec![],
                    num_heads: num_q_heads, seq_len, d_head, exp_scale,
                    k_values, v_values,
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                });
            }
            12 => {
                // llama_layer: config + all 9 weight matrices
                let d_model = read_u32(&mut pos) as usize;
                let d_ff = read_u32(&mut pos) as usize;
                let num_q_heads = read_u32(&mut pos) as usize;
                let num_kv_heads = read_u32(&mut pos) as usize;
                let d_head = read_u32(&mut pos) as usize;
                let silu_scale = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;

                let q_dim = num_q_heads * d_head;
                let kv_dim = num_kv_heads * d_head;

                // Bulk read: reinterpret byte slice as u32 slice (little-endian, aligned)
                // Zero-copy read: Mersenne31 is repr(transparent) over u32,
                // and data is little-endian (matching native ARM byte order).
                let read_fields = |pos: &mut usize, count: usize| -> Vec<F> {
                    let byte_len = count * 4;
                    let slice = &data[*pos..*pos + byte_len];
                    *pos += byte_len;
                    let mut result = Vec::<F>::with_capacity(count);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            slice.as_ptr(),
                            result.as_mut_ptr() as *mut u8,
                            byte_len,
                        );
                        result.set_len(count);
                    }
                    result
                };

                let norm1_gamma = read_fields(&mut pos, d_model);
                let w_q = read_fields(&mut pos, q_dim * d_model);
                let w_k = read_fields(&mut pos, kv_dim * d_model);
                let w_v = read_fields(&mut pos, kv_dim * d_model);
                let w_o = read_fields(&mut pos, d_model * q_dim);
                let norm2_gamma = read_fields(&mut pos, d_model);
                let w_gate = read_fields(&mut pos, d_ff * d_model);
                let w_up = read_fields(&mut pos, d_ff * d_model);
                let w_down = read_fields(&mut pos, d_model * d_ff);

                ops.push(OpDesc {
                    op_type: "llama_layer".into(), name,
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
                    llama_weights: Some(LlamaLayerWeightData {
                        norm1_gamma, w_q, w_k, w_v, w_o,
                        norm2_gamma, w_gate, w_up, w_down,
                    }),
                    qwen_config: None, qwen_weights: None,
                });
            }
            14 => {
                // qwen_layer: config + all 10 weight matrices (includes g_proj + sigmoid_scale)
                let d_model = read_u32(&mut pos) as usize;
                let d_ff = read_u32(&mut pos) as usize;
                let num_q_heads = read_u32(&mut pos) as usize;
                let num_kv_heads = read_u32(&mut pos) as usize;
                let d_head = read_u32(&mut pos) as usize;
                let silu_scale = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;
                let sigmoid_scale = i32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
                pos += 4;

                let q_dim = num_q_heads * d_head;
                let kv_dim = num_kv_heads * d_head;

                let read_fields = |pos: &mut usize, count: usize| -> Vec<F> {
                    let byte_len = count * 4;
                    let slice = &data[*pos..*pos + byte_len];
                    *pos += byte_len;
                    let mut result = Vec::<F>::with_capacity(count);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            slice.as_ptr(),
                            result.as_mut_ptr() as *mut u8,
                            byte_len,
                        );
                        result.set_len(count);
                    }
                    result
                };

                let norm1_gamma = read_fields(&mut pos, d_model);
                let w_q = read_fields(&mut pos, q_dim * d_model);
                let w_k = read_fields(&mut pos, kv_dim * d_model);
                let w_v = read_fields(&mut pos, kv_dim * d_model);
                let w_o = read_fields(&mut pos, d_model * q_dim);
                let w_g_proj = read_fields(&mut pos, q_dim * d_model);
                let norm2_gamma = read_fields(&mut pos, d_model);
                let w_gate = read_fields(&mut pos, d_ff * d_model);
                let w_up = read_fields(&mut pos, d_ff * d_model);
                let w_down = read_fields(&mut pos, d_model * d_ff);

                ops.push(OpDesc {
                    op_type: "qwen_layer".into(), name,
                    m: 0, n: 0, w_q: vec![], b_q: vec![],
                    gelu_scale: silu_scale,
                    gamma: vec![], beta: vec![],
                    save_name: String::new(), add_name: String::new(),
                    new_input: vec![], ln_output: vec![],
                    num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                    k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: Some(QwenLayerConfig {
                        d_model, d_ff, num_q_heads, num_kv_heads, d_head,
                        silu_scale, sigmoid_scale,
                    }),
                    qwen_weights: Some(QwenLayerWeightData {
                        norm1_gamma, w_q, w_k, w_v, w_o, w_g_proj,
                        norm2_gamma, w_gate, w_up, w_down,
                    }),
                });
            }
            _ => panic!("Unknown binary op_type byte: {}", op_type_byte),
        }
    }

    ProveRequest {
        mode: "mlp".into(),
        input,
        ops,
        gpt2: None,
    }
}
