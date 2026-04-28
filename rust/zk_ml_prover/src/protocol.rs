// ===== Phase 10 P10-8 (E7 closure): OpDesc → tagged-enum at the deserialization edge =====
//
// WHY: `OpDesc` was a 30-field god-struct with `#[serde(default)]` on every
// payload field. That made every variant superficially valid: a hostile or
// mistyped JSON like `{"type":"layer_norm","name":"x"}` (missing gamma/beta)
// silently parsed as zero-default vectors and only blew up later inside the
// dispatcher with an unrelated panic, miles from the actual bug. Adding a
// new op_type was equally unsafe — the compiler could not tell you which
// fields the dispatcher would read.
//
// WHAT CHANGED: a tagged-union `Op` enum is the canonical wire-shape contract
// for the JSON `ops:` array. Each variant carries exactly the fields the
// runtime reads, so missing-field bugs surface at parse time. `OpDesc` keeps
// its existing flat shape for now (compat with ~30 struct-literal call sites
// in pipeline/forward.rs, server.rs, and tests). A custom `Deserialize` impl
// on `OpDesc` parses through `Op` first and then unconditionally widens to
// the flat representation — so the wire JSON contract is unchanged but
// invalid payloads are rejected early.
//
// WIRE FORMAT (UNCHANGED): `{"type": "<snake_case>", "name": "...", ...}`
// where `<snake_case>` is the kebab/snake-case form of the variant name
// (`linear`, `relu`, `set_input`, `layer_norm`, `add_saved`, `gelu`,
// `attention`, `rms_norm`, `silu`, `swiglu`, `gqa_attention`, `llama_layer`,
// `qwen_layer`, `passthrough`). Existing model traces deserialize unchanged
// — this is verified by the `op_wire_format_pin` regression test.
//
// DEFERRED: dispatch sites in `pipeline/forward.rs`, `pipeline/prove.rs`,
// `verification.rs`, `server.rs`, and `pipeline/mod.rs` still match on
// `OpDesc.op_type: String`. Migrating those to match on `Op` directly is
// follow-up work — the win above (compile-time variant safety at the parse
// boundary) lands now without touching any of the proof logic.

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

// `OpDesc` no longer derives `Deserialize` directly — see `impl Deserialize`
// below, which routes through the tagged-union `Op` enum.
//
// INVARIANT: at the JSON parse boundary every field is populated by the
// `Op → OpDesc` lowering with a known-correct value (a real payload for
// the variant, or a typed-zero/empty for fields the variant doesn't carry).
// Binary-protocol callers (`parse_binary_checked`) and in-process callers
// (`server.rs`, test helpers) still build `OpDesc` via struct literal and
// have always been responsible for populating all fields.
pub(crate) struct OpDesc {
    pub(crate) op_type: String,
    pub(crate) name: String,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) w_q: Vec<u32>,
    pub(crate) b_q: Vec<u32>,
    // --- GELU fields ---
    pub(crate) gelu_scale: i32,
    // --- LayerNorm fields ---
    pub(crate) gamma: Vec<u32>,
    pub(crate) beta: Vec<u32>,
    // --- save / add_saved fields ---
    pub(crate) save_name: String,
    pub(crate) add_name: String,
    // --- set_input fields ---
    pub(crate) new_input: Vec<u32>,
    // --- Pre-computed LN output (from Python float computation) ---
    #[allow(dead_code)]
    pub(crate) ln_output: Vec<u32>,
    // --- Attention fields ---
    pub(crate) num_heads: usize,
    pub(crate) seq_len: usize,
    pub(crate) d_head: usize,
    pub(crate) exp_scale: i32,
    pub(crate) k_values: Vec<u32>,
    pub(crate) v_values: Vec<u32>,
    // --- Llama layer fields ---
    pub(crate) llama_config: Option<LlamaLayerConfig>,
    pub(crate) llama_weights: Option<LlamaLayerWeightData>,
    // --- Qwen layer fields ---
    pub(crate) qwen_config: Option<QwenLayerConfig>,
    pub(crate) qwen_weights: Option<QwenLayerWeightData>,
}

/// Tagged-union shape of the JSON `ops:` array. The discriminator is the
/// `"type"` field; payload fields per variant exactly mirror what each
/// dispatcher reads downstream. Unknown discriminators or missing required
/// payload fields fail fast at parse time instead of zero-defaulting.
///
/// SAFETY (E7): variant rename strings MUST exactly match the discriminator
/// strings emitted by the Python compiler / produced by `parse_binary_checked`
/// (search for `"linear".into()` etc.). Any divergence here breaks the wire
/// contract and is caught by `op_wire_format_pin` below — that test pins
/// every variant's tag string, so renaming a variant without updating
/// the test (and the corresponding sender) won't compile-pass silently.
#[derive(Deserialize)]
#[serde(tag = "type")]
pub(crate) enum Op {
    #[serde(rename = "linear")]
    Linear {
        name: String,
        m: usize,
        n: usize,
        w_q: Vec<u32>,
        b_q: Vec<u32>,
    },
    #[serde(rename = "relu")]
    Relu { name: String },
    #[serde(rename = "set_input")]
    SetInput { name: String, new_input: Vec<u32> },
    /// Wire discriminator is the historical `"layernorm"` (one word). The
    /// task spec mentioned `"layer_norm"`, but the actual senders (Python
    /// compiler, `parse_binary_checked`) emit the no-underscore form, so
    /// that's what we honor.
    #[serde(rename = "layernorm")]
    LayerNorm {
        name: String,
        gamma: Vec<u32>,
        beta: Vec<u32>,
        /// Pre-computed LN output (Python float compute). Optional — older
        /// traces omit it and the prover recomputes via QR perturbation.
        #[serde(default)]
        ln_output: Vec<u32>,
    },
    #[serde(rename = "save")]
    Save { name: String, save_name: String },
    #[serde(rename = "add_saved")]
    AddSaved { name: String, add_name: String },
    #[serde(rename = "gelu")]
    Gelu {
        name: String,
        /// Pre-quantized inputs/outputs in the historical `w_q`/`b_q` slots
        /// — same shape on the wire as the binary parser produces.
        #[serde(default)]
        w_q: Vec<u32>,
        #[serde(default)]
        b_q: Vec<u32>,
        #[serde(default = "default_gelu_scale")]
        gelu_scale: i32,
    },
    #[serde(rename = "attention")]
    Attention {
        name: String,
        num_heads: usize,
        seq_len: usize,
        d_head: usize,
        exp_scale: i32,
        k_values: Vec<u32>,
        v_values: Vec<u32>,
    },
    /// Historical no-underscore form, matches `parse_binary_checked`.
    #[serde(rename = "rmsnorm")]
    RmsNorm {
        name: String,
        gamma: Vec<u32>,
        /// Pre-computed RMSNorm output (Python float compute), reused by
        /// the prover. Stored in the legacy `ln_output` slot in OpDesc.
        #[serde(default)]
        ln_output: Vec<u32>,
    },
    #[serde(rename = "silu")]
    Silu {
        name: String,
        #[serde(default)]
        w_q: Vec<u32>,
        #[serde(default)]
        b_q: Vec<u32>,
        /// Reuses the `gelu_scale` slot in `OpDesc` (single-scalar field
        /// shared between GELU and SiLU lookup variants).
        #[serde(default = "default_gelu_scale")]
        gelu_scale: i32,
    },
    #[serde(rename = "swiglu")]
    SwiGlu {
        name: String,
        /// gate values (i16-as-u32) in legacy w_q slot.
        #[serde(default)]
        w_q: Vec<u32>,
        /// gate-after-silu values in legacy b_q slot.
        #[serde(default)]
        b_q: Vec<u32>,
        /// up-projection values in legacy ln_output slot.
        #[serde(default)]
        ln_output: Vec<u32>,
        #[serde(default = "default_gelu_scale")]
        gelu_scale: i32,
    },
    /// Grouped-query attention. Wire packs `num_kv_heads` in the legacy
    /// `n` slot (see `parse_binary_checked` byte-12); we preserve that
    /// quirk so dispatchers don't need touching.
    #[serde(rename = "gqa_attention")]
    GqaAttention {
        name: String,
        #[serde(default)]
        m: usize,
        #[serde(default)]
        n: usize,
        num_heads: usize,
        seq_len: usize,
        d_head: usize,
        exp_scale: i32,
        k_values: Vec<u32>,
        v_values: Vec<u32>,
    },
    #[serde(rename = "llama_layer")]
    LlamaLayer {
        name: String,
        llama_config: LlamaLayerConfig,
        llama_weights: LlamaLayerWeightData,
        /// SiLU scale travels in the shared `gelu_scale` field for parity
        /// with binary parsing. Variant default keeps wire-compat with
        /// older traces that didn't include it.
        #[serde(default = "default_gelu_scale")]
        gelu_scale: i32,
    },
    #[serde(rename = "qwen_layer")]
    QwenLayer {
        name: String,
        qwen_config: QwenLayerConfig,
        qwen_weights: QwenLayerWeightData,
        #[serde(default = "default_gelu_scale")]
        gelu_scale: i32,
    },
    /// Internal-only marker emitted by the prover/verifier coverage logic.
    /// It is NOT produced by any current sender, but the verifier matches
    /// on it (`verification.rs:187`) so JSON traces could in principle
    /// contain it. Carrying just `name` keeps round-trip safe.
    #[serde(rename = "passthrough")]
    PassThrough { name: String },
}

impl Op {
    /// Lower a parsed variant into the legacy flat `OpDesc` shape. This is
    /// the conversion shim that lets us land compile-time variant safety at
    /// the parse boundary without touching any of the ~30 dispatch sites
    /// that still match on `OpDesc.op_type: String`.
    ///
    /// INVARIANT: the produced `OpDesc.op_type` string here MUST match
    /// what those dispatchers compare against — that's the same string the
    /// binary parser uses, so we keep them synchronized in one spot.
    pub(crate) fn into_op_desc(self) -> OpDesc {
        // Helper for the "all the empty defaults" ceremony — every variant
        // overwrites the few slots it actually populates and the rest stay
        // as typed-zero / empty-vec. Centralizing avoids drift if a new
        // field is added to OpDesc.
        fn blank(name: String, op_type: &str) -> OpDesc {
            OpDesc {
                op_type: op_type.to_string(),
                name,
                m: 0,
                n: 0,
                w_q: Vec::new(),
                b_q: Vec::new(),
                gelu_scale: default_gelu_scale(),
                gamma: Vec::new(),
                beta: Vec::new(),
                save_name: String::new(),
                add_name: String::new(),
                new_input: Vec::new(),
                ln_output: Vec::new(),
                num_heads: 0,
                seq_len: 0,
                d_head: 0,
                exp_scale: 0,
                k_values: Vec::new(),
                v_values: Vec::new(),
                llama_config: None,
                llama_weights: None,
                qwen_config: None,
                qwen_weights: None,
            }
        }

        match self {
            Op::Linear { name, m, n, w_q, b_q } => {
                let mut o = blank(name, "linear");
                o.m = m; o.n = n; o.w_q = w_q; o.b_q = b_q;
                o
            }
            Op::Relu { name } => blank(name, "relu"),
            Op::SetInput { name, new_input } => {
                let mut o = blank(name, "set_input");
                o.new_input = new_input;
                o
            }
            Op::LayerNorm { name, gamma, beta, ln_output } => {
                let mut o = blank(name, "layernorm");
                o.gamma = gamma; o.beta = beta; o.ln_output = ln_output;
                o
            }
            Op::Save { name, save_name } => {
                let mut o = blank(name, "save");
                o.save_name = save_name;
                o
            }
            Op::AddSaved { name, add_name } => {
                let mut o = blank(name, "add_saved");
                o.add_name = add_name;
                o
            }
            Op::Gelu { name, w_q, b_q, gelu_scale } => {
                let mut o = blank(name, "gelu");
                o.w_q = w_q; o.b_q = b_q; o.gelu_scale = gelu_scale;
                o
            }
            Op::Attention { name, num_heads, seq_len, d_head, exp_scale, k_values, v_values } => {
                let mut o = blank(name, "attention");
                o.num_heads = num_heads;
                o.seq_len = seq_len;
                o.d_head = d_head;
                o.exp_scale = exp_scale;
                o.k_values = k_values;
                o.v_values = v_values;
                o
            }
            Op::RmsNorm { name, gamma, ln_output } => {
                let mut o = blank(name, "rmsnorm");
                o.gamma = gamma;
                o.ln_output = ln_output;
                o
            }
            Op::Silu { name, w_q, b_q, gelu_scale } => {
                let mut o = blank(name, "silu");
                o.w_q = w_q; o.b_q = b_q; o.gelu_scale = gelu_scale;
                o
            }
            Op::SwiGlu { name, w_q, b_q, ln_output, gelu_scale } => {
                let mut o = blank(name, "swiglu");
                o.w_q = w_q; o.b_q = b_q; o.ln_output = ln_output; o.gelu_scale = gelu_scale;
                o
            }
            Op::GqaAttention { name, m, n, num_heads, seq_len, d_head, exp_scale, k_values, v_values } => {
                let mut o = blank(name, "gqa_attention");
                // INVARIANT (binary parity): `num_kv_heads` is packed into the
                // shared `n` slot — see `parse_binary_checked` byte-12 emitter.
                // JSON callers that supply `n` directly land in the same slot.
                o.m = m;
                o.n = n;
                o.num_heads = num_heads;
                o.seq_len = seq_len;
                o.d_head = d_head;
                o.exp_scale = exp_scale;
                o.k_values = k_values;
                o.v_values = v_values;
                o
            }
            Op::LlamaLayer { name, llama_config, llama_weights, gelu_scale } => {
                let mut o = blank(name, "llama_layer");
                o.gelu_scale = gelu_scale;
                o.llama_config = Some(llama_config);
                o.llama_weights = Some(llama_weights);
                o
            }
            Op::QwenLayer { name, qwen_config, qwen_weights, gelu_scale } => {
                let mut o = blank(name, "qwen_layer");
                o.gelu_scale = gelu_scale;
                o.qwen_config = Some(qwen_config);
                o.qwen_weights = Some(qwen_weights);
                o
            }
            Op::PassThrough { name } => blank(name, "passthrough"),
        }
    }
}

impl<'de> Deserialize<'de> for OpDesc {
    /// SAFETY (E7): JSON deserialization of an op MUST go through the
    /// `Op` enum so that unknown discriminators and missing required
    /// payload fields surface as parse errors rather than zero-defaulting
    /// into a footgun for the dispatcher. The shim then unconditionally
    /// widens to the legacy flat shape.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Op::deserialize(deserializer).map(Op::into_op_desc)
    }
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
    /// V head count (asymmetric GDN). 0 = fall back to num_kv_heads.
    #[serde(default)]
    pub(crate) v_num_heads: usize,
    /// V head dim. 0 = fall back to d_head.
    #[serde(default)]
    pub(crate) v_d_head: usize,
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

/// Compile-time invariants required for the zero-copy transmute below.
///
/// SAFETY (E1): upstream `p3_mersenne_31` does NOT publicly guarantee
/// `#[repr(transparent)]` on `Mersenne31`. If a future version adds a tag byte
/// or changes the layout, our `Vec::from_raw_parts` cast becomes UB. Rather
/// than ride that risk silently, these assertions force a compile-time abort
/// if `size_of::<Mersenne31>() != size_of::<u32>()` or alignments diverge.
/// Bumping the dependency in Cargo.toml is then the only way to break the
/// build, and reviewers can decide whether to keep the optimization.
const _ASSERT_M31_SIZE_MATCHES_U32: () =
    assert!(std::mem::size_of::<Mersenne31>() == std::mem::size_of::<u32>());
const _ASSERT_M31_ALIGN_MATCHES_U32: () =
    assert!(std::mem::align_of::<Mersenne31>() == std::mem::align_of::<u32>());

/// Zero-copy transmute Vec<u32> → Vec<F> (Mersenne31).
///
/// SAFETY (E1):
///   - size and alignment of `Mersenne31` and `u32` are equal (compile-time
///     asserted by `_ASSERT_M31_SIZE_MATCHES_U32` / `_ALIGN_`).
///   - `Mersenne31` is currently `#[repr(transparent)]` around `u32` in
///     `p3_mersenne_31`. We do NOT take that as a stable contract; the
///     compile-time size/align checks are the only structural guarantees we
///     rely on. If those hold, reinterpreting the buffer is sound.
///
/// INVARIANT: callers must ensure every `u32` value is canonical
/// (`< Mersenne31::ORDER == 2^31 - 1`). Non-canonical values would still
/// pass type-checks but break field arithmetic (Mersenne31 expects values
/// in `[0, p)`). In debug builds we sample-check the first few entries.
pub(crate) fn transmute_u32_to_f(v: Vec<u32>) -> Vec<F> {
    // Force the compile-time asserts to be evaluated: a `let _ = CONST;`
    // ensures rustc emits the diagnostic at this site if the const fails.
    let () = _ASSERT_M31_SIZE_MATCHES_U32;
    let () = _ASSERT_M31_ALIGN_MATCHES_U32;
    debug_assert!(
        v.iter().take(8).all(|&u| u < (1u32 << 31) - 1),
        "transmute_u32_to_f: non-canonical value found in first 8 entries"
    );
    let mut v = std::mem::ManuallyDrop::new(v);
    let ptr = v.as_mut_ptr() as *mut F;
    let len = v.len();
    let cap = v.capacity();
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

/// Bulk-copy `count` little-endian u32s from a byte slice directly into a
/// `Vec<F>` of length `count`. Requires `slice.len() == count * 4`.
///
/// SAFETY (E1): same compile-time invariants as `transmute_u32_to_f`. We
/// `copy_nonoverlapping` `count * 4` bytes into a fresh `Vec<F>` whose
/// allocation is `count * size_of::<F>() = count * 4`. Because size and
/// alignment of `F` and `u32` are equal (compile-time asserted), the copy
/// is byte-identical to a u32-by-u32 read followed by `From<u32>`. On
/// little-endian targets (ARM, x86) the field elements end up in canonical
/// in-memory form. On big-endian targets the byte order would mismatch,
/// but Plonky3 / our targets are LE-only — debug builds catch this via
/// the `cfg(target_endian)` check.
pub(crate) fn read_fields_zero_copy(slice: &[u8], count: usize) -> Vec<F> {
    debug_assert_eq!(slice.len(), count * 4,
        "read_fields_zero_copy: slice length must equal count * 4");
    // SAFETY: explicit endianness gate — fail fast on big-endian rather than
    // silently produce wrong field elements. (No supported target is BE today.)
    #[cfg(target_endian = "big")]
    compile_error!("read_fields_zero_copy assumes little-endian; current target is BE");

    let mut result = Vec::<F>::with_capacity(count);
    let byte_len = count * 4; // overflow checked by callers via checked_count
    unsafe {
        std::ptr::copy_nonoverlapping(
            slice.as_ptr(),
            result.as_mut_ptr() as *mut u8,
            byte_len,
        );
        result.set_len(count);
    }
    result
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

// ===== Binary parse error type (E2 hardening) =====

/// Error returned by `parse_binary` for malformed / hostile input.
///
/// SOUNDNESS / SAFETY (E2): the binary parser is reachable from untrusted senders
/// (server.rs, pipeline/mod.rs reading stdin). Previous version panicked on bad
/// input via `unwrap`/`expect`/`data[pos..pos+N]` indexing. A malicious client
/// could DoS the prover with truncated frames, oversized dims, or `m*n` overflow.
/// This Result-returning variant is the canonical entry point; callers must
/// propagate the error rather than unwrap.
#[derive(Debug, Clone)]
pub(crate) enum BinaryParseError {
    Truncated { needed: usize, remaining: usize, ctx: &'static str },
    InvalidUtf8(&'static str),
    DimensionOverflow { ctx: &'static str, a: usize, b: usize },
    DimensionTooLarge { ctx: &'static str, value: usize, max: usize },
    OpCountTooLarge { count: u32, max: u32 },
    UnknownOpType(u8),
}

impl std::fmt::Display for BinaryParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated { needed, remaining, ctx } =>
                write!(f, "binary frame truncated at {ctx}: need {needed} bytes, have {remaining}"),
            Self::InvalidUtf8(ctx) => write!(f, "invalid UTF-8 in {ctx}"),
            Self::DimensionOverflow { ctx, a, b } =>
                write!(f, "dimension overflow at {ctx}: {a} * {b} overflows usize"),
            Self::DimensionTooLarge { ctx, value, max } =>
                write!(f, "dimension at {ctx} ({value}) exceeds max {max}"),
            Self::OpCountTooLarge { count, max } =>
                write!(f, "op count {count} exceeds max {max}"),
            Self::UnknownOpType(b) => write!(f, "unknown binary op_type byte: {b}"),
        }
    }
}
impl std::error::Error for BinaryParseError {}

/// SAFETY: caps on per-op dimension fields. Conservative — accommodates Qwen 9B
/// (`d_model=4096`, `d_ff=14336`, weight matrices ~58M elements) plus headroom.
/// Anything larger is almost certainly hostile or a bug; better to fail fast.
const MAX_DIM: usize = 1 << 24;            // 16,777,216 — covers all production model sizes
const MAX_OPS: u32 = 1 << 20;              // 1,048,576 — far above any real graph
const MAX_ELEMENT_COUNT: usize = 1 << 30;  // 1,073,741,824 = 4 GiB at 4 B/elem (matmul weights)

#[inline]
fn check_remaining(data: &[u8], pos: usize, needed: usize, ctx: &'static str)
    -> Result<(), BinaryParseError>
{
    let remaining = data.len().saturating_sub(pos);
    if remaining < needed {
        return Err(BinaryParseError::Truncated { needed, remaining, ctx });
    }
    Ok(())
}

#[inline]
fn read_u32_le(data: &[u8], pos: &mut usize, ctx: &'static str)
    -> Result<u32, BinaryParseError>
{
    check_remaining(data, *pos, 4, ctx)?;
    let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

#[inline]
fn read_i32_le(data: &[u8], pos: &mut usize, ctx: &'static str)
    -> Result<i32, BinaryParseError>
{
    check_remaining(data, *pos, 4, ctx)?;
    let v = i32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

#[inline]
fn read_i16_le(data: &[u8], pos: &mut usize, ctx: &'static str)
    -> Result<i16, BinaryParseError>
{
    check_remaining(data, *pos, 2, ctx)?;
    let v = i16::from_le_bytes(data[*pos..*pos + 2].try_into().unwrap());
    *pos += 2;
    Ok(v)
}

#[inline]
fn read_dim(data: &[u8], pos: &mut usize, ctx: &'static str)
    -> Result<usize, BinaryParseError>
{
    let v = read_u32_le(data, pos, ctx)? as usize;
    if v > MAX_DIM {
        return Err(BinaryParseError::DimensionTooLarge { ctx, value: v, max: MAX_DIM });
    }
    Ok(v)
}

/// SAFETY: usize-checked product for element counts, with absolute cap.
/// Prevents `m * n` from wrapping (hostile dims) or producing a Vec capacity
/// that exceeds available memory.
#[inline]
fn checked_count(a: usize, b: usize, ctx: &'static str)
    -> Result<usize, BinaryParseError>
{
    let prod = a.checked_mul(b)
        .ok_or(BinaryParseError::DimensionOverflow { ctx, a, b })?;
    if prod > MAX_ELEMENT_COUNT {
        return Err(BinaryParseError::DimensionTooLarge { ctx, value: prod, max: MAX_ELEMENT_COUNT });
    }
    Ok(prod)
}

#[inline]
fn read_string(data: &[u8], pos: &mut usize, ctx: &'static str)
    -> Result<String, BinaryParseError>
{
    let len = read_dim(data, pos, ctx)?;
    check_remaining(data, *pos, len, ctx)?;
    let s = String::from_utf8(data[*pos..*pos + len].to_vec())
        .map_err(|_| BinaryParseError::InvalidUtf8(ctx))?;
    *pos += len;
    Ok(s)
}

#[inline]
fn read_u32_vec(data: &[u8], pos: &mut usize, len: usize, ctx: &'static str)
    -> Result<Vec<u32>, BinaryParseError>
{
    let bytes = checked_count(len, 4, ctx)?;
    check_remaining(data, *pos, bytes, ctx)?;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()));
        *pos += 4;
    }
    Ok(out)
}

/// SAFETY (E3a): Result-returning variant for the server preload loop.
/// IO errors during preload (truncated weights, disconnect) propagate as
/// `io::Error` instead of panicking the prover. Callers in production paths
/// should prefer this over `read_u32_from`. Generic over any `Read` so it can
/// be exercised in unit tests with a `Cursor<&[u8]>`.
pub(crate) fn read_u32_from_checked<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read `len` bytes into a Vec, returning an io::Error on truncation.
pub(crate) fn read_exact_vec<R: Read>(reader: &mut R, len: usize)
    -> io::Result<Vec<u8>>
{
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(buf)
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
/// SAFETY (E2): all field reads are bounds-checked, all multiplications use
/// `checked_mul`, and dimension fields are clamped to `MAX_DIM` /
/// `MAX_ELEMENT_COUNT` to prevent OOM via hostile inputs.
///
/// Format:
///   [u32 LE] num_ops
///   [u32 LE] input_len
///   [u32 LE * input_len] input values
///   For each op:
///     [u8] op_type: 0=linear, 1=relu, 2=set_input, 3=layernorm, 4=save, 5=add_saved, 6=gelu, …
///     [u32 LE] name_len
///     [u8 * name_len] name bytes
///     If linear: [u32 LE] m, [u32 LE] n, [u32 LE * m*n] w_q, [u32 LE * m] b_q
///     If relu: (nothing)
///     If set_input: [u32 LE] new_input_len, [u32 LE * new_input_len] values
pub(crate) fn parse_binary(data: &[u8]) -> ProveRequest {
    parse_binary_checked(data)
        .unwrap_or_else(|e| panic!("parse_binary: {}", e))
}

/// Result-returning variant. Production callers should prefer this; the
/// `parse_binary` wrapper exists only for legacy tests / pipeline paths.
pub(crate) fn parse_binary_checked(data: &[u8]) -> Result<ProveRequest, BinaryParseError> {
    let mut pos = 0;

    let num_ops_raw = read_u32_le(data, &mut pos, "num_ops")?;
    if num_ops_raw > MAX_OPS {
        return Err(BinaryParseError::OpCountTooLarge { count: num_ops_raw, max: MAX_OPS });
    }
    let num_ops = num_ops_raw;

    let input_len = read_dim(data, &mut pos, "input_len")?;
    let input = read_u32_vec(data, &mut pos, input_len, "input")?;

    let mut ops = Vec::with_capacity(num_ops as usize);
    for _ in 0..num_ops {
        check_remaining(data, pos, 1, "op_type")?;
        let op_type_byte = data[pos];
        pos += 1;

        let name = read_string(data, &mut pos, "op name")?;

        match op_type_byte {
            0 => {
                // linear
                let m = read_dim(data, &mut pos, "linear.m")?;
                let n = read_dim(data, &mut pos, "linear.n")?;
                let mn = checked_count(m, n, "linear.m*n")?;
                let w_q = read_u32_vec(data, &mut pos, mn, "linear.w_q")?;
                let b_q = read_u32_vec(data, &mut pos, m, "linear.b_q")?;
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
                let new_input_len = read_dim(data, &mut pos, "set_input.len")?;
                let new_input = read_u32_vec(data, &mut pos, new_input_len, "set_input")?;
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
                let gamma_len = read_dim(data, &mut pos, "layernorm.gamma_len")?;
                let gamma = read_u32_vec(data, &mut pos, gamma_len, "layernorm.gamma")?;
                let beta_len = read_dim(data, &mut pos, "layernorm.beta_len")?;
                let beta = read_u32_vec(data, &mut pos, beta_len, "layernorm.beta")?;
                let ln_out_len = read_dim(data, &mut pos, "layernorm.out_len")?;
                let ln_output = read_u32_vec(data, &mut pos, ln_out_len, "layernorm.output")?;
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
                let save_name = read_string(data, &mut pos, "save_name")?;
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
                let add_name = read_string(data, &mut pos, "add_name")?;
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
                let gelu_scale = read_i32_le(data, &mut pos, "gelu.scale")?;
                let n_gelu = read_dim(data, &mut pos, "gelu.n")?;
                let pair_bytes = checked_count(n_gelu, 4, "gelu.pair_bytes")?;
                check_remaining(data, pos, pair_bytes, "gelu.pairs")?;
                let mut gelu_input_f = Vec::with_capacity(n_gelu);
                let mut gelu_output_f = Vec::with_capacity(n_gelu);
                for _ in 0..n_gelu {
                    let inp = read_i16_le(data, &mut pos, "gelu.in")?;
                    let out = read_i16_le(data, &mut pos, "gelu.out")?;
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
                let num_heads = read_dim(data, &mut pos, "attn.num_heads")?;
                let seq_len = read_dim(data, &mut pos, "attn.seq_len")?;
                let d_head = read_dim(data, &mut pos, "attn.d_head")?;
                let exp_scale = read_i32_le(data, &mut pos, "attn.exp_scale")?;
                let n_kv = read_dim(data, &mut pos, "attn.n_kv")?;
                let k_values = read_u32_vec(data, &mut pos, n_kv, "attn.k_values")?;
                let v_values = read_u32_vec(data, &mut pos, n_kv, "attn.v_values")?;
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
                let gamma_len = read_dim(data, &mut pos, "rmsnorm.gamma_len")?;
                let gamma = read_u32_vec(data, &mut pos, gamma_len, "rmsnorm.gamma")?;
                let out_len = read_dim(data, &mut pos, "rmsnorm.out_len")?;
                let rmsnorm_output = read_u32_vec(data, &mut pos, out_len, "rmsnorm.output")?;
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
                let silu_scale = read_i32_le(data, &mut pos, "silu.scale")?;
                let n_silu = read_dim(data, &mut pos, "silu.n")?;
                let pair_bytes = checked_count(n_silu, 4, "silu.pair_bytes")?;
                check_remaining(data, pos, pair_bytes, "silu.pairs")?;
                let mut silu_input_f = Vec::with_capacity(n_silu);
                let mut silu_output_f = Vec::with_capacity(n_silu);
                for _ in 0..n_silu {
                    let inp = read_i16_le(data, &mut pos, "silu.in")?;
                    let out = read_i16_le(data, &mut pos, "silu.out")?;
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
                let silu_scale = read_i32_le(data, &mut pos, "swiglu.scale")?;
                let n = read_dim(data, &mut pos, "swiglu.n")?;
                let i16_bytes = checked_count(n, 2, "swiglu.i16_bytes")?;
                let mut gate_f = Vec::with_capacity(n);
                check_remaining(data, pos, i16_bytes, "swiglu.gate")?;
                for _ in 0..n {
                    let v = read_i16_le(data, &mut pos, "swiglu.gate.elem")?;
                    gate_f.push(i16_to_field(v).as_canonical_u32());
                }
                let mut gate_silu_f = Vec::with_capacity(n);
                check_remaining(data, pos, i16_bytes, "swiglu.gate_silu")?;
                for _ in 0..n {
                    let v = read_i16_le(data, &mut pos, "swiglu.gate_silu.elem")?;
                    gate_silu_f.push(i16_to_field(v).as_canonical_u32());
                }
                let up_values = read_u32_vec(data, &mut pos, n, "swiglu.up_values")?;
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
                let num_q_heads = read_dim(data, &mut pos, "gqa.num_q_heads")?;
                let num_kv_heads = read_dim(data, &mut pos, "gqa.num_kv_heads")?;
                let seq_len = read_dim(data, &mut pos, "gqa.seq_len")?;
                let d_head = read_dim(data, &mut pos, "gqa.d_head")?;
                let exp_scale = read_i32_le(data, &mut pos, "gqa.exp_scale")?;
                let n_kv = read_dim(data, &mut pos, "gqa.n_kv")?;
                let k_values = read_u32_vec(data, &mut pos, n_kv, "gqa.k_values")?;
                let v_values = read_u32_vec(data, &mut pos, n_kv, "gqa.v_values")?;
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
                let d_model = read_dim(data, &mut pos, "llama.d_model")?;
                let d_ff = read_dim(data, &mut pos, "llama.d_ff")?;
                let num_q_heads = read_dim(data, &mut pos, "llama.num_q_heads")?;
                let num_kv_heads = read_dim(data, &mut pos, "llama.num_kv_heads")?;
                let d_head = read_dim(data, &mut pos, "llama.d_head")?;
                let silu_scale = read_i32_le(data, &mut pos, "llama.silu_scale")?;

                let q_dim = checked_count(num_q_heads, d_head, "llama.q_dim")?;
                let kv_dim = checked_count(num_kv_heads, d_head, "llama.kv_dim")?;

                // Bulk read via centralized helper (E1 SAFETY: compile-time
                // size/align asserts + endianness gate live in
                // `read_fields_zero_copy`).
                let read_fields = |data: &[u8], pos: &mut usize, count: usize, ctx: &'static str|
                    -> Result<Vec<F>, BinaryParseError>
                {
                    let byte_len = checked_count(count, 4, ctx)?;
                    check_remaining(data, *pos, byte_len, ctx)?;
                    let slice = &data[*pos..*pos + byte_len];
                    *pos += byte_len;
                    Ok(read_fields_zero_copy(slice, count))
                };

                let norm1_gamma = read_fields(data, &mut pos, d_model, "llama.norm1_gamma")?;
                let w_q_count = checked_count(q_dim, d_model, "llama.w_q.count")?;
                let w_q = read_fields(data, &mut pos, w_q_count, "llama.w_q")?;
                let w_k_count = checked_count(kv_dim, d_model, "llama.w_k.count")?;
                let w_k = read_fields(data, &mut pos, w_k_count, "llama.w_k")?;
                let w_v_count = checked_count(kv_dim, d_model, "llama.w_v.count")?;
                let w_v = read_fields(data, &mut pos, w_v_count, "llama.w_v")?;
                let w_o_count = checked_count(d_model, q_dim, "llama.w_o.count")?;
                let w_o = read_fields(data, &mut pos, w_o_count, "llama.w_o")?;
                let norm2_gamma = read_fields(data, &mut pos, d_model, "llama.norm2_gamma")?;
                let w_gate_count = checked_count(d_ff, d_model, "llama.w_gate.count")?;
                let w_gate = read_fields(data, &mut pos, w_gate_count, "llama.w_gate")?;
                let w_up = read_fields(data, &mut pos, w_gate_count, "llama.w_up")?;
                let w_down_count = checked_count(d_model, d_ff, "llama.w_down.count")?;
                let w_down = read_fields(data, &mut pos, w_down_count, "llama.w_down")?;

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
                // qwen_layer: config + all 10 weight matrices (includes g_proj + sigmoid_scale).
                // Wire format adds v_num_heads/v_d_head after d_head for asymmetric GDN
                // (Qwen3.5-4B/9B). Old senders break — bump prover when client updates.
                let d_model = read_dim(data, &mut pos, "qwen.d_model")?;
                let d_ff = read_dim(data, &mut pos, "qwen.d_ff")?;
                let num_q_heads = read_dim(data, &mut pos, "qwen.num_q_heads")?;
                let num_kv_heads = read_dim(data, &mut pos, "qwen.num_kv_heads")?;
                let d_head = read_dim(data, &mut pos, "qwen.d_head")?;
                let v_num_heads = read_dim(data, &mut pos, "qwen.v_num_heads")?;
                let v_d_head = read_dim(data, &mut pos, "qwen.v_d_head")?;
                let silu_scale = read_i32_le(data, &mut pos, "qwen.silu_scale")?;
                let sigmoid_scale = read_i32_le(data, &mut pos, "qwen.sigmoid_scale")?;

                let q_dim = checked_count(num_q_heads, d_head, "qwen.q_dim")?;
                let k_dim = checked_count(num_kv_heads, d_head, "qwen.k_dim")?;
                let v_dim = checked_count(v_num_heads, v_d_head, "qwen.v_dim")?;
                // For GDN, attention output is v_dim (gate also v_dim, o_proj input v_dim).
                // For full attention, v_dim = k_dim and out = q_dim (GQA-replicated).
                // Heuristic: full-attn when num_q_heads != num_kv_heads (GQA pattern); else GDN.
                let attn_out_dim = if num_q_heads != num_kv_heads { q_dim } else { v_dim };

                let read_fields = |data: &[u8], pos: &mut usize, count: usize, ctx: &'static str|
                    -> Result<Vec<F>, BinaryParseError>
                {
                    let byte_len = checked_count(count, 4, ctx)?;
                    check_remaining(data, *pos, byte_len, ctx)?;
                    let slice = &data[*pos..*pos + byte_len];
                    *pos += byte_len;
                    Ok(read_fields_zero_copy(slice, count))
                };

                let norm1_gamma = read_fields(data, &mut pos, d_model, "qwen.norm1_gamma")?;
                let w_q_count = checked_count(q_dim, d_model, "qwen.w_q.count")?;
                let w_q = read_fields(data, &mut pos, w_q_count, "qwen.w_q")?;
                let w_k_count = checked_count(k_dim, d_model, "qwen.w_k.count")?;
                let w_k = read_fields(data, &mut pos, w_k_count, "qwen.w_k")?;
                let w_v_count = checked_count(v_dim, d_model, "qwen.w_v.count")?;
                let w_v = read_fields(data, &mut pos, w_v_count, "qwen.w_v")?;
                let w_o_count = checked_count(d_model, attn_out_dim, "qwen.w_o.count")?;
                let w_o = read_fields(data, &mut pos, w_o_count, "qwen.w_o")?;
                let w_g_count = checked_count(attn_out_dim, d_model, "qwen.w_g_proj.count")?;
                let w_g_proj = read_fields(data, &mut pos, w_g_count, "qwen.w_g_proj")?;
                let norm2_gamma = read_fields(data, &mut pos, d_model, "qwen.norm2_gamma")?;
                let w_gate_count = checked_count(d_ff, d_model, "qwen.w_gate.count")?;
                let w_gate = read_fields(data, &mut pos, w_gate_count, "qwen.w_gate")?;
                let w_up = read_fields(data, &mut pos, w_gate_count, "qwen.w_up")?;
                let w_down_count = checked_count(d_model, d_ff, "qwen.w_down.count")?;
                let w_down = read_fields(data, &mut pos, w_down_count, "qwen.w_down")?;

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
                        v_num_heads, v_d_head,
                    }),
                    qwen_weights: Some(QwenLayerWeightData {
                        norm1_gamma, w_q, w_k, w_v, w_o, w_g_proj,
                        norm2_gamma, w_gate, w_up, w_down,
                    }),
                });
            }
            _ => return Err(BinaryParseError::UnknownOpType(op_type_byte)),
        }
    }

    Ok(ProveRequest {
        mode: "mlp".into(),
        input,
        ops,
        gpt2: None,
    })
}

// ===== E2 regression tests: hostile binary inputs are rejected, not panicked =====

#[cfg(test)]
mod binary_hardening_tests {
    use super::*;

    /// SOUNDNESS / SAFETY (E2): truncated frame must return Truncated, not panic.
    #[test]
    fn test_parse_binary_truncated_header_rejects() {
        // Less than 4 bytes — can't even read num_ops.
        let r = parse_binary_checked(&[0u8; 2]);
        assert!(matches!(r, Err(BinaryParseError::Truncated { .. })));
    }

    /// SAFETY (E2): num_ops above MAX_OPS rejected before allocation.
    #[test]
    fn test_parse_binary_oversized_op_count_rejects() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&u32::MAX.to_le_bytes()); // num_ops = 2^32 - 1
        buf.extend_from_slice(&0u32.to_le_bytes());     // input_len
        let r = parse_binary_checked(&buf);
        assert!(matches!(r, Err(BinaryParseError::OpCountTooLarge { .. })));
    }

    /// SAFETY (E2): input_len above MAX_DIM rejected.
    #[test]
    fn test_parse_binary_oversized_input_len_rejects() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_le_bytes());        // num_ops = 0
        buf.extend_from_slice(&u32::MAX.to_le_bytes());    // input_len = 2^32 - 1
        let r = parse_binary_checked(&buf);
        assert!(matches!(r, Err(BinaryParseError::DimensionTooLarge { .. })));
    }

    /// SAFETY (E2): linear `m * n` overflow returns DimensionOverflow, no panic.
    /// Builds a frame with m = n = 2^31; product overflows usize on 64-bit.
    #[test]
    fn test_parse_binary_linear_mn_overflow_rejects() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes());         // num_ops = 1
        buf.extend_from_slice(&0u32.to_le_bytes());         // input_len = 0
        buf.push(0u8);                                       // op_type = 0 (linear)
        buf.extend_from_slice(&3u32.to_le_bytes());         // name_len = 3
        buf.extend_from_slice(b"foo");
        // m and n each at MAX_DIM cap (16M). m * n = 2^48 → exceeds MAX_ELEMENT_COUNT (2^30).
        buf.extend_from_slice(&((1u32 << 24) - 1).to_le_bytes());
        buf.extend_from_slice(&((1u32 << 24) - 1).to_le_bytes());
        let err = parse_binary_checked(&buf).err()
            .expect("expected element-count cap to reject");
        assert!(matches!(err, BinaryParseError::DimensionTooLarge { .. }),
            "expected element-count cap, got {}", err);
    }

    /// SAFETY (E2): unknown op_type byte returns error rather than panicking.
    #[test]
    fn test_parse_binary_unknown_op_type_rejects() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes());     // num_ops = 1
        buf.extend_from_slice(&0u32.to_le_bytes());     // input_len = 0
        buf.push(99u8);                                  // unknown op_type
        buf.extend_from_slice(&0u32.to_le_bytes());     // name_len = 0
        let r = parse_binary_checked(&buf);
        assert!(matches!(r, Err(BinaryParseError::UnknownOpType(99))));
    }

    /// SAFETY (E2): truncated linear payload (declares m,n but not enough bytes
    /// for w_q) returns Truncated, not panic via slice indexing.
    #[test]
    fn test_parse_binary_truncated_linear_payload_rejects() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes());     // num_ops = 1
        buf.extend_from_slice(&0u32.to_le_bytes());     // input_len = 0
        buf.push(0u8);                                   // op_type = linear
        buf.extend_from_slice(&3u32.to_le_bytes());     // name_len
        buf.extend_from_slice(b"foo");
        buf.extend_from_slice(&4u32.to_le_bytes());     // m = 4
        buf.extend_from_slice(&4u32.to_le_bytes());     // n = 4
        // Should expect 16 u32 weights + 4 u32 biases = 80 bytes; provide nothing.
        let r = parse_binary_checked(&buf);
        assert!(matches!(r, Err(BinaryParseError::Truncated { .. })));
    }

    /// SAFETY (E2): well-formed minimal frame (zero ops, zero input) parses.
    #[test]
    fn test_parse_binary_empty_frame_parses() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_le_bytes());     // num_ops = 0
        buf.extend_from_slice(&0u32.to_le_bytes());     // input_len = 0
        let r = parse_binary_checked(&buf).expect("valid empty frame should parse");
        assert_eq!(r.ops.len(), 0);
        assert_eq!(r.input.len(), 0);
    }

    /// SAFETY (E3a): truncated stdin during preload returns io::Error rather
    /// than panicking via `unwrap`. Exercised via `Cursor<&[u8]>`.
    #[test]
    fn test_read_u32_from_checked_truncated_returns_err() {
        use std::io::Cursor;
        let buf = [0u8; 2]; // only 2 bytes — read_exact(4) must fail
        let mut c = Cursor::new(&buf[..]);
        let r = read_u32_from_checked(&mut c);
        assert!(r.is_err(), "expected truncated read to fail");
    }

    /// SAFETY (E3a): well-formed 4 bytes round-trip through Result reader.
    #[test]
    fn test_read_u32_from_checked_round_trip() {
        use std::io::Cursor;
        let buf = 0xDEAD_BEEFu32.to_le_bytes();
        let mut c = Cursor::new(&buf[..]);
        let v = read_u32_from_checked(&mut c).expect("round-trip read");
        assert_eq!(v, 0xDEAD_BEEF);
    }

    /// SAFETY (E3a): `read_exact_vec` propagates io::Error on truncation.
    #[test]
    fn test_read_exact_vec_truncated_returns_err() {
        use std::io::Cursor;
        let buf = [1u8, 2, 3];
        let mut c = Cursor::new(&buf[..]);
        let r = read_exact_vec(&mut c, 16);
        assert!(r.is_err(), "expected truncated read_exact_vec to fail");
    }

    /// SAFETY (E1): structural invariants Mersenne31 vs u32 must hold at
    /// compile time. This test re-asserts at runtime so a CI failure is
    /// loud and explicit even if the const assertion somehow optimizes
    /// away (it shouldn't, but defense in depth).
    #[test]
    fn test_mersenne31_layout_matches_u32() {
        assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<u32>(),
            "Mersenne31 size diverged from u32 — zero-copy transmute is UB");
        assert_eq!(std::mem::align_of::<F>(), std::mem::align_of::<u32>(),
            "Mersenne31 alignment diverged from u32 — zero-copy transmute is UB");
    }

    /// SAFETY (E1): round-trip a known little-endian byte buffer through
    /// `read_fields_zero_copy` and confirm canonical Mersenne31 values.
    #[test]
    fn test_read_fields_zero_copy_round_trip() {
        use p3_field::PrimeField32;
        let values: Vec<u32> = vec![0, 1, 2, 100, (1u32 << 31) - 2]; // all canonical
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let f_vec = read_fields_zero_copy(&bytes, values.len());
        assert_eq!(f_vec.len(), values.len());
        for (i, expected) in values.iter().enumerate() {
            assert_eq!(f_vec[i].as_canonical_u32(), *expected,
                "round-trip mismatch at index {}", i);
        }
    }

    /// SAFETY (E1): `transmute_u32_to_f` round-trips canonical values.
    #[test]
    fn test_transmute_u32_to_f_round_trip() {
        use p3_field::PrimeField32;
        let values: Vec<u32> = (0..32).collect();
        let f_vec = transmute_u32_to_f(values.clone());
        for (i, expected) in values.iter().enumerate() {
            assert_eq!(f_vec[i].as_canonical_u32(), *expected,
                "transmute round-trip mismatch at index {}", i);
        }
    }

    // ===== E7 (P10-8): wire-format pin for the OpDesc → Op enum refactor =====
    //
    // SAFETY: this is the contract test — every variant supported on the
    // JSON wire is deserialized here and asserted to land in the right
    // `OpDesc.op_type` slot with its payload fields populated. If any
    // sender's wire shape changes, or a future refactor renames a variant,
    // this test fires loudly. The discriminator strings here are
    // load-bearing — keep them in lock-step with the senders.

    /// SAFETY (E7): all known JSON discriminators round-trip through
    /// `OpDesc`'s custom `Deserialize` (which goes via `Op`). The op_type
    /// strings asserted below are the canonical wire form; any mismatch is
    /// a wire-break and a soundness regression for the prover.
    #[test]
    fn op_wire_format_pin() {
        // Each entry: (json fragment, expected op_type, payload assertion).
        let cases: Vec<(&str, &str)> = vec![
            (r#"{"type":"linear","name":"fc1","m":2,"n":3,"w_q":[1,2,3,4,5,6],"b_q":[7,8]}"#, "linear"),
            (r#"{"type":"relu","name":"r1"}"#, "relu"),
            (r#"{"type":"set_input","name":"si","new_input":[42,43]}"#, "set_input"),
            (r#"{"type":"layernorm","name":"ln","gamma":[1,2],"beta":[3,4]}"#, "layernorm"),
            (r#"{"type":"save","name":"s","save_name":"buf"}"#, "save"),
            (r#"{"type":"add_saved","name":"a","add_name":"buf"}"#, "add_saved"),
            (r#"{"type":"gelu","name":"g","w_q":[1],"b_q":[2],"gelu_scale":500}"#, "gelu"),
            (r#"{"type":"attention","name":"at","num_heads":2,"seq_len":4,"d_head":8,"exp_scale":100,"k_values":[1,2],"v_values":[3,4]}"#, "attention"),
            (r#"{"type":"rmsnorm","name":"rn","gamma":[1,2]}"#, "rmsnorm"),
            (r#"{"type":"silu","name":"si","w_q":[1],"b_q":[2],"gelu_scale":1000}"#, "silu"),
            (r#"{"type":"swiglu","name":"sg","w_q":[1],"b_q":[2],"ln_output":[3],"gelu_scale":10}"#, "swiglu"),
            (r#"{"type":"gqa_attention","name":"gqa","num_heads":4,"seq_len":8,"d_head":2,"exp_scale":1,"k_values":[1],"v_values":[2]}"#, "gqa_attention"),
            (r#"{"type":"passthrough","name":"pt"}"#, "passthrough"),
        ];

        for (json, expected_type) in &cases {
            let parsed: OpDesc = serde_json::from_str(json)
                .unwrap_or_else(|e| panic!("failed to parse {} variant: {}", expected_type, e));
            assert_eq!(parsed.op_type, *expected_type,
                "wire discriminator mismatch for {}: got {}", expected_type, parsed.op_type);
            assert!(!parsed.name.is_empty(), "name field lost during lowering for {}", expected_type);
        }

        // Per-variant payload spot-checks: ensure fields land in the
        // expected `OpDesc` slots (not zero-defaulted).
        let linear: OpDesc = serde_json::from_str(
            r#"{"type":"linear","name":"fc1","m":2,"n":3,"w_q":[1,2,3,4,5,6],"b_q":[7,8]}"#
        ).unwrap();
        assert_eq!(linear.m, 2);
        assert_eq!(linear.n, 3);
        assert_eq!(linear.w_q, vec![1,2,3,4,5,6]);
        assert_eq!(linear.b_q, vec![7,8]);

        let set_in: OpDesc = serde_json::from_str(
            r#"{"type":"set_input","name":"x","new_input":[42,43,44]}"#
        ).unwrap();
        assert_eq!(set_in.new_input, vec![42, 43, 44]);

        let ln: OpDesc = serde_json::from_str(
            r#"{"type":"layernorm","name":"x","gamma":[1,2],"beta":[3,4]}"#
        ).unwrap();
        assert_eq!(ln.gamma, vec![1, 2]);
        assert_eq!(ln.beta, vec![3, 4]);
        assert_eq!(ln.ln_output, Vec::<u32>::new());

        let attn: OpDesc = serde_json::from_str(
            r#"{"type":"attention","name":"a","num_heads":2,"seq_len":4,"d_head":8,"exp_scale":100,"k_values":[1,2],"v_values":[3,4]}"#
        ).unwrap();
        assert_eq!(attn.num_heads, 2);
        assert_eq!(attn.seq_len, 4);
        assert_eq!(attn.d_head, 8);
        assert_eq!(attn.exp_scale, 100);
        assert_eq!(attn.k_values, vec![1, 2]);
        assert_eq!(attn.v_values, vec![3, 4]);

        let save: OpDesc = serde_json::from_str(
            r#"{"type":"save","name":"x","save_name":"buf42"}"#
        ).unwrap();
        assert_eq!(save.save_name, "buf42");

        let add_saved: OpDesc = serde_json::from_str(
            r#"{"type":"add_saved","name":"x","add_name":"buf42"}"#
        ).unwrap();
        assert_eq!(add_saved.add_name, "buf42");

        let gelu: OpDesc = serde_json::from_str(
            r#"{"type":"gelu","name":"g","w_q":[10],"b_q":[20],"gelu_scale":500}"#
        ).unwrap();
        assert_eq!(gelu.w_q, vec![10]);
        assert_eq!(gelu.b_q, vec![20]);
        assert_eq!(gelu.gelu_scale, 500);
    }

    /// SAFETY (E7): unknown discriminators MUST fail at parse time. This
    /// is the headline win of the refactor — the old `OpDesc` would happily
    /// accept `{"type":"bogus"}` and only blow up later with an unrelated
    /// panic in the dispatcher.
    #[test]
    fn op_wire_unknown_discriminator_rejected() {
        let r: Result<OpDesc, _> = serde_json::from_str(r#"{"type":"definitely_not_an_op","name":"x"}"#);
        assert!(r.is_err(), "unknown op discriminator must fail at parse time");
    }

    /// SAFETY (E7): missing required payload fields per variant fail at
    /// parse time instead of zero-defaulting. e.g. `linear` without `m`.
    #[test]
    fn op_wire_missing_required_field_rejected() {
        // `linear` missing `m` field — used to silently parse as m=0.
        let r: Result<OpDesc, _> = serde_json::from_str(
            r#"{"type":"linear","name":"fc","n":3,"w_q":[1,2,3],"b_q":[4,5,6]}"#
        );
        assert!(r.is_err(), "linear missing m field must fail at parse time");
    }

    /// SAFETY (E7): the `gelu_scale` default must still apply for variants
    /// that omit the field (preserves wire compat with older traces).
    #[test]
    fn op_wire_gelu_scale_default_applied() {
        let g: OpDesc = serde_json::from_str(
            r#"{"type":"gelu","name":"g"}"#
        ).expect("gelu with all-defaults should parse");
        assert_eq!(g.gelu_scale, default_gelu_scale());
    }
}
