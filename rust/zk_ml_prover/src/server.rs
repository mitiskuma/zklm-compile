//! Binary protocol server loop for persistent-connection proving.

use std::collections::HashMap;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::time::Instant;

use p3_field::{AbstractField, PrimeField32};
use p3_mersenne_31::Mersenne31;

use crate::pipeline::run_mlp_mode_returning;
use crate::protocol::*;
use crate::proving::weight_commitment::commit_weights_fast;

type F = Mersenne31;

/// SAFETY (E3a): caps for server preload phase. Match parser caps from
/// protocol.rs. The Python orchestrator is trusted but bugs can still produce
/// hostile-looking dims; these caps keep the prover process alive.
const PRELOAD_MAX_WEIGHTS: u32 = 1 << 16;          // 65,536 — far above any model
const PRELOAD_MAX_DIM: usize = 1 << 24;            // 16M
const PRELOAD_MAX_NAME_LEN: usize = 1 << 12;       // 4 KiB — names are short
const PRELOAD_MAX_ELEMENT_COUNT: usize = 1 << 30;  // 4 GiB at 4 B/elem

pub(crate) fn run_server_mode(reader: &mut BufReader<io::Stdin>, writer: &mut BufWriter<io::Stdout>) {
    if let Err(e) = run_server_mode_inner(reader, writer) {
        eprintln!("  server: fatal error during preload — {}", e);
        // Drain stdin briefly so the orchestrator notices the EPIPE rather than
        // hanging on a half-written frame; then return cleanly so caller exits.
        let _ = reader;
    }
}

fn run_server_mode_inner(reader: &mut BufReader<io::Stdin>, writer: &mut BufWriter<io::Stdout>)
    -> io::Result<()>
{
    let t_preload = Instant::now();

    // Read preload header: num_weight_matrices
    let num_weights = read_u32_from_checked(reader)?;
    if num_weights > PRELOAD_MAX_WEIGHTS {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("num_weights {} exceeds cap {}", num_weights, PRELOAD_MAX_WEIGHTS)));
    }
    eprintln!("  server: preloading {} weight matrices", num_weights);

    let mut weights: HashMap<String, PreloadedLinear> = HashMap::new();
    for i in 0..num_weights {
        let t_w = Instant::now();
        let name_len = read_u32_from_checked(reader)? as usize;
        if name_len > PRELOAD_MAX_NAME_LEN {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("preload weight[{}] name_len {} exceeds cap {}",
                    i, name_len, PRELOAD_MAX_NAME_LEN)));
        }
        let name_buf = read_exact_vec(reader, name_len)?;
        let name = String::from_utf8(name_buf)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData,
                format!("preload weight[{}] name not valid UTF-8", i)))?;

        let m = read_u32_from_checked(reader)? as usize;
        let n = read_u32_from_checked(reader)? as usize;
        if m > PRELOAD_MAX_DIM || n > PRELOAD_MAX_DIM {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("preload weight[{}] '{}' dim ({}, {}) exceeds cap {}",
                    i, name, m, n, PRELOAD_MAX_DIM)));
        }

        // SAFETY: checked_mul prevents `m * n` overflow on hostile dims;
        // additional cap guards against OOM via huge legal-but-unreasonable dims.
        let total = m.checked_mul(n)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData,
                format!("preload weight[{}] '{}' m*n overflow ({} * {})",
                    i, name, m, n)))?;
        if total > PRELOAD_MAX_ELEMENT_COUNT {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("preload weight[{}] '{}' element count {} exceeds cap {}",
                    i, name, total, PRELOAD_MAX_ELEMENT_COUNT)));
        }

        let mut w_f = Vec::with_capacity(total);
        for _ in 0..total {
            w_f.push(F::from_canonical_u32(read_u32_from_checked(reader)?));
        }

        let commitment = commit_weights_fast(&w_f);
        eprintln!(
            "    weight[{}] '{}' {}x{} ({:.1}MB): {:.1}ms",
            i, name, m, n,
            (total * 4) as f64 / 1e6,
            t_w.elapsed().as_secs_f64() * 1000.0
        );

        weights.insert(name, PreloadedLinear { w_f, m, n, commitment });
    }

    eprintln!(
        "  server: preload complete in {:.1}ms",
        t_preload.elapsed().as_secs_f64() * 1000.0
    );

    // Prove request loop
    let mut request_num = 0u64;
    loop {
        // Read payload_size (4 bytes). EOF = shutdown.
        let payload_size = match read_u32_from_opt(reader) {
            Some(v) => v as usize,
            None => {
                eprintln!("  server: EOF, shutting down");
                break;
            }
        };
        if payload_size == 0 {
            eprintln!("  server: shutdown signal (payload_size=0)");
            break;
        }

        request_num += 1;
        let t_req = Instant::now();

        // Read entire payload
        let mut payload = vec![0u8; payload_size];
        reader.read_exact(&mut payload)
            .expect("IO error reading request payload — client likely disconnected");

        // Safe payload reading helpers — return None on truncated data instead of panicking.
        let safe_read_u32 = |payload: &[u8], pos: &mut usize| -> Option<u32> {
            if *pos + 4 > payload.len() {
                eprintln!("  server: malformed payload, unexpected end at offset {} (need 4 bytes, have {})", *pos, payload.len() - *pos);
                return None;
            }
            let v = u32::from_le_bytes(payload[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Some(v)
        };
        let safe_read_i32 = |payload: &[u8], pos: &mut usize| -> Option<i32> {
            if *pos + 4 > payload.len() {
                eprintln!("  server: malformed payload, unexpected end at offset {} (need 4 bytes for i32)", *pos);
                return None;
            }
            let v = i32::from_le_bytes(payload[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Some(v)
        };
        let safe_read_i16 = |payload: &[u8], pos: &mut usize| -> Option<i16> {
            if *pos + 2 > payload.len() {
                eprintln!("  server: malformed payload, unexpected end at offset {} (need 2 bytes for i16)", *pos);
                return None;
            }
            let v = i16::from_le_bytes(payload[*pos..*pos + 2].try_into().unwrap());
            *pos += 2;
            Some(v)
        };
        let safe_read_bytes = |payload: &[u8], pos: &mut usize, len: usize| -> Option<Vec<u8>> {
            if *pos + len > payload.len() {
                eprintln!("  server: malformed payload, unexpected end at offset {} (need {} bytes)", *pos, len);
                return None;
            }
            let v = payload[*pos..*pos + len].to_vec();
            *pos += len;
            Some(v)
        };
        let safe_read_string = |payload: &[u8], pos: &mut usize| -> Option<String> {
            let len = safe_read_u32(payload, pos)? as usize;
            let bytes = safe_read_bytes(payload, pos, len)?;
            match String::from_utf8(bytes) {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("  server: invalid UTF-8 in string at offset {}: {}", *pos - len, e);
                    None
                }
            }
        };
        // Read N u32 values as F field elements.
        let safe_read_f_vec = |payload: &[u8], pos: &mut usize, count: usize| -> Option<Vec<F>> {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(F::from_canonical_u32(safe_read_u32(payload, pos)?));
            }
            Some(v)
        };
        // Read N u32 values as raw u32.
        let safe_read_u32_vec = |payload: &[u8], pos: &mut usize, count: usize| -> Option<Vec<u32>> {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(safe_read_u32(payload, pos)?);
            }
            Some(v)
        };
        // Bulk read: read count u32 values and transmute to F (for large weight matrices).
        // SAFETY (E3a): use checked_mul on `count * 4` and saturating_sub on
        // remaining-bytes so a hostile `count` cannot wrap or panic on subtraction.
        let safe_read_bulk_f = |payload: &[u8], pos: &mut usize, count: usize| -> Option<Vec<F>> {
            let byte_len = match count.checked_mul(4) {
                Some(v) => v,
                None => {
                    eprintln!("  server: malformed payload, count*4 overflow (count={})", count);
                    return None;
                }
            };
            if payload.len().saturating_sub(*pos) < byte_len {
                eprintln!(
                    "  server: malformed payload, unexpected end at offset {} (need {} bytes for bulk read)",
                    *pos, byte_len,
                );
                return None;
            }
            let slice = &payload[*pos..*pos + byte_len];
            *pos += byte_len;
            Some(transmute_u32_to_f(slice.chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()))
        };

        // Parse payload using safe helpers. On any truncation, skip this request.
        let mut pos = 0;
        // Use a closure-like block to allow early exit via break on parse failure.
        let parsed = 'parse: {
            macro_rules! try_read {
                ($expr:expr) => {
                    match $expr {
                        Some(v) => v,
                        None => break 'parse None,
                    }
                };
            }

            let num_ops = try_read!(safe_read_u32(&payload, &mut pos));
            let input_len = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
            let input_u32 = try_read!(safe_read_u32_vec(&payload, &mut pos, input_len));

            // Parse ops
            let mut ops: Vec<ServerOp> = Vec::with_capacity(num_ops as usize);
            for _ in 0..num_ops {
                if pos >= payload.len() {
                    eprintln!("  server: malformed payload, unexpected end reading op_type byte at offset {}", pos);
                    break 'parse None;
                }
                let op_type_byte = payload[pos];
                pos += 1;

                let name = try_read!(safe_read_string(&payload, &mut pos));

                match op_type_byte {
                    0 => {
                        // linear_ref
                        let weight_name = try_read!(safe_read_string(&payload, &mut pos));
                        let bias_len = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let bias = try_read!(safe_read_f_vec(&payload, &mut pos, bias_len));
                        ops.push(ServerOp::LinearRef { name, weight_name, bias });
                    }
                    1 => {
                        ops.push(ServerOp::Relu { name });
                    }
                    2 => {
                        let new_input_len = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let new_input = try_read!(safe_read_u32_vec(&payload, &mut pos, new_input_len));
                        ops.push(ServerOp::SetInput { name, new_input });
                    }
                    3 => {
                        // layernorm: gamma_len, gamma[], beta_len, beta[], ln_out_len, ln_output[]
                        let gamma_len = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let gamma = try_read!(safe_read_f_vec(&payload, &mut pos, gamma_len));
                        let beta_len = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let beta = try_read!(safe_read_f_vec(&payload, &mut pos, beta_len));
                        let ln_out_len = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let ln_output = try_read!(safe_read_u32_vec(&payload, &mut pos, ln_out_len));
                        ops.push(ServerOp::LayerNorm { name, gamma, beta, ln_output });
                    }
                    4 => {
                        // save: save_name
                        let save_name = try_read!(safe_read_string(&payload, &mut pos));
                        ops.push(ServerOp::Save { name, save_name });
                    }
                    5 => {
                        // add_saved: add_name
                        let add_name = try_read!(safe_read_string(&payload, &mut pos));
                        ops.push(ServerOp::AddSaved { name, add_name });
                    }
                    6 => {
                        // gelu: scale (i32), n_gelu (u32), then n_gelu pairs of (input_i16, output_i16)
                        let gelu_scale = try_read!(safe_read_i32(&payload, &mut pos));
                        let n_gelu = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let mut gelu_input_i16 = Vec::with_capacity(n_gelu);
                        let mut gelu_output_i16 = Vec::with_capacity(n_gelu);
                        for _ in 0..n_gelu {
                            gelu_input_i16.push(try_read!(safe_read_i16(&payload, &mut pos)));
                            gelu_output_i16.push(try_read!(safe_read_i16(&payload, &mut pos)));
                        }
                        ops.push(ServerOp::Gelu { name, gelu_scale, gelu_input_i16, gelu_output_i16 });
                    }
                    7 => {
                        // attention: num_heads, seq_len, d_head, exp_scale, k_values, v_values
                        let num_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let seq_len = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let d_head = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let exp_scale = try_read!(safe_read_i32(&payload, &mut pos));
                        let n_kv = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let k_values = try_read!(safe_read_u32_vec(&payload, &mut pos, n_kv));
                        let v_values = try_read!(safe_read_u32_vec(&payload, &mut pos, n_kv));
                        ops.push(ServerOp::Attention {
                            name, num_heads, seq_len, d_head, exp_scale, k_values, v_values,
                        });
                    }
                    8 => {
                        // rmsnorm: gamma_len, gamma[], rmsnorm_output_len, rmsnorm_output[]
                        let gamma_len = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let gamma = try_read!(safe_read_f_vec(&payload, &mut pos, gamma_len));
                        let out_len = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let rmsnorm_output = try_read!(safe_read_u32_vec(&payload, &mut pos, out_len));
                        ops.push(ServerOp::RmsNorm { name, gamma, rmsnorm_output });
                    }
                    9 => {
                        // silu: silu_scale, n_entries, (input_i16, output_i16)*
                        let silu_scale = try_read!(safe_read_i32(&payload, &mut pos));
                        let n_silu = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let mut silu_input_i16 = Vec::with_capacity(n_silu);
                        let mut silu_output_i16 = Vec::with_capacity(n_silu);
                        for _ in 0..n_silu {
                            silu_input_i16.push(try_read!(safe_read_i16(&payload, &mut pos)));
                            silu_output_i16.push(try_read!(safe_read_i16(&payload, &mut pos)));
                        }
                        ops.push(ServerOp::Silu { name, silu_scale, silu_input_i16, silu_output_i16 });
                    }
                    10 => {
                        // swiglu: silu_scale, n_entries, gate_i16[], gate_silu_i16[], up_values[]
                        let silu_scale = try_read!(safe_read_i32(&payload, &mut pos));
                        let n = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let mut gate_i16 = Vec::with_capacity(n);
                        for _ in 0..n {
                            gate_i16.push(try_read!(safe_read_i16(&payload, &mut pos)));
                        }
                        let mut gate_silu_i16 = Vec::with_capacity(n);
                        for _ in 0..n {
                            gate_silu_i16.push(try_read!(safe_read_i16(&payload, &mut pos)));
                        }
                        let up_values = try_read!(safe_read_u32_vec(&payload, &mut pos, n));
                        ops.push(ServerOp::SwiGlu { name, silu_scale, gate_i16, gate_silu_i16, up_values });
                    }
                    11 => {
                        // gqa_attention: num_q_heads, num_kv_heads, seq_len, d_head, exp_scale, k_values, v_values
                        let num_q_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let num_kv_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let seq_len = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let d_head = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let exp_scale = try_read!(safe_read_i32(&payload, &mut pos));
                        let n_kv = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let k_values = try_read!(safe_read_u32_vec(&payload, &mut pos, n_kv));
                        let v_values = try_read!(safe_read_u32_vec(&payload, &mut pos, n_kv));
                        ops.push(ServerOp::GqaAttention {
                            name, num_q_heads, num_kv_heads, seq_len, d_head, exp_scale, k_values, v_values,
                        });
                    }
                    12 => {
                        // llama_layer
                        let d_model = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let d_ff = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let num_q_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let num_kv_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let d_head = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let silu_scale = try_read!(safe_read_i32(&payload, &mut pos));

                        let q_dim = num_q_heads * d_head;
                        let kv_dim = num_kv_heads * d_head;

                        let norm1_gamma = try_read!(safe_read_bulk_f(&payload, &mut pos, d_model));
                        let w_q = try_read!(safe_read_bulk_f(&payload, &mut pos, q_dim * d_model));
                        let w_k = try_read!(safe_read_bulk_f(&payload, &mut pos, kv_dim * d_model));
                        let w_v = try_read!(safe_read_bulk_f(&payload, &mut pos, kv_dim * d_model));
                        let w_o = try_read!(safe_read_bulk_f(&payload, &mut pos, d_model * q_dim));
                        let norm2_gamma = try_read!(safe_read_bulk_f(&payload, &mut pos, d_model));
                        let w_gate = try_read!(safe_read_bulk_f(&payload, &mut pos, d_ff * d_model));
                        let w_up = try_read!(safe_read_bulk_f(&payload, &mut pos, d_ff * d_model));
                        let w_down = try_read!(safe_read_bulk_f(&payload, &mut pos, d_model * d_ff));

                        ops.push(ServerOp::LlamaLayer {
                            name,
                            config: LlamaLayerConfig {
                                d_model, d_ff, num_q_heads, num_kv_heads, d_head, silu_scale,
                            },
                            weights: LlamaLayerWeightData {
                                norm1_gamma, w_q, w_k, w_v, w_o,
                                norm2_gamma, w_gate, w_up, w_down,
                            },
                        });
                    }
                    13 => {
                        // llama_layer_ref: config + 9 weight name references
                        let d_model = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let d_ff = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let num_q_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let num_kv_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let d_head = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let silu_scale = try_read!(safe_read_i32(&payload, &mut pos));

                        let weight_names: [String; 9] = [
                            try_read!(safe_read_string(&payload, &mut pos)), // norm1_gamma
                            try_read!(safe_read_string(&payload, &mut pos)), // w_q
                            try_read!(safe_read_string(&payload, &mut pos)), // w_k
                            try_read!(safe_read_string(&payload, &mut pos)), // w_v
                            try_read!(safe_read_string(&payload, &mut pos)), // w_o
                            try_read!(safe_read_string(&payload, &mut pos)), // norm2_gamma
                            try_read!(safe_read_string(&payload, &mut pos)), // w_gate
                            try_read!(safe_read_string(&payload, &mut pos)), // w_up
                            try_read!(safe_read_string(&payload, &mut pos)), // w_down
                        ];

                        ops.push(ServerOp::LlamaLayerRef {
                            name,
                            config: LlamaLayerConfig {
                                d_model, d_ff, num_q_heads, num_kv_heads, d_head, silu_scale,
                            },
                            weight_names,
                        });
                    }
                    14 => {
                        // qwen_layer (inline weights). v_num_heads/v_d_head added for asymmetric GDN.
                        let d_model = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let d_ff = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let num_q_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let num_kv_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let d_head = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let v_num_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let v_d_head = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let silu_scale = try_read!(safe_read_i32(&payload, &mut pos));
                        let sigmoid_scale = try_read!(safe_read_i32(&payload, &mut pos));

                        let q_dim = num_q_heads * d_head;
                        let k_dim = num_kv_heads * d_head;
                        let v_dim = v_num_heads * v_d_head;
                        // Heuristic: GQA full-attn has q_heads != kv_heads (output q_dim).
                        // GDN has q_heads == kv_heads (output v_dim).
                        let attn_out_dim = if num_q_heads != num_kv_heads { q_dim } else { v_dim };

                        let norm1_gamma = try_read!(safe_read_bulk_f(&payload, &mut pos, d_model));
                        let w_q = try_read!(safe_read_bulk_f(&payload, &mut pos, q_dim * d_model));
                        let w_k = try_read!(safe_read_bulk_f(&payload, &mut pos, k_dim * d_model));
                        let w_v = try_read!(safe_read_bulk_f(&payload, &mut pos, v_dim * d_model));
                        let w_o = try_read!(safe_read_bulk_f(&payload, &mut pos, d_model * attn_out_dim));
                        let w_g_proj = try_read!(safe_read_bulk_f(&payload, &mut pos, attn_out_dim * d_model));
                        let norm2_gamma = try_read!(safe_read_bulk_f(&payload, &mut pos, d_model));
                        let w_gate = try_read!(safe_read_bulk_f(&payload, &mut pos, d_ff * d_model));
                        let w_up = try_read!(safe_read_bulk_f(&payload, &mut pos, d_ff * d_model));
                        let w_down = try_read!(safe_read_bulk_f(&payload, &mut pos, d_model * d_ff));

                        ops.push(ServerOp::QwenLayer {
                            name,
                            config: QwenLayerConfig {
                                d_model, d_ff, num_q_heads, num_kv_heads, d_head,
                                silu_scale, sigmoid_scale,
                                v_num_heads, v_d_head,
                            },
                            weights: QwenLayerWeightData {
                                norm1_gamma, w_q, w_k, w_v, w_o, w_g_proj,
                                norm2_gamma, w_gate, w_up, w_down,
                            },
                        });
                    }
                    15 => {
                        // qwen_layer_ref: config + 10 weight name references.
                        // Wire format adds v_num_heads/v_d_head after d_head for asymmetric GDN.
                        let d_model = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let d_ff = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let num_q_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let num_kv_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let d_head = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let v_num_heads = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let v_d_head = try_read!(safe_read_u32(&payload, &mut pos)) as usize;
                        let silu_scale = try_read!(safe_read_i32(&payload, &mut pos));
                        let sigmoid_scale = try_read!(safe_read_i32(&payload, &mut pos));

                        let weight_names: [String; 10] = [
                            try_read!(safe_read_string(&payload, &mut pos)), // norm1_gamma
                            try_read!(safe_read_string(&payload, &mut pos)), // w_q
                            try_read!(safe_read_string(&payload, &mut pos)), // w_k
                            try_read!(safe_read_string(&payload, &mut pos)), // w_v
                            try_read!(safe_read_string(&payload, &mut pos)), // w_o
                            try_read!(safe_read_string(&payload, &mut pos)), // w_g_proj
                            try_read!(safe_read_string(&payload, &mut pos)), // norm2_gamma
                            try_read!(safe_read_string(&payload, &mut pos)), // w_gate
                            try_read!(safe_read_string(&payload, &mut pos)), // w_up
                            try_read!(safe_read_string(&payload, &mut pos)), // w_down
                        ];

                        ops.push(ServerOp::QwenLayerRef {
                            name,
                            config: QwenLayerConfig {
                                d_model, d_ff, num_q_heads, num_kv_heads, d_head,
                                silu_scale, sigmoid_scale,
                                v_num_heads, v_d_head,
                            },
                            weight_names,
                        });
                    }
                    _ => {
                        eprintln!("  server req#{}: unknown op_type byte {} at offset {}, skipping request", request_num, op_type_byte, pos - 1);
                        break 'parse None;
                    }
                }
            }

            Some((ops, input_u32))
        };

        let (ops, input_u32) = match parsed {
            Some(v) => v,
            None => {
                eprintln!("  server req#{}: skipping malformed request", request_num);
                continue;
            }
        };

        eprintln!(
            "  server req#{}: parsed {:.1}ms, {} ops, input_len={}",
            request_num,
            t_req.elapsed().as_secs_f64() * 1000.0,
            ops.len(),
            input_u32.len()
        );

        // Build a ProveRequest from server ops, referencing preloaded weights
        let mut req_ops = Vec::with_capacity(ops.len());
        for op in &ops {
            match op {
                ServerOp::LinearRef { name: _, weight_name, bias } => {
                    let pw = weights.get(weight_name)
                        .unwrap_or_else(|| panic!("Unknown preloaded weight: {}", weight_name));
                    // Use weight_name as op name so preloaded lookup works in run_mlp_mode_returning
                    req_ops.push(OpDesc {
                        op_type: "linear".into(),
                        name: weight_name.clone(),
                        m: pw.m,
                        n: pw.n,
                        w_q: vec![],  // empty — will use preloaded via name lookup
                        b_q: bias.iter().map(|v| v.as_canonical_u32()).collect(),
                        gelu_scale: default_gelu_scale(),
                        gamma: vec![],
                        beta: vec![],
                        save_name: String::new(),
                        add_name: String::new(),
                        new_input: vec![],
                        ln_output: vec![],
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::Relu { name } => {
                    req_ops.push(OpDesc {
                        op_type: "relu".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: vec![], b_q: vec![],
                        gelu_scale: default_gelu_scale(),
                        gamma: vec![], beta: vec![],
                        save_name: String::new(),
                        add_name: String::new(),
                        new_input: vec![],
                        ln_output: vec![],
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::SetInput { name, new_input } => {
                    req_ops.push(OpDesc {
                        op_type: "set_input".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: vec![], b_q: vec![],
                        gelu_scale: default_gelu_scale(),
                        gamma: vec![], beta: vec![],
                        save_name: String::new(),
                        add_name: String::new(),
                        new_input: new_input.clone(),
                        ln_output: vec![],
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::LayerNorm { name, gamma, beta, ln_output } => {
                    req_ops.push(OpDesc {
                        op_type: "layernorm".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: vec![], b_q: vec![],
                        gelu_scale: default_gelu_scale(),
                        gamma: gamma.iter().map(|v| v.as_canonical_u32()).collect(),
                        beta: beta.iter().map(|v| v.as_canonical_u32()).collect(),
                        save_name: String::new(),
                        add_name: String::new(),
                        new_input: vec![],
                        ln_output: ln_output.clone(),
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::Save { name, save_name } => {
                    req_ops.push(OpDesc {
                        op_type: "save".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: vec![], b_q: vec![],
                        gelu_scale: default_gelu_scale(),
                        gamma: vec![], beta: vec![],
                        save_name: save_name.clone(),
                        add_name: String::new(),
                        new_input: vec![],
                        ln_output: vec![],
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::AddSaved { name, add_name } => {
                    req_ops.push(OpDesc {
                        op_type: "add_saved".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: vec![], b_q: vec![],
                        gelu_scale: default_gelu_scale(),
                        gamma: vec![], beta: vec![],
                        save_name: String::new(),
                        add_name: add_name.clone(),
                        new_input: vec![],
                        ln_output: vec![],
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::Gelu { name, gelu_scale, gelu_input_i16, gelu_output_i16 } => {
                    // Convert i16 vectors to M31 field elements for the lookup proof
                    let gelu_input_f: Vec<F> = gelu_input_i16.iter()
                        .map(|&v| crate::field::common::i16_to_field(v))
                        .collect();
                    let gelu_output_f: Vec<F> = gelu_output_i16.iter()
                        .map(|&v| crate::field::common::i16_to_field(v))
                        .collect();
                    req_ops.push(OpDesc {
                        op_type: "gelu".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: gelu_input_f.iter().map(|v| v.as_canonical_u32()).collect(),
                        b_q: gelu_output_f.iter().map(|v| v.as_canonical_u32()).collect(),
                        gelu_scale: *gelu_scale,
                        gamma: vec![], beta: vec![],
                        save_name: String::new(),
                        add_name: String::new(),
                        new_input: vec![],
                        ln_output: vec![],
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::Attention { name, num_heads, seq_len, d_head, exp_scale, k_values, v_values } => {
                    req_ops.push(OpDesc {
                        op_type: "attention".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: vec![], b_q: vec![],
                        gelu_scale: default_gelu_scale(),
                        gamma: vec![], beta: vec![],
                        save_name: String::new(),
                        add_name: String::new(),
                        new_input: vec![],
                        ln_output: vec![],
                        num_heads: *num_heads, seq_len: *seq_len, d_head: *d_head,
                        exp_scale: *exp_scale,
                        k_values: k_values.clone(), v_values: v_values.clone(),
                        llama_config: None, llama_weights: None,
                        qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::RmsNorm { name, gamma, rmsnorm_output } => {
                    req_ops.push(OpDesc {
                        op_type: "rmsnorm".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: vec![], b_q: vec![],
                        gelu_scale: default_gelu_scale(),
                        gamma: gamma.iter().map(|v| v.as_canonical_u32()).collect(),
                        beta: vec![],
                        save_name: String::new(),
                        add_name: String::new(),
                        new_input: vec![],
                        ln_output: rmsnorm_output.clone(),
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::Silu { name, silu_scale, silu_input_i16, silu_output_i16 } => {
                    let silu_input_f: Vec<F> = silu_input_i16.iter()
                        .map(|&v| crate::field::common::i16_to_field(v))
                        .collect();
                    let silu_output_f: Vec<F> = silu_output_i16.iter()
                        .map(|&v| crate::field::common::i16_to_field(v))
                        .collect();
                    req_ops.push(OpDesc {
                        op_type: "silu".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: silu_input_f.iter().map(|v| v.as_canonical_u32()).collect(),
                        b_q: silu_output_f.iter().map(|v| v.as_canonical_u32()).collect(),
                        gelu_scale: *silu_scale,
                        gamma: vec![], beta: vec![],
                        save_name: String::new(),
                        add_name: String::new(),
                        new_input: vec![],
                        ln_output: vec![],
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::SwiGlu { name, silu_scale, gate_i16, gate_silu_i16, up_values } => {
                    let gate_f: Vec<u32> = gate_i16.iter()
                        .map(|&v| crate::field::common::i16_to_field(v).as_canonical_u32())
                        .collect();
                    let gate_silu_f: Vec<u32> = gate_silu_i16.iter()
                        .map(|&v| crate::field::common::i16_to_field(v).as_canonical_u32())
                        .collect();
                    req_ops.push(OpDesc {
                        op_type: "swiglu".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: gate_f,
                        b_q: gate_silu_f,
                        gelu_scale: *silu_scale,
                        gamma: vec![],
                        beta: vec![],
                        save_name: String::new(),
                        add_name: String::new(),
                        new_input: vec![],
                        ln_output: up_values.clone(),
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                    llama_config: None, llama_weights: None,
                    qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::GqaAttention { name, num_q_heads, num_kv_heads, seq_len, d_head, exp_scale, k_values, v_values } => {
                    req_ops.push(OpDesc {
                        op_type: "gqa_attention".into(),
                        name: name.clone(),
                        m: *num_q_heads, n: *num_kv_heads,
                        w_q: vec![], b_q: vec![],
                        gelu_scale: default_gelu_scale(),
                        gamma: vec![], beta: vec![],
                        save_name: String::new(),
                        add_name: String::new(),
                        new_input: vec![],
                        ln_output: vec![],
                        num_heads: *num_q_heads, seq_len: *seq_len, d_head: *d_head,
                        exp_scale: *exp_scale,
                        k_values: k_values.clone(), v_values: v_values.clone(),
                        llama_config: None, llama_weights: None,
                        qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::LlamaLayer { name, config, weights } => {
                    req_ops.push(OpDesc {
                        op_type: "llama_layer".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: vec![], b_q: vec![],
                        gelu_scale: config.silu_scale,
                        gamma: vec![], beta: vec![],
                        save_name: String::new(), add_name: String::new(),
                        new_input: vec![], ln_output: vec![],
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                        llama_config: Some(config.clone()),
                        llama_weights: Some(weights.clone()),
                        qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::LlamaLayerRef { name, config, weight_names } => {
                    // Look up each of the 9 weight matrices from preloaded HashMap
                    let get_w = |wn: &str| -> Vec<F> {
                        let pw = weights.get(wn)
                            .unwrap_or_else(|| panic!("Unknown preloaded weight: {}", wn));
                        pw.w_f.clone()
                    };
                    let llama_weights = LlamaLayerWeightData {
                        norm1_gamma: get_w(&weight_names[0]),
                        w_q: get_w(&weight_names[1]),
                        w_k: get_w(&weight_names[2]),
                        w_v: get_w(&weight_names[3]),
                        w_o: get_w(&weight_names[4]),
                        norm2_gamma: get_w(&weight_names[5]),
                        w_gate: get_w(&weight_names[6]),
                        w_up: get_w(&weight_names[7]),
                        w_down: get_w(&weight_names[8]),
                    };
                    req_ops.push(OpDesc {
                        op_type: "llama_layer".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: vec![], b_q: vec![],
                        gelu_scale: config.silu_scale,
                        gamma: vec![], beta: vec![],
                        save_name: String::new(), add_name: String::new(),
                        new_input: vec![], ln_output: vec![],
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                        llama_config: Some(config.clone()),
                        llama_weights: Some(llama_weights),
                        qwen_config: None, qwen_weights: None,
                    });
                }
                ServerOp::QwenLayer { name, config, weights } => {
                    req_ops.push(OpDesc {
                        op_type: "qwen_layer".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: vec![], b_q: vec![],
                        gelu_scale: config.silu_scale,
                        gamma: vec![], beta: vec![],
                        save_name: String::new(), add_name: String::new(),
                        new_input: vec![], ln_output: vec![],
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                        llama_config: None, llama_weights: None,
                        qwen_config: Some(config.clone()),
                        qwen_weights: Some(weights.clone()),
                    });
                }
                ServerOp::QwenLayerRef { name, config, weight_names } => {
                    let get_w = |wn: &str| -> Vec<F> {
                        let pw = weights.get(wn)
                            .unwrap_or_else(|| panic!("Unknown preloaded weight: {}", wn));
                        pw.w_f.clone()
                    };
                    let qwen_weights = QwenLayerWeightData {
                        norm1_gamma: get_w(&weight_names[0]),
                        w_q: get_w(&weight_names[1]),
                        w_k: get_w(&weight_names[2]),
                        w_v: get_w(&weight_names[3]),
                        w_o: get_w(&weight_names[4]),
                        w_g_proj: get_w(&weight_names[5]),
                        norm2_gamma: get_w(&weight_names[6]),
                        w_gate: get_w(&weight_names[7]),
                        w_up: get_w(&weight_names[8]),
                        w_down: get_w(&weight_names[9]),
                    };
                    req_ops.push(OpDesc {
                        op_type: "qwen_layer".into(),
                        name: name.clone(),
                        m: 0, n: 0,
                        w_q: vec![], b_q: vec![],
                        gelu_scale: config.silu_scale,
                        gamma: vec![], beta: vec![],
                        save_name: String::new(), add_name: String::new(),
                        new_input: vec![], ln_output: vec![],
                        num_heads: 0, seq_len: 0, d_head: 0, exp_scale: 0,
                        k_values: vec![], v_values: vec![],
                        llama_config: None, llama_weights: None,
                        qwen_config: Some(config.clone()),
                        qwen_weights: Some(qwen_weights),
                    });
                }
            }
        }

        let req = ProveRequest {
            mode: "mlp".into(),
            input: input_u32,
            ops: req_ops,
            gpt2: None,
        };

        // Run the MLP prove, but capture output as string instead of printing
        let (response, _) = run_mlp_mode_returning(req, Some(&weights));

        let json = serde_json::to_string(&response)
            .expect("failed to serialize ProveResponse to JSON");
        writeln!(writer, "{}", json)
            .expect("IO error writing response — client likely disconnected");
        writer.flush()
            .expect("IO error flushing response — client likely disconnected");

        eprintln!(
            "  server req#{}: total {:.1}ms, valid={}",
            request_num,
            t_req.elapsed().as_secs_f64() * 1000.0,
            response.valid
        );
    }

    Ok(())
}

#[allow(dead_code)]
enum ServerOp {
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
        weight_names: [String; 9],
    },
    QwenLayer {
        name: String,
        config: QwenLayerConfig,
        weights: QwenLayerWeightData,
    },
    QwenLayerRef {
        name: String,
        config: QwenLayerConfig,
        weight_names: [String; 10],
    },
}
