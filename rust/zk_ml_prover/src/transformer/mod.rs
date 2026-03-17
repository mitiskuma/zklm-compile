//! ZK proofs for transformer layers over M31.
//!
//! Supports two architectures:
//!
//! **GPT-2 style** (pre-norm, simplified attention):
//!   h = x + attention(layernorm1(x))
//!   output = h + mlp(layernorm2(h))
//!   where mlp(x) = gelu(x @ W1 + b1) @ W2 + b2
//!
//! **Llama/Mistral/Qwen style** (pre-norm, GQA + SwiGLU):
//!   h = x + gqa(rmsnorm(x))
//!   output = h + swiglu_mlp(rmsnorm(h))
//!   where swiglu_mlp(x) = (silu(gate_proj(x)) ⊙ up_proj(x)) @ down_proj
//!
//! Use `ModelConfig` to select architecture and dimensions.

pub mod gpt2;
pub mod llama;
pub mod qwen;

// Re-export everything for backward compatibility
pub use gpt2::*;
pub use llama::*;
pub use qwen::*;

use p3_field::{AbstractField, Field, PrimeField32};
use p3_mersenne_31::Mersenne31;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::proving::lookup::LookupTable;
use crate::field::common::{mod_sqrt_m31, is_qr_m31, i16_to_field};
use crate::proving::weight_commitment::{commit_weights_fast, WeightCommitment};

type F = Mersenne31;

// reduce_u64_m31 moved to m31_ops.rs

/// Fast M31 dot product: work on raw u32 arrays via transmute, accumulate in u64
/// with periodic reduction. Mersenne31 is repr(transparent) u32.
/// Batches of 3 products per u64 accumulator to avoid overflow.
#[inline(always)]
pub(crate) fn m31_dot(a: &[F], b: &[F]) -> F {
    let n = a.len();
    // Safety: Mersenne31 is #[repr(transparent)] around u32
    let a_raw: &[u32] = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u32, n) };
    let b_raw: &[u32] = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u32, n) };
    m31_dot_u32(a_raw, b_raw)
}

/// Dot product on raw u32 slices, returning M31.
/// Uses NEON SIMD on aarch64: vmull_u32 processes 2 elements per instruction,
/// batching 6 products per reduction (3 × 2-wide vmull/vmlal).
#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(crate) fn m31_dot_u32(a: &[u32], b: &[u32]) -> F {
    use std::arch::aarch64::*;
    const P: u64 = (1u64 << 31) - 1;
    let n = a.len();

    unsafe {
        let p_vec = vdupq_n_u64(P);
        let mut total = vdupq_n_u64(0);
        let chunks = n / 6;
        let mut j = 0;

        for _ in 0..chunks {
            // 3 × vmull/vmlal = 6 products in 2 lanes
            // Max per lane: 3 × (2^31-1)^2 < 3 × 2^62 < 2^64 ✓
            let a0 = vld1_u32(a.as_ptr().add(j));
            let b0 = vld1_u32(b.as_ptr().add(j));
            let mut acc = vmull_u32(a0, b0);

            let a1 = vld1_u32(a.as_ptr().add(j + 2));
            let b1 = vld1_u32(b.as_ptr().add(j + 2));
            acc = vmlal_u32(acc, a1, b1);

            let a2 = vld1_u32(a.as_ptr().add(j + 4));
            let b2 = vld1_u32(b.as_ptr().add(j + 4));
            acc = vmlal_u32(acc, a2, b2);

            // Partial reduction: (acc & P) + (acc >> 31)
            let lo = vandq_u64(acc, p_vec);
            let hi = vshrq_n_u64::<31>(acc);
            total = vaddq_u64(total, vaddq_u64(lo, hi));
            j += 6;
        }

        // Extract both lanes
        let mut v = vgetq_lane_u64::<0>(total) + vgetq_lane_u64::<1>(total);

        // Scalar remainder
        while j < n {
            let prod = a[j] as u64 * b[j] as u64;
            v += (prod & P) + (prod >> 31);
            j += 1;
        }

        // Final reductions
        v = (v & P) + (v >> 31);
        v = (v & P) + (v >> 31);
        if v >= P { v -= P; }
        F::from_canonical_u32(v as u32)
    }
}

/// Dot product on raw u32 slices, returning M31 (non-aarch64 fallback).
/// Accumulates in u64, reduces every 3 products (3 × 2^62 < 2^64).
#[inline(always)]
#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn m31_dot_u32(a: &[u32], b: &[u32]) -> F {
    const P: u64 = (1u64 << 31) - 1;
    let n = a.len();
    let mut total = 0u64;
    let mut acc = 0u64;
    let chunks = n / 3;
    let mut j = 0;
    for _ in 0..chunks {
        acc += a[j] as u64 * b[j] as u64;
        acc += a[j+1] as u64 * b[j+1] as u64;
        acc += a[j+2] as u64 * b[j+2] as u64;
        total += (acc & P) + (acc >> 31);
        acc = 0;
        j += 3;
    }
    while j < n {
        acc += a[j] as u64 * b[j] as u64;
        j += 1;
    }
    if acc > 0 {
        total += (acc & P) + (acc >> 31);
    }
    let mut v = total;
    v = (v & P) + (v >> 31);
    v = (v & P) + (v >> 31);
    if v >= P { v -= P; }
    F::from_canonical_u32(v as u32)
}

pub(crate) fn matmul_forward(w: &[F], x: &[F], m: usize, n: usize, bias: Option<&[F]>) -> Vec<F> {
    if m * n >= 4096 {
        // Parallel rows for large matrices
        (0..m).into_par_iter().map(|i| {
            let row = &w[i * n..(i + 1) * n];
            let mut acc = m31_dot(row, x);
            if let Some(b) = bias {
                acc += b[i];
            }
            acc
        }).collect()
    } else {
        let mut y = Vec::with_capacity(m);
        for i in 0..m {
            let row = &w[i * n..(i + 1) * n];
            let mut acc = m31_dot(row, x);
            if let Some(b) = bias {
                acc += b[i];
            }
            y.push(acc);
        }
        y
    }
}

/// Look up GELU value from table. The field element encodes an INT16 value.
#[allow(dead_code)]
pub(crate) fn lookup_gelu(v: F, table: &LookupTable) -> F {
    let key = v.as_canonical_u32();
    for &(inp, out) in &table.entries {
        if inp == key {
            return F::from_canonical_u32(out);
        }
    }
    panic!("lookup_gelu: value {} not found in GELU table", key);
}

/// Compute r such that r^2 * sum_sq = d (mod p).
/// r = sqrt(d / sum_sq) in M31.
pub(crate) fn compute_r(sum_sq: F, d: usize) -> F {
    let d_field = F::from_canonical_u32(d as u32);
    let target = d_field * sum_sq.inverse();
    assert!(is_qr_m31(target), "compute_r: d/sum_sq is not a quadratic residue in M31");
    mod_sqrt_m31(target)
}

/// Model architecture configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub d_model: usize,
    pub d_ff: usize,       // intermediate size (Llama: 11008 for 7B)
    pub num_q_heads: usize, // query heads
    pub num_kv_heads: usize, // key/value heads (GQA; =num_q_heads for MHA)
    pub d_head: usize,      // d_model / num_q_heads
    pub n_layers: usize,
    pub vocab_size: usize,
    pub norm_type: NormType,
    pub activation: ActivationType,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum NormType {
    LayerNorm,
    RMSNorm,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    GELU,
    SwiGLU,
}

/// Simple RMSNorm forward: y = gamma * x / rms(x)
/// Uses QR perturbation on x[0] when d/sum_sq is not a quadratic residue in M31.
/// Returns (output, perturbed_input) — the input may differ from x at index 0.
pub fn rmsnorm_forward(x: &[F], gamma: &[F]) -> (Vec<F>, Vec<F>) {
    let d = x.len();
    let d_field = F::from_canonical_u32(d as u32);

    // Try delta=0 first (no perturbation, no clone needed for check)
    let sum_sq: F = x.iter().map(|&v| v * v).sum();
    if sum_sq != F::zero() {
        let target = d_field * sum_sq.inverse();
        if is_qr_m31(target) {
            let r = mod_sqrt_m31(target);
            let out = x.iter().zip(gamma.iter()).map(|(&xi, &gi)| gi * xi * r).collect();
            return (out, x.to_vec());
        }
    }

    // Need perturbation — compute sum_sq incrementally
    let base_sum_sq = sum_sq;
    let x0 = x[0];
    for delta in 1i32..100 {
        for sign in &[1i32, -1i32] {
            let perturbation = delta * sign;
            let adj = if perturbation > 0 {
                F::from_canonical_u32(perturbation as u32)
            } else {
                F::zero() - F::from_canonical_u32((-perturbation) as u32)
            };
            let new_x0 = x0 + adj;
            // sum_sq_new = sum_sq_old - x0^2 + new_x0^2
            let sum_sq_new = base_sum_sq - x0 * x0 + new_x0 * new_x0;
            if sum_sq_new == F::zero() { continue; }
            let target = d_field * sum_sq_new.inverse();

            if is_qr_m31(target) {
                let r = mod_sqrt_m31(target);
                eprintln!("  rmsnorm: perturbed x[0] by {} to get QR", perturbation);
                let mut x_p = x.to_vec();
                x_p[0] = new_x0;
                let out = x_p.iter().zip(gamma.iter()).map(|(&xi, &gi)| gi * xi * r).collect();
                return (out, x_p);
            }
        }
    }
    panic!("rmsnorm_forward: could not find QR perturbation in 200 attempts");
}

/// Requantize M31 field elements to i16 range for lookup table compatibility.
/// Converts M31 → signed → i16 (by finding max abs and scaling to fit i16).
/// Returns field elements in i16_to_field encoding.
pub(crate) fn requantize_to_i16_field(values: &[F], _table: &LookupTable) -> Vec<F> {
    const M31: u64 = (1u64 << 31) - 1;
    const HALF: u64 = M31 / 2;

    // Convert to signed integers and find max absolute value in single pass
    let mut max_abs: u64 = 1;
    let signed: Vec<i64> = values.iter().map(|v| {
        let u = v.as_canonical_u32() as u64;
        let s = if u > HALF { u as i64 - M31 as i64 } else { u as i64 };
        let abs = s.unsigned_abs();
        if abs > max_abs { max_abs = abs; }
        s
    }).collect();

    // Scale to fit in i16 range using integer division with rounding
    // scaled = v * 32767 / max_abs (with rounding)
    signed.iter().map(|&v| {
        let scaled = if max_abs <= 32767 {
            v // already fits
        } else {
            // Integer division with rounding: (v * 32767 + sign(v) * max_abs/2) / max_abs
            let half = max_abs as i64 / 2;
            if v >= 0 {
                (v * 32767 + half) / max_abs as i64
            } else {
                (v * 32767 - half) / max_abs as i64
            }
        };
        let clamped = scaled.max(-32768).min(32767) as i16;
        i16_to_field(clamped)
    }).collect()
}

/// Look up a value from a table.
#[allow(dead_code)]
pub fn lookup_value(v: F, table: &LookupTable) -> F {
    let key = v.as_canonical_u32();
    for &(inp, out) in &table.entries {
        if inp == key {
            return F::from_canonical_u32(out);
        }
    }
    panic!("lookup_value: value {} not found in table '{}'", key, table.name);
}

/// Fast lookup using a pre-built HashMap index.
#[inline(always)]
pub(crate) fn lookup_fast(key: u32, index: &std::collections::HashMap<u32, u32>) -> F {
    F::from_canonical_u32(*index.get(&key).expect("lookup_fast: key not found"))
}

/// Build a HashMap index from a LookupTable for O(1) lookups.
pub fn build_table_index(table: &LookupTable) -> std::collections::HashMap<u32, u32> {
    table.entries.iter().copied().collect()
}

/// Type alias for pre-built table index.
pub type TableIndex = std::collections::HashMap<u32, u32>;

/// Commit a weight matrix with padding.
pub fn commit_weight_matrix(w: &[F], _rows: usize, _cols: usize) -> WeightCommitment {
    // Commit on raw unpadded weights — no need to allocate padded array.
    // The commitment is a binding hash; padding zeros don't add security.
    commit_weights_fast(w)
}
