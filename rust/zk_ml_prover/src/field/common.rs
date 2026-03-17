//! Shared utilities for the ZK-ML prover.
//!
//! Consolidates functions that were duplicated across 10+ modules.

use p3_field::{AbstractField, Field};
use p3_mersenne_31::Mersenne31;
use sha2::{Digest, Sha256};

use crate::field::m31_ops::to_field;

pub type F = Mersenne31;

// ===== Math utilities =====

/// Ceiling of log2(n), minimum 1.
pub fn log2_ceil(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

/// Compute eq(a, b) = Π_i (a_i · b_i + (1-a_i)(1-b_i))
pub fn compute_eq_at_point(a: &[F], b: &[F]) -> F {
    let mut result = F::one();
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        result *= ai * bi + (F::one() - ai) * (F::one() - bi);
    }
    result
}

/// Modular square root in M31 (p = 2^31 - 1 ≡ 3 mod 4).
/// sqrt(a) = a^((p+1)/4) = a^(2^29).
pub fn mod_sqrt_m31(a: F) -> F {
    let mut result = a;
    for _ in 0..29 {
        result = result * result;
    }
    result
}

/// Check if `a` is a quadratic residue in M31 (p ≡ 3 mod 4).
/// Uses Euler's criterion: a^((p-1)/2) == 1.
pub fn is_qr_m31(a: F) -> bool {
    if a == F::zero() {
        return true;
    }
    let mut r = a;
    for _ in 0..30 {
        r = r * r;
    }
    r * a.inverse() == F::one()
}

// ===== Quantization =====

/// Convert a signed i16 value to M31 field element.
pub fn i16_to_field(x: i16) -> F {
    to_field(x as i64)
}

/// Quantize a f64 value to INT16 with saturation.
pub fn quantize_i16(x: f64, scale: f64) -> i16 {
    let v = (x * scale).round();
    if v > 32767.0 {
        32767
    } else if v < -32768.0 {
        -32768
    } else {
        v as i16
    }
}

// ===== Hashing =====

/// SHA-256 hash of field element values.
/// Format: len_u32_le || v[0]_u32_le || v[1]_u32_le || ...
pub fn hash_values(values: &[u32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update((values.len() as u32).to_le_bytes());
    for &v in values {
        h.update(v.to_le_bytes());
    }
    h.finalize().into()
}

// ===== Extension field utilities =====

use crate::proving::sumcheck::EF;

/// Compute eq(a, b) = Π_i (a_i · b_i + (1-a_i)(1-b_i)) in extension field.
pub fn compute_eq_at_point_ef(a: &[EF], b: &[EF]) -> EF {
    let mut result = EF::one();
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        result *= ai * bi + (EF::one() - ai) * (EF::one() - bi);
    }
    result
}
