//! ZK proof for GELU activation using lookup tables.
//!
//! GELU(x) = x * 0.5 * (1 + erf(x/sqrt(2))) is too complex for arithmetic circuits.
//! We use a lookup table over the INT16 domain and prove each (input, output) pair
//! exists in the table via the LogUp lookup argument.
//!
//! Approximation used: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

use p3_field::PrimeField32;
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::proving::lookup::{self, LookupTable, LookupProof, LookupProofEF};
use crate::proving::sumcheck::Transcript;
use crate::proving::weight_commitment::WeightCommitment;
use crate::field::common::{i16_to_field, quantize_i16};

type F = Mersenne31;

/// Compute GELU using the tanh approximation:
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
fn gelu_f64(x: f64) -> f64 {
    let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Build a GELU lookup table for INT16 inputs with given scale.
///
/// Input domain: INT16 [-32768, 32767], representing x/scale in fixed-point.
/// Output: gelu(input/scale) * scale, quantized to INT16.
pub fn build_gelu_table(scale: i32) -> LookupTable {
    let s = scale as f64;
    let mut entries = Vec::with_capacity(65536);
    for raw in 0u32..65536 {
        let input_i16 = raw as i16;
        let x = input_i16 as f64 / s;
        let y = gelu_f64(x);
        let output_i16 = quantize_i16(y, s);
        entries.push((
            i16_to_field(input_i16).as_canonical_u32(),
            i16_to_field(output_i16).as_canonical_u32(),
        ));
    }
    LookupTable {
        name: "gelu".to_string(),
        entries,
        commitment: WeightCommitment {
            root: [0u8; 32],
            num_weights: 65536,
            log_height: 16,
        },
    }
}

/// Proof that all GELU lookups are valid.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeluProof {
    pub lookup_proof: LookupProof,
}

/// Prove that y[i] = gelu(x[i]) for all i, using the lookup table.
pub fn prove_gelu(
    x: &[F],
    y: &[F],
    table: &LookupTable,
    transcript: &mut Transcript,
) -> GeluProof {
    let lookup_proof = lookup::prove_lookup(table, x, y, transcript);
    GeluProof { lookup_proof }
}

/// Verify a GELU proof.
pub fn verify_gelu(
    proof: &GeluProof,
    num_elements: usize,
    transcript: &mut Transcript,
) -> bool {
    lookup::verify_lookup(
        &proof.lookup_proof.table_commitment,
        &proof.lookup_proof,
        num_elements,
        transcript,
    )
}

/// EF GELU proof (124-bit challenge soundness).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeluProofEF {
    pub lookup_proof: LookupProofEF,
}

/// Prove GELU with 124-bit EF challenges.
#[allow(dead_code)]
pub fn prove_gelu_ef(
    x: &[F],
    y: &[F],
    table: &LookupTable,
    transcript: &mut Transcript,
) -> GeluProofEF {
    let lookup_proof = lookup::prove_lookup_ef(table, x, y, transcript);
    GeluProofEF { lookup_proof }
}

/// Verify EF GELU proof.
#[allow(dead_code)]
pub fn verify_gelu_ef(
    proof: &GeluProofEF,
    num_elements: usize,
    transcript: &mut Transcript,
) -> bool {
    lookup::verify_lookup_ef_with_data(
        &proof.lookup_proof.table_commitment,
        &proof.lookup_proof,
        num_elements,
        None,
        transcript,
    )
}

/// Verify EF GELU with externally-provided inputs/outputs.
#[allow(dead_code)]
pub fn verify_gelu_ef_with_data(
    proof: &GeluProofEF,
    num_elements: usize,
    external_data: Option<(&[u32], &[u32])>,
    transcript: &mut Transcript,
) -> bool {
    lookup::verify_lookup_ef_with_data(
        &proof.lookup_proof.table_commitment,
        &proof.lookup_proof,
        num_elements,
        external_data,
        transcript,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::m31_ops::*;
    use p3_field::AbstractField;

    /// Build a small GELU table (256 entries, i8 range) for faster tests.
    fn build_small_gelu_table(scale: i32) -> LookupTable {
        let s = scale as f64;
        let mut entries = Vec::with_capacity(256);
        for raw in 0u32..256 {
            let input_i16 = raw as i8 as i16; // -128..127
            let x = input_i16 as f64 / s;
            let y = gelu_f64(x);
            let output_i16 = quantize_i16(y, s);
            entries.push((
                i16_to_field(input_i16).as_canonical_u32(),
                i16_to_field(output_i16).as_canonical_u32(),
            ));
        }
        LookupTable {
            name: "gelu_small".to_string(),
            entries,
            commitment: WeightCommitment {
                root: [0u8; 32],
                num_weights: 256,
                log_height: 8,
            },
        }
    }

    /// Pick valid lookups from a table at given indices.
    fn pick_lookups(table: &LookupTable, indices: &[usize]) -> (Vec<F>, Vec<F>) {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for &idx in indices {
            let (inp, out) = table.entries[idx];
            inputs.push(F::from_canonical_u32(inp));
            outputs.push(F::from_canonical_u32(out));
        }
        (inputs, outputs)
    }

    #[test]
    fn test_gelu_table_known_values() {
        let scale = 1000;
        let table = build_small_gelu_table(scale);
        assert_eq!(table.entries.len(), 256);

        // gelu(0) = 0
        // Index 0 in i8 range maps to input_i16 = 0
        let (_, out_raw) = table.entries[0];
        let out_val = from_field(F::from_canonical_u32(out_raw));
        assert_eq!(out_val, 0, "gelu(0) should be 0");

        // gelu(1.0) ≈ 0.8412 => output ≈ 841
        // input_i16 = 1000 is outside i8 range, so use full table
        let full_table = build_gelu_table(scale);
        // Index for input_i16=1000: raw = 1000 (positive, direct)
        let (_, out_raw) = full_table.entries[1000];
        let out_val = from_field(F::from_canonical_u32(out_raw));
        assert!((out_val - 841).abs() <= 1, "gelu(1.0) ≈ 0.841, got {}", out_val);

        // gelu(-1.0) ≈ -0.1588 => output ≈ -159
        // input_i16 = -1000 => raw = 65536 - 1000 = 64536
        let (_, out_raw) = full_table.entries[64536];
        let out_val = from_field(F::from_canonical_u32(out_raw));
        assert!((out_val - (-159)).abs() <= 1, "gelu(-1.0) ≈ -0.159, got {}", out_val);
    }

    #[test]
    fn test_gelu_proof_small() {
        let table = build_small_gelu_table(1000);
        // Pick 8 valid lookups
        let (inputs, outputs) = pick_lookups(&table, &[0, 1, 2, 3, 128, 200, 0, 1]);

        let mut p_transcript = Transcript::new(b"gelu-test");
        let proof = prove_gelu(&inputs, &outputs, &table, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"gelu-test");
        assert!(verify_gelu(&proof, inputs.len(), &mut v_transcript));
    }

    #[test]
    #[should_panic(expected = "LogUp identity failed")]
    fn test_gelu_proof_tampered_output() {
        let table = build_small_gelu_table(1000);
        let (inputs, mut outputs) = pick_lookups(&table, &[0, 1, 2, 3]);

        // Tamper with one output
        outputs[2] = outputs[2] + F::one();

        let mut p_transcript = Transcript::new(b"gelu-bad");
        let _proof = prove_gelu(&inputs, &outputs, &table, &mut p_transcript);
    }

    #[test]
    fn test_gelu_proof_large_batch() {
        let table = build_small_gelu_table(1000);
        // 64 elements — indices cycling through table
        let indices: Vec<usize> = (0..64).map(|i| i % 256).collect();
        let (inputs, outputs) = pick_lookups(&table, &indices);

        let mut p_transcript = Transcript::new(b"gelu-large");
        let proof = prove_gelu(&inputs, &outputs, &table, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"gelu-large");
        assert!(verify_gelu(&proof, inputs.len(), &mut v_transcript));
    }

    #[test]
    fn test_gelu_verification_fails_with_tampered_proof() {
        let table = build_small_gelu_table(1000);
        let (inputs, outputs) = pick_lookups(&table, &[0, 1, 2, 3]);

        let mut p_transcript = Transcript::new(b"gelu-tamper");
        let mut proof = prove_gelu(&inputs, &outputs, &table, &mut p_transcript);

        // Tamper with the final evaluation
        proof.lookup_proof.finals.1 = (proof.lookup_proof.finals.1 + 1) % ((1u32 << 31) - 1);

        let mut v_transcript = Transcript::new(b"gelu-tamper");
        assert!(!verify_gelu(&proof, inputs.len(), &mut v_transcript));
    }

    // ===== Extension field GELU tests =====

    #[test]
    fn test_gelu_proof_ef_small() {
        let table = build_small_gelu_table(1000);
        let (inputs, outputs) = pick_lookups(&table, &[0, 1, 2, 3, 128, 200, 0, 1]);

        let mut p_transcript = Transcript::new(b"gelu-ef-test");
        let proof = prove_gelu_ef(&inputs, &outputs, &table, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"gelu-ef-test");
        assert!(verify_gelu_ef(&proof, inputs.len(), &mut v_transcript));
    }

    #[test]
    #[should_panic(expected = "LogUp identity failed")]
    fn test_gelu_proof_ef_tampered() {
        let table = build_small_gelu_table(1000);
        let (inputs, mut outputs) = pick_lookups(&table, &[0, 1, 2, 3]);
        outputs[2] = outputs[2] + F::one();

        let mut p_transcript = Transcript::new(b"gelu-ef-bad");
        let _proof = prove_gelu_ef(&inputs, &outputs, &table, &mut p_transcript);
    }

    #[test]
    fn test_gelu_proof_ef_large_batch() {
        let table = build_small_gelu_table(1000);
        let indices: Vec<usize> = (0..64).map(|i| i % 256).collect();
        let (inputs, outputs) = pick_lookups(&table, &indices);

        let mut p_transcript = Transcript::new(b"gelu-ef-large");
        let proof = prove_gelu_ef(&inputs, &outputs, &table, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"gelu-ef-large");
        assert!(verify_gelu_ef(&proof, inputs.len(), &mut v_transcript));
    }
}
