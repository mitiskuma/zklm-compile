//! ZK proof for SiLU (Swish) activation using lookup tables.
//!
//! SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//! Used by Llama, Mistral, Qwen in their gated MLPs.
//!
//! Same lookup table approach as GELU: precompute over INT16 domain,
//! prove each (input, output) pair via LogUp lookup argument.

use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::proving::lookup::{self, LookupProof, LookupProofEF, LookupTable};
use crate::proving::sumcheck::Transcript;

type F = Mersenne31;

/// Proof that all SiLU lookups are valid.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SiluProof {
    pub lookup_proof: LookupProof,
}

/// Prove that y[i] = silu(x[i]) for all i, using the lookup table.
pub fn prove_silu(
    x: &[F],
    y: &[F],
    table: &LookupTable,
    transcript: &mut Transcript,
) -> SiluProof {
    let lookup_proof = lookup::prove_lookup(table, x, y, transcript);
    SiluProof { lookup_proof }
}

/// Verify a SiLU proof.
pub fn verify_silu(
    proof: &SiluProof,
    num_elements: usize,
    transcript: &mut Transcript,
) -> bool {
    verify_silu_with_data(proof, num_elements, None, transcript)
}

/// Verify SiLU with externally-provided inputs/outputs from the forward pass.
pub fn verify_silu_with_data(
    proof: &SiluProof,
    num_elements: usize,
    external_data: Option<(&[u32], &[u32])>,
    transcript: &mut Transcript,
) -> bool {
    lookup::verify_lookup_with_data(
        &proof.lookup_proof.table_commitment,
        &proof.lookup_proof,
        num_elements,
        external_data,
        transcript,
    )
}

/// EF SiLU proof (124-bit challenge soundness).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SiluProofEF {
    pub lookup_proof: LookupProofEF,
}

/// Prove SiLU with 124-bit EF challenges.
pub fn prove_silu_ef(
    x: &[F],
    y: &[F],
    table: &LookupTable,
    transcript: &mut Transcript,
) -> SiluProofEF {
    let lookup_proof = lookup::prove_lookup_ef(table, x, y, transcript);
    SiluProofEF { lookup_proof }
}

/// Verify EF SiLU proof.
#[allow(dead_code)]
pub fn verify_silu_ef(
    proof: &SiluProofEF,
    num_elements: usize,
    transcript: &mut Transcript,
) -> bool {
    verify_silu_ef_with_data(proof, num_elements, None, transcript)
}

/// Verify EF SiLU with externally-provided inputs/outputs.
pub fn verify_silu_ef_with_data(
    proof: &SiluProofEF,
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
    use crate::field::common::{i16_to_field, quantize_i16};
    use crate::proving::lookup::build_silu_table;
    use crate::field::m31_ops::*;
    use crate::proving::weight_commitment::WeightCommitment;
    use p3_field::{AbstractField, PrimeField32};

    /// Build a small SiLU table (256 entries, i8 range) for faster tests.
    fn build_small_silu_table(scale: i32) -> LookupTable {
        let s = scale as f64;
        let mut entries = Vec::with_capacity(256);
        for raw in 0u32..256 {
            let input_i16 = raw as i8 as i16;
            let x = input_i16 as f64 / s;
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            let y = x * sigmoid;
            let output_i16 = quantize_i16(y, s);
            entries.push((
                i16_to_field(input_i16).as_canonical_u32(),
                i16_to_field(output_i16).as_canonical_u32(),
            ));
        }
        LookupTable {
            name: "silu_small".to_string(),
            entries,
            commitment: WeightCommitment {
                root: [0u8; 32],
                num_weights: 256,
                log_height: 8,
                kind: crate::proving::weight_commitment::WeightDigestKind::Blake3Fast,
            },
        }
    }

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
    fn test_silu_table_known_values() {
        let table = build_small_silu_table(1000);
        // silu(0) = 0 * sigmoid(0) = 0
        let (_, out_raw) = table.entries[0];
        let out_val = from_field(F::from_canonical_u32(out_raw));
        assert_eq!(out_val, 0, "silu(0) should be 0");
    }

    #[test]
    fn test_silu_proof_small() {
        let table = build_small_silu_table(1000);
        let (inputs, outputs) = pick_lookups(&table, &[0, 1, 2, 3, 128, 200, 0, 1]);

        let mut p_transcript = Transcript::new(b"silu-test");
        let proof = prove_silu(&inputs, &outputs, &table, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"silu-test");
        assert!(verify_silu(&proof, inputs.len(), &mut v_transcript));
    }

    #[test]
    #[should_panic(expected = "LogUp identity failed")]
    fn test_silu_proof_tampered_output() {
        let table = build_small_silu_table(1000);
        let (inputs, mut outputs) = pick_lookups(&table, &[0, 1, 2, 3]);
        outputs[2] = outputs[2] + F::one();
        let mut p_transcript = Transcript::new(b"silu-bad");
        let _proof = prove_silu(&inputs, &outputs, &table, &mut p_transcript);
    }

    #[test]
    fn test_silu_proof_large_batch() {
        let table = build_small_silu_table(1000);
        let indices: Vec<usize> = (0..64).map(|i| i % 256).collect();
        let (inputs, outputs) = pick_lookups(&table, &indices);

        let mut p_transcript = Transcript::new(b"silu-large");
        let proof = prove_silu(&inputs, &outputs, &table, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"silu-large");
        assert!(verify_silu(&proof, inputs.len(), &mut v_transcript));
    }

    #[test]
    fn test_full_silu_table() {
        let table = build_silu_table(1000);
        assert_eq!(table.entries.len(), 65536);
    }
}
