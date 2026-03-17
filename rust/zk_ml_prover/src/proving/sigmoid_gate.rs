//! ZK proof for output gating: output = input ⊙ sigmoid(gate).
//!
//! Used by Qwen3.5 (GatedDeltaNet + full attention layers) for output gating:
//!   gated_out = o_proj_out * sigmoid(g_proj(x))
//!
//! Decomposes into:
//! 1. Sigmoid lookup proof: proves sigmoid_out[i] = sigmoid(gate[i])
//! 2. Hadamard product proof: proves output[i] = input[i] * sigmoid_out[i]

use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::proving::elementwise::{prove_hadamard, verify_hadamard, HadamardProof, prove_hadamard_ef, verify_hadamard_ef, HadamardProofEF};
use crate::proving::lookup::LookupTable;
use crate::proving::sigmoid::{prove_sigmoid, verify_sigmoid, SigmoidProof, prove_sigmoid_ef, SigmoidProofEF};
use crate::proving::sumcheck::Transcript;

type F = Mersenne31;

/// Proof for output gating: output = input ⊙ sigmoid(gate).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SigmoidGateProof {
    /// Proves sigmoid_out[i] = sigmoid(gate[i]) via lookup
    pub sigmoid_proof: SigmoidProof,
    /// Proves output[i] = input[i] * sigmoid_out[i] via Hadamard product sumcheck
    pub hadamard_proof: HadamardProof,
}

/// Prove the output gating: output = input ⊙ sigmoid(gate).
pub fn prove_sigmoid_gate(
    gate: &[F],
    gate_sigmoid: &[F],
    input: &[F],
    output: &[F],
    sigmoid_table: &LookupTable,
    transcript: &mut Transcript,
) -> SigmoidGateProof {
    let log_n = crate::field::common::log2_ceil(output.len());

    // 1. Sigmoid lookup proof: gate -> gate_sigmoid
    let sigmoid_proof = prove_sigmoid(gate, gate_sigmoid, sigmoid_table, transcript);

    // 2. Hadamard product proof: output = input ⊙ gate_sigmoid
    let point = transcript.squeeze_many(log_n);
    let hadamard_proof = prove_hadamard(input, gate_sigmoid, output, &point, transcript);

    SigmoidGateProof {
        sigmoid_proof,
        hadamard_proof,
    }
}

/// Verify a sigmoid gate proof.
pub fn verify_sigmoid_gate(
    proof: &SigmoidGateProof,
    num_elements: usize,
    transcript: &mut Transcript,
) -> bool {
    let log_n = crate::field::common::log2_ceil(num_elements);

    // 1. Verify sigmoid lookup
    if !verify_sigmoid(&proof.sigmoid_proof, num_elements, transcript) {
        eprintln!("SigmoidGate: sigmoid lookup verification failed");
        return false;
    }

    // 2. Verify Hadamard product
    let point = transcript.squeeze_many(log_n);
    if !verify_hadamard(&proof.hadamard_proof, &point, transcript) {
        eprintln!("SigmoidGate: Hadamard product verification failed");
        return false;
    }

    true
}

/// Extension field proof for sigmoid gating.
/// Both sigmoid lookup and Hadamard product use EF challenges (124-bit soundness).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SigmoidGateProofEF {
    /// Sigmoid lookup proof with EF challenges (124-bit soundness)
    pub sigmoid_proof: SigmoidProofEF,
    /// Hadamard product proof over extension field
    pub hadamard_proof: HadamardProofEF,
}

/// Prove the sigmoid gating with extension field challenges.
///
/// Both sigmoid lookup and Hadamard product use EF challenges for 124-bit soundness.
pub fn prove_sigmoid_gate_ef(
    gate: &[F],
    gate_sigmoid: &[F],
    input: &[F],
    output: &[F],
    sigmoid_table: &LookupTable,
    transcript: &mut Transcript,
) -> SigmoidGateProofEF {
    let log_n = crate::field::common::log2_ceil(output.len());

    // 1. Sigmoid lookup proof: gate -> gate_sigmoid (EF challenges)
    let sigmoid_proof = prove_sigmoid_ef(gate, gate_sigmoid, sigmoid_table, transcript);

    // 2. Hadamard product proof: output = input ⊙ gate_sigmoid (EF challenge point)
    let point = transcript.squeeze_ef_many(log_n);
    let hadamard_proof = prove_hadamard_ef(input, gate_sigmoid, output, &point, transcript);

    SigmoidGateProofEF {
        sigmoid_proof,
        hadamard_proof,
    }
}

/// Verify a sigmoid gate proof with extension field challenges.
#[allow(dead_code)]
pub fn verify_sigmoid_gate_ef(
    proof: &SigmoidGateProofEF,
    num_elements: usize,
    transcript: &mut Transcript,
) -> bool {
    verify_sigmoid_gate_ef_with_data(proof, num_elements, None, transcript)
}

/// Verify sigmoid gate with externally-provided sigmoid lookup data.
pub fn verify_sigmoid_gate_ef_with_data(
    proof: &SigmoidGateProofEF,
    num_elements: usize,
    sigmoid_data: Option<(&[u32], &[u32])>,
    transcript: &mut Transcript,
) -> bool {
    let log_n = crate::field::common::log2_ceil(num_elements);

    // 1. Verify sigmoid lookup (EF challenges — 124-bit soundness)
    let sig_ok = if let Some(data) = sigmoid_data {
        crate::proving::sigmoid::verify_sigmoid_ef_with_data(&proof.sigmoid_proof, num_elements, Some(data), transcript)
    } else {
        crate::proving::sigmoid::verify_sigmoid_ef(&proof.sigmoid_proof, num_elements, transcript)
    };
    if !sig_ok {
        eprintln!("SigmoidGate EF: sigmoid lookup verification failed");
        return false;
    }

    // 2. Verify Hadamard product (EF challenge point)
    let point = transcript.squeeze_ef_many(log_n);
    if !verify_hadamard_ef(&proof.hadamard_proof, &point, transcript) {
        eprintln!("SigmoidGate EF: Hadamard product verification failed");
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::common::{i16_to_field, quantize_i16};
    use crate::proving::lookup::LookupTable;
    use crate::field::m31_ops::*;
    use crate::proving::weight_commitment::WeightCommitment;
    use p3_field::{AbstractField, PrimeField32};

    fn build_small_sigmoid_table(scale: i32) -> LookupTable {
        let s = scale as f64;
        let mut entries = Vec::with_capacity(256);
        for raw in 0u32..256 {
            let input_i16 = raw as i8 as i16;
            let x = input_i16 as f64 / s;
            let y = 1.0 / (1.0 + (-x).exp());
            let output_i16 = quantize_i16(y, s);
            entries.push((
                i16_to_field(input_i16).as_canonical_u32(),
                i16_to_field(output_i16).as_canonical_u32(),
            ));
        }
        LookupTable {
            name: "sigmoid_small".to_string(),
            entries,
            commitment: WeightCommitment {
                root: [0u8; 32],
                num_weights: 256,
                log_height: 8,
            },
        }
    }

    fn lookup_sigmoid(v: F, table: &LookupTable) -> F {
        let key = v.as_canonical_u32();
        for &(inp, out) in &table.entries {
            if inp == key { return F::from_canonical_u32(out); }
        }
        F::zero()
    }

    #[test]
    fn test_sigmoid_gate_basic() {
        let table = build_small_sigmoid_table(10);
        let n = 8;

        // gate values in table range (i8 as field elements)
        let gate: Vec<F> = (0..n).map(|i| to_field((i as i64 % 20) - 10)).collect();
        let gate_sigmoid: Vec<F> = gate.iter().map(|&v| lookup_sigmoid(v, &table)).collect();
        // input values (arbitrary small field elements)
        let input: Vec<F> = (0..n).map(|i| F::from_canonical_u32(i as u32 + 1)).collect();
        // output = input ⊙ gate_sigmoid
        let output: Vec<F> = input.iter().zip(gate_sigmoid.iter()).map(|(&a, &b)| a * b).collect();

        let mut pt = Transcript::new(b"sigmoid-gate-test");
        let proof = prove_sigmoid_gate(&gate, &gate_sigmoid, &input, &output, &table, &mut pt);

        let mut vt = Transcript::new(b"sigmoid-gate-test");
        assert!(verify_sigmoid_gate(&proof, n, &mut vt));
    }

    #[test]
    fn test_sigmoid_gate_wrong_output() {
        let table = build_small_sigmoid_table(10);
        let n = 4;

        let gate: Vec<F> = (0..n).map(|i| to_field(i as i64)).collect();
        let gate_sigmoid: Vec<F> = gate.iter().map(|&v| lookup_sigmoid(v, &table)).collect();
        let input: Vec<F> = (0..n).map(|i| F::from_canonical_u32(i as u32 + 1)).collect();
        let mut output: Vec<F> = input.iter().zip(gate_sigmoid.iter()).map(|(&a, &b)| a * b).collect();
        output[0] = output[0] + F::one(); // tamper

        let mut pt = Transcript::new(b"sigmoid-gate-bad");
        let proof = prove_sigmoid_gate(&gate, &gate_sigmoid, &input, &output, &table, &mut pt);

        let mut vt = Transcript::new(b"sigmoid-gate-bad");
        assert!(!verify_sigmoid_gate(&proof, n, &mut vt), "Should reject tampered output");
    }

    #[test]
    fn test_sigmoid_gate_larger() {
        let table = build_small_sigmoid_table(10);
        let n = 32;

        let gate: Vec<F> = (0..n).map(|i| to_field((i as i64 % 20) - 10)).collect();
        let gate_sigmoid: Vec<F> = gate.iter().map(|&v| lookup_sigmoid(v, &table)).collect();
        let input: Vec<F> = (0..n).map(|i| F::from_canonical_u32(i as u32 * 3 + 1)).collect();
        let output: Vec<F> = input.iter().zip(gate_sigmoid.iter()).map(|(&a, &b)| a * b).collect();

        let mut pt = Transcript::new(b"sigmoid-gate-large");
        let proof = prove_sigmoid_gate(&gate, &gate_sigmoid, &input, &output, &table, &mut pt);

        let mut vt = Transcript::new(b"sigmoid-gate-large");
        assert!(verify_sigmoid_gate(&proof, n, &mut vt));
    }
}
