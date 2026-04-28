//! ZK proof for Gated MLP (SwiGLU) used by Llama/Mistral/Qwen.
//!
//! SwiGLU(x) = SiLU(gate_proj(x)) ⊙ up_proj(x)
//!
//! Decomposes into:
//! 1. SiLU lookup proof: proves silu_out[i] = silu(gate_out[i])
//! 2. Hadamard product proof: proves output[i] = silu_out[i] * up_out[i]
//!
//! The matmul proofs for gate_proj and up_proj are handled separately
//! (in the transformer layer orchestration).

use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::proving::elementwise::{prove_hadamard, verify_hadamard, HadamardProof, prove_hadamard_ef, verify_hadamard_ef, HadamardProofEF};
use crate::proving::lookup::LookupTable;
use crate::proving::silu::{prove_silu, verify_silu, SiluProof, prove_silu_ef, SiluProofEF};
use crate::proving::sumcheck::Transcript;

type F = Mersenne31;

/// Proof for the gating operation: output = SiLU(gate) ⊙ up.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwiGluProof {
    /// Proves silu_out[i] = silu(gate[i]) via lookup
    pub silu_proof: SiluProof,
    /// Proves output[i] = silu_out[i] * up[i] via Hadamard product sumcheck
    pub hadamard_proof: HadamardProof,
}

/// Prove the SwiGLU gating: output = SiLU(gate) ⊙ up.
///
/// Arguments:
/// - `gate`: output of gate_proj matmul (INT16 quantized field elements for SiLU lookup)
/// - `gate_silu`: SiLU(gate) values (from lookup table)
/// - `up`: output of up_proj matmul
/// - `output`: gate_silu ⊙ up (element-wise product)
/// - `silu_table`: precomputed SiLU lookup table
pub fn prove_swiglu(
    gate: &[F],
    gate_silu: &[F],
    up: &[F],
    output: &[F],
    silu_table: &LookupTable,
    transcript: &mut Transcript,
) -> SwiGluProof {
    let log_n = crate::field::common::log2_ceil(output.len());

    // 1. SiLU lookup proof: gate -> gate_silu
    let silu_proof = prove_silu(gate, gate_silu, silu_table, transcript);

    // 2. Hadamard product proof: output = gate_silu ⊙ up
    let point = transcript.squeeze_many(log_n);
    let hadamard_proof = prove_hadamard(gate_silu, up, output, &point, transcript);

    SwiGluProof {
        silu_proof,
        hadamard_proof,
    }
}

/// Verify a SwiGLU proof.
pub fn verify_swiglu(
    proof: &SwiGluProof,
    num_elements: usize,
    transcript: &mut Transcript,
) -> bool {
    let log_n = crate::field::common::log2_ceil(num_elements);

    // 1. Verify SiLU lookup
    if !verify_silu(&proof.silu_proof, num_elements, transcript) {
        eprintln!("SwiGLU: SiLU lookup verification failed");
        return false;
    }

    // 2. Verify Hadamard product
    let point = transcript.squeeze_many(log_n);
    if !verify_hadamard(&proof.hadamard_proof, &point, transcript) {
        eprintln!("SwiGLU: Hadamard product verification failed");
        return false;
    }

    true
}

/// Extension field proof for SwiGLU gating.
/// Both SiLU lookup and Hadamard product use EF challenges (124-bit soundness).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwiGluProofEF {
    /// SiLU lookup proof with EF challenges (124-bit soundness)
    pub silu_proof: SiluProofEF,
    /// Hadamard product proof over extension field
    pub hadamard_proof: HadamardProofEF,
}

/// Prove the SwiGLU gating with extension field challenges.
///
/// Both the SiLU lookup and Hadamard product use EF challenges for 124-bit soundness.
pub fn prove_swiglu_ef(
    gate: &[F],
    gate_silu: &[F],
    up: &[F],
    output: &[F],
    silu_table: &LookupTable,
    transcript: &mut Transcript,
) -> SwiGluProofEF {
    let log_n = crate::field::common::log2_ceil(output.len());

    // 1. SiLU lookup proof: gate -> gate_silu (EF challenges)
    let silu_proof = prove_silu_ef(gate, gate_silu, silu_table, transcript);

    // 2. Hadamard product proof: output = gate_silu ⊙ up (EF challenge point)
    let point = transcript.squeeze_ef_many(log_n);
    let hadamard_proof = prove_hadamard_ef(gate_silu, up, output, &point, transcript);

    SwiGluProofEF {
        silu_proof,
        hadamard_proof,
    }
}

/// Verify a SwiGLU proof with extension field challenges.
#[allow(dead_code)]
pub fn verify_swiglu_ef(
    proof: &SwiGluProofEF,
    num_elements: usize,
    transcript: &mut Transcript,
) -> bool {
    verify_swiglu_ef_with_data(proof, num_elements, None, transcript)
}

/// Verify SwiGLU with externally-provided SiLU lookup data from the forward pass.
/// When provided, uses trace values instead of proof.inputs/outputs for transcript.
pub fn verify_swiglu_ef_with_data(
    proof: &SwiGluProofEF,
    num_elements: usize,
    silu_data: Option<(&[u32], &[u32])>,
    transcript: &mut Transcript,
) -> bool {
    let log_n = crate::field::common::log2_ceil(num_elements);

    // 1. Verify SiLU lookup (EF challenges — 124-bit soundness)
    let silu_ok = if let Some(data) = silu_data {
        crate::proving::silu::verify_silu_ef_with_data(&proof.silu_proof, num_elements, Some(data), transcript)
    } else {
        crate::proving::silu::verify_silu_ef(&proof.silu_proof, num_elements, transcript)
    };
    if !silu_ok {
        eprintln!("SwiGLU EF: SiLU lookup verification failed");
        return false;
    }

    // 2. Verify Hadamard product (EF challenge point)
    let point = transcript.squeeze_ef_many(log_n);
    if !verify_hadamard_ef(&proof.hadamard_proof, &point, transcript) {
        eprintln!("SwiGLU EF: Hadamard product verification failed");
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
            commitment: WeightCommitment { root: [0u8; 32], num_weights: 256, log_height: 8, kind: crate::proving::weight_commitment::WeightDigestKind::Blake3Fast },
        }
    }

    fn lookup_silu(v: F, table: &LookupTable) -> F {
        let key = v.as_canonical_u32();
        for &(inp, out) in &table.entries {
            if inp == key { return F::from_canonical_u32(out); }
        }
        F::zero()
    }

    #[test]
    fn test_swiglu_basic() {
        let table = build_small_silu_table(10);
        let n = 8;

        // gate values in table range (i8 as field elements)
        let gate: Vec<F> = (0..n).map(|i| to_field((i as i64 % 20) - 10)).collect();
        let gate_silu: Vec<F> = gate.iter().map(|&v| lookup_silu(v, &table)).collect();
        // up values (arbitrary small field elements)
        let up: Vec<F> = (0..n).map(|i| F::from_canonical_u32(i as u32 + 1)).collect();
        // output = gate_silu ⊙ up
        let output: Vec<F> = gate_silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();

        let mut pt = Transcript::new(b"swiglu-test");
        let proof = prove_swiglu(&gate, &gate_silu, &up, &output, &table, &mut pt);

        let mut vt = Transcript::new(b"swiglu-test");
        assert!(verify_swiglu(&proof, n, &mut vt));
    }

    #[test]
    fn test_swiglu_wrong_output() {
        let table = build_small_silu_table(10);
        let n = 4;

        let gate: Vec<F> = (0..n).map(|i| to_field(i as i64)).collect();
        let gate_silu: Vec<F> = gate.iter().map(|&v| lookup_silu(v, &table)).collect();
        let up: Vec<F> = (0..n).map(|i| F::from_canonical_u32(i as u32 + 1)).collect();
        let mut output: Vec<F> = gate_silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();
        output[0] = output[0] + F::one(); // tamper

        let mut pt = Transcript::new(b"swiglu-bad");
        let proof = prove_swiglu(&gate, &gate_silu, &up, &output, &table, &mut pt);

        let mut vt = Transcript::new(b"swiglu-bad");
        assert!(!verify_swiglu(&proof, n, &mut vt), "Should reject tampered output");
    }

    #[test]
    fn test_swiglu_larger() {
        let table = build_small_silu_table(10);
        let n = 32;

        let gate: Vec<F> = (0..n).map(|i| to_field((i as i64 % 20) - 10)).collect();
        let gate_silu: Vec<F> = gate.iter().map(|&v| lookup_silu(v, &table)).collect();
        let up: Vec<F> = (0..n).map(|i| F::from_canonical_u32(i as u32 * 3 + 1)).collect();
        let output: Vec<F> = gate_silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();

        let mut pt = Transcript::new(b"swiglu-large");
        let proof = prove_swiglu(&gate, &gate_silu, &up, &output, &table, &mut pt);

        let mut vt = Transcript::new(b"swiglu-large");
        assert!(verify_swiglu(&proof, n, &mut vt));
    }
}
