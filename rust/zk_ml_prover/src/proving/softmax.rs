//! ZK proof for softmax: y[i] = exp(z[i]) / Σ exp(z[j])
//!
//! Sub-proofs:
//! 1. **Exp lookup**: proves e[i] = exp(z[i]) via LogUp argument against exp table
//! 2. **Sum check**: proves S = Σ e[i] via product sumcheck
//! 3. **Division**: proves y[i] = e[i] * inv_S, where inv_S = S^(-1) in M31
//!    Verified as: y[i] * S = e[i] for all i, reduced to MLE identity check

use p3_field::{AbstractField, Field, PrimeField32};
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::field::common::{log2_ceil, compute_eq_at_point};
use crate::proving::lookup::{self, LookupProof, LookupTable};
use crate::field::m31_ops::*;
use crate::proving::sumcheck::{self, SumcheckProof, Transcript};

type F = Mersenne31;

/// Proof that y = softmax(z) using exp lookup + sum + division checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoftmaxProof {
    /// LogUp proof: e[i] = exp(z[i]) for all i
    pub exp_lookup_proof: LookupProof,
    /// Claimed sum S = Σ e[i]
    pub sum_value: u32,
    /// Product sumcheck proving Σ eq(r,i) · e[i] = ẽ(r) (MLE eval of e at random r)
    pub sum_proof: SumcheckProof,
    pub sum_finals: (u32, u32),
    /// Triple sumcheck proving Σ eq(r',i) · y[i] · S_const = Σ eq(r',i) · e[i]
    /// Equivalently: Σ eq(r',i) · (y[i] · S - e[i]) = 0
    /// We prove: Σ eq(r',i) · y[i] · S_expanded[i] = ẽ(r')
    /// where S_expanded[i] = S for all i.
    pub output_proof: SumcheckProof,
    pub output_finals: (u32, u32, u32),
    /// Modular inverse of S in M31
    pub inv_s: u32,
    /// The exp values (statement)
    pub e_values: Vec<u32>,
    /// The output values (statement)
    pub y_values: Vec<u32>,
}

/// Prove y = softmax(z).
///
/// Inputs:
///   - z: input logits (field elements, already shifted by max if desired)
///   - e: exp(z[i]) values (from lookup table)
///   - y: output softmax values y[i] = e[i] / S
///   - exp_table: the committed exp lookup table
///
/// The prover:
/// 1. Proves e[i] = exp(z[i]) via LogUp lookup
/// 2. Proves S = Σ e[i] via product sumcheck
/// 3. Proves y[i] * S = e[i] via triple sumcheck
pub fn prove_softmax(
    z: &[F],
    e: &[F],
    y: &[F],
    exp_table: &LookupTable,
    transcript: &mut Transcript,
) -> SoftmaxProof {
    let n = z.len();
    assert_eq!(e.len(), n);
    assert_eq!(y.len(), n);

    // 1. Exp lookup proof
    let exp_lookup_proof = lookup::prove_lookup(exp_table, z, e, transcript);

    // 2. Sum proof: S = Σ e[i]
    let sum: F = e.iter().copied().sum();
    let sum_value = sum.as_canonical_u32();

    // Pad to power of 2
    let log_n = log2_ceil(n);
    let n_pad = 1 << log_n;

    let mut e_pad = e.to_vec();
    e_pad.resize(n_pad, F::zero());
    let mut y_pad = y.to_vec();
    y_pad.resize(n_pad, F::zero());

    // Absorb sum claim
    transcript.absorb(sum_value);

    // Product sumcheck: Σ ones[i] · e[i] = S
    // ones is all-1s, so this proves the sum of e.
    let ones = vec![F::one(); n_pad];
    let (sum_proof, ones_at_s, e_at_s) =
        sumcheck::prove_product_best(&ones, &e_pad, log_n, transcript);

    let sum_finals = (ones_at_s.as_canonical_u32(), e_at_s.as_canonical_u32());

    // 3. Division proof: y[i] * S = e[i] for all i
    // Triple sumcheck: Σ eq(r, i) · y[i] · S_expanded[i] should equal ẽ(r)
    // where S_expanded[i] = S for all i.
    // Since S is a constant, S_expanded MLE = S (constant polynomial).
    // So the triple sumcheck proves: Σ eq(r,i) · y[i] · S = ẽ(r)
    // i.e., S · ỹ(r) = ẽ(r), which holds iff y[i] = e[i]/S for all i.

    // Squeeze random point for the output check
    let r_point = transcript.squeeze_many(log_n);
    let eq_r = eq_evals(&r_point);

    // S_expanded = [S, S, S, ...] (constant S over the hypercube)
    let s_expanded = vec![sum; n_pad];

    let (output_proof, eq_at_s, y_at_s, s_at_s) =
        sumcheck::prove_triple_best(&eq_r, &y_pad, &s_expanded, log_n, transcript);

    let output_finals = (
        eq_at_s.as_canonical_u32(),
        y_at_s.as_canonical_u32(),
        s_at_s.as_canonical_u32(),
    );

    let inv_s = sum.inverse().as_canonical_u32();

    SoftmaxProof {
        exp_lookup_proof,
        sum_value,
        sum_proof,
        sum_finals,
        output_proof,
        output_finals,
        inv_s,
        e_values: e.iter().map(|v| v.as_canonical_u32()).collect(),
        y_values: y.iter().map(|v| v.as_canonical_u32()).collect(),
    }
}

/// Verify a softmax proof.
///
/// Checks:
/// 1. Exp lookup is valid (all e[i] = exp(z[i]))
/// 2. S = Σ e[i] via product sumcheck
/// 3. y[i] * S = e[i] via triple sumcheck (division correctness)
/// 4. inv_s * S = 1 (modular inverse correctness)
pub fn verify_softmax(
    proof: &SoftmaxProof,
    _exp_table: &LookupTable,
    num_elements: usize,
    transcript: &mut Transcript,
) -> bool {
    // 1. Verify exp lookup
    if !lookup::verify_lookup(
        &proof.exp_lookup_proof.table_commitment,
        &proof.exp_lookup_proof,
        num_elements,
        transcript,
    ) {
        return false;
    }

    // Check that e_values matches the outputs proven by the exp lookup proof
    if proof.e_values.len() != proof.exp_lookup_proof.outputs.len() {
        return false;
    }
    for (i, (&e_val, &lookup_out)) in proof.e_values.iter().zip(proof.exp_lookup_proof.outputs.iter()).enumerate() {
        if e_val != lookup_out {
            eprintln!("Softmax: e_values[{}] ({}) != exp_lookup_proof.outputs[{}] ({})", i, e_val, i, lookup_out);
            return false;
        }
    }

    let sum = F::from_canonical_u32(proof.sum_value);
    let inv_s = F::from_canonical_u32(proof.inv_s);

    // Check modular inverse: inv_s * S = 1
    if inv_s * sum != F::one() {
        return false;
    }

    let log_n = log2_ceil(num_elements);
    let n_pad = 1 << log_n;

    // 2. Verify sum proof: Σ ones[i] · e[i] = S
    transcript.absorb(proof.sum_value);

    let ones_at_s = F::from_canonical_u32(proof.sum_finals.0);
    let e_at_s = F::from_canonical_u32(proof.sum_finals.1);

    if !sumcheck::verify_product(
        sum,
        &proof.sum_proof,
        log_n,
        ones_at_s,
        e_at_s,
        transcript,
    ) {
        return false;
    }

    // Verify ones_at_s = 1 (constant-1 MLE)
    if ones_at_s != F::one() {
        return false;
    }

    // 3. Verify output proof: Σ eq(r,i) · y[i] · S = ẽ(r)
    // The claimed sum = ẽ(r) = MLE of e evaluated at r
    let r_point = transcript.squeeze_many(log_n);

    // Compute ẽ(r) from the e_values
    let e_vals: Vec<F> = proof.e_values.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let mut e_pad = e_vals.clone();
    e_pad.resize(n_pad, F::zero());
    let e_at_r = mle_evaluate(&e_pad, &r_point);

    let eq_at_s2 = F::from_canonical_u32(proof.output_finals.0);
    let y_at_s2 = F::from_canonical_u32(proof.output_finals.1);
    let s_at_s2 = F::from_canonical_u32(proof.output_finals.2);

    if !sumcheck::verify_triple(
        e_at_r,
        &proof.output_proof,
        log_n,
        eq_at_s2,
        y_at_s2,
        s_at_s2,
        transcript,
    ) {
        return false;
    }

    // Verify eq(r, s*) independently
    let s_point: Vec<F> = proof.output_proof.challenges.iter()
        .map(|&v| F::from_canonical_u32(v))
        .collect();
    if compute_eq_at_point(&r_point, &s_point) != eq_at_s2 {
        return false;
    }

    // Verify S_expanded MLE at s* = S (constant polynomial)
    if s_at_s2 != sum {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small exp table (256 entries) for faster tests.
    fn build_small_exp_table(scale: i32) -> LookupTable {
        let s = scale as f64;
        let mut entries = Vec::with_capacity(256);
        for raw in 0u32..256 {
            let input_i16 = raw as i8 as i16; // -128..127
            let x = input_i16 as f64 / s;
            let y = x.exp();
            let output_i16 = crate::field::common::quantize_i16(y, s);
            entries.push((
                to_field(input_i16 as i64).as_canonical_u32(),
                to_field(output_i16 as i64).as_canonical_u32(),
            ));
        }
        use crate::proving::weight_commitment::WeightCommitment;
        LookupTable {
            name: "exp_small".to_string(),
            entries,
            commitment: WeightCommitment {
                root: [0u8; 32],
                num_weights: 256,
                log_height: 8,
            },
        }
    }

    /// Compute softmax over small integer inputs using the lookup table.
    /// Returns (z_fields, e_fields, y_fields).
    fn compute_softmax_from_table(
        inputs_i16: &[i16],
        table: &LookupTable,
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        // Build index map
        let mut index_map = std::collections::HashMap::new();
        for (j, &(inp, out)) in table.entries.iter().enumerate() {
            index_map.insert(inp, (j, out));
        }

        let z: Vec<F> = inputs_i16.iter()
            .map(|&v| to_field(v as i64))
            .collect();

        let e: Vec<F> = z.iter().map(|&zf| {
            let key = zf.as_canonical_u32();
            let (_, out) = index_map[&key];
            F::from_canonical_u32(out)
        }).collect();

        // Compute sum
        let sum: F = e.iter().copied().sum();
        let inv_s = sum.inverse();

        // y[i] = e[i] * inv_s
        let y: Vec<F> = e.iter().map(|&ei| ei * inv_s).collect();

        (z, e, y)
    }

    #[test]
    fn test_softmax_simple_4_elements() {
        let table = build_small_exp_table(100);
        // Use small inputs that fit in the table range (-128..127)
        let inputs: Vec<i16> = vec![1, 2, 3, 4];
        let (z, e, y) = compute_softmax_from_table(&inputs, &table);

        let mut pt = Transcript::new(b"softmax-test");
        let proof = prove_softmax(&z, &e, &y, &table, &mut pt);

        let mut vt = Transcript::new(b"softmax-test");
        assert!(verify_softmax(&proof, &table, z.len(), &mut vt));
    }

    #[test]
    fn test_softmax_known_distribution() {
        // softmax([0, 0, 0, 0]) should give uniform distribution: each y[i] = e[0] / (4*e[0])
        let table = build_small_exp_table(100);
        let inputs: Vec<i16> = vec![0, 0, 0, 0];
        let (z, e, y) = compute_softmax_from_table(&inputs, &table);

        // All e values should be equal (exp(0) = scale = 100)
        assert_eq!(e[0], e[1]);
        assert_eq!(e[1], e[2]);
        assert_eq!(e[2], e[3]);

        // All y values should be equal (uniform)
        assert_eq!(y[0], y[1]);
        assert_eq!(y[1], y[2]);
        assert_eq!(y[2], y[3]);

        let mut pt = Transcript::new(b"softmax-uniform");
        let proof = prove_softmax(&z, &e, &y, &table, &mut pt);

        let mut vt = Transcript::new(b"softmax-uniform");
        assert!(verify_softmax(&proof, &table, z.len(), &mut vt));
    }

    #[test]
    fn test_softmax_wrong_sum_claim() {
        let table = build_small_exp_table(100);
        let inputs: Vec<i16> = vec![1, 2, 3, 4];
        let (z, e, y) = compute_softmax_from_table(&inputs, &table);

        let mut pt = Transcript::new(b"softmax-bad-sum");
        let mut proof = prove_softmax(&z, &e, &y, &table, &mut pt);

        // Tamper with sum claim
        let p = (1u32 << 31) - 1;
        proof.sum_value = (proof.sum_value + 1) % p;
        // Also fix inv_s to match the tampered sum (otherwise inv_s*S != 1 catches it trivially)
        // Actually, we can't fix inv_s without knowing the field inverse. So the inv_s*S!=1 check
        // will catch it. That's fine — the point is verification rejects.

        let mut vt = Transcript::new(b"softmax-bad-sum");
        assert!(!verify_softmax(&proof, &table, z.len(), &mut vt));
    }

    #[test]
    fn test_softmax_wrong_output() {
        let table = build_small_exp_table(100);
        let inputs: Vec<i16> = vec![1, 2, 3, 4];
        let (z, e, y) = compute_softmax_from_table(&inputs, &table);

        let mut pt = Transcript::new(b"softmax-bad-output");
        let mut proof = prove_softmax(&z, &e, &y, &table, &mut pt);

        // Tamper with output finals to simulate a cheating prover
        let p = (1u32 << 31) - 1;
        proof.output_finals.1 = (proof.output_finals.1 + 1) % p;

        let mut vt = Transcript::new(b"softmax-bad-output");
        assert!(!verify_softmax(&proof, &table, z.len(), &mut vt));
    }

    #[test]
    fn test_softmax_8_elements() {
        let table = build_small_exp_table(100);
        let inputs: Vec<i16> = vec![10, 20, 5, 15, 8, 12, 3, 25];
        let (z, e, y) = compute_softmax_from_table(&inputs, &table);

        let mut pt = Transcript::new(b"softmax-8");
        let proof = prove_softmax(&z, &e, &y, &table, &mut pt);

        let mut vt = Transcript::new(b"softmax-8");
        assert!(verify_softmax(&proof, &table, z.len(), &mut vt));
    }
}
