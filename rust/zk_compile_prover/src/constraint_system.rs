use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystemRef, SynthesisError, Variable,
};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Debug)]
pub struct SerializedConstraint {
    pub a: HashMap<String, String>,
    pub b: HashMap<String, String>,
    pub c: HashMap<String, String>,
}

#[derive(Deserialize, Debug)]
pub struct SerializedCircuit {
    pub field: String,
    pub field_modulus: String,
    pub num_wires: usize,
    pub num_public_inputs: usize,
    pub num_constraints: usize,
    pub constraints: Vec<SerializedConstraint>,
    pub witness: Vec<String>,
}

pub struct ZkCompileCircuit {
    pub num_wires: usize,
    pub num_public_inputs: usize,
    pub constraints: Vec<SerializedConstraint>,
    pub witness: Vec<String>,
}

impl ZkCompileCircuit {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        const BN254_MODULUS: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";

        let data = std::fs::read_to_string(path)?;
        let circuit: SerializedCircuit = serde_json::from_str(&data)?;

        if circuit.field_modulus != BN254_MODULUS {
            return Err(format!(
                "Unsupported field modulus: {}. Expected BN254: {}",
                circuit.field_modulus, BN254_MODULUS
            ).into());
        }

        Ok(Self {
            num_wires: circuit.num_wires,
            num_public_inputs: circuit.num_public_inputs,
            constraints: circuit.constraints,
            witness: circuit.witness,
        })
    }

    pub fn str_to_fr(s: &str) -> Fr {
        let val: num_bigint::BigUint = s.parse().expect("invalid number in witness");
        let bytes = val.to_bytes_le();
        Fr::from_le_bytes_mod_order(&bytes)
    }
}

impl ConstraintSynthesizer<Fr> for ZkCompileCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        // Validate witness length: must have at least num_wires entries.
        // Without this, missing witness values would silently become zero,
        // which could produce valid-looking proofs for wrong computations.
        if self.witness.len() < self.num_wires {
            return Err(SynthesisError::Unsatisfiable);
        }

        let mut variables: Vec<Variable> = Vec::with_capacity(self.num_wires);

        // Wire 0 → constant 1
        variables.push(Variable::One);

        // Allocate public inputs
        for i in 1..=self.num_public_inputs {
            let val = if i < self.witness.len() {
                Self::str_to_fr(&self.witness[i])
            } else {
                Fr::from(0u64)
            };
            let var = cs.new_input_variable(|| Ok(val))?;
            variables.push(var);
        }

        // Allocate private witnesses
        for i in (self.num_public_inputs + 1)..self.num_wires {
            let val = if i < self.witness.len() {
                Self::str_to_fr(&self.witness[i])
            } else {
                Fr::from(0u64)
            };
            let var = cs.new_witness_variable(|| Ok(val))?;
            variables.push(var);
        }

        // Add constraints
        for constraint in &self.constraints {
            let mut a_lc = ark_relations::r1cs::LinearCombination::<Fr>::zero();
            for (wire_str, coeff_str) in &constraint.a {
                let wire_idx: usize = wire_str.parse().map_err(|_| SynthesisError::Unsatisfiable)?;
                if wire_idx >= variables.len() {
                    return Err(SynthesisError::Unsatisfiable);
                }
                let var = variables[wire_idx];
                a_lc = a_lc + (Self::str_to_fr(coeff_str), var);
            }

            let mut b_lc = ark_relations::r1cs::LinearCombination::<Fr>::zero();
            for (wire_str, coeff_str) in &constraint.b {
                let wire_idx: usize = wire_str.parse().map_err(|_| SynthesisError::Unsatisfiable)?;
                if wire_idx >= variables.len() {
                    return Err(SynthesisError::Unsatisfiable);
                }
                let var = variables[wire_idx];
                b_lc = b_lc + (Self::str_to_fr(coeff_str), var);
            }

            let mut c_lc = ark_relations::r1cs::LinearCombination::<Fr>::zero();
            for (wire_str, coeff_str) in &constraint.c {
                let wire_idx: usize = wire_str.parse().map_err(|_| SynthesisError::Unsatisfiable)?;
                if wire_idx >= variables.len() {
                    return Err(SynthesisError::Unsatisfiable);
                }
                let var = variables[wire_idx];
                c_lc = c_lc + (Self::str_to_fr(coeff_str), var);
            }

            cs.enforce_constraint(a_lc, b_lc, c_lc)?;
        }

        Ok(())
    }
}
