mod constraint_system;

use ark_bn254::{Bn254, Fr};
use ark_groth16::Groth16;
use ark_snark::SNARK;
use rand::rngs::OsRng;
use clap::{Parser, Subcommand};
use std::time::Instant;

#[allow(unused_imports)]
use ark_ff::{Field, One, Zero};

use constraint_system::ZkCompileCircuit;

#[derive(Parser)]
#[command(name = "zk-compile-prover")]
#[command(about = "ZK-Compile: Prove and verify R1CS circuits from TVM")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate proof from circuit file
    Prove {
        /// Path to circuit JSON file
        circuit: String,
        /// Path to write proof output
        #[arg(short, long, default_value = "proof.bin")]
        output: String,
    },
    /// Verify a proof
    Verify {
        /// Path to circuit JSON file (for public inputs + verifying key)
        circuit: String,
        /// Path to proof file
        #[arg(short, long, default_value = "proof.bin")]
        proof: String,
    },
    /// Prove and verify in one step (for benchmarking)
    ProveAndVerify {
        /// Path to circuit JSON file
        circuit: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Prove { circuit, output } => {
            let c = ZkCompileCircuit::from_file(&circuit)?;
            println!("Circuit loaded: {} wires, {} constraints, {} public inputs",
                c.num_wires, c.constraints.len(), c.num_public_inputs);

            let rng = &mut OsRng;

            // Setup
            let t = Instant::now();
            let c_setup = ZkCompileCircuit::from_file(&circuit)?;
            let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c_setup, rng)?;
            println!("Setup: {:.2}s", t.elapsed().as_secs_f64());

            // Prove
            let t = Instant::now();
            let c_prove = ZkCompileCircuit::from_file(&circuit)?;
            let proof = Groth16::<Bn254>::prove(&pk, c_prove, rng)?;
            println!("Prove: {:.2}s", t.elapsed().as_secs_f64());

            // Serialize proof
            let proof_bytes = ark_serialize::CanonicalSerialize::compressed_size(&proof);
            let mut buf = Vec::with_capacity(proof_bytes);
            ark_serialize::CanonicalSerialize::serialize_compressed(&proof, &mut buf)?;
            std::fs::write(&output, &buf)?;
            println!("Proof written to {} ({} bytes)", output, buf.len());

            // Also serialize verifying key
            let vk_path = format!("{}.vk", output);
            let mut vk_buf = Vec::new();
            ark_serialize::CanonicalSerialize::serialize_compressed(&vk, &mut vk_buf)?;
            std::fs::write(&vk_path, &vk_buf)?;
            println!("Verifying key written to {} ({} bytes)", vk_path, vk_buf.len());
        }

        Commands::Verify { circuit, proof: proof_path } => {
            let c = ZkCompileCircuit::from_file(&circuit)?;

            // Extract public inputs from witness
            let public_inputs: Vec<Fr> = (1..=c.num_public_inputs)
                .map(|i| ZkCompileCircuit::str_to_fr(&c.witness[i]))
                .collect();

            // Read proof and vk
            let proof_bytes = std::fs::read(&proof_path)?;
            let proof: ark_groth16::Proof<Bn254> =
                ark_serialize::CanonicalDeserialize::deserialize_compressed(&proof_bytes[..])?;

            let vk_path = format!("{}.vk", &proof_path);
            let vk_bytes = std::fs::read(&vk_path)?;
            let vk: ark_groth16::VerifyingKey<Bn254> =
                ark_serialize::CanonicalDeserialize::deserialize_compressed(&vk_bytes[..])?;

            let pvk = ark_groth16::prepare_verifying_key(&vk);

            let t = Instant::now();
            let valid = Groth16::<Bn254>::verify_with_processed_vk(&pvk, &public_inputs, &proof)?;
            println!("Verify: {:.4}s — {}", t.elapsed().as_secs_f64(),
                if valid { "VALID" } else { "INVALID" });
        }

        Commands::ProveAndVerify { circuit } => {
            let c = ZkCompileCircuit::from_file(&circuit)?;
            let num_constraints = c.constraints.len();
            let num_wires = c.num_wires;
            let num_public = c.num_public_inputs;
            println!("Circuit: {} wires, {} constraints, {} public inputs",
                num_wires, num_constraints, num_public);

            let rng = &mut OsRng;

            // Setup
            let t = Instant::now();
            let c_setup = ZkCompileCircuit::from_file(&circuit)?;
            let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c_setup, rng)?;
            let setup_time = t.elapsed().as_secs_f64();
            println!("Setup: {:.3}s", setup_time);

            // Prove
            let t = Instant::now();
            let c_prove = ZkCompileCircuit::from_file(&circuit)?;
            let proof = Groth16::<Bn254>::prove(&pk, c_prove, rng)?;
            let prove_time = t.elapsed().as_secs_f64();
            println!("Prove: {:.3}s", prove_time);

            // Verify
            let c_verify = ZkCompileCircuit::from_file(&circuit)?;
            let public_inputs: Vec<Fr> = (1..=c_verify.num_public_inputs)
                .map(|i| ZkCompileCircuit::str_to_fr(&c_verify.witness[i]))
                .collect();

            let pvk = ark_groth16::prepare_verifying_key(&vk);
            let t = Instant::now();
            let valid = Groth16::<Bn254>::verify_with_processed_vk(&pvk, &public_inputs, &proof)?;
            let verify_time = t.elapsed().as_secs_f64();

            // Proof size
            let mut proof_buf = Vec::new();
            ark_serialize::CanonicalSerialize::serialize_compressed(&proof, &mut proof_buf)?;

            println!("Verify: {:.4}s — {}", verify_time,
                if valid { "VALID ✓" } else { "INVALID ✗" });
            println!("\n--- Summary ---");
            println!("Constraints: {}", num_constraints);
            println!("Wires: {}", num_wires);
            println!("Public inputs: {}", num_public);
            println!("Setup time: {:.3}s", setup_time);
            println!("Prove time: {:.3}s", prove_time);
            println!("Verify time: {:.4}s", verify_time);
            println!("Proof size: {} bytes", proof_buf.len());
        }
    }

    Ok(())
}
