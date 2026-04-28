//! P10-7: proptest cross-language harness for GDN
//! recurrence audit-mode math.
//!
//! Generates random small fixtures, runs both the Rust prover's
//! `run_recurrence_with_digest` and the numpy reference
//! (`tests/reference/gdn_recurrence.py`) via the JSON shim, then
//! asserts:
//!   1. The trajectory digests match bytewise.
//!   2. The final state vectors match field-for-field.
//!   3. The per-step output vectors match field-for-field.
//!
//! Mirrors the `tests/property/harness.rs` (P11-2) pattern. Subprocess
//! overhead caps wall-clock at ~5/s; default 20 cases ≈ 4 s.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use p3_field::{AbstractField, PrimeField32};
use p3_mersenne_31::Mersenne31;

use zk_ml_prover::proving::gdn_recurrence::{
    run_recurrence_with_digest, GdnRecurrenceConfig, GdnRecurrenceStep,
};

type F = Mersenne31;

/// Path to `tests/reference/run.py`. Mirrors `harness.rs::run_py_path`.
fn run_py_path() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("CARGO_MANIFEST_DIR has at least two parents")
        .join("tests")
        .join("reference")
        .join("run.py")
}

fn call_reference(request: &serde_json::Value) -> Result<serde_json::Value, String> {
    let script = run_py_path();
    let mut child = Command::new("python3")
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn python3: {e}"))?;

    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| "no stdin pipe".to_string())?;
        let body = serde_json::to_vec(request).map_err(|e| format!("serialize: {e}"))?;
        stdin
            .write_all(&body)
            .map_err(|e| format!("write stdin: {e}"))?;
    }

    let out = child
        .wait_with_output()
        .map_err(|e| format!("wait: {e}"))?;
    if !out.status.success() && out.stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "python3 run.py exited {}: stderr={}",
            out.status, stderr
        ));
    }
    let resp: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("parse stdout: {e}"))?;
    if resp["ok"].as_bool() != Some(true) {
        return Err(format!(
            "shim returned error: {}",
            resp.get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("<no error message>")
        ));
    }
    Ok(resp["result"].clone())
}

/// Sample a small GDN recurrence fixture (H ∈ {1,2}, d_k ∈ {2,4},
/// d_v ∈ {2,4}, T ∈ {1..=4}). Field elements stay in `[0, 64)` so
/// the recurrence stays in the M31 positive half-field — keeps the
/// numpy reference fast (it's pure-Python loops, not vectorized).
fn sample_fixture(seed: u64) -> (Vec<F>, Vec<GdnRecurrenceStep>, GdnRecurrenceConfig) {
    let mut rng = StdRng::seed_from_u64(seed);
    let num_heads = *[1usize, 2].choose(&mut rng).expect("non-empty");
    let d_k = *[2usize, 4].choose(&mut rng).expect("non-empty");
    let d_v = *[2usize, 4].choose(&mut rng).expect("non-empty");
    let seq_len = rng.gen_range(1..=4usize);
    let cfg = GdnRecurrenceConfig {
        num_heads,
        d_k,
        d_v,
    };
    let small = |rng: &mut StdRng| rng.gen_range(0u32..64);
    let s0: Vec<F> = (0..cfg.state_size())
        .map(|_| F::from_canonical_u32(small(&mut rng)))
        .collect();
    let steps: Vec<GdnRecurrenceStep> = (0..seq_len)
        .map(|_| GdnRecurrenceStep {
            q: (0..num_heads * d_k).map(|_| small(&mut rng)).collect(),
            k: (0..num_heads * d_k).map(|_| small(&mut rng)).collect(),
            v: (0..num_heads * d_v).map(|_| small(&mut rng)).collect(),
            gate: (0..num_heads).map(|_| small(&mut rng)).collect(),
            beta: (0..num_heads).map(|_| small(&mut rng)).collect(),
        })
        .collect();
    (s0, steps, cfg)
}

trait ChooseExt<T> {
    fn choose<R: Rng>(&self, rng: &mut R) -> Option<&T>;
}

impl<T> ChooseExt<T> for [T] {
    fn choose<R: Rng>(&self, rng: &mut R) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            Some(&self[rng.gen_range(0..self.len())])
        }
    }
}

fn build_request(
    s0: &[F],
    steps: &[GdnRecurrenceStep],
    cfg: &GdnRecurrenceConfig,
) -> serde_json::Value {
    let initial_state: Vec<u32> = s0.iter().map(|f| f.as_canonical_u32()).collect();
    let steps_json: Vec<serde_json::Value> = steps
        .iter()
        .map(|s| {
            serde_json::json!({
                "q": s.q,
                "k": s.k,
                "v": s.v,
                "gate": s.gate,
                "beta": s.beta,
            })
        })
        .collect();
    serde_json::json!({
        "op": "gdn_recurrence",
        "args": {
            "initial_state": initial_state,
            "steps": steps_json,
            "config": {
                "num_heads": cfg.num_heads,
                "d_k": cfg.d_k,
                "d_v": cfg.d_v,
            },
        },
    })
}

proptest! {
    #![proptest_config(ProptestConfig {
        // Subprocess overhead caps us at ~5/s. 20 cases ≈ 4 s wall-clock.
        cases: 20,
        max_shrink_iters: 16,
        .. ProptestConfig::default()
    })]

    /// **End-to-end Rust ↔ numpy parity for the GDN recurrence
    /// trajectory digest.** A random small fixture must produce
    /// identical (digest, S_T, o_seq) from
    /// `run_recurrence_with_digest` (Rust) and `gdn_recurrence_forward`
    /// (numpy). Bytewise digest equality is the load-bearing assertion
    /// — the audit-mode binding rests on it (verifier rejects on any
    /// digest divergence).
    #[test]
    fn prop_gdn_recurrence_matches_reference(seed in any::<u64>()) {
        let (s0, steps, cfg) = sample_fixture(seed);

        // Rust forward.
        let trace = run_recurrence_with_digest(&s0, &steps, &cfg);

        // numpy forward.
        let req = build_request(&s0, &steps, &cfg);
        let resp = call_reference(&req).map_err(|e| TestCaseError::fail(e))?;

        // (1) Digest bytewise equality.
        let py_digest_hex = resp["trajectory_digest"]
            .as_str()
            .expect("trajectory_digest hex string");
        let rust_digest_hex = hex::encode(trace.trajectory_digest);
        prop_assert_eq!(rust_digest_hex.as_str(), py_digest_hex,
            "trajectory_digest mismatch (cfg={:?}, T={})", cfg, steps.len());

        // (2) S_final field-for-field equality.
        let py_s_final = resp["s_final"].as_array().expect("s_final array");
        let rust_s_final: Vec<u32> = trace.s_final.iter().map(|f| f.as_canonical_u32()).collect();
        prop_assert_eq!(py_s_final.len(), rust_s_final.len(),
            "s_final length mismatch");
        for (i, py_v) in py_s_final.iter().enumerate() {
            let pyv = py_v.as_u64().expect("u64") as u32;
            prop_assert_eq!(pyv, rust_s_final[i],
                "s_final[{}] mismatch: py={} rust={}", i, pyv, rust_s_final[i]);
        }

        // (3) o_seq field-for-field equality.
        let py_o_seq = resp["o_seq"].as_array().expect("o_seq array");
        prop_assert_eq!(py_o_seq.len(), trace.o_seq.len(),
            "o_seq length mismatch");
        for (t, py_o_t) in py_o_seq.iter().enumerate() {
            let py_o_t = py_o_t.as_array().expect("o_t array");
            let rust_o_t: Vec<u32> = trace.o_seq[t].iter().map(|f| f.as_canonical_u32()).collect();
            prop_assert_eq!(py_o_t.len(), rust_o_t.len(),
                "o_seq[{}] length mismatch", t);
            for (i, py_v) in py_o_t.iter().enumerate() {
                let pyv = py_v.as_u64().expect("u64") as u32;
                prop_assert_eq!(pyv, rust_o_t[i],
                    "o_seq[{}][{}] mismatch", t, i);
            }
        }
    }
}

/// Smoke test: confirms `run.py` is reachable AND that the
/// `gdn_recurrence` op is registered. Fails fast (no proptest
/// shrinking) if the JSON shim wiring regresses.
#[test]
fn smoke_gdn_recurrence_shim_reachable() {
    let cfg = GdnRecurrenceConfig {
        num_heads: 1,
        d_k: 2,
        d_v: 2,
    };
    let s0: Vec<F> = (0..cfg.state_size())
        .map(|_| F::zero())
        .collect();
    let steps = vec![GdnRecurrenceStep {
        q: vec![0; 2],
        k: vec![0; 2],
        v: vec![0; 2],
        gate: vec![0],
        beta: vec![0],
    }];
    let req = build_request(&s0, &steps, &cfg);
    let resp = call_reference(&req).expect("run.py shim must be reachable");
    let py_digest = resp["trajectory_digest"]
        .as_str()
        .expect("trajectory_digest hex");
    let rust_digest = hex::encode(run_recurrence_with_digest(&s0, &steps, &cfg).trajectory_digest);
    assert_eq!(rust_digest, py_digest, "smoke test: digest mismatch");
    eprintln!("p10-7 gdn-recurrence shim digest: {}", py_digest);
}
