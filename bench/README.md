# bench/ — head-to-head benchmark harness

Runs ZK-Compile and competing zk-ML systems on identical workloads,
identical hardware, identical Python env. The output CSV is the
load-bearing artifact — replaces "vendor-reported" footnotes in the
README with measurements anyone can replicate.

## Quick start

```bash
# Install requirements (Python 3.9+, recommend a venv)
pip install ezkl onnx torch numpy

# Build ZK-Compile prover binary
cd rust/zk_ml_prover && cargo build --release && cd ../..

# Run the comparison
python bench/compare.py
```

Output:

```
  → bench/results/compare.csv

========================================================================
  Dense-4M MLP head-to-head (Apple M4 Max, identical workload)
========================================================================
  System               Setup       Prove      Verify       Proof
  zk-compile         (none)     26.3 ms    0.200 ms    2,120 B
  ezkl              74737 ms  62456.6 ms 1281.800 ms  169,827 B

  Ratios (zk-compile vs ezkl):
    Prove:          2375× faster
    Verify:         6409× faster
    Proof:            80× smaller
```

## Files

- `run_ezkl.py` — exports a Dense-4M MLP from PyTorch, drives the EZKL
  pipeline (`gen_settings` → `calibrate_settings` → `compile_circuit` →
  `get_srs` → `gen_witness` → `setup` → `prove` → `verify`), and writes
  `bench/results/ezkl_dense4m.csv`.
- `compare.py` — wraps both `python/benchmark_dense4m.py` (ZK-Compile)
  and `run_ezkl.py`, parses outputs, emits the unified
  `bench/results/compare.csv` and a side-by-side console summary.
- `work/` — ONNX, witness, proving/verifying keys, proof bytes (gitignored).
- `results/` — CSV outputs (committed; that's the deliverable).

## Workload — Dense-4M MLP

Architecture (mirrored across both systems):
```
1024 → 2048 → ReLU → 1024 → ReLU → 10
≈ 4.2 M parameters
```

Same as Lagrange's "Dense 4M" benchmark. We pick this size because it
matches the reference number EZKL/DeepProve publish on, and because it
is small enough to fit comfortably in memory on both systems.

## Caveats

- **Numbers are machine-specific.** The README's headline ratios (2,375×
  prove, 6,409× verify, 80× proof) are from Apple M4 Max. Linux x86 will
  differ. The CSV is what's reproducible; the absolute milliseconds
  scale with your hardware. Both systems get the same penalty/bonus.
- **EZKL is at the default scale (`scales=[7]`).** Higher scales improve
  EZKL's numerical fidelity at the cost of significantly more prove
  time. We pick the lowest scale EZKL accepts to give it the best-case
  performance number — the ratio still holds.
- **Soundness regimes differ.** ZK-Compile's default is sumcheck over
  Mersenne-31 with M31⁴ extension-field challenges (~124-bit per round)
  and Fiat-Shamir binding. EZKL uses Halo2/KZG (~128-bit security at
  the SNARK level, with KZG trusted-setup assumption). The CSV records
  `soundness_bits_per_round` so the comparison is honest.
- **EZKL needs a KZG trusted setup.** The `get_srs` step downloads ~32MB
  of SRS data. Subsequent runs reuse the cached file under
  `bench/work/kzg.srs`.

## DeepProve (queued)

DeepProve has worse-documented input adapters and a custom binary
format; ~1 week of additional adapter code. Tracked as P10-1 stretch
goal. The harness already has the slot — just add `run_deepprove.py`
mirroring `run_ezkl.py` and a third row in `compare.py`.

## Adding a new model

1. Define the architecture in PyTorch in `run_ezkl.py` (or a new
   `run_<model>.py`).
2. Mirror it in `python/benchmark_<model>.py` for the ZK-Compile side.
3. Add a `run_<system>(model)` function in `compare.py` that parses
   each system's output.

The harness is intentionally simple: one Python file per system. No
Docker, no orchestrator, no hidden state. Anyone reading the file can
audit exactly what was timed.
