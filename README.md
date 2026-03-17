# ZK-Compile

Verifiable ML inference via structured sumcheck over Mersenne-31. Proves that a transformer's output was computed correctly — cryptographically, without re-execution.

## Benchmarks

| Model | Params | Prove | Verify | Proof Size | Soundness |
|-------|--------|-------|--------|------------|-----------|
| Qwen3.5-0.8B | 800M | **89ms** | 17ms | 789KB | 124-bit |
| GPT-2 | 117M | **247ms** | 5ms | 939KB | 124-bit |
| Dense MLP | 4.2M | **24ms** | 0.1ms | 2.1KB | 124-bit |
| MNIST MLP | 101K | **2.8ms** | 0.1ms | 908B | 124-bit |
| Qwen3.5-0.8B (Basefold) | 800M | 2.3s | 358ms | 32MB | 124 + 219-bit |

Apple M4 Max. Comparisons on identical workloads:

| | ZK-Compile | ZK-Compile (Basefold) | DeepProve | EZKL |
|---|---|---|---|---|
| Dense 4M MLP | **24ms** | **124ms** | 2,335ms | 126,831ms |
| MNIST MLP | **2.8ms** | — | — | 1,310,000ms |

## How it works

Every transformer operation maps to a specialized sub-protocol over M31 (p = 2³¹ − 1):

```
Matmul          → product sumcheck + MLE eval proof
LayerNorm       → product + triple sumcheck + QR perturbation
RMSNorm         → product sumcheck + inverse proof
GELU / SiLU     → LogUp lookup (extension field challenges)
SwiGLU          → LogUp + Hadamard product proof
Softmax         → exp lookup + sum + division proof
Attention (GQA) → row-wise product sumcheck
Residual add    → linear combination proof
PCS             → Basefold with AES-MMO Merkle commitments
```

Three model families: GPT-2 (LayerNorm + GELU), Llama (RMSNorm + SwiGLU + GQA), Qwen3.5 (hybrid GatedDeltaNet + attention — first ZK proof of a hybrid recurrent-attention architecture).

124-bit evaluation soundness via M31⁴ extension field challenges across all sub-protocols. Optional Basefold PCS adds 219-bit proximity soundness per polynomial commitment.

## Install

```bash
# Local circuit proving (Groth16 backend, cross-platform)
pip install zk-compile
zk-compile prove-and-verify circuit.json

# Proof auditing (lightweight, numpy-only)
pip install zk-ml-audit
zk-audit proof.json --model gpt2 --input "hello world"

# ML proof server
docker compose -f docker/docker-compose.yml up
```

```python
from zk_compile import prove_and_verify
result = prove_and_verify("circuit.json")

from zk_ml_audit import audit_full
report = audit_full(proof_data, model_name="gpt2")
```

## Build from source

```bash
cd rust/zk_ml_prover
cargo test --release                       # default mode
cargo test --release --features pcs        # + Basefold on w_partial
cargo test --release --features pcs-full   # + Basefold on full weight matrix
```

Reproduce benchmarks:
```bash
cargo build --release
cd ../..
python python/benchmark_mnist.py
python python/benchmark_dense4m.py
```

## Build modes

| Mode | Flag | Commitment | Soundness |
|------|------|------------|-----------|
| Default | — | Fiat-Shamir binding | 124-bit evaluation |
| PCS | `--features pcs` | Basefold on projected weights | + 219-bit proximity |
| PCS-Full | `--features pcs-full` | Basefold on full weight matrix | + 219-bit proximity |

## Project layout

```
rust/zk_ml_prover/src/
├── field/           M31 arithmetic, NEON SIMD, MLE evaluation
├── proving/
│   ├── sumcheck     product / triple sumcheck (base + extension field)
│   ├── matmul       weight commitment + product sumcheck + MLE eval
│   ├── basefold     Basefold PCS (butterfly encode, fold, commit)
│   ├── lookup       LogUp argument (GELU, SiLU, sigmoid, softmax)
│   ├── layernorm    LayerNorm + RMSNorm proofs
│   └── ...          attention, elementwise, softmax, swiglu
├── transformer/     model-specific proving (GPT-2, Llama, Qwen3.5)
└── pipeline/        orchestration, forward pass, proof assembly
```

## Paper

See [PAPER.md](PAPER.md) for the full technical description.

## License

MIT OR Apache-2.0