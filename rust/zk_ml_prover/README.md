# ZK-Compile: Verifiable ML Inference

Structured sumcheck prover with Basefold PCS for cryptographic proof of transformer inference. Sub-100ms for 800M parameter models. 219-bit proximity soundness.

## Performance

Qwen3.5-0.8B (24 layers, 800M parameters) on Apple M4 Max:

| Mode | Prove | Verify | Proof Size | Soundness | What it proves |
|------|-------|--------|-----------|-----------|---------------|
| Default | **92ms** | 16ms | 4.7MB | 124-bit (EF sumcheck) | Fiat-Shamir binding |
| `--features pcs` | **134ms** | 20ms | 56MB | 124 + 219-bit (Basefold) | Basefold on w_partial |
| `--features pcs-full` | **14.3s** | 683ms | 391MB | 124 + 219-bit (Basefold) | Full standalone verification |

GPT-2 (12 layers, 117M parameters): **62ms** prove.

## Soundness

| Layer | Bits | Mechanism |
|-------|------|-----------|
| Proximity (PCS) | **219** | 24 queries × ~22 rounds × 0.415 bits. Recursively foldable code — proximity gap theorem (Basefold §4). |
| Fold consistency | **219** | Merkle-verified left + right + folded values at every query, every round. No skipped rounds. |
| Evaluation | **124** | M31^4 extension field sumcheck. Schwartz-Zippel over 124-bit field. |
| Commitment binding | **128+** | Blake3 Merkle tree for codewords. Collision resistance at blake3 security level. |
| Fiat-Shamir | **256** | SHA-256 transcript. All commitment roots absorbed before challenges squeezed. |

## Build

```bash
cargo build --release                    # Default: fastest, Fiat-Shamir binding
cargo build --release --features pcs     # Basefold on w_partial
cargo build --release --features pcs-full # Basefold on full W (standalone verify)
```

## Test

```bash
cargo test --release                     # 130 tests
cargo test --release --features pcs      # With Basefold on w_partial
cargo test --release --features pcs-full # With Basefold on full W
```

## Architecture

```
src/
├── main.rs                    # CLI entry point
├── protocol.rs                # Wire format, types, binary parsing
├── server.rs                  # Persistent binary protocol server
├── verification.rs            # Proof verification + standalone verifier
├── field/                     # M31 field math
│   ├── common.rs              # log2_ceil, hash, mod_sqrt, QR check
│   └── m31_ops.rs             # MLE evaluate, eq_evals, fold, f_to_ef
├── proving/                   # All proof protocols
│   ├── basefold.rs            # Basefold PCS (recursively foldable code, field-agnostic)
│   ├── sumcheck.rs            # Sumcheck prover/verifier, Transcript, EF
│   ├── matmul.rs              # Matmul proof + bind_weights (PCS dispatch)
│   ├── weight_commitment.rs   # Blake3 commitments, MLE eval proofs
│   ├── pcs.rs                 # Merkle tree infrastructure
│   ├── attention.rs           # Row attention + GQA
│   ├── layernorm.rs           # LayerNorm (squared variant)
│   ├── rmsnorm.rs             # RMSNorm
│   ├── lookup.rs              # LogUp table proofs (GELU, SiLU, sigmoid, exp)
│   ├── elementwise.rs         # Hadamard, add, scalar_mul
│   ├── sigmoid_gate.rs        # Sigmoid gate (sigmoid + hadamard)
│   ├── swiglu.rs              # SwiGLU (SiLU + hadamard)
│   ├── gelu.rs, silu.rs, sigmoid.rs, softmax.rs
│   └── mod.rs
├── pipeline/                  # Execution orchestration
│   ├── mod.rs                 # run_gpt2_mode, run_mlp_mode, types
│   ├── forward.rs             # Forward pass (all op types)
│   └── prove.rs               # Proving pass (all op types)
├── transformer/               # Model-specific proving
│   ├── mod.rs                 # Shared SIMD utilities
│   ├── gpt2.rs                # GPT-2 prove/verify
│   ├── llama.rs               # Llama prove/verify
│   └── qwen.rs                # Qwen3.5 prove/verify (base + EF)
└── gpu/                       # Metal GPU acceleration (optional)
```

## How It Works

### Proving

1. **Forward pass**: Execute the model in M31 arithmetic, recording all intermediates
2. **Structured sumcheck**: For each matmul `y = Wx`, prove `ỹ(r) = W̃(r,s) · x̃(s)` via product sumcheck
3. **Weight binding**: Commit to weights via Basefold PCS (recursively foldable code + Merkle tree), absorb into Fiat-Shamir transcript
4. **Nonlinear ops**: LogUp lookup argument for GELU, SiLU, sigmoid, softmax, exp
5. **Normalization**: Algebraic proofs for LayerNorm and RMSNorm (QR perturbation for non-QR cases)

### Verification

The verifier checks the sumcheck proofs, Fiat-Shamir transcript consistency, and (in default/pcs modes) re-runs the forward pass to verify the output. In `pcs-full` mode, a standalone verifier can check proofs against published commitments without access to raw weights.

### PCS Modes

| Feature | Commitment | Binding | Standalone? |
|---------|-----------|---------|------------|
| (none) | blake3 hash | Fiat-Shamir transcript | No — verifier needs weights |
| `pcs` | Basefold on w_partial | Basefold opening proof | No — needs trusted hash(W) |
| `pcs-full` | Basefold on full W | Basefold opening proof at (r,s) | **Yes** — needs only published commitment |

### Basefold PCS

Field-agnostic polynomial commitment scheme over Mersenne-31, following [Zeilberger-Chen-Fisch 2023](https://eprint.iacr.org/2023/1705).

- **Encoding**: Recursively foldable random linear code (rate 1/2). Butterfly construction: `Enc_{i+1}(m_L || m_R) = (Enc_i(m_L) + T_i · Enc_i(m_R)) || (Enc_i(m_L) - T_i · Enc_i(m_R))` where T_i are deterministic random diagonal matrices.
- **Folding**: Challenge α produces `Enc_i(m_L + α · m_R)` — a valid codeword of the halved code. Preserves code structure at every level (proximity gap theorem applies).
- **Commitment**: Blake3 Merkle tree over codeword elements.
- **Opening proof**: Interleaved fold rounds with 24 random query positions per level. Each query verifies: Merkle path on previous codeword (left + right), Merkle path on folded codeword, arithmetic fold consistency check.
- **Table generation**: Random T vectors via blake3 XOF. Fold weights via Montgomery batch inverse (1 inversion for all elements).

## Cryptographic Details

- **Field**: Mersenne-31 (p = 2^31 - 1) with M31^4 extension for 124-bit evaluation soundness
- **Proximity soundness**: 219 bits (24 queries × ~22 rounds × 0.415 bits per query)
- **Sumcheck**: Product and triple sumcheck with PackedMersenne31Neon SIMD (4-wide NEON)
- **PCS**: Basefold with recursively foldable code. No FFT required.
- **Lookup**: LogUp with active subtable optimization (16x smaller sumcheck domain)
- **Transcript**: SHA-256 Fiat-Shamir with domain separation

## Supported Operations

| Op | Proof Method |
|----|-------------|
| Linear (matmul + bias) | Structured product sumcheck |
| ReLU | Triple sumcheck (z = a · b, b ∈ {0,1}) |
| GELU | LogUp lookup (65536-entry table) |
| SiLU | LogUp lookup |
| Sigmoid | LogUp lookup |
| SwiGLU | SiLU lookup + hadamard |
| Sigmoid gate | Sigmoid lookup + hadamard |
| Softmax | Exp lookup + inverse + output sumcheck |
| LayerNorm | Algebraic (mean + variance + QR perturbation) |
| RMSNorm | Algebraic (sum of squares + QR) |
| Attention | Per-row: score matmul + softmax + output matmul |
| GQA Attention | Grouped query attention (Qwen3.5 hybrid) |
| Residual add | Elementwise add proof |
| Save/restore | Commitment-based state management |

## Benchmarks vs Competition

| System | Model | Prove | Verify | Proof | PCS |
|--------|-------|-------|--------|-------|-----|
| **ZK-Compile** | Qwen3.5-0.8B (800M) | **92ms** | 16ms | 4.7MB | Fiat-Shamir |
| **ZK-Compile (pcs-full)** | Qwen3.5-0.8B (800M) | **14.3s** | 683ms | 391MB | Basefold |
| **ZK-Compile** | GPT-2 (117M) | **62ms** | — | — | Fiat-Shamir |
| Deep-Prove | MLP (4M) | 2,335ms | 520ms | — | Basefold |
| Deep-Prove | CNN (264K) | 1,242ms | 599ms | — | Basefold |
| EZKL | MNIST MLP | ~1,310s | 5.4s | 127KB | Halo2/KZG |

Deep-Prove (Lagrange Labs) claims GPT-2 and Gemma 3 but has not published benchmark numbers for transformers.

## License

[TBD]
