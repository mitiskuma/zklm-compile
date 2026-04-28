# ZK-Compile

**376 ms to prove correctness of a 4 B-parameter hybrid transformer's
forward pass — verifier-trusted-weights mode, with a documented path to
true ZK.**

Structured sumcheck over Mersenne-31 (M31⁴ extension-field challenges,
~124 bits per round). On Apple M4 Max, EF default:

| | Prove | Verify | Proof | Notes |
|---|---|---|---|---|
| **Qwen3.5-4B** (32 hybrid layers) | **376 ms** | 74 ms | 1.25 MB | proofground |
| GPT-2 (124 M, 12 layers) | 253 ms | 5.2 ms | 963 KB | 182/182 ops proved |
| Dense-4M MLP | 24.7 ms | 0.1 ms | 2.1 KB | head-to-head vs EZKL: ~2,375× faster prove, ~6,409× faster verify, ~80× smaller proof — `python bench/compare.py` reproduces |

> **What "verifier-trusted-weights mode" means.**
> The verifier today knows the model weights and re-runs `qwen_forward(x, weights, config)`
> internally; sub-proof MLE evaluations bind to its own canonical trace. This is
> sound and fast (74 ms verify on 4B) but is NOT the "true ZK with hidden
> weights" model. Migration path:
> 1. Already shipped: Seq1VConsistency v↔attn_out proof-level binding at seq_len=1
>    (the sub-proof that survives the audit-mode → true-ZK transition).
> 2. Already documented: LogUp `_with_data` audit-mode digest contract + tamper-test.
> 3. Roadmap: replace verifier's `qwen_forward` recomputation with chained
>    MLE-eval-bound claims rooted at the input commitment. ~1 quarter of work,
>    not 6 weeks. See "Limitations" and Phase 10 in `ROADMAP.md` for the
>    structural plan.

## Benchmarks

> **Reproducibility note.** The April 2026
> benchmark re-run measured all numbers above marked "(re-measured)"
> directly under the now-default EF challenges on Apple M4 Max. The
> GPT-2 prove-time (~253 ms) is ~4× the pre-EF-default snapshot
> (62 ms) — the expected cost of the EF default replacing base-field
> challenges (M1). Qwen3.5-4B/9B were unblocked for this run by a
> direct-safetensors loader path in `python/tvm_to_gkr/model_extractor.py`
> (since stable transformers ≤ 4.57.6 doesn't recognize the `qwen3_5`
> architecture); the loader synthesizes hidden-state inputs for
> benchmark timing, which is dimension-equivalent to real token-stream
> inputs. Reproduction harness in `python/benchmark_*.py`.

Apple M4 Max, default mode (extension-field, ~124-bit soundness).
Numbers marked **(re-measured)** were verified in the April 2026
re-run under the now-default EF challenges; the others are from the
pre-EF-default run and are queued for re-verification (see
Limitations § benchmark re-run).

| Model | Params | Layers | Prove | Verify | Proof Size | Soundness |
|-------|--------|--------|-------|--------|------------|-----------|
| Dense MLP | 4.2M | — | **24.7 ms** (re-measured) | 0.1 ms | 2.1 KB | 124-bit |
| GPT-2 | 124M | 12 | **253 ms** (re-measured, avg 248–260 over 3 runs) | 5.16 ms | 963 KB | 124-bit |
| Qwen3.5-4B (GDN+attn hybrid) | 4B | 32 (24 GDN + 8 full-attn) | **376 ms** (re-measured, avg 356–392 over 3 runs) | 74 ms | 1.25 MB | 124-bit |
| Qwen3.5-9B (GDN+attn hybrid) | 9B | 32 (24 GDN + 8 full-attn) | **17.8 s** (re-measured, single run; memory-bound — see note) | 2.6 s | 1.25 MB | 124-bit |
| Qwen3.5-0.8B | 800M | 24 | 92 ms (pre-EF-default) | 16 ms | 4.7 MB | 124-bit |
| MNIST MLP | 101K | — | 2.8 ms (pre-EF-default) | 0.1 ms | 908 B | 124-bit |
| Qwen3.5-0.8B (`--features pcs-full`) | 800M | 24 | 14.3 s (pre-EF-default) | 683 ms | 391 MB | 124 + 219-bit |

> **Qwen3.5-4B vs 9B scaling note:** the 4B prove time (376 ms) reflects
> the parallel `all_qwen` path (32 layers fanned out via rayon, weight
> matrices fit comfortably in M4 Max's 64 GB RAM). The 9B run (17.8 s)
> is memory-bound — at `d_model = 4096, d_ff = 12288` the per-layer
> weights are ~200 MB, so 32 simultaneous rayon threads saturate RAM
> and degrade to swap-bound serial execution. The 4B 376 ms is the
> meaningful "shipping production" number; the 9B 17.8 s is the
> "scale ceiling on this hardware" number. Closer-to-linear scaling
> for 9B requires either a `--features pcs-full` (which already caps
> `par_chunk_size = 8`) or a memory-aware scheduler — tracked as
> future work.

Head-to-head, identical M4 Max, identical Dense-4M workload (April 2026,
P10-1 harness — `python bench/compare.py` reproduces these on your hardware):

| | ZK-Compile | EZKL (re-measured) |
|---|---|---|
| Setup | none (Fiat-Shamir) | **74.7 s** (KZG trusted setup) |
| Prove | **26.3 ms** | 62,456.6 ms (**~2,375× slower than ZK-Compile**) |
| Verify | **0.20 ms** | 1,281.8 ms (**~6,409× slower**) |
| Proof | **2.1 KB** | 169.8 KB (**~80× larger**) |
| Soundness/round | 124-bit (EF sumcheck) | ~128-bit (KZG/Halo2) |

Comparison harness: `bench/compare.py` (ezkl 23.0.5 + ZK-Compile EF default).

Other comparisons (still vendor-reported, harness queued — see Limitations):

| | ZK-Compile (Basefold) | DeepProve |
|---|---|---|
| Dense 4M MLP | **124 ms** | 2,335 ms (vendor-reported, not re-run on M4 Max) |
| MNIST MLP | (queued, env-blocked) | (no published number) |

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

Three model families: GPT-2 (LayerNorm + GELU), Llama (RMSNorm + SwiGLU + GQA), Qwen3.5 (hybrid GatedDeltaNet + attention layers).

> **Honest scope on the GDN claim.** The Qwen3.5
> support proves the *layer-structure* portion of the hybrid architecture
> — RMSNorm, fused Q/K/V via `in_proj_qkv`, attention output, sigmoid-gate
> on top of the attention output, residual, RMSNorm again, gate/up/down
> SwiGLU MLP, residual. The Conv1D pre-step is **statically folded into
> the projection weights in Python before proving** (`_fold_conv_into_proj`
> in `python/tvm_to_gkr/qwen35_pipeline.py`); the recurrent delta-rule
> state update of the GDN layer is **not proven** in this release —
> `A_log`, `dt_bias`, `in_proj_a`, `in_proj_b` are extracted but not
> instantiated as a sumcheck-provable recurrence. A full proof of the
> recurrent state update is queued as future work.

124-bit evaluation soundness via M31⁴ extension field challenges across all sub-protocols. Optional Basefold PCS adds 219-bit proximity soundness per polynomial commitment.

## Install

> **Status:** the `zk-compile` and `zk-ml-audit` PyPI packages are not yet
> published. Use the source builds below. PyPI wheels are tracked under a
> future release once the API stabilizes.

```bash
# Local circuit proving (Groth16 backend, cross-platform) — source build
git clone https://github.com/mitiskuma/zklm-compile && cd zklm-compile
pip install -e ./rust/zk_compile_prover    # provides `zk-compile` CLI
zk-compile prove-and-verify circuit.json

# Proof auditing (lightweight, numpy-only) — source build
pip install -e ./python/zk_ml_audit         # provides `zk-audit` CLI
zk-audit proof.json --model gpt2 --input "hello world"

# ML proof server (Docker)
docker compose -f docker/docker-compose.yml up
# → tier the `memory:` limit per model size; see comment in compose.yml
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

### Quickstart smoke-test (R8)

After cloning, run the bundled five-line proof to confirm your toolchain
works before spending time on the larger model benchmarks:

```bash
./examples/quickstart.sh
# → builds zk_compile_prover, proves and verifies examples/square_circuit.json
#   (proves x*x = 25 over BN254 — proof in ~1 ms, verify in ~1 ms).
```

`examples/square_circuit.json` is the canonical "does this even build?"
input. The Rust unit test
`zk_compile_prover::constraint_system::tests::test_example_square_circuit_loads_and_satisfies`
re-validates the file in CI, so it can't silently rot when the schema
changes.

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

## Mode semantics (C4 — Option C)

The prover ships two evaluation regimes that the verifier interoperates with
via Fiat-Shamir transcript binding. **Default is extension-field** so that a
fresh `cargo run` of the prover lands on the funding-grade soundness path.

| Mode | CLI flag | Sumcheck challenges | Per-round soundness | Use case | Verifier needs weights? |
|------|---------|--------------------|---------------------|----------|-------------------------|
| **EF (default)** | *(no flag)* | M31⁴ extension (~124-bit) | ~2⁻¹²⁴ per round | true ZK production proofs | optional — commitments suffice |
| **Fast** | `--fast` | base M31 (~31-bit per round) | ~2⁻³¹ per round | fast audit / dev mode | yes — verifier re-executes |
| **PCS** | `--features pcs` (also opt-in `pcs-full`) | EF + Basefold proximity | + 219-bit proximity | weight-commit binding (no re-execution) | no |

The CLI flag and feature flag compose: a typical funding-grade run is
`cargo run --release --features pcs` (no `--fast`), which selects EF
challenges *and* enables Basefold proximity binding. `--fast` is provided
for dev iteration where 31-bit-per-round soundness is acceptable; the
prover prints the active mode at startup so a misconfigured run can't
hide.

## Limitations (C6 — what this release does NOT do)

- **GDN recurrence is not proven (C1).** Qwen3.5 support proves the layer
  *structure* (RMSNorm → fused QKV → attention → sigmoid-gate → residual
  → RMSNorm → SwiGLU MLP → residual). Conv1D is statically folded into
  the projection weights *in Python before proving*; the gated-delta-rule
  recurrent state update (`A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`) is
  extracted but not instantiated as a sumcheck. A full proof of the
  recurrent state update is on the roadmap; today's prover proves a
  hybrid layer *with statically-folded recurrence*, which is weaker than
  "first ZK proof of GatedDeltaNet" — see the explicit scope note above.
- **Conv1D fold tap convention is configurable but not yet verified
  against an HF reference (G1).** `_fold_conv_into_proj` in
  `python/tvm_to_gkr/qwen35_pipeline.py` collapses the seq_len=1 Conv1D
  to an element-wise projection scale by reading
  `conv_weight[:, 0, causal_tap_index]`. Standard PyTorch causal Conv1d
  with left-padding implies the active tap is the LAST kernel position
  (`-1`), but some HF Mamba/GDN custom kernels store weights in
  reverse-time order making index `0` correct. The parameter defaults
  to `0` (historical pipeline behavior); flipping requires a forward-pass
  match against an HF reference. The Python regression test
  `test_fold_causal_tap_distinguishes_first_and_last` pins both
  branches so a refactor cannot silently change the default.
- **Post-Conv1D SiLU is dropped by the static fold (G2).** The real
  Qwen3.5 GDN forward pass is `y = silu(conv1d(in_proj(x)))` followed
  by Q/K/V split. The static fold in `_fold_conv_into_proj` captures
  `conv1d(in_proj(x))` but not the SiLU — the Rust prover therefore
  proves an approximation where the Q/K/V inputs are pre-SiLU values.
  Drift is small at typical GDN scales (post-RMSNorm inputs are O(1),
  where SiLU ≈ x/2 + small correction), but it is an approximation.
  A faithful proof would absorb a per-channel SiLU lookup between the
  folded matmul and the Q/K/V split. Tracked as future work alongside
  the GDN recurrent-state proof. Pinned by
  `test_fold_post_silu_is_dropped_pinning` so the approximation is
  explicit and cannot silently change.
- **LogUp `_with_data` verifiers depend on audit-mode-recomputed
  inputs/outputs (M3).** `verify_lookup_with_data` and
  `verify_lookup_ef_with_data` accept `external_data: Option<(&[u32],
  &[u32])>` and absorb the values directly into the FS transcript to
  derive the LogUp α/β challenges. In the current architecture the
  caller is always a `verify_*_layer` function that has just recomputed
  the canonical trace via `*_forward(...)`, so the absorbed values are
  the model's actual lookup arguments. In a future "true ZK" mode where
  the verifier no longer recomputes the trace, the lookup verifier
  would need to bind these values to a commitment via
  `prove_mle_eval_bound` rather than accept them as plain bytes.
  Pinned by `test_lookup_ef_with_data_tamper_rejects` which confirms
  the binding mechanism is functioning today (a tampered input vector
  diverges the recomputed α from `proof.alpha` and the verifier
  rejects).
- **TVM integration is design-only, not implemented (C5).** PAPER.md
  describes a TVM-as-ZK-backend lowering. The shipping prover does not
  call TVM; `python/tvm_to_gkr/` is the *target name* but the current
  pipeline is direct PyTorch → Mersenne-31. Treat the TVM section as
  a roadmap item.
- **Competitor numbers are vendor-reported, not re-run on M4 Max (C3).**
  The DeepProve / EZKL rows in the comparison table above come from the
  respective project READMEs and papers. We have not re-executed those
  systems on the same hardware in the same environment. A like-for-like
  benchmark harness is queued; until it lands, treat those rows as
  approximate ratios rather than precise measurements.
- **Attention at `seq_len=1` has an empty `row_proofs: vec![]`
  binding mechanism that depends on audit-mode verifier (S4).** At
  seq_len=1 the layer proof does NOT include a sub-proof binding
  `attn_out` to `v`. The current architecture is *audit mode* — the
  verifier has the weights and re-runs `qwen_forward(x, weights, config)`
  internally. Every sub-proof's MLE evaluation is checked against the
  *verifier's* canonical trace, NOT the prover's declared trace. So
  even though the prover *could* construct a proof against a tampered
  `attn_out`, the verifier's canonical recomputation produces the
  correct attn_out and the sub-proof bindings mismatch — verifier
  rejects. Pinned both ways:
  `test_qwen_seq_len_1_attn_proof_is_empty_pinning` (proof structure
  is empty at seq_len=1) and
  `test_qwen_seq_len_1_attn_out_tamper_rejected_via_canonical_trace`
  (tampered attn_out is rejected by audit-mode recomputation).
  When the architecture moves toward true ZK with hidden weights —
  removing the verifier's `qwen_forward` call — the binding mechanism
  is gone and a proof-level v↔attn_out consistency sub-proof will be
  required. The full GDN-degenerate form `attn_out = β·(q·k)·v` is
  also queued alongside the recurrent-state proof. Prefill (≥ 2 tokens)
  is fully proven by the row-attention sumcheck.
- **`--fast` is below funding-grade soundness.** ~31-bit per round is
  acceptable for dev iteration, audit, and CI smoke tests, but it is
  NOT a production soundness level. The default path (EF, ~124-bit) is
  what the headline benchmarks use.

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