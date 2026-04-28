# Roadmap

Status snapshot for the Rust prover. Each row links a deliverable to its current state, the failure mode it catches, and the gap that remains.

## Verifier modes

| Mode | Default? | Soundness | Use case | Verifier holds weights? |
|------|----------|-----------|----------|------------------------|
| EF (M31⁴) | yes | ~124-bit per round | audit-mode attestation | yes (re-execute) |
| Base (M31) | flag `--fast` | ~31-bit/round, ~80–100 bit composed | dev / fast audit only | yes |
| Basefold PCS | flag `--features pcs` | + 219-bit proximity | weight-binding | yes (still recomputes trace) |
| Basefold PCS-full | flag `--features pcs-full` | + 219-bit proximity | weight-commit binding | partial (commit suffices for weights; activations still re-executed) |

`--fast` is below funding-grade soundness and prints a warning at startup. `pcs-full` upgrades weight binding but does **not** by itself close the audit-mode gap — see `PAPER.md` §7.4 for the migration plan to true-ZK verifier-hidden-weights mode.

---

## Soundness

Mode-independent. Must hold in every configuration.

| ID | Item | Status |
|----|------|--------|
| S1 | `Transcript::squeeze` full-digest rejection-sample (no biased `% p`) | DONE |
| S2 | All inner MLE evals thread the main transcript (no fresh `Transcript::new`) | DONE |
| S3 | Qwen prove absorbs all dimensions before challenges (config-injection guard) | DONE |
| S4 | Seq1V audit-mode binding: tampered `attn_out` rejected via canonical-trace recomputation | DONE |
| S5 | RMSNorm QR perturbation amount transcript-bound | DONE |

## Mode semantics

| ID | Item | Status |
|----|------|--------|
| M1 | EF (124-bit) is the default; `--fast` opts into base-field | DONE |
| M2 | Mode semantics documented in README + PAPER | DONE |
| M3 | LogUp `_with_data` audit-mode digest binding (blake3 of canonical inputs/outputs) | DONE (defense-in-depth; full `(WeightCommitment, MleEvalProof)` migration queued with true-ZK) |

## Engineering hygiene

| ID | Item | Status |
|----|------|--------|
| E1 | Mersenne31 transmute: compile-time size+align asserts, LE-only target gate, centralized helper | DONE |
| E2 | Binary parser: bounded dimensions + `checked_mul` | DONE |
| E3 | `protocol.rs` + `server.rs` request loop replace panics with `Result` | DONE |
| E5 | Metal `cmd.status()` checks + Send/Sync invariant documented | DONE |
| E6 | `placeholder_qwen_commitments` gated behind `cfg(test)` | DONE |
| E7 | `OpDesc` god-struct → `Op` tagged-union enum (parse-time variant safety) | PARTIAL — `Op` enum + conversion shim shipped; full match-on-`Op` migration is mechanical follow-up |
| E8 | `WeightCommitment` digest discriminant clarified | DONE |

## Tests

| ID | Item | Status |
|----|------|--------|
| T1 | Asymmetric V dim test (kv_dim=8, v_dim=16) | DONE |
| T2 | Forgery test: substitute committed weights, verifier rejects | DONE |
| T3 | `cargo-fuzz` target on binary parser | DEFERRED (parser has no `unsafe`, all arithmetic is checked) |
| T4 | Tampered transcript byte-flip → verifier rejects | DONE |

## Reproducibility / DevOps

| ID | Item | Status |
|----|------|--------|
| R1 | `Cargo.lock` committed for both crates | DONE |
| R2 | `rust-toolchain.toml` pinned (channel 1.85.0) | DONE |
| R3 | Remove `target-cpu=native` from `.cargo/config.toml` | DONE |
| R4 | Minimal `.github/workflows/ci.yml` (ubuntu-latest + macos-latest) | DONE |
| R5 | `transformers >= 4.49` pinned | DONE |
| R6 | `torchvision` added to requirements | DONE |
| R7 | Metallib embedded via `include_bytes!` | DONE |
| R8 | `examples/quickstart.sh` + `square_circuit.json` + Rust regression | DONE |
| R10 | `Cargo.toml` `repository` field set | DONE |
| R11 | Docker memory limits documented per benchmark tier | DONE |

## GDN math correctness

| ID | Item | Status |
|----|------|--------|
| G1 | `causal_tap_index` parametrized + Python pytest pins both branches | DONE (default flip queued behind HF forward-pass match) |
| G2 | `_fold_conv_into_proj` post-SiLU drop documented as approximation + pytest pin | DONE |
| G3 | seq_len=1 q/k proofs documented as soundness-neutral perf overhead + pinned | DONE |

---

## Phase 10 — post-test-infrastructure deliverables

| ID | Item | Sized | Status |
|----|------|-------|--------|
| P10-1 | EZKL competitor harness on M4 Max | 1–2 d | **DONE** — `bench/{run_ezkl.py, compare.py, README.md, results/*.csv}`. Dense-4M MLP: 26.3 ms / 0.20 ms / 2.1 KB vs EZKL 62.5 s / 1.28 s / 170 KB. ~2,375× / ~6,409× / ~80×. Reproducible via `python bench/compare.py`. |
| P10-2 | README headline + audit-mode-vs-true-ZK on slide 1 | ~1 d | **DONE** — root README opens with the audit-mode-honest framing; benchmark table follows; migration path to true-ZK documented. |
| P10-3 | Seq1VConsistency proof-level v↔attn_out at seq_len=1 | 1 d | **DONE** — Both Qwen prove paths squeeze `r` and commit MLE evals; verifier mirrors with GDN identity check. GQA r-coordinate slicing fix: `[group_prefix(log_kv) ‖ d_suffix(log_d)]`, not contiguous prefix. +3 tamper regressions. |
| P10-4 | M3 commitment binding (LogUp `_with_data` → MleEvalProof) | 2 d | **PARTIAL** — audit-mode digest binding (`external_data_digest: [u8; 32]`) on both LookupProof variants, blake3 of domain-separated byte stream, checked before sumcheck. Legacy zero-digest sentinel kept for back-compat. Full structural `(WeightCommitment, MleEvalProof)` binding queued with the true-ZK verifier-without-weights migration. |
| P10-5 | Loom recording — fresh clone → quickstart → 4B benchmark | 2–4 h | TODO (manual) |
| P10-6 | 9B memory-aware scheduler | 3–5 d | **PARTIAL** — `par_chunk_size` formula in `pipeline/prove.rs`: per-layer weight bytes × 4× overhead vs 32 GB peak (overridable via `ZKMLP_TARGET_PEAK_GB`). 9B 17.8 → 14.8 s. Weight streaming follow-up tracked. |
| P10-7 | GDN recurrent-state proof | 3–5 d | **DONE (audit-mode)** — `proving/gdn_recurrence.rs`: pure-M31 integer recurrence, blake3 trajectory digest over `(S_t for t∈0..=T, o_t for t∈0..T)` + shape footer, final-state weight commitment, two MLE-eval claims. Verifier re-runs forward and asserts bytewise digest equality + Fiat-Shamir mirror. Proof size constant in T (~150 B). 10 unit tests + cross-language proptest harness vs `tests/reference/gdn_recurrence.py`. Structural sumcheck decomposition queued with true-ZK migration. |
| P10-8 | `OpDesc` → tagged-union enum refactor | 2–3 d | **PARTIAL** — `Op` enum (`#[serde(tag = "type")]`) shipped alongside legacy `OpDesc`. Parse-time variant safety captured (malformed JSON now rejected at deserialization). Full match-on-`Op` migration across dispatch sites is mechanical follow-up. |
| P10-9 | GPU default-on for macOS + Metal kernels for hot paths | ~1 wk | **DEFERRED** — needs target_os+target_arch conditional defaults, runtime Metal-device detection, Apple-Silicon CI runner (current hosted `macos-latest` is Intel x86_64, panics on metal link). `cargo install --features metal_gpu` is the documented opt-in. |
| P10-10 | Server amortization with warm caches | 4–6 d | **PARTIAL** — process-global SiLU + sigmoid table caches in `proving/lookup.rs` keyed by scale. 4B prove 376 → 328 ms. Table-index HashMap cache + weight-commitment-by-hash cache + all-qwen layer-data preallocation are tracked follow-ups. |

---

## Phase 11 — bulletproof test infrastructure

The motivation: a soundness bug (the GQA r-coord slicing error caught in P10-3) passed every hand-written test because tests written by the same author as the prover do not reliably catch MLE-coordinate-slicing errors. Phase 11 ships a property-test harness with an independently-authored numpy reference + tamper coverage matrix so this class of bug fails fast.

| ID | Deliverable | What it catches | Status |
|----|-------------|-----------------|--------|
| P11-1 | `tests/reference/*.py` — numpy implementations of every provable op (matmul, RMSNorm, GELU/SiLU/sigmoid lookup, attention with GDN + GQA branches incl. head replication, LayerNorm, residual, GDN recurrence forward) | Quantization drift; prover-side index bugs | **DONE** — 5 modules, MSB-first folding pinned against Rust fixtures, overflow-safe. |
| P11-2 | `tests/property/harness.rs` — proptest-driven Rust ↔ numpy differential. Random `(num_q_heads, num_kv_heads, d_head, seq_len, seed)`; runs Rust prover end-to-end; runs P11-1 reference; asserts outputs match within quantization tolerance + verifier accepts. | Coverage gaps in hand-written tests; misuse across config branches (the GQA-bug class) | **DONE** — 4 tests, 60 cases + smoke + branch-coverage in 2.40 s. 200-seed enumeration: GDN=47, GQA=82, asymmetric-V=71. |
| P11-3 | `tests/property/mle_relations.rs` — isolated tests for every MLE identity (`MLE(replicate(v), r) == MLE(v, [r_g ‖ r_d])`, `MLE(eq_table, r, x) == eq(r, x)`, `MLE(matmul(W, x), r) == Σ MLE(W,...)·MLE(x,...)`). Replays the GQA bug as a passing-then-reverted test. | MLE-coordinate-slicing bugs (the GQA bug class) | **DONE** — 7 proptest tests at 100 cases each. Bug-replay verified: deliberately reintroducing `r[..log_v]` contiguous-prefix fails `prop_mle_replicate_identity` at seed=0 in <60 s. |
| P11-4 | `tests/tamper_matrix/main.rs` — codegen walking every proof type's serde fields; one tamper test per field × config branch × proof type. | "I forgot to test the GQA branch of seq1_consistency" → impossible by construction | **DONE** — 144 tests: 24 fields × 3 branches × 2 proof types (`QwenLayerProof` base + EF). |
| P11-5 | `tests/property/quantization_invariants.rs` — M31-in-range after every layer, RMSNorm bounded, attention-bounded-by-v, GDN state norm bounded across timesteps (regression target for P10-7). | Compounding quantization errors | **DONE** — 7 tests in 0.09 s. 64-step recurrence proxy locks the regression target for P10-7. Structural invariant: every `attn_out` entry literally equals some `v` entry (covers GDN identity AND GQA head-replication). |
| P11-6 | CI: run P11-2/-3/-4/-5 on every PR. `PROPTEST_CASES=100` PR cap, `10000` nightly soak. Fail-fast on any rejection. | Regressions across commits | **DONE** — `.github/workflows/ci.yml` with `actions/setup-python@v5` + numpy install before `cargo test --release`; new `phase11-soak` job (cron `0 6 * * *` + `workflow_dispatch`, 240-min timeout); top-level `concurrency:` block. |

After Phase 11:
- P10-7 (GDN recurrent state) becomes safe to write — wrong MLE coordinate slice caught by P11-3 in <60 s; wrong outer-product decomposition caught by P11-2; wrong quantization scale caught by P11-5.
- True-ZK migration becomes auditable — P11-2 fails the moment a binding goes implicit, because the property holds at the MLE level not the audit-mode level.

**Acceptance gates:**
1. **Replay-the-bug**: deliberately reintroduce the GQA r-coord bug; P11-3 fails on `mle_replicate_identity` within 60 s.
2. **Replay S5**: deliberately reintroduce the perturbation_delta-not-absorbed bug; P11-4 tamper test rejection within 60 s.
3. **Soak**: 24-h proptest run via `cargo test --release -- --ignored long-soak` exercises ~500 K random configs without finding any failure.
4. **CI**: `.github/workflows/ci.yml` includes the four targets; PR diff visibly runs them.

---

## Out of scope

Explicitly deferred after weighing cost vs. ROI for the current scope:

- **`cargo-fuzz` on binary parser** — parser has no `unsafe`, all arithmetic is checked; marginal value
- **Migration from Mersenne-31 to Goldilocks/BabyBear** — EF default already gives funding-grade soundness; would lose NEON SIMD + Basefold parameter set
- **SNARK recursion / proof composition** — premature; not on the critical path for single-host inference
- **CUDA path** — defer until an Apple-Silicon-only customer requires more
- **Multi-host distribution** — premature; single-host with P10-6 + P10-9 hits the ceiling first
- **True-ZK migration (verifier without weights)** — 8–12 weeks (1 quarter); structural rewrite, not a feature flag. See `PAPER.md` §7.4 for the cost extrapolation.
- **`pcs-full` optimization** — not the production path; ~30× slower than default by design

---

## Benchmark snapshot

Apple M4 Max, EF default (~124-bit per round), audit-mode.

| Model | Params | Layers | Prove | Verify | Proof |
|-------|--------|--------|-------|--------|-------|
| Dense MLP | 4.2M | — | 24.7 ms | 0.1 ms | 2.1 KB |
| GPT-2 | 124M | 12 | 253 ms | 5.2 ms | 963 KB |
| Qwen3.5-4B (GDN+attn hybrid) | 4B | 32 (24 GDN + 8 full-attn) | 376 ms | 74 ms | 1.25 MB |
| Qwen3.5-9B (GDN+attn hybrid) | 9B | 32 | 17.8 s | 2.6 s | 1.25 MB |

9B is the M4-Max scale ceiling; per-layer weights at d_model=4096 saturate RAM and degrade to swap-bound serial. Closer-to-linear scaling needs weight streaming + a memory-aware scheduler, tracked under P10-6.

EZKL head-to-head on Dense-4M MLP (audit-mode use case): ~2,375× faster prove, ~6,409× faster verify, ~80× smaller proof. See `bench/README.md` for the comparison harness; `PAPER.md` §7.4 for the verifier-model side-by-side that puts those numbers in context.
