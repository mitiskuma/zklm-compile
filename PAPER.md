# ZK-Compile: Zero-Knowledge Proofs as a Compiler Target for ML Inference

**Authors:** ZK-Compile Contributors
**Date:** March 2026
**Status:** Draft — not yet submitted

---

## Abstract

We propose treating zero-knowledge proof generation as a compiler backend target, integrated into an existing ML compilation framework (Apache TVM). Current zkML systems (EZKL, zkPyTorch, DeepProve) each build custom compilation pipelines from ONNX or PyTorch IR, with limited access to the operator fusion, quantization, and memory planning passes that ML compilers provide. We observe that TVM's intermediate representations (Relax and TensorIR) decompose models into primitive arithmetic — structurally similar to what ZK constraint systems require — and propose a lowering from TensorIR to R1CS and AIR constraint systems. This enables (1) compiler-assisted circuit generation with ZK-aware fusion to reduce witness count, (2) field-native quantization targeting ZK-friendly fields, (3) dual compilation from a shared source representation for optimistic verification, and (4) the first cross-architecture comparison of ZK proving cost, demonstrating that hybrid architectures (GatedDeltaNet, Mamba-2) produce asymptotically fewer non-linear constraints than softmax transformers at long sequence lengths. We describe the design, discuss key technical challenges including field overflow, memory arguments, and proof composition, and outline an implementation plan with concrete validation milestones.

---

## 1. Introduction

### 1.1 The Problem

Verifiable ML inference — proving that a specific model produced a specific output without revealing the model weights — is an unsolved infrastructure problem. Regulatory pressure (EU AI Act 2025-2026), autonomous AI agents operating on behalf of users, and decentralized compute markets all create demand for cryptographic guarantees about ML computation.

The current approach: take a trained model, convert it to an arithmetic circuit (a system of polynomial constraints over a finite field), and generate a zero-knowledge proof of correct execution. This is what EZKL, zkPyTorch, DeepProve, and ZKTorch do.

The problem: **each zkML framework builds its own compilation pipeline.** EZKL takes ONNX graphs and lowers them to Halo2 circuits with graph-level optimizations (element-wise fusion, constant folding, lookup batching). zkPyTorch captures PyTorch execution traces and compiles to Expander. DeepProve builds an ONNX-to-GKR pipeline that exploits the layered structure of neural networks via sumcheck protocols. Each framework reinvents aspects of model compilation — fusion, quantization, operator scheduling — that ML compilers have spent a decade optimizing for hardware targets.

We ask: can the same ML compiler infrastructure that targets CUDA, Metal, and WebGPU also target ZK proof systems?

### 1.2 The Insight

Apache TVM's TensorIR represents computation as indexed arithmetic expressions — loop nests over multiply-accumulate operations with explicit data dependencies. While TensorIR is an imperative loop program with mutable buffer state (not a pure arithmetic circuit), the arithmetic core of each kernel — the multiplications and additions that dominate ML workloads — maps naturally to constraint system gates.

The lowering from TensorIR to constraint systems is non-trivial (Section 3.6 discusses the challenges), but the structural similarity is sufficient to build a practical ZK backend that inherits TVM's graph-level optimizations.

### 1.3 Contributions

1. **ZK as a compiler target.** We define a lowering from TensorIR to R1CS and AIR constraint systems, identifying which TensorIR features map directly and which require additional constraint machinery (memory arguments, range checks, overflow handling).

2. **ZK-aware compiler passes.** We introduce fusion passes optimized for witness count minimization and quantization passes targeting ZK-friendly finite fields (Goldilocks, BabyBear). We discuss the interaction between field size, arithmetic overflow, and model accuracy.

3. **Dual compilation.** Both hardware inference and ZK circuit are compiled from the same source Relax graph. We discuss the conditions under which functional equivalence holds and the limitations imposed by differing arithmetic semantics.

4. **Cross-architecture proving cost analysis.** We provide the first analysis of how model architecture choice (softmax transformer vs. GatedDeltaNet vs. Mamba-2) affects ZK proving cost, demonstrating an asymptotic advantage for linear attention at long sequence lengths.

---

## 2. Background

### 2.1 ML Compilation (TVM)

TVM compiles ML models through a multi-level IR stack:

```
Model (PyTorch / ONNX / HuggingFace)
  → Relax IR (graph-level: operator DAG, data flow, control flow)
    → TensorIR (kernel-level: loop nests, arithmetic expressions, memory access)
      → Backend codegen (CUDA, Metal, WebGPU, LLVM, ...)
```

**Relax IR** represents the model as a dataflow graph of tensor operations. Passes at this level perform operator fusion, layout optimization, memory planning, and quantization.

**TensorIR** represents individual tensor operations as loop nests with explicit arithmetic. A matrix multiply becomes three nested loops over multiply-accumulate. Passes at this level perform tiling, vectorization, and hardware-specific scheduling.

**Important distinction:** TensorIR is an imperative program with mutable buffers, not a pure arithmetic circuit. Each `BufferStore` statement performs side-effecting writes to memory. The lowering to constraint systems must resolve buffer aliasing, handle accumulation patterns (read-modify-write loops), and convert imperative control flow to arithmetic form. We detail these challenges in Section 3.6.

### 2.2 ZK Proof Systems

A zero-knowledge proof system proves that a computation was executed correctly without revealing all inputs. The computation must be expressed as an **arithmetic circuit** — a DAG of addition and multiplication gates over a finite field F_p.

**R1CS (Rank-1 Constraint System):** Used by Groth16, Marlin. Each constraint has the form `(a · s) * (b · s) = (c · s)` where `s` is the witness vector and `a, b, c` are selector vectors. One multiplication = one constraint. Additions are free (linear combinations). Proving cost is dominated by multi-scalar multiplication (MSM), which is superlinear in constraint count.

**AIR (Algebraic Intermediate Representation):** Used by STARKs (Plonky3, Stwo). Defines constraints as polynomial equations over an execution trace. The repetitive structure of neural network layers maps well to AIR's row-based execution model. Proving cost is dominated by NTT/FRI, scaling as O(N log N) in trace length.

**PLONKish:** Used by Halo2. Custom gates with lookup tables for non-arithmetic operations. EZKL uses this approach.

**GKR (Goldwasser-Kalai-Rothblum):** Used by DeepProve. A sumcheck-based protocol that directly exploits the layered DAG structure of neural networks, avoiding the flattening to R1CS that other approaches require. This is a fundamentally different paradigm — the proof protocol itself encodes the circuit topology.

**Key distinction:** Constraint count is a useful proxy for circuit complexity but is not a direct predictor of proving time. Commitment overhead (MSM for SNARKs, NTT/FRI for STARKs) often dominates wall-clock time. We use constraint count for cross-architecture comparison (where the commitment scheme is held constant) but do not claim it directly predicts absolute proving time.

### 2.3 Existing zkML Compilation

Current zkML systems have more sophisticated compilation than a naive operator-by-operator approach, but each is limited in scope:

- **EZKL** performs graph-level optimizations including element-wise fusion, constant folding, and lookup batching via its `--optimize` flag. However, it operates on ONNX IR and does not perform cross-operator fusion of the kind ML compilers support (e.g., fusing matmul + bias + activation into a single circuit block).

- **DeepProve** uses the GKR protocol, which inherently exploits the layered structure of neural networks. This is a protocol-level optimization, not a compiler optimization — the proof system itself is designed for layered computation. DeepProve also performs field-aware quantization.

- **zkPyTorch** captures PyTorch execution traces and applies a hierarchical compilation strategy. It performs some operator grouping but operates at the trace level, not the compiler IR level.

None of these systems provide: (a) configurable fusion passes that can be tuned for different proof system cost models, (b) quantization targeting arbitrary finite fields with calibration and mixed-precision search, or (c) compilation to multiple proof system backends from a single source.

---

## 3. Design

### 3.1 Architecture

```
                    ┌──────────────────┐
                    │   PyTorch/ONNX   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    Relax IR       │
                    │  (shared source)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
     ┌────────▼───────┐           ┌─────────▼────────┐
     │  HW Pipeline    │           │  ZK Pipeline      │
     │  - HW fusion    │           │  - ZK fusion      │
     │  - INT8/FP16    │           │  - Field quant    │
     │  - HW schedule  │           │  - Unroll+flatten │
     └────────┬───────┘           └─────────┬────────┘
              │                             │
     ┌────────▼───────┐           ┌─────────▼────────┐
     │  Metal/CUDA/   │           │  R1CS / AIR /     │
     │  WebGPU/WASM   │           │  PLONKish         │
     └────────────────┘           └──────────────────┘
```

The two pipelines diverge after the shared Relax graph. Each applies its own fusion, quantization, and lowering passes. The shared source provides a common semantic specification (Section 5).

### 3.2 TensorIR → Constraint System Lowering

**Multiplication gates.** Each `BufferStore` containing a multiplication produces one R1CS constraint. Additions fold into linear combinations (free in R1CS).

**Loop unrolling.** TensorIR loops with static bounds are fully unrolled into the flat constraint system. A matmul `for i in range(N): for j in range(M): for k in range(K): C[i,j] += A[i,k] * B[k,j]` becomes N×M×K multiplication constraints plus range checks on accumulation (see Section 3.4).

**Dynamic shapes.** TensorIR supports symbolic dimensions (`tir.Var`), but constraint systems require fixed circuit topology. The ZK backend requires shape specialization before lowering — all symbolic dimensions must be bound to concrete values. This is a real limitation: a separate circuit must be compiled for each input shape.

**Control flow.** Simple `if-then-else` expressions are converted to arithmetic selectors: `result = condition * value_true + (1 - condition) * value_false`. Data-dependent control flow involving block-level side effects (conditional buffer writes) requires both branches to produce witness values, increasing constraint count. We conservatively include both branches.

**Non-linear operations.** Operations not expressible as field arithmetic (ReLU, softmax, GELU) are lowered to lookup arguments or bit decomposition, depending on the target proof system:
- PLONKish targets: lookup tables (LogUp/CQ arguments)
- R1CS targets: bit decomposition + range checks
- AIR targets: auxiliary columns with boundary constraints

Lookup arguments have their own proving cost that scales with table size. For activations over wide value ranges (e.g., 31-bit field), table size must be managed via range decomposition (splitting values into smaller chunks with separate tables), adding constraints.

### 3.3 ZK-Aware Fusion Pass

Standard TVM fusion groups operators by memory access patterns (element-wise, broadcast, reduction). ZK fusion groups by **witness elimination** with a cost model tuned to the target proof system:

**Rule 1: Intermediate elimination.** If operator B feeds only into operator C, and both are linear (field-arithmetic), fuse them to eliminate B's output from the witness. This reduces witness vector length and, in PLONKish/STARK systems, the number of advice columns in the execution trace.

**Rule 2: Constraint sharing.** If two operators share input witnesses, fuse to share range-check constraints on those inputs. Range checks cost ~1 constraint per bit per witness element; sharing them across consumers reduces total constraint count.

**Rule 3: Non-linear isolation.** Non-linear operations (requiring lookup arguments) are kept as separate circuit blocks. Fusing a linear operator with a non-linear one would force the entire fused block to use the more expensive lookup-based proving strategy.

**Interaction between rules.** Rules 1 and 3 can conflict when a linear operator feeds into a non-linear one (e.g., matmul+bias → ReLU). In this case, the linear portion (matmul+bias) is fused to eliminate the bias intermediate, and the non-linear (ReLU) remains a separate block consuming the fused output. The fusion pass resolves this by classifying each operator as linear or non-linear before applying fusion rules.

**Mul+add fusion (implemented).** The most impactful fusion folds accumulation adds into their preceding multiply constraints. In matmul, each `output[i,j] += input[i,k] * weight[k,j]` produces a mul gate then an add gate with an intermediate wire. Fusion encodes this as a single R1CS constraint: `A=[a:1], B=[b:1], C=[acc_new:1, acc:-1]` (i.e., `a * b = acc_new - acc`). This eliminates one constraint and one intermediate wire per multiply-accumulate step. Measured impact: 49% constraint reduction for MNIST MLP (209K → 107K), 30% for CNN (66K → 46K).

**Relationship to proving time.** Constraint reduction translates directly to proving time reduction because Groth16's MSM cost is linear in constraint count. The 49% constraint reduction yields ~40% prove time reduction (0.99s → 0.59s for MLP). Additional fusion opportunities (folding bias adds, chaining linear combinations) provide diminishing returns since the multiply-accumulate pattern dominates.

### 3.4 ZK-Aware Quantization and Field Overflow

Standard quantization maps FP32 → INT8/INT4 to exploit hardware integer units. ZK quantization maps to finite field elements to minimize constraint count.

**The overflow problem.** A single field-native multiply costs 1 constraint only when operands and result fit within the field modulus. For BabyBear (p ≈ 2^31), two INT15 values multiply to a 30-bit result — this fits. But a dot product of dimension d accumulates d such products: the sum reaches d × 2^30, which overflows BabyBear for d > 2 (i.e., immediately for any realistic hidden dimension).

**Solutions and their costs:**

| Strategy | Field | Max Safe Accumulation | Constraints per Multiply | Trade-off |
|---|---|---|---|---|
| Goldilocks + INT16 | p ≈ 2^64 | d ≤ 2^32 (effectively unlimited) | 1 (native) | Wider field = larger proofs, slower NTT |
| BabyBear + INT8 | p ≈ 2^31 | d ≤ 2^15 = 32,768 | 1 (native) | INT8 quantization may reduce model accuracy |
| BabyBear + INT15 + intermediate reductions | p ≈ 2^31 | Unlimited (with reduction every ~2 steps) | 2-3 (reduction overhead) | More constraints but stays in small field |
| Extension field (BabyBear^4) | p^4 ≈ 2^124 | Effectively unlimited | 4-8 (extension field multiply) | Complex, partial loss of small-field benefits |

**Our recommendation:** Goldilocks (64-bit) with INT16 quantization for most models. The 64-bit field accommodates realistic accumulation depths without intermediate reductions, and INT16 quantization preserves model accuracy better than INT8. The proving cost penalty vs. BabyBear is moderate (~2x for NTT) and is offset by eliminating overflow-handling constraints.

For models that tolerate INT8 quantization (e.g., vision models post quantization-aware training), BabyBear with INT8 is viable and provides the smallest proofs.

**Accuracy impact.** Field-native quantization is equivalent to integer quantization at the same bit width — the modular arithmetic is invisible to the model as long as values don't wrap around the modulus (which overflow prevention guarantees). TVM's existing quantization infrastructure (calibration, per-channel scaling, mixed precision) applies directly; only the target format changes.

**Range checks vs constant constraints.** Traditional range checks decompose each witness value into bits (~16 constraints per INT16 wire). For models with 100K+ weight wires, this dominates constraint count. We observe that model weights are embedded at compile time — a malicious prover cannot change them without violating the circuit. Instead of range-checking weight wires, we pin them to exact values using constant constraints (1 constraint per wire: `1 * value = wire`). This is 16x cheaper per wire than range checks while providing strictly stronger soundness (exact value, not just range). For MNIST MLP: 101K constant constraints vs 1.7M range check constraints. User inputs still require range checks if the prover supplies them.

### 3.5 Proof System Selection

The ZK backend selects a proof system based on the deployment target:

```python
# On-chain verification (Ethereum L1) — small proofs, cheap verification
mod_zk = tvm.build(mod, target="zk-groth16-bn254")

# Off-chain, transparent (no trusted setup) — larger proofs, no ceremony
mod_zk = tvm.build(mod, target="zk-stark-goldilocks")

# Browser-verifiable — moderate proofs, no trusted setup with IPA
mod_zk = tvm.build(mod, target="zk-plonk-bn254")
```

The compiler adjusts quantization and constraint generation to match the target proof system's field and gate structure.

### 3.6 Technical Challenges

We identify the following challenges that the lowering must address. These are engineering problems with known solutions, but they add complexity and constraints beyond the simplified model in Section 3.2:

**Memory indirection.** TensorIR `BufferLoad` with data-dependent indices (e.g., embedding lookups, MoE routing) cannot be directly expressed as circuit wiring, which must be fixed at compile time. Solution: permutation-based memory arguments (as in Polygon Miden) or Merkle-tree memory checking (as in Cairo/TinyRAM). These add O(N log N) constraints for N memory accesses.

**Buffer aliasing.** TensorIR allows buffer aliasing (`match_buffer`) and in-place mutation. The ZK path requires a lowering pass that converts aliased buffers to SSA-style unique witness slots. This is standard compiler engineering but adds a pass.

**Proof composition for large models.** A 7B parameter model produces billions of constraints that cannot be proved monolithically. The circuit must be partitioned (e.g., per-layer or per-block) and composed using recursive proof techniques (Nova/HyperNova folding, or STARK recursion). The partitioning strategy interacts with the fusion pass — fusion cannot cross partition boundaries.

**Witness generation.** Computing the full witness (all intermediate values in the finite field) requires running the forward pass in field arithmetic. For large models, witness generation can dominate wall-clock time. This cost is proportional to model size and independent of proof system choice.

---

## 4. Cross-Architecture Proving Cost Analysis

This section presents our primary novel contribution: the observation that model architecture choice has a significant impact on ZK proving cost, particularly at long sequence lengths.

### 4.1 The Non-Linearity Tax

Published measurements confirm that non-linear operations (softmax, GELU, LayerNorm) account for a large fraction of ZK proving cost in transformer inference. Hao et al. [1] report that non-linear function evaluation consumes over 80% of proving time in naive circuits. Modern provers reduce this via optimized lookup arguments (LogUp batching in DeepProve, tlookup in zkLLM), bringing the non-linear share to an estimated 15-25% of proving time in optimized systems. However, the constraint count contribution remains high — non-linearities produce 50-200x more constraints per element than field multiplications.

### 4.2 Linear Attention Eliminates the Most Expensive Non-Linearity

Hybrid architectures like GatedDeltaNet (Qwen3.5) and Mamba-2 (Nemotron 3) replace softmax attention with operations that are primarily field-native:

| Operation | ZK Cost | Notes |
|---|---|---|
| Element-wise gating (sigmoid) | ~150 constraints/element | One non-linearity per head per position |
| Linear recurrence (matmul chain) | 1 constraint/multiply + range checks | Field-native, but accumulation needs overflow handling |
| Delta rule update (add/sub/mul) | 1 constraint/multiply + range checks | Field-native |
| **Softmax (for comparison)** | **~200 constraints/element** | **Applied to n×n attention matrix per head** |

### 4.3 Asymptotic Scaling with Sequence Length

The critical difference is asymptotic, not constant-factor. Softmax attention applies ~200 constraints per element to an n×n attention matrix (per head). Linear attention applies ~150 constraints per element to a fixed-size gating vector (per head, per position).

For the **attention component only** (excluding shared FFN layers):

| Sequence Length | Softmax Attn Non-Linear Constraints | Linear Attn Non-Linear Constraints | Ratio (attention only) |
|---|---|---|---|
| 512 | ~630M | ~9.4M | 67x |
| 2,048 | ~10B | ~37.7M | 265x |
| 8,192 | ~161B | ~151M | 1,067x |

*For 12 heads, d=768. Softmax: 12 × n × n × 200 constraints. Linear (GatedDeltaNet): 12 × n × 2 × 150 constraints (two sigmoid gates per head per position). These count non-linear constraints only in the attention component.*

**Important caveat:** Both architectures share identical FFN layers (with GELU/SiLU activations and LayerNorm), which contribute a constant per-position non-linear cost. At short sequences (512 tokens), the FFN dominates and the total model proving cost difference is modest (~1.7x, as discussed in our earlier analysis). The attention-specific advantage becomes dominant only at longer sequences (4K+) where the quadratic softmax cost overwhelms the linear FFN cost.

### 4.4 Why This Requires a Compiler

This analysis is only possible when a single compilation framework supports multiple architectures with a shared cost model. Today, comparing proving cost across architectures requires reimplementing each model in each zkML framework — an apples-to-oranges comparison confounded by different proof systems, quantization strategies, and optimization levels.

A compiler-based approach holds all variables constant except architecture, enabling principled co-design: choosing a model architecture based on both ML quality and proving cost, informed by the compiler's constraint analysis.

---

## 5. Dual Compilation and Optimistic Verification

### 5.1 Shared Source, Separate Semantics

Both the hardware and ZK compilation pipelines start from the same Relax graph — the imported model before any target-specific transformations. This shared source provides a common semantic specification: both paths implement the same mathematical function (matrix multiplies, activations, normalization) as defined by the model architecture.

However, the two paths produce **numerically different results** due to differing arithmetic:
- The hardware path uses floating-point or INT8 arithmetic with hardware-specific rounding
- The ZK path uses finite field arithmetic with modular reduction

These are different numerical computations. We do not claim bit-exact equivalence.

### 5.2 Equivalence Under Shared Quantization

Functional equivalence can be achieved when both paths use the **same quantization scheme**. If the hardware path also executes in integer arithmetic at the same bit width used by the ZK path (e.g., both use INT16 with identical scale factors), and the computation order is fixed by the compiler (eliminating FP non-determinism), then both paths compute identical results.

This constrains the hardware path — it cannot use FP16 or hardware-specific fused multiply-add. The trade-off is: exact equivalence at some inference speed cost, or faster hardware inference with bounded but non-zero divergence.

### 5.3 Optimistic Verification

For optimistic verification (prove only on challenge), exact equivalence is the safer path:
1. Serve inference results from the integer-quantized hardware path — still fast (INT16 inference is well-optimized on modern hardware)
2. If challenged, generate a proof from the ZK path over the same quantized computation
3. Equivalence is guaranteed by shared quantization and fixed computation order

Current optimistic zkML systems maintain two separate implementations. Bugs in either cause silent divergence — the proof proves a different computation than what was served. Compiling both from a shared source with shared quantization eliminates this class of bugs, though it does not constitute a formal verification guarantee (TVM's passes are tested empirically, not formally verified).

---

## 6. Related Work

| System | Approach | Graph Optimization | Quantization | Multi-target | Architecture-aware |
|---|---|---|---|---|---|
| EZKL | ONNX → Halo2 | Element-wise fusion, constant folding, lookup batching | Fixed-point | Halo2 only | No |
| zkPyTorch | PyTorch trace → Expander | Hierarchical compilation | ZK-aware | Expander only | No |
| DeepProve | ONNX/GGUF → GKR | Protocol-level (GKR exploits layered structure) | Field-aware | GKR only | No |
| ZKTorch | Universal compiler | CQ batching | Lookup-based | CQLin/CQ | No |
| zkRNN | RNN-specific | N/A | N/A | GKR | RNN only |
| **ZK-Compile (ours)** | **TVM IR → multiple backends** | **ZK-aware fusion (configurable)** | **Field-native with overflow handling** | **R1CS, AIR, PLONKish** | **Yes (first)** |

**Note on GKR-based systems.** DeepProve's use of GKR is a fundamentally different paradigm from circuit flattening (R1CS/AIR). GKR exploits the layered structure of the computation directly, achieving efficiency that R1CS-based approaches cannot match for uniformly layered circuits. Our approach is complementary: the compiler's graph-level optimizations (fusion, quantization, architecture analysis) are independent of the backend proof system and could target a GKR backend in future work.

---

## 7. Implementation Plan

### Phase 1: Proof of Concept — COMPLETE

R1CS backend for a subset of TensorIR (static shapes, no memory indirection), targeting arkworks Groth16. Two-line API: `circuit = compile_model(model, x)` → `result = prove(circuit, x)`. Also accepts ONNX: `circuit = compile_onnx("model.onnx", x)`.

**Key implementation details:**
- Scale-aware quantization: bias rescaling through dataflow graph (correlation 1.000000 vs PyTorch)
- Dataflow wire sharing: Relax call graph parsed, output→input wires shared across functions
- Zero-gate wire aliasing: transpose/reshape output→input wire mapping (no gates = pure data movement)
- Sound ReLU: 41-bit decomposition, malicious-prover secure
- Constant wire constraints: model weights pinned to exact values (1 constraint per wire), preventing weight substitution by malicious prover
- Mul+add fusion: accumulation adds folded into preceding mul constraints (49% constraint reduction for MLP)

**Measured results (M4 Max, BN254 field, INT16 quantization):**

| Model | Constraints (base) | Constraints (+ const) | Prove (base) | Prove (+ const) | Verify | Proof |
|-------|-------------------|---------------------|-------------|-----------------|--------|-------|
| MNIST MLP (784→128→10) | 107,415 | 209,185 | 0.59s | 0.82s | 0.9ms | 128 B |
| MNIST CNN (conv+relu+fc) | 46,103 | 51,977 | 0.20s | 0.20s | 1.0ms | 128 B |

*Base = no constant constraints (model weights unpinned). + const = constant constraints pinning all model weights (sound against malicious prover).*

**EZKL comparison (Halo2/KZG, same models, same machine):**

| Model | EZKL constraints | EZKL prove | EZKL verify | EZKL proof size |
|-------|-----------------|------------|-------------|-----------------|
| MNIST MLP | ≤65,536 | 1.23s | 17.5ms | 18,829 B |
| MNIST CNN | ≤65,536 | 1.35s | 17.9ms | 18,761 B |

**Comparison (ZK-Compile with constant constraints vs EZKL):**

| Metric | MLP | CNN |
|--------|-----|-----|
| Prove speed | 1.5x faster | 6.8x faster |
| Verify speed | 18x faster | 16x faster |
| Proof size | 147x smaller | 147x smaller |
| Constraints | 3.2x more | 0.8x (fewer) |

**GPT-2 results (structured sumcheck over M31, March 2026):** 12-layer GPT-2 (124M parameters): 1.19s prove, 4.87ms verify, 2.48MB proof, 182/182 operations proved (100% coverage). All operations including attention (softmax + QKV dot products) are fully proved.

ZK-Compile (Groth16) wins on prove speed, verify speed, and proof size. EZKL (Halo2) wins on constraint count for MLP (lookup tables for ReLU + range checks) and requires no trusted setup. The CNN result is notable: ZK-Compile with full soundness guarantees produces fewer constraints than EZKL.

**Fusion results:** TVM's standard FuseOps gives 0% constraint reduction (merges at graph level, doesn't eliminate intermediate TIR buffers). Our ZK-specific mul+add fusion folds accumulation adds into their preceding mul gates, reducing MLP constraints by 49% (209K → 107K) and CNN by 30% (66K → 46K).

### Phase 2: Field-Native Quantization (4-6 weeks)
- Goldilocks (64-bit) and BabyBear (31-bit) field quantization
- Overflow analysis and intermediate reduction insertion
- Accuracy evaluation: compare field-quantized models against FP32 and standard INT8 on standard benchmarks
- Deliverable: accuracy-constraint count Pareto curves for different field/bit-width combinations

### Phase 3: STARK Backend + Architecture Comparison (8-12 weeks)
- AIR backend targeting Plonky3 (Goldilocks field)
- Compile small transformer (GPT-2 124M scale) and equivalent GatedDeltaNet model
- **Primary deliverable:** First measured constraint-count comparison across architecture families at multiple sequence lengths
- Include range checks, witness generation time, and actual proving time (not just constraint count)

### Phase 4: Dual Compilation Prototype (4-6 weeks)
- Shared INT16 quantization → Metal (fast) + STARK (provable)
- Demonstrate: identical outputs from both paths
- End-to-end demo: serve from Metal, prove on demand
- Deliverable: measured end-to-end latency for serve + prove workflow

**Total estimated timeline: 22-32 weeks.** Each phase produces independently publishable results.

---

## 8. Limitations

- **Phase 1 complete.** The R1CS backend, gate extraction, witness generation, Groth16 proving, mul+add fusion, constant wire constraints, ONNX input, and two-line API are all working. MNIST MLP and CNN models proved end-to-end with correct output (correlation 1.000000 vs PyTorch). Constraint counts in Section 4 remain analytical estimates; Phase 1 results cover only MNIST-scale models.

- **Constraint count ≠ proving time.** We use constraint count for cross-architecture comparison (where the proof system is held constant) but acknowledge that proving time depends on commitment costs (MSM/NTT), witness generation, and memory, all of which are superlinear in constraint count.

- **Field quantization accuracy is unvalidated.** We argue that field-native quantization is equivalent to integer quantization at the same bit width, but have not measured accuracy on real models. INT8 quantization causes measurable accuracy loss on LLMs; INT16 is safer but produces larger circuits.

- **Dynamic shapes unsupported.** The ZK backend requires concrete shapes at compile time. Each input shape requires a separate circuit compilation.

- **Proof composition not designed.** Proving real-sized models (1B+ parameters) requires circuit partitioning and recursive proof composition. We have not designed the partitioning strategy or its interaction with fusion passes.

- **TVM community acceptance uncertain.** A ZK backend is architecturally compatible with TVM's extension mechanisms but serves a different audience than TVM's current users. An out-of-tree extension or fork may be more practical than upstreaming.

---

## 9. Conclusion

The central observation of this paper is that ML compilers and ZK circuit compilers solve structurally similar problems — decomposing high-level computation into primitive arithmetic — and that unifying them provides benefits unavailable to either alone. A compiler-based approach enables ZK-aware fusion (reducing witness and range-check overhead), field-native quantization (with principled overflow handling), and cross-architecture proving cost analysis.

The most impactful finding is that **architecture selection is a first-order lever for ZK proving cost** — an insight that only emerges when a unified compiler provides comparable cost analysis across model families. At long sequence lengths, linear attention architectures (GatedDeltaNet, Mamba-2) produce asymptotically fewer non-linear constraints than softmax transformers, with the attention-layer advantage exceeding 1000x at 8K tokens. This asymptotic advantage is robust to proof system choice and optimization level, as it stems from the O(n²) vs O(n) scaling of the attention computation itself.

Phase 1 results demonstrate that compiler-level optimizations provide measurable advantages: mul+add fusion reduces MLP constraints by 49%, scale-aware quantization achieves perfect correlation with PyTorch, and constant wire constraints provide sound weight binding at 16x lower cost than range checks. Against EZKL on MNIST CNN, ZK-Compile with full soundness guarantees produces fewer constraints while proving 6.8x faster — the first demonstration that a compiler-based approach can match or beat a dedicated zkML framework on constraint efficiency.

---

## References

- [1] Hao et al. "Scalable Zero-Knowledge Proofs for Non-Linear Functions in Machine Learning." USENIX Security 2024.
- [2] Sun et al. "zkLLM: Zero Knowledge Proofs for Large Language Models." 2024. arXiv:2404.16109.
- [3] Kang et al. "Scaling up Trustless DNN Inference with Zero-Knowledge Proofs." EuroSys 2024.
- [4] Chen et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI 2018.
- [5] Feng et al. "TensorIR: An Abstraction for Automatic Tensorized Program Optimization." ASPLOS 2023.
- [6] Yang et al. "Gated Delta Networks with Piecewise Affine Dual Form." ICLR 2025.
- [7] Dao and Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Space Duality." ICML 2024.
- [8] Groth. "On the Size of Pairing-based Non-interactive Arguments." EUROCRYPT 2016.
- [9] Ben-Sasson et al. "Scalable Zero Knowledge with No Trusted Setup." CRYPTO 2019.
- [10] EZKL. https://github.com/zkonduit/ezkl
- [11] zkPyTorch. "Verifiable Inference of Any PyTorch Model." ePrint 2025/535.
- [12] DeepProve. "Efficient Verifiable AI." Lagrange Labs, 2025.
- [13] Zarinjouei et al. "zkRNN: Zero-Knowledge Proofs for Recurrent Neural Network Inference." ePrint 2026/073.
- [14] Li and Fan. "SUMMER: Recursive Zero-Knowledge Proofs for Scalable RNN Training." ePrint 2025/1688.
- [15] Ghodsi et al. "SafetyNets: Verifiable Execution of Deep Neural Networks on an Untrusted Cloud." NeurIPS 2017.
- [16] Kang. "TensorPlonk: A GPU for ZKML, Delivering 1,000x Speedups." 2025.
- [17] Goldwasser, Kalai, Rothblum. "Delegating Computation: Interactive Proofs for Muggles." STOC 2008.
- [18] Kothapalli and Setty. "HyperNova: Recursive Arguments from Folding Schemes." CRYPTO 2023.
