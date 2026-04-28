# tests/reference/ — Phase 11 numpy reference SPEC

This file is the **API contract** the numpy reference (P11-1) must
implement and that the Rust property harness (P11-2), MLE-identity
tests (P11-3), and quantization-invariant tests (P11-5) must call.
Lock the API here once; do not edit while parallel agents are
implementing it.

## Why a SPEC

The Phase 11 plan parallelizes 3-4 agents. They cannot agree on
function signatures by reading each other's code mid-implementation.
This file is the single source of truth so:
  - Agent A (P11-1) implements TO this spec.
  - Agents D/E (P11-2/P11-5) call AGAINST this spec.
  - Agent F (P11-6) wires CI assuming this spec.

If the spec proves wrong during implementation, raise it as a comment
in the relevant agent's PR/branch — don't unilaterally diverge.

## File layout

```
tests/reference/
├── SPEC.md                  ← this file
├── reference.py             ← P11-1, the numpy implementations
├── m31.py                   ← M31 field arithmetic helpers (mod p, signed canonicalization)
├── mle.py                   ← multilinear extension primitives (mle_evaluate, eq_evals)
├── attention.py             ← attention sub-references (full attn, GDN identity, GQA replicate)
└── transformer.py           ← layer-level reference (RMSNorm, residual, full Qwen layer forward)
```

Python ≥ 3.9; numpy only. **No torch.** Keep dependencies minimal so the
reference is auditable in isolation.

## M31 field

```python
M31 = (1 << 31) - 1  # = 2_147_483_647

def to_signed(x: np.ndarray) -> np.ndarray:
    """Canonical signed view of M31 elements: values > p/2 → negative."""

def from_signed(x: np.ndarray) -> np.ndarray:
    """Inverse of to_signed: x mod p with the canonical positive rep."""
```

## Multilinear extension primitives

`reference/mle.py`:

```python
def mle_evaluate(evals: np.ndarray, point: np.ndarray) -> int:
    """Multilinear extension of `evals` (length 2^k) at `point` (length k).

    Folding convention: MSB-first. point[0] folds the highest-order bit
    of the index. This MUST match the Rust prover's `field/m31_ops::mle_evaluate`
    semantics exactly. The GQA bug from P10-3 was a slicing error
    against this exact convention.

    Returns int in [0, M31).
    """

def eq_evals(point: np.ndarray) -> np.ndarray:
    """For point of length k, return [eq(point, x) for x in {0,1}^k] of length 2^k.

    Used by both the prover and the reference; the reference must match
    Rust `field/m31_ops::eq_evals` value-for-value.
    """
```

## Reference forwards (per-op)

All return values are M31 field elements (positive ints in [0, M31)) unless
explicitly noted as floats.

### `reference.matmul`

```python
def matmul(W: np.ndarray, x: np.ndarray, m: int, n: int,
           bias: Optional[np.ndarray] = None) -> np.ndarray:
    """y = W @ x + bias (mod M31). W is shape (m, n) row-major flat-encoded;
    x is shape (n,); y is shape (m,). All inputs are M31 field elements."""
```

### `reference.rmsnorm`

```python
def rmsnorm_forward(x: np.ndarray, gamma: np.ndarray, d: int,
                     perturbation_delta: int = 0) -> tuple[np.ndarray, int]:
    """RMSNorm with QR-perturbation loop. Returns (output, applied_delta).

    Implements the prover's QR-search: increment x[0] by ±delta until
    d / sum(x²) is a quadratic residue in M31. The applied_delta value
    is what the prover transcript-binds (P10-S5).
    """
```

### `reference.attention`

```python
def attention_seq_len_1(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                         num_q_heads: int, num_kv_heads: int, d_head: int,
                         is_gqa_full_attn: bool) -> np.ndarray:
    """seq_len=1 attention output.

    GDN-style (q == kv): `attn_out = v.clone()` (identity).
    GQA full-attn (q != kv): `attn_out[h, d] = v[h / heads_per_group, d]`
                              (head replication).

    The MLE relation that the v↔attn_out P10-3 sub-proof relies on:
        MLE(attn_out, r_attn) = MLE(v, r_v)
    where for GDN r_v = r_attn, and for GQA r_v is the
    [group_prefix(log_kv) || d_head_suffix(log_d)] slice (NOT the
    contiguous prefix — that was the GQA bug, fixed in commit 86f3ab6).

    Used by P11-3 to assert the relation directly from random inputs.
    """

def attention_full(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    num_heads: int, seq_len: int, d_head: int,
                    exp_scale: int) -> np.ndarray:
    """seq_len ≥ 2 row-attention. Reference for proptest harness."""
```

### `reference.lookup`

```python
def silu_table(scale: int) -> np.ndarray:
    """Returns 65536-row (input, output) pairs as int32 array of shape
    (65536, 2). Must match Rust `proving::lookup::build_silu_table` byte-
    for-byte after canonicalization to M31."""

def sigmoid_table(scale: int) -> np.ndarray: ...
def exp_table(scale: int) -> np.ndarray: ...
```

### `reference.transformer`

```python
def qwen_layer_forward(x, weights: dict, config: dict) -> dict:
    """Full Qwen layer forward. Returns dict with all intermediates that
    the prover trace exposes (norm1_out, q, k, v, attn_out, gated_attn,
    o_proj_out, h, norm2_out, gate_out, up_out, swiglu_out, down_out,
    output).

    `weights` keys: norm1_gamma, w_q, w_k, w_v, w_o, w_g_proj,
                    norm2_gamma, w_gate, w_up, w_down. All M31 arrays.
    `config` keys: d_model, d_ff, num_q_heads, num_kv_heads, d_head,
                   v_num_heads, v_d_head, silu_scale, sigmoid_scale.

    Reference for the proptest harness's end-to-end test:
        1. Sample random config + weights.
        2. Run `qwen_layer_forward` (reference).
        3. Run `prove_qwen_layer_with_trace` (Rust prover).
        4. Assert canonical_trace.output ≡ rust_trace.output (mod M31).
        5. Assert proof verifies against Rust verifier.
    """
```

## Quantization tolerance

Most ops are exact in M31 (no quantization at the field level). Three
ops have explicit tolerance:

- **RMSNorm**: QR-perturbation may flip `x[0]` by ±delta; the reference
  exposes the applied delta, prover and reference must agree on it.
- **LogUp lookups (SiLU/sigmoid/GELU)**: floating-point tables
  quantized to int16. Allowed tolerance: ±1 LSB at the int16 level for
  any single lookup.
- **Attention**: softmax goes through exp lookup → sum → divide. End-
  to-end tolerance: ‖rust − reference‖_∞ ≤ 4 LSBs at the int16 level.

For end-to-end proptest assertions, use **exact** equality at the M31
level for everything except the three ops above.

## Random seed convention

All proptest entry points take `seed: u64`. Convert to numpy:
```python
rng = np.random.default_rng(seed)
```
For Rust:
```rust
let mut rng = StdRng::seed_from_u64(seed);
```
Same seed → same weights / inputs / model config across both languages.

## Config branch coverage

Property tests MUST exercise all three branches:
1. **GDN-style** (`num_q_heads == num_kv_heads`).
2. **GQA full-attn** (`num_q_heads != num_kv_heads`, `num_q_heads % num_kv_heads == 0`).
3. **Asymmetric V** (`v_num_heads != num_kv_heads` or `v_d_head != d_head`).

Per-branch coverage is enforced by the proptest config; if a branch is
under-sampled, the harness fails with a "branch coverage" warning.

## Don't break

The reference and prover/verifier MUST agree on:
1. **MLE folding direction** (MSB-first per Rust `mle_evaluate`).
2. **Tensor flattening order** (row-major: `W[i, j]` at flat index `i*n + j`).
3. **Head-replication semantics for GQA** (`attn_out[h, d] = v[h / heads_per_group, d]`,
   where `heads_per_group = num_q_heads / num_kv_heads`).
4. **Quantization rounding** (round-half-to-even for int16, matching numpy default).
5. **Field canonicalization** (positive representation `[0, M31)`).
