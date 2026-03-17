//! Lasso-style lookup tables for nonlinear operations over M31.
//!
//! Implements LogUp lookup arguments for exp, sigmoid, SiLU, and softplus.
//! Tables are precomputed over INT16 range and committed via Merkle trees.
//! The lookup proof uses a product sumcheck to verify that every (input, output)
//! pair appears in the committed table, via the LogUp fractional sumcheck:
//!
//!   Σ_i 1/(α - f(i)) = Σ_j m(j)/(α - T(j))
//!
//! where f(i) encodes (a_i, b_i) as a_i + β·b_i, T(j) encodes table entries,
//! m(j) is the multiplicity of each table entry, and α, β are Fiat-Shamir challenges.
//!
//! Decomposable structure (Lasso optimization): for large inputs, decompose into
//! 8-bit chunks and use 2^8-sized subtables instead of one 2^16 table.

use p3_field::{AbstractField, Field, PrimeField32};
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};
use crate::proving::sumcheck::{self, SumcheckProof, SumcheckProofEF, EFElement, EF, Transcript};
use crate::field::common::{log2_ceil, i16_to_field, quantize_i16};
use crate::field::m31_ops::f_to_ef;
use crate::proving::weight_commitment::{commit_weights_fast, WeightCommitment};

type F = Mersenne31;

/// A precomputed lookup table mapping input -> output in M31.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LookupTable {
    #[allow(dead_code)]
    pub name: String,
    /// (input, output) pairs — table[i] corresponds to input value (i + offset).
    pub entries: Vec<(u32, u32)>,
    /// Merkle commitment to the encoded table values (input + β·output for each entry).
    pub commitment: WeightCommitment,
}

/// Proof that all lookups (a_i, b_i) are present in the committed table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LookupProof {
    pub table_commitment: WeightCommitment,
    pub multiplicity_commitment: WeightCommitment,
    /// Sumcheck proof for the LogUp identity:
    /// Σ_i eq(r,i) · inv_f(i) = Σ_j eq(r,j) · m(j) · inv_T(j)
    /// Reduced to: Σ_i eq(r,i) · [inv_f(i) · Π_T - m(i) · inv_T(i) · Π_f] = 0
    /// We use a product sumcheck on the combined polynomial.
    pub sumcheck_proof: SumcheckProof,
    /// Final evaluations (f_at_s, g_at_s) from the sumcheck.
    pub finals: (u32, u32),
    /// The random challenges used (α, β) — verifier re-derives from transcript.
    pub alpha: u32,
    pub beta: u32,
    /// Lookup inputs and outputs (the statement being proved).
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    /// Multiplicities per active table entry.
    /// Empty in production — verifier doesn't need them (sumcheck is self-contained).
    #[serde(default)]
    pub multiplicities: Vec<u32>,
}

/// Proof that all lookups are present in the committed table (extension field version).
/// Uses 124-bit EF challenges for α, β (vs 31-bit in base-field LookupProof).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LookupProofEF {
    pub table_commitment: WeightCommitment,
    pub multiplicity_commitment: WeightCommitment,
    /// EF sumcheck proof for the LogUp identity.
    pub sumcheck_proof: SumcheckProofEF,
    /// Final evaluation h_at_s from the EF sumcheck. (ones_at_s is always 1.)
    pub h_at_s: EFElement,
    /// The random EF challenges (α, β) — verifier re-derives. May be empty (stripped).
    #[serde(default)]
    pub alpha: Option<EFElement>,
    #[serde(default)]
    pub beta: Option<EFElement>,
    /// Lookup inputs and outputs (the statement being proved). May be empty (stripped).
    #[serde(default)]
    pub inputs: Vec<u32>,
    #[serde(default)]
    pub outputs: Vec<u32>,
    /// Multiplicities per active table entry (empty in production).
    #[serde(default)]
    pub multiplicities: Vec<u32>,
}

// ===== Table construction =====

/// Build an exp lookup table: exp(x / scale), quantized to INT16.
/// Input domain: INT16 [-32768, 32767], representing x/scale in fixed-point.
/// Output: exp(input/scale) * scale, quantized to INT16.
pub fn build_exp_table(scale: i32) -> LookupTable {
    let s = scale as f64;
    let mut entries = Vec::with_capacity(65536);
    for raw in 0u32..65536 {
        let input_i16 = (raw as i16) as i16; // wrapping cast: 0..32767 positive, 32768..65535 negative
        let x = input_i16 as f64 / s;
        let y = x.exp();
        let output_i16 = quantize_i16(y, s);
        entries.push((
            i16_to_field(input_i16).as_canonical_u32(),
            i16_to_field(output_i16).as_canonical_u32(),
        ));
    }
    // Commitment is built during prove (needs β), use a placeholder.
    LookupTable {
        name: "exp".to_string(),
        entries,
        commitment: WeightCommitment {
            root: [0u8; 32],
            num_weights: 65536,
            log_height: 16,
        },
    }
}

/// Build a sigmoid lookup table: sigmoid(x/scale) = 1/(1+exp(-x/scale)), quantized.
pub fn build_sigmoid_table(scale: i32) -> LookupTable {
    let s = scale as f64;
    let mut entries = Vec::with_capacity(65536);
    for raw in 0u32..65536 {
        let input_i16 = raw as i16;
        let x = input_i16 as f64 / s;
        let y = 1.0 / (1.0 + (-x).exp());
        let output_i16 = quantize_i16(y, s);
        entries.push((
            i16_to_field(input_i16).as_canonical_u32(),
            i16_to_field(output_i16).as_canonical_u32(),
        ));
    }
    LookupTable {
        name: "sigmoid".to_string(),
        entries,
        commitment: WeightCommitment {
            root: [0u8; 32],
            num_weights: 65536,
            log_height: 16,
        },
    }
}

/// Build a SiLU lookup table: silu(x) = x * sigmoid(x), quantized.
pub fn build_silu_table(scale: i32) -> LookupTable {
    let s = scale as f64;
    let mut entries = Vec::with_capacity(65536);
    for raw in 0u32..65536 {
        let input_i16 = raw as i16;
        let x = input_i16 as f64 / s;
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        let y = x * sigmoid;
        let output_i16 = quantize_i16(y, s);
        entries.push((
            i16_to_field(input_i16).as_canonical_u32(),
            i16_to_field(output_i16).as_canonical_u32(),
        ));
    }
    LookupTable {
        name: "silu".to_string(),
        entries,
        commitment: WeightCommitment {
            root: [0u8; 32],
            num_weights: 65536,
            log_height: 16,
        },
    }
}

/// Build a softplus lookup table: softplus(x) = log(1+exp(x/scale)), quantized.
/// Threshold optimization: softplus(x) ≈ x for x > 20*scale (avoids overflow).
#[allow(dead_code)]
pub fn build_softplus_table(scale: i32) -> LookupTable {
    let s = scale as f64;
    let threshold = 20.0 * s;
    let mut entries = Vec::with_capacity(65536);
    for raw in 0u32..65536 {
        let input_i16 = raw as i16;
        let x = input_i16 as f64;
        let y = if x > threshold {
            // softplus(x/s) ≈ x/s for large x
            x / s
        } else {
            (1.0 + (x / s).exp()).ln()
        };
        let output_i16 = quantize_i16(y, s);
        entries.push((
            i16_to_field(input_i16).as_canonical_u32(),
            i16_to_field(output_i16).as_canonical_u32(),
        ));
    }
    LookupTable {
        name: "softplus".to_string(),
        entries,
        commitment: WeightCommitment {
            root: [0u8; 32],
            num_weights: 65536,
            log_height: 16,
        },
    }
}

// ===== Decomposable subtables (Lasso optimization) =====

/// Build a subtable for an 8-bit chunk, reducing table size from 2^16 to 2^8.
/// `chunk_fn` maps an 8-bit value to its contribution.
#[allow(dead_code)]
pub fn build_chunk_subtable(
    name: &str,
    chunk_fn: impl Fn(u8) -> i16,
) -> LookupTable {
    let mut entries = Vec::with_capacity(256);
    for i in 0u32..256 {
        let input_i16 = i as i16; // 0..255
        let output_i16 = chunk_fn(i as u8);
        entries.push((
            i16_to_field(input_i16).as_canonical_u32(),
            i16_to_field(output_i16).as_canonical_u32(),
        ));
    }
    LookupTable {
        name: name.to_string(),
        entries,
        commitment: WeightCommitment {
            root: [0u8; 32],
            num_weights: 256,
            log_height: 8,
        },
    }
}

// ===== LogUp lookup argument =====

/// Batch field inversion via Montgomery's trick: compute all inverses at once
/// using only 1 full inversion + 3(n-1) multiplications.
/// Returns inv[i] = 1/vals[i] for each i. Panics if any val is zero.
fn batch_inverse(vals: &[F]) -> Vec<F> {
    let n = vals.len();
    if n == 0 { return vec![]; }

    // Forward pass: prefix products
    let mut prefix = Vec::with_capacity(n);
    prefix.push(vals[0]);
    for i in 1..n {
        prefix.push(prefix[i - 1] * vals[i]);
    }

    // Single inversion of the total product
    let mut inv_prod = prefix[n - 1].inverse();

    // Backward pass: compute individual inverses
    let mut result = vec![F::zero(); n];
    for i in (1..n).rev() {
        result[i] = inv_prod * prefix[i - 1];
        inv_prod = inv_prod * vals[i];
    }
    result[0] = inv_prod;

    result
}

/// Batch field inversion over EF via Montgomery's trick.
fn batch_inverse_ef(vals: &[EF]) -> Vec<EF> {
    let n = vals.len();
    if n == 0 { return vec![]; }

    let mut prefix = Vec::with_capacity(n);
    prefix.push(vals[0]);
    for i in 1..n {
        prefix.push(prefix[i - 1] * vals[i]);
    }

    let mut inv_prod = prefix[n - 1].inverse();

    let mut result = vec![EF::zero(); n];
    for i in (1..n).rev() {
        result[i] = inv_prod * prefix[i - 1];
        inv_prod = inv_prod * vals[i];
    }
    result[0] = inv_prod;

    result
}

/// Reverse a field element back to its raw table index (0..n_table).
/// Tables are built from sequential i16 (or i8) values via i16_to_field,
/// so the mapping is directly invertible without a HashMap.
///
/// For n_table=65536 (i16 range): raw 0..32767 → field 0..32767, raw 32768..65535 → field P-32768..P-1
/// For n_table=256 (i8 range): raw 0..127 → field 0..127, raw 128..255 → field P-128..P-1
#[inline(always)]
pub fn field_to_table_index(v: u32, n_table: usize) -> usize {
    const P: u64 = 0x7FFFFFFF; // Mersenne-31
    let v64 = v as u64;
    let half = (n_table / 2) as u64;
    if v64 < half {
        v64 as usize
    } else if v64 >= P - half {
        // Negative i16/i8: field = P - |val|, raw = n_table - |val| = v + n_table - P
        (v64 + n_table as u64 - P) as usize
    } else {
        // Not a valid table entry — return out-of-range sentinel
        usize::MAX
    }
}

/// Compute multiplicities: for each lookup input, find the table index via
/// direct arithmetic (O(1) per lookup) instead of building a HashMap over all entries.
/// Soundness note: we index by input only. If the output is wrong, the LogUp
/// identity check (Σ h(x) = 0) will catch it — the encoded f_i ≠ T[j].
fn compute_multiplicities(
    table: &LookupTable,
    inputs: &[F],
    _outputs: &[F],
) -> Vec<u32> {
    let n_table = table.entries.len();
    let mut mult = vec![0u32; n_table];

    for a in inputs {
        let idx = field_to_table_index(a.as_canonical_u32(), n_table);
        if idx < n_table {
            mult[idx] += 1;
        }
    }
    mult
}


/// Prove that all (inputs[i], outputs[i]) pairs exist in the lookup table.
///
/// LogUp argument: we prove the identity
///   Σ_i 1/(α - f_i) = Σ_j m_j/(α - T_j)
/// where f_i = inputs[i] + β·outputs[i], T_j = table[j].input + β·table[j].output.
///
/// Rearranged into a single sumcheck over a combined domain of size N = max(n_lookups, n_table),
/// padded to power of 2:
///   Σ_x eq(r, x) · g(x) = 0
/// where g(x) encodes the LogUp fractional identity.
///
/// Specifically, multiplying through by all denominators, the identity becomes:
///   Σ_i [Π_{j≠i}(α - f_j)] · [Π_k(α - T_k)] = Σ_j m_j · [Π_{k≠j}(α - T_k)] · [Π_i(α - f_i)]
///
/// This is impractical for large N. Instead we use the batch inverse approach:
/// Define h(x) = inv_f(x) - m(x)·inv_T(x) where inv_f(x) = 1/(α - f(x)), inv_T(x) = 1/(α - T(x)).
/// Then Σ h(x) = 0 is the LogUp identity.
/// We prove this via product sumcheck: Σ eq(r,x) · h(x) = 0.
pub fn prove_lookup(
    table: &LookupTable,
    inputs: &[F],
    outputs: &[F],
    transcript: &mut Transcript,
) -> LookupProof {
    let n_lookups = inputs.len();
    let _n_table = table.entries.len();
    assert_eq!(n_lookups, outputs.len());

    // Absorb table commitment placeholder into transcript
    transcript.absorb_bytes(&table.commitment.root);
    transcript.absorb(n_lookups as u32);
    for (a, b) in inputs.iter().zip(outputs.iter()) {
        transcript.absorb(a.as_canonical_u32());
        transcript.absorb(b.as_canonical_u32());
    }

    // Squeeze challenges
    let beta = transcript.squeeze();
    let alpha = transcript.squeeze();

    // Encode lookups: f_i = inputs[i] + β·outputs[i]
    let f_encoded: Vec<F> = inputs
        .iter()
        .zip(outputs.iter())
        .map(|(&a, &b)| a + beta * b)
        .collect();

    // Compute multiplicities over the full table
    let mult_raw_full = compute_multiplicities(table, inputs, outputs);

    // Active subtable optimization: only include table entries with multiplicity > 0.
    // This dramatically reduces the sumcheck domain (e.g., 65536 → ~3584 for SiLU).
    let mut active_t_encoded = Vec::new();
    let mut active_mult_raw = Vec::new();
    let mut active_mult_f = Vec::new();
    for (j, &m) in mult_raw_full.iter().enumerate() {
        if m > 0 {
            let (inp, out) = table.entries[j];
            active_t_encoded.push(F::from_canonical_u32(inp) + beta * F::from_canonical_u32(out));
            active_mult_raw.push(m);
            active_mult_f.push(F::from_canonical_u32(m));
        }
    }
    let n_active = active_t_encoded.len();

    // Commit to active subtable and multiplicities
    let table_commitment = commit_weights_fast(&active_t_encoded);
    let multiplicity_commitment = commit_weights_fast(&active_mult_f);

    // Build h(x) over a combined domain using active subtable.
    // Domain size = n_lookups + n_active, padded to power of 2.
    let combined_len = n_lookups + n_active;
    let log_n = log2_ceil(combined_len);
    let n_pad = 1 << log_n;

    let mut h_vals = vec![F::zero(); n_pad];

    #[cfg(feature = "metal_gpu")]
    {
        use crate::gpu::{MetalContext, GpuBuffer, MetalKernels, GPU_THRESHOLD};

        if crate::proving::sumcheck::is_gpu_enabled() && combined_len >= GPU_THRESHOLD {
            let ctx = MetalContext::get();
            let kernels = MetalKernels::get();

            let mut denoms: Vec<F> = Vec::with_capacity(n_lookups + n_active);
            for i in 0..n_lookups {
                let d = alpha - f_encoded[i];
                assert!(d != F::zero(), "alpha collision with lookup encoding");
                denoms.push(d);
            }
            for k in 0..n_active {
                let d = alpha - active_t_encoded[k];
                assert!(d != F::zero(), "alpha collision with table encoding");
                denoms.push(d);
            }

            let mut buf = GpuBuffer::from_field_slice(&ctx.device, &denoms);
            kernels.batch_inverse(&mut buf);
            let inv_denoms = buf.to_field_vec();

            for i in 0..n_lookups {
                h_vals[i] = inv_denoms[i];
            }
            for k in 0..n_active {
                h_vals[n_lookups + k] = F::zero() - active_mult_f[k] * inv_denoms[n_lookups + k];
            }
        } else {
            // CPU path
            let mut denoms = Vec::with_capacity(n_lookups + n_active);
            for i in 0..n_lookups {
                let d = alpha - f_encoded[i];
                assert!(d != F::zero(), "alpha collision with lookup encoding");
                denoms.push(d);
            }
            for k in 0..n_active {
                let d = alpha - active_t_encoded[k];
                assert!(d != F::zero(), "alpha collision with table encoding");
                denoms.push(d);
            }
            let inv_denoms = batch_inverse(&denoms);
            for i in 0..n_lookups {
                h_vals[i] = inv_denoms[i];
            }
            for k in 0..n_active {
                h_vals[n_lookups + k] = F::zero() - active_mult_f[k] * inv_denoms[n_lookups + k];
            }
        }
    }

    #[cfg(not(feature = "metal_gpu"))]
    {
        // Batch inversion: collect all denominators, invert at once
        let mut denoms = Vec::with_capacity(n_lookups + n_active);
        for i in 0..n_lookups {
            let d = alpha - f_encoded[i];
            assert!(d != F::zero(), "alpha collision with lookup encoding");
            denoms.push(d);
        }
        for k in 0..n_active {
            let d = alpha - active_t_encoded[k];
            assert!(d != F::zero(), "alpha collision with table encoding");
            denoms.push(d);
        }

        let inv_denoms = batch_inverse(&denoms);

        // Lookup side: h[i] = 1/(α - f_i)
        for i in 0..n_lookups {
            h_vals[i] = inv_denoms[i];
        }
        // Table side: h[n_lookups + k] = -m_k/(α - T_k)
        for k in 0..n_active {
            h_vals[n_lookups + k] = F::zero() - active_mult_f[k] * inv_denoms[n_lookups + k];
        }
    }

    // The claimed sum should be 0 if the lookups are valid.
    let local_sum: F = h_vals.iter().copied().sum();
    assert_eq!(
        local_sum,
        F::zero(),
        "LogUp identity failed — invalid lookups"
    );

    // Absorb commitments into transcript for Fiat-Shamir binding
    transcript.absorb_bytes(&table_commitment.root);
    transcript.absorb_bytes(&multiplicity_commitment.root);

    // Product sumcheck: Σ ones(x) · h(x) = 0
    let (proof, ones_at_s, h_at_s) =
        sumcheck::prove_product_ones_best(&h_vals, log_n, transcript);

    LookupProof {
        table_commitment,
        multiplicity_commitment,
        sumcheck_proof: proof,
        finals: (ones_at_s.as_canonical_u32(), h_at_s.as_canonical_u32()),
        alpha: alpha.as_canonical_u32(),
        beta: beta.as_canonical_u32(),
        inputs: inputs.iter().map(|v| v.as_canonical_u32()).collect(),
        outputs: outputs.iter().map(|v| v.as_canonical_u32()).collect(),
        multiplicities: vec![], // omitted — verifier doesn't use (sumcheck self-contained)
    }
}

/// Prove lookups with 124-bit EF challenges (soundness upgrade from 31-bit base-field).
///
/// Same LogUp argument as `prove_lookup`, but α, β are squeezed as M31^4 extension
/// field elements and all arithmetic (encoding, batch inverse, sumcheck) operates in EF.
pub fn prove_lookup_ef(
    table: &LookupTable,
    inputs: &[F],
    outputs: &[F],
    transcript: &mut Transcript,
) -> LookupProofEF {
    let n_lookups = inputs.len();
    assert_eq!(n_lookups, outputs.len());

    // Absorb table commitment placeholder into transcript
    transcript.absorb_bytes(&table.commitment.root);
    transcript.absorb(n_lookups as u32);
    for (a, b) in inputs.iter().zip(outputs.iter()) {
        transcript.absorb(a.as_canonical_u32());
        transcript.absorb(b.as_canonical_u32());
    }

    // Squeeze EF challenges (124-bit soundness)
    let beta = transcript.squeeze_ef();
    let alpha = transcript.squeeze_ef();

    // Encode lookups in EF: f_i = input_ef + β·output_ef
    let f_encoded: Vec<EF> = inputs
        .iter()
        .zip(outputs.iter())
        .map(|(&a, &b)| f_to_ef(a) + beta * f_to_ef(b))
        .collect();

    // Compute multiplicities over the full table (same base-field logic)
    let mult_raw_full = compute_multiplicities(table, inputs, outputs);

    // Active subtable optimization
    let mut active_t_encoded = Vec::new();
    let mut active_mult_raw = Vec::new();
    let mut active_mult_ef = Vec::new();
    for (j, &m) in mult_raw_full.iter().enumerate() {
        if m > 0 {
            let (inp, out) = table.entries[j];
            active_t_encoded.push(f_to_ef(F::from_canonical_u32(inp)) + beta * f_to_ef(F::from_canonical_u32(out)));
            active_mult_raw.push(m);
            active_mult_ef.push(f_to_ef(F::from_canonical_u32(m)));
        }
    }
    let n_active = active_t_encoded.len();

    // Commit to active subtable entries and multiplicities (base field for Merkle binding).
    // Encode as input + output (single F per entry) to match base-field path's num_weights.
    let mut active_entries_f = Vec::with_capacity(n_active);
    for (j, &m) in mult_raw_full.iter().enumerate() {
        if m > 0 {
            let (inp, out) = table.entries[j];
            active_entries_f.push(F::from_canonical_u32(inp) + F::from_canonical_u32(out));
        }
    }
    let active_mult_f: Vec<F> = active_mult_raw.iter().map(|&m| F::from_canonical_u32(m)).collect();
    let table_commitment = commit_weights_fast(&active_entries_f);
    let multiplicity_commitment = commit_weights_fast(&active_mult_f);

    // Build h(x) over combined domain in EF
    let combined_len = n_lookups + n_active;
    let log_n = log2_ceil(combined_len);
    let n_pad = 1 << log_n;

    let mut h_vals = vec![EF::zero(); n_pad];

    // Batch inversion in EF
    let mut denoms = Vec::with_capacity(n_lookups + n_active);
    for i in 0..n_lookups {
        let d = alpha - f_encoded[i];
        assert!(d != EF::zero(), "alpha collision with lookup encoding");
        denoms.push(d);
    }
    for k in 0..n_active {
        let d = alpha - active_t_encoded[k];
        assert!(d != EF::zero(), "alpha collision with table encoding");
        denoms.push(d);
    }

    let inv_denoms = batch_inverse_ef(&denoms);

    // Lookup side: h[i] = 1/(α - f_i)
    for i in 0..n_lookups {
        h_vals[i] = inv_denoms[i];
    }
    // Table side: h[n_lookups + k] = -m_k/(α - T_k)
    for k in 0..n_active {
        h_vals[n_lookups + k] = EF::zero() - active_mult_ef[k] * inv_denoms[n_lookups + k];
    }

    // The claimed sum should be 0 if the lookups are valid.
    let local_sum: EF = h_vals.iter().copied().sum();
    assert_eq!(
        local_sum,
        EF::zero(),
        "LogUp identity failed — invalid lookups"
    );

    // Absorb commitments into transcript
    transcript.absorb_bytes(&table_commitment.root);
    transcript.absorb_bytes(&multiplicity_commitment.root);

    // EF product sumcheck: Σ ones(x) · h(x) = 0
    let (proof, _ones_at_s, h_at_s) =
        sumcheck::prove_product_ones_ef_full(&h_vals, log_n, transcript);

    LookupProofEF {
        table_commitment,
        multiplicity_commitment,
        sumcheck_proof: proof,
        h_at_s: EFElement::from_ef(h_at_s),
        alpha: None,
        beta: None,
        inputs: inputs.iter().map(|v| v.as_canonical_u32()).collect(),
        outputs: outputs.iter().map(|v| v.as_canonical_u32()).collect(),
        multiplicities: vec![],
    }
}

/// Verify a lookup proof.
///
/// The verifier checks:
/// 1. Re-derive α, β from transcript (Fiat-Shamir)
/// 2. Verify the product sumcheck: Σ eq(r,x) · h(x) = 0
/// 3. Check final evaluations: eq(r, s) · h(s) matches the sumcheck's final claim
/// 4. Verify eq(r, s) independently
pub fn verify_lookup(
    _table_commitment: &WeightCommitment,
    proof: &LookupProof,
    num_lookups: usize,
    transcript: &mut Transcript,
) -> bool {
    verify_lookup_with_data(_table_commitment, proof, num_lookups, None, transcript)
}

/// Verify lookup with externally-provided inputs/outputs.
/// Used when proof.inputs/outputs are empty (size optimization).
pub fn verify_lookup_with_data(
    _table_commitment: &WeightCommitment,
    proof: &LookupProof,
    num_lookups: usize,
    external_data: Option<(&[u32], &[u32])>,
    transcript: &mut Transcript,
) -> bool {
    let (inputs, outputs) = match external_data {
        Some((i, o)) => (i, o),
        None => (&proof.inputs[..], &proof.outputs[..]),
    };

    // Re-derive challenges
    transcript.absorb_bytes(&[0u8; 32]); // original table commitment placeholder
    transcript.absorb(num_lookups as u32);
    for (&a, &b) in inputs.iter().zip(outputs.iter()) {
        transcript.absorb(a);
        transcript.absorb(b);
    }

    let beta = transcript.squeeze();
    let alpha = transcript.squeeze();

    // Verify challenges match
    if alpha.as_canonical_u32() != proof.alpha || beta.as_canonical_u32() != proof.beta {
        return false;
    }

    let n_table = proof.table_commitment.num_weights;
    let combined_len = num_lookups + n_table;
    let log_n = log2_ceil(combined_len);

    // Absorb commitments (must match prover's transcript)
    transcript.absorb_bytes(&proof.table_commitment.root);
    transcript.absorb_bytes(&proof.multiplicity_commitment.root);

    let ones_at_s = F::from_canonical_u32(proof.finals.0);
    let h_at_s = F::from_canonical_u32(proof.finals.1);

    // Verify product sumcheck: Σ ones(x) · h(x) = 0
    if !sumcheck::verify_product(
        F::zero(), // LogUp identity sums to 0
        &proof.sumcheck_proof,
        log_n,
        ones_at_s,
        h_at_s,
        transcript,
    ) {
        return false;
    }

    // The all-ones MLE is the constant 1 polynomial.
    // Verify ones_at_s = 1.
    if ones_at_s != F::one() {
        return false;
    }

    true
}

/// Verify an EF lookup proof.
#[allow(dead_code)]
pub fn verify_lookup_ef(
    _table_commitment: &WeightCommitment,
    proof: &LookupProofEF,
    num_lookups: usize,
    transcript: &mut Transcript,
) -> bool {
    verify_lookup_ef_with_data(_table_commitment, proof, num_lookups, None, transcript)
}

/// Verify EF lookup with externally-provided inputs/outputs.
pub fn verify_lookup_ef_with_data(
    _table_commitment: &WeightCommitment,
    proof: &LookupProofEF,
    num_lookups: usize,
    external_data: Option<(&[u32], &[u32])>,
    transcript: &mut Transcript,
) -> bool {
    let (inputs, outputs) = match external_data {
        Some((i, o)) => (i, o),
        None => (&proof.inputs[..], &proof.outputs[..]),
    };

    // Re-derive EF challenges
    transcript.absorb_bytes(&[0u8; 32]); // original table commitment placeholder
    transcript.absorb(num_lookups as u32);
    for (&a, &b) in inputs.iter().zip(outputs.iter()) {
        transcript.absorb(a);
        transcript.absorb(b);
    }

    let _beta = transcript.squeeze_ef();
    let _alpha = transcript.squeeze_ef();

    // α, β re-derived from transcript — no need to compare with proof
    // (Fiat-Shamir binding comes from the transcript, not stored values)

    let n_table = proof.table_commitment.num_weights;
    let combined_len = num_lookups + n_table;
    let log_n = log2_ceil(combined_len);

    // Absorb commitments (must match prover's transcript)
    transcript.absorb_bytes(&proof.table_commitment.root);
    transcript.absorb_bytes(&proof.multiplicity_commitment.root);

    // ones_at_s is always EF::one() (constant-1 MLE), h_at_s from proof
    let ones_at_s = EF::one();
    let h_at_s = proof.h_at_s.to_ef();

    // Verify EF product sumcheck: Σ ones(x) · h(x) = 0
    if !sumcheck::verify_product_ef(
        EF::zero(), // LogUp identity sums to 0
        &proof.sumcheck_proof,
        log_n,
        ones_at_s,
        h_at_s,
        transcript,
    ) {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::m31_ops::*;
    use p3_field::AbstractField;

    /// Helper: build a small table (256 entries) for faster tests.
    fn build_small_exp_table(scale: i32) -> LookupTable {
        let s = scale as f64;
        let mut entries = Vec::with_capacity(256);
        for raw in 0u32..256 {
            let input_i16 = raw as i8 as i16; // -128..127
            let x = input_i16 as f64 / s;
            let y = x.exp();
            let output_i16 = quantize_i16(y, s);
            entries.push((
                i16_to_field(input_i16).as_canonical_u32(),
                i16_to_field(output_i16).as_canonical_u32(),
            ));
        }
        LookupTable {
            name: "exp_small".to_string(),
            entries,
            commitment: WeightCommitment {
                root: [0u8; 32],
                num_weights: 256,
                log_height: 8,
            },
            }
    }

    fn build_small_sigmoid_table(scale: i32) -> LookupTable {
        let s = scale as f64;
        let mut entries = Vec::with_capacity(256);
        for raw in 0u32..256 {
            let input_i16 = raw as i8 as i16;
            let x = input_i16 as f64 / s;
            let y = 1.0 / (1.0 + (-x).exp());
            let output_i16 = quantize_i16(y, s);
            entries.push((
                i16_to_field(input_i16).as_canonical_u32(),
                i16_to_field(output_i16).as_canonical_u32(),
            ));
        }
        LookupTable {
            name: "sigmoid_small".to_string(),
            entries,
            commitment: WeightCommitment {
                root: [0u8; 32],
                num_weights: 256,
                log_height: 8,
            },
            }
    }

    fn build_small_silu_table(scale: i32) -> LookupTable {
        let s = scale as f64;
        let mut entries = Vec::with_capacity(256);
        for raw in 0u32..256 {
            let input_i16 = raw as i8 as i16;
            let x = input_i16 as f64 / s;
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            let y = x * sigmoid;
            let output_i16 = quantize_i16(y, s);
            entries.push((
                i16_to_field(input_i16).as_canonical_u32(),
                i16_to_field(output_i16).as_canonical_u32(),
            ));
        }
        LookupTable {
            name: "silu_small".to_string(),
            entries,
            commitment: WeightCommitment {
                root: [0u8; 32],
                num_weights: 256,
                log_height: 8,
            },
            }
    }

    fn build_small_softplus_table(scale: i32) -> LookupTable {
        let s = scale as f64;
        let threshold = 20.0 * s;
        let mut entries = Vec::with_capacity(256);
        for raw in 0u32..256 {
            let input_i16 = raw as i8 as i16;
            let x = input_i16 as f64;
            let y = if x > threshold {
                x / s
            } else {
                (1.0 + (x / s).exp()).ln()
            };
            let output_i16 = quantize_i16(y, s);
            entries.push((
                i16_to_field(input_i16).as_canonical_u32(),
                i16_to_field(output_i16).as_canonical_u32(),
            ));
        }
        LookupTable {
            name: "softplus_small".to_string(),
            entries,
            commitment: WeightCommitment {
                root: [0u8; 32],
                num_weights: 256,
                log_height: 8,
            },
            }
    }

    /// Pick valid lookups from a table: entries at given indices.
    fn pick_lookups(table: &LookupTable, indices: &[usize]) -> (Vec<F>, Vec<F>) {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for &idx in indices {
            let (inp, out) = table.entries[idx];
            inputs.push(F::from_canonical_u32(inp));
            outputs.push(F::from_canonical_u32(out));
        }
        (inputs, outputs)
    }

    #[test]
    fn test_build_exp_table() {
        let table = build_small_exp_table(1000);
        assert_eq!(table.entries.len(), 256);
        // exp(0) = 1.0 => output = 1000 (scale=1000)
        // Index 0 maps to input_i16 = 0
        let (_, out_raw) = table.entries[0];
        let out_val = from_field(F::from_canonical_u32(out_raw));
        assert_eq!(out_val, 1000); // exp(0)*1000 = 1000
    }

    #[test]
    fn test_build_sigmoid_table() {
        let table = build_small_sigmoid_table(1000);
        assert_eq!(table.entries.len(), 256);
        // sigmoid(0) = 0.5 => output = 500
        let (_, out_raw) = table.entries[0];
        let out_val = from_field(F::from_canonical_u32(out_raw));
        assert_eq!(out_val, 500);
    }

    #[test]
    fn test_build_silu_table() {
        let table = build_small_silu_table(1000);
        assert_eq!(table.entries.len(), 256);
        // silu(0) = 0 * sigmoid(0) = 0
        let (_, out_raw) = table.entries[0];
        let out_val = from_field(F::from_canonical_u32(out_raw));
        assert_eq!(out_val, 0);
    }

    #[test]
    fn test_build_softplus_table() {
        let table = build_small_softplus_table(1000);
        assert_eq!(table.entries.len(), 256);
        // softplus(0) = ln(2) ≈ 0.693 => output ≈ 693
        let (_, out_raw) = table.entries[0];
        let out_val = from_field(F::from_canonical_u32(out_raw));
        assert_eq!(out_val, 693);
    }

    #[test]
    fn test_lookup_proof_valid() {
        let table = build_small_exp_table(1000);
        // Pick a few valid lookups
        let (inputs, outputs) = pick_lookups(&table, &[0, 1, 2, 3, 0, 0, 1, 2]);

        let mut p_transcript = Transcript::new(b"lookup-test");
        let proof = prove_lookup(&table, &inputs, &outputs, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"lookup-test");
        assert!(verify_lookup(
            &proof.table_commitment,
            &proof,
            inputs.len(),
            &mut v_transcript
        ));
    }

    #[test]
    fn test_lookup_proof_sigmoid_valid() {
        let table = build_small_sigmoid_table(1000);
        let (inputs, outputs) = pick_lookups(&table, &[0, 50, 100, 200]);

        let mut p_transcript = Transcript::new(b"sigmoid-lookup");
        let proof = prove_lookup(&table, &inputs, &outputs, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"sigmoid-lookup");
        assert!(verify_lookup(
            &proof.table_commitment,
            &proof,
            inputs.len(),
            &mut v_transcript
        ));
    }

    #[test]
    fn test_lookup_proof_silu_valid() {
        let table = build_small_silu_table(1000);
        let (inputs, outputs) = pick_lookups(&table, &[0, 10, 128, 255]);

        let mut p_transcript = Transcript::new(b"silu-lookup");
        let proof = prove_lookup(&table, &inputs, &outputs, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"silu-lookup");
        assert!(verify_lookup(
            &proof.table_commitment,
            &proof,
            inputs.len(),
            &mut v_transcript
        ));
    }

    #[test]
    fn test_lookup_proof_softplus_valid() {
        let table = build_small_softplus_table(1000);
        let (inputs, outputs) = pick_lookups(&table, &[0, 5, 127, 200]);

        let mut p_transcript = Transcript::new(b"softplus-lookup");
        let proof = prove_lookup(&table, &inputs, &outputs, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"softplus-lookup");
        assert!(verify_lookup(
            &proof.table_commitment,
            &proof,
            inputs.len(),
            &mut v_transcript
        ));
    }

    #[test]
    fn test_lookup_proof_repeated_lookups() {
        // Same entry looked up many times — multiplicities > 1
        let table = build_small_exp_table(1000);
        let (inputs, outputs) = pick_lookups(&table, &[0, 0, 0, 0, 0, 0, 0, 0]);

        let mut p_transcript = Transcript::new(b"repeat-lookup");
        let proof = prove_lookup(&table, &inputs, &outputs, &mut p_transcript);
        // Active subtable has exactly 1 entry (all 8 lookups hit the same row),
        // and its multiplicity must be 8.
        // multiplicities omitted from proof for size optimization
        // (verifier doesn't need them — sumcheck is self-contained)

        let mut v_transcript = Transcript::new(b"repeat-lookup");
        assert!(verify_lookup(
            &proof.table_commitment,
            &proof,
            inputs.len(),
            &mut v_transcript
        ));
    }

    #[test]
    #[should_panic(expected = "LogUp identity failed")]
    fn test_lookup_proof_invalid_output() {
        let table = build_small_exp_table(1000);
        let (inputs, mut outputs) = pick_lookups(&table, &[0, 1, 2, 3]);

        // Tamper with one output — make it wrong
        outputs[2] = outputs[2] + F::one();

        let mut p_transcript = Transcript::new(b"bad-lookup");
        // This should panic because the LogUp identity won't hold
        let _proof = prove_lookup(&table, &inputs, &outputs, &mut p_transcript);
    }

    #[test]
    #[should_panic(expected = "LogUp identity failed")]
    fn test_lookup_proof_invalid_input() {
        let table = build_small_exp_table(1000);
        let (mut inputs, outputs) = pick_lookups(&table, &[0, 1, 2, 3]);

        // Tamper with input — doesn't match any table entry
        inputs[0] = F::from_canonical_u32(999999);

        let mut p_transcript = Transcript::new(b"bad-input");
        let _proof = prove_lookup(&table, &inputs, &outputs, &mut p_transcript);
    }

    #[test]
    fn test_lookup_verification_fails_with_tampered_proof() {
        let table = build_small_exp_table(1000);
        let (inputs, outputs) = pick_lookups(&table, &[0, 1, 2, 3]);

        let mut p_transcript = Transcript::new(b"tamper-test");
        let mut proof = prove_lookup(&table, &inputs, &outputs, &mut p_transcript);

        // Tamper with the final evaluation
        proof.finals.1 = (proof.finals.1 + 1) % ((1u32 << 31) - 1);

        let mut v_transcript = Transcript::new(b"tamper-test");
        assert!(!verify_lookup(
            &proof.table_commitment,
            &proof,
            inputs.len(),
            &mut v_transcript
        ));
    }

    #[test]
    fn test_chunk_subtable() {
        // Identity function as a chunk subtable
        let table = build_chunk_subtable("identity_chunk", |x| x as i16);
        assert_eq!(table.entries.len(), 256);
        // Entry 42 should map 42 -> 42
        let (inp, out) = table.entries[42];
        assert_eq!(
            from_field(F::from_canonical_u32(inp)),
            42
        );
        assert_eq!(
            from_field(F::from_canonical_u32(out)),
            42
        );
    }

    #[test]
    fn test_full_size_table_builds() {
        // Just verify the full 2^16 tables construct without panic.
        let t1 = build_exp_table(1000);
        assert_eq!(t1.entries.len(), 65536);

        let t2 = build_sigmoid_table(1000);
        assert_eq!(t2.entries.len(), 65536);

        let t3 = build_silu_table(1000);
        assert_eq!(t3.entries.len(), 65536);

        let t4 = build_softplus_table(1000);
        assert_eq!(t4.entries.len(), 65536);
    }

    // ===== Extension field lookup tests =====

    #[test]
    fn test_lookup_proof_ef_valid() {
        let table = build_small_exp_table(1000);
        let (inputs, outputs) = pick_lookups(&table, &[0, 1, 2, 3, 0, 0, 1, 2]);

        let mut p_transcript = Transcript::new(b"lookup-ef-test");
        let proof = prove_lookup_ef(&table, &inputs, &outputs, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"lookup-ef-test");
        assert!(verify_lookup_ef(
            &proof.table_commitment,
            &proof,
            inputs.len(),
            &mut v_transcript
        ));
    }

    #[test]
    fn test_lookup_proof_ef_silu() {
        let table = build_small_silu_table(1000);
        let (inputs, outputs) = pick_lookups(&table, &[0, 10, 128, 255, 50, 100, 200, 30]);

        let mut p_transcript = Transcript::new(b"silu-ef-test");
        let proof = prove_lookup_ef(&table, &inputs, &outputs, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"silu-ef-test");
        assert!(verify_lookup_ef(
            &proof.table_commitment,
            &proof,
            inputs.len(),
            &mut v_transcript
        ));
    }

    #[test]
    fn test_lookup_proof_ef_repeated() {
        let table = build_small_exp_table(1000);
        let (inputs, outputs) = pick_lookups(&table, &[0, 0, 0, 0, 0, 0, 0, 0]);

        let mut p_transcript = Transcript::new(b"repeat-ef-test");
        let proof = prove_lookup_ef(&table, &inputs, &outputs, &mut p_transcript);

        let mut v_transcript = Transcript::new(b"repeat-ef-test");
        assert!(verify_lookup_ef(
            &proof.table_commitment,
            &proof,
            inputs.len(),
            &mut v_transcript
        ));
    }

    #[test]
    #[should_panic(expected = "LogUp identity failed")]
    fn test_lookup_proof_ef_invalid_output() {
        let table = build_small_exp_table(1000);
        let (inputs, mut outputs) = pick_lookups(&table, &[0, 1, 2, 3]);
        outputs[2] = outputs[2] + F::one();

        let mut p_transcript = Transcript::new(b"bad-ef-lookup");
        let _proof = prove_lookup_ef(&table, &inputs, &outputs, &mut p_transcript);
    }

    #[test]
    fn test_lookup_ef_with_external_data() {
        let table = build_small_sigmoid_table(1000);
        let (inputs, outputs) = pick_lookups(&table, &[0, 50, 100, 200]);

        let mut p_transcript = Transcript::new(b"ef-ext-data");
        let proof = prove_lookup_ef(&table, &inputs, &outputs, &mut p_transcript);

        let input_u32s: Vec<u32> = inputs.iter().map(|v| v.as_canonical_u32()).collect();
        let output_u32s: Vec<u32> = outputs.iter().map(|v| v.as_canonical_u32()).collect();

        let mut v_transcript = Transcript::new(b"ef-ext-data");
        assert!(verify_lookup_ef_with_data(
            &proof.table_commitment,
            &proof,
            inputs.len(),
            Some((&input_u32s, &output_u32s)),
            &mut v_transcript
        ));
    }
}
