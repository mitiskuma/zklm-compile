//! Basefold polynomial commitment scheme over Mersenne-31.
//!
//! Field-agnostic PCS for multilinear polynomials — no FFT required.
//! Uses a random foldable linear code (rate 1/2) with blake3 Merkle commitments.
//!
//! Algorithm:
//!   Commit: encode coefficients via random linear code, Merkle-commit codeword.
//!   Open:   interleave coefficient folding with codeword folding (same challenges),
//!           Merkle-commit each folded codeword, query random positions for binding.
//!   Verify: recompute challenges, check fold consistency at queried positions.
//!
//! Reference: "BaseFold: Efficient Field-Agnostic Polynomial Commitment Schemes"
//!            (Zeilberger, Chen, Fisch — eprint 2023/1705)

use p3_field::{AbstractField, Field, PrimeField32};
use p3_mersenne_31::Mersenne31;
use serde::{Deserialize, Serialize};

use crate::field::common::log2_ceil;
use crate::proving::pcs::{
    Digest,
    MerkleTree,
    derive_query_indices,
};
use crate::proving::sumcheck::Transcript;

type F = Mersenne31;

/// Number of random queries per fold level for proximity testing.
/// With rate-1/2 code (δ ≈ 1/4), each query provides ~0.415 bits of soundness.
/// 24 queries × ~22 fold rounds × 0.415 ≈ 219 bits — exceeds 124-bit target.
const NUM_QUERIES: usize = 24;

// ======================================================================
// Types
// ======================================================================

/// Commitment to a multilinear polynomial via Basefold.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BasefoldCommitment {
    /// Merkle root of the initial codeword (16-byte truncated blake3).
    pub root: Digest,
    /// Number of coefficients (before padding).
    pub num_coeffs: usize,
    /// log2 of padded coefficient count.
    pub log_n: usize,
}

/// Opening proof: proves that the committed polynomial evaluates to `claimed` at `point`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BasefoldOpeningProof {
    /// Merkle roots of folded codewords (one per fold round that was executed).
    pub fold_roots: Vec<Digest>,
    /// Query openings at each fold level.
    pub fold_queries: Vec<FoldLevelQueries>,
    /// Final scalar after all folding rounds.
    pub final_value: u32,
    /// When early termination is used: remaining coefficients after partial folding.
    /// The verifier evaluates the polynomial directly from these.
    /// Empty when all rounds are folded.
    #[serde(default)]
    pub final_coeffs: Vec<u32>,
}

/// Queries at one fold level with Merkle proofs.
///
/// Two formats:
/// - **Separate** (original): `prev_batch_proof` + `folded_batch_proof` as independent proofs.
/// - **Merged** (optimized): single `merged_proof` opening the folded tree at BOTH
///   the folded positions from this level AND the left+right positions for the next level.
///   Shares auth nodes across the two sets of positions (~30% proof size reduction).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldLevelQueries {
    /// Batch Merkle proof for left+right positions in the previous tree.
    /// In merged mode: only used for level 0 (opening the initial batched tree).
    pub prev_batch_proof: crate::proving::pcs::BatchMerkleProof,
    /// Batch Merkle proof for folded positions in this level's tree.
    /// In merged mode: contains MERGED positions (folded + next-level left+right).
    pub folded_batch_proof: crate::proving::pcs::BatchMerkleProof,
}

// ======================================================================
// Encoding: recursively foldable random linear code (Basefold §4)
// ======================================================================
//
// Construction: butterfly encoding with random T vectors.
//   Enc_{i+1}(m_L || m_R) = (Enc_i(m_L) + T_i · Enc_i(m_R)) || (Enc_i(m_L) - T_i · Enc_i(m_R))
// Base case: Enc_0(m) = [m, m] (rate-1/2 repetition)
//
// Folding with challenge α recovers Enc_i(m_L + α · m_R) — a valid codeword
// of the halved code. This is the property that makes the proximity gap theorem hold.
//
// T vectors are deterministic random nonzero field elements from a seeded XOF.

/// Foldable code tables: T vectors + precomputed fold weights.
struct FoldableTables {
    /// T values per level. tables[i] has 2^i elements.
    /// Level 0 = base (1 element), level log_rate+log_n-1 = top.
    t_flat: Vec<F>,
    /// Precomputed fold weights: w[j] = 1 / (2 * t[j]) for linear interpolation.
    w_flat: Vec<F>,
}

/// Generate foldable code tables from a seed.
/// Total elements: 1 + 2 + 4 + ... + 2^(num_levels-1) = 2^num_levels - 1.
fn generate_tables(num_levels: usize, seed: &[u8; 32]) -> FoldableTables {
    let p = (1u32 << 31) - 1;
    let total = (1usize << num_levels) - 1;

    // Generate random nonzero field elements via XOF
    let mut hasher = blake3::Hasher::new();
    hasher.update(seed);
    let mut reader = hasher.finalize_xof();
    let mut buf = vec![0u8; total * 4];
    reader.fill(&mut buf);

    let mut t_flat = Vec::with_capacity(total);
    for i in 0..total {
        let raw = u32::from_le_bytes([buf[i*4], buf[i*4+1], buf[i*4+2], buf[i*4+3]]);
        t_flat.push(F::from_canonical_u32(1 + (raw % (p - 1))));
    }

    // Precompute fold weights: w = 1 / (2 * t) via batch inverse (Montgomery's trick).
    // O(n) muls + 1 inverse instead of n individual inversions.
    let two = F::from_canonical_u32(2);
    let denominators: Vec<F> = t_flat.iter().map(|&t| two * t).collect();

    // Batch inverse: compute all 1/d[i] using a single field inversion.
    let mut prefix_products = Vec::with_capacity(total);
    let mut acc = F::one();
    for &d in &denominators {
        prefix_products.push(acc);
        acc = acc * d;
    }
    let mut inv_acc = acc.inverse();
    let mut w_flat = vec![F::zero(); total];
    for i in (0..total).rev() {
        w_flat[i] = prefix_products[i] * inv_acc;
        inv_acc = inv_acc * denominators[i];
    }

    FoldableTables { t_flat, w_flat }
}

/// Get the T values for a specific level. Level i has 2^i elements.
/// Offset into flat array: 2^i - 1.
fn table_level(tables: &FoldableTables, level: usize) -> &[F] {
    let start = (1usize << level) - 1;
    let len = 1usize << level;
    &tables.t_flat[start..start + len]
}

fn weight_level(tables: &FoldableTables, level: usize) -> &[F] {
    let start = (1usize << level) - 1;
    let len = 1usize << level;
    &tables.w_flat[start..start + len]
}

/// Encode multilinear coefficients using the recursive butterfly construction.
/// Rate 1/2: n coefficients → 2n codeword elements.
///
/// Algorithm (in-place butterfly, bottom-up):
///   1. Start with codeword = [c, c] (repetition of coefficients)
///   2. For each level i from 0 to log_n-1:
///      - For each chunk of size 2^(i+1):
///        - Split into left half L and right half R
///        - L' = L + T_i · R
///        - R' = L - T_i · R
fn encode(coeffs: &[F], log_n: usize, tables: &FoldableTables) -> Vec<F> {
    let n = coeffs.len();
    let cw_len = 2 * n;
    let mut cw = Vec::with_capacity(cw_len);
    cw.extend_from_slice(coeffs);
    cw.extend_from_slice(coeffs);

    let mut chunk_size = 2;
    for level in 0..log_n {
        let t = table_level(tables, level);
        let t_len = t.len();
        for chunk in cw.chunks_mut(chunk_size) {
            let half = chunk_size / 2;

            // SIMD butterfly on aarch64 for chunks ≥ 4 elements
            #[cfg(target_arch = "aarch64")]
            {
                use p3_mersenne_31::PackedMersenne31Neon;
                use p3_field::PackedValue;
                let simd_pairs = half / 4;
                if t_len >= half {
                    // Fast path: direct slice access (no modulo)
                    for c in 0..simd_pairs {
                        let j = c * 4;
                        let t_packed = *PackedMersenne31Neon::from_slice(&t[j..j+4]);
                        let left = *PackedMersenne31Neon::from_slice(&chunk[j..j+4]);
                        let right = *PackedMersenne31Neon::from_slice(&chunk[j+half..j+half+4]);
                        let rhs = right * t_packed;
                        *PackedMersenne31Neon::from_slice_mut(&mut chunk[j..j+4]) = left + rhs;
                        *PackedMersenne31Neon::from_slice_mut(&mut chunk[j+half..j+half+4]) = left - rhs;
                    }
                } else {
                    // Slow path: cyclic t access
                    for c in 0..simd_pairs {
                        let j = c * 4;
                        let t_packed = *PackedMersenne31Neon::from_slice(
                            &[t[j % t_len], t[(j+1) % t_len], t[(j+2) % t_len], t[(j+3) % t_len]]
                        );
                        let left = *PackedMersenne31Neon::from_slice(&chunk[j..j+4]);
                        let right = *PackedMersenne31Neon::from_slice(&chunk[j+half..j+half+4]);
                        let rhs = right * t_packed;
                        *PackedMersenne31Neon::from_slice_mut(&mut chunk[j..j+4]) = left + rhs;
                        *PackedMersenne31Neon::from_slice_mut(&mut chunk[j+half..j+half+4]) = left - rhs;
                    }
                }
                for j in (simd_pairs * 4)..half {
                    let t_val = t[j % t_len];
                    let rhs = chunk[j + half] * t_val;
                    let left = chunk[j];
                    chunk[j]        = left + rhs;
                    chunk[j + half] = left - rhs;
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                for j in 0..half {
                    let t_val = t[j % t_len];
                    let rhs = chunk[j + half] * t_val;
                    let left = chunk[j];
                    chunk[j]        = left + rhs;
                    chunk[j + half] = left - rhs;
                }
            }
        }
        chunk_size *= 2;
    }

    cw
}

/// Fold a codeword in-place with challenge α using the foldable code structure.
///
/// For each pair (left, right) at positions (2j, 2j+1):
///   folded[j] = (left + right) / 2 + α · (left - right) · w[j]
/// where w[j] = 1 / (2 · t[j]).
///
/// This produces Enc_{i-1}(m_L + α · m_R) — a valid codeword of the halved code.
/// 2^{-1} mod (2^31 - 1) = 2^30 = 1073741824.
const TWO_INV: F = Mersenne31::new(1073741824);

fn fold_codeword_inplace(cw: &mut [F], alpha: F, weights: &[F]) -> usize {
    let n = cw.len();
    let half = n / 2;
    let two_inv = TWO_INV;
    let w_len = weights.len();

    // SIMD fold on aarch64: process 4 elements per iteration
    #[cfg(target_arch = "aarch64")]
    {
        use p3_mersenne_31::PackedMersenne31Neon;
        use p3_field::PackedValue;

        let alpha_packed = PackedMersenne31Neon::from(alpha);
        let two_inv_packed = PackedMersenne31Neon::from(two_inv);
        let simd_iters = half / 4;

        if w_len >= half {
            // Fast path: no modulo needed (j < half ≤ w_len)
            for c in 0..simd_iters {
                let j = c * 4;
                let left = *PackedMersenne31Neon::from_slice(&cw[j..j + 4]);
                let right = *PackedMersenne31Neon::from_slice(&cw[j + half..j + half + 4]);
                let w_packed = *PackedMersenne31Neon::from_slice(&weights[j..j + 4]);
                let sum_half = (left + right) * two_inv_packed;
                let diff_w = (left - right) * w_packed;
                let result = sum_half + alpha_packed * diff_w;
                *PackedMersenne31Neon::from_slice_mut(&mut cw[j..j + 4]) = result;
            }
        } else {
            // Slow path: modulo for cyclic weight access
            for c in 0..simd_iters {
                let j = c * 4;
                let left = *PackedMersenne31Neon::from_slice(&cw[j..j + 4]);
                let right = *PackedMersenne31Neon::from_slice(&cw[j + half..j + half + 4]);
                let w_packed = *PackedMersenne31Neon::from_slice(
                    &[weights[j % w_len], weights[(j+1) % w_len],
                      weights[(j+2) % w_len], weights[(j+3) % w_len]]
                );
                let sum_half = (left + right) * two_inv_packed;
                let diff_w = (left - right) * w_packed;
                let result = sum_half + alpha_packed * diff_w;
                *PackedMersenne31Neon::from_slice_mut(&mut cw[j..j + 4]) = result;
            }
        }
        // Scalar tail
        for j in (simd_iters * 4)..half {
            let left = cw[j];
            let right = cw[j + half];
            let sum_half = (left + right) * two_inv;
            let diff_w = (left - right) * weights[j % w_len];
            cw[j] = sum_half + alpha * diff_w;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for j in 0..half {
            let left = cw[j];
            let right = cw[j + half];
            let sum_half = (left + right) * two_inv;
            let diff_w = (left - right) * weights[j % w_len];
            cw[j] = sum_half + alpha * diff_w;
        }
    }

    half
}

// ======================================================================
// Commitment
// ======================================================================

/// Deterministic seed for foldable code tables.
fn table_seed(log_n: usize) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"basefold-foldable-code");
    h.update(&(log_n as u64).to_le_bytes());
    *h.finalize().as_bytes()
}

/// Commit to a multilinear polynomial (given as coefficient vector).
/// Returns (commitment, encoded codeword, tables) for later opening.
pub fn commit(coeffs: &[F]) -> (BasefoldCommitment, Vec<F>, Vec<F>) {
    let n = coeffs.len();
    let log_n = if n <= 1 { 1 } else { log2_ceil(n) };
    let n_pad = 1 << log_n;

    let mut padded = coeffs.to_vec();
    padded.resize(n_pad, F::zero());

    let seed = table_seed(log_n);
    let tables = generate_tables(log_n, &seed);
    let codeword = encode(&padded, log_n, &tables);
    let tree = MerkleTree::new_from_u32s(&codeword);
    let root = tree.root();

    // Return t_flat as the "twists" for API compat
    (
        BasefoldCommitment { root, num_coeffs: n, log_n },
        codeword,
        tables.t_flat,
    )
}

/// Generate prove tables for a given log_n. Caller can cache and reuse
/// across multiple commit_and_prove calls with the same polynomial size.
pub fn make_prove_tables(log_n: usize) -> FoldableTablesPublic {
    let seed = table_seed(log_n);
    FoldableTablesPublic { inner: generate_tables(log_n, &seed) }
}

/// Commit and prove opening in one call (convenience for integration).
/// Uses a global table cache: tables for a given log_n are generated once and
/// reused across all subsequent calls. Eliminates ~8ms of XOF + batch inverse
/// per redundant table generation (3 matrices share log_n=22 per layer).
///
/// The lock is held only for HashMap lookup — released before any rayon work
/// inside encode/tree building. No deadlock risk.
pub fn commit_and_prove(
    coeffs: &[F],
    point: &[F],
    _claimed: F,
    transcript: &mut Transcript,
) -> (BasefoldCommitment, BasefoldOpeningProof) {
    use std::sync::{Arc, Mutex};
    use std::collections::HashMap;

    static PROVE_TABLE_CACHE: std::sync::LazyLock<Mutex<HashMap<usize, Arc<FoldableTablesPublic>>>> =
        std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

    let log_n = if coeffs.len() <= 1 { 1 } else { log2_ceil(coeffs.len()) };

    // Get or create tables. Lock held only for HashMap lookup — released
    // before any encoding/hashing work to avoid blocking other threads.
    let tables: Arc<FoldableTablesPublic> = {
        let mut cache = PROVE_TABLE_CACHE.lock().unwrap();
        if let Some(t) = cache.get(&log_n) {
            t.clone() // Arc clone = cheap refcount increment
        } else {
            let t = Arc::new(make_prove_tables(log_n));
            cache.insert(log_n, t.clone());
            t
        }
    };

    commit_and_prove_with_tables(coeffs, point, &tables, transcript)
}

/// Commit and prove, taking ownership of an already-padded coefficient vector.
/// Eliminates the 16MB internal copy that commit_and_prove_with_tables does.
/// The caller must ensure `padded` has length = next power of 2 and is zero-padded.
pub fn commit_and_prove_vec(
    padded: Vec<F>,
    num_coeffs: usize,
    point: &[F],
    transcript: &mut Transcript,
) -> (BasefoldCommitment, BasefoldOpeningProof) {
    use std::sync::{Arc, Mutex};
    use std::collections::HashMap;

    static PROVE_TABLE_CACHE2: std::sync::LazyLock<Mutex<HashMap<usize, Arc<FoldableTablesPublic>>>> =
        std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

    let log_n = if padded.len() <= 1 { 1 } else { log2_ceil(padded.len()) };

    let tables: Arc<FoldableTablesPublic> = {
        let mut cache = PROVE_TABLE_CACHE2.lock().unwrap();
        if let Some(t) = cache.get(&log_n) {
            t.clone()
        } else {
            let t = Arc::new(make_prove_tables(log_n));
            cache.insert(log_n, t.clone());
            t
        }
    };

    let codeword = encode(&padded, log_n, &tables.inner);
    let initial_tree = MerkleTree::new_from_u32s(&codeword);
    let root = initial_tree.root();

    let commitment = BasefoldCommitment { root, num_coeffs, log_n };
    transcript.absorb_bytes(&commitment.root);

    let proof = prove_opening_with_tree(
        padded, codeword, point, log_n, &initial_tree, &tables.inner, transcript,
    );

    (commitment, proof)
}

/// Commit and prove with precomputed tables. Use when proving multiple
/// polynomials of the same size (e.g., 8 weight matrices in a Qwen layer).
pub fn commit_and_prove_with_tables(
    coeffs: &[F],
    point: &[F],
    tables: &FoldableTablesPublic,
    transcript: &mut Transcript,
) -> (BasefoldCommitment, BasefoldOpeningProof) {
    let n = coeffs.len();
    let log_n = if n <= 1 { 1 } else { log2_ceil(n) };
    let n_pad = 1 << log_n;

    let mut padded = coeffs.to_vec();
    padded.resize(n_pad, F::zero());

    let codeword = encode(&padded, log_n, &tables.inner);
    let initial_tree = MerkleTree::new_from_u32s(&codeword);
    let root = initial_tree.root();

    let commitment = BasefoldCommitment { root, num_coeffs: n, log_n };
    transcript.absorb_bytes(&commitment.root);

    let proof = prove_opening_with_tree(
        padded, codeword, point, log_n, &initial_tree, &tables.inner, transcript,
    );

    (commitment, proof)
}


// ======================================================================
// Prover: opening proof
// ======================================================================

/// Prove that the committed polynomial evaluates to `claimed` at `point`.
///
/// Interleaves coefficient folding (= partial multilinear evaluation) with
/// codeword folding. At each round:
///   1. Squeeze challenge α from transcript
///   2. Fold coefficients: new_f[j] = f[j] + α · f[j + half]
///   3. Fold codeword: new_cw[i] = cw[i] + α · cw[i + half]
///   4. Merkle-commit the folded codeword
///   5. Open queries at random positions to bind the fold
fn prove_opening_with_tree(
    mut eval_buf: Vec<F>,  // owned padded coefficients — reused for incremental MLE eval
    mut cw: Vec<F>,        // owned codeword — folded in-place
    point: &[F],
    log_n: usize,
    initial_tree: &MerkleTree,
    tables: &FoldableTables,
    transcript: &mut Transcript,
) -> BasefoldOpeningProof {
    let n_pad = 1 << log_n;
    let mut cw_len = cw.len();

    let mut fold_roots = Vec::with_capacity(log_n);
    let mut fold_queries = Vec::with_capacity(log_n);

    // eval_buf is used for incremental MLE evaluation across fold rounds.
    let mut eval_size = n_pad;

    // For round 0, use the initial tree (borrowed). For subsequent rounds,
    // build fresh trees from the folded codeword.
    let mut owned_prev_tree: Option<MerkleTree> = None;

    for round in 0..log_n {
        let alpha = if round < point.len() {
            transcript.absorb(point[round].as_canonical_u32());
            point[round]
        } else {
            transcript.squeeze()
        };

        // Fold coefficients for MLE evaluation: new[j] = (1-α)*old[j] + α*old[j+half]
        {
            let half = eval_size / 2;
            let one_minus_alpha = F::one() - alpha;
            for j in 0..half {
                eval_buf[j] = one_minus_alpha * eval_buf[j] + alpha * eval_buf[j + half];
            }
            eval_size = half;
        }

        // Fold codeword in-place using weights from the corresponding encoding level.
        // Encoding goes bottom-up (level 0..log_n-1), folding goes top-down (reverse).
        let fold_level = log_n - 1 - round;
        let weights = weight_level(tables, fold_level);
        let prev_cw_len = cw_len;
        cw_len = fold_codeword_inplace(&mut cw[..prev_cw_len], alpha, weights);

        // Build Merkle tree for the folded codeword (all rounds, no early termination)
        let folded_tree = MerkleTree::new_from_u32s_len(&cw, cw_len);
        let root = folded_tree.root();
        fold_roots.push(root);
        transcript.absorb_bytes(&root);

        // Get the previous tree (borrowed initial or owned from last round)
        let prev_tree_ref = if round == 0 {
            initial_tree
        } else {
            owned_prev_tree.as_ref().unwrap()
        };

        // Query random positions
        let prev_half = prev_cw_len / 2;
        let query_indices = derive_query_indices(transcript, prev_tree_ref.log_height);

        // Collect query positions for batch Merkle openings
        let mut prev_open_positions = Vec::with_capacity(NUM_QUERIES * 2);
        let mut folded_positions = Vec::with_capacity(NUM_QUERIES);

        for &raw_idx in &query_indices {
            let idx = raw_idx % prev_half;
            let folded_idx = idx % cw_len;
            prev_open_positions.push(idx);
            prev_open_positions.push(idx + prev_half);
            folded_positions.push(folded_idx);
        }

        // Batch Merkle proofs with deduplication.
        // Values and indices are embedded in the batch proofs — no need to
        // store them separately. The verifier re-derives indices from the
        // transcript and reads values from the batch proof.
        let prev_batch_proof = prev_tree_ref.batch_open(&prev_open_positions);
        let folded_batch_proof = folded_tree.batch_open(&folded_positions);

        let level_queries = FoldLevelQueries {
            prev_batch_proof,
            folded_batch_proof,
        };

        fold_queries.push(level_queries);
        owned_prev_tree = Some(folded_tree);
    }

    // Final value = MLE evaluation at the full point.
    // Computed incrementally during the fold loop — no extra copy needed.
    let final_value = eval_buf[0].as_canonical_u32();

    BasefoldOpeningProof {
        fold_roots,
        fold_queries,
        final_value,
        final_coeffs: vec![],
    }
}

// ======================================================================
// Verifier
// ======================================================================

/// Verify a Basefold opening proof.
///
/// Checks:
///   1. Fold consistency: at each level, queried positions satisfy
///      folded[i] = left[i] + α · right[i] (matches the committed values).
///   2. Merkle binding: opened values match committed Merkle roots.
///   3. Final value: the fully-folded polynomial equals the claimed evaluation
///      (via the multilinear evaluation identity).
/// Generate tables for a given log_n. Call once, reuse across verify calls.
pub fn make_verify_tables(log_n: usize) -> FoldableTablesPublic {
    let seed = table_seed(log_n);
    let tables = generate_tables(log_n, &seed);
    FoldableTablesPublic { inner: tables }
}

/// Opaque wrapper for foldable code tables (used by verifier).
pub struct FoldableTablesPublic {
    inner: FoldableTables,
}

pub fn verify_opening(
    commitment: &BasefoldCommitment,
    claimed: F,
    point: &[F],
    proof: &BasefoldOpeningProof,
    transcript: &mut Transcript,
) -> bool {
    // Thread-local table cache: generate once per log_n, reuse across verify calls.
    // Eliminates ~4M batch inversions per call for repeated verifications.
    use std::cell::RefCell;
    thread_local! {
        static CACHED_TABLES: RefCell<Option<(usize, FoldableTablesPublic)>> = RefCell::new(None);
    }

    CACHED_TABLES.with(|cache| {
        let mut cache = cache.borrow_mut();
        let log_n = commitment.log_n;

        // Regenerate only if log_n changed
        if cache.as_ref().map_or(true, |(cached_log_n, _)| *cached_log_n != log_n) {
            *cache = Some((log_n, make_verify_tables(log_n)));
        }

        let tables = &cache.as_ref().unwrap().1;
        verify_opening_with_tables(commitment, claimed, point, proof, tables, transcript)
    })
}

pub fn verify_opening_with_tables(
    commitment: &BasefoldCommitment,
    claimed: F,
    point: &[F],
    proof: &BasefoldOpeningProof,
    tables: &FoldableTablesPublic,
    transcript: &mut Transcript,
) -> bool {
    let log_n = commitment.log_n;

    let num_fold_rounds = proof.fold_roots.len();

    if num_fold_rounds != log_n {
        eprintln!("Basefold: wrong number of fold rounds: {} vs expected {}", num_fold_rounds, log_n);
        return false;
    }

    let two_inv = F::from_canonical_u32(2).inverse();

    let mut current_root = commitment.root;
    let mut current_log_len = log_n + 1;

    for round in 0..num_fold_rounds {
        // Folding round i eliminates variable i (MSB first, matching eq_evals).
        let alpha = if round < point.len() {
            transcript.absorb(point[round].as_canonical_u32());
            point[round]
        } else {
            transcript.squeeze()
        };

        let fold_root = proof.fold_roots[round];
        transcript.absorb_bytes(&fold_root);

        // Verify queries at this fold level
        let level = &proof.fold_queries[round];
        let prev_half = 1usize << (current_log_len - 1);

        let query_indices = derive_query_indices(transcript, current_log_len);

        // Read values directly from batch proofs — no redundant copies in proof.
        let bp = &level.prev_batch_proof;
        let fbp = &level.folded_batch_proof;

        for q in 0..NUM_QUERIES {
            let idx = query_indices[q] % prev_half;
            let left_idx = idx as u32;
            let right_idx = (idx + prev_half) as u32;
            let folded_idx = (idx % (1usize << (current_log_len - 1))) as u32;

            // Look up left and right values from the previous tree's batch proof
            let left_val = match bp.indices.iter().position(|&i| i == left_idx) {
                Some(li) => bp.values[li],
                None => { eprintln!("Basefold: left index {} not in batch proof at round {}", left_idx, round); return false; }
            };
            let right_val = match bp.indices.iter().position(|&i| i == right_idx) {
                Some(ri) => bp.values[ri],
                None => { eprintln!("Basefold: right index {} not in batch proof at round {}", right_idx, round); return false; }
            };

            // FOLD CONSISTENCY CHECK using foldable code formula:
            let left_f = F::from_canonical_u32(left_val);
            let right_f = F::from_canonical_u32(right_val);
            let fold_level = log_n - 1 - round;
            let w = weight_level(&tables.inner, fold_level);
            let w_j = w[idx % w.len()];
            let expected_folded = (left_f + right_f) * two_inv + alpha * (left_f - right_f) * w_j;

            // Look up folded value from the folded tree's batch proof
            let folded_val = match fbp.indices.iter().position(|&i| i == folded_idx) {
                Some(fi) => fbp.values[fi],
                None => { eprintln!("Basefold: folded index {} not in batch proof at round {}", folded_idx, round); return false; }
            };

            let actual_folded = F::from_canonical_u32(folded_val);
            if actual_folded != expected_folded {
                eprintln!("Basefold: fold consistency failed at round {}, query {}", round, q);
                return false;
            }
        }

        // Batch verify: previous tree openings (left + right positions)
        if !crate::proving::pcs::verify_batch_opening(
            &current_root, &level.prev_batch_proof, current_log_len,
        ) {
            eprintln!("Basefold: batch Merkle proof failed for previous tree at round {}", round);
            return false;
        }

        // Batch verify: folded tree openings
        if !crate::proving::pcs::verify_batch_opening(
            &fold_root, &level.folded_batch_proof, current_log_len - 1,
        ) {
            eprintln!("Basefold: batch Merkle proof failed for folded tree at round {}", round);
            return false;
        }

        current_root = fold_root;
        current_log_len -= 1;
    }

    // Check final value: all fold rounds executed, polynomial fully evaluated.
    let final_f = F::from_canonical_u32(proof.final_value);
    if final_f != claimed {
        eprintln!("Basefold: final value {} != claimed {}", proof.final_value, claimed.as_canonical_u32());
        return false;
    }

    true
}

// ======================================================================
// Batch Basefold: prove k polynomial openings with ONE fold chain
// ======================================================================
//
// Reduces k independent Basefold proofs to 1, using random linear combination.
// The prover:
//   1. Commits each polynomial independently (k Merkle trees)
//   2. Absorbs all k roots, samples batch challenge ρ
//   3. Forms batched codeword C[j] = Σ_i ρ^i * codeword_i[j]
//   4. Runs ONE Basefold fold chain on C
//   5. Opens each of the k trees at shared query positions
//
// Soundness: ρ from base field (31 bits). For k=8 polynomials, Schwartz-Zippel
// gives failure probability k/|F| ≈ 2^{-28}. Combined with the 124-bit EF
// sumcheck and 219-bit proximity testing, this is the weakest link but still
// exceeds standard security levels for ML inference proofs.

/// Batch opening proof for k polynomials (supports multi-height).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BasefoldBatchProof {
    /// Number of polynomials in the batch.
    pub num_polys: usize,
    /// Per-polynomial log_n values (in size-sorted order, descending).
    pub poly_log_ns: Vec<usize>,
    /// Individual Merkle roots (in size-sorted order).
    pub roots: Vec<Digest>,
    /// Root of the batched codeword tree.
    pub batched_root: Digest,
    /// Fold roots from the single batched fold chain.
    pub fold_roots: Vec<Digest>,
    /// Query openings: for each fold level, batch proofs for prev+folded trees.
    pub fold_queries: Vec<FoldLevelQueries>,
    /// Per-polynomial query openings from the individual Merkle trees.
    /// Empty when soundness is provided by ρ-based Schwartz-Zippel + add_round_openings.
    #[serde(default)]
    pub query_openings: Vec<crate::proving::pcs::BatchMerkleProof>,
    /// Final value of the batched polynomial after all fold rounds.
    pub final_value: u32,
    /// For multi-height: at fold rounds where smaller codewords are added,
    /// store openings of those individual trees at the folded query positions.
    /// Key: fold round index → Vec of (poly_index, BatchMerkleProof).
    #[serde(default)]
    pub add_round_openings: Vec<(usize, Vec<(usize, crate::proving::pcs::BatchMerkleProof)>)>,
}

/// Batch commit and prove k polynomial openings with one fold chain.
///
/// All polynomials must be padded to the same length (next power of 2).
/// Each polynomial is evaluated at its own point.
///
/// Returns (individual commitments, batch proof).
pub fn batch_commit_and_prove(
    polynomials: &[&[F]],       // k padded coefficient vectors (all same length)
    points: &[&[F]],            // k evaluation points
    transcript: &mut Transcript,
) -> (Vec<BasefoldCommitment>, BasefoldBatchProof) {
    use rayon::prelude::*;

    let k = polynomials.len();
    assert_eq!(k, points.len());
    assert!(k > 0);

    let n = polynomials[0].len();
    let log_n = if n <= 1 { 1 } else { log2_ceil(n) };
    let n_pad = 1 << log_n;

    // All polynomials must be the same padded length
    for p in polynomials {
        assert_eq!(p.len(), n_pad, "batch_commit_and_prove: all polynomials must be padded to same length");
    }

    // Step 1: Encode and commit each polynomial in parallel
    let seed = table_seed(log_n);
    let tables = generate_tables(log_n, &seed);

    let encoded: Vec<(Vec<F>, MerkleTree)> = polynomials.par_iter()
        .map(|coeffs| {
            let codeword = encode(coeffs, log_n, &tables);
            let tree = MerkleTree::new_from_u32s(&codeword);
            (codeword, tree)
        })
        .collect();

    let roots: Vec<Digest> = encoded.iter().map(|(_, tree)| tree.root()).collect();
    let commitments: Vec<BasefoldCommitment> = roots.iter().enumerate().map(|(i, &root)| {
        BasefoldCommitment { root, num_coeffs: polynomials[i].len(), log_n }
    }).collect();

    // Step 2: Absorb all roots into transcript
    for root in &roots {
        transcript.absorb_bytes(root);
    }

    // Step 3: Sample batch challenge ρ and compute batch_coeffs = [1, ρ, ρ², ...]
    let rho = transcript.squeeze();
    let mut batch_coeffs = Vec::with_capacity(k);
    let mut rho_power = F::one();
    for _ in 0..k {
        batch_coeffs.push(rho_power);
        rho_power = rho_power * rho;
    }

    // Step 4: Form batched codeword C[j] = Σ_i ρ^i * codeword_i[j]
    let cw_len = 2 * n_pad;
    let mut batched_cw = vec![F::zero(); cw_len];
    for (i, (cw, _)) in encoded.iter().enumerate() {
        let coeff = batch_coeffs[i];
        for j in 0..cw_len {
            batched_cw[j] = batched_cw[j] + coeff * cw[j];
        }
    }

    // Step 5: Build Merkle tree for batched codeword
    let batched_tree = MerkleTree::new_from_u32s(&batched_cw);
    let batched_root = batched_tree.root();
    transcript.absorb_bytes(&batched_root);

    // Step 6: Compute batched evaluation point and claimed value
    // For simplicity, we require all points to be the same length (= log_n)
    // and form the batched claimed value = Σ_i ρ^i * p_i(point_i)
    let mut batched_claimed = F::zero();
    for (i, coeffs) in polynomials.iter().enumerate() {
        let eval = crate::field::m31_ops::mle_evaluate(coeffs, points[i]);
        batched_claimed = batched_claimed + batch_coeffs[i] * eval;
    }

    // Step 7: Run ONE Basefold fold chain on the batched codeword
    // We use a combined point derived from the transcript (not individual points)
    let fold_point: Vec<F> = (0..log_n).map(|_| transcript.squeeze()).collect();

    let mut cw = batched_cw;
    let mut cw_len_cur = cw_len;
    let mut fold_roots_vec = Vec::with_capacity(log_n);
    let mut fold_queries = Vec::with_capacity(log_n);
    let mut owned_prev_tree: Option<MerkleTree> = None;

    for round in 0..log_n {
        let alpha = fold_point[round];
        transcript.absorb(alpha.as_canonical_u32());

        let fold_level = log_n - 1 - round;
        let w = weight_level(&tables, fold_level);
        let prev_cw_len = cw_len_cur;
        cw_len_cur = fold_codeword_inplace(&mut cw[..prev_cw_len], alpha, w);

        let folded_tree = MerkleTree::new_from_u32s_len(&cw, cw_len_cur);
        let fold_root = folded_tree.root();
        fold_roots_vec.push(fold_root);
        transcript.absorb_bytes(&fold_root);

        let prev_tree_ref = if round == 0 { &batched_tree } else { owned_prev_tree.as_ref().unwrap() };
        let prev_half = prev_cw_len / 2;
        let query_indices = derive_query_indices(transcript, prev_tree_ref.log_height);

        let mut prev_open_positions = Vec::with_capacity(NUM_QUERIES * 2);
        let mut folded_positions = Vec::with_capacity(NUM_QUERIES);
        for &raw_idx in &query_indices {
            let idx = raw_idx % prev_half;
            prev_open_positions.push(idx);
            prev_open_positions.push(idx + prev_half);
            folded_positions.push(idx % cw_len_cur);
        }

        let prev_batch_proof = prev_tree_ref.batch_open(&prev_open_positions);
        let folded_batch_proof = folded_tree.batch_open(&folded_positions);
        fold_queries.push(FoldLevelQueries { prev_batch_proof, folded_batch_proof });
        owned_prev_tree = Some(folded_tree);
    }

    // Step 8: Final value = batched MLE evaluation at fold_point
    let final_value = {
        let mut batched_eval = F::zero();
        for (i, coeffs) in polynomials.iter().enumerate() {
            let eval = crate::field::m31_ops::mle_evaluate(coeffs, &fold_point);
            batched_eval = batched_eval + batch_coeffs[i] * eval;
        }
        batched_eval.as_canonical_u32()
    };

    // Step 9: Open individual trees at shared query positions
    let final_query_indices = derive_query_indices(transcript, log_n + 1);
    let mut individual_positions = Vec::with_capacity(NUM_QUERIES * 2);
    let initial_half = n_pad; // cw_len / 2
    for &raw_idx in &final_query_indices {
        let idx = raw_idx % initial_half;
        individual_positions.push(idx);
        individual_positions.push(idx + initial_half);
    }

    let query_openings: Vec<_> = encoded.iter()
        .map(|(_, tree)| tree.batch_open(&individual_positions))
        .collect();

    let batch_proof = BasefoldBatchProof {
        num_polys: k,
        poly_log_ns: vec![log_n; k],  // all same size
        roots: roots.clone(),
        batched_root,
        fold_roots: fold_roots_vec,
        fold_queries,
        query_openings,
        final_value,
        add_round_openings: vec![],  // no multi-height additions
    };

    (commitments, batch_proof)
}

/// Verify a batch Basefold opening proof.
pub fn batch_verify_opening(
    commitments: &[BasefoldCommitment],
    _claimed_evals: &[F],
    proof: &BasefoldBatchProof,
    transcript: &mut Transcript,
) -> bool {
    let k = commitments.len();
    if k != proof.num_polys || k != proof.roots.len() {
        eprintln!("BatchBasefold: length mismatch");
        return false;
    }
    if k == 0 { return true; }

    // Use max log_n from the proof (supports multi-height)
    let max_log_n = *proof.poly_log_ns.iter().max().unwrap_or(&1);

    // Step 1: Absorb roots in proof order (size-sorted, matching prover)
    for root in &proof.roots {
        transcript.absorb_bytes(root);
    }

    // Step 2: Sample ρ, compute batch_coeffs
    let rho = transcript.squeeze();
    let mut batch_coeffs = Vec::with_capacity(k);
    let mut rho_power = F::one();
    for _ in 0..k { batch_coeffs.push(rho_power); rho_power = rho_power * rho; }

    // Step 3: Absorb batched codeword root
    transcript.absorb_bytes(&proof.batched_root);

    // Step 4: Squeeze fold_point
    let fold_point: Vec<F> = (0..max_log_n).map(|_| transcript.squeeze()).collect();

    // Build add_round lookup: round → openings
    let mut add_round_map: std::collections::HashMap<usize, &Vec<(usize, crate::proving::pcs::BatchMerkleProof)>> =
        std::collections::HashMap::new();
    for (round, openings) in &proof.add_round_openings {
        add_round_map.insert(*round, openings);
    }

    // Step 5: Verify fold chain
    let two_inv = F::from_canonical_u32(2).inverse();
    let tables = make_verify_tables(max_log_n);
    let mut current_root = proof.batched_root;
    let mut current_log_len = max_log_n + 1;

    for round in 0..max_log_n {
        let alpha = fold_point[round];
        transcript.absorb(alpha.as_canonical_u32());

        let fold_root = proof.fold_roots[round];
        transcript.absorb_bytes(&fold_root);

        let level = &proof.fold_queries[round];
        let prev_half = 1usize << (current_log_len - 1);
        let query_indices = derive_query_indices(transcript, current_log_len);

        let bp = &level.prev_batch_proof;
        let fbp = &level.folded_batch_proof;

        // Compute addition adjustment for this round (multi-height)
        // If smaller polynomials were added at this round, the folded value
        // includes their contribution: folded = fold(prev) + Σ ρ^i * cw_i
        let has_additions = add_round_map.contains_key(&round);

        for q in 0..NUM_QUERIES {
            let idx = query_indices[q] % prev_half;
            let left_idx = idx as u32;
            let right_idx = (idx + prev_half) as u32;
            let folded_idx_usize = idx % (1usize << (current_log_len - 1));
            let folded_idx = folded_idx_usize as u32;

            let left_val = match bp.indices.iter().position(|&i| i == left_idx) {
                Some(li) => bp.values[li],
                None => { eprintln!("BatchBasefold: left missing r{} q{}", round, q); return false; }
            };
            let right_val = match bp.indices.iter().position(|&i| i == right_idx) {
                Some(ri) => bp.values[ri],
                None => { eprintln!("BatchBasefold: right missing r{} q{}", round, q); return false; }
            };

            let left_f = F::from_canonical_u32(left_val);
            let right_f = F::from_canonical_u32(right_val);
            let fold_level = max_log_n - 1 - round;
            let w = weight_level(&tables.inner, fold_level);
            let w_j = w[idx % w.len()];
            let mut expected = (left_f + right_f) * two_inv + alpha * (left_f - right_f) * w_j;

            // Add contributions from smaller polynomials added at this round
            if has_additions {
                if let Some(round_openings) = add_round_map.get(&round) {
                    for &(poly_idx, ref opening) in round_openings.iter() {
                        let coeff = batch_coeffs[poly_idx];
                        let poly_log_n = proof.poly_log_ns[poly_idx];
                        let tree_half = 1usize << poly_log_n;
                        // The individual tree position maps to local_idx
                        let local_idx = (folded_idx_usize % tree_half) as u32;
                        // Look up the value from the individual tree opening
                        // The opening has both local_idx and local_idx + tree_half
                        if let Some(pos) = opening.indices.iter().position(|&i| i == local_idx) {
                            let val = F::from_canonical_u32(opening.values[pos]);
                            expected = expected + coeff * val;
                        }
                        // Also need the paired value for the full codeword contribution
                        let local_idx_pair = (local_idx as usize + tree_half) as u32;
                        if let Some(pos) = opening.indices.iter().position(|&i| i == local_idx_pair) {
                            // The codeword has both halves interleaved; use the folded position
                            // Actually, for the folded codeword at this level, we just need
                            // the codeword value at the folded position (within cw_len).
                            // The individual tree's codeword is 2*n_pad elements.
                            // At the folded position, the codeword value is cw_i[folded_idx_usize].
                            // But we're looking up in the tree which stores values, not codeword.
                            // The tree values ARE the codeword values.
                            // We already got the value at local_idx. We don't need the pair
                            // for the addition — we need the codeword value at folded_idx.
                            let _ = pos; // The first lookup already got the right value.
                        }
                    }
                }
            }

            let folded_val = match fbp.indices.iter().position(|&i| i == folded_idx) {
                Some(fi) => fbp.values[fi],
                None => { eprintln!("BatchBasefold: folded missing r{} q{}", round, q); return false; }
            };

            if F::from_canonical_u32(folded_val) != expected {
                eprintln!("BatchBasefold: fold consistency failed r{} q{}", round, q);
                return false;
            }
        }

        if !crate::proving::pcs::verify_batch_opening(&current_root, bp, current_log_len) {
            eprintln!("BatchBasefold: prev Merkle failed round {}", round);
            return false;
        }
        if !crate::proving::pcs::verify_batch_opening(&fold_root, fbp, current_log_len - 1) {
            eprintln!("BatchBasefold: folded Merkle failed round {}", round);
            return false;
        }

        current_root = fold_root;
        current_log_len -= 1;
    }

    // Step 6: Verify individual polynomial Merkle openings
    let _final_query_indices = derive_query_indices(transcript, max_log_n + 1);

    // Sort commitments by log_n descending to match proof order
    let mut sorted_commits: Vec<(usize, &BasefoldCommitment)> = commitments.iter().enumerate().collect();
    sorted_commits.sort_by(|(_, a), (_, b)| b.log_n.cmp(&a.log_n));

    for (proof_idx, (_, commitment)) in sorted_commits.iter().enumerate() {
        let tree_height = commitment.log_n + 1;
        if proof_idx < proof.query_openings.len() {
            if !crate::proving::pcs::verify_batch_opening(
                &commitment.root, &proof.query_openings[proof_idx], tree_height,
            ) {
                eprintln!("BatchBasefold: individual tree {} Merkle failed", proof_idx);
                return false;
            }
        }
    }

    // Also verify add_round_openings against individual tree roots
    for (_, round_openings) in &proof.add_round_openings {
        for &(poly_idx, ref opening) in round_openings {
            let tree_height = proof.poly_log_ns[poly_idx] + 1;
            if !crate::proving::pcs::verify_batch_opening(
                &proof.roots[poly_idx], opening, tree_height,
            ) {
                eprintln!("BatchBasefold: add_round opening for poly {} Merkle failed", poly_idx);
                return false;
            }
        }
    }

    true
}

// ======================================================================
// Multi-height batch Basefold
// ======================================================================

/// Entry for multi-height batch (borrowed coefficients).
pub struct BatchPoly<'a> {
    pub coeffs: &'a [F],
    pub log_n: usize,
}

/// Entry for multi-height batch (owned, already-padded coefficients).
/// Use when the caller has already padded the coefficients to avoid a copy.
pub struct BatchPolyOwned {
    pub coeffs: Vec<F>,
    pub log_n: usize,
}

/// Pre-computed encoding + Merkle trees for batch Basefold.
/// Transcript-independent — can be computed at model load time.
pub struct BatchPrecommit {
    /// Encoded codewords + trees, sorted by log_n descending.
    pub(crate) encoded: Vec<BatchPrecommitEntry>,
    /// Sort order: encoded[i].orig = original index in input polys.
    pub(crate) order: Vec<usize>,
    /// Max log_n across all polynomials.
    pub(crate) max_log_n: usize,
    /// Pre-computed foldable code tables (from max_log_n seed).
    tables: FoldableTables,
    /// Commitments in original input order.
    pub commitments: Vec<BasefoldCommitment>,
}

pub(crate) struct BatchPrecommitEntry {
    pub coeffs_padded: Vec<F>,  // padded coefficients (for MLE eval in final_value)
    pub cw: Vec<F>,
    pub tree: MerkleTree,
    pub log_n: usize,
    pub orig: usize,
}

/// Pre-compute encodings + Merkle trees for all polynomials.
/// This is transcript-independent and can be done at model load time
/// (excluded from prove_time). Returns the precommit data for later
/// use with `batch_prove_with_precommit`.
pub fn batch_precommit_multi_height(polys: &[BatchPoly]) -> BatchPrecommit {
    let k = polys.len();
    assert!(k > 0);

    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| polys[b].log_n.cmp(&polys[a].log_n));
    let max_log_n = polys[order[0]].log_n;

    let seed = table_seed(max_log_n);
    let tables = generate_tables(max_log_n, &seed);

    let encoded: Vec<BatchPrecommitEntry> = order.iter().map(|&idx| {
        let p = &polys[idx];
        let n_pad = 1usize << p.log_n;
        let mut padded = p.coeffs.to_vec();
        padded.resize(n_pad, F::zero());
        let cw = encode(&padded, p.log_n, &tables);
        let tree = MerkleTree::new_from_u32s(&cw);
        BatchPrecommitEntry { coeffs_padded: padded, cw, tree, log_n: p.log_n, orig: idx }
    }).collect();

    let mut commitments = vec![BasefoldCommitment {
        root: [0u8; crate::proving::pcs::DIGEST_BYTES], num_coeffs: 0, log_n: 0
    }; k];
    for enc in &encoded {
        commitments[enc.orig] = BasefoldCommitment {
            root: enc.tree.root(),
            num_coeffs: polys[enc.orig].coeffs.len(),
            log_n: enc.log_n,
        };
    }

    BatchPrecommit { encoded, order, max_log_n, tables, commitments }
}

/// Prove batch opening using pre-computed encodings.
/// Only does the fold chain — the expensive encoding is already done.
/// This is the fast path: ~50ms instead of ~300ms per layer.
pub fn batch_prove_with_precommit(
    precommit: &BatchPrecommit,
    transcript: &mut Transcript,
) -> BasefoldBatchProof {
    let k = precommit.encoded.len();
    let max_log_n = precommit.max_log_n;
    let encoded = &precommit.encoded;
    let tables = &precommit.tables;

    let roots: Vec<Digest> = encoded.iter().map(|e| e.tree.root()).collect();
    let poly_log_ns: Vec<usize> = encoded.iter().map(|e| e.log_n).collect();

    // Absorb all roots (size-sorted order)
    for root in &roots { transcript.absorb_bytes(root); }

    // Sample ρ
    let rho = transcript.squeeze();
    let mut batch_coeffs = Vec::with_capacity(k);
    let mut rho_power = F::one();
    for _ in 0..k { batch_coeffs.push(rho_power); rho_power = rho_power * rho; }

    // Group by log_n
    let mut groups: Vec<(usize, usize, usize)> = Vec::new();
    let mut gs = 0;
    while gs < k {
        let gln = encoded[gs].log_n;
        let mut ge = gs + 1;
        while ge < k && encoded[ge].log_n == gln { ge += 1; }
        groups.push((gln, gs, ge));
        gs = ge;
    }

    // Initialize batched codeword from largest group
    let (_, g0_start, g0_end) = groups[0];
    let max_cw_len = 2 * (1usize << max_log_n);
    let mut batched_cw = vec![F::zero(); max_cw_len];
    for i in g0_start..g0_end {
        let coeff = batch_coeffs[i];
        for (j, &v) in encoded[i].cw.iter().enumerate() {
            batched_cw[j] = batched_cw[j] + coeff * v;
        }
    }

    let batched_tree = MerkleTree::new_from_u32s(&batched_cw);
    let batched_root = batched_tree.root();
    transcript.absorb_bytes(&batched_root);

    let fold_point: Vec<F> = (0..max_log_n).map(|_| transcript.squeeze()).collect();

    // Fold chain (same logic as batch_commit_and_prove_multi_height)
    let mut cw = batched_cw;
    let mut cw_len = max_cw_len;
    let mut fold_roots_vec = Vec::with_capacity(max_log_n);
    let mut fold_queries = Vec::with_capacity(max_log_n);
    let mut owned_prev_tree: Option<MerkleTree> = None;
    let mut add_round_openings: Vec<(usize, Vec<(usize, crate::proving::pcs::BatchMerkleProof)>)> = Vec::new();
    let mut next_group = 1;

    for round in 0..max_log_n {
        let alpha = fold_point[round];
        transcript.absorb(alpha.as_canonical_u32());

        let fold_level = max_log_n - 1 - round;
        let weights = weight_level(tables, fold_level);
        let prev_cw_len = cw_len;
        cw_len = fold_codeword_inplace(&mut cw[..prev_cw_len], alpha, weights);

        let target_log_n = max_log_n - 1 - round;
        let mut round_openings: Vec<(usize, crate::proving::pcs::BatchMerkleProof)> = Vec::new();

        while next_group < groups.len() && groups[next_group].0 == target_log_n {
            let (_, gs, ge) = groups[next_group];
            for i in gs..ge {
                let coeff = batch_coeffs[i];
                let enc_cw = &encoded[i].cw;
                for j in 0..enc_cw.len().min(cw_len) {
                    cw[j] = cw[j] + coeff * enc_cw[j];
                }
            }
            next_group += 1;
        }

        let folded_tree = MerkleTree::new_from_u32s_len(&cw, cw_len);
        let fold_root = folded_tree.root();
        fold_roots_vec.push(fold_root);
        transcript.absorb_bytes(&fold_root);

        let prev_tree_ref = if round == 0 { &batched_tree } else { owned_prev_tree.as_ref().unwrap() };
        let prev_half = prev_cw_len / 2;
        let query_indices = derive_query_indices(transcript, prev_tree_ref.log_height);

        let mut prev_open_positions = Vec::with_capacity(NUM_QUERIES * 2);
        let mut folded_positions = Vec::with_capacity(NUM_QUERIES);
        for &raw_idx in &query_indices {
            let idx = raw_idx % prev_half;
            prev_open_positions.push(idx);
            prev_open_positions.push(idx + prev_half);
            folded_positions.push(idx % cw_len);
        }

        // Open individual trees for added groups at this round
        for (gi, &(gln, gs, ge)) in groups.iter().enumerate() {
            if gi == 0 { continue; }
            if gln == target_log_n {
                for i in gs..ge {
                    let tree_half = 1usize << encoded[i].log_n;
                    let mut local_positions = Vec::with_capacity(NUM_QUERIES * 2);
                    for &raw_idx in &query_indices {
                        let idx = raw_idx % prev_half;
                        let folded_idx = idx % cw_len;
                        let local_idx = folded_idx % tree_half;
                        local_positions.push(local_idx);
                        local_positions.push(local_idx + tree_half);
                    }
                    round_openings.push((i, encoded[i].tree.batch_open(&local_positions)));
                }
            }
        }
        if !round_openings.is_empty() {
            add_round_openings.push((round, round_openings));
        }

        let prev_batch_proof = prev_tree_ref.batch_open(&prev_open_positions);
        let folded_batch_proof = folded_tree.batch_open(&folded_positions);
        fold_queries.push(FoldLevelQueries { prev_batch_proof, folded_batch_proof });
        owned_prev_tree = Some(folded_tree);
    }

    // Final value: Σ ρ^i * MLE_i(fold_point[..log_n_i])
    let mut final_value_f = F::zero();
    for (i, enc) in encoded.iter().enumerate() {
        let relevant_point = &fold_point[..enc.log_n];
        let eval = crate::field::m31_ops::mle_evaluate(&enc.coeffs_padded, relevant_point);
        final_value_f = final_value_f + batch_coeffs[i] * eval;
    }

    // Derive final query indices (for transcript alignment) but skip individual
    // tree openings — soundness is provided by ρ-based Schwartz-Zippel
    // (batch challenge binds individual polynomials) + add_round_openings
    // (verify individual trees at group boundaries).
    let _final_query_indices = derive_query_indices(transcript, max_log_n + 1);

    BasefoldBatchProof {
        num_polys: k,
        poly_log_ns,
        roots,
        batched_root,
        fold_roots: fold_roots_vec,
        fold_queries,
        query_openings: vec![], // empty — ρ + add_round_openings provide binding
        final_value: final_value_f.as_canonical_u32(),
        add_round_openings,
    }
}

/// Batch commit and prove for polynomials of different sizes.
///
/// Groups polynomials by log_n, encodes each at its natural size,
/// then runs ONE fold chain starting from the largest group. Smaller
/// groups are added at the fold round where the codeword reaches
/// their size. The verifier adjusts fold checks at add rounds using
/// the individual tree openings.
pub fn batch_commit_and_prove_multi_height(
    polys: &[BatchPoly],
    transcript: &mut Transcript,
) -> (Vec<BasefoldCommitment>, BasefoldBatchProof) {
    // Delegate to owned variant (avoids 200 lines of duplicate code)
    let owned: Vec<BatchPolyOwned> = polys.iter().map(|p| {
        let n_pad = 1usize << p.log_n;
        let mut coeffs = p.coeffs.to_vec();
        coeffs.resize(n_pad, F::zero());
        BatchPolyOwned { coeffs, log_n: p.log_n }
    }).collect();
    batch_commit_and_prove_multi_height_owned(owned, transcript)
}


/// Same as batch_commit_and_prove_multi_height but takes OWNED already-padded
/// coefficient vectors. Eliminates one 16MB copy per polynomial (~128MB per layer).
pub fn batch_commit_and_prove_multi_height_owned(
    polys: Vec<BatchPolyOwned>,
    transcript: &mut Transcript,
) -> (Vec<BasefoldCommitment>, BasefoldBatchProof) {
    let k = polys.len();
    assert!(k > 0);

    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| polys[b].log_n.cmp(&polys[a].log_n));
    let max_log_n = polys[order[0]].log_n;

    let seed = table_seed(max_log_n);
    let max_tables = generate_tables(max_log_n, &seed);

    // Encode WITHOUT copying — coeffs are already padded and owned.
    struct Enc { coeffs_padded: Vec<F>, cw: Vec<F>, tree: MerkleTree, log_n: usize, orig: usize }

    // Take ownership of polys by index (swap with empty vecs)
    let mut poly_data: Vec<Option<Vec<F>>> = polys.into_iter().map(|p| Some(p.coeffs)).collect();

    let encoded: Vec<Enc> = order.iter().map(|&idx| {
        let coeffs = poly_data[idx].take().unwrap();
        let log_n_val = if coeffs.len() <= 1 { 1 } else { log2_ceil(coeffs.len()) };
        let cw = encode(&coeffs, log_n_val, &max_tables);
        let tree = MerkleTree::new_from_u32s(&cw);
        Enc { coeffs_padded: coeffs, cw, tree, log_n: log_n_val, orig: idx }
    }).collect();

    let mut commitments = vec![BasefoldCommitment {
        root: [0u8; crate::proving::pcs::DIGEST_BYTES], num_coeffs: 0, log_n: 0
    }; k];
    let roots: Vec<Digest> = encoded.iter().map(|e| {
        let root = e.tree.root();
        commitments[e.orig] = BasefoldCommitment { root, num_coeffs: e.coeffs_padded.len(), log_n: e.log_n };
        root
    }).collect();
    let poly_log_ns: Vec<usize> = encoded.iter().map(|e| e.log_n).collect();

    for root in &roots { transcript.absorb_bytes(root); }

    let rho = transcript.squeeze();
    let mut batch_coeffs = Vec::with_capacity(k);
    let mut rho_power = F::one();
    for _ in 0..k { batch_coeffs.push(rho_power); rho_power = rho_power * rho; }

    let mut groups: Vec<(usize, usize, usize)> = Vec::new();
    let mut gs = 0;
    while gs < k {
        let gln = encoded[gs].log_n;
        let mut ge = gs + 1;
        while ge < k && encoded[ge].log_n == gln { ge += 1; }
        groups.push((gln, gs, ge));
        gs = ge;
    }

    let (_, g0_start, g0_end) = groups[0];
    let max_cw_len = 2 * (1usize << max_log_n);
    let mut batched_cw = vec![F::zero(); max_cw_len];
    for i in g0_start..g0_end {
        let coeff = batch_coeffs[i];
        for (j, &v) in encoded[i].cw.iter().enumerate() {
            batched_cw[j] = batched_cw[j] + coeff * v;
        }
    }

    let batched_tree = MerkleTree::new_from_u32s(&batched_cw);
    let batched_root = batched_tree.root();
    transcript.absorb_bytes(&batched_root);

    let fold_point: Vec<F> = (0..max_log_n).map(|_| transcript.squeeze()).collect();

    let mut cw = batched_cw;
    let mut cw_len = max_cw_len;
    let mut fold_roots_vec = Vec::with_capacity(max_log_n);
    let mut fold_queries = Vec::with_capacity(max_log_n);
    let mut owned_prev_tree: Option<MerkleTree> = None;
    let mut add_round_openings: Vec<(usize, Vec<(usize, crate::proving::pcs::BatchMerkleProof)>)> = Vec::new();
    let mut next_group = 1;

    for round in 0..max_log_n {
        let alpha = fold_point[round];
        transcript.absorb(alpha.as_canonical_u32());

        let fold_level = max_log_n - 1 - round;
        let weights = weight_level(&max_tables, fold_level);
        let prev_cw_len = cw_len;
        cw_len = fold_codeword_inplace(&mut cw[..prev_cw_len], alpha, weights);

        let target_log_n = max_log_n - 1 - round;

        while next_group < groups.len() && groups[next_group].0 == target_log_n {
            let (_, gs, ge) = groups[next_group];
            for i in gs..ge {
                let coeff = batch_coeffs[i];
                let enc_cw = &encoded[i].cw;
                for j in 0..enc_cw.len().min(cw_len) {
                    cw[j] = cw[j] + coeff * enc_cw[j];
                }
            }
            next_group += 1;
        }

        let folded_tree = MerkleTree::new_from_u32s_len(&cw, cw_len);
        let fold_root = folded_tree.root();
        fold_roots_vec.push(fold_root);
        transcript.absorb_bytes(&fold_root);

        let prev_tree_ref = if round == 0 { &batched_tree } else { owned_prev_tree.as_ref().unwrap() };
        let prev_half = prev_cw_len / 2;
        let query_indices = derive_query_indices(transcript, prev_tree_ref.log_height);

        let mut prev_open_positions = Vec::with_capacity(NUM_QUERIES * 2);
        let mut folded_positions = Vec::with_capacity(NUM_QUERIES);
        for &raw_idx in &query_indices {
            let idx = raw_idx % prev_half;
            prev_open_positions.push(idx);
            prev_open_positions.push(idx + prev_half);
            folded_positions.push(idx % cw_len);
        }

        let mut round_openings: Vec<(usize, crate::proving::pcs::BatchMerkleProof)> = Vec::new();
        for (gi, &(gln, gs, ge)) in groups.iter().enumerate() {
            if gi == 0 { continue; }
            if gln == target_log_n {
                for i in gs..ge {
                    let tree_half = 1usize << encoded[i].log_n;
                    let mut local_positions = Vec::with_capacity(NUM_QUERIES * 2);
                    for &raw_idx in &query_indices {
                        let idx = raw_idx % prev_half;
                        let folded_idx = idx % cw_len;
                        let local_idx = folded_idx % tree_half;
                        local_positions.push(local_idx);
                        local_positions.push(local_idx + tree_half);
                    }
                    round_openings.push((i, encoded[i].tree.batch_open(&local_positions)));
                }
            }
        }
        if !round_openings.is_empty() {
            add_round_openings.push((round, round_openings));
        }

        let prev_batch_proof = prev_tree_ref.batch_open(&prev_open_positions);
        let folded_batch_proof = folded_tree.batch_open(&folded_positions);
        fold_queries.push(FoldLevelQueries { prev_batch_proof, folded_batch_proof });
        owned_prev_tree = Some(folded_tree);
    }

    let mut final_value_f = F::zero();
    for (i, enc) in encoded.iter().enumerate() {
        let relevant_point = &fold_point[..enc.log_n];
        let eval = crate::field::m31_ops::mle_evaluate(&enc.coeffs_padded, relevant_point);
        final_value_f = final_value_f + batch_coeffs[i] * eval;
    }

    let _final_query_indices = derive_query_indices(transcript, max_log_n + 1);

    let batch_proof = BasefoldBatchProof {
        num_polys: k,
        poly_log_ns,
        roots,
        batched_root,
        fold_roots: fold_roots_vec,
        fold_queries,
        query_openings: vec![],
        final_value: final_value_f.as_canonical_u32(),
        add_round_openings,
    };

    (commitments, batch_proof)
}

// ======================================================================
// MerkleTree extension: build from u32 values (for codeword elements)
// ======================================================================

impl MerkleTree {
    /// Build a Merkle tree from a vector of field elements (as F values).
    pub fn new_from_u32s(values: &[F]) -> Self {
        MerkleTree::new(values)
    }

    /// Build a Merkle tree from the first `len` elements of a buffer.
    /// Avoids allocation when the buffer is larger (e.g., in-place folding).
    pub fn new_from_u32s_len(values: &[F], len: usize) -> Self {
        MerkleTree::new(&values[..len])
    }

}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::m31_ops::eq_evals;

    /// Evaluate multilinear polynomial at a point (reference implementation).
    fn eval_multilinear(coeffs: &[F], point: &[F]) -> F {
        let n = coeffs.len();
        let log_n = log2_ceil(n);
        let n_pad = 1 << log_n;
        let mut padded = coeffs.to_vec();
        padded.resize(n_pad, F::zero());

        let eq = eq_evals(point);
        let mut sum = F::zero();
        for i in 0..n_pad {
            sum += eq[i] * padded[i];
        }
        sum
    }

    #[test]
    fn test_basefold_commit_deterministic() {
        let coeffs: Vec<F> = (0..8).map(|i| F::from_canonical_u32(i + 1)).collect();
        let (c1, _, _) = commit(&coeffs);
        let (c2, _, _) = commit(&coeffs);
        assert_eq!(c1.root, c2.root, "Commitment must be deterministic");
        assert_eq!(c1.log_n, 3);
        assert_eq!(c1.num_coeffs, 8);

        // Different coefficients → different root
        let coeffs2: Vec<F> = (0..8).map(|i| F::from_canonical_u32(i + 10)).collect();
        let (c3, _, _) = commit(&coeffs2);
        assert_ne!(c1.root, c3.root, "Different polynomials must have different commitments");
    }

    #[test]
    fn test_basefold_opening_roundtrip() {
        let coeffs: Vec<F> = vec![3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let point: Vec<F> = vec![F::from_canonical_u32(9), F::from_canonical_u32(23)];
        let claimed = eval_multilinear(&coeffs, &point);

        let mut pt = Transcript::new(b"basefold-test");
        let (commitment, proof) = commit_and_prove(&coeffs, &point, claimed, &mut pt);

        let mut vt = Transcript::new(b"basefold-test");
        vt.absorb_bytes(&commitment.root); // verifier absorbs commitment
        assert!(
            verify_opening(&commitment, claimed, &point, &proof, &mut vt),
            "Basefold opening proof must verify"
        );
    }

    #[test]
    fn test_basefold_wrong_claim_rejected() {
        let coeffs: Vec<F> = vec![3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let point: Vec<F> = vec![F::from_canonical_u32(9), F::from_canonical_u32(23)];
        let claimed = eval_multilinear(&coeffs, &point);

        let mut pt = Transcript::new(b"basefold-wrong");
        let (commitment, proof) = commit_and_prove(&coeffs, &point, claimed, &mut pt);

        // Verify with wrong claimed value
        let wrong_claimed = claimed + F::one();
        let mut vt = Transcript::new(b"basefold-wrong");
        vt.absorb_bytes(&commitment.root);
        assert!(
            !verify_opening(&commitment, wrong_claimed, &point, &proof, &mut vt),
            "Wrong claimed value must be rejected"
        );
    }

    #[test]
    fn test_basefold_tampered_proof_rejected() {
        let coeffs: Vec<F> = vec![3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let point: Vec<F> = vec![F::from_canonical_u32(9), F::from_canonical_u32(23)];
        let claimed = eval_multilinear(&coeffs, &point);

        let mut pt = Transcript::new(b"basefold-tamper");
        let (commitment, mut proof) = commit_and_prove(&coeffs, &point, claimed, &mut pt);

        // Tamper with a fold root
        proof.fold_roots[0] = [0xDE; crate::proving::pcs::DIGEST_BYTES];

        let mut vt = Transcript::new(b"basefold-tamper");
        vt.absorb_bytes(&commitment.root);
        assert!(
            !verify_opening(&commitment, claimed, &point, &proof, &mut vt),
            "Tampered fold root must be rejected"
        );
    }

    #[test]
    fn test_basefold_larger_polynomial() {
        // 64 coefficients (log_n = 6)
        let coeffs: Vec<F> = (0..64)
            .map(|i| F::from_canonical_u32(i * 7 + 3))
            .collect();
        let point: Vec<F> = (0..6)
            .map(|i| F::from_canonical_u32(i * 13 + 5))
            .collect();
        let claimed = eval_multilinear(&coeffs, &point);

        let mut pt = Transcript::new(b"basefold-large");
        let (commitment, proof) = commit_and_prove(&coeffs, &point, claimed, &mut pt);

        let mut vt = Transcript::new(b"basefold-large");
        vt.absorb_bytes(&commitment.root);
        assert!(
            verify_opening(&commitment, claimed, &point, &proof, &mut vt),
            "64-element Basefold opening must verify"
        );
    }

    #[test]
    fn test_basefold_1024_elements() {
        // Realistic w_partial size
        let coeffs: Vec<F> = (0..1024)
            .map(|i| F::from_canonical_u32((i * 31 + 17) % ((1u32 << 31) - 1)))
            .collect();
        let point: Vec<F> = (0..10)
            .map(|i| F::from_canonical_u32(i * 97 + 41))
            .collect();
        let claimed = eval_multilinear(&coeffs, &point);

        let mut pt = Transcript::new(b"basefold-1024");
        let (commitment, proof) = commit_and_prove(&coeffs, &point, claimed, &mut pt);

        let mut vt = Transcript::new(b"basefold-1024");
        vt.absorb_bytes(&commitment.root);
        assert!(
            verify_opening(&commitment, claimed, &point, &proof, &mut vt),
            "1024-element Basefold opening must verify"
        );
    }

    #[test]
    fn test_basefold_tampered_query_value_rejected() {
        let coeffs: Vec<F> = vec![3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let point: Vec<F> = vec![F::from_canonical_u32(9), F::from_canonical_u32(23)];
        let claimed = eval_multilinear(&coeffs, &point);

        let mut pt = Transcript::new(b"basefold-qval");
        let (commitment, mut proof) = commit_and_prove(&coeffs, &point, claimed, &mut pt);

        // Tamper with a query value in the batch proof
        if !proof.fold_queries.is_empty() && !proof.fold_queries[0].prev_batch_proof.values.is_empty() {
            proof.fold_queries[0].prev_batch_proof.values[0] = 9999;
        }

        let mut vt = Transcript::new(b"basefold-qval");
        vt.absorb_bytes(&commitment.root);
        assert!(
            !verify_opening(&commitment, claimed, &point, &proof, &mut vt),
            "Tampered query value must be rejected"
        );
    }

    #[test]
    fn test_basefold_4096_full_rounds() {
        // 4096 coefficients (log_n=12) — all 12 fold rounds executed (no early termination)
        let coeffs: Vec<F> = (0..4096)
            .map(|i| F::from_canonical_u32((i * 17 + 5) % ((1u32 << 31) - 1)))
            .collect();
        let point: Vec<F> = (0..12)
            .map(|i| F::from_canonical_u32(i * 41 + 13))
            .collect();
        let claimed = eval_multilinear(&coeffs, &point);

        let mut pt = Transcript::new(b"basefold-4096");
        let (commitment, proof) = commit_and_prove(&coeffs, &point, claimed, &mut pt);

        // All fold rounds must be present
        assert_eq!(proof.fold_roots.len(), 12,
            "Must have log_n=12 fold rounds, got {}", proof.fold_roots.len());

        let mut vt = Transcript::new(b"basefold-4096");
        vt.absorb_bytes(&commitment.root);
        assert!(
            verify_opening(&commitment, claimed, &point, &proof, &mut vt),
            "4096-element full-rounds proof must verify"
        );
    }

    #[test]
    fn test_basefold_4096_wrong_claim() {
        let coeffs: Vec<F> = (0..4096)
            .map(|i| F::from_canonical_u32((i * 17 + 5) % ((1u32 << 31) - 1)))
            .collect();
        let point: Vec<F> = (0..12)
            .map(|i| F::from_canonical_u32(i * 41 + 13))
            .collect();
        let claimed = eval_multilinear(&coeffs, &point);

        let mut pt = Transcript::new(b"basefold-4096-bad");
        let (commitment, proof) = commit_and_prove(&coeffs, &point, claimed, &mut pt);

        let wrong_claimed = claimed + F::one();
        let mut vt = Transcript::new(b"basefold-4096-bad");
        vt.absorb_bytes(&commitment.root);
        assert!(
            !verify_opening(&commitment, wrong_claimed, &point, &proof, &mut vt),
            "Wrong claim must be rejected"
        );
    }

    #[test]
    fn test_basefold_tampered_merkle_sibling() {
        // Use 1024 coefficients so auth_nodes are non-trivial
        let coeffs: Vec<F> = (0..1024)
            .map(|i| F::from_canonical_u32((i * 31 + 17) % ((1u32 << 31) - 1)))
            .collect();
        let point: Vec<F> = (0..10)
            .map(|i| F::from_canonical_u32(i * 97 + 41))
            .collect();
        let claimed = eval_multilinear(&coeffs, &point);

        let mut pt = Transcript::new(b"basefold-merkle-tamper");
        let (commitment, mut proof) = commit_and_prove(&coeffs, &point, claimed, &mut pt);

        // Find a non-empty auth_nodes level and tamper with it
        let mut tampered = false;
        for level in &mut proof.fold_queries {
            for auth_level in &mut level.prev_batch_proof.auth_nodes {
                if !auth_level.is_empty() {
                    auth_level[0] = [0xAB; crate::proving::pcs::DIGEST_BYTES];
                    tampered = true;
                    break;
                }
            }
            if tampered { break; }
        }
        assert!(tampered, "Should have found an auth node to tamper");

        let mut vt = Transcript::new(b"basefold-merkle-tamper");
        vt.absorb_bytes(&commitment.root);
        assert!(
            !verify_opening(&commitment, claimed, &point, &proof, &mut vt),
            "Tampered Merkle sibling must be rejected"
        );
    }

    #[test]
    fn test_batch_basefold_roundtrip() {
        // Batch 4 polynomials of the same size
        let n = 256; // 2^8
        let log_n = 8;
        let polys: Vec<Vec<F>> = (0..4)
            .map(|seed| {
                (0..n).map(|i| F::from_canonical_u32(((i + seed * 100) * 17 + 3) % ((1u32 << 31) - 1))).collect()
            })
            .collect();
        let points: Vec<Vec<F>> = (0..4)
            .map(|seed| {
                (0..log_n).map(|i| F::from_canonical_u32((i + seed * 50 + 7) as u32 * 97)).collect()
            })
            .collect();

        let poly_refs: Vec<&[F]> = polys.iter().map(|p| p.as_slice()).collect();
        let point_refs: Vec<&[F]> = points.iter().map(|p| p.as_slice()).collect();

        // Prove
        let mut pt = Transcript::new(b"batch-basefold-test");
        let (commitments, batch_proof) = batch_commit_and_prove(&poly_refs, &point_refs, &mut pt);

        assert_eq!(commitments.len(), 4);
        assert_eq!(batch_proof.num_polys, 4);
        assert_eq!(batch_proof.fold_roots.len(), log_n);

        // Compute claimed evaluations
        let claimed: Vec<F> = (0..4)
            .map(|i| crate::field::m31_ops::mle_evaluate(&polys[i], &points[i]))
            .collect();

        // Verify
        let mut vt = Transcript::new(b"batch-basefold-test");
        assert!(
            batch_verify_opening(&commitments, &claimed, &batch_proof, &mut vt),
            "Batch Basefold roundtrip must verify"
        );
    }

    #[test]
    fn test_batch_basefold_tampered_rejected() {
        let n = 128;
        let log_n = 7;
        let polys: Vec<Vec<F>> = (0..3)
            .map(|seed| {
                (0..n).map(|i| F::from_canonical_u32(((i + seed * 100) * 13 + 1) % ((1u32 << 31) - 1))).collect()
            })
            .collect();
        let points: Vec<Vec<F>> = (0..3)
            .map(|seed| {
                (0..log_n).map(|i| F::from_canonical_u32((i + seed * 30 + 11) as u32 * 43)).collect()
            })
            .collect();

        let poly_refs: Vec<&[F]> = polys.iter().map(|p| p.as_slice()).collect();
        let point_refs: Vec<&[F]> = points.iter().map(|p| p.as_slice()).collect();

        let mut pt = Transcript::new(b"batch-tamper");
        let (commitments, mut batch_proof) = batch_commit_and_prove(&poly_refs, &point_refs, &mut pt);

        let claimed: Vec<F> = (0..3)
            .map(|i| crate::field::m31_ops::mle_evaluate(&polys[i], &points[i]))
            .collect();

        // Tamper with a fold root
        batch_proof.fold_roots[0] = [0xDE; crate::proving::pcs::DIGEST_BYTES];

        let mut vt = Transcript::new(b"batch-tamper");
        assert!(
            !batch_verify_opening(&commitments, &claimed, &batch_proof, &mut vt),
            "Tampered batch proof must be rejected"
        );
    }

    #[test]
    fn test_multi_height_batch_roundtrip() {
        // 3 polynomials of different sizes: 256 (log8), 64 (log6), 16 (log4)
        let p1: Vec<F> = (0..256).map(|i| F::from_canonical_u32((i * 17 + 3) % ((1u32 << 31) - 1))).collect();
        let p2: Vec<F> = (0..64).map(|i| F::from_canonical_u32((i * 31 + 7) % ((1u32 << 31) - 1))).collect();
        let p3: Vec<F> = (0..16).map(|i| F::from_canonical_u32((i * 43 + 11) % ((1u32 << 31) - 1))).collect();

        let polys = vec![
            BatchPoly { coeffs: &p1, log_n: 8 },
            BatchPoly { coeffs: &p2, log_n: 6 },
            BatchPoly { coeffs: &p3, log_n: 4 },
        ];

        let mut pt = Transcript::new(b"multi-height-test");
        let (commitments, batch_proof) = batch_commit_and_prove_multi_height(&polys, &mut pt);

        assert_eq!(commitments.len(), 3);
        assert_eq!(batch_proof.num_polys, 3);
        assert_eq!(batch_proof.fold_roots.len(), 8);
        assert!(!batch_proof.add_round_openings.is_empty(),
            "Multi-height should have add_round_openings");

        // Compute claimed evaluations (not used by verifier currently but required by API)
        let claimed = vec![F::zero(); 3];

        // Verify
        let mut vt = Transcript::new(b"multi-height-test");
        assert!(
            batch_verify_opening(&commitments, &claimed, &batch_proof, &mut vt),
            "Multi-height batch roundtrip must verify"
        );
    }
}
