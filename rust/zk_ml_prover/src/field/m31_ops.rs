//! Multilinear extension operations over M31.

use p3_field::{AbstractField, AbstractExtensionField, PrimeField32};
use p3_mersenne_31::Mersenne31;

type F = Mersenne31;

/// Evaluate the MLE of `evals` at `point`.
/// evals has 2^k entries, point has k coordinates.
/// Uses stride-based MSB-first folding (matches Python implementation).
pub fn mle_evaluate(evals: &[F], point: &[F]) -> F {
    let mut table = evals.to_vec();
    let num_vars = point.len();
    let mut size = table.len();

    for i in 0..num_vars {
        let half = size / 2;
        let r = point[i];
        let one_minus_r = F::one() - r;
        for j in 0..half {
            table[j] = one_minus_r * table[j] + r * table[j + half];
        }
        size = half;
    }
    table[0]
}

/// Compute eq(r, x) for all x in {0,1}^k.
/// eq(r, x) = prod_i (r_i * x_i + (1 - r_i) * (1 - x_i))
pub fn eq_evals(r: &[F]) -> Vec<F> {
    let k = r.len();
    let n = 1 << k;
    let mut evals = vec![F::one(); n];

    // Build eq(r, x) via butterfly: each round i doubles the number of
    // populated entries by splitting on r[i].
    // After round i, entries 0..2^(i+1) are populated.
    let mut populated = 1usize; // number of populated entries
    for i in 0..k {
        let ri = r[i];
        let one_minus_ri = F::one() - ri;
        // Process in reverse to avoid overwriting
        for j in (0..populated).rev() {
            evals[2 * j + 1] = evals[j] * ri;
            evals[2 * j] = evals[j] * one_minus_ri;
        }
        populated *= 2;
    }
    evals
}

/// GPU-accelerated eq_evals using Metal butterfly kernel.
#[cfg(feature = "metal_gpu")]
pub fn eq_evals_gpu(r: &[F]) -> Vec<F> {
    use crate::gpu::{MetalContext, GpuBuffer, MetalKernels, GPU_THRESHOLD};

    let k = r.len();
    let n = 1 << k;

    if n < GPU_THRESHOLD {
        return eq_evals(r);
    }

    let ctx = MetalContext::get();
    let kernels = MetalKernels::new();

    let mut init = vec![F::one()];
    init.resize(n, F::zero());
    let mut buf = GpuBuffer::from_field_slice(&ctx.device, &init);

    let mut populated = 1usize;
    for i in 0..k {
        kernels.eq_butterfly(&mut buf, populated, r[i]);
        populated *= 2;
    }

    buf.to_field_vec()
}

/// Inner product of two vectors in M31.
#[allow(dead_code)]
pub fn inner_product(a: &[F], b: &[F]) -> F {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Fold the first (MSB) variable of a table to `val`.
/// Input: 2^v entries. Output: 2^(v-1) entries.
#[allow(dead_code)]
pub fn fold_first_var(evals: &[F], val: F) -> Vec<F> {
    let half = evals.len() / 2;
    let one_minus_val = F::one() - val;
    (0..half)
        .map(|j| one_minus_val * evals[j] + val * evals[j + half])
        .collect()
}

/// Convert a signed integer to M31 field element.
pub fn to_field(x: i64) -> F {
    if x >= 0 {
        F::from_canonical_u32((x as u64 % (M31)) as u32)
    } else {
        let pos = ((-x) as u64) % M31;
        F::zero() - F::from_canonical_u32(pos as u32)
    }
}

/// Convert M31 field element to signed integer (centered representation).
pub fn from_field(x: F) -> i64 {
    let v = x.as_canonical_u32();
    let half = (M31 / 2) as u32;
    if v > half {
        v as i64 - M31 as i64
    } else {
        v as i64
    }
}

/// M31 prime: 2^31 - 1.
pub const M31: u64 = (1u64 << 31) - 1;

/// Reduce a u64 value modulo M31 (2^31 - 1).
/// Input can be up to ~2^63 safely.
#[allow(dead_code)]
#[inline(always)]
pub fn reduce_u64_m31(v: u64) -> F {
    const P: u64 = (1u64 << 31) - 1;
    // Split into 31-bit chunks: v = v2*2^62 + v1*2^31 + v0
    // 2^31 ≡ 1 (mod P), so 2^62 ≡ 1 (mod P)
    // Result = v0 + v1 + v2 (mod P)
    let v0 = (v & P) as u32;
    let v1 = ((v >> 31) & P) as u32;
    let v2 = (v >> 62) as u32;
    let sum = v0 as u64 + v1 as u64 + v2 as u64;
    // sum < 3 × 2^31, one more reduction
    let lo = (sum & P) as u32;
    let hi = (sum >> 31) as u32;
    F::from_canonical_u32(if lo + hi >= P as u32 { lo + hi - P as u32 } else { lo + hi })
}

// =============================================================================
// Extension-field MLE operations (base-field polynomials at EF points)
// =============================================================================

use p3_field::extension::Complex;
use crate::proving::sumcheck::EF;

/// Embed base field element into EF.
#[inline(always)]
pub fn f_to_ef(v: F) -> EF {
    EF::from_base(Complex::new(v, F::zero()))
}

/// Evaluate the MLE of base-field `evals` at an extension-field `point`.
/// Same folding algorithm as `mle_evaluate`, but mixing F coefficients with EF challenges.
pub fn mle_evaluate_ef(evals: &[F], point: &[EF]) -> EF {
    let num_vars = point.len();
    let mut size = evals.len();
    // After first fold, table becomes EF
    let mut table: Vec<EF> = evals.iter().map(|&v| f_to_ef(v)).collect();

    for i in 0..num_vars {
        let half = size / 2;
        let r = point[i];
        let one_minus_r = EF::one() - r;
        for j in 0..half {
            table[j] = one_minus_r * table[j] + r * table[j + half];
        }
        size = half;
    }
    table[0]
}

/// Compute eq(r, x) for all x in {0,1}^k where r is an EF point.
/// eq(r, x) = prod_i (r_i * x_i + (1 - r_i) * (1 - x_i))
pub fn eq_evals_ef(r: &[EF]) -> Vec<EF> {
    let k = r.len();
    let n = 1 << k;
    let mut evals = vec![EF::one(); n];

    let mut populated = 1usize;
    for i in 0..k {
        let ri = r[i];
        let one_minus_ri = EF::one() - ri;
        for j in (0..populated).rev() {
            evals[2 * j + 1] = evals[j] * ri;
            evals[2 * j] = evals[j] * one_minus_ri;
        }
        populated *= 2;
    }
    evals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mle_evaluate() {
        // f(x) = [3, 5, 7, 11] on {0,1}^2
        // f(0,0)=3, f(0,1)=5, f(1,0)=7, f(1,1)=11
        let evals: Vec<F> = [3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let point = vec![F::zero(), F::zero()];
        assert_eq!(mle_evaluate(&evals, &point), F::from_canonical_u32(3));

        let point2 = vec![F::one(), F::one()];
        assert_eq!(mle_evaluate(&evals, &point2), F::from_canonical_u32(11));
    }

    #[test]
    fn test_eq_evals() {
        let r = vec![F::from_canonical_u32(2), F::from_canonical_u32(3)];
        let eqs = eq_evals(&r);
        // MSB-first butterfly: index bits map as (r[0]->MSB, r[1]->LSB)
        // idx 0 = (0,0): (1-2)(1-3) = 2
        // idx 1 = (0,1): (1-2)(3) = -3
        // idx 2 = (1,0): (2)(1-3) = -4
        // idx 3 = (1,1): (2)(3) = 6
        // But butterfly processes MSB first, so actual layout:
        // Sum should equal product of (r_i + (1-r_i)) = 1 for each var → sum = 1
        assert_eq!(eqs.len(), 4);
        assert_eq!(eqs[0], to_field(2));
        assert_eq!(eqs[1], to_field(-3));
        assert_eq!(eqs[2], to_field(-4));
        assert_eq!(eqs[3], to_field(6));
    }

    #[test]
    fn test_to_from_field() {
        assert_eq!(from_field(to_field(42)), 42);
        assert_eq!(from_field(to_field(-42)), -42);
        assert_eq!(from_field(to_field(0)), 0);
    }
}
