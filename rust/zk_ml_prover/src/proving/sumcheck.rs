//! Product and triple sumcheck prover/verifier over M31.
//!
//! Product sumcheck: prove Σ f(x)·g(x) = c
//! Triple sumcheck: prove Σ f(x)·g(x)·h(x) = c

use p3_field::{AbstractField, AbstractExtensionField, Field, PackedValue, PrimeField32};
use p3_field::extension::{BinomialExtensionField, Complex};
use p3_mersenne_31::Mersenne31;
#[cfg(target_arch = "aarch64")]
use p3_mersenne_31::PackedMersenne31Neon;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::field::m31_ops::f_to_ef;

type F = Mersenne31;
/// Degree-4 extension field: M31 → Complex<M31> (x²+1) → EF (y²-(2+i)).
/// Provides 124-bit soundness for Fiat-Shamir challenges.
pub type EF = BinomialExtensionField<Complex<Mersenne31>, 2>;

/// Runtime toggle for GPU acceleration. Set via `--gpu` CLI flag.
#[cfg(feature = "metal_gpu")]
static GPU_ENABLED: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "metal_gpu")]
pub fn set_gpu_enabled(enabled: bool) {
    GPU_ENABLED.store(enabled, Ordering::Relaxed);
}

#[cfg(feature = "metal_gpu")]
pub fn is_gpu_enabled() -> bool {
    GPU_ENABLED.load(Ordering::Relaxed)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckProof {
    /// Round polynomials: for product sumcheck, each has 3 evaluations (degree 2);
    /// for triple sumcheck, each has 4 evaluations (degree 3).
    pub round_polys: Vec<Vec<u32>>,
    /// Challenges derived via Fiat-Shamir. May be empty (verifier re-derives).
    #[serde(default)]
    pub challenges: Vec<u32>,
}

/// Serializable representation of an extension field element (4 base-field u32s).
/// Layout: [re(c0), im(c0), re(c1), im(c1)] where EF = c0 + c1·Y.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EFElement(pub [u32; 4]);

impl EFElement {
    pub fn from_ef(v: EF) -> Self {
        let base_slice: &[Complex<Mersenne31>] = v.as_base_slice();
        let c0: &[Mersenne31] = base_slice[0].as_base_slice();
        let c1: &[Mersenne31] = base_slice[1].as_base_slice();
        EFElement([
            c0[0].as_canonical_u32(),
            c0[1].as_canonical_u32(),
            c1[0].as_canonical_u32(),
            c1[1].as_canonical_u32(),
        ])
    }

    pub fn to_ef(&self) -> EF {
        let c0 = Complex::new(
            F::from_canonical_u32(self.0[0]),
            F::from_canonical_u32(self.0[1]),
        );
        let c1 = Complex::new(
            F::from_canonical_u32(self.0[2]),
            F::from_canonical_u32(self.0[3]),
        );
        EF::from_base_slice(&[c0, c1])
    }
}

/// Sumcheck proof with extension-field challenges (124-bit soundness).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckProofEF {
    /// Round polynomials: evaluations are EF elements.
    /// Product: 3 EF evals per round; triple: 4 EF evals per round.
    pub round_polys: Vec<Vec<EFElement>>,
    /// Challenges in EF derived via Fiat-Shamir. May be empty (verifier re-derives).
    #[serde(default)]
    pub challenges: Vec<EFElement>,
}

/// Simple Fiat-Shamir transcript using SHA-256.
pub struct Transcript {
    hasher: Sha256,
}

impl Transcript {
    pub fn new(label: &[u8]) -> Self {
        Self {
            hasher: Sha256::new_with_prefix(label),
        }
    }

    pub fn absorb(&mut self, value: u32) {
        self.hasher.update(value.to_le_bytes());
    }

    pub fn absorb_many(&mut self, values: &[F]) {
        for v in values {
            self.absorb(v.as_canonical_u32());
        }
    }

    pub fn squeeze(&mut self) -> F {
        let digest = self.hasher.finalize_reset();
        let raw = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
        let challenge = raw % ((1u32 << 31) - 1);
        // Chain: re-seed with digest
        self.hasher = Sha256::new_with_prefix(&digest);
        F::from_canonical_u32(challenge)
    }

    pub fn squeeze_many(&mut self, n: usize) -> Vec<F> {
        (0..n).map(|_| self.squeeze()).collect()
    }

    pub fn absorb_bytes(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
    }

    /// Squeeze an extension field challenge (4 base-field elements → 124-bit soundness).
    pub fn squeeze_ef(&mut self) -> EF {
        let c0 = self.squeeze();
        let c1 = self.squeeze();
        let c2 = self.squeeze();
        let c3 = self.squeeze();
        EF::from_base_slice(&[Complex::new(c0, c1), Complex::new(c2, c3)])
    }

    /// Squeeze multiple extension field challenges.
    pub fn squeeze_ef_many(&mut self, n: usize) -> Vec<EF> {
        (0..n).map(|_| self.squeeze_ef()).collect()
    }

    /// Absorb an extension field element (4 base-field u32s).
    pub fn absorb_ef(&mut self, v: EF) {
        let base: &[Complex<Mersenne31>] = v.as_base_slice();
        let c0: &[Mersenne31] = base[0].as_base_slice();
        let c1: &[Mersenne31] = base[1].as_base_slice();
        self.absorb(c0[0].as_canonical_u32());
        self.absorb(c0[1].as_canonical_u32());
        self.absorb(c1[0].as_canonical_u32());
        self.absorb(c1[1].as_canonical_u32());
    }

}

/// Prove a product sumcheck: claimed_sum = Σ_x f[x] * g[x]
/// Returns (proof, f_at_s, g_at_s) where s is the challenge point.
#[allow(dead_code)]
pub fn prove_product(
    f: &[F],
    g: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof, F, F) {
    let mut f = f.to_vec();
    let mut g = g.to_vec();
    let mut size = f.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        // Compute round polynomial at t=0, t=1, t=2
        let mut s0 = F::zero();
        let mut s1 = F::zero();
        let mut s2 = F::zero();

        for j in 0..half {
            let f0 = f[j];
            let f1 = f[j + half];
            let g0 = g[j];
            let g1 = g[j + half];

            s0 += f0 * g0;
            s1 += f1 * g1;

            // At t=2: f(2) = f0 + 2*(f1-f0) = 2*f1 - f0
            let f2 = f1 + f1 - f0;
            let g2 = g1 + g1 - g0;
            s2 += f2 * g2;
        }

        // Keep all 3 evals — mle_evaluate_from_sumcheck_claim reads poly[1]
        let poly = vec![
            s0.as_canonical_u32(),
            s1.as_canonical_u32(),
            s2.as_canonical_u32(),
        ];
        transcript.absorb_many(&[s0, s1, s2]);
        let r = transcript.squeeze();

        // Fold
        let one_minus_r = F::one() - r;
        for j in 0..half {
            f[j] = one_minus_r * f[j] + r * f[j + half];
            g[j] = one_minus_r * g[j] + r * g[j + half];
        }

        round_polys.push(poly);
        challenges.push(r.as_canonical_u32());
        size = half;
    }

    let proof = SumcheckProof {
        round_polys,
        challenges,
    };
    (proof, f[0], g[0])
}

/// Prove a triple product sumcheck: claimed_sum = Σ_x f[x] * g[x] * h[x]
/// Returns (proof, f_at_s, g_at_s, h_at_s).
#[allow(dead_code)]
pub fn prove_triple(
    f: &[F],
    g: &[F],
    h: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof, F, F, F) {
    let mut f = f.to_vec();
    let mut g = g.to_vec();
    let mut h = h.to_vec();
    let mut size = f.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        let mut s0 = F::zero();
        let mut s1 = F::zero();
        let mut s2 = F::zero();
        let mut s3 = F::zero();

        for j in 0..half {
            let f0 = f[j];
            let f1 = f[j + half];
            let g0 = g[j];
            let g1 = g[j + half];
            let h0 = h[j];
            let h1 = h[j + half];

            s0 += f0 * g0 * h0;
            s1 += f1 * g1 * h1;

            let f2 = f1 + f1 - f0;
            let g2 = g1 + g1 - g0;
            let h2 = h1 + h1 - h0;
            s2 += f2 * g2 * h2;

            // t=3: f(3) = f0 + 3*(f1-f0) = 3*f1 - 2*f0
            let three = F::from_canonical_u32(3);
            let two = F::two();
            let f3 = three * f1 - two * f0;
            let g3 = three * g1 - two * g0;
            let h3 = three * h1 - two * h0;
            s3 += f3 * g3 * h3;
        }

        let poly = vec![
            s0.as_canonical_u32(),
            s1.as_canonical_u32(),
            s2.as_canonical_u32(),
            s3.as_canonical_u32(),
        ];
        transcript.absorb_many(&[s0, s1, s2, s3]);
        let r = transcript.squeeze();

        let one_minus_r = F::one() - r;
        for j in 0..half {
            f[j] = one_minus_r * f[j] + r * f[j + half];
            g[j] = one_minus_r * g[j] + r * g[j + half];
            h[j] = one_minus_r * h[j] + r * h[j + half];
        }

        round_polys.push(poly);
        challenges.push(r.as_canonical_u32());
        size = half;
    }

    (
        SumcheckProof {
            round_polys,
            challenges,
        },
        f[0],
        g[0],
        h[0],
    )
}

/// Parallelism threshold: below this half-size, sequential is faster.
const PAR_THRESHOLD: usize = 1024;

/// SIMD-accelerated linear interpolation: lo[j] = (1-r) * lo[j] + r * hi[j]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fold_lerp_simd(lo: &mut [F], hi: &[F], one_minus_r: F, r: F, n: usize) {
    const W: usize = 4;
    let omr = PackedMersenne31Neon::from(one_minus_r);
    let rp = PackedMersenne31Neon::from(r);
    let chunks = n / W;
    for c in 0..chunks {
        let j = c * W;
        let hi_packed = *PackedMersenne31Neon::from_slice(&hi[j..j + W]);
        let lo_packed = PackedMersenne31Neon::from_slice_mut(&mut lo[j..j + W]);
        *lo_packed = omr * *lo_packed + rp * hi_packed;
    }
    for j in (chunks * W)..n {
        lo[j] = one_minus_r * lo[j] + r * hi[j];
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn fold_lerp_simd(lo: &mut [F], hi: &[F], one_minus_r: F, r: F, n: usize) {
    for j in 0..n {
        lo[j] = one_minus_r * lo[j] + r * hi[j];
    }
}

/// Parallel version of `prove_product`. Uses rayon for the reduction and folding
/// steps when the working size >= PAR_THRESHOLD. Produces identical proofs
/// (same Fiat-Shamir transcript) as the sequential version.
pub fn prove_product_parallel(
    f: &[F],
    g: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof, F, F) {
    let mut f = f.to_vec();
    let mut g = g.to_vec();
    let mut size = f.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        let (s0, s1, s2) = if half >= PAR_THRESHOLD {
            (0..half)
                .into_par_iter()
                .map(|j| {
                    let f0 = f[j];
                    let f1 = f[j + half];
                    let g0 = g[j];
                    let g1 = g[j + half];
                    let s0 = f0 * g0;
                    let s1 = f1 * g1;
                    let f2 = f1 + f1 - f0;
                    let g2 = g1 + g1 - g0;
                    let s2 = f2 * g2;
                    (s0, s1, s2)
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |(a0, a1, a2), (b0, b1, b2)| (a0 + b0, a1 + b1, a2 + b2),
                )
        } else {
            let mut s0 = F::zero();
            let mut s1 = F::zero();
            let mut s2 = F::zero();
            for j in 0..half {
                let f0 = f[j];
                let f1 = f[j + half];
                let g0 = g[j];
                let g1 = g[j + half];
                s0 += f0 * g0;
                s1 += f1 * g1;
                let f2 = f1 + f1 - f0;
                let g2 = g1 + g1 - g0;
                s2 += f2 * g2;
            }
            (s0, s1, s2)
        };

        // Keep all 3 evals — mle_evaluate_from_sumcheck_claim reads poly[1]
        let poly = vec![
            s0.as_canonical_u32(),
            s1.as_canonical_u32(),
            s2.as_canonical_u32(),
        ];
        transcript.absorb_many(&[s0, s1, s2]);
        let r = transcript.squeeze();

        // Fold f and g
        let one_minus_r = F::one() - r;
        if half >= PAR_THRESHOLD {
            let (f_lo, f_hi) = f.split_at_mut(half);
            f_lo.par_iter_mut()
                .zip(f_hi.par_iter())
                .for_each(|(lo, hi)| {
                    *lo = one_minus_r * *lo + r * *hi;
                });
            let (g_lo, g_hi) = g.split_at_mut(half);
            g_lo.par_iter_mut()
                .zip(g_hi.par_iter())
                .for_each(|(lo, hi)| {
                    *lo = one_minus_r * *lo + r * *hi;
                });
        } else {
            let (f_lo, f_hi) = f.split_at_mut(half);
            fold_lerp_simd(f_lo, &f_hi[..half], one_minus_r, r, half);
            let (g_lo, g_hi) = g.split_at_mut(half);
            fold_lerp_simd(g_lo, &g_hi[..half], one_minus_r, r, half);
        }

        round_polys.push(poly);
        challenges.push(r.as_canonical_u32());
        size = half;
    }

    let proof = SumcheckProof {
        round_polys,
        challenges,
    };
    (proof, f[0], g[0])
}

/// Parallel version of `prove_triple`. Uses rayon for the reduction and folding
/// steps when the working size >= PAR_THRESHOLD. Produces identical proofs
/// (same Fiat-Shamir transcript) as the sequential version.
pub fn prove_triple_parallel(
    f: &[F],
    g: &[F],
    h: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof, F, F, F) {
    let mut f = f.to_vec();
    let mut g = g.to_vec();
    let mut h = h.to_vec();
    let mut size = f.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        let (s0, s1, s2, s3) = if half >= PAR_THRESHOLD {
            (0..half)
                .into_par_iter()
                .map(|j| {
                    let f0 = f[j];
                    let f1 = f[j + half];
                    let g0 = g[j];
                    let g1 = g[j + half];
                    let h0 = h[j];
                    let h1 = h[j + half];

                    let s0 = f0 * g0 * h0;
                    let s1 = f1 * g1 * h1;

                    let f2 = f1 + f1 - f0;
                    let g2 = g1 + g1 - g0;
                    let h2 = h1 + h1 - h0;
                    let s2 = f2 * g2 * h2;

                    let three = F::from_canonical_u32(3);
                    let two = F::two();
                    let f3 = three * f1 - two * f0;
                    let g3 = three * g1 - two * g0;
                    let h3 = three * h1 - two * h0;
                    let s3 = f3 * g3 * h3;

                    (s0, s1, s2, s3)
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero(), F::zero()),
                    |(a0, a1, a2, a3), (b0, b1, b2, b3)| {
                        (a0 + b0, a1 + b1, a2 + b2, a3 + b3)
                    },
                )
        } else {
            let mut s0 = F::zero();
            let mut s1 = F::zero();
            let mut s2 = F::zero();
            let mut s3 = F::zero();
            for j in 0..half {
                let f0 = f[j];
                let f1 = f[j + half];
                let g0 = g[j];
                let g1 = g[j + half];
                let h0 = h[j];
                let h1 = h[j + half];
                s0 += f0 * g0 * h0;
                s1 += f1 * g1 * h1;
                let f2 = f1 + f1 - f0;
                let g2 = g1 + g1 - g0;
                let h2 = h1 + h1 - h0;
                s2 += f2 * g2 * h2;
                let three = F::from_canonical_u32(3);
                let two = F::two();
                let f3 = three * f1 - two * f0;
                let g3 = three * g1 - two * g0;
                let h3 = three * h1 - two * h0;
                s3 += f3 * g3 * h3;
            }
            (s0, s1, s2, s3)
        };

        let poly = vec![
            s0.as_canonical_u32(),
            s1.as_canonical_u32(),
            s2.as_canonical_u32(),
            s3.as_canonical_u32(),
        ];
        transcript.absorb_many(&[s0, s1, s2, s3]);
        let r = transcript.squeeze();

        // Fold f, g, h
        let one_minus_r = F::one() - r;
        if half >= PAR_THRESHOLD {
            let (f_lo, f_hi) = f.split_at_mut(half);
            f_lo.par_iter_mut()
                .zip(f_hi.par_iter())
                .for_each(|(lo, hi)| {
                    *lo = one_minus_r * *lo + r * *hi;
                });
            let (g_lo, g_hi) = g.split_at_mut(half);
            g_lo.par_iter_mut()
                .zip(g_hi.par_iter())
                .for_each(|(lo, hi)| {
                    *lo = one_minus_r * *lo + r * *hi;
                });
            let (h_lo, h_hi) = h.split_at_mut(half);
            h_lo.par_iter_mut()
                .zip(h_hi.par_iter())
                .for_each(|(lo, hi)| {
                    *lo = one_minus_r * *lo + r * *hi;
                });
        } else {
            let (f_lo, f_hi) = f.split_at_mut(half);
            fold_lerp_simd(f_lo, &f_hi[..half], one_minus_r, r, half);
            let (g_lo, g_hi) = g.split_at_mut(half);
            fold_lerp_simd(g_lo, &g_hi[..half], one_minus_r, r, half);
            let (h_lo, h_hi) = h.split_at_mut(half);
            fold_lerp_simd(h_lo, &h_hi[..half], one_minus_r, r, half);
        }

        round_polys.push(poly);
        challenges.push(r.as_canonical_u32());
        size = half;
    }

    (
        SumcheckProof {
            round_polys,
            challenges,
        },
        f[0],
        g[0],
        h[0],
    )
}

/// GPU-accelerated version of `prove_product_parallel`. Uses Metal for reduce
/// and fold when half >= GPU_THRESHOLD, falls back to CPU otherwise.
/// Produces identical proofs (same Fiat-Shamir transcript) as the other variants.
#[cfg(feature = "metal_gpu")]
pub fn prove_product_gpu(
    f: &[F],
    g: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof, F, F) {
    use crate::gpu::{MetalContext, GpuBuffer, MetalKernels, GPU_THRESHOLD};

    let ctx = MetalContext::get();
    let kernels = MetalKernels::get();

    let mut f_buf = GpuBuffer::from_field_slice(&ctx.device, f);
    let mut g_buf = GpuBuffer::from_field_slice(&ctx.device, g);
    let mut size = f.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();
    let mut on_gpu = true;

    let mut f_cpu: Vec<F> = Vec::new();
    let mut g_cpu: Vec<F> = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        let (s0, s1, s2) = if on_gpu && half >= GPU_THRESHOLD {
            kernels.sumcheck_reduce_product(&f_buf, &g_buf, half)
        } else {
            if on_gpu {
                f_cpu = f_buf.to_field_vec()[..size].to_vec();
                g_cpu = g_buf.to_field_vec()[..size].to_vec();
                on_gpu = false;
            }
            let mut s0 = F::zero();
            let mut s1 = F::zero();
            let mut s2 = F::zero();
            for j in 0..half {
                let f0 = f_cpu[j]; let f1 = f_cpu[j + half];
                let g0 = g_cpu[j]; let g1 = g_cpu[j + half];
                s0 += f0 * g0;
                s1 += f1 * g1;
                let f2 = f1 + f1 - f0;
                let g2 = g1 + g1 - g0;
                s2 += f2 * g2;
            }
            (s0, s1, s2)
        };

        // Keep all 3 evals — mle_evaluate_from_sumcheck_claim reads poly[1]
        let poly = vec![
            s0.as_canonical_u32(),
            s1.as_canonical_u32(),
            s2.as_canonical_u32(),
        ];
        transcript.absorb_many(&[s0, s1, s2]);
        let r = transcript.squeeze();

        if on_gpu && half >= GPU_THRESHOLD {
            kernels.sumcheck_fold_pair(&mut f_buf, &mut g_buf, half, r);
        } else {
            if on_gpu {
                f_cpu = f_buf.to_field_vec()[..size].to_vec();
                g_cpu = g_buf.to_field_vec()[..size].to_vec();
                on_gpu = false;
            }
            let one_minus_r = F::one() - r;
            for j in 0..half {
                f_cpu[j] = one_minus_r * f_cpu[j] + r * f_cpu[j + half];
                g_cpu[j] = one_minus_r * g_cpu[j] + r * g_cpu[j + half];
            }
        }

        round_polys.push(poly);
        challenges.push(r.as_canonical_u32());
        size = half;
    }

    let (f_final, g_final) = if on_gpu {
        let fv = f_buf.to_field_vec();
        let gv = g_buf.to_field_vec();
        (fv[0], gv[0])
    } else {
        (f_cpu[0], g_cpu[0])
    };

    (SumcheckProof { round_polys, challenges }, f_final, g_final)
}

/// GPU-accelerated version of `prove_triple_parallel`.
#[cfg(feature = "metal_gpu")]
pub fn prove_triple_gpu(
    f: &[F],
    g: &[F],
    h: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof, F, F, F) {
    use crate::gpu::{MetalContext, GpuBuffer, MetalKernels, GPU_THRESHOLD};

    let ctx = MetalContext::get();
    let kernels = MetalKernels::get();

    let mut f_buf = GpuBuffer::from_field_slice(&ctx.device, f);
    let mut g_buf = GpuBuffer::from_field_slice(&ctx.device, g);
    let mut h_buf = GpuBuffer::from_field_slice(&ctx.device, h);
    let mut size = f.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();
    let mut on_gpu = true;

    let mut f_cpu: Vec<F> = Vec::new();
    let mut g_cpu: Vec<F> = Vec::new();
    let mut h_cpu: Vec<F> = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        let (s0, s1, s2, s3) = if on_gpu && half >= GPU_THRESHOLD {
            kernels.sumcheck_reduce_triple(&f_buf, &g_buf, &h_buf, half)
        } else {
            if on_gpu {
                f_cpu = f_buf.to_field_vec()[..size].to_vec();
                g_cpu = g_buf.to_field_vec()[..size].to_vec();
                h_cpu = h_buf.to_field_vec()[..size].to_vec();
                on_gpu = false;
            }
            let mut s0 = F::zero();
            let mut s1 = F::zero();
            let mut s2 = F::zero();
            let mut s3 = F::zero();
            for j in 0..half {
                let f0 = f_cpu[j]; let f1 = f_cpu[j + half];
                let g0 = g_cpu[j]; let g1 = g_cpu[j + half];
                let h0 = h_cpu[j]; let h1 = h_cpu[j + half];
                s0 += f0 * g0 * h0;
                s1 += f1 * g1 * h1;
                let f2 = f1 + f1 - f0;
                let g2 = g1 + g1 - g0;
                let h2 = h1 + h1 - h0;
                s2 += f2 * g2 * h2;
                let three = F::from_canonical_u32(3);
                let two = F::two();
                s3 += (three * f1 - two * f0) * (three * g1 - two * g0) * (three * h1 - two * h0);
            }
            (s0, s1, s2, s3)
        };

        let poly = vec![
            s0.as_canonical_u32(),
            s1.as_canonical_u32(),
            s2.as_canonical_u32(),
            s3.as_canonical_u32(),
        ];
        transcript.absorb_many(&[s0, s1, s2, s3]);
        let r = transcript.squeeze();

        if on_gpu && half >= GPU_THRESHOLD {
            kernels.sumcheck_fold_triple(&mut f_buf, &mut g_buf, &mut h_buf, half, r);
        } else {
            if on_gpu {
                f_cpu = f_buf.to_field_vec()[..size].to_vec();
                g_cpu = g_buf.to_field_vec()[..size].to_vec();
                h_cpu = h_buf.to_field_vec()[..size].to_vec();
                on_gpu = false;
            }
            let one_minus_r = F::one() - r;
            for j in 0..half {
                f_cpu[j] = one_minus_r * f_cpu[j] + r * f_cpu[j + half];
                g_cpu[j] = one_minus_r * g_cpu[j] + r * g_cpu[j + half];
                h_cpu[j] = one_minus_r * h_cpu[j] + r * h_cpu[j + half];
            }
        }

        round_polys.push(poly);
        challenges.push(r.as_canonical_u32());
        size = half;
    }

    let (f_final, g_final, h_final) = if on_gpu {
        let fv = f_buf.to_field_vec();
        let gv = g_buf.to_field_vec();
        let hv = h_buf.to_field_vec();
        (fv[0], gv[0], hv[0])
    } else {
        (f_cpu[0], g_cpu[0], h_cpu[0])
    };

    (SumcheckProof { round_polys, challenges }, f_final, g_final, h_final)
}

/// Auto-dispatch: GPU if enabled at runtime and large enough, else parallel.
pub fn prove_product_best(
    f: &[F],
    g: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof, F, F) {
    #[cfg(feature = "metal_gpu")]
    {
        if is_gpu_enabled() && f.len() >= crate::gpu::GPU_THRESHOLD * 2 {
            return prove_product_gpu(f, g, num_vars, transcript);
        }
    }
    prove_product_parallel(f, g, num_vars, transcript)
}

/// Auto-dispatch triple sumcheck.
pub fn prove_triple_best(
    f: &[F],
    g: &[F],
    h: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof, F, F, F) {
    #[cfg(feature = "metal_gpu")]
    {
        if is_gpu_enabled() && f.len() >= crate::gpu::GPU_THRESHOLD * 2 {
            return prove_triple_gpu(f, g, h, num_vars, transcript);
        }
    }
    prove_triple_parallel(f, g, h, num_vars, transcript)
}

/// Prove a product sumcheck where f is all-ones: claimed_sum = Σ_x 1 · g[x] = Σ g[x].
/// Skips folding the ones vector entirely (f is always 1 regardless of challenge).
/// Returns (proof, F::one(), g_at_s) — f_at_s is always 1.
///
/// The round polynomial simplifies because f(t) = 1 for all t:
///   s0 = Σ g[j]           (t=0, f=1)
///   s1 = Σ g[j+half]      (t=1, f=1)
///   s2 = Σ (2·g1 - g0)    (t=2, f=1, g(2) = 2·g1 - g0)
/// Note: s2 = 2·s1 - s0.
#[allow(dead_code)]
pub fn prove_product_ones(
    g: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof, F, F) {
    let mut g = g.to_vec();
    let mut size = g.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        let mut s0 = F::zero();
        let mut s1 = F::zero();

        for j in 0..half {
            s0 += g[j];
            s1 += g[j + half];
        }

        // s2 = 2·s1 - s0 (since f(2)=1 and g(2) = 2g1 - g0, product = g(2))
        let s2 = s1 + s1 - s0;

        // Compressed: [s0, s2], verifier derives s1 = claim - s0
        let poly = vec![s0.as_canonical_u32(), s2.as_canonical_u32()];
        transcript.absorb_many(&[s0, s1, s2]);
        let r = transcript.squeeze();

        // Only fold g — f stays 1
        let one_minus_r = F::one() - r;
        for j in 0..half {
            g[j] = one_minus_r * g[j] + r * g[j + half];
        }

        round_polys.push(poly);
        challenges.push(r.as_canonical_u32());
        size = half;
    }

    let proof = SumcheckProof {
        round_polys,
        challenges,
    };
    (proof, F::one(), g[0])
}

/// Parallel version of `prove_product_ones`. Uses rayon when half >= PAR_THRESHOLD.
/// Produces identical proofs (same Fiat-Shamir transcript) as the sequential version.
pub fn prove_product_ones_parallel(
    g: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof, F, F) {
    let mut g = g.to_vec();
    let mut size = g.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        let (s0, s1) = if half >= PAR_THRESHOLD {
            (0..half)
                .into_par_iter()
                .map(|j| (g[j], g[j + half]))
                .reduce(
                    || (F::zero(), F::zero()),
                    |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1),
                )
        } else {
            let mut s0 = F::zero();
            let mut s1 = F::zero();
            for j in 0..half {
                s0 += g[j];
                s1 += g[j + half];
            }
            (s0, s1)
        };

        let s2 = s1 + s1 - s0;

        // Compressed: [s0, s2], verifier derives s1 = claim - s0
        let poly = vec![s0.as_canonical_u32(), s2.as_canonical_u32()];
        transcript.absorb_many(&[s0, s1, s2]);
        let r = transcript.squeeze();

        // Only fold g — f stays 1
        let one_minus_r = F::one() - r;
        if half >= PAR_THRESHOLD {
            let (g_lo, g_hi) = g.split_at_mut(half);
            g_lo.par_iter_mut()
                .zip(g_hi.par_iter())
                .for_each(|(lo, hi)| {
                    *lo = one_minus_r * *lo + r * *hi;
                });
        } else {
            let (g_lo, g_hi) = g.split_at_mut(half);
            fold_lerp_simd(g_lo, &g_hi[..half], one_minus_r, r, half);
        }

        round_polys.push(poly);
        challenges.push(r.as_canonical_u32());
        size = half;
    }

    let proof = SumcheckProof {
        round_polys,
        challenges,
    };
    (proof, F::one(), g[0])
}

/// Auto-dispatch for all-ones product sumcheck: GPU if enabled, else parallel.
pub fn prove_product_ones_best(
    g: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProof, F, F) {
    // No GPU variant yet — use parallel
    prove_product_ones_parallel(g, num_vars, transcript)
}

/// Verify a product sumcheck proof.
pub fn verify_product(
    claimed_sum: F,
    proof: &SumcheckProof,
    _num_vars: usize,
    f_at_s: F,
    g_at_s: F,
    transcript: &mut Transcript,
) -> bool {
    let mut claim = claimed_sum;

    for (i, poly_raw) in proof.round_polys.iter().enumerate() {
        let vals: Vec<F> = poly_raw.iter().map(|&v| F::from_canonical_u32(v)).collect();
        // Support compressed (2-elem: [s0,s2]) and full (3-elem: [s0,s1,s2])
        let (s0, s1, s2) = if vals.len() == 2 {
            (vals[0], claim - vals[0], vals[1])
        } else {
            (vals[0], vals[1], vals[2])
        };

        if s0 + s1 != claim { return false; }

        transcript.absorb_many(&[s0, s1, s2]);
        let r = transcript.squeeze();

        if !proof.challenges.is_empty() && r.as_canonical_u32() != proof.challenges[i] {
            return false;
        }

        claim = interpolate_quadratic(&[s0, s1, s2], r);
    }

    // Final check
    claim == f_at_s * g_at_s
}

/// Same as verify_product but returns derived challenges (= evaluation point s*).
/// Handles compressed round_polys: if each round has 2 elements [s0, s2],
/// the verifier reconstructs s1 = claim - s0.
pub fn verify_product_with_challenges(
    claimed_sum: F,
    proof: &SumcheckProof,
    f_at_s: F,
    g_at_s: F,
    transcript: &mut Transcript,
) -> (bool, Vec<F>) {
    let mut claim = claimed_sum;
    let mut derived = Vec::with_capacity(proof.round_polys.len());

    for (i, poly_raw) in proof.round_polys.iter().enumerate() {
        let vals: Vec<F> = poly_raw.iter().map(|&v| F::from_canonical_u32(v)).collect();
        let (s0, s1, s2) = if vals.len() == 2 {
            (vals[0], claim - vals[0], vals[1])
        } else {
            (vals[0], vals[1], vals[2])
        };

        if s0 + s1 != claim { return (false, vec![]); }

        transcript.absorb_many(&[s0, s1, s2]);
        let r = transcript.squeeze();

        if !proof.challenges.is_empty() && r.as_canonical_u32() != proof.challenges[i] {
            return (false, vec![]);
        }

        derived.push(r);
        claim = interpolate_quadratic(&[s0, s1, s2], r);
    }

    (claim == f_at_s * g_at_s, derived)
}

/// Verify a triple product sumcheck proof.
pub fn verify_triple(
    claimed_sum: F,
    proof: &SumcheckProof,
    _num_vars: usize,
    f_at_s: F,
    g_at_s: F,
    h_at_s: F,
    transcript: &mut Transcript,
) -> bool {
    let mut claim = claimed_sum;

    for (i, poly_raw) in proof.round_polys.iter().enumerate() {
        let poly: Vec<F> = poly_raw.iter().map(|&v| F::from_canonical_u32(v)).collect();
        let s0 = poly[0];
        let s1 = poly[1];

        if s0 + s1 != claim {
            return false;
        }

        transcript.absorb_many(&poly);
        let r = transcript.squeeze();

        if !proof.challenges.is_empty() && r.as_canonical_u32() != proof.challenges[i] {
            return false;
        }

        claim = interpolate_cubic(&poly, r);
    }

    claim == f_at_s * g_at_s * h_at_s
}

/// Same as verify_triple but returns derived challenges (= evaluation point s*).
#[allow(dead_code)]
pub fn verify_triple_with_challenges(
    claimed_sum: F,
    proof: &SumcheckProof,
    f_at_s: F,
    g_at_s: F,
    h_at_s: F,
    transcript: &mut Transcript,
) -> (bool, Vec<F>) {
    let mut claim = claimed_sum;
    let mut derived = Vec::with_capacity(proof.round_polys.len());

    for (i, poly_raw) in proof.round_polys.iter().enumerate() {
        let poly: Vec<F> = poly_raw.iter().map(|&v| F::from_canonical_u32(v)).collect();
        let s0 = poly[0];
        let s1 = poly[1];

        if s0 + s1 != claim { return (false, vec![]); }

        transcript.absorb_many(&poly);
        let r = transcript.squeeze();

        if !proof.challenges.is_empty() && r.as_canonical_u32() != proof.challenges[i] {
            return (false, vec![]);
        }

        derived.push(r);
        claim = interpolate_cubic(&poly, r);
    }

    (claim == f_at_s * g_at_s * h_at_s, derived)
}

/// Lagrange interpolation through (0, y0), (1, y1), (2, y2) at x.
fn interpolate_quadratic(evals: &[F], x: F) -> F {
    let y0 = evals[0];
    let y1 = evals[1];
    let y2 = evals[2];

    let x0 = F::zero();
    let x1 = F::one();
    let x2 = F::two();

    let l0 = (x - x1) * (x - x2) * (F::two().inverse());
    let l1 = (x - x0) * (x - x2) * (F::zero() - F::one()).inverse();
    let l2 = (x - x0) * (x - x1) * (F::two().inverse());

    y0 * l0 + y1 * l1 + y2 * l2
}

/// Lagrange interpolation through (0,y0), (1,y1), (2,y2), (3,y3) at x.
fn interpolate_cubic(evals: &[F], x: F) -> F {
    let n = evals.len();
    let mut result = F::zero();
    for i in 0..n {
        let mut num = F::one();
        let mut den = F::one();
        for j in 0..n {
            if j == i {
                continue;
            }
            let xj = F::from_canonical_u32(j as u32);
            let xi = F::from_canonical_u32(i as u32);
            num *= x - xj;
            den *= xi - xj;
        }
        result += evals[i] * num * den.inverse();
    }
    result
}

// =============================================================================
// Extension-field sumcheck: 124-bit soundness via M31^4 challenges
// =============================================================================

/// Lagrange interpolation through (0, y0), (1, y1), (2, y2) at EF point x.
fn interpolate_quadratic_ef(evals: &[EF], x: EF) -> EF {
    let y0 = evals[0];
    let y1 = evals[1];
    let y2 = evals[2];

    let x0 = EF::zero();
    let x1 = EF::one();
    let x2 = EF::two();

    let l0 = (x - x1) * (x - x2) * (EF::two().inverse());
    let l1 = (x - x0) * (x - x2) * (EF::zero() - EF::one()).inverse();
    let l2 = (x - x0) * (x - x1) * (EF::two().inverse());

    y0 * l0 + y1 * l1 + y2 * l2
}

/// Lagrange interpolation through (0,y0), (1,y1), (2,y2), (3,y3) at EF point x.
fn interpolate_cubic_ef(evals: &[EF], x: EF) -> EF {
    let n = evals.len();
    let mut result = EF::zero();
    for i in 0..n {
        let mut num = EF::one();
        let mut den = EF::one();
        for j in 0..n {
            if j == i { continue; }
            let xj = EF::from_base(Complex::new(F::from_canonical_u32(j as u32), F::zero()));
            let xi = EF::from_base(Complex::new(F::from_canonical_u32(i as u32), F::zero()));
            num *= x - xj;
            den *= xi - xj;
        }
        result += evals[i] * num * den.inverse();
    }
    result
}

/// Product sumcheck with EF challenges: Σ f(x)·g(x) = c.
///
/// Inputs f, g are base-field vectors. Round 0 computes in base field,
/// then challenges are EF, so after first fold working arrays become Vec<EF>.
/// Returns (proof, f_at_s, g_at_s) where all are EF.
pub fn prove_product_ef(
    f: &[F],
    g: &[F],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProofEF, EF, EF) {
    // After first round, we work in EF. Start with base field.
    let mut f_ef: Vec<EF> = f.iter().map(|&v| f_to_ef(v)).collect();
    let mut g_ef: Vec<EF> = g.iter().map(|&v| f_to_ef(v)).collect();
    let mut size = f_ef.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        let mut s0 = EF::zero();
        let mut s1 = EF::zero();
        let mut s2 = EF::zero();

        for j in 0..half {
            let f0 = f_ef[j];
            let f1 = f_ef[j + half];
            let g0 = g_ef[j];
            let g1 = g_ef[j + half];

            s0 += f0 * g0;
            s1 += f1 * g1;

            let f2 = f1 + f1 - f0;
            let g2 = g1 + g1 - g0;
            s2 += f2 * g2;
        }

        // Store only [s0, s2] — s1 derivable as (claim - s0). Saves 16B/round.
        let poly = vec![EFElement::from_ef(s0), EFElement::from_ef(s2)];
        transcript.absorb_ef(s0);
        transcript.absorb_ef(s1);
        transcript.absorb_ef(s2);
        let r = transcript.squeeze_ef();

        let one_minus_r = EF::one() - r;
        for j in 0..half {
            f_ef[j] = one_minus_r * f_ef[j] + r * f_ef[j + half];
            g_ef[j] = one_minus_r * g_ef[j] + r * g_ef[j + half];
        }

        round_polys.push(poly);
        challenges.push(EFElement::from_ef(r));
        size = half;
    }

    let proof = SumcheckProofEF { round_polys, challenges };
    (proof, f_ef[0], g_ef[0])
}

/// Product sumcheck where both inputs are already in EF.
/// Used for MLE eval proofs on extension-field vectors (e.g., w_partial from EF matmul).
pub fn prove_product_ef_full(
    f: &[EF],
    g: &[EF],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProofEF, EF, EF) {
    let mut f_ef = f.to_vec();
    let mut g_ef = g.to_vec();
    let mut size = f_ef.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        let mut s0 = EF::zero();
        let mut s1 = EF::zero();
        let mut s2 = EF::zero();

        for j in 0..half {
            let f0 = f_ef[j];
            let f1 = f_ef[j + half];
            let g0 = g_ef[j];
            let g1 = g_ef[j + half];

            s0 += f0 * g0;
            s1 += f1 * g1;

            let f2 = f1 + f1 - f0;
            let g2 = g1 + g1 - g0;
            s2 += f2 * g2;
        }

        // Keep all 3 evals — used by mle_evaluate_from_sumcheck_claim_ef.
        let poly = vec![EFElement::from_ef(s0), EFElement::from_ef(s1), EFElement::from_ef(s2)];
        transcript.absorb_ef(s0);
        transcript.absorb_ef(s1);
        transcript.absorb_ef(s2);
        let r = transcript.squeeze_ef();

        let one_minus_r = EF::one() - r;
        for j in 0..half {
            f_ef[j] = one_minus_r * f_ef[j] + r * f_ef[j + half];
            g_ef[j] = one_minus_r * g_ef[j] + r * g_ef[j + half];
        }

        round_polys.push(poly);
        challenges.push(EFElement::from_ef(r));
        size = half;
    }

    let proof = SumcheckProofEF { round_polys, challenges };
    (proof, f_ef[0], g_ef[0])
}

/// Triple sumcheck where all inputs are already in EF.
pub fn prove_triple_ef_full(
    f: &[EF],
    g: &[EF],
    h: &[EF],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProofEF, EF, EF, EF) {
    let mut f_ef = f.to_vec();
    let mut g_ef = g.to_vec();
    let mut h_ef = h.to_vec();
    let mut size = f_ef.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        let mut s0 = EF::zero();
        let mut s1 = EF::zero();
        let mut s2 = EF::zero();
        let mut s3 = EF::zero();

        for j in 0..half {
            let f0 = f_ef[j];
            let f1 = f_ef[j + half];
            let g0 = g_ef[j];
            let g1 = g_ef[j + half];
            let h0 = h_ef[j];
            let h1 = h_ef[j + half];

            s0 += f0 * g0 * h0;
            s1 += f1 * g1 * h1;

            let f2 = f1 + f1 - f0;
            let g2 = g1 + g1 - g0;
            let h2 = h1 + h1 - h0;
            s2 += f2 * g2 * h2;

            let f3 = f2 + (f1 - f0);
            let g3 = g2 + (g1 - g0);
            let h3 = h2 + (h1 - h0);
            s3 += f3 * g3 * h3;
        }

        // Store [s0, s2, s3] — s1 derivable as (claim - s0). Saves 16B/round.
        let poly = vec![EFElement::from_ef(s0), EFElement::from_ef(s2), EFElement::from_ef(s3)];
        transcript.absorb_ef(s0);
        transcript.absorb_ef(s1);
        transcript.absorb_ef(s2);
        transcript.absorb_ef(s3);
        let r = transcript.squeeze_ef();

        let one_minus_r = EF::one() - r;
        for j in 0..half {
            f_ef[j] = one_minus_r * f_ef[j] + r * f_ef[j + half];
            g_ef[j] = one_minus_r * g_ef[j] + r * g_ef[j + half];
            h_ef[j] = one_minus_r * h_ef[j] + r * h_ef[j + half];
        }

        round_polys.push(poly);
        challenges.push(EFElement::from_ef(r));
        size = half;
    }

    let proof = SumcheckProofEF { round_polys, challenges };
    (proof, f_ef[0], g_ef[0], h_ef[0])
}

/// Product-ones sumcheck where the input is already in EF.
/// Proves Σ 1·g[i] = claim where g is Vec<EF>.
#[allow(dead_code)]
pub fn prove_product_ones_ef_full(
    g: &[EF],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (SumcheckProofEF, EF, EF) {
    let mut g_ef = g.to_vec();
    let mut size = g_ef.len();
    let mut round_polys = Vec::new();
    let mut challenges = Vec::new();

    for _ in 0..num_vars {
        let half = size / 2;

        let mut s0 = EF::zero();
        let mut s1 = EF::zero();

        for j in 0..half {
            s0 += g_ef[j];
            s1 += g_ef[j + half];
        }

        // s2 = 2*s1 - s0 (since f=1: g(2) = 2g(half+j) - g(j), product = g(2))
        let s2 = s1 + s1 - s0;

        // Compressed: [s0, s2], verifier derives s1 = claim - s0
        let poly = vec![
            EFElement::from_ef(s0),
            EFElement::from_ef(s2),
        ];
        transcript.absorb_ef(s0);
        transcript.absorb_ef(s1);
        transcript.absorb_ef(s2);
        let r = transcript.squeeze_ef();

        let one_minus_r = EF::one() - r;
        for j in 0..half {
            g_ef[j] = one_minus_r * g_ef[j] + r * g_ef[j + half];
        }

        round_polys.push(poly);
        challenges.push(EFElement::from_ef(r));
        size = half;
    }

    let proof = SumcheckProofEF { round_polys, challenges };
    (proof, EF::one(), g_ef[0])
}

/// Verify a product sumcheck proof with EF challenges.
#[allow(dead_code)]
pub fn verify_product_ef(
    claimed_sum: EF,
    proof: &SumcheckProofEF,
    _num_vars: usize,
    f_at_s: EF,
    g_at_s: EF,
    transcript: &mut Transcript,
) -> bool {
    let mut claim = claimed_sum;

    for (i, poly_raw) in proof.round_polys.iter().enumerate() {
        let vals: Vec<EF> = poly_raw.iter().map(|e| e.to_ef()).collect();
        let (s0, s1, s2) = if vals.len() == 2 {
            (vals[0], claim - vals[0], vals[1])
        } else {
            (vals[0], vals[1], vals[2])
        };

        if s0 + s1 != claim { return false; }

        transcript.absorb_ef(s0);
        transcript.absorb_ef(s1);
        transcript.absorb_ef(s2);
        let r = transcript.squeeze_ef();

        if !proof.challenges.is_empty() && EFElement::from_ef(r) != proof.challenges[i] {
            return false;
        }

        claim = interpolate_quadratic_ef(&[s0, s1, s2], r);
    }

    claim == f_at_s * g_at_s
}

/// Same as verify_product_ef but returns the derived challenges (= evaluation point s*).
/// Handles compressed round_polys: if each round has 2 elements [s0, s2],
/// the verifier reconstructs s1 = claim - s0.
pub fn verify_product_ef_with_challenges(
    claimed_sum: EF,
    proof: &SumcheckProofEF,
    f_at_s: EF,
    g_at_s: EF,
    transcript: &mut Transcript,
) -> (bool, Vec<EF>) {
    let mut claim = claimed_sum;
    let mut derived = Vec::with_capacity(proof.round_polys.len());

    for (i, poly_raw) in proof.round_polys.iter().enumerate() {
        let vals: Vec<EF> = poly_raw.iter().map(|e| e.to_ef()).collect();
        // Support both compressed (2-element) and full (3-element) round polys
        let (s0, s1, s2) = if vals.len() == 2 {
            (vals[0], claim - vals[0], vals[1])
        } else {
            (vals[0], vals[1], vals[2])
        };

        if s0 + s1 != claim { return (false, vec![]); }

        transcript.absorb_ef(s0);
        transcript.absorb_ef(s1);
        transcript.absorb_ef(s2);
        let r = transcript.squeeze_ef();

        if !proof.challenges.is_empty() && EFElement::from_ef(r) != proof.challenges[i] {
            return (false, vec![]);
        }

        derived.push(r);
        claim = interpolate_quadratic_ef(&[s0, s1, s2], r);
    }

    (claim == f_at_s * g_at_s, derived)
}

/// Verify a triple product sumcheck proof with EF challenges, returning derived challenges.
/// Used for stripping challenges from proof (future RmsNorm optimization).
#[allow(dead_code)]
pub fn verify_triple_ef_with_challenges(
    claimed_sum: EF,
    proof: &SumcheckProofEF,
    f_at_s: EF,
    g_at_s: EF,
    h_at_s: EF,
    transcript: &mut Transcript,
) -> (bool, Vec<EF>) {
    let mut claim = claimed_sum;
    let mut derived = Vec::with_capacity(proof.round_polys.len());

    for (i, poly_raw) in proof.round_polys.iter().enumerate() {
        let vals: Vec<EF> = poly_raw.iter().map(|e| e.to_ef()).collect();
        let (s0, s1, s2, s3) = if vals.len() == 3 {
            (vals[0], claim - vals[0], vals[1], vals[2])
        } else {
            (vals[0], vals[1], vals[2], vals[3])
        };

        if s0 + s1 != claim { return (false, vec![]); }

        transcript.absorb_ef(s0);
        transcript.absorb_ef(s1);
        transcript.absorb_ef(s2);
        transcript.absorb_ef(s3);
        let r = transcript.squeeze_ef();

        if !proof.challenges.is_empty() && EFElement::from_ef(r) != proof.challenges[i] {
            return (false, vec![]);
        }

        derived.push(r);
        claim = interpolate_cubic_ef(&[s0, s1, s2, s3], r);
    }

    (claim == f_at_s * g_at_s * h_at_s, derived)
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_product_sumcheck() {
        // f = [3, 5], g = [2, 4]
        // sum = 3*2 + 5*4 = 26
        let f: Vec<F> = vec![3, 5].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let g: Vec<F> = vec![2, 4].iter().map(|&v| F::from_canonical_u32(v)).collect();
        let claimed = F::from_canonical_u32(26);

        let mut t = Transcript::new(b"test");
        let (proof, f_at_s, g_at_s) = prove_product(&f, &g, 1, &mut t);

        let mut t2 = Transcript::new(b"test");
        assert!(verify_product(claimed, &proof, 1, f_at_s, g_at_s, &mut t2));
    }

    #[test]
    fn test_triple_sumcheck() {
        let f: Vec<F> = vec![1, 2, 3, 4]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let g = f.clone();
        let h: Vec<F> = vec![1, 1, 1, 1]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        // sum = 1*1*1 + 2*2*1 + 3*3*1 + 4*4*1 = 1+4+9+16 = 30
        let claimed = F::from_canonical_u32(30);

        let mut t = Transcript::new(b"test");
        let (proof, fa, ga, ha) = prove_triple(&f, &g, &h, 2, &mut t);

        let mut t2 = Transcript::new(b"test");
        assert!(verify_triple(claimed, &proof, 2, fa, ga, ha, &mut t2));
    }

    /// Helper to generate random field elements for testing.
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    static TEST_SEED: AtomicU64 = AtomicU64::new(0);

    fn random_field_vec(n: usize) -> Vec<F> {
        use rand::SeedableRng;
        use rand::Rng;
        let seed = TEST_SEED.fetch_add(1, AtomicOrdering::Relaxed);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| F::from_canonical_u32(rng.gen_range(0..((1u32 << 31) - 1))))
            .collect()
    }

    #[test]
    fn test_parallel_product_small() {
        // Small input (below threshold) — should still produce identical results.
        let f: Vec<F> = vec![3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let g: Vec<F> = vec![2, 4, 6, 8]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();

        let mut t_seq = Transcript::new(b"par_test");
        let (proof_seq, fs, gs) = prove_product(&f, &g, 2, &mut t_seq);

        let mut t_par = Transcript::new(b"par_test");
        let (proof_par, fp, gp) = prove_product_parallel(&f, &g, 2, &mut t_par);

        assert_eq!(proof_seq.round_polys, proof_par.round_polys);
        assert_eq!(proof_seq.challenges, proof_par.challenges);
        assert_eq!(fs, fp);
        assert_eq!(gs, gp);
    }

    #[test]
    fn test_parallel_product_large() {
        // Large input (above threshold) — exercises the rayon path.
        let num_vars = 12; // 4096 elements
        let n = 1 << num_vars;
        let f = random_field_vec(n);
        let g = random_field_vec(n);

        let mut t_seq = Transcript::new(b"large_product");
        let (proof_seq, fs, gs) = prove_product(&f, &g, num_vars, &mut t_seq);

        let mut t_par = Transcript::new(b"large_product");
        let (proof_par, fp, gp) = prove_product_parallel(&f, &g, num_vars, &mut t_par);

        assert_eq!(proof_seq.round_polys, proof_par.round_polys);
        assert_eq!(proof_seq.challenges, proof_par.challenges);
        assert_eq!(fs, fp);
        assert_eq!(gs, gp);

        // Verify the parallel proof
        let claimed: F = f.iter().zip(g.iter()).map(|(&a, &b)| a * b).sum();
        let mut t_v = Transcript::new(b"large_product");
        assert!(verify_product(claimed, &proof_par, num_vars, fp, gp, &mut t_v));
    }

    #[test]
    fn test_parallel_triple_small() {
        let f: Vec<F> = vec![1, 2, 3, 4]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let g = f.clone();
        let h: Vec<F> = vec![1, 1, 1, 1]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();

        let mut t_seq = Transcript::new(b"triple_par");
        let (proof_seq, fs, gs, hs) = prove_triple(&f, &g, &h, 2, &mut t_seq);

        let mut t_par = Transcript::new(b"triple_par");
        let (proof_par, fp, gp, hp) = prove_triple_parallel(&f, &g, &h, 2, &mut t_par);

        assert_eq!(proof_seq.round_polys, proof_par.round_polys);
        assert_eq!(proof_seq.challenges, proof_par.challenges);
        assert_eq!(fs, fp);
        assert_eq!(gs, gp);
        assert_eq!(hs, hp);
    }

    #[cfg(feature = "metal_gpu")]
    #[test]
    fn test_gpu_product_proof_matches_cpu() {
        super::set_gpu_enabled(true);
        let num_vars = 16; // 65536 elements — well above GPU_THRESHOLD
        let n = 1 << num_vars;
        let f = random_field_vec(n);
        let g = random_field_vec(n);

        let mut t_seq = Transcript::new(b"gpu_product");
        let (proof_seq, fs, gs) = prove_product(&f, &g, num_vars, &mut t_seq);

        let mut t_gpu = Transcript::new(b"gpu_product");
        let (proof_gpu, fg, gg) = super::prove_product_gpu(&f, &g, num_vars, &mut t_gpu);

        assert_eq!(proof_seq.round_polys, proof_gpu.round_polys, "round polys mismatch");
        assert_eq!(proof_seq.challenges, proof_gpu.challenges, "challenges mismatch");
        assert_eq!(fs, fg);
        assert_eq!(gs, gg);

        // Verify the GPU proof
        let claimed: F = f.iter().zip(g.iter()).map(|(&a, &b)| a * b).sum();
        let mut t_v = Transcript::new(b"gpu_product");
        assert!(verify_product(claimed, &proof_gpu, num_vars, fg, gg, &mut t_v));
    }

    #[cfg(feature = "metal_gpu")]
    #[test]
    fn test_gpu_triple_proof_matches_cpu() {
        super::set_gpu_enabled(true);
        let num_vars = 16;
        let n = 1 << num_vars;
        let f = random_field_vec(n);
        let g = random_field_vec(n);
        let h = random_field_vec(n);

        let mut t_seq = Transcript::new(b"gpu_triple");
        let (proof_seq, fs, gs, hs) = prove_triple(&f, &g, &h, num_vars, &mut t_seq);

        let mut t_gpu = Transcript::new(b"gpu_triple");
        let (proof_gpu, fg, gg, hg) = super::prove_triple_gpu(&f, &g, &h, num_vars, &mut t_gpu);

        assert_eq!(proof_seq.round_polys, proof_gpu.round_polys, "round polys mismatch");
        assert_eq!(proof_seq.challenges, proof_gpu.challenges, "challenges mismatch");
        assert_eq!(fs, fg);
        assert_eq!(gs, gg);
        assert_eq!(hs, hg);

        let claimed: F = f.iter().zip(g.iter()).zip(h.iter()).map(|((&a, &b), &c)| a * b * c).sum();
        let mut t_v = Transcript::new(b"gpu_triple");
        assert!(verify_triple(claimed, &proof_gpu, num_vars, fg, gg, hg, &mut t_v));
    }

    #[cfg(feature = "metal_gpu")]
    #[test]
    fn bench_gpu_vs_cpu_product() {
        super::set_gpu_enabled(true);
        use std::time::Instant;

        let num_vars = 18; // 256K elements
        let n = 1 << num_vars;
        let f = random_field_vec(n);
        let g = random_field_vec(n);

        // Warmup GPU
        {
            let mut t = Transcript::new(b"warmup");
            let _ = super::prove_product_gpu(&f, &g, num_vars, &mut t);
        }

        // GPU
        let start = Instant::now();
        let mut t_gpu = Transcript::new(b"bench");
        let _ = super::prove_product_gpu(&f, &g, num_vars, &mut t_gpu);
        let gpu_ms = start.elapsed().as_secs_f64() * 1000.0;

        // CPU parallel
        let start = Instant::now();
        let mut t_cpu = Transcript::new(b"bench");
        let _ = prove_product_parallel(&f, &g, num_vars, &mut t_cpu);
        let cpu_ms = start.elapsed().as_secs_f64() * 1000.0;

        eprintln!("Product sumcheck 2^{num_vars}: GPU={gpu_ms:.2}ms, CPU={cpu_ms:.2}ms, speedup={:.1}x", cpu_ms / gpu_ms);
    }

    #[test]
    fn test_parallel_triple_large() {
        let num_vars = 12;
        let n = 1 << num_vars;
        let f = random_field_vec(n);
        let g = random_field_vec(n);
        let h = random_field_vec(n);

        let mut t_seq = Transcript::new(b"large_triple");
        let (proof_seq, fs, gs, hs) = prove_triple(&f, &g, &h, num_vars, &mut t_seq);

        let mut t_par = Transcript::new(b"large_triple");
        let (proof_par, fp, gp, hp) = prove_triple_parallel(&f, &g, &h, num_vars, &mut t_par);

        assert_eq!(proof_seq.round_polys, proof_par.round_polys);
        assert_eq!(proof_seq.challenges, proof_par.challenges);
        assert_eq!(fs, fp);
        assert_eq!(gs, gp);
        assert_eq!(hs, hp);

        // Verify the parallel proof
        let claimed: F = f
            .iter()
            .zip(g.iter())
            .zip(h.iter())
            .map(|((&a, &b), &c)| a * b * c)
            .sum();
        let mut t_v = Transcript::new(b"large_triple");
        assert!(verify_triple(claimed, &proof_par, num_vars, fp, gp, hp, &mut t_v));
    }

    #[test]
    fn test_product_ones_matches_product() {
        // prove_product_ones(g) should produce identical proofs to prove_product(ones, g)
        let g: Vec<F> = vec![3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();
        let ones = vec![F::one(); 4];

        let mut t_prod = Transcript::new(b"ones_test");
        let (proof_prod, f_prod, g_prod) = prove_product(&ones, &g, 2, &mut t_prod);

        let mut t_ones = Transcript::new(b"ones_test");
        let (proof_ones, f_ones, g_ones) = prove_product_ones(&g, 2, &mut t_ones);

        // Round polys differ structurally (ones is compressed [s0,s2] vs full [s0,s1,s2])
        // but both verify correctly and produce the same final values.
        assert_eq!(f_prod, f_ones);
        assert_eq!(f_ones, F::one());
        assert_eq!(g_prod, g_ones);
        // Verify both proofs pass
        let mut v1 = Transcript::new(b"ones_test");
        assert!(verify_product(F::from_canonical_u32(26), &proof_prod, 2, f_prod, g_prod, &mut v1));
        let mut v2 = Transcript::new(b"ones_test");
        assert!(verify_product(F::from_canonical_u32(26), &proof_ones, 2, f_ones, g_ones, &mut v2));
    }

    #[test]
    fn test_product_ones_parallel_matches_sequential() {
        let g: Vec<F> = vec![3, 5, 7, 11]
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect();

        let mut t_seq = Transcript::new(b"ones_par");
        let (proof_seq, f_seq, g_seq) = prove_product_ones(&g, 2, &mut t_seq);

        let mut t_par = Transcript::new(b"ones_par");
        let (proof_par, f_par, g_par) = prove_product_ones_parallel(&g, 2, &mut t_par);

        assert_eq!(proof_seq.round_polys, proof_par.round_polys);
        assert_eq!(proof_seq.challenges, proof_par.challenges);
        assert_eq!(f_seq, f_par);
        assert_eq!(g_seq, g_par);
    }

    #[test]
    fn test_product_ones_large_matches_product() {
        let num_vars = 12; // 4096 elements — exercises parallel path
        let n = 1 << num_vars;
        let g = random_field_vec(n);
        let ones = vec![F::one(); n];

        let mut t_prod = Transcript::new(b"ones_large");
        let (proof_prod, f_prod, g_prod) = prove_product_parallel(&ones, &g, num_vars, &mut t_prod);

        let mut t_ones = Transcript::new(b"ones_large");
        let (proof_ones, f_ones, g_ones) = prove_product_ones_parallel(&g, num_vars, &mut t_ones);

        // Round polys differ structurally (compressed vs full) but both verify.
        assert_eq!(f_prod, f_ones);
        assert_eq!(f_ones, F::one());
        assert_eq!(g_prod, g_ones);

        // Verify both proofs
        let claimed: F = g.iter().copied().sum();
        let mut t_v1 = Transcript::new(b"ones_large");
        assert!(verify_product(claimed, &proof_prod, num_vars, f_prod, g_prod, &mut t_v1));
        let mut t_v2 = Transcript::new(b"ones_large");
        assert!(verify_product(claimed, &proof_ones, num_vars, f_ones, g_ones, &mut t_v2));
    }
}
