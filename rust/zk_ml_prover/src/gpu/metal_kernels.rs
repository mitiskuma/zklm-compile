//! Typed kernel dispatch wrappers for Metal compute shaders.

use metal::*;
use p3_field::{AbstractField, PrimeField32};
use p3_mersenne_31::Mersenne31;

use super::metal_buffer::GpuBuffer;
use super::metal_device::{MetalContext, wait_and_check};

type F = Mersenne31;

const THREADGROUP_SIZE: u64 = 256;

pub struct MetalKernels {
    fold_pipeline: ComputePipelineState,
    reduce_product_pipeline: ComputePipelineState,
    reduce_triple_pipeline: ComputePipelineState,
    batch_inverse_pipeline: ComputePipelineState,
    eq_butterfly_pipeline: ComputePipelineState,
}

static KERNELS: std::sync::OnceLock<MetalKernels> = std::sync::OnceLock::new();

impl MetalKernels {
    pub fn get() -> &'static MetalKernels {
        KERNELS.get_or_init(|| {
            let ctx = MetalContext::get();

            let make_pipeline = |name: &str| {
                let func = ctx.library.get_function(name, None)
                    .unwrap_or_else(|e| panic!("kernel '{name}' not found: {e}"));
                ctx.device.new_compute_pipeline_state_with_function(&func)
                    .unwrap_or_else(|e| panic!("failed to create pipeline for '{name}': {e}"))
            };

            MetalKernels {
                fold_pipeline: make_pipeline("sumcheck_fold"),
                reduce_product_pipeline: make_pipeline("sumcheck_reduce_product"),
                reduce_triple_pipeline: make_pipeline("sumcheck_reduce_triple"),
                batch_inverse_pipeline: make_pipeline("batch_m31_inverse"),
                eq_butterfly_pipeline: make_pipeline("eq_evals_butterfly"),
            }
        })
    }

    #[allow(dead_code)]
    pub fn new() -> &'static MetalKernels {
        Self::get()
    }

    /// Dispatch sumcheck_fold in-place on `f`.
    pub fn sumcheck_fold(&self, f: &mut GpuBuffer, half: usize, r: F) {
        let ctx = MetalContext::get();
        let r_val: u32 = r.as_canonical_u32();
        let one_minus_r: u32 = (F::one() - r).as_canonical_u32();
        let half_u32: u32 = half as u32;

        let cmd = ctx.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.fold_pipeline);
        enc.set_buffer(0, Some(&f.buffer), 0);
        enc.set_bytes(1, 4, &half_u32 as *const u32 as *const _);
        enc.set_bytes(2, 4, &r_val as *const u32 as *const _);
        enc.set_bytes(3, 4, &one_minus_r as *const u32 as *const _);

        let grid = MTLSize::new(half as u64, 1, 1);
        let tg = MTLSize::new(THREADGROUP_SIZE.min(half as u64), 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        wait_and_check(&cmd, "sumcheck_fold");
    }

    /// Fold two buffers in one command buffer (saves a dispatch round-trip).
    pub fn sumcheck_fold_pair(&self, f: &mut GpuBuffer, g: &mut GpuBuffer, half: usize, r: F) {
        let ctx = MetalContext::get();
        let r_val: u32 = r.as_canonical_u32();
        let one_minus_r: u32 = (F::one() - r).as_canonical_u32();
        let half_u32: u32 = half as u32;

        let cmd = ctx.queue.new_command_buffer();
        let grid = MTLSize::new(half as u64, 1, 1);
        let tg = MTLSize::new(THREADGROUP_SIZE.min(half as u64), 1, 1);

        // Fold f
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.fold_pipeline);
        enc.set_buffer(0, Some(&f.buffer), 0);
        enc.set_bytes(1, 4, &half_u32 as *const u32 as *const _);
        enc.set_bytes(2, 4, &r_val as *const u32 as *const _);
        enc.set_bytes(3, 4, &one_minus_r as *const u32 as *const _);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();

        // Fold g
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.fold_pipeline);
        enc.set_buffer(0, Some(&g.buffer), 0);
        enc.set_bytes(1, 4, &half_u32 as *const u32 as *const _);
        enc.set_bytes(2, 4, &r_val as *const u32 as *const _);
        enc.set_bytes(3, 4, &one_minus_r as *const u32 as *const _);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();

        cmd.commit();
        wait_and_check(&cmd, "sumcheck_fold_pair");
    }

    /// Fold three buffers in one command buffer.
    pub fn sumcheck_fold_triple(&self, f: &mut GpuBuffer, g: &mut GpuBuffer, h: &mut GpuBuffer, half: usize, r: F) {
        let ctx = MetalContext::get();
        let r_val: u32 = r.as_canonical_u32();
        let one_minus_r: u32 = (F::one() - r).as_canonical_u32();
        let half_u32: u32 = half as u32;

        let cmd = ctx.queue.new_command_buffer();
        let grid = MTLSize::new(half as u64, 1, 1);
        let tg = MTLSize::new(THREADGROUP_SIZE.min(half as u64), 1, 1);

        for buf in [&f.buffer, &g.buffer, &h.buffer] {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.fold_pipeline);
            enc.set_buffer(0, Some(buf), 0);
            enc.set_bytes(1, 4, &half_u32 as *const u32 as *const _);
            enc.set_bytes(2, 4, &r_val as *const u32 as *const _);
            enc.set_bytes(3, 4, &one_minus_r as *const u32 as *const _);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
        }

        cmd.commit();
        wait_and_check(&cmd, "sumcheck_fold_triple");
    }

    /// Dispatch sumcheck_reduce_product, return (s0, s1, s2).
    pub fn sumcheck_reduce_product(&self, f: &GpuBuffer, g: &GpuBuffer, half: usize) -> (F, F, F) {
        let ctx = MetalContext::get();
        let num_groups = ((half as u64) + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
        let mut partials = GpuBuffer::new_uninit(&ctx.device, (num_groups as usize) * 3);
        let half_u32: u32 = half as u32;

        let cmd = ctx.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.reduce_product_pipeline);
        enc.set_buffer(0, Some(&f.buffer), 0);
        enc.set_buffer(1, Some(&g.buffer), 0);
        enc.set_buffer(2, Some(&partials.buffer), 0);
        enc.set_bytes(3, 4, &half_u32 as *const u32 as *const _);

        let grid = MTLSize::new(num_groups * THREADGROUP_SIZE, 1, 1);
        let tg = MTLSize::new(THREADGROUP_SIZE, 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        wait_and_check(&cmd, "sumcheck_reduce_product");

        let data = partials.as_u32_slice_mut();
        let mut s0 = F::zero();
        let mut s1 = F::zero();
        let mut s2 = F::zero();
        for i in 0..num_groups as usize {
            s0 += F::from_canonical_u32(data[i * 3]);
            s1 += F::from_canonical_u32(data[i * 3 + 1]);
            s2 += F::from_canonical_u32(data[i * 3 + 2]);
        }
        (s0, s1, s2)
    }

    /// Dispatch sumcheck_reduce_triple, return (s0, s1, s2, s3).
    pub fn sumcheck_reduce_triple(
        &self, f: &GpuBuffer, g: &GpuBuffer, h: &GpuBuffer, half: usize,
    ) -> (F, F, F, F) {
        let ctx = MetalContext::get();
        let num_groups = ((half as u64) + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
        let mut partials = GpuBuffer::new_uninit(&ctx.device, (num_groups as usize) * 4);
        let half_u32: u32 = half as u32;

        let cmd = ctx.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.reduce_triple_pipeline);
        enc.set_buffer(0, Some(&f.buffer), 0);
        enc.set_buffer(1, Some(&g.buffer), 0);
        enc.set_buffer(2, Some(&h.buffer), 0);
        enc.set_buffer(3, Some(&partials.buffer), 0);
        enc.set_bytes(4, 4, &half_u32 as *const u32 as *const _);

        let grid = MTLSize::new(num_groups * THREADGROUP_SIZE, 1, 1);
        let tg = MTLSize::new(THREADGROUP_SIZE, 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        wait_and_check(&cmd, "sumcheck_reduce_triple");

        let data = partials.as_u32_slice_mut();
        let mut s0 = F::zero();
        let mut s1 = F::zero();
        let mut s2 = F::zero();
        let mut s3 = F::zero();
        for i in 0..num_groups as usize {
            s0 += F::from_canonical_u32(data[i * 4]);
            s1 += F::from_canonical_u32(data[i * 4 + 1]);
            s2 += F::from_canonical_u32(data[i * 4 + 2]);
            s3 += F::from_canonical_u32(data[i * 4 + 3]);
        }
        (s0, s1, s2, s3)
    }

    /// Batch modular inverse in-place: vals[i] = vals[i]^(p-2).
    pub fn batch_inverse(&self, vals: &mut GpuBuffer) {
        let ctx = MetalContext::get();
        let count: u32 = vals.len as u32;

        let cmd = ctx.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.batch_inverse_pipeline);
        enc.set_buffer(0, Some(&vals.buffer), 0);
        enc.set_bytes(1, 4, &count as *const u32 as *const _);

        let grid = MTLSize::new(vals.len as u64, 1, 1);
        let tg = MTLSize::new(THREADGROUP_SIZE.min(vals.len as u64), 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        wait_and_check(&cmd, "batch_inverse");
    }

    /// eq_evals butterfly: one level.
    pub fn eq_butterfly(&self, evals: &mut GpuBuffer, populated: usize, ri: F) {
        let ctx = MetalContext::get();
        let ri_val: u32 = ri.as_canonical_u32();
        let one_minus_ri: u32 = (F::one() - ri).as_canonical_u32();
        let pop_u32: u32 = populated as u32;

        let cmd = ctx.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.eq_butterfly_pipeline);
        enc.set_buffer(0, Some(&evals.buffer), 0);
        enc.set_bytes(1, 4, &pop_u32 as *const u32 as *const _);
        enc.set_bytes(2, 4, &ri_val as *const u32 as *const _);
        enc.set_bytes(3, 4, &one_minus_ri as *const u32 as *const _);

        let grid = MTLSize::new(populated as u64, 1, 1);
        let tg = MTLSize::new(THREADGROUP_SIZE.min(populated as u64), 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        wait_and_check(&cmd, "eq_butterfly");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_field_vec(n: usize) -> Vec<F> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| F::from_canonical_u32(rng.gen_range(0..((1u32 << 31) - 1))))
            .collect()
    }

    fn cpu_fold(data: &[F], r: F) -> Vec<F> {
        let half = data.len() / 2;
        let one_minus_r = F::one() - r;
        (0..half)
            .map(|j| one_minus_r * data[j] + r * data[j + half])
            .collect()
    }

    #[test]
    fn test_gpu_fold_matches_cpu() {
        let ctx = MetalContext::get();
        let kernels = MetalKernels::get();

        for log_n in [12, 14, 16, 18, 20] {
            let n = 1usize << log_n;
            let data = random_field_vec(n);
            let r = F::from_canonical_u32(12345);

            let cpu_result = cpu_fold(&data, r);

            let mut gpu_buf = GpuBuffer::from_field_slice(&ctx.device, &data);
            kernels.sumcheck_fold(&mut gpu_buf, n / 2, r);
            let gpu_result = gpu_buf.to_field_vec()[..n / 2].to_vec();

            assert_eq!(cpu_result, gpu_result, "fold mismatch at 2^{log_n}");
        }
    }

    #[test]
    fn test_gpu_reduce_product_matches_cpu() {
        let ctx = MetalContext::get();
        let kernels = MetalKernels::get();

        for log_n in [12, 14, 16, 18] {
            let n = 1usize << log_n;
            let f_data = random_field_vec(n);
            let g_data = random_field_vec(n);
            let half = n / 2;

            let mut cs0 = F::zero();
            let mut cs1 = F::zero();
            let mut cs2 = F::zero();
            for j in 0..half {
                let f0 = f_data[j]; let f1 = f_data[j + half];
                let g0 = g_data[j]; let g1 = g_data[j + half];
                cs0 += f0 * g0;
                cs1 += f1 * g1;
                let f2 = f1 + f1 - f0;
                let g2 = g1 + g1 - g0;
                cs2 += f2 * g2;
            }

            let f_buf = GpuBuffer::from_field_slice(&ctx.device, &f_data);
            let g_buf = GpuBuffer::from_field_slice(&ctx.device, &g_data);
            let (gs0, gs1, gs2) = kernels.sumcheck_reduce_product(&f_buf, &g_buf, half);

            assert_eq!((cs0, cs1, cs2), (gs0, gs1, gs2), "reduce product mismatch at 2^{log_n}");
        }
    }

    #[test]
    fn test_gpu_reduce_triple_matches_cpu() {
        let ctx = MetalContext::get();
        let kernels = MetalKernels::get();

        for log_n in [12, 14, 16] {
            let n = 1usize << log_n;
            let f_data = random_field_vec(n);
            let g_data = random_field_vec(n);
            let h_data = random_field_vec(n);
            let half = n / 2;

            let mut cs0 = F::zero();
            let mut cs1 = F::zero();
            let mut cs2 = F::zero();
            let mut cs3 = F::zero();
            for j in 0..half {
                let f0 = f_data[j]; let f1 = f_data[j + half];
                let g0 = g_data[j]; let g1 = g_data[j + half];
                let h0 = h_data[j]; let h1 = h_data[j + half];
                cs0 += f0 * g0 * h0;
                cs1 += f1 * g1 * h1;
                let f2 = f1 + f1 - f0;
                let g2 = g1 + g1 - g0;
                let h2 = h1 + h1 - h0;
                cs2 += f2 * g2 * h2;
                let three = F::from_canonical_u32(3);
                let two = F::two();
                cs3 += (three * f1 - two * f0) * (three * g1 - two * g0) * (three * h1 - two * h0);
            }

            let f_buf = GpuBuffer::from_field_slice(&ctx.device, &f_data);
            let g_buf = GpuBuffer::from_field_slice(&ctx.device, &g_data);
            let h_buf = GpuBuffer::from_field_slice(&ctx.device, &h_data);
            let (gs0, gs1, gs2, gs3) = kernels.sumcheck_reduce_triple(&f_buf, &g_buf, &h_buf, half);

            assert_eq!((cs0, cs1, cs2, cs3), (gs0, gs1, gs2, gs3), "reduce triple mismatch at 2^{log_n}");
        }
    }

    #[test]
    fn test_gpu_batch_inverse() {
        let ctx = MetalContext::get();
        let kernels = MetalKernels::get();

        let data: Vec<F> = (1..=1024).map(|v| F::from_canonical_u32(v)).collect();
        let mut buf = GpuBuffer::from_field_slice(&ctx.device, &data);
        kernels.batch_inverse(&mut buf);
        let result = buf.to_field_vec();

        for (i, (&orig, &inv)) in data.iter().zip(result.iter()).enumerate() {
            assert_eq!(orig * inv, F::one(), "inverse failed at index {i}");
        }
    }

    #[test]
    fn test_gpu_eq_evals_matches_cpu() {
        use crate::field::m31_ops::eq_evals;
        let ctx = MetalContext::get();
        let kernels = MetalKernels::get();

        let k = 14usize;
        let r: Vec<F> = (0..k).map(|i| F::from_canonical_u32(100 + i as u32)).collect();

        let cpu_result = eq_evals(&r);

        let n = 1 << k;
        let mut init = vec![F::one()];
        init.resize(n, F::zero());
        let mut buf = GpuBuffer::from_field_slice(&ctx.device, &init);

        let mut populated = 1usize;
        for i in 0..k {
            kernels.eq_butterfly(&mut buf, populated, r[i]);
            populated *= 2;
        }

        let gpu_result = buf.to_field_vec();
        assert_eq!(cpu_result, gpu_result, "eq_evals mismatch at k={k}");
    }
}
