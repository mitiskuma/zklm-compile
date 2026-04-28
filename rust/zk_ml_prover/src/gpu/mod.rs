//! Metal GPU acceleration for sumcheck operations over M31.

pub mod metal_device;
pub mod metal_buffer;
pub mod metal_kernels;

/// Below this half-size, CPU (rayon) is faster than GPU dispatch overhead (~5μs).
pub const GPU_THRESHOLD: usize = 16384;

pub use metal_device::{MetalContext, wait_and_check};
pub use metal_buffer::GpuBuffer;
pub use metal_kernels::MetalKernels;
