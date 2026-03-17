//! GPU buffer wrapper for zero-copy M31 field element arrays.

use metal::{Buffer, Device, MTLResourceOptions};
use p3_field::{AbstractField, PrimeField32};
use p3_mersenne_31::Mersenne31;

type F = Mersenne31;

pub struct GpuBuffer {
    pub(crate) buffer: Buffer,
    /// Number of u32 elements
    pub(crate) len: usize,
}

impl GpuBuffer {
    /// Allocate a shared-memory buffer and copy field elements into it.
    pub fn from_field_slice(device: &Device, data: &[F]) -> Self {
        let byte_len = data.len() * 4;
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        GpuBuffer { buffer, len: data.len() }
    }

    /// Allocate an uninitialized shared-memory buffer for `count` u32 elements.
    pub fn new_uninit(device: &Device, count: usize) -> Self {
        let byte_len = (count * 4) as u64;
        let buffer = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
        GpuBuffer { buffer, len: count }
    }

    /// Get a mutable u32 slice of the buffer contents (zero-copy on unified memory).
    pub fn as_u32_slice_mut(&mut self) -> &mut [u32] {
        let ptr = self.buffer.contents() as *mut u32;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }
    }

    /// Get an immutable u32 slice.
    pub fn as_u32_slice(&self) -> &[u32] {
        let ptr = self.buffer.contents() as *const u32;
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    /// Read buffer contents back as Vec<F>.
    pub fn to_field_vec(&self) -> Vec<F> {
        self.as_u32_slice()
            .iter()
            .map(|&v| F::from_canonical_u32(v))
            .collect()
    }
}
