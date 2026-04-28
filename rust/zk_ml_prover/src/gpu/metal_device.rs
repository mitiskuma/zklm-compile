//! Metal device context: singleton Device + CommandQueue + compiled shader library.

use metal::{Device, CommandQueue, Library, CommandBufferRef, MTLCommandBufferStatus};
use std::sync::OnceLock;

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub library: Library,
}

/// SAFETY (E5): `MetalContext` is wrapped in a `OnceLock` global and only
/// constructed once (`MetalContext::get`). The Metal-rs `Device`, `CommandQueue`,
/// and `Library` types wrap Objective-C objects whose underlying Metal API is
/// internally thread-safe per Apple's Metal documentation: a `CommandQueue`
/// can be used to enqueue work from multiple threads concurrently. The
/// `unsafe impl Send/Sync` is therefore an explicit contract that we use these
/// types only via Apple-blessed APIs (no mutation of internal state) and never
/// hand out a `&mut MetalContext`. Reviewer flagged the previous code as
/// silently relying on this; the comment now documents the invariant.
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

/// SAFETY (E5): wait for `cmd` to complete and verify it actually completed
/// successfully (status == `Completed`). Apple's Metal API does NOT panic on
/// kernel error / timeout / aborted status — the buffer's `status()` reflects
/// the failure but the host blissfully continues with garbage in the GPU
/// buffers. For ZK proving that's a soundness hazard: a corrupted intermediate
/// product in a sumcheck round would still produce a "valid-looking" proof.
/// This helper turns the silent failure into a panic. Tests that don't have
/// Metal access can still be exercised via the non-GPU path.
///
/// `ctx` is included in the panic message so failures are traceable to the
/// originating kernel without a debugger.
#[track_caller]
pub fn wait_and_check(cmd: &CommandBufferRef, ctx: &'static str) {
    cmd.wait_until_completed();
    let status = cmd.status();
    if status != MTLCommandBufferStatus::Completed {
        panic!(
            "MetalCommandBuffer at {} did not complete: status={:?}",
            ctx, status,
        );
    }
}

static INSTANCE: OnceLock<MetalContext> = OnceLock::new();

impl MetalContext {
    pub fn get() -> &'static MetalContext {
        INSTANCE.get_or_init(|| {
            let device = Device::system_default()
                .expect("No Metal device found");

            let queue = device.new_command_queue();

            // R7: embed the metallib bytes directly into
            // the binary instead of reading them from `OUT_DIR` at runtime.
            // The previous `new_library_with_file(format!("{}/m31.metallib",
            // env!("OUT_DIR")))` fixed the path at build time, so moving the
            // compiled binary to another machine (or a docker layer that
            // didn't include the same `target/` directory) caused
            // "Failed to load m31.metallib" at first GPU dispatch.
            // `include_bytes!` evaluates relative to this source file and
            // resolves to the build-time `OUT_DIR`, so the bytes get baked
            // into the executable. Runtime is then a single
            // `new_library_with_data(&BYTES)` call — no filesystem
            // dependency.
            const METALLIB_BYTES: &[u8] = include_bytes!(
                concat!(env!("OUT_DIR"), "/m31.metallib")
            );
            let library = device.new_library_with_data(METALLIB_BYTES)
                .expect("Failed to load embedded m31.metallib");

            MetalContext { device, queue, library }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_device_init() {
        let ctx = MetalContext::get();
        eprintln!("Metal device: {}", ctx.device.name());
    }

    /// SAFETY regression (E5): an empty (no encoder) command buffer that's
    /// committed cleanly must complete with status `Completed`, so
    /// `wait_and_check` returns without panicking. If `wait_and_check`
    /// is ever rewired to skip the status check, this test would still
    /// pass — but if the helper is rewired to incorrectly *fail* on a
    /// successful buffer (e.g. comparing the wrong enum variant), it'd
    /// catch that.
    #[test]
    fn test_wait_and_check_completes_empty_buffer() {
        let ctx = MetalContext::get();
        let cmd = ctx.queue.new_command_buffer();
        cmd.commit();
        wait_and_check(&cmd, "test_wait_and_check_completes_empty_buffer");
    }
}
