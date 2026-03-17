//! Metal device context: singleton Device + CommandQueue + compiled shader library.

use metal::{Device, CommandQueue, Library};
use std::sync::OnceLock;

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub library: Library,
}

unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

static INSTANCE: OnceLock<MetalContext> = OnceLock::new();

impl MetalContext {
    pub fn get() -> &'static MetalContext {
        INSTANCE.get_or_init(|| {
            let device = Device::system_default()
                .expect("No Metal device found");

            let queue = device.new_command_queue();

            let metallib_path = format!("{}/m31.metallib", env!("OUT_DIR"));
            let library = device.new_library_with_file(metallib_path)
                .expect("Failed to load m31.metallib");

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
}
