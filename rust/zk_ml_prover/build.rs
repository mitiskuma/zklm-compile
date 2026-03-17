use std::process::Command;

fn main() {
    if std::env::var("CARGO_FEATURE_METAL_GPU").is_ok() {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let shader_src = "src/gpu/shaders/m31.metal";

        println!("cargo:rerun-if-changed={shader_src}");

        // Compile .metal → .air
        let air_path = format!("{out_dir}/m31.air");
        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c", shader_src, "-o", &air_path])
            .status()
            .expect("failed to run xcrun metal");
        assert!(status.success(), "metal shader compilation failed");

        // Link .air → .metallib
        let lib_path = format!("{out_dir}/m31.metallib");
        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib", &air_path, "-o", &lib_path])
            .status()
            .expect("failed to run xcrun metallib");
        assert!(status.success(), "metallib linking failed");
    }
}
