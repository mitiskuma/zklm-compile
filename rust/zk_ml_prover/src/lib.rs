//! Library re-exports for integration tests.
//!
//! `zk_ml_prover` is primarily a binary crate (`src/main.rs`). The Phase 11
//! integration tests under `tests/` need access to the prover's internal
//! types (`QwenLayerProof`, `prove_qwen_layer_with_trace`, etc.). Cargo
//! does NOT expose binary-crate internals to integration tests, so we ship
//! a thin `lib.rs` that re-declares the same module tree. The compilation
//! cost is that each module is built twice (once for the bin, once for the
//! lib); this is acceptable for our test surface and avoids restructuring
//! the binary.
//!
//! `main.rs` retains its own `mod` declarations and is unchanged.

pub mod field;
pub mod proving;
#[cfg(feature = "metal_gpu")]
pub mod gpu;
pub mod pipeline;
pub mod protocol;
pub mod server;
pub mod transformer;
pub mod verification;
