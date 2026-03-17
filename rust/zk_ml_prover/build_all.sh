#!/bin/bash
set -e

echo "Building default binary..."
cargo build --release
cp target/release/zk_ml_prover target/release/zk_ml_prover_default

echo "Building PCS (basefold w_partial) binary..."
cargo build --release --features pcs
cp target/release/zk_ml_prover target/release/zk_ml_prover_pcs

echo "Building PCS-Full (basefold full W) binary..."
cargo build --release --features pcs-full
cp target/release/zk_ml_prover target/release/zk_ml_prover_pcs_full

# Restore default as the main binary
cp target/release/zk_ml_prover_default target/release/zk_ml_prover

echo "Done. Binaries:"
ls -lh target/release/zk_ml_prover target/release/zk_ml_prover_pcs target/release/zk_ml_prover_pcs_full
