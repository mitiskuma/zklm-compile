#!/usr/bin/env bash
# R8 (funding-readiness): five-minute smoke test for a fresh checkout.
#
# Builds zk_compile_prover, runs `prove` then `verify` against the bundled
# `square_circuit.json` example (proves x*x = 25, public input 25, private
# witness x = 5 over BN254), and prints the timings.
#
# Exits non-zero on any failure so this can drive CI / readiness checks.

set -euo pipefail

# Resolve repo root regardless of where the script is invoked from.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd )"

CIRCUIT="$SCRIPT_DIR/square_circuit.json"
PROOF_PATH="${PROOF_PATH:-/tmp/zk_compile_quickstart_proof.bin}"

cd "$REPO_ROOT/rust/zk_compile_prover"

echo "[1/3] cargo build --release"
cargo build --release --quiet

BIN="$REPO_ROOT/rust/zk_compile_prover/target/release/zk_compile_prover"
if [ ! -x "$BIN" ]; then
    echo "ERROR: prover binary not found at $BIN" >&2
    exit 2
fi

echo "[2/3] prove $CIRCUIT"
"$BIN" prove "$CIRCUIT" -o "$PROOF_PATH"

echo "[3/3] verify $PROOF_PATH"
"$BIN" verify "$CIRCUIT" -p "$PROOF_PATH"

echo
echo "OK — quickstart proves and verifies. Proof: $PROOF_PATH"
