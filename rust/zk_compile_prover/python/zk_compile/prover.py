"""Subprocess wrapper around the zk_compile_prover binary."""

import re
import subprocess
import shutil
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProofResult:
    """Result from a prove or prove-and-verify operation."""
    valid: bool
    setup_time_s: float
    prove_time_s: float
    verify_time_s: float
    proof_size_bytes: Optional[int]
    num_constraints: Optional[int]
    num_wires: Optional[int]
    proof_path: Optional[str]
    vk_path: Optional[str]


def _find_binary() -> str:
    """Find the zk_compile_prover binary.

    Search order:
    1. ZK_COMPILE_PROVER_BIN environment variable
    2. PATH (installed via maturin/pip -- maturin bin mode puts it on PATH)
    3. Cargo release build relative to this file (development mode)
    4. Cargo debug build relative to this file (development mode)
    """
    # 1. Env var override
    env_path = os.environ.get("ZK_COMPILE_PROVER_BIN")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. PATH (maturin bin mode installs the binary into the venv's bin/)
    on_path = shutil.which("zk_compile_prover")
    if on_path:
        return on_path

    # 3. Development: relative to this file (python/zk_compile/prover.py -> ../../target/release/)
    pkg_root = Path(__file__).resolve().parent.parent.parent
    for profile in ("release", "debug"):
        dev_path = pkg_root / "target" / profile / "zk_compile_prover"
        if dev_path.is_file():
            return str(dev_path)

    raise FileNotFoundError(
        "zk_compile_prover binary not found. "
        "Install via: pip install zk-compile, "
        "or build from source: cd rust/zk_compile_prover && cargo build --release"
    )


def prove(circuit_path: str, output_path: str = "proof.bin") -> ProofResult:
    """Generate a Groth16 proof for a circuit.

    Args:
        circuit_path: Path to the circuit JSON file.
        output_path: Path to write the proof binary (default: proof.bin).
            The verifying key is written to <output_path>.vk.

    Returns:
        ProofResult with timing and file paths.
    """
    binary = _find_binary()
    result = subprocess.run(
        [binary, "prove", circuit_path, "--output", output_path],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Proving failed:\n{result.stdout}\n{result.stderr}")

    parsed = _parse_output(result.stdout)
    parsed.proof_path = output_path
    parsed.vk_path = f"{output_path}.vk"
    return parsed


def verify(circuit_path: str, proof_path: str = "proof.bin") -> bool:
    """Verify a Groth16 proof.

    Args:
        circuit_path: Path to the circuit JSON file.
        proof_path: Path to the proof binary (default: proof.bin).
            Expects the verifying key at <proof_path>.vk.

    Returns:
        True if the proof is valid, False otherwise.
    """
    binary = _find_binary()
    result = subprocess.run(
        [binary, "verify", circuit_path, "--proof", proof_path],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return False

    # Check output for VALID/INVALID
    return "VALID" in result.stdout and "INVALID" not in result.stdout


def prove_and_verify(circuit_path: str) -> ProofResult:
    """Generate and verify a proof in one step.

    This is the most common operation for benchmarking. It runs setup,
    prove, and verify sequentially, printing a summary at the end.

    Args:
        circuit_path: Path to the circuit JSON file.

    Returns:
        ProofResult with all timing data.
    """
    binary = _find_binary()
    result = subprocess.run(
        [binary, "prove-and-verify", circuit_path],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Prove-and-verify failed:\n{result.stdout}\n{result.stderr}"
        )

    return _parse_output(result.stdout)


def _parse_output(stdout: str) -> ProofResult:
    """Parse prover stdout into a ProofResult.

    The binary prints timing in seconds:
        Setup: 0.990s
        Prove: 0.870s
        Verify: 0.0006s -- VALID
    And a summary block:
        Constraints: 200705
        Wires: 201490
        Proof size: 128 bytes
    """
    setup_time = 0.0
    prove_time = 0.0
    verify_time = 0.0
    valid = False
    num_constraints = None
    num_wires = None
    proof_size = None

    for line in stdout.splitlines():
        # Setup: 0.990s
        m = re.search(r'Setup:\s*([\d.]+)s', line)
        if m:
            setup_time = float(m.group(1))

        # Prove: 0.870s
        m = re.search(r'Prove:\s*([\d.]+)s', line)
        if m:
            prove_time = float(m.group(1))

        # Verify: 0.0006s -- VALID / INVALID
        m = re.search(r'Verify:\s*([\d.]+)s', line)
        if m:
            verify_time = float(m.group(1))

        if "VALID" in line and "INVALID" not in line:
            valid = True

        # Constraints: 200705
        m = re.search(r'Constraints:\s*(\d+)', line)
        if m:
            num_constraints = int(m.group(1))

        # Wires: 201490
        m = re.search(r'Wires:\s*(\d+)', line)
        if m:
            num_wires = int(m.group(1))

        # Proof size: 128 bytes
        m = re.search(r'Proof size:\s*(\d+)\s*bytes', line)
        if m:
            proof_size = int(m.group(1))

    return ProofResult(
        valid=valid,
        setup_time_s=setup_time,
        prove_time_s=prove_time,
        verify_time_s=verify_time,
        proof_size_bytes=proof_size,
        num_constraints=num_constraints,
        num_wires=num_wires,
        proof_path=None,
        vk_path=None,
    )
