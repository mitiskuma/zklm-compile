#!/usr/bin/env python3
"""
P10-1: unified Dense-4M head-to-head benchmark.

Runs ZK-Compile (this repo) and EZKL on the same Dense-4M MLP, on the
same hardware, in the same Python env. Emits one CSV row per system
into `bench/results/compare.csv` so anyone can clone, run, and verify
the ratios in the README.

Usage:
    pip install ezkl onnx torch numpy
    cd zklm-compile
    cargo build --release
    python bench/compare.py

The script prints a side-by-side summary at the end. Numbers are
machine-specific — they match the README rows when run on Apple M4 Max.
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "bench" / "work"
RESULTS = REPO / "bench" / "results"
WORK.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def run_zkcompile() -> dict:
    """Invoke the existing python/benchmark_dense4m.py and parse its JSON."""
    print("[zk-compile] running python/benchmark_dense4m.py")
    t = time.time()
    proc = subprocess.run(
        [sys.executable, str(REPO / "python" / "benchmark_dense4m.py")],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - t
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("zk-compile benchmark failed")

    # Parse the relevant lines from stdout. The script prints the timings
    # in a fixed format under "[Default (Fiat-Shamir)] Results:".
    out = proc.stdout
    prove_ms = _parse_line(out, "Prove:", "ms")
    verify_ms = _parse_line(out, "Verify:", "ms")
    proof_bytes = _parse_line(out, "Proof size:", "bytes", strip_commas=True)
    print(f"  zk-compile: {prove_ms:.1f} ms prove / "
          f"{verify_ms:.3f} ms verify / {int(proof_bytes):,} bytes "
          f"(harness wallclock {elapsed:.1f}s)")
    return {
        "system": "zk-compile",
        "model": "dense_4m",
        "version": "@HEAD",
        "setup_ms": 0.0,  # Fiat-Shamir, no trusted setup
        "prove_ms": prove_ms,
        "verify_ms": verify_ms,
        "proof_size_bytes": int(proof_bytes),
        "soundness_bits_per_round": 124,
    }


def _parse_line(text: str, key: str, unit: str, strip_commas: bool = False) -> float:
    """Extract the numeric value following `key` in `text`. Used because
    benchmark_dense4m.py prints free-form ASCII tables."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(key):
            tail = line[len(key):].strip()
            num_str = tail.split()[0]
            if strip_commas:
                num_str = num_str.replace(",", "")
            return float(num_str)
    raise RuntimeError(f"Could not find {key!r} in zk-compile output")


def run_ezkl() -> dict:
    """Invoke bench/run_ezkl.py and read the CSV row it emits."""
    print("[ezkl] running bench/run_ezkl.py")
    t = time.time()
    proc = subprocess.run(
        [sys.executable, str(REPO / "bench" / "run_ezkl.py")],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    elapsed = time.time() - t
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("ezkl benchmark failed")

    csv_path = RESULTS / "ezkl_dense4m.csv"
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    assert rows, "ezkl_dense4m.csv was empty"
    r = rows[-1]
    prove_ms = float(r["prove_ms"])
    verify_ms = float(r["verify_ms"])
    proof_bytes = int(r["proof_size_bytes"])
    print(f"  ezkl: {prove_ms:.1f} ms prove / "
          f"{verify_ms:.1f} ms verify / {proof_bytes:,} bytes "
          f"(harness wallclock {elapsed:.1f}s)")
    return {
        "system": "ezkl",
        "model": "dense_4m",
        "version": f"ezkl=={r.get('ezkl_version', 'unknown')}",
        "setup_ms": float(r["setup_ms"]),
        "prove_ms": prove_ms,
        "verify_ms": verify_ms,
        "proof_size_bytes": proof_bytes,
        "soundness_bits_per_round": 128,  # KZG/Halo2 ~128 bit security
    }


def main():
    rows = []
    rows.append(run_zkcompile())
    rows.append(run_ezkl())

    fieldnames = [
        "system", "model", "version",
        "setup_ms", "prove_ms", "verify_ms", "proof_size_bytes",
        "soundness_bits_per_round",
    ]
    csv_path = RESULTS / "compare.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n  → {csv_path}")

    # Side-by-side summary
    print()
    print("=" * 72)
    print(f"  Dense-4M MLP head-to-head (Apple M4 Max, identical workload)")
    print("=" * 72)
    print(f"  {'System':<14} {'Setup':>11} {'Prove':>11} {'Verify':>11} {'Proof':>11}")
    for r in rows:
        prove_str = f"{r['prove_ms']:>9.1f} ms"
        verify_str = f"{r['verify_ms']:>9.3f} ms"
        proof_str = f"{r['proof_size_bytes']:>9,} B"
        setup_str = (f"{r['setup_ms']:>9.0f} ms" if r['setup_ms']
                     else f"{'(none)':>11}")
        print(f"  {r['system']:<14}{setup_str:>11}{prove_str:>11}{verify_str:>11}{proof_str:>11}")

    if len(rows) == 2:
        zk = next(r for r in rows if r["system"] == "zk-compile")
        ez = next(r for r in rows if r["system"] == "ezkl")
        print("\n  Ratios (zk-compile vs ezkl):")
        print(f"    Prove:      {ez['prove_ms'] / zk['prove_ms']:>8.0f}× faster")
        print(f"    Verify:     {ez['verify_ms'] / zk['verify_ms']:>8.0f}× faster")
        print(f"    Proof:      {ez['proof_size_bytes'] / zk['proof_size_bytes']:>8.0f}× smaller")


if __name__ == "__main__":
    main()
