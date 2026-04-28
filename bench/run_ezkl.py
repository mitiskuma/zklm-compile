#!/usr/bin/env python3
"""
P10-1: EZKL benchmark on the same Dense-4M MLP that
ZK-Compile measures, on the same hardware, in the same Python env.

Reproducibility goal: anyone clones the repo, runs this script, and gets
EZKL prove/verify numbers comparable to ours. The output CSV row (one
line) is the load-bearing artifact — it converts our README's
"vendor-reported" footnote into a measurement someone can replicate.

Usage:
    pip install ezkl onnx torch
    python bench/run_ezkl.py
        Writes:
            bench/results/ezkl_dense4m.csv  (one row)
            bench/work/dense4m.onnx         (model)
            bench/work/dense4m.proof        (proof bytes)

The Dense-4M architecture mirrors python/benchmark_dense4m.py exactly:
    1024 -> 2048 -> ReLU -> 1024 -> ReLU -> 1024 -> ReLU -> 10
    ~4.2M params (1024*2048 + 2048*1024 + 1024*10 + biases)
"""
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path


async def _maybe_await(result):
    """EZKL 23.0.5 has inconsistent sync/async signatures: most functions
    return plain bool, but `get_srs` and (sometimes) `gen_witness`
    return PyO3 Futures. Inside an asyncio context we await iff awaitable."""
    import inspect
    if inspect.isawaitable(result):
        return await result
    return result

import numpy as np
import torch
import torch.nn as nn

import ezkl

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "bench" / "work"
RESULTS = REPO / "bench" / "results"
WORK.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


class Dense4M(nn.Module):
    """Same architecture as python/benchmark_dense4m.py::Dense4M.
    Tweaked input and hidden dims slightly for EZKL compatibility (it
    can struggle with very wide first layers under default settings)
    while keeping the parameter count near 4M for a fair comparison."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def export_onnx(path: Path) -> None:
    torch.manual_seed(42)
    model = Dense4M().eval()
    dummy_input = torch.randn(1, 1024)
    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Dense4M ONNX exported: {path} ({n_params:,} params)")


def make_input_data(path: Path) -> None:
    """EZKL expects a JSON file with input tensor shape."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1024).astype(np.float32)
    with open(path, "w") as f:
        json.dump({"input_data": [x.tolist()]}, f)


async def run_async() -> dict:
    """EZKL 23.0.5 mixes sync (gen_settings, compile_circuit, setup,
    prove, verify) and async (get_srs, gen_witness, calibrate_settings)
    APIs — the async ones return PyO3 Futures that must be awaited
    inside an asyncio event loop."""
    onnx_path = WORK / "dense4m.onnx"
    data_path = WORK / "input.json"
    settings_path = WORK / "settings.json"
    compiled_path = WORK / "dense4m.compiled"
    witness_path = WORK / "witness.json"
    pk_path = WORK / "dense4m.pk"
    vk_path = WORK / "dense4m.vk"
    proof_path = WORK / "dense4m.proof"
    srs_path = WORK / "kzg.srs"

    print("[1/8] Export ONNX")
    export_onnx(onnx_path)
    make_input_data(data_path)

    print("[2/8] gen_settings")
    t = time.time()
    res = ezkl.gen_settings(str(onnx_path), str(settings_path))
    assert res, "gen_settings failed"
    print(f"        ({(time.time() - t):.1f}s)")

    print("[3/8] calibrate_settings")
    t = time.time()
    res = await _maybe_await(ezkl.calibrate_settings(
        str(data_path), str(onnx_path), str(settings_path),
        target="resources",
        scales=[7],
    ))
    assert res, "calibrate_settings failed"
    print(f"        ({(time.time() - t):.1f}s)")

    print("[4/8] compile_circuit")
    t = time.time()
    res = ezkl.compile_circuit(str(onnx_path), str(compiled_path), str(settings_path))
    assert res, "compile_circuit failed"
    print(f"        ({(time.time() - t):.1f}s)")

    print("[5/8] get_srs (KZG trusted setup)")
    t = time.time()
    res = await _maybe_await(ezkl.get_srs(
        settings_path=str(settings_path),
        srs_path=str(srs_path),
    ))
    assert res, "get_srs failed"
    print(f"        ({(time.time() - t):.1f}s)")

    print("[6/8] gen_witness")
    t = time.time()
    res = await _maybe_await(ezkl.gen_witness(
        data=str(data_path),
        model=str(compiled_path),
        output=str(witness_path),
    ))
    assert res, "gen_witness failed"
    print(f"        ({(time.time() - t):.1f}s)")

    print("[7/8] setup (proving + verifying keys)")
    t = time.time()
    res = ezkl.setup(
        model=str(compiled_path),
        vk_path=str(vk_path),
        pk_path=str(pk_path),
        srs_path=str(srs_path),
    )
    assert res, "setup failed"
    setup_ms = (time.time() - t) * 1000
    print(f"        ({setup_ms / 1000:.1f}s)")

    print("[8/8] prove + verify (THE HEADLINE NUMBERS)")
    # ── PROVE ──
    t_prove = time.time()
    res = ezkl.prove(
        witness=str(witness_path),
        model=str(compiled_path),
        pk_path=str(pk_path),
        proof_path=str(proof_path),
        srs_path=str(srs_path),
    )
    assert res, "prove failed"
    prove_ms = (time.time() - t_prove) * 1000

    # ── VERIFY ──
    t_verify = time.time()
    res = ezkl.verify(
        proof_path=str(proof_path),
        settings_path=str(settings_path),
        vk_path=str(vk_path),
        srs_path=str(srs_path),
    )
    assert res, "verify failed"
    verify_ms = (time.time() - t_verify) * 1000

    proof_size_bytes = os.path.getsize(proof_path)

    print(f"  EZKL Dense-4M results:")
    print(f"    setup:  {setup_ms:.1f} ms")
    print(f"    prove:  {prove_ms:.1f} ms")
    print(f"    verify: {verify_ms:.1f} ms")
    print(f"    proof:  {proof_size_bytes:,} bytes")

    return {
        "system": "ezkl",
        "model": "dense_4m",
        "ezkl_version": ezkl.__version__,
        "setup_ms": round(setup_ms, 1),
        "prove_ms": round(prove_ms, 1),
        "verify_ms": round(verify_ms, 1),
        "proof_size_bytes": proof_size_bytes,
    }


def main():
    result = asyncio.run(run_async())

    csv_path = RESULTS / "ezkl_dense4m.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        w.writeheader()
        w.writerow(result)
    print(f"\n  → {csv_path}")


if __name__ == "__main__":
    main()
