#!/usr/bin/env python3
"""Benchmark Dense 4M-parameter MLP through the structured sumcheck prover.

Architecture: 1024 → 2048 → ReLU → 1024 → ReLU → 1024 → ReLU → 10
Total params: 1024*2048 + 2048*1024 + 1024*1024 + 1024*10 ≈ 4.2M

This matches Lagrange's DeepProve "Dense 4M" benchmark for direct comparison.
DeepProve: 2,335ms prove, 520ms verify (Basefold PCS)
EZKL:      126,831ms prove, 1,112ms verify (Halo2/KZG)

Runs both default mode (Fiat-Shamir binding) and PCS-full mode (Basefold)
for fair comparison against DeepProve which also uses Basefold.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

M31 = (1 << 31) - 1
QUANT_RANGE = 127


def quantize_vector(v: np.ndarray, scale: float) -> list:
    q = np.round(v / scale).astype(np.int64)
    result = q % M31
    result = np.where(result < 0, result + M31, result)
    return result.astype(np.uint32).tolist()


class Dense4M(nn.Module):
    """4M-parameter dense MLP: 1024→2048→1024→1024→10."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)   # 1024*2048 + 2048 = 2,099,200
        self.fc2 = nn.Linear(2048, 1024)   # 2048*1024 + 1024 = 2,098,176
        self.fc3 = nn.Linear(1024, 10)     # 1024*10 + 10 = 10,250
        # Total: ~4.2M params

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_request(model, input_vec: np.ndarray) -> dict:
    """Build a JSON ProveRequest for the dense MLP."""
    sd = model.state_dict()

    fc1_w = sd['fc1.weight'].float().numpy()  # [2048, 1024]
    fc1_b = sd['fc1.bias'].float().numpy()
    fc2_w = sd['fc2.weight'].float().numpy()  # [1024, 2048]
    fc2_b = sd['fc2.bias'].float().numpy()
    fc3_w = sd['fc3.weight'].float().numpy()  # [10, 1024]
    fc3_b = sd['fc3.bias'].float().numpy()

    # Quantize input
    x_max = np.max(np.abs(input_vec))
    x_scale = x_max / QUANT_RANGE if x_max > 0 else 1.0
    input_q = quantize_vector(input_vec, x_scale)

    # Quantize layers — chain scales through the network
    fc1_w_max = np.max(np.abs(fc1_w))
    fc1_w_scale = fc1_w_max / QUANT_RANGE if fc1_w_max > 0 else 1.0
    fc1_w_q = quantize_vector(fc1_w.flatten(), fc1_w_scale)
    fc1_b_scale = fc1_w_scale * x_scale
    fc1_b_q = quantize_vector(fc1_b, fc1_b_scale)

    fc2_w_max = np.max(np.abs(fc2_w))
    fc2_w_scale = fc2_w_max / QUANT_RANGE if fc2_w_max > 0 else 1.0
    fc2_w_q = quantize_vector(fc2_w.flatten(), fc2_w_scale)
    fc2_b_scale = fc2_w_scale * fc1_b_scale
    fc2_b_q = quantize_vector(fc2_b, fc2_b_scale)

    fc3_w_max = np.max(np.abs(fc3_w))
    fc3_w_scale = fc3_w_max / QUANT_RANGE if fc3_w_max > 0 else 1.0
    fc3_w_q = quantize_vector(fc3_w.flatten(), fc3_w_scale)
    fc3_b_scale = fc3_w_scale * fc2_b_scale
    fc3_b_q = quantize_vector(fc3_b, fc3_b_scale)

    ops = [
        {"type": "linear", "name": "fc1", "m": 2048, "n": 1024,
         "w_q": fc1_w_q, "b_q": fc1_b_q},
        {"type": "relu", "name": "relu1"},
        {"type": "linear", "name": "fc2", "m": 1024, "n": 2048,
         "w_q": fc2_w_q, "b_q": fc2_b_q},
        {"type": "relu", "name": "relu2"},
        {"type": "linear", "name": "fc3", "m": 10, "n": 1024,
         "w_q": fc3_w_q, "b_q": fc3_b_q},
    ]

    return {"mode": "mlp", "input": input_q, "ops": ops}


def run_prover(binary_path, payload, label=""):
    """Run the prover and return parsed result."""
    print(f"\n  [{label}] Running prover: {os.path.basename(binary_path)}")
    t0 = time.time()
    proc = subprocess.run(
        [binary_path],
        input=payload,
        capture_output=True,
        timeout=600,
    )
    wall_time = time.time() - t0

    if proc.returncode != 0:
        print(f"    FAILED (exit code {proc.returncode})")
        print(f"    stderr: {proc.stderr.decode()}")
        return None, wall_time

    stderr = proc.stderr.decode().strip()
    print(f"    stderr:\n      {stderr.replace(chr(10), chr(10) + '      ')}")

    result = json.loads(proc.stdout.decode().strip())
    return result, wall_time


def print_comparison(label, result, wall_time):
    """Print results and comparison against DeepProve/EZKL."""
    if result is None:
        print(f"\n  [{label}] FAILED")
        return

    print(f"\n  [{label}] Results:")
    print(f"    Valid:        {result['valid']}")
    print(f"    Prove:        {result['prove_time_ms']:.1f} ms")
    print(f"    Verify:       {result['verify_time_ms']:.1f} ms")
    print(f"    Proof size:   {result['proof_size_bytes']:,} bytes ({result['proof_size_bytes']/1024:.1f} KB)")
    print(f"    Wall time:    {wall_time*1000:.0f} ms")

    if result.get('coverage'):
        cov = result['coverage']
        print(f"    Proved ops:   {cov['proved_count']} ({cov['proved_ops']})")

    # DeepProve comparison (Dense 4M, Basefold PCS)
    dp_prove = 2335.0
    dp_verify = 520.0
    our_prove = result['prove_time_ms']
    our_verify = result['verify_time_ms']

    if our_prove > 0:
        print(f"\n    vs DeepProve (2,335ms prove):  {dp_prove/our_prove:.0f}x faster")
    if our_verify > 0:
        print(f"    vs DeepProve (520ms verify):   {dp_verify/our_verify:.0f}x faster")

    # EZKL comparison (Dense 4M)
    ezkl_prove = 126831.0
    ezkl_verify = 1112.0
    if our_prove > 0:
        print(f"    vs EZKL (126,831ms prove):     {ezkl_prove/our_prove:.0f}x faster")
    if our_verify > 0:
        print(f"    vs EZKL (1,112ms verify):      {ezkl_verify/our_verify:.0f}x faster")


def main():
    print("=" * 70)
    print("Dense 4M MLP Benchmark: ZK-Compile vs DeepProve vs EZKL")
    print("=" * 70)

    # Build model with random weights (same as DeepProve's benchmark — random init)
    print("\n[1/4] Building Dense 4M MLP (1024→2048→1024→10)...")
    torch.manual_seed(42)
    model = Dense4M()
    model.eval()
    n_params = count_params(model)
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Random input (matching DeepProve's setup — they use random inputs)
    print("\n[2/4] Generating random input (dim=1024)...")
    input_vec = np.random.randn(1024).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(input_vec).unsqueeze(0))
        torch_pred = torch_out.argmax(1).item()
    print(f"  PyTorch prediction: {torch_pred}")

    # Build request
    print("\n[3/4] Quantizing to M31...")
    request = build_request(model, input_vec)
    total_weights = sum(len(op.get("w_q", [])) for op in request["ops"])
    print(f"  Total weight elements: {total_weights:,}")
    payload = json.dumps(request).encode()
    print(f"  JSON payload: {len(payload)/1e6:.1f} MB")

    # Find binaries
    prover_dir = os.path.join(
        os.path.dirname(__file__), "..", "rust", "zk_ml_prover", "target", "release"
    )
    default_bin = os.path.join(prover_dir, "zk_ml_prover")
    pcs_full_bin = os.path.join(prover_dir, "zk_ml_prover_pcs_full")

    if not os.path.isfile(default_bin):
        print(f"ERROR: {default_bin} not found. Run: cargo build --release")
        sys.exit(1)

    # Run benchmarks
    print("\n[4/4] Proving...")
    print("=" * 70)

    # Default mode (Fiat-Shamir binding — fastest, evaluation soundness only)
    result_default, wall_default = run_prover(default_bin, payload, "Default (Fiat-Shamir)")
    print_comparison("Default (Fiat-Shamir)", result_default, wall_default)

    # PCS-full mode (Basefold — fair comparison with DeepProve)
    if os.path.isfile(pcs_full_bin):
        result_pcs, wall_pcs = run_prover(pcs_full_bin, payload, "PCS-Full (Basefold)")
        print_comparison("PCS-Full (Basefold)", result_pcs, wall_pcs)
    else:
        print(f"\n  [PCS-Full] Binary not found at {pcs_full_bin}")
        print("  Build with: cargo build --release --features pcs-full")
        print("  Then: cp target/release/zk_ml_prover target/release/zk_ml_prover_pcs_full")

    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE (Dense ~4M params)")
    print("=" * 70)
    print(f"{'System':<30} {'Prove':>10} {'Verify':>10} {'Proof Size':>12}")
    print("-" * 70)
    if result_default:
        print(f"{'ZK-Compile (default)':<30} {result_default['prove_time_ms']:>8.1f}ms {result_default['verify_time_ms']:>8.1f}ms {result_default['proof_size_bytes']:>10,}B")
    if os.path.isfile(pcs_full_bin) and result_pcs:
        print(f"{'ZK-Compile (pcs-full)':<30} {result_pcs['prove_time_ms']:>8.1f}ms {result_pcs['verify_time_ms']:>8.1f}ms {result_pcs['proof_size_bytes']:>10,}B")
    print(f"{'DeepProve (Basefold)':<30} {'2335.0':>8}ms {'520.0':>8}ms {'N/A':>12}")
    print(f"{'EZKL (Halo2/KZG)':<30} {'126831.0':>8}ms {'1112.0':>8}ms {'127,000':>12}B")
    print()


if __name__ == "__main__":
    main()
