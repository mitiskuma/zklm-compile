#!/usr/bin/env python3
"""Benchmark MNIST MLP (784→128→10) through the structured sumcheck prover.

Trains a simple MNIST MLP, quantizes weights to M31, sends through the Rust
prover in MLP mode (linear + relu ops), and reports prove/verify times.

Comparison target: EZKL takes ~1,310s on the same model.
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


class MNISTNet(nn.Module):
    """Simple 784→128→ReLU→10 MLP."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_mnist(epochs=3):
    """Train the MNIST MLP. Returns trained model."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('/tmp/mnist_data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    model = MNISTNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = F.cross_entropy(out, batch_y)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)
        print(f"  Epoch {epoch+1}/{epochs}: accuracy {correct/total:.1%}")

    return model


def get_test_sample(model):
    """Get a single MNIST test sample and its prediction."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = datasets.MNIST('/tmp/mnist_data', train=False, download=True, transform=transform)
    x, label = test_data[0]
    with torch.no_grad():
        pred = model(x.unsqueeze(0)).argmax(1).item()
    return x.view(-1).numpy(), label, pred


def build_mnist_request(model, input_vec: np.ndarray) -> dict:
    """Build a JSON ProveRequest for the MNIST MLP."""
    sd = model.state_dict()

    # Extract weights and biases
    fc1_w = sd['fc1.weight'].float().numpy()  # [128, 784]
    fc1_b = sd['fc1.bias'].float().numpy()    # [128]
    fc2_w = sd['fc2.weight'].float().numpy()  # [10, 128]
    fc2_b = sd['fc2.bias'].float().numpy()    # [10]

    # Quantize input
    x_max = np.max(np.abs(input_vec))
    x_scale = x_max / QUANT_RANGE if x_max > 0 else 1.0
    input_q = quantize_vector(input_vec, x_scale)

    # Quantize fc1 weights
    fc1_w_max = np.max(np.abs(fc1_w))
    fc1_w_scale = fc1_w_max / QUANT_RANGE if fc1_w_max > 0 else 1.0
    fc1_w_q = quantize_vector(fc1_w.flatten(), fc1_w_scale)
    fc1_b_scale = fc1_w_scale * x_scale
    fc1_b_q = quantize_vector(fc1_b, fc1_b_scale)

    # Compute fc1 output scale for fc2 input quantization
    # After matmul: output is in fc1_b_scale domain
    # After ReLU: same scale (ReLU doesn't change scale)
    # For fc2, we need to quantize fc2 weights relative to the activation scale

    # Quantize fc2 weights
    fc2_w_max = np.max(np.abs(fc2_w))
    fc2_w_scale = fc2_w_max / QUANT_RANGE if fc2_w_max > 0 else 1.0
    fc2_w_q = quantize_vector(fc2_w.flatten(), fc2_w_scale)
    fc2_b_scale = fc2_w_scale * fc1_b_scale  # chain the scales
    fc2_b_q = quantize_vector(fc2_b, fc2_b_scale)

    ops = [
        {
            "type": "linear",
            "name": "fc1",
            "m": 128,
            "n": 784,
            "w_q": fc1_w_q,
            "b_q": fc1_b_q,
        },
        {
            "type": "relu",
            "name": "relu1",
        },
        {
            "type": "linear",
            "name": "fc2",
            "m": 10,
            "n": 128,
            "w_q": fc2_w_q,
            "b_q": fc2_b_q,
        },
    ]

    return {
        "mode": "mlp",
        "input": input_q,
        "ops": ops,
    }


def find_prover_binary():
    """Find the zk_ml_prover binary."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "rust", "zk_ml_prover", "target", "release", "zk_ml_prover"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "zk_ml_prover binary not found. Build with: "
        "cd rust/zk_ml_prover && cargo build --release"
    )


def main():
    print("=" * 60)
    print("MNIST MLP Benchmark: ZK-Compile Structured Sumcheck Prover")
    print("=" * 60)

    # Step 1: Train
    print("\n[1/4] Training MNIST MLP (784→128→ReLU→10)...")
    model = train_mnist(epochs=3)
    model.eval()

    # Step 2: Get test sample
    print("\n[2/4] Getting test sample...")
    input_vec, true_label, torch_pred = get_test_sample(model)
    print(f"  True label: {true_label}, PyTorch prediction: {torch_pred}")

    # Step 3: Build request
    print("\n[3/4] Quantizing weights to M31 and building request...")
    request = build_mnist_request(model, input_vec)

    # Count ops
    n_linear = sum(1 for op in request["ops"] if op["type"] == "linear")
    n_relu = sum(1 for op in request["ops"] if op["type"] == "relu")
    total_weights = sum(len(op.get("w_q", [])) for op in request["ops"])
    print(f"  Ops: {n_linear} linear + {n_relu} relu")
    print(f"  Total weight elements: {total_weights:,}")

    payload = json.dumps(request).encode()
    print(f"  JSON payload size: {len(payload) / 1e6:.2f} MB")

    # Step 4: Prove
    print("\n[4/4] Proving with structured sumcheck over M31...")
    binary = find_prover_binary()
    print(f"  Binary: {binary}")

    t0 = time.time()
    proc = subprocess.run(
        [binary],
        input=payload,
        capture_output=True,
        timeout=300,
    )
    wall_time = time.time() - t0

    if proc.returncode != 0:
        print(f"\n  PROVER FAILED (exit code {proc.returncode})")
        print(f"  stderr: {proc.stderr.decode()}")
        sys.exit(1)

    # Parse result
    stdout = proc.stdout.decode().strip()
    stderr = proc.stderr.decode().strip()
    result = json.loads(stdout)

    print(f"\n  stderr (timing):\n    {stderr.replace(chr(10), chr(10) + '    ')}")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Valid:          {result['valid']}")
    print(f"  Prediction:     {result['prediction']}")
    print(f"  Prove time:     {result['prove_time_ms']:.1f} ms")
    print(f"  Verify time:    {result['verify_time_ms']:.1f} ms")
    print(f"  Proof size:     {result['proof_size_bytes']:,} bytes")
    print(f"  Wall time:      {wall_time*1000:.1f} ms")

    if result.get('coverage'):
        cov = result['coverage']
        print(f"  Proved ops:     {cov['proved_count']}")
        print(f"  State ops:      {cov['state_count']}")
        print(f"  Op types:       {cov['proved_ops']}")

    # Comparison
    ezkl_prove_s = 1310.0  # From rust/zk_ml_prover/README.md
    our_prove_s = result['prove_time_ms'] / 1000.0
    speedup = ezkl_prove_s / our_prove_s if our_prove_s > 0 else float('inf')

    print(f"\n  vs EZKL ({ezkl_prove_s:.0f}s): {speedup:.0f}x faster proving")

    ezkl_verify_s = 5.4  # From README
    our_verify_s = result['verify_time_ms'] / 1000.0
    verify_speedup = ezkl_verify_s / our_verify_s if our_verify_s > 0 else float('inf')
    print(f"  vs EZKL ({ezkl_verify_s}s verify): {verify_speedup:.0f}x faster verification")

    ezkl_proof_kb = 127.0  # From README
    our_proof_kb = result['proof_size_bytes'] / 1024.0
    if our_proof_kb > 0:
        size_ratio = ezkl_proof_kb / our_proof_kb
        print(f"  vs EZKL ({ezkl_proof_kb:.0f}KB proof): {size_ratio:.1f}x {'smaller' if size_ratio > 1 else 'larger'}")

    print()


if __name__ == "__main__":
    main()
