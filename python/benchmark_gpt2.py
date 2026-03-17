"""Benchmark ZK proof of GPT-2 transformer inference.

Measures prove time, verify time, proof size, and wall clock for varying
numbers of transformer layers. Produces a clean benchmark table.

Usage:
    cd zk && .venv/bin/python3 python/benchmark_gpt2.py
"""
import time
import json
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tvm_to_gkr.structured_pipeline import (
    PrecompiledTransformerWeights,
    RustProverServer,
    prove_rust_transformer_v2,
)


def benchmark():
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Pre-compiling weights...")
    precompiled = PrecompiledTransformerWeights(model)

    print("Starting Rust prover server...")
    server = RustProverServer(precompiled)

    # Warmup
    input_ids = tokenizer.encode("Hello world", return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        h = outputs.hidden_states[0][0, -1, :]

    print("Warmup...")
    prove_rust_transformer_v2(model, h, prove_layers=1, verbose=False,
                               precompiled=precompiled, server=server)

    # Benchmark
    layer_configs = [1, 2, 4, 6, 8, 12]
    prompts = [
        "The capital of France is",
        "In machine learning, a neural network",
        "The quick brown fox jumps over",
    ]

    print("\n" + "="*80)
    print("GPT-2 (124M params) ZK Proof Benchmark — Sumcheck over Mersenne-31")
    print("="*80)
    print(f"{'Layers':>7} {'Prove(ms)':>10} {'Verify(ms)':>11} "
          f"{'Proof(B)':>9} {'Wall(ms)':>9} {'Comp%':>6} {'Total%':>7} {'Ops':>5}")
    print("-"*80)

    for n_layers in layer_configs:
        prove_times = []
        verify_times = []
        proof_sizes = []
        wall_times = []
        last_coverage = None

        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                layer_idx = 12 - n_layers
                h = outputs.hidden_states[layer_idx][0, -1, :]

            t0 = time.time()
            result = prove_rust_transformer_v2(
                model, h, prove_layers=n_layers, verbose=False,
                precompiled=precompiled, server=server,
            )
            wall = (time.time() - t0) * 1000

            assert result["valid"], f"Proof FAILED for {n_layers} layers!"
            prove_times.append(result["prove_time_ms"])
            verify_times.append(result["verify_time_ms"])
            proof_sizes.append(result["proof_size_bytes"])
            wall_times.append(wall)
            if result.get("coverage"):
                last_coverage = result["coverage"]

        avg_prove = np.mean(prove_times)
        avg_verify = np.mean(verify_times)
        avg_proof = int(np.mean(proof_sizes))
        avg_wall = np.mean(wall_times)

        comp_pct = last_coverage["computational_pct"] if last_coverage else 0.0
        total_pct = last_coverage["total_pct"] if last_coverage else 0.0
        total_ops = last_coverage["total_count"] if last_coverage else 0

        print(f"{n_layers:>7} {avg_prove:>10.1f} {avg_verify:>11.3f} "
              f"{avg_proof:>9} {avg_wall:>9.0f} {comp_pct:>5.1f}% {total_pct:>6.1f}% {total_ops:>5}")

    print("-"*80)
    print("\nProved computational ops/layer: c_attn, c_proj, c_fc, mlp_proj (sumcheck),")
    print("  ln1, ln2 (squared proof + QR perturbation), GELU (LogUp lookup),")
    print("  attention (trivial seq_len=1), 2x residual adds (claim chaining)")
    print("Verified state ops: save (content commitment), set_input (cross-segment commitment)")
    print("Comp% = computational ops only (per zkGPT methodology)")
    print("Total% = all ops incl. committed state management")
    print("Field: Mersenne-31 (p = 2^31 - 1), INT8 quantization")
    print("Platform: Apple M4 Max, single-threaded Rust prover")

    server.shutdown()


if __name__ == "__main__":
    benchmark()
