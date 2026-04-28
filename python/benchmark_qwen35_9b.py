#!/usr/bin/env python3
"""Benchmark Qwen3.5-9B full proof via server-mode prover."""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tvm_to_gkr.model_extractor import ModelExtractor
from tvm_to_gkr.structured_pipeline import RustProverServer
from tvm_to_gkr.qwen35_pipeline import (
    QwenPrecompiledWeights,
    build_qwen_layer_ref_op,
)
from tvm_to_gkr.model_extractor import quantize_symmetric


def main():
    args = sys.argv[1:]
    model_name = args[0] if len(args) > 0 else "Qwen/Qwen3.5-9B"
    prove_layers = int(args[1]) if len(args) > 1 else 32
    text = "The quick brown fox jumps over the lazy dog."

    print(f"=== Qwen3.5-9B benchmark ===")
    print(f"Model: {model_name}")
    print(f"Prove layers: {prove_layers}")
    print(f"Input: {text!r}")
    print()

    print("[1/4] Loading model extractor (bf16 to save memory)...")
    t0 = time.time()
    import torch
    extractor = ModelExtractor(model_name, dtype=torch.bfloat16)
    extractor.model_name = model_name
    print(f"      done in {time.time() - t0:.1f}s")
    print(f"      config: d={extractor.config.d_model} ff={extractor.config.d_ff} layers={extractor.config.n_layers}")
    if extractor.config.layer_types:
        gdn = sum(1 for x in extractor.config.layer_types if x == "gdn")
        attn = sum(1 for x in extractor.config.layer_types if x == "attn")
        print(f"      layer types: {gdn} gdn + {attn} attn")

    print("[2/4] Extracting weights...")
    t0 = time.time()
    weights = extractor.extract()
    print(f"      done in {time.time() - t0:.1f}s ({len(weights)} weight tensors)")

    print("[3/4] Pre-compiling for server...")
    t0 = time.time()
    precompiled = QwenPrecompiledWeights(extractor, weights)
    print(f"      done in {time.time() - t0:.1f}s ({len(precompiled.weight_entries)} weight entries)")

    # Pre-compute hidden states for prove input now (still need model loaded).
    start_layer = extractor.config.n_layers - prove_layers
    print(f"      Pre-computing hidden states at layer {start_layer}...")
    hidden_q_input = extractor.get_hidden_states(text, layer_idx=start_layer)
    config_snapshot = extractor.config

    # Free model + raw weights — saves ~2x model size of RAM (huge for 9B).
    print("      Freeing source model + raw weights to reclaim RAM...")
    extractor.model = None
    del extractor
    weights.clear()
    del weights
    import gc
    gc.collect()

    print("[4/4] Starting prover server + loading weights...")
    t0 = time.time()
    server = RustProverServer(weight_entries=precompiled.weight_entries)
    print(f"      server ready in {time.time() - t0:.1f}s")

    # Server has now copied weights to Rust process. Free Python copies.
    print("      Freeing Python-side weights (server has them)...")
    precompiled.weight_entries = []
    del precompiled
    gc.collect()

    print()
    print(f"=== Proving {prove_layers} layers ===")
    silu_scale = 1000
    sigmoid_scale = 1000
    input_q, _ = quantize_symmetric(hidden_q_input)
    server_ops = []
    for layer_offset in range(prove_layers):
        layer_idx = start_layer + layer_offset
        server_ops.append(build_qwen_layer_ref_op(
            f"layer_{layer_idx}", config_snapshot, silu_scale, sigmoid_scale, layer_idx
        ))

    t0 = time.time()
    result = server.prove(input_q, server_ops)
    result['model'] = model_name
    result['prove_layers'] = prove_layers
    wall = time.time() - t0

    print()
    print(f"=== Results ===")
    print(json.dumps(
        {k: v for k, v in result.items() if not isinstance(v, (bytes, bytearray))},
        indent=2,
        default=str,
    ))
    print(f"wall_time_total_s: {wall:.3f}")

    server.shutdown()


if __name__ == "__main__":
    main()
