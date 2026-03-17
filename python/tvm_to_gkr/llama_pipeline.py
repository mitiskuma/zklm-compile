"""Llama/Mistral/Qwen proving pipeline.

Sends composite llama_layer ops to the Rust prover. Rust owns the forward pass,
proving, and verification. Python is a dumb weight extractor.

Usage:
    from tvm_to_gkr.llama_pipeline import prove_llama
    result = prove_llama("meta-llama/Llama-2-7b-hf", "Hello world", prove_layers=1)
"""

import struct
import subprocess
import json
import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

from .constants import M31, QUANT_RANGE
from .model_extractor import ModelExtractor, ModelConfig, QuantizedWeights, quantize_symmetric


# Binary protocol op type bytes
OP_LLAMA_LAYER = 0x0C


def _pack_u32(v: int) -> bytes:
    return struct.pack('<I', v & 0xFFFFFFFF)


def _pack_i32(v: int) -> bytes:
    return struct.pack('<i', v)


def _pack_string(s: str) -> bytes:
    encoded = s.encode('utf-8')
    return _pack_u32(len(encoded)) + encoded


def _pack_u32_array(arr: np.ndarray) -> bytes:
    return arr.astype(np.uint32).tobytes()


def to_m31(val: int) -> int:
    """Convert signed integer to M31 field element."""
    return val % M31


def build_llama_layer_op(name: str, config: ModelConfig, silu_scale: int,
                          layer_idx: int, sd: dict, weights: dict) -> bytes:
    """Build binary payload for a single llama_layer composite op.

    Format: [0x0C] [name] [d_model] [d_ff] [num_q_heads] [num_kv_heads] [d_head] [silu_scale]
            [norm1_gamma] [w_q] [w_k] [w_v] [w_o] [norm2_gamma] [w_gate] [w_up] [w_down]
    """
    prefix = f"model.layers.{layer_idx}"

    data = bytes([OP_LLAMA_LAYER])
    data += _pack_string(name)
    data += _pack_u32(config.d_model)
    data += _pack_u32(config.d_ff)
    data += _pack_u32(config.num_q_heads)
    data += _pack_u32(config.num_kv_heads)
    data += _pack_u32(config.d_head)
    data += _pack_i32(silu_scale)

    # norm1_gamma
    norm1_gamma = sd[f"{prefix}.input_layernorm.weight"].float().numpy()
    norm1_gamma_q, _ = quantize_symmetric(norm1_gamma)
    data += _pack_u32_array(norm1_gamma_q)

    # QKV + O projections
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        w_q = weights[f"layer{layer_idx}.{proj}"].w_q
        data += _pack_u32_array(w_q)

    # norm2_gamma
    norm2_gamma = sd[f"{prefix}.post_attention_layernorm.weight"].float().numpy()
    norm2_gamma_q, _ = quantize_symmetric(norm2_gamma)
    data += _pack_u32_array(norm2_gamma_q)

    # gate, up, down projections
    for proj in ['gate_proj', 'up_proj', 'down_proj']:
        w_q = weights[f"layer{layer_idx}.{proj}"].w_q
        data += _pack_u32_array(w_q)

    return data


def prove_llama(
    model_name: str,
    text: str,
    prove_layers: int = 1,
    silu_scale: int = 1000,
    exp_scale: int = 100,
) -> dict:
    """Prove Llama/Mistral/Qwen inference end-to-end.

    Args:
        model_name: HuggingFace model name (e.g. "meta-llama/Llama-2-7b-hf")
        text: Input text
        prove_layers: Number of layers to prove (from the end)
        silu_scale: Quantization scale for SiLU lookup table
        exp_scale: Quantization scale for exp (softmax) lookup table

    Returns:
        dict with prove/verify times, proof size, coverage, etc.
    """
    import torch
    from transformers import AutoTokenizer

    print(f"Loading model: {model_name}")
    extractor = ModelExtractor(model_name)
    config = extractor.config
    weights = extractor.extract()

    print(f"Architecture: {config.model_type}, d_model={config.d_model}, "
          f"d_ff={config.d_ff}, heads={config.num_q_heads}q/{config.num_kv_heads}kv, "
          f"layers={config.n_layers}")

    # Get hidden states at the start of the proved layers
    start_layer = config.n_layers - prove_layers
    hidden = extractor.get_hidden_states(text, layer_idx=start_layer)
    print(f"Hidden states shape: {hidden.shape} at layer {start_layer}")

    # Quantize input hidden states
    input_q, input_scale = quantize_symmetric(hidden)

    # Build one llama_layer op per proved layer
    model = extractor.model
    sd = model.state_dict()

    all_ops_bytes = b''
    num_ops = prove_layers
    for layer_offset in range(prove_layers):
        layer_idx = start_layer + layer_offset
        all_ops_bytes += build_llama_layer_op(
            f"layer_{layer_idx}", config, silu_scale, layer_idx, sd, weights
        )

    print(f"Built {num_ops} llama_layer op(s) for {prove_layers} layer(s)")

    # Build binary payload: magic + num_ops + input + ops
    payload = bytes([0x00])  # binary mode magic
    payload += _pack_u32(num_ops)
    payload += _pack_u32(len(input_q))
    payload += _pack_u32_array(input_q)
    payload += all_ops_bytes

    # Run Rust prover
    rust_binary = os.path.join(
        os.path.dirname(__file__), '..', '..', 'rust', 'zk_ml_prover',
        'target', 'release', 'zk_ml_prover'
    )
    if not os.path.exists(rust_binary):
        raise FileNotFoundError(f"Build first: cd rust/zk_ml_prover && cargo build --release")

    print(f"Running Rust prover ({len(payload) / 1e6:.1f}MB payload)...")
    t0 = time.time()
    result = subprocess.run(
        [rust_binary],
        input=payload,
        capture_output=True,
    )
    wall_time = time.time() - t0

    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='replace')
        print(f"Prover stderr:\n{stderr}")
        raise RuntimeError(f"Prover failed with code {result.returncode}")

    # Parse JSON output
    stdout = result.stdout.decode('utf-8').strip()
    stderr = result.stderr.decode('utf-8', errors='replace')
    print(f"Prover stderr:\n{stderr}")

    try:
        response = json.loads(stdout)
    except json.JSONDecodeError:
        print(f"Raw stdout: {stdout[:500]}")
        raise

    response['wall_time_ms'] = wall_time * 1000
    response['model'] = model_name
    response['prove_layers'] = prove_layers
    response['config'] = {
        'd_model': config.d_model,
        'd_ff': config.d_ff,
        'num_q_heads': config.num_q_heads,
        'num_kv_heads': config.num_kv_heads,
        'n_layers': config.n_layers,
    }

    return response


def prove_llama_token(
    extractor: ModelExtractor,
    weights: dict,
    text: str,
    prove_layers: int = 1,
    silu_scale: int = 1000,
) -> dict:
    """Prove a single next-token prediction using a pre-loaded Llama model.

    Like prove_llama() but reuses an already-loaded ModelExtractor and weights dict,
    avoiding the expensive model load on every call.

    Args:
        extractor: Pre-loaded ModelExtractor instance
        weights: Pre-extracted weight dict from extractor.extract()
        text: Full context string (all tokens so far)
        prove_layers: Number of layers to prove (from the end)
        silu_scale: Quantization scale for SiLU lookup table

    Returns:
        dict with valid, prove_time_ms, verify_time_ms, proof_size_bytes, prediction, etc.
    """
    config = extractor.config
    start_layer = config.n_layers - prove_layers

    hidden = extractor.get_hidden_states(text, layer_idx=start_layer)
    input_q, input_scale = quantize_symmetric(hidden)

    model = extractor.model
    sd = model.state_dict()

    all_ops_bytes = b''
    for layer_offset in range(prove_layers):
        layer_idx = start_layer + layer_offset
        all_ops_bytes += build_llama_layer_op(
            f"layer_{layer_idx}", config, silu_scale, layer_idx, sd, weights
        )

    payload = bytes([0x00])
    payload += _pack_u32(prove_layers)
    payload += _pack_u32(len(input_q))
    payload += _pack_u32_array(input_q)
    payload += all_ops_bytes

    rust_binary = os.path.join(
        os.path.dirname(__file__), '..', '..', 'rust', 'zk_ml_prover',
        'target', 'release', 'zk_ml_prover'
    )
    if not os.path.exists(rust_binary):
        raise FileNotFoundError("Build first: cd rust/zk_ml_prover && cargo build --release")

    t0 = time.time()
    result = subprocess.run([rust_binary], input=payload, capture_output=True)
    wall_time = time.time() - t0

    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='replace')
        raise RuntimeError(f"Prover failed (code {result.returncode}): {stderr}")

    stdout = result.stdout.decode('utf-8').strip()
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Bad prover output: {stdout[:200]}")

    response['wall_time_ms'] = wall_time * 1000
    response['model'] = extractor.model_name if hasattr(extractor, 'model_name') else 'unknown'
    response['prove_layers'] = prove_layers
    return response


def build_llama_layer_ref_op(name: str, config, silu_scale: int, layer_idx: int) -> dict:
    """Build a server-mode llama_layer_ref op dict (references preloaded weights by name).

    Returns a dict suitable for RustProverServer.prove() server_ops list.
    """
    return {
        "type": "llama_layer_ref",
        "name": name,
        "config": {
            "d_model": config.d_model,
            "d_ff": config.d_ff,
            "num_q_heads": config.num_q_heads,
            "num_kv_heads": config.num_kv_heads,
            "d_head": config.d_head,
            "silu_scale": silu_scale,
        },
        "weight_names": [
            f"layer{layer_idx}.norm1_gamma",
            f"layer{layer_idx}.q_proj",
            f"layer{layer_idx}.k_proj",
            f"layer{layer_idx}.v_proj",
            f"layer{layer_idx}.o_proj",
            f"layer{layer_idx}.norm2_gamma",
            f"layer{layer_idx}.gate_proj",
            f"layer{layer_idx}.up_proj",
            f"layer{layer_idx}.down_proj",
        ],
    }


def prove_llama_token_server(
    server,
    extractor,
    text: str,
    prove_layers: int = 1,
    silu_scale: int = 1000,
) -> dict:
    """Prove a Llama token using server-mode prover (preloaded weights).

    Args:
        server: RustProverServer with Llama weights preloaded
        extractor: Pre-loaded ModelExtractor instance
        text: Full context string
        prove_layers: Number of layers to prove (from the end)
        silu_scale: SiLU quantization scale

    Returns:
        dict with valid, prove_time_ms, verify_time_ms, proof_size_bytes, etc.
    """
    config = extractor.config
    start_layer = config.n_layers - prove_layers

    hidden = extractor.get_hidden_states(text, layer_idx=start_layer)
    input_q, _ = quantize_symmetric(hidden)

    server_ops = []
    for layer_offset in range(prove_layers):
        layer_idx = start_layer + layer_offset
        server_ops.append(build_llama_layer_ref_op(
            f"layer_{layer_idx}", config, silu_scale, layer_idx
        ))

    result = server.prove(input_q, server_ops)
    result['model'] = extractor.model_name if hasattr(extractor, 'model_name') else 'unknown'
    result['prove_layers'] = prove_layers
    return result
