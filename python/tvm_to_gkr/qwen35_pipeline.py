"""Qwen3.5 (GatedDeltaNet hybrid) proving pipeline.

Sends composite qwen_layer ops to the Rust prover. Like llama_pipeline but
with output gating (g_proj + sigmoid) and Conv1D folding for GDN layers.

At seq_len=1, Conv1D with zero history = element-wise multiply by conv_weight[:,:,0].
We fold this into the Q/K/V projection weights in Python so Rust never sees Conv1D.

Usage:
    from tvm_to_gkr.qwen35_pipeline import prove_qwen35_token_server
    result = prove_qwen35_token_server(server, extractor, text, prove_layers=1)
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


# Binary protocol op type byte for qwen_layer
OP_QWEN_LAYER = 0x0E


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


def _fold_conv_into_proj(w_proj_q: np.ndarray, conv_weight: np.ndarray,
                          proj_rows: int, d_model: int, proj_offset: int) -> np.ndarray:
    """Fold Conv1D weight[:,:,0] into projection weights for seq_len=1.

    At seq_len=1 with zero history, Conv1D is just element-wise multiply
    by conv_weight[:, 0, 0]. So effective_w = w_proj * conv_scale[row].

    Args:
        w_proj_q: Quantized projection weights as uint32 M31 elements (flat, row-major)
        conv_weight: Conv1D weight tensor (total_dim, 1, kernel_size)
        proj_rows: Number of rows in this projection
        d_model: Number of columns
        proj_offset: Offset into conv_weight for this projection's rows

    Returns:
        New uint32 M31 array with conv folded in (same shape as w_proj_q)
    """
    # conv_weight[:, 0, 0] gives the scaling factor per output channel
    conv_scale = conv_weight[proj_offset:proj_offset + proj_rows, 0, 0]

    # Convert M31 field elements back to signed integers
    w_signed = np.where(w_proj_q > M31 // 2,
                         w_proj_q.astype(np.int64) - M31,
                         w_proj_q.astype(np.int64))
    w_2d = w_signed.reshape(proj_rows, d_model)

    # Scale each row by its conv factor
    for i in range(proj_rows):
        w_2d[i] = np.round(w_2d[i] * conv_scale[i]).astype(np.int64)

    # Convert back to M31
    w_flat = w_2d.flatten()
    result = np.where(w_flat >= 0, w_flat, M31 + w_flat).astype(np.uint64) % M31
    return result.astype(np.uint32)


def build_qwen_layer_op(name: str, config: ModelConfig, silu_scale: int,
                          sigmoid_scale: int, layer_idx: int, sd: dict,
                          weights: dict) -> bytes:
    """Build binary payload for a single qwen_layer composite op.

    Format: [0x0E] [name] [d_model] [d_ff] [num_q_heads] [num_kv_heads] [d_head]
            [silu_scale] [sigmoid_scale]
            [norm1_gamma] [w_q] [w_k] [w_v] [w_o] [w_g_proj]
            [norm2_gamma] [w_gate] [w_up] [w_down]
    """
    prefix = f"model.layers.{layer_idx}"
    layer_type = config.layer_types[layer_idx] if config.layer_types else 'attn'

    data = bytes([OP_QWEN_LAYER])
    data += _pack_string(name)
    data += _pack_u32(config.d_model)
    data += _pack_u32(config.d_ff)
    data += _pack_u32(config.num_q_heads)
    data += _pack_u32(config.num_kv_heads)
    data += _pack_u32(config.d_head)
    data += _pack_i32(silu_scale)
    data += _pack_i32(sigmoid_scale)

    # norm1_gamma
    norm1_gamma = sd[f"{prefix}.input_layernorm.weight"].float().numpy()
    norm1_gamma_q, _ = quantize_symmetric(norm1_gamma)
    data += _pack_u32_array(norm1_gamma_q)

    # QKV + O projections (potentially conv-folded for GDN layers)
    q_dim = config.num_q_heads * config.d_head
    kv_dim = config.num_kv_heads * config.d_head

    for proj, dim in [('q_proj', q_dim), ('k_proj', kv_dim), ('v_proj', kv_dim), ('o_proj', None)]:
        w_q = weights[f"layer{layer_idx}.{proj}"].w_q

        # Conv1D folding for GDN layers (skip for o_proj — it's after attention)
        if layer_type == 'gdn' and proj != 'o_proj':
            conv_key = f"layer{layer_idx}.short_conv"
            if conv_key in weights:
                conv_w = weights[conv_key]
                # Reconstruct conv weight shape
                conv_data = conv_w.w_q.view(np.float32).reshape(conv_w.shape)
                # Offset into conv weight for this projection
                if proj == 'q_proj':
                    offset = 0
                elif proj == 'k_proj':
                    offset = q_dim
                else:  # v_proj
                    offset = q_dim + kv_dim
                w_q = _fold_conv_into_proj(w_q, conv_data, dim, config.d_model, offset)

        data += _pack_u32_array(w_q)

    # g_proj (output gate)
    g_proj_key = f"layer{layer_idx}.g_proj"
    if g_proj_key in weights:
        data += _pack_u32_array(weights[g_proj_key].w_q)
    else:
        # Fallback: zero g_proj (sigmoid(0)=0.5, half gating)
        data += _pack_u32_array(np.zeros(q_dim * config.d_model, dtype=np.uint32))

    # norm2_gamma
    norm2_gamma = sd[f"{prefix}.post_attention_layernorm.weight"].float().numpy()
    norm2_gamma_q, _ = quantize_symmetric(norm2_gamma)
    data += _pack_u32_array(norm2_gamma_q)

    # gate, up, down projections
    for proj in ['gate_proj', 'up_proj', 'down_proj']:
        w_q = weights[f"layer{layer_idx}.{proj}"].w_q
        data += _pack_u32_array(w_q)

    return data


def prove_qwen35(
    model_name: str,
    text: str,
    prove_layers: int = 1,
    silu_scale: int = 1000,
    sigmoid_scale: int = 1000,
) -> dict:
    """Prove Qwen3.5 inference end-to-end."""
    import torch
    from transformers import AutoTokenizer

    print(f"Loading model: {model_name}")
    extractor = ModelExtractor(model_name)
    config = extractor.config
    weights = extractor.extract()

    print(f"Architecture: {config.model_type}, d_model={config.d_model}, "
          f"d_ff={config.d_ff}, heads={config.num_q_heads}q/{config.num_kv_heads}kv, "
          f"layers={config.n_layers}")
    if config.layer_types:
        gdn_count = sum(1 for t in config.layer_types if t == 'gdn')
        attn_count = sum(1 for t in config.layer_types if t == 'attn')
        print(f"Layer types: {gdn_count} GatedDeltaNet + {attn_count} attention")

    start_layer = config.n_layers - prove_layers
    hidden = extractor.get_hidden_states(text, layer_idx=start_layer)
    print(f"Hidden states shape: {hidden.shape} at layer {start_layer}")

    input_q, input_scale = quantize_symmetric(hidden)

    model = extractor.model
    sd = model.state_dict()

    all_ops_bytes = b''
    for layer_offset in range(prove_layers):
        layer_idx = start_layer + layer_offset
        all_ops_bytes += build_qwen_layer_op(
            f"layer_{layer_idx}", config, silu_scale, sigmoid_scale,
            layer_idx, sd, weights
        )

    print(f"Built {prove_layers} qwen_layer op(s)")

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
        raise FileNotFoundError(
            f"Rust prover binary not found at: {os.path.abspath(rust_binary)}\n"
            f"Build it with:\n"
            f"  cd rust/zk_ml_prover && cargo build --release\n"
            f"This compiles the structured sumcheck prover (~30s on M4 Max)."
        )

    print(f"Running Rust prover ({len(payload) / 1e6:.1f}MB payload)...")
    t0 = time.time()
    result = subprocess.run([rust_binary], input=payload, capture_output=True)
    wall_time = time.time() - t0

    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='replace')
        print(f"Prover stderr:\n{stderr}")
        raise RuntimeError(f"Prover failed with code {result.returncode}")

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
    return response


def _layer_head_config(config: ModelConfig, layer_idx: int) -> dict:
    """Return per-layer head config (GDN and full attention have different head counts)."""
    layer_type = config.layer_types[layer_idx] if config.layer_types else 'attn'
    if layer_type == 'gdn':
        return {
            "num_q_heads": config.gdn_num_heads,
            "num_kv_heads": config.gdn_num_heads,  # GDN: same for Q/K/V
            "d_head": config.gdn_d_head,
        }
    else:
        return {
            "num_q_heads": config.num_q_heads,
            "num_kv_heads": config.num_kv_heads,
            "d_head": config.d_head,
        }


def build_qwen_layer_ref_op(name: str, config: ModelConfig, silu_scale: int,
                              sigmoid_scale: int, layer_idx: int) -> dict:
    """Build a server-mode qwen_layer_ref op dict (references preloaded weights by name)."""
    heads = _layer_head_config(config, layer_idx)
    return {
        "type": "qwen_layer_ref",
        "name": name,
        "config": {
            "d_model": config.d_model,
            "d_ff": config.d_ff,
            **heads,
            "silu_scale": silu_scale,
            "sigmoid_scale": sigmoid_scale,
        },
        "weight_names": [
            f"layer{layer_idx}.norm1_gamma",
            f"layer{layer_idx}.q_proj",
            f"layer{layer_idx}.k_proj",
            f"layer{layer_idx}.v_proj",
            f"layer{layer_idx}.o_proj",
            f"layer{layer_idx}.g_proj",
            f"layer{layer_idx}.norm2_gamma",
            f"layer{layer_idx}.gate_proj",
            f"layer{layer_idx}.up_proj",
            f"layer{layer_idx}.down_proj",
        ],
    }


def prove_qwen35_token_server(
    server,
    extractor: ModelExtractor,
    text: str,
    prove_layers: int = 1,
    silu_scale: int = 1000,
    sigmoid_scale: int = 1000,
) -> dict:
    """Prove a Qwen3.5 token using server-mode prover (preloaded weights).

    Args:
        server: RustProverServer with Qwen weights preloaded
        extractor: Pre-loaded ModelExtractor instance
        text: Full context string
        prove_layers: Number of layers to prove (from the end)
        silu_scale: SiLU quantization scale
        sigmoid_scale: Sigmoid quantization scale

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
        server_ops.append(build_qwen_layer_ref_op(
            f"layer_{layer_idx}", config, silu_scale, sigmoid_scale, layer_idx
        ))

    result = server.prove(input_q, server_ops)
    result['model'] = extractor.model_name if hasattr(extractor, 'model_name') else 'unknown'
    result['prove_layers'] = prove_layers
    return result


class QwenPrecompiledWeights:
    """Pre-quantized weight matrices for Qwen3.5 transformer blocks.

    Weights are already split/extracted by ModelExtractor. This class handles
    Conv1D folding for GDN layers and builds the weight_entries list for the
    Rust server.
    """

    def __init__(self, extractor: ModelExtractor, weights: dict):
        import time

        t0 = time.time()
        self.config = extractor.config
        self.n_layers = self.config.n_layers

        self.weight_entries = []  # list of (name, w_q_array, m, n)

        gdn_dim = self.config.gdn_num_heads * self.config.gdn_d_head  # 2048
        attn_q_dim = self.config.num_q_heads * self.config.d_head       # 2048
        attn_kv_dim = self.config.num_kv_heads * self.config.d_head     # 512

        for i in range(self.n_layers):
            layer_type = self.config.layer_types[i] if self.config.layer_types else 'attn'
            is_gdn = layer_type == 'gdn'

            # Per-layer dimensions
            if is_gdn:
                q_dim = gdn_dim
                kv_dim = gdn_dim
            else:
                q_dim = attn_q_dim
                kv_dim = attn_kv_dim

            # RMSNorm gammas
            self.weight_entries.append((f"layer{i}.norm1_gamma", weights[f"layer{i}.norm1"].w_q, self.config.d_model, 1))
            self.weight_entries.append((f"layer{i}.norm2_gamma", weights[f"layer{i}.norm2"].w_q, self.config.d_model, 1))

            # Q, K, V projections (potentially conv-folded for GDN)
            proj_dims = {
                'q_proj': (q_dim, self.config.d_model),
                'k_proj': (kv_dim, self.config.d_model),
                'v_proj': (kv_dim, self.config.d_model),
                'o_proj': (self.config.d_model, q_dim),
            }

            for proj, (m, n) in proj_dims.items():
                w_q = weights[f"layer{i}.{proj}"].w_q

                # Conv1D folding for GDN layers (skip o_proj)
                if is_gdn and proj != 'o_proj':
                    conv_key = f"layer{i}.short_conv"
                    if conv_key in weights:
                        conv_w = weights[conv_key]
                        conv_data = conv_w.w_q.view(np.float32).reshape(conv_w.shape)
                        if proj == 'q_proj':
                            offset = 0
                        elif proj == 'k_proj':
                            offset = gdn_dim
                        else:  # v_proj
                            offset = 2 * gdn_dim
                        w_q = _fold_conv_into_proj(w_q, conv_data, m, n, offset)

                self.weight_entries.append((f"layer{i}.{proj}", w_q, m, n))

            # g_proj (output gate — from in_proj_z for GDN, from packed q_proj for attn)
            gw = weights[f"layer{i}.g_proj"]
            self.weight_entries.append((f"layer{i}.g_proj", gw.w_q, q_dim, self.config.d_model))

            # MLP projections
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                qw = weights[f"layer{i}.{proj}"]
                total = len(qw.w_q)
                if proj in ('gate_proj', 'up_proj'):
                    pn = self.config.d_model
                    pm = total // pn
                else:
                    pm = self.config.d_model
                    pn = total // pm
                self.weight_entries.append((f"layer{i}.{proj}", qw.w_q, pm, pn))

        elapsed = time.time() - t0
        print(f"Pre-compiled {self.n_layers} Qwen layers ({len(self.weight_entries)} weight matrices) in {elapsed:.2f}s")
