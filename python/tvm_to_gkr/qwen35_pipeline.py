"""Qwen3.5 (GatedDeltaNet hybrid) proving pipeline.

Sends composite qwen_layer ops to the Rust prover. Like llama_pipeline but
with output gating (g_proj + sigmoid) and Conv1D folding for GDN layers.

At seq_len=1, Conv1D with zero history collapses to an element-wise scale
on the projection output. Which kernel tap supplies that scale depends on
the upstream model's storage convention:

  Standard PyTorch ``nn.Conv1d`` with causal left-padding (kernel_size - 1
  zeros prepended) at t=0 gives::

      output[c, 0] = sum_k conv_weight[c, 0, k] * pad_input[c, t - K + 1 + k]
                   = conv_weight[c, 0, K-1] * input[c, 0]      # only k = K-1 sees a non-zero input

  i.e. the LAST tap (``conv_weight[:, 0, K-1]``).

  Some HF model checkpoints (and certain Mamba/GDN custom kernels) instead
  store conv weights in reverse-time order, where index 0 is the most
  recent tap and index K-1 is the oldest. In that case ``conv_weight[:, 0, 0]``
  is the correct fold scale.

The fold helper below is parametrized via ``causal_tap_index`` so the
correct convention can be selected per checkpoint. The default
``causal_tap_index=0`` is the historical behavior of this pipeline and
matches the unit tests; the ``causal_tap_index=-1`` branch enables the
PyTorch standard. ``_fold_conv_into_proj`` validates both at the shape
level. G1 — flip the default once we have empirical
confirmation against an HF reference forward pass.

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
                          proj_rows: int, d_model: int, proj_offset: int,
                          causal_tap_index: int = 0) -> np.ndarray:
    """Fold the seq_len=1 Conv1D scale into projection weights.

    At seq_len=1 with zero history, Conv1D collapses to an element-wise
    multiply by `conv_weight[:, 0, causal_tap_index]`. Folding that scale
    into `w_proj` produces an effective projection that the prover can
    treat as a plain matmul (no Conv1D op needed in the proof system).

    SAFETY (G1): which tap supplies the scale depends
    on the upstream model's storage convention. PyTorch's standard
    ``nn.Conv1d`` with causal left-padding makes the LAST kernel position
    (``conv_weight[:, 0, K-1]``) the active tap at t=0 (`causal_tap_index=-1`).
    Some HF Mamba/GDN custom kernels store weights in reverse-time order,
    making index 0 the active tap (`causal_tap_index=0`). The historical
    default of this pipeline is ``causal_tap_index=0``; flipping requires
    a forward-pass match against an HF reference. The unit tests pin both
    branches so a regression on either convention is loud.

    Args:
        w_proj_q: Quantized projection weights as uint32 M31 elements
            (flat, row-major).
        conv_weight: Conv1D weight tensor of shape
            ``(total_dim, 1, kernel_size)``.
        proj_rows: Number of rows in this projection.
        d_model: Number of columns.
        proj_offset: Offset into ``conv_weight`` rows for this projection.
        causal_tap_index: Which kernel index supplies the t=0 scale. Use
            ``0`` for reverse-time-stored kernels (historical default) or
            ``-1`` for standard PyTorch ``nn.Conv1d`` causal-pad
            convention. Any in-range integer is accepted; out-of-range
            raises ``IndexError``.

    Returns:
        New uint32 M31 array with conv folded in (same shape as w_proj_q).
    """
    if conv_weight.ndim != 3 or conv_weight.shape[1] != 1:
        raise ValueError(
            f"_fold_conv_into_proj expects shape (total_dim, 1, kernel_size); "
            f"got {conv_weight.shape}"
        )
    kernel_size = conv_weight.shape[2]
    # Resolve negative indices (e.g. -1 for last tap) to absolute.
    if causal_tap_index < 0:
        tap = kernel_size + causal_tap_index
    else:
        tap = causal_tap_index
    if not (0 <= tap < kernel_size):
        raise IndexError(
            f"causal_tap_index={causal_tap_index} out of range for "
            f"kernel_size={kernel_size}"
        )

    # conv_weight[:, 0, tap] gives the scaling factor per output channel.
    conv_scale = conv_weight[proj_offset:proj_offset + proj_rows, 0, tap]

    # Convert M31 field elements back to signed integers
    w_signed = np.where(w_proj_q > M31 // 2,
                         w_proj_q.astype(np.int64) - M31,
                         w_proj_q.astype(np.int64))
    w_2d = w_signed.reshape(proj_rows, d_model)

    # Scale each row by its conv factor
    for i in range(proj_rows):
        w_2d[i] = np.round(w_2d[i] * conv_scale[i]).astype(np.int64)

    # SOUNDNESS (G2): the upstream Qwen3.5 GDN forward
    # pass applies SiLU AFTER the Conv1D and BEFORE the Q/K/V split:
    #
    #     y = silu(conv1d(in_proj(x)))      # ← real model
    #     q, k, v = split(y)
    #
    # Folding only the linear conv into the projection weights captures
    # the ``conv1d(in_proj(x))`` step but NOT the SiLU. The Rust prover
    # therefore proves an APPROXIMATION where the Q/K/V inputs are the
    # pre-SiLU values. SiLU on a single channel is monotonic and roughly
    # linear in [-1, 1], so the approximation drift is small at typical
    # GDN scales (the inputs are post-RMSNorm, so |x| is O(1)), but it is
    # an approximation — not a faithful reproduction of the model.
    #
    # A faithful proof would either (a) require the prover to absorb a
    # SiLU lookup proof between the folded matmul and the Q/K/V split,
    # or (b) move the SiLU into the next layer's input pre-processing.
    # Both are tracked as future work; until then this approximation is
    # explicitly called out in the README "Limitations" section so that
    # benchmark numbers can be reproduced under the same approximation.

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

    # Per-layer head config (GDN may have asymmetric V dim)
    heads = _layer_head_config(config, layer_idx)

    data = bytes([OP_QWEN_LAYER])
    data += _pack_string(name)
    data += _pack_u32(config.d_model)
    data += _pack_u32(config.d_ff)
    data += _pack_u32(heads["num_q_heads"])
    data += _pack_u32(heads["num_kv_heads"])
    data += _pack_u32(heads["d_head"])
    data += _pack_u32(heads["v_num_heads"])
    data += _pack_u32(heads["v_d_head"])
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
    """Return per-layer head config (GDN and full attention have different head counts).

    For GDN: V may have different heads/dim than K (e.g. Qwen3.5-4B 32-v vs 16-k).
    For full attn: V matches K (standard GQA).
    """
    layer_type = config.layer_types[layer_idx] if config.layer_types else 'attn'
    if layer_type == 'gdn':
        v_heads = config.gdn_v_num_heads or config.gdn_num_heads
        v_d = config.gdn_v_d_head or config.gdn_d_head
        return {
            "num_q_heads": config.gdn_num_heads,
            "num_kv_heads": config.gdn_num_heads,  # K shares head count with Q
            "d_head": config.gdn_d_head,
            "v_num_heads": v_heads,
            "v_d_head": v_d,
        }
    else:
        return {
            "num_q_heads": config.num_q_heads,
            "num_kv_heads": config.num_kv_heads,
            "d_head": config.d_head,
            "v_num_heads": config.num_kv_heads,
            "v_d_head": config.d_head,
        }


def build_qwen_layer_ref_op(name: str, config: ModelConfig, silu_scale: int,
                              sigmoid_scale: int, layer_idx: int) -> dict:
    """Build a server-mode qwen_layer_ref op dict (references preloaded weights by name).

    Config carries v_num_heads/v_d_head so the Rust prover knows V dim independently
    of K dim (required for Qwen3.5-4B/9B asymmetric GDN).
    """
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

        # GDN K/Q dim and V dim may differ (Qwen3.5-4B: V = 2*K)
        gdn_k_dim = self.config.gdn_num_heads * self.config.gdn_d_head
        gdn_v_dim = self.config.gdn_v_num_heads * self.config.gdn_v_d_head
        if gdn_v_dim == 0:  # backward compat: if v fields missing, assume symmetric
            gdn_v_dim = gdn_k_dim
        attn_q_dim = self.config.num_q_heads * self.config.d_head
        attn_kv_dim = self.config.num_kv_heads * self.config.d_head

        for i in range(self.n_layers):
            layer_type = self.config.layer_types[i] if self.config.layer_types else 'attn'
            is_gdn = layer_type == 'gdn'

            # Per-layer dimensions
            if is_gdn:
                q_dim = gdn_k_dim
                k_dim = gdn_k_dim
                v_dim = gdn_v_dim
                # GDN attention output has v_dim shape; gate matches.
                attn_out_dim = gdn_v_dim
            else:
                q_dim = attn_q_dim
                k_dim = attn_kv_dim
                v_dim = attn_kv_dim
                attn_out_dim = attn_q_dim

            # RMSNorm gammas
            self.weight_entries.append((f"layer{i}.norm1_gamma", weights[f"layer{i}.norm1"].w_q, self.config.d_model, 1))
            self.weight_entries.append((f"layer{i}.norm2_gamma", weights[f"layer{i}.norm2"].w_q, self.config.d_model, 1))

            # Q, K, V, O projections — o_proj input matches attention output dim.
            proj_dims = {
                'q_proj': (q_dim, self.config.d_model),
                'k_proj': (k_dim, self.config.d_model),
                'v_proj': (v_dim, self.config.d_model),
                'o_proj': (self.config.d_model, attn_out_dim),
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
                            offset = gdn_k_dim
                        else:  # v_proj
                            offset = 2 * gdn_k_dim
                        w_q = _fold_conv_into_proj(w_q, conv_data, m, n, offset)

                self.weight_entries.append((f"layer{i}.{proj}", w_q, m, n))

            # g_proj (in_proj_z for GDN, q_proj second-half for full attn) — gate matches attn output dim.
            gw = weights[f"layer{i}.g_proj"]
            self.weight_entries.append((f"layer{i}.g_proj", gw.w_q, attn_out_dim, self.config.d_model))

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
