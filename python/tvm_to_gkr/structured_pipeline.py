"""GPT-2 ZK proof pipeline using structured sumcheck over Mersenne-31.

Python handles model loading, weight quantization, and binary protocol
encoding. The Rust prover does the actual proving via structured sumcheck.

Usage:
    from tvm_to_gkr.structured_pipeline import (
        PrecompiledTransformerWeights,
        RustProverServer,
        prove_rust_transformer_v2,
    )

    precompiled = PrecompiledTransformerWeights(gpt2_model)
    server = RustProverServer(precompiled)
    result = prove_rust_transformer_v2(
        gpt2_model, hidden_state,
        prove_layers=12,
        precompiled=precompiled,
        server=server,
    )
"""

from __future__ import annotations

import numpy as np
import torch

from .constants import M31, QUANT_RANGE


def quantize_vector(v: np.ndarray, scale: float, as_numpy: bool = False):
    """Quantize a float vector to M31 field elements (u32-ready).

    Args:
        as_numpy: If True, return numpy uint32 array (faster for binary protocol).
                  If False, return Python list (for JSON/M31 field ops).
    """
    q = np.round(v / scale).astype(np.int64)
    result = q % M31
    result = np.where(result < 0, result + M31, result)
    arr = result.astype(np.uint32)
    return arr if as_numpy else arr.tolist()


# --- Removed: compile_model, prove, verify, prove_rust, prove_rust_gpt2 ---
# These were the legacy MLP proving path (Python-side sumcheck).
# All GPT-2 proving now goes through prove_rust_transformer_v2 below.

def _build_binary_payload(first_input_q, all_ops):
    """Build binary payload for the Rust prover (avoids JSON serialization overhead).

    Format: [0x00 magic] [u32 num_ops] [u32 input_len] [u32*input_len input]
    Per op: [u8 type] [u32 name_len] [name bytes] [op-specific data]
    """
    import struct

    def pack_u32_array(arr):
        """Pack a list/array of u32 values as little-endian bytes."""
        return np.asarray(arr, dtype=np.uint32).tobytes()

    parts = []
    parts.append(b'\x00')  # magic byte
    parts.append(struct.pack('<I', len(all_ops)))
    parts.append(struct.pack('<I', len(first_input_q)))
    parts.append(pack_u32_array(first_input_q))

    for op in all_ops:
        op_type = op["type"]
        name_bytes = op["name"].encode('utf-8')
        if op_type == "linear":
            parts.append(b'\x00')
            parts.append(struct.pack('<I', len(name_bytes)))
            parts.append(name_bytes)
            parts.append(struct.pack('<II', op["m"], op["n"]))
            parts.append(pack_u32_array(op["w_q"]))
            assert len(op["b_q"]) == op["m"], (
                f"Bias length mismatch for {op['name']}: "
                f"len(b_q)={len(op['b_q'])} != m={op['m']}"
            )
            parts.append(pack_u32_array(op["b_q"]))
        elif op_type == "relu":
            parts.append(b'\x01')
            parts.append(struct.pack('<I', len(name_bytes)))
            parts.append(name_bytes)
        elif op_type == "set_input":
            parts.append(b'\x02')
            parts.append(struct.pack('<I', len(name_bytes)))
            parts.append(name_bytes)
            new_input = op["new_input"]
            parts.append(struct.pack('<I', len(new_input)))
            parts.append(pack_u32_array(new_input))
        elif op_type == "layernorm":
            parts.append(b'\x03')
            parts.append(struct.pack('<I', len(name_bytes)))
            parts.append(name_bytes)
            gamma_q = op["gamma_q"]
            parts.append(struct.pack('<I', len(gamma_q)))
            parts.append(pack_u32_array(gamma_q))
            beta_q = op["beta_q"]
            parts.append(struct.pack('<I', len(beta_q)))
            parts.append(pack_u32_array(beta_q))
            # ln_output removed: Rust computes LN via QR perturbation
            parts.append(struct.pack('<I', 0))  # empty ln_output for protocol compat
        elif op_type == "save":
            parts.append(b'\x04')
            parts.append(struct.pack('<I', len(name_bytes)))
            parts.append(name_bytes)
            sname_bytes = op["save_name"].encode('utf-8')
            parts.append(struct.pack('<I', len(sname_bytes)))
            parts.append(sname_bytes)
        elif op_type == "add_saved":
            parts.append(b'\x05')
            parts.append(struct.pack('<I', len(name_bytes)))
            parts.append(name_bytes)
            aname_bytes = op["add_name"].encode('utf-8')
            parts.append(struct.pack('<I', len(aname_bytes)))
            parts.append(aname_bytes)
        elif op_type == "gelu":
            parts.append(b'\x06')
            parts.append(struct.pack('<I', len(name_bytes)))
            parts.append(name_bytes)
            # Send gelu_scale, gelu_input_i16, gelu_output_i16
            gelu_inp = op.get("gelu_input_i16")
            gelu_out = op.get("gelu_output_i16")
            gscale = op.get("gelu_scale", 1000)
            parts.append(struct.pack('<i', gscale))
            if gelu_inp is not None and gelu_out is not None:
                n_gelu = len(gelu_inp)
                parts.append(struct.pack('<I', n_gelu))
                # Send as i16 pairs
                for j in range(n_gelu):
                    parts.append(struct.pack('<hh', int(gelu_inp[j]), int(gelu_out[j])))
            else:
                parts.append(struct.pack('<I', 0))
        elif op_type == "attention":
            parts.append(b'\x07')
            parts.append(struct.pack('<I', len(name_bytes)))
            parts.append(name_bytes)
            parts.append(struct.pack('<III', op["num_heads"], op["seq_len"], op["d_head"]))
            parts.append(struct.pack('<i', op["exp_scale"]))
            # No external K/V for seq_len=1 (Rust splits c_attn output)
            parts.append(struct.pack('<I', 0))  # n_kv = 0 means split from current
        else:
            raise ValueError(f"Binary protocol does not support op type: {op_type}")

    return b''.join(parts)


class PrecompiledTransformerWeights:
    """Pre-quantized weight matrices for GPT-2 transformer blocks.

    Weight quantization is input-independent (scale = max(|w|) / 127),
    so we do it once at startup. Only biases need per-request quantization
    because their scale depends on the input activation scale chain.
    """

    def __init__(self, gpt2_model):
        import time
        t0 = time.time()

        try:
            from transformers.pytorch_utils import Conv1D
        except ImportError:
            Conv1D = None

        def get_weight(module) -> np.ndarray:
            w = module.weight.detach().numpy()
            if Conv1D is not None and isinstance(module, Conv1D):
                return w.T
            return w

        transformer = gpt2_model.transformer
        self.d_model = transformer.config.n_embd
        self.d_ff = transformer.config.n_inner or 4 * self.d_model
        self.n_layers = len(transformer.h)

        self.layers = []
        for i in range(self.n_layers):
            block = transformer.h[i]

            # c_attn (QKV projection) weights
            c_attn_w = get_weight(block.attn.c_attn)
            c_attn_w_s = np.abs(c_attn_w).max() / QUANT_RANGE
            c_attn_w_q = quantize_vector(c_attn_w.flatten(), c_attn_w_s, as_numpy=True)
            c_attn_b = block.attn.c_attn.bias.detach().numpy()

            # attn_proj (output projection) weights
            attn_w = get_weight(block.attn.c_proj)
            attn_w_s = np.abs(attn_w).max() / QUANT_RANGE
            attn_w_q = quantize_vector(attn_w.flatten(), attn_w_s, as_numpy=True)
            attn_b = block.attn.c_proj.bias.detach().numpy()

            # MLP c_fc weights
            fc1_w = get_weight(block.mlp.c_fc)
            fc1_w_s = np.abs(fc1_w).max() / QUANT_RANGE
            fc1_w_q = quantize_vector(fc1_w.flatten(), fc1_w_s, as_numpy=True)
            fc1_b = block.mlp.c_fc.bias.detach().numpy()

            # MLP c_proj weights
            fc2_w = get_weight(block.mlp.c_proj)
            fc2_w_s = np.abs(fc2_w).max() / QUANT_RANGE
            fc2_w_q = quantize_vector(fc2_w.flatten(), fc2_w_s, as_numpy=True)
            fc2_b = block.mlp.c_proj.bias.detach().numpy()

            # LayerNorm parameters (quantized for proving)
            ln1_g = block.ln_1.weight.detach().numpy()
            ln1_b = block.ln_1.bias.detach().numpy()
            ln1_g_s = np.abs(ln1_g).max() / QUANT_RANGE if np.abs(ln1_g).max() > 0 else 1.0
            ln1_b_s = np.abs(ln1_b).max() / QUANT_RANGE if np.abs(ln1_b).max() > 0 else 1.0
            ln1_g_q = quantize_vector(ln1_g, ln1_g_s, as_numpy=True)
            ln1_b_q = quantize_vector(ln1_b, ln1_b_s, as_numpy=True)

            ln2_g = block.ln_2.weight.detach().numpy()
            ln2_b = block.ln_2.bias.detach().numpy()
            ln2_g_s = np.abs(ln2_g).max() / QUANT_RANGE if np.abs(ln2_g).max() > 0 else 1.0
            ln2_b_s = np.abs(ln2_b).max() / QUANT_RANGE if np.abs(ln2_b).max() > 0 else 1.0
            ln2_g_q = quantize_vector(ln2_g, ln2_g_s, as_numpy=True)
            ln2_b_q = quantize_vector(ln2_b, ln2_b_s, as_numpy=True)

            self.layers.append({
                "c_attn_w": c_attn_w, "c_attn_w_q": c_attn_w_q, "c_attn_w_s": c_attn_w_s, "c_attn_b": c_attn_b,
                "attn_w": attn_w, "attn_w_q": attn_w_q, "attn_w_s": attn_w_s, "attn_b": attn_b,
                "fc1_w": fc1_w, "fc1_w_q": fc1_w_q, "fc1_w_s": fc1_w_s, "fc1_b": fc1_b,
                "fc2_w": fc2_w, "fc2_w_q": fc2_w_q, "fc2_w_s": fc2_w_s, "fc2_b": fc2_b,
                "ln1_g_q": ln1_g_q, "ln1_b_q": ln1_b_q,
                "ln2_g_q": ln2_g_q, "ln2_b_q": ln2_b_q,
            })

        # Final LayerNorm + LM head
        final_ln = transformer.ln_f
        final_ln_g = final_ln.weight.detach().numpy()
        final_ln_b = final_ln.bias.detach().numpy()
        final_ln_g_s = np.abs(final_ln_g).max() / QUANT_RANGE if np.abs(final_ln_g).max() > 0 else 1.0
        final_ln_b_s = np.abs(final_ln_b).max() / QUANT_RANGE if np.abs(final_ln_b).max() > 0 else 1.0
        self.final_ln_g_q = quantize_vector(final_ln_g, final_ln_g_s, as_numpy=True)
        self.final_ln_b_q = quantize_vector(final_ln_b, final_ln_b_s, as_numpy=True)

        # lm_head (50257 × 768, bias=False, weight-tied with wte)
        lm_head_w = gpt2_model.lm_head.weight.detach().numpy()  # (50257, 768)
        self.lm_head_m = lm_head_w.shape[0]  # 50257
        self.lm_head_n = lm_head_w.shape[1]  # 768
        self.lm_head_w_s = np.abs(lm_head_w).max() / QUANT_RANGE
        self.lm_head_w_q = quantize_vector(lm_head_w.flatten(), self.lm_head_w_s, as_numpy=True)

        elapsed = time.time() - t0
        print(f"Pre-compiled {self.n_layers} transformer layers + lm_head ({self.lm_head_m}x{self.lm_head_n}) in {elapsed:.2f}s")


class LlamaPrecompiledWeights:
    """Pre-quantized weight matrices for Llama-style transformer blocks.

    Quantizes all 9 weight matrices per layer (q/k/v/o proj, gate/up/down proj,
    2x RMSNorm gamma) once at init time. Returns named weight entries for
    RustProverServer preloading.
    """

    def __init__(self, extractor, weights: dict):
        import time
        from .model_extractor import quantize_symmetric

        t0 = time.time()
        self.config = extractor.config
        self.n_layers = self.config.n_layers
        sd = extractor.model.state_dict()

        self.weight_entries = []  # list of (name, w_q_array, m, n)

        for i in range(self.n_layers):
            prefix = f"model.layers.{i}"

            # RMSNorm gammas (d_model × 1, treated as vectors)
            norm1_gamma = sd[f"{prefix}.input_layernorm.weight"].float().numpy()
            norm1_q, _ = quantize_symmetric(norm1_gamma)
            self.weight_entries.append((f"layer{i}.norm1_gamma", norm1_q, len(norm1_q), 1))

            norm2_gamma = sd[f"{prefix}.post_attention_layernorm.weight"].float().numpy()
            norm2_q, _ = quantize_symmetric(norm2_gamma)
            self.weight_entries.append((f"layer{i}.norm2_gamma", norm2_q, len(norm2_q), 1))

            # Projection weights
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                qw = weights[f"layer{i}.{proj}"]
                w_q = qw.w_q  # already uint32 numpy array
                # Determine dimensions from the weight shape
                total = len(w_q)
                if proj in ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'):
                    # (out_dim, d_model) flattened
                    n = self.config.d_model
                    m = total // n
                elif proj == 'o_proj':
                    # (d_model, q_dim) flattened
                    m = self.config.d_model
                    n = total // m
                else:  # down_proj
                    # (d_model, d_ff) flattened
                    m = self.config.d_model
                    n = total // m
                self.weight_entries.append((f"layer{i}.{proj}", w_q, m, n))

        elapsed = time.time() - t0
        print(f"Pre-compiled {self.n_layers} Llama layers ({len(self.weight_entries)} weight matrices) in {elapsed:.2f}s")


class RustProverServer:
    """Persistent Rust prover subprocess with pre-loaded weights.

    Eliminates per-request overhead of: process spawn, weight transfer,
    u32→F conversion, and weight commitment computation.
    """

    # Map pcs_mode to binary suffix
    PCS_BINARY_MAP = {
        "default": "zk_ml_prover",
        "pcs": "zk_ml_prover_pcs",
        "pcs-full": "zk_ml_prover_pcs_full",
    }

    def __init__(self, precompiled=None, weight_entries=None, gpu=False, pcs_mode="default"):
        """Start server with preloaded weights.

        Args:
            precompiled: PrecompiledTransformerWeights (GPT-2 path)
            weight_entries: list of (name, w_q_array, m, n) tuples (generic path)
                           Used by LlamaPrecompiledWeights.
            gpu: if True, pass --gpu flag to enable Metal GPU acceleration.
            pcs_mode: "default", "pcs", or "pcs-full". Selects which binary to use.
        """
        import struct, subprocess, os, time

        self.pcs_mode = pcs_mode
        binary_name = self.PCS_BINARY_MAP.get(pcs_mode, "zk_ml_prover")
        rust_bin = os.path.join(
            os.path.dirname(__file__), "..", "..", "rust",
            "zk_ml_prover", "target", "release", binary_name
        )
        if not os.path.exists(rust_bin):
            raise FileNotFoundError(f"Rust prover not found at {rust_bin}")

        cmd = [rust_bin]
        if gpu:
            cmd.append("--gpu")

        t0 = time.time()
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Drain stderr in background thread to prevent pipe buffer deadlock
        import threading, io
        self._stderr_buf = io.BytesIO()
        def _drain_stderr():
            try:
                while True:
                    chunk = self.proc.stderr.read(4096)
                    if not chunk:
                        break
                    self._stderr_buf.write(chunk)
            except Exception:
                pass
        self._stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        self._stderr_thread.start()

        # Send server mode magic byte + preload weights
        parts = [b'\x01']  # server mode magic

        # Build weight entries from either source
        if weight_entries is None:
            weight_entries = []
        else:
            weight_entries = list(weight_entries)

        if precompiled is not None and isinstance(precompiled, PrecompiledTransformerWeights):
            for i in range(precompiled.n_layers):
                lw = precompiled.layers[i]
                weight_entries.append((f"c_attn_{i}", lw["c_attn_w_q"], lw["c_attn_w"].shape[0], lw["c_attn_w"].shape[1]))
                weight_entries.append((f"c_proj_{i}", lw["attn_w_q"], lw["attn_w"].shape[0], lw["attn_w"].shape[1]))
                weight_entries.append((f"mlp_fc_{i}", lw["fc1_w_q"], lw["fc1_w"].shape[0], lw["fc1_w"].shape[1]))
                weight_entries.append((f"mlp_proj_{i}", lw["fc2_w_q"], lw["fc2_w"].shape[0], lw["fc2_w"].shape[1]))
            weight_entries.append(("lm_head", precompiled.lm_head_w_q, precompiled.lm_head_m, precompiled.lm_head_n))
            self._precompiled = precompiled

        # Write the small magic + preamble parts first.
        parts.append(struct.pack('<I', len(weight_entries)))
        preamble = b''.join(parts)
        self.proc.stdin.write(preamble)

        # Stream weights one at a time. Avoids holding all bytes in Python
        # heap simultaneously (critical for 9B+ models on memory-constrained
        # systems — concatenating 36GB+ before writing OOMs).
        for name, w_q, m, n in weight_entries:
            name_bytes = name.encode('utf-8')
            self.proc.stdin.write(struct.pack('<I', len(name_bytes)))
            self.proc.stdin.write(name_bytes)
            self.proc.stdin.write(struct.pack('<II', m, n))
            arr = np.asarray(w_q, dtype=np.uint32)
            self.proc.stdin.write(arr.tobytes())
            del arr  # release temp copy if asarray made one
        self.proc.stdin.flush()

        elapsed = time.time() - t0
        print(f"Rust prover server started, {len(weight_entries)} weights preloaded in {elapsed:.2f}s")

    def prove(self, first_input_q, server_ops) -> dict:
        """Send a prove request and return the result.

        server_ops is a list of dicts:
          {"type": "linear_ref", "name": ..., "weight_name": ..., "bias_q": np.array}
          {"type": "relu", "name": ...}
          {"type": "set_input", "name": ..., "new_input": np.array}
        """
        import struct, json

        def pack_u32_array(arr):
            return np.asarray(arr, dtype=np.uint32).tobytes()

        parts = []
        parts.append(struct.pack('<I', len(server_ops)))
        parts.append(struct.pack('<I', len(first_input_q)))
        parts.append(pack_u32_array(first_input_q))

        for op in server_ops:
            name_bytes = op["name"].encode('utf-8')
            if op["type"] == "linear_ref":
                parts.append(b'\x00')
                parts.append(struct.pack('<I', len(name_bytes)))
                parts.append(name_bytes)
                wname_bytes = op["weight_name"].encode('utf-8')
                parts.append(struct.pack('<I', len(wname_bytes)))
                parts.append(wname_bytes)
                bias_q = op["bias_q"]
                parts.append(struct.pack('<I', len(bias_q)))
                parts.append(pack_u32_array(bias_q))
            elif op["type"] == "relu":
                parts.append(b'\x01')
                parts.append(struct.pack('<I', len(name_bytes)))
                parts.append(name_bytes)
            elif op["type"] == "set_input":
                parts.append(b'\x02')
                parts.append(struct.pack('<I', len(name_bytes)))
                parts.append(name_bytes)
                new_input = op["new_input"]
                parts.append(struct.pack('<I', len(new_input)))
                parts.append(pack_u32_array(new_input))
            elif op["type"] == "layernorm":
                parts.append(b'\x03')
                parts.append(struct.pack('<I', len(name_bytes)))
                parts.append(name_bytes)
                gamma_q = op["gamma_q"]
                parts.append(struct.pack('<I', len(gamma_q)))
                parts.append(pack_u32_array(gamma_q))
                beta_q = op["beta_q"]
                parts.append(struct.pack('<I', len(beta_q)))
                parts.append(pack_u32_array(beta_q))
                # ln_output removed: Rust computes LN via QR perturbation
                parts.append(struct.pack('<I', 0))  # empty ln_output for protocol compat
            elif op["type"] == "save":
                parts.append(b'\x04')
                parts.append(struct.pack('<I', len(name_bytes)))
                parts.append(name_bytes)
                sname_bytes = op["save_name"].encode('utf-8')
                parts.append(struct.pack('<I', len(sname_bytes)))
                parts.append(sname_bytes)
            elif op["type"] == "add_saved":
                parts.append(b'\x05')
                parts.append(struct.pack('<I', len(name_bytes)))
                parts.append(name_bytes)
                aname_bytes = op["add_name"].encode('utf-8')
                parts.append(struct.pack('<I', len(aname_bytes)))
                parts.append(aname_bytes)
            elif op["type"] == "gelu":
                parts.append(b'\x06')
                parts.append(struct.pack('<I', len(name_bytes)))
                parts.append(name_bytes)
                # Send gelu_scale, gelu_input_i16, gelu_output_i16
                gelu_inp = op.get("gelu_input_i16")
                gelu_out_arr = op.get("gelu_output_i16")
                gscale = op.get("gelu_scale", 1000)
                parts.append(struct.pack('<i', gscale))
                if gelu_inp is not None and gelu_out_arr is not None:
                    n_gelu = len(gelu_inp)
                    parts.append(struct.pack('<I', n_gelu))
                    for j in range(n_gelu):
                        parts.append(struct.pack('<hh', int(gelu_inp[j]), int(gelu_out_arr[j])))
                else:
                    parts.append(struct.pack('<I', 0))
            elif op["type"] == "attention":
                parts.append(b'\x07')
                parts.append(struct.pack('<I', len(name_bytes)))
                parts.append(name_bytes)
                parts.append(struct.pack('<III', op["num_heads"], op["seq_len"], op["d_head"]))
                parts.append(struct.pack('<i', op["exp_scale"]))
                parts.append(struct.pack('<I', 0))  # n_kv = 0 (split from current)
            elif op["type"] == "llama_layer_ref":
                parts.append(b'\x0D')  # 13 = llama_layer_ref
                parts.append(struct.pack('<I', len(name_bytes)))
                parts.append(name_bytes)
                cfg = op["config"]
                parts.append(struct.pack('<IIIII',
                    cfg["d_model"], cfg["d_ff"], cfg["num_q_heads"],
                    cfg["num_kv_heads"], cfg["d_head"]))
                parts.append(struct.pack('<i', cfg["silu_scale"]))
                # 9 weight name references
                for wname in op["weight_names"]:
                    wb = wname.encode('utf-8')
                    parts.append(struct.pack('<I', len(wb)))
                    parts.append(wb)
            elif op["type"] == "qwen_layer_ref":
                parts.append(b'\x0F')  # 15 = qwen_layer_ref
                parts.append(struct.pack('<I', len(name_bytes)))
                parts.append(name_bytes)
                cfg = op["config"]
                # Header: d_model, d_ff, num_q_heads, num_kv_heads, d_head, v_num_heads, v_d_head
                # v_num_heads/v_d_head added for asymmetric GDN (Qwen3.5-4B/9B). Defaults to
                # num_kv_heads/d_head when missing for backward compat.
                v_num_heads = cfg.get("v_num_heads", cfg["num_kv_heads"])
                v_d_head = cfg.get("v_d_head", cfg["d_head"])
                parts.append(struct.pack('<IIIIIII',
                    cfg["d_model"], cfg["d_ff"], cfg["num_q_heads"],
                    cfg["num_kv_heads"], cfg["d_head"],
                    v_num_heads, v_d_head))
                parts.append(struct.pack('<i', cfg["silu_scale"]))
                parts.append(struct.pack('<i', cfg["sigmoid_scale"]))
                # 10 weight name references
                for wname in op["weight_names"]:
                    wb = wname.encode('utf-8')
                    parts.append(struct.pack('<I', len(wb)))
                    parts.append(wb)

        payload = b''.join(parts)
        # Send length-prefixed payload
        self.proc.stdin.write(struct.pack('<I', len(payload)))
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

        # Read response line
        line = self.proc.stdout.readline()
        if not line:
            stderr_out = self._stderr_buf.getvalue().decode('utf-8', errors='replace')
            raise RuntimeError(f"Rust prover server died. stderr: {stderr_out}")
        return json.loads(line)

    def shutdown(self):
        import struct
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.write(struct.pack('<I', 0))  # shutdown signal
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)

    def stop(self):
        """Stop the server process. Used for on-demand PCS mode switching."""
        self.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


def prove_rust_transformer_v2(
    gpt2_model,
    input_hidden: torch.Tensor,
    prove_layers: int = 1,
    start_layer: int = -1,
    gelu_scale: int = 1000,
    verbose: bool = True,
    precompiled: PrecompiledTransformerWeights = None,
    server: RustProverServer = None,
) -> dict:
    """Prove GPT-2 transformer blocks via MLP-mode ops with scale tracking.

    Builds the full op list (LayerNorm, matmul, GELU, residual add) and sends
    to the Rust prover in MLP mode. This avoids the broken quantization in
    the GPT-2 mode by using the same scale-tracking approach as prove_rust().

    Each transformer block becomes:
        save("residual")  → LN1 → c_proj (attention proxy) → add_saved("residual")
        → save("residual2") → LN2 → c_fc → GELU → mlp_proj → add_saved("residual2")
    """
    import json, os, subprocess

    transformer = gpt2_model.transformer
    d_model = transformer.config.n_embd
    d_ff = transformer.config.n_inner or 4 * d_model
    n_layers = len(transformer.h)

    # Use precompiled weights if available, otherwise extract on the fly
    has_precompiled = precompiled is not None

    if not has_precompiled:
        try:
            from transformers.pytorch_utils import Conv1D
        except ImportError:
            Conv1D = None

        def get_weight(module) -> np.ndarray:
            w = module.weight.detach().numpy()
            if Conv1D is not None and isinstance(module, Conv1D):
                return w.T
            return w

    if start_layer < 0:
        start_layer = n_layers - prove_layers
    prove_layers = min(prove_layers, n_layers - start_layer)

    if verbose:
        print(f"GPT-2 v2: d_model={d_model}, d_ff={d_ff}")
        print(f"Proving layers {start_layer}..{start_layer + prove_layers - 1} of {n_layers}")

    import time as _time
    _t_start = _time.time()

    # Quantize input
    x_float = input_hidden.detach().numpy().flatten()
    x_max = np.abs(x_float).max()
    x_scale = x_max / QUANT_RANGE if x_max > 0 else 1.0
    x_q = quantize_vector(x_float, x_scale, as_numpy=True)

    # Strategy: LayerNorm and residual connections are computed in Python (float).
    # Only matmul + ReLU chains are sent to the Rust prover.
    # Each transformer layer produces 2 proof segments:
    #   Segment A: LN1_output → attn_proj (1 linear)
    #   Segment B: LN2_output → c_fc → ReLU → mlp_proj (2 linear + 1 relu)
    # Each segment is an independent proof with its own quantized input.

    import json, os, subprocess

    rust_bin = os.path.join(
        os.path.dirname(__file__), "..", "..", "rust",
        "zk_ml_prover", "target", "release", "zk_ml_prover"
    )
    if not os.path.exists(rust_bin):
        raise FileNotFoundError(f"Rust prover not found at {rust_bin}")

    # Run PyTorch forward pass to get exact intermediate values
    current_float = x_float.copy()
    segments = []  # list of (name, input_float, ops_list)

    for i in range(start_layer, start_layer + prove_layers):
        block = transformer.h[i]
        residual = current_float.copy()

        # LN1 (in float)
        with torch.no_grad():
            ln1_out = block.ln_1(torch.from_numpy(current_float).unsqueeze(0).float())
            ln1_float = ln1_out.squeeze(0).numpy()

        if has_precompiled:
            lw = precompiled.layers[i]
            c_attn_w, c_attn_b = lw["c_attn_w"], lw["c_attn_b"]
            attn_w, attn_b = lw["attn_w"], lw["attn_b"]
            fc1_w, fc1_b = lw["fc1_w"], lw["fc1_b"]
            fc2_w, fc2_b = lw["fc2_w"], lw["fc2_b"]
            ln1_g_q, ln1_b_q = lw["ln1_g_q"], lw["ln1_b_q"]
            ln2_g_q, ln2_b_q = lw["ln2_g_q"], lw["ln2_b_q"]
        else:
            c_attn_w = get_weight(block.attn.c_attn)
            c_attn_b = block.attn.c_attn.bias.detach().numpy()
            attn_w = get_weight(block.attn.c_proj)
            attn_b = block.attn.c_proj.bias.detach().numpy()
            fc1_w = get_weight(block.mlp.c_fc)
            fc1_b = block.mlp.c_fc.bias.detach().numpy()
            fc2_w = get_weight(block.mlp.c_proj)
            fc2_b = block.mlp.c_proj.bias.detach().numpy()
            ln1_g_q = quantize_vector(block.ln_1.weight.detach().numpy(),
                                       np.abs(block.ln_1.weight.detach().numpy()).max() / QUANT_RANGE, as_numpy=True)
            ln1_b_q = quantize_vector(block.ln_1.bias.detach().numpy(),
                                       max(np.abs(block.ln_1.bias.detach().numpy()).max() / QUANT_RANGE, 1e-10), as_numpy=True)
            ln2_g_q = quantize_vector(block.ln_2.weight.detach().numpy(),
                                       np.abs(block.ln_2.weight.detach().numpy()).max() / QUANT_RANGE, as_numpy=True)
            ln2_b_q = quantize_vector(block.ln_2.bias.detach().numpy(),
                                       max(np.abs(block.ln_2.bias.detach().numpy()).max() / QUANT_RANGE, 1e-10), as_numpy=True)

        # Run attention in PyTorch to get reference values
        with torch.no_grad():
            ln1_3d = ln1_out.unsqueeze(0) if ln1_out.dim() == 2 else ln1_out  # (1, 1, 768)
            attn_output = block.attn(ln1_3d)[0]  # (1, 1, 768)
            attn_float = attn_output.squeeze().numpy()

        # Compute attention exp_scale for softmax lookup table.
        # For seq_len=1, scores are scalar per head (Q·K = dot product of d_head values).
        # Score range depends on c_attn output magnitude. Use conservative scale.
        # Score values are in M31, so we requantize to INT16 range for exp lookup.
        # For seq_len=1, softmax of single element = 1.0, output = V.
        # We still need the exp_table for the proof machinery.
        d_model = c_attn_w.shape[1]  # 768
        num_heads = 12  # GPT-2 standard
        d_head = d_model // num_heads  # 64
        seq_len = 1  # single-token inference

        # Segment A: save → LN1 → c_attn → attention → c_proj → add_saved
        # Attention splits c_attn output (2304) into Q(768), K(768), V(768),
        # runs multi-head attention, produces 768-dim output for c_proj.
        # exp_scale controls the INT16 range for exp lookup table.
        # For seq_len=1, scores are tiny (single dot product in M31), so large scale is fine.
        exp_scale = 1000  # conservative scale for exp lookup
        segments.append((f"layer_{i}_a", current_float, [
            {"type": "save", "save_name": f"res1_{i}", "name": f"save_res1_{i}"},
            {"type": "layernorm", "gamma_q": ln1_g_q, "beta_q": ln1_b_q,
             "name": f"ln1_{i}"},
            {"w": c_attn_w, "b": c_attn_b, "name": f"c_attn_{i}"},
            {"type": "attention", "name": f"attn_{i}",
             "num_heads": num_heads, "seq_len": seq_len, "d_head": d_head,
             "exp_scale": exp_scale},
            {"w": attn_w, "b": attn_b, "name": f"c_proj_{i}"},
            {"type": "add_saved", "add_name": f"res1_{i}", "name": f"add_res1_{i}"},
        ]))

        # Continue float forward: attention output + residual
        current_float = attn_float + residual

        # LN2 (in float)
        with torch.no_grad():
            ln2_out = block.ln_2(torch.from_numpy(current_float).unsqueeze(0).float())
            ln2_float = ln2_out.squeeze(0).numpy()
        residual2 = current_float.copy()

        # Float forward: compute fc1 → GELU → fc2 + residual
        fc1_out = fc1_w @ ln2_float + fc1_b
        # GPT-2 uses GELU activation (not ReLU)
        # numpy tanh approximation: GELU(x) ≈ 0.5x(1 + tanh(sqrt(2/π)(x + 0.044715x³)))
        gelu_out = 0.5 * fc1_out * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (fc1_out + 0.044715 * fc1_out**3)))

        # Requantize fc1 output to INT16 for GELU lookup proof
        # Convention: input_i16 = round(x * scale), so x = input_i16 / scale
        # Choose scale so max(|fc1_out|) * scale fits in i16 (±32767)
        fc1_max = float(np.abs(fc1_out).max())
        if fc1_max > 0:
            gelu_scale = int(32767.0 / fc1_max)
        else:
            gelu_scale = 1000
        gelu_scale = max(1, gelu_scale)
        gelu_input_i16 = np.clip(np.round(fc1_out * gelu_scale), -32768, 32767).astype(np.int16)
        # Compute GELU output from quantized inputs (must match Rust table exactly)
        gelu_input_float = gelu_input_i16.astype(np.float64) / gelu_scale
        gelu_out_from_table = 0.5 * gelu_input_float * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (gelu_input_float + 0.044715 * gelu_input_float**3))
        )
        gelu_output_i16 = np.clip(np.round(gelu_out_from_table * gelu_scale), -32768, 32767).astype(np.int16)

        # Segment B1: post_attn+residual → save(res2) → layernorm(ln2) → c_fc → gelu
        # fc1 matmul proved via sumcheck. GELU proved via LogUp lookup on requantized vectors.
        # Requantization (fc1_float → gelu_input_i16) is trusted.
        segments.append((f"mlp_fc_{i}", current_float, [
            {"type": "save", "save_name": f"res2_{i}", "name": f"save_res2_{i}"},
            {"type": "layernorm", "gamma_q": ln2_g_q, "beta_q": ln2_b_q,
             "name": f"ln2_{i}"},
            {"w": fc1_w, "b": fc1_b, "name": f"mlp_fc_{i}"},
            {"type": "gelu", "name": f"gelu_{i}",
             "gelu_input_i16": gelu_input_i16, "gelu_output_i16": gelu_output_i16,
             "gelu_scale": gelu_scale},
        ]))

        # Segment B2: GELU output (from Python float forward) → fc2 → add_saved(res2)
        segments.append((f"mlp_proj_{i}", gelu_out, [
            {"w": fc2_w, "b": fc2_b, "name": f"mlp_proj_{i}"},
            {"type": "add_saved", "add_name": f"res2_{i}", "name": f"add_res2_{i}"},
        ]))

        mlp_out = fc2_w @ gelu_out + fc2_b
        current_float = mlp_out + residual2

        if verbose:
            print(f"  Layer {i}: 3 segments (LN1+c_attn+attn+c_proj+res, LN2+fc1+gelu, fc2+res)")

    # Final segment: final_ln → lm_head
    if has_precompiled and hasattr(precompiled, 'lm_head_w_q'):
        # Compute final LN in float for reference
        with torch.no_grad():
            final_ln_out = transformer.ln_f(torch.from_numpy(current_float).unsqueeze(0).float())
            final_ln_float = final_ln_out.squeeze(0).numpy()

        lm_head_w = gpt2_model.lm_head.weight.detach().numpy()  # (50257, 768)
        lm_head_b = np.zeros(precompiled.lm_head_m)  # no bias

        segments.append((f"lm_head", current_float, [
            {"type": "layernorm", "gamma_q": precompiled.final_ln_g_q,
             "beta_q": precompiled.final_ln_b_q,
             "name": "final_ln"},
            {"w": lm_head_w, "b": lm_head_b, "name": "lm_head"},
        ]))
        if verbose:
            print(f"  Final: LN + lm_head ({precompiled.lm_head_m}x{precompiled.lm_head_n})")

    _t_segments = _time.time()
    if verbose:
        print(f"  [timing] Float forward + segments: {(_t_segments - _t_start)*1000:.0f}ms")

    # Build a single combined ops list. Each segment gets its own quantized input
    # injected via a special "set_input" mechanism. Since the Rust prover doesn't
    # Concatenate all segments into one op list using set_input between segments.
    # Single Rust call — avoids subprocess overhead.
    all_ops = []
    first_input_q = None

    for idx, (seg_name, seg_input, seg_ops) in enumerate(segments):
        seg_max = np.abs(seg_input).max()
        seg_scale = seg_max / QUANT_RANGE if seg_max > 0 else 1.0
        seg_x_q = quantize_vector(seg_input, seg_scale, as_numpy=True)

        if idx == 0:
            first_input_q = seg_x_q
        else:
            # Inject new quantized input for this segment
            all_ops.append({
                "type": "set_input",
                "name": seg_name,
                "new_input": seg_x_q,
            })

        cur_scale = seg_scale
        for op in seg_ops:
            op_type = op.get("type", "linear")
            if op_type == "relu":
                all_ops.append({"type": "relu", "name": op["name"]})
            elif op_type == "gelu":
                all_ops.append({
                    "type": "gelu", "name": op["name"],
                    "gelu_input_i16": op.get("gelu_input_i16"),
                    "gelu_output_i16": op.get("gelu_output_i16"),
                    "gelu_scale": op.get("gelu_scale", 1000),
                })
            elif op_type == "attention":
                all_ops.append({
                    "type": "attention", "name": op["name"],
                    "num_heads": op["num_heads"], "seq_len": op["seq_len"],
                    "d_head": op["d_head"], "exp_scale": op["exp_scale"],
                })
            elif op_type == "save":
                all_ops.append({"type": "save", "name": op["name"], "save_name": op["save_name"]})
            elif op_type == "add_saved":
                all_ops.append({"type": "add_saved", "name": op["name"], "add_name": op["add_name"]})
            elif op_type == "layernorm":
                all_ops.append({
                    "type": "layernorm", "name": op["name"],
                    "gamma_q": op["gamma_q"], "beta_q": op["beta_q"],
                })
                # LayerNorm doesn't change the scale (it normalizes)
                # After LN, output scale is determined by gamma scale
                # For simplicity, keep cur_scale unchanged — LN output is re-quantized
            else:
                # Linear op
                w = op["w"]
                b = op["b"]
                op_name = op["name"]

                # Look up precompiled weight quantization if available
                pre_w_q = None
                pre_w_s = None
                if has_precompiled:
                    if op_name == "lm_head":
                        pre_w_q, pre_w_s = precompiled.lm_head_w_q, precompiled.lm_head_w_s
                    else:
                        layer_idx = int(op_name.split("_")[-1])
                        lw = precompiled.layers[layer_idx]
                        if "c_attn" in op_name:
                            pre_w_q, pre_w_s = lw["c_attn_w_q"], lw["c_attn_w_s"]
                        elif "c_proj" in op_name:
                            pre_w_q, pre_w_s = lw["attn_w_q"], lw["attn_w_s"]
                        elif "mlp_fc" in op_name:
                            pre_w_q, pre_w_s = lw["fc1_w_q"], lw["fc1_w_s"]
                        elif "mlp_proj" in op_name:
                            pre_w_q, pre_w_s = lw["fc2_w_q"], lw["fc2_w_s"]

                if pre_w_q is not None:
                    w_s = pre_w_s
                    w_q = pre_w_q
                else:
                    w_s = np.abs(w).max() / QUANT_RANGE
                    w_q = quantize_vector(w.flatten(), w_s, as_numpy=True)

                out_s = w_s * cur_scale
                all_ops.append({
                    "type": "linear",
                    "name": op_name,
                    "m": w.shape[0],
                    "n": w.shape[1],
                    "w_q": w_q,
                    "b_q": quantize_vector(b, out_s, as_numpy=True),
                })
                cur_scale = out_s

    _t_ops = _time.time()
    if verbose:
        print(f"  [timing] Ops + quantization: {(_t_ops - _t_segments)*1000:.0f}ms")

    if server is not None:
        # Server mode: build lightweight ops referencing preloaded weights
        server_ops = []
        for op in all_ops:
            if op["type"] == "linear":
                server_ops.append({
                    "type": "linear_ref",
                    "name": op["name"],
                    "weight_name": op["name"],
                    "bias_q": op["b_q"],
                })
            elif op["type"] == "relu":
                server_ops.append({"type": "relu", "name": op["name"]})
            elif op["type"] == "gelu":
                server_ops.append({
                    "type": "gelu", "name": op["name"],
                    "gelu_input_i16": op.get("gelu_input_i16"),
                    "gelu_output_i16": op.get("gelu_output_i16"),
                    "gelu_scale": op.get("gelu_scale", 1000),
                })
            elif op["type"] == "set_input":
                server_ops.append({
                    "type": "set_input",
                    "name": op["name"],
                    "new_input": op["new_input"],
                })
            elif op["type"] == "layernorm":
                server_ops.append({
                    "type": "layernorm",
                    "name": op["name"],
                    "gamma_q": op["gamma_q"],
                    "beta_q": op["beta_q"],
                })
            elif op["type"] == "save":
                server_ops.append({
                    "type": "save",
                    "name": op["name"],
                    "save_name": op["save_name"],
                })
            elif op["type"] == "add_saved":
                server_ops.append({
                    "type": "add_saved",
                    "name": op["name"],
                    "add_name": op["add_name"],
                })
            elif op["type"] == "attention":
                server_ops.append({
                    "type": "attention",
                    "name": op["name"],
                    "num_heads": op["num_heads"],
                    "seq_len": op["seq_len"],
                    "d_head": op["d_head"],
                    "exp_scale": op["exp_scale"],
                })

        _t_send = _time.time()
        result = server.prove(first_input_q, server_ops)
        _t_rust = _time.time()
        if verbose:
            print(f"  [timing] Server request: {(_t_send - _t_ops)*1000:.0f}ms build, {(_t_rust - _t_send)*1000:.0f}ms prove")
            print(f"  [timing] Total Python overhead: {(_t_send - _t_start)*1000:.0f}ms")
    else:
        binary_payload = _build_binary_payload(first_input_q, all_ops)
        _t_binary = _time.time()
        if verbose:
            print(f"  [timing] Binary payload: {(_t_binary - _t_ops)*1000:.0f}ms ({len(binary_payload)/1024/1024:.1f}MB)")

        proc = subprocess.run(
            [rust_bin],
            input=binary_payload,
            capture_output=True, timeout=300,
        )
        _t_rust = _time.time()
        if verbose:
            print(f"  [timing] Rust subprocess: {(_t_rust - _t_binary)*1000:.0f}ms")
            print(f"  [timing] Total Python overhead: {(_t_binary - _t_start)*1000:.0f}ms")

        result = None

    if result is None:
        # One-shot subprocess path
        if proc.returncode != 0:
            raise RuntimeError(f"Rust prover failed: {proc.stderr.decode()}")
        result = json.loads(proc.stdout.decode())

    try:
        total_prove_ms = result.get("prove_time_ms", 0)
        total_verify_ms = result.get("verify_time_ms", 0)
        total_proof_bytes = result.get("proof_size_bytes", 0)
        all_valid = result.get("valid", False)
    except (TypeError, AttributeError) as e:
        raise RuntimeError(
            f"Invalid Rust prover response (expected dict with prove_time_ms, "
            f"verify_time_ms, proof_size_bytes, valid): got {type(result).__name__}: {e}"
        ) from e
    total_ops = len([o for o in all_ops if o["type"] in ("linear", "relu", "gelu", "attention")])

    if verbose:
        print(f"\n--- Rust Transformer Prover v2 ---")
        print(f"  Valid: {all_valid}")
        print(f"  Segments: {len(segments)} ({total_ops} ops)")
        print(f"  Prove: {total_prove_ms:.1f}ms")
        print(f"  Verify: {total_verify_ms:.3f}ms")
        print(f"  Proof size: {total_proof_bytes} bytes")

    # Parse coverage from Rust response
    coverage = None
    if "coverage" in result and result["coverage"]:
        cov = result["coverage"]
        comp_count = cov.get("computational_count", cov.get("proved_count", 0))
        comp_total = cov.get("computational_total", comp_count)
        total_count = cov.get("total_count", 0)
        state_count = cov.get("state_count", 0)
        coverage = {
            "proved": cov.get("proved_ops", []),
            "state": cov.get("state_ops", []),
            "proved_count": cov.get("proved_count", 0),
            "state_count": state_count,
            "computational_count": comp_count,
            "computational_total": comp_total,
            "total_count": total_count,
            "computational_pct": round(
                comp_count / comp_total * 100, 1
            ) if comp_total > 0 else 0.0,
            "total_pct": round(
                (comp_count + state_count) / total_count * 100, 1
            ) if total_count > 0 else 0.0,
        }
        if verbose:
            print(f"  Computational: {coverage['computational_count']}/{coverage['computational_total']} ({coverage['computational_pct']}%)")
            print(f"  Total verified: {coverage['proved_count'] + state_count}/{total_count} ({coverage['total_pct']}%)")

    return {
        "valid": all_valid,
        "prediction": -1,  # not meaningful for intermediate layers
        "prove_time_ms": total_prove_ms,
        "verify_time_ms": total_verify_ms,
        "proof_size_bytes": total_proof_bytes,
        "weight_commitments": result.get("weight_commitments", []),
        "input_commitment": result.get("input_commitment", ""),
        "coverage": coverage,
    }

