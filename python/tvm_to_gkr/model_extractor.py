"""Generic HuggingFace model weight extraction and quantization.

Extracts weights from any transformer model (Llama, Mistral, Qwen, GPT-2, etc.)
and quantizes them for the ZK prover.

Usage:
    extractor = ModelExtractor("meta-llama/Llama-2-7b-hf")
    weights = extractor.extract()  # dict of name -> quantized numpy arrays
    config = extractor.config      # ModelConfig for Rust prover
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .constants import M31, QUANT_RANGE


@dataclass
class ModelConfig:
    """Architecture configuration matching Rust ModelConfig."""
    d_model: int
    d_ff: int
    num_q_heads: int
    num_kv_heads: int
    d_head: int
    n_layers: int
    vocab_size: int
    norm_type: str   # "layernorm" or "rmsnorm"
    activation: str  # "gelu" or "swiglu"
    model_type: str  # "gpt2", "llama", "mistral", "qwen2", "qwen3", "qwen3_5"
    layer_types: Optional[List[str]] = None  # per-layer type, e.g. ["gdn", "gdn", "gdn", "attn", ...]
    sigmoid_scale: int = 1000  # scale for sigmoid lookup (output gating)
    # GDN-specific config. K dim and V dim may differ (Qwen3.5-4B: V = 2*K).
    gdn_num_heads: int = 0      # GDN K/Q head count (e.g. 16)
    gdn_d_head: int = 0         # GDN K/Q head dim (e.g. 128)
    gdn_v_num_heads: int = 0    # GDN V head count (often == gdn_num_heads but not always)
    gdn_v_d_head: int = 0       # GDN V head dim


@dataclass
class QuantizedWeights:
    """Quantized weight matrix with scale info."""
    w_q: np.ndarray    # uint32 M31 field elements
    scale: float       # quantization scale
    shape: Tuple[int, ...]  # original shape
    name: str


def quantize_symmetric(w: np.ndarray, quant_range: int = QUANT_RANGE) -> Tuple[np.ndarray, float]:
    """INT8 symmetric quantization: scale = max(|w|) / quant_range."""
    w_flat = w.flatten().astype(np.float64)
    max_abs = np.max(np.abs(w_flat))
    if max_abs == 0:
        return np.zeros(len(w_flat), dtype=np.uint32), 1.0
    scale = max_abs / quant_range
    w_int = np.round(w_flat / scale).astype(np.int64)
    # Convert to M31 field elements (negative values become p - |val|)
    w_q = np.where(w_int >= 0, w_int, M31 + w_int).astype(np.uint64) % M31
    return w_q.astype(np.uint32), scale


def detect_model_config(hf_config) -> ModelConfig:
    """Detect architecture from HuggingFace config."""
    model_type = getattr(hf_config, 'model_type', 'unknown')

    if model_type == 'gpt2':
        return ModelConfig(
            d_model=hf_config.n_embd,
            d_ff=hf_config.n_inner or 4 * hf_config.n_embd,
            num_q_heads=hf_config.n_head,
            num_kv_heads=hf_config.n_head,  # GPT-2 is MHA
            d_head=hf_config.n_embd // hf_config.n_head,
            n_layers=hf_config.n_layer,
            vocab_size=hf_config.vocab_size,
            norm_type="layernorm",
            activation="gelu",
            model_type="gpt2",
        )
    elif model_type in ('llama', 'mistral', 'qwen2'):
        num_kv_heads = getattr(hf_config, 'num_key_value_heads',
                               getattr(hf_config, 'num_attention_heads', 32))
        d_head = hf_config.hidden_size // hf_config.num_attention_heads
        return ModelConfig(
            d_model=hf_config.hidden_size,
            d_ff=hf_config.intermediate_size,
            num_q_heads=hf_config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            d_head=d_head,
            n_layers=hf_config.num_hidden_layers,
            vocab_size=hf_config.vocab_size,
            norm_type="rmsnorm",
            activation="swiglu",
            model_type=model_type,
        )
    elif model_type in ('qwen3', 'qwen3_5', 'qwen3_5_text'):
        # Qwen3.5: hybrid GatedDeltaNet + full attention with output gating
        # May be a multimodal config with text_config nested inside
        tc = getattr(hf_config, 'text_config', hf_config)

        num_kv_heads = getattr(tc, 'num_key_value_heads',
                               getattr(tc, 'num_attention_heads', 32))
        d_head = getattr(tc, 'head_dim', tc.hidden_size // tc.num_attention_heads)

        # Build per-layer type list from config
        layer_types_raw = getattr(tc, 'layer_types', None)
        if layer_types_raw is not None:
            # Map HF names to our names
            layer_types_config = []
            for lt in layer_types_raw:
                if lt in ('linear_attention',):
                    layer_types_config.append('gdn')
                else:
                    layer_types_config.append('attn')
        else:
            # Default: 3 GDN + 1 Attn pattern repeated
            n = tc.num_hidden_layers
            layer_types_config = []
            for i in range(n):
                if (i + 1) % 4 == 0:
                    layer_types_config.append('attn')
                else:
                    layer_types_config.append('gdn')

        gdn_num_heads = getattr(tc, 'linear_num_key_heads', 16)
        gdn_d_head = getattr(tc, 'linear_key_head_dim', 128)
        # V may have different head config (e.g. Qwen3.5-4B: 32 v-heads vs 16 k-heads)
        gdn_v_num_heads = getattr(tc, 'linear_num_value_heads', gdn_num_heads)
        gdn_v_d_head = getattr(tc, 'linear_value_head_dim', gdn_d_head)

        return ModelConfig(
            d_model=tc.hidden_size,
            d_ff=tc.intermediate_size,
            num_q_heads=tc.num_attention_heads,
            num_kv_heads=num_kv_heads,
            d_head=d_head,
            n_layers=tc.num_hidden_layers,
            vocab_size=tc.vocab_size,
            norm_type="rmsnorm",
            activation="swiglu",
            model_type='qwen3_5',
            layer_types=layer_types_config,
            gdn_num_heads=gdn_num_heads,
            gdn_d_head=gdn_d_head,
            gdn_v_num_heads=gdn_v_num_heads,
            gdn_v_d_head=gdn_v_d_head,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _resolve_local_snapshot(model_name_or_path: str):
    """Find a local HF cache snapshot dir for `org/model`, else return path
    if it's already a local directory.

    Used by `_load_via_safetensors_direct` to bypass `AutoModel` for
    architectures that the installed `transformers` doesn't recognize
    (e.g. `qwen3_5` on transformers <= 4.57.6 — see 
    benchmark re-run notes)."""
    import os
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    # HF cache layout: ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<sha>/
    cache_root = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface/hub")
    folder = "models--" + model_name_or_path.replace("/", "--")
    snap_root = os.path.join(cache_root, folder, "snapshots")
    if not os.path.isdir(snap_root):
        return None
    snaps = sorted(os.listdir(snap_root))
    if not snaps:
        return None
    return os.path.join(snap_root, snaps[-1])


def _load_via_safetensors_direct(snapshot_dir: str, dtype=None):
    """Load `(config_namespace, state_dict)` directly from a HF snapshot
    without going through `AutoModel`. Returns a `(SimpleNamespace, dict)`
    pair that mimics the surface area `ModelExtractor` consumes.

    SAFETY (benchmark re-run): this path exists
    specifically for `qwen3_5` checkpoints where the installed
    `transformers` lacks the architecture class. It reads config.json
    as a nested `SimpleNamespace` (to match `hf_config.text_config.X`
    attribute access) and loads safetensors shards via `safetensors.torch`.
    Weights are kept in the file's stored dtype unless `dtype` is
    explicitly passed; the rest of the pipeline calls `.float().numpy()`
    on each tensor, so the in-memory dtype only affects peak RSS during
    extraction."""
    import json
    import os
    import torch
    from types import SimpleNamespace
    from safetensors.torch import load_file

    config_path = os.path.join(snapshot_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    def _to_namespace(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_namespace(x) for x in obj]
        return obj

    cfg_ns = _to_namespace(config_dict)

    # Discover all safetensors shards in the snapshot dir. The
    # safetensors.index.json lists which key lives in which shard, but
    # we just slurp every file and merge — same result, simpler code.
    raw_state_dict = {}
    shard_files = sorted(
        f for f in os.listdir(snapshot_dir)
        if f.endswith(".safetensors")
    )
    if not shard_files:
        raise RuntimeError(
            f"No .safetensors shards found in {snapshot_dir}; cannot bypass AutoModel"
        )
    for shard in shard_files:
        shard_path = os.path.join(snapshot_dir, shard)
        loaded = load_file(shard_path, device="cpu")
        if dtype is not None:
            loaded = {k: v.to(dtype) for k, v in loaded.items()}
        raw_state_dict.update(loaded)

    # SHAPE NORMALIZATION (benchmark re-run): Qwen3.5-VLM
    # checkpoints store the text backbone under `model.language_model.*` and
    # extra heads under `mtp.*` / `model.visual.*`. The extractor's
    # `_extract_qwen35_style` was written against the AutoModel-loaded view
    # which strips the `language_model` indirection, so we replicate that
    # here:
    #   - `model.language_model.X` → `model.X`
    #   - skip `model.visual.*` (vision encoder, not part of the text proof)
    #   - skip `mtp.*` (multi-token prediction head, not used at inference)
    # Tied embeddings: when `tie_word_embeddings: true` (Qwen3.5 default), the
    # checkpoint omits `lm_head.weight`; we synthesize it as a view onto
    # `embed_tokens.weight` so the extractor's lm_head lookup succeeds.
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith("model.visual.") or k.startswith("mtp."):
            continue
        if k.startswith("model.language_model."):
            new_k = "model." + k[len("model.language_model."):]
        else:
            new_k = k
        state_dict[new_k] = v

    tie_word_embeddings = bool(getattr(cfg_ns, "tie_word_embeddings", False)) or \
        bool(getattr(getattr(cfg_ns, "text_config", cfg_ns), "tie_word_embeddings", False))
    if "lm_head.weight" not in state_dict and tie_word_embeddings:
        embed_key = "model.embed_tokens.weight"
        if embed_key in state_dict:
            state_dict["lm_head.weight"] = state_dict[embed_key]

    return cfg_ns, state_dict


class _StaticStateDictModel:
    """Minimal `nn.Module`-shaped stand-in for the bits of `self.model`
    that `ModelExtractor` touches: just `.state_dict()` and `.eval()`.

    Used when `_load_via_safetensors_direct` is the loading path."""
    def __init__(self, sd):
        self._sd = sd
    def state_dict(self):
        return self._sd
    def eval(self):
        return self


class ModelExtractor:
    """Extract and quantize weights from any HuggingFace model."""

    def __init__(self, model_name_or_path: str, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig

        self.model_name = model_name_or_path
        load_dtype = dtype or torch.float32

        # Fast path: try AutoConfig + AutoModel. Falls back to a direct
        # safetensors load when the installed `transformers` doesn't
        # know the architecture (e.g. qwen3_5 on transformers <= 4.57.6).
        try:
            hf_config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True,
            )
            self.config = detect_model_config(hf_config)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=load_dtype,
                trust_remote_code=True,
            )
            self.model.eval()
            return
        except (ValueError, KeyError) as e:
            msg = str(e)
            if "model type" not in msg and "qwen3_5" not in msg:
                raise

        # Slow path: direct safetensors load. Only triggered when AutoConfig
        # raised a "model type X not recognized" — for any other failure
        # we re-raised above.
        snapshot_dir = _resolve_local_snapshot(model_name_or_path)
        if snapshot_dir is None:
            raise RuntimeError(
                f"AutoConfig rejected {model_name_or_path!r} and no local "
                f"HF cache snapshot was found. Either upgrade transformers "
                f"(`pip install git+https://github.com/huggingface/transformers.git`) "
                f"or pre-download the model into ~/.cache/huggingface/hub/."
            )
        cfg_ns, state_dict = _load_via_safetensors_direct(snapshot_dir, dtype=load_dtype)
        self.config = detect_model_config(cfg_ns)
        self.model = _StaticStateDictModel(state_dict)
        self.model.eval()

    def extract(self) -> Dict[str, QuantizedWeights]:
        """Extract and quantize all weights."""
        if self.config.model_type == 'gpt2':
            return self._extract_gpt2()
        elif self.config.model_type in ('llama', 'mistral', 'qwen2'):
            return self._extract_llama_style()
        elif self.config.model_type in ('qwen3', 'qwen3_5'):
            return self._extract_qwen35_style()
        else:
            raise ValueError(f"Unsupported: {self.config.model_type}")

    def _extract_llama_style(self) -> Dict[str, QuantizedWeights]:
        """Extract Llama/Mistral/Qwen weights."""
        import torch
        weights = {}
        sd = self.model.state_dict()
        c = self.config

        for layer_idx in range(c.n_layers):
            prefix = f"model.layers.{layer_idx}"

            # RMSNorm + attention + MLP weights for this layer
            try:
                norm1_w = sd[f"{prefix}.input_layernorm.weight"].float().numpy()
                norm2_w = sd[f"{prefix}.post_attention_layernorm.weight"].float().numpy()
            except KeyError as e:
                raise ValueError(
                    f"Missing norm weight in state dict for layer {layer_idx} "
                    f"(prefix={prefix}): {e}. Available keys with this prefix: "
                    f"{[k for k in sd.keys() if k.startswith(prefix)][:10]}"
                ) from e

            norm1_q, norm1_s = quantize_symmetric(norm1_w)
            norm2_q, norm2_s = quantize_symmetric(norm2_w)
            weights[f"layer{layer_idx}.norm1"] = QuantizedWeights(norm1_q, norm1_s, norm1_w.shape, f"layer{layer_idx}.norm1")
            weights[f"layer{layer_idx}.norm2"] = QuantizedWeights(norm2_q, norm2_s, norm2_w.shape, f"layer{layer_idx}.norm2")

            # QKV projections
            try:
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    key = f"{prefix}.self_attn.{proj_name}.weight"
                    w = sd[key].float().numpy()
                    w_q, scale = quantize_symmetric(w)
                    weights[f"layer{layer_idx}.{proj_name}"] = QuantizedWeights(
                        w_q, scale, w.shape, f"layer{layer_idx}.{proj_name}"
                    )
            except KeyError as e:
                raise ValueError(
                    f"Missing attention weight in state dict for layer {layer_idx}: {e}"
                ) from e

            # MLP: gate_proj, up_proj, down_proj
            try:
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    key = f"{prefix}.mlp.{proj_name}.weight"
                    w = sd[key].float().numpy()
                    w_q, scale = quantize_symmetric(w)
                    weights[f"layer{layer_idx}.{proj_name}"] = QuantizedWeights(
                        w_q, scale, w.shape, f"layer{layer_idx}.{proj_name}"
                    )
            except KeyError as e:
                raise ValueError(
                    f"Missing MLP weight in state dict for layer {layer_idx}: {e}"
                ) from e

        # Final RMSNorm
        try:
            final_norm = sd["model.norm.weight"].float().numpy()
        except KeyError as e:
            raise ValueError(f"Missing final norm weight 'model.norm.weight' in state dict: {e}") from e
        fq, fs = quantize_symmetric(final_norm)
        weights["final_norm"] = QuantizedWeights(fq, fs, final_norm.shape, "final_norm")

        # LM head
        try:
            lm_head = sd["lm_head.weight"].float().numpy()
        except KeyError as e:
            raise ValueError(f"Missing LM head weight 'lm_head.weight' in state dict: {e}") from e
        lq, ls = quantize_symmetric(lm_head)
        weights["lm_head"] = QuantizedWeights(lq, ls, lm_head.shape, "lm_head")

        return weights

    def _extract_qwen35_style(self) -> Dict[str, QuantizedWeights]:
        """Extract Qwen3.5 weights (GatedDeltaNet + full attention with output gating).

        GDN layers: linear_attn.in_proj_qkv (fused) → split to Q/K/V,
                    linear_attn.in_proj_z → gate, linear_attn.conv1d → for folding
        Full attn:  self_attn.q_proj (packed Q+gate) → split,
                    self_attn.{k,v,o}_proj as-is
        """
        weights = {}
        sd = self.model.state_dict()
        c = self.config
        gdn_k_dim = c.gdn_num_heads * c.gdn_d_head      # K (and Q) dim for GDN
        gdn_v_dim = c.gdn_v_num_heads * c.gdn_v_d_head  # V dim for GDN (may differ from K)
        attn_q_dim = c.num_q_heads * c.d_head

        for layer_idx in range(c.n_layers):
            prefix = f"model.layers.{layer_idx}"
            layer_type = c.layer_types[layer_idx] if c.layer_types else 'attn'

            # RMSNorm weights (same for both layer types)
            try:
                norm1_w = sd[f"{prefix}.input_layernorm.weight"].float().numpy()
                norm2_w = sd[f"{prefix}.post_attention_layernorm.weight"].float().numpy()
            except KeyError as e:
                raise ValueError(
                    f"Missing norm weight in state dict for Qwen3.5 layer {layer_idx} "
                    f"(prefix={prefix}): {e}"
                ) from e
            norm1_q, norm1_s = quantize_symmetric(norm1_w)
            norm2_q, norm2_s = quantize_symmetric(norm2_w)
            weights[f"layer{layer_idx}.norm1"] = QuantizedWeights(norm1_q, norm1_s, norm1_w.shape, f"layer{layer_idx}.norm1")
            weights[f"layer{layer_idx}.norm2"] = QuantizedWeights(norm2_q, norm2_s, norm2_w.shape, f"layer{layer_idx}.norm2")

            if layer_type == 'gdn':
                # --- GatedDeltaNet layer ---
                try:
                    # Fused QKV → split into separate Q, K, V (Q and K share gdn_k_dim, V uses gdn_v_dim)
                    qkv_w = sd[f"{prefix}.linear_attn.in_proj_qkv.weight"].float().numpy()
                    expected_rows = 2 * gdn_k_dim + gdn_v_dim
                    if qkv_w.shape[0] != expected_rows:
                        raise ValueError(
                            f"Layer {layer_idx} fused QKV rows {qkv_w.shape[0]} != "
                            f"2*gdn_k_dim + gdn_v_dim ({2*gdn_k_dim}+{gdn_v_dim}={expected_rows}). "
                            f"Config mismatch: linear_num_key_heads={c.gdn_num_heads}, "
                            f"linear_num_value_heads={c.gdn_v_num_heads}"
                        )
                    q_w = qkv_w[:gdn_k_dim, :]
                    k_w = qkv_w[gdn_k_dim:2*gdn_k_dim, :]
                    v_w = qkv_w[2*gdn_k_dim:2*gdn_k_dim + gdn_v_dim, :]

                    # Output gate (in_proj_z) → stored as g_proj
                    z_w = sd[f"{prefix}.linear_attn.in_proj_z.weight"].float().numpy()

                    # Output projection
                    o_w = sd[f"{prefix}.linear_attn.out_proj.weight"].float().numpy()

                    # Conv1D weights (for folding at seq_len=1)
                    conv_w = sd[f"{prefix}.linear_attn.conv1d.weight"].float().numpy()
                except KeyError as e:
                    raise ValueError(
                        f"Missing GatedDeltaNet weight in state dict for layer {layer_idx} "
                        f"(prefix={prefix}): {e}. Expected linear_attn.{{in_proj_qkv, "
                        f"in_proj_z, out_proj, conv1d}}.weight"
                    ) from e

                for name, w in [('q_proj', q_w), ('k_proj', k_w), ('v_proj', v_w)]:
                    w_q, scale = quantize_symmetric(w)
                    weights[f"layer{layer_idx}.{name}"] = QuantizedWeights(
                        w_q, scale, w.shape, f"layer{layer_idx}.{name}")

                z_q, z_s = quantize_symmetric(z_w)
                weights[f"layer{layer_idx}.g_proj"] = QuantizedWeights(
                    z_q, z_s, z_w.shape, f"layer{layer_idx}.g_proj")

                o_q, o_s = quantize_symmetric(o_w)
                weights[f"layer{layer_idx}.o_proj"] = QuantizedWeights(
                    o_q, o_s, o_w.shape, f"layer{layer_idx}.o_proj")

                weights[f"layer{layer_idx}.short_conv"] = QuantizedWeights(
                    conv_w.flatten().astype(np.float32).view(np.uint32),
                    1.0, conv_w.shape, f"layer{layer_idx}.short_conv")

            else:
                # --- Full attention layer ---
                try:
                    # q_proj is packed: [Q (first half) | gate (second half)]
                    q_full = sd[f"{prefix}.self_attn.q_proj.weight"].float().numpy()

                    # K, V, O projections
                    kvo_weights = {}
                    for proj_name in ['k_proj', 'v_proj', 'o_proj']:
                        kvo_weights[proj_name] = sd[f"{prefix}.self_attn.{proj_name}.weight"].float().numpy()
                except KeyError as e:
                    raise ValueError(
                        f"Missing attention weight in state dict for Qwen3.5 layer {layer_idx} "
                        f"(prefix={prefix}): {e}. Expected self_attn.{{q,k,v,o}}_proj.weight"
                    ) from e

                q_w = q_full[:attn_q_dim, :]
                gate_w = q_full[attn_q_dim:, :]

                q_q, q_s = quantize_symmetric(q_w)
                weights[f"layer{layer_idx}.q_proj"] = QuantizedWeights(
                    q_q, q_s, q_w.shape, f"layer{layer_idx}.q_proj")

                gate_q, gate_s = quantize_symmetric(gate_w)
                weights[f"layer{layer_idx}.g_proj"] = QuantizedWeights(
                    gate_q, gate_s, gate_w.shape, f"layer{layer_idx}.g_proj")

                for proj_name in ['k_proj', 'v_proj', 'o_proj']:
                    w = kvo_weights[proj_name]
                    w_q, scale = quantize_symmetric(w)
                    weights[f"layer{layer_idx}.{proj_name}"] = QuantizedWeights(
                        w_q, scale, w.shape, f"layer{layer_idx}.{proj_name}")

            # MLP: gate_proj, up_proj, down_proj (same for both layer types)
            try:
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    w = sd[f"{prefix}.mlp.{proj_name}.weight"].float().numpy()
                    w_q, scale = quantize_symmetric(w)
                    weights[f"layer{layer_idx}.{proj_name}"] = QuantizedWeights(
                        w_q, scale, w.shape, f"layer{layer_idx}.{proj_name}")
            except KeyError as e:
                raise ValueError(
                    f"Missing MLP weight in state dict for Qwen3.5 layer {layer_idx}: {e}"
                ) from e

        # Final RMSNorm
        if "model.norm.weight" in sd:
            final_norm = sd["model.norm.weight"].float().numpy()
        elif "model.text_model.norm.weight" in sd:
            final_norm = sd["model.text_model.norm.weight"].float().numpy()
        else:
            raise ValueError(
                "Missing final norm weight: neither 'model.norm.weight' nor "
                "'model.text_model.norm.weight' found in state dict"
            )
        fq, fs = quantize_symmetric(final_norm)
        weights["final_norm"] = QuantizedWeights(fq, fs, final_norm.shape, "final_norm")

        # LM head
        try:
            lm_head = sd["lm_head.weight"].float().numpy()
        except KeyError as e:
            raise ValueError(f"Missing LM head weight 'lm_head.weight' in state dict: {e}") from e
        lq, ls = quantize_symmetric(lm_head)
        weights["lm_head"] = QuantizedWeights(lq, ls, lm_head.shape, "lm_head")

        return weights

    def _extract_gpt2(self) -> Dict[str, QuantizedWeights]:
        """Extract GPT-2 weights (backward compatible)."""
        weights = {}
        sd = self.model.state_dict()
        c = self.config

        for layer_idx in range(c.n_layers):
            prefix = f"transformer.h.{layer_idx}"

            # LayerNorm (gamma + beta)
            try:
                for ln_name in ['ln_1', 'ln_2']:
                    gamma = sd[f"{prefix}.{ln_name}.weight"].float().numpy()
                    beta = sd[f"{prefix}.{ln_name}.bias"].float().numpy()
                    gq, gs = quantize_symmetric(gamma)
                    bq, bs = quantize_symmetric(beta)
                    weights[f"layer{layer_idx}.{ln_name}_gamma"] = QuantizedWeights(gq, gs, gamma.shape, f"layer{layer_idx}.{ln_name}_gamma")
                    weights[f"layer{layer_idx}.{ln_name}_beta"] = QuantizedWeights(bq, bs, beta.shape, f"layer{layer_idx}.{ln_name}_beta")
            except KeyError as e:
                raise ValueError(
                    f"Missing LayerNorm weight in state dict for GPT-2 layer {layer_idx} "
                    f"(prefix={prefix}): {e}"
                ) from e

            # Attention: c_attn (combined QKV), c_proj
            try:
                for attn_name in ['c_attn', 'c_proj']:
                    key = f"{prefix}.attn.{attn_name}.weight"
                    w = sd[key].float().numpy()
                    # GPT-2 uses transposed weights (d_model, out_features)
                    w = w.T  # → (out_features, d_model) for matmul y = W @ x
                    wq, ws = quantize_symmetric(w)
                    weights[f"layer{layer_idx}.{attn_name}"] = QuantizedWeights(wq, ws, w.shape, f"layer{layer_idx}.{attn_name}")
                    # Bias
                    bkey = f"{prefix}.attn.{attn_name}.bias"
                    if bkey in sd:
                        b = sd[bkey].float().numpy()
                        bq, bs = quantize_symmetric(b)
                        weights[f"layer{layer_idx}.{attn_name}_bias"] = QuantizedWeights(bq, bs, b.shape, f"layer{layer_idx}.{attn_name}_bias")
            except KeyError as e:
                raise ValueError(
                    f"Missing attention weight in state dict for GPT-2 layer {layer_idx}: {e}"
                ) from e

            # MLP: c_fc, c_proj
            try:
                for mlp_name in ['c_fc', 'c_proj']:
                    key = f"{prefix}.mlp.{mlp_name}.weight"
                    w = sd[key].float().numpy().T
                    wq, ws = quantize_symmetric(w)
                    weights[f"layer{layer_idx}.mlp_{mlp_name}"] = QuantizedWeights(wq, ws, w.shape, f"layer{layer_idx}.mlp_{mlp_name}")
                    bkey = f"{prefix}.mlp.{mlp_name}.bias"
                    if bkey in sd:
                        b = sd[bkey].float().numpy()
                        bq, bs = quantize_symmetric(b)
                        weights[f"layer{layer_idx}.mlp_{mlp_name}_bias"] = QuantizedWeights(bq, bs, b.shape, f"layer{layer_idx}.mlp_{mlp_name}_bias")
            except KeyError as e:
                raise ValueError(
                    f"Missing MLP weight in state dict for GPT-2 layer {layer_idx}: {e}"
                ) from e

        # Final LayerNorm
        try:
            for name in ['weight', 'bias']:
                key = f"transformer.ln_f.{name}"
                w = sd[key].float().numpy()
                wq, ws = quantize_symmetric(w)
                weights[f"final_ln_{name}"] = QuantizedWeights(wq, ws, w.shape, f"final_ln_{name}")
        except KeyError as e:
            raise ValueError(f"Missing final LayerNorm weight 'transformer.ln_f.{name}' in state dict: {e}") from e

        # LM head
        try:
            lm_head = sd["lm_head.weight"].float().numpy()
        except KeyError as e:
            raise ValueError(f"Missing LM head weight 'lm_head.weight' in state dict: {e}") from e
        lq, ls = quantize_symmetric(lm_head)
        weights["lm_head"] = QuantizedWeights(lq, ls, lm_head.shape, "lm_head")

        return weights

    def get_hidden_states(self, text: str, layer_idx: int = 0):
        """Get hidden states at a specific layer for a given input text.

        Falls back to a synthetic input vector when the loaded model is the
        `_StaticStateDictModel` stand-in (i.e. when AutoModel was bypassed
        via `_load_via_safetensors_direct`). Synthetic input has the same
        shape as a real hidden state and uses values small enough to pass
        the prover's RMSNorm QR-perturbation loop without much rejection.
        For benchmarking, prove cost depends only on dimensions, so the
        synthetic input gives identical timing — only correctness against
        a specific token stream is lost. Reported in benchmark output."""
        import torch

        if isinstance(self.model, _StaticStateDictModel):
            # Direct-safetensors path: no callable model.
            import numpy as np
            import sys
            print(
                "      [info] AutoModel bypass active — using synthetic "
                "hidden-state input (benchmark prove cost is unaffected; "
                "semantic correctness against the token stream is not).",
                file=sys.stderr,
            )
            d_model = self.config.d_model
            rng = np.random.default_rng(seed=42)
            # Small symmetric-around-zero values so that RMSNorm sum_sq ≠ 0
            # and the QR-perturbation loop converges fast.
            return rng.standard_normal(d_model).astype(np.float32) * 0.1

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        # hidden_states[0] = embeddings, hidden_states[i] = after layer i-1
        hidden = outputs.hidden_states[layer_idx]
        # Take last token position
        return hidden[0, -1, :].float().numpy()
