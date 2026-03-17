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
    # GDN-specific config (may differ from full attention head config)
    gdn_num_heads: int = 0     # GDN head count (e.g. 16)
    gdn_d_head: int = 0        # GDN head dim (e.g. 128)


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
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class ModelExtractor:
    """Extract and quantize weights from any HuggingFace model."""

    def __init__(self, model_name_or_path: str, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig

        self.model_name = model_name_or_path
        hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.config = detect_model_config(hf_config)

        # Load model (use float32 for quantization accuracy)
        load_dtype = dtype or torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=load_dtype,
            trust_remote_code=True,
        )
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
        gdn_dim = c.gdn_num_heads * c.gdn_d_head  # 2048
        attn_q_dim = c.num_q_heads * c.d_head       # 2048

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
                    # Fused QKV → split into separate Q, K, V
                    qkv_w = sd[f"{prefix}.linear_attn.in_proj_qkv.weight"].float().numpy()
                    q_w = qkv_w[:gdn_dim, :]
                    k_w = qkv_w[gdn_dim:2*gdn_dim, :]
                    v_w = qkv_w[2*gdn_dim:, :]

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
        """Get hidden states at a specific layer for a given input text."""
        import torch
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
