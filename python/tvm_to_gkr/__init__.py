"""ZK-Compile: Structured sumcheck prover over M31 field.

Supports GPT-2 (original) and Llama/Mistral/Qwen (Phase 1).
Python handles model loading, quantization, and binary protocol encoding.
Rust does the actual proving.
"""

from .m31 import M31
from .structured_pipeline import (
    PrecompiledTransformerWeights,
    RustProverServer,
    prove_rust_transformer_v2,
)
from .model_extractor import ModelExtractor, ModelConfig, detect_model_config
from .llama_pipeline import prove_llama
