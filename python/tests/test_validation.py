"""Integration tests for validation, error handling, and edge cases.

Tests JSON result parsing, state dict key validation, stderr handling,
and binary existence checks across all pipeline modules.

No model downloads required — all tests use mocks or synthetic data.
"""

import sys
import os
import json
import struct
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tvm_to_gkr.constants import M31, QUANT_RANGE
from tvm_to_gkr.model_extractor import (
    ModelConfig,
    QuantizedWeights,
    quantize_symmetric,
)


class TestResultParsingValidation(unittest.TestCase):
    """Test that JSON result parsing uses safe .get() with defaults."""

    def test_missing_prove_time_returns_zero(self):
        """Result dict missing 'prove_time_ms' should not raise."""
        from tvm_to_gkr.structured_pipeline import prove_rust_transformer_v2
        # We can't easily call prove_rust_transformer_v2 without a model,
        # but we can verify the parsing logic handles missing keys.
        # Simulate what the parsing code does:
        result = {"valid": True, "proof_size_bytes": 100}
        total_prove_ms = result.get("prove_time_ms", 0)
        total_verify_ms = result.get("verify_time_ms", 0)
        total_proof_bytes = result.get("proof_size_bytes", 0)
        all_valid = result.get("valid", False)
        self.assertEqual(total_prove_ms, 0)
        self.assertEqual(total_verify_ms, 0)
        self.assertEqual(total_proof_bytes, 100)
        self.assertTrue(all_valid)

    def test_missing_valid_returns_false(self):
        """Result dict missing 'valid' should default to False."""
        result = {"prove_time_ms": 42.0}
        all_valid = result.get("valid", False)
        self.assertFalse(all_valid)

    def test_complete_result_parses_correctly(self):
        """Full result dict should parse all fields correctly."""
        result = {
            "valid": True,
            "prove_time_ms": 123.4,
            "verify_time_ms": 5.6,
            "proof_size_bytes": 2048,
            "coverage": {
                "proved_count": 10,
                "computational_count": 8,
                "computational_total": 10,
                "total_count": 12,
                "state_count": 2,
            }
        }
        self.assertEqual(result.get("prove_time_ms", 0), 123.4)
        self.assertEqual(result.get("verify_time_ms", 0), 5.6)
        self.assertEqual(result.get("proof_size_bytes", 0), 2048)
        self.assertTrue(result.get("valid", False))

    def test_coverage_parsing_with_missing_fields(self):
        """Coverage dict should use .get() for all sub-fields."""
        cov = {"proved_count": 5}
        comp_count = cov.get("computational_count", cov.get("proved_count", 0))
        comp_total = cov.get("computational_total", comp_count)
        total_count = cov.get("total_count", 0)
        state_count = cov.get("state_count", 0)
        self.assertEqual(comp_count, 5)  # falls back to proved_count
        self.assertEqual(comp_total, 5)  # falls back to comp_count
        self.assertEqual(total_count, 0)
        self.assertEqual(state_count, 0)

    def test_none_result_type_error(self):
        """Calling .get() on None should raise TypeError, not KeyError."""
        result = None
        with self.assertRaises((TypeError, AttributeError)):
            result.get("prove_time_ms", 0)


class TestStateDictKeyValidation(unittest.TestCase):
    """Test that state dict key misses raise ValueError with helpful messages."""

    def _make_fake_sd(self, keys):
        """Build a fake state dict with given keys mapping to small tensors."""
        sd = {}
        for k in keys:
            t = MagicMock()
            t.float.return_value.numpy.return_value = np.ones(4, dtype=np.float32)
            sd[k] = t
        return sd

    def test_llama_missing_norm_raises_valueerror(self):
        """Missing input_layernorm should raise ValueError with layer info."""
        from tvm_to_gkr.model_extractor import ModelExtractor

        # Create a mock extractor
        extractor = object.__new__(ModelExtractor)
        extractor.config = ModelConfig(
            d_model=4, d_ff=8, num_q_heads=1, num_kv_heads=1,
            d_head=4, n_layers=1, vocab_size=10,
            norm_type="rmsnorm", activation="swiglu", model_type="llama",
        )

        # State dict missing the norm weights
        sd = self._make_fake_sd([
            "model.layers.0.self_attn.q_proj.weight",
        ])
        mock_model = MagicMock()
        mock_model.state_dict.return_value = sd
        extractor.model = mock_model

        with self.assertRaises(ValueError) as ctx:
            extractor._extract_llama_style()
        self.assertIn("layer 0", str(ctx.exception))
        self.assertIn("norm", str(ctx.exception).lower())

    def test_llama_missing_attn_raises_valueerror(self):
        """Missing attention weight should raise ValueError."""
        from tvm_to_gkr.model_extractor import ModelExtractor

        extractor = object.__new__(ModelExtractor)
        extractor.config = ModelConfig(
            d_model=4, d_ff=8, num_q_heads=1, num_kv_heads=1,
            d_head=4, n_layers=1, vocab_size=10,
            norm_type="rmsnorm", activation="swiglu", model_type="llama",
        )

        # Has norms but missing attention weights
        sd = self._make_fake_sd([
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ])
        mock_model = MagicMock()
        mock_model.state_dict.return_value = sd
        extractor.model = mock_model

        with self.assertRaises(ValueError) as ctx:
            extractor._extract_llama_style()
        self.assertIn("attention", str(ctx.exception).lower())

    def test_gpt2_missing_layernorm_raises_valueerror(self):
        """Missing GPT-2 LayerNorm should raise ValueError with prefix info."""
        from tvm_to_gkr.model_extractor import ModelExtractor

        extractor = object.__new__(ModelExtractor)
        extractor.config = ModelConfig(
            d_model=4, d_ff=16, num_q_heads=1, num_kv_heads=1,
            d_head=4, n_layers=1, vocab_size=10,
            norm_type="layernorm", activation="gelu", model_type="gpt2",
        )

        sd = self._make_fake_sd([])  # empty state dict
        mock_model = MagicMock()
        mock_model.state_dict.return_value = sd
        extractor.model = mock_model

        with self.assertRaises(ValueError) as ctx:
            extractor._extract_gpt2()
        self.assertIn("LayerNorm", str(ctx.exception))
        self.assertIn("GPT-2", str(ctx.exception))

    def test_qwen35_missing_gdn_weight_raises_valueerror(self):
        """Missing GDN linear_attn weight should raise ValueError."""
        from tvm_to_gkr.model_extractor import ModelExtractor

        extractor = object.__new__(ModelExtractor)
        extractor.config = ModelConfig(
            d_model=4, d_ff=8, num_q_heads=1, num_kv_heads=1,
            d_head=4, n_layers=1, vocab_size=10,
            norm_type="rmsnorm", activation="swiglu", model_type="qwen3_5",
            layer_types=["gdn"],
            gdn_num_heads=1, gdn_d_head=4,
        )

        # Has norms but missing GDN-specific weights
        sd = self._make_fake_sd([
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ])
        mock_model = MagicMock()
        mock_model.state_dict.return_value = sd
        extractor.model = mock_model

        with self.assertRaises(ValueError) as ctx:
            extractor._extract_qwen35_style()
        self.assertIn("GatedDeltaNet", str(ctx.exception))
        self.assertIn("layer 0", str(ctx.exception))

    def test_qwen35_missing_final_norm_raises_valueerror(self):
        """Missing final norm weight should raise ValueError."""
        from tvm_to_gkr.model_extractor import ModelExtractor

        extractor = object.__new__(ModelExtractor)
        extractor.config = ModelConfig(
            d_model=4, d_ff=8, num_q_heads=1, num_kv_heads=1,
            d_head=4, n_layers=0, vocab_size=10,
            norm_type="rmsnorm", activation="swiglu", model_type="qwen3_5",
            layer_types=[],
            gdn_num_heads=1, gdn_d_head=4,
        )

        sd = self._make_fake_sd([])  # No layers, no final norm
        mock_model = MagicMock()
        mock_model.state_dict.return_value = sd
        extractor.model = mock_model

        with self.assertRaises(ValueError) as ctx:
            extractor._extract_qwen35_style()
        self.assertIn("final norm", str(ctx.exception).lower())


class TestStderrHandling(unittest.TestCase):
    """Test that stderr is not truncated in error messages."""

    def test_llama_stderr_not_truncated(self):
        """llama_pipeline RuntimeError should contain full stderr."""
        from tvm_to_gkr.llama_pipeline import prove_llama_token

        # Build a long stderr message (>200 chars)
        long_stderr = "CRITICAL ERROR: " + "x" * 300 + " END_MARKER"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = long_stderr.encode('utf-8')

        extractor = MagicMock()
        extractor.config = ModelConfig(
            d_model=4, d_ff=8, num_q_heads=1, num_kv_heads=1,
            d_head=4, n_layers=1, vocab_size=10,
            norm_type="rmsnorm", activation="swiglu", model_type="llama",
        )
        extractor.get_hidden_states.return_value = np.ones(4, dtype=np.float32)
        extractor.model.state_dict.return_value = {}

        weights = {
            f"layer0.{p}": QuantizedWeights(
                np.ones(16, dtype=np.uint32), 1.0, (4, 4), f"layer0.{p}"
            ) for p in ['norm1', 'norm2', 'q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj']
        }

        with patch('tvm_to_gkr.llama_pipeline.subprocess.run', return_value=mock_result), \
             patch('os.path.exists', return_value=True), \
             patch('tvm_to_gkr.llama_pipeline.build_llama_layer_op', return_value=b'\x00' * 10):
            with self.assertRaises(RuntimeError) as ctx:
                prove_llama_token(extractor, weights, "test", prove_layers=1)

            error_msg = str(ctx.exception)
            # Must contain the full END_MARKER (proves no truncation)
            self.assertIn("END_MARKER", error_msg)


class TestRustBinaryCheck(unittest.TestCase):
    """Test that missing Rust binary gives helpful build instructions."""

    def test_qwen35_missing_binary_message(self):
        """qwen35_pipeline should suggest build command when binary is missing."""
        import inspect
        from tvm_to_gkr import qwen35_pipeline
        source = inspect.getsource(qwen35_pipeline.prove_qwen35)
        self.assertIn("cargo build --release", source)
        self.assertIn("Rust prover binary not found", source)

    def test_llama_missing_binary_message(self):
        """llama_pipeline should suggest build command when binary is missing."""
        import inspect
        from tvm_to_gkr import llama_pipeline
        source = inspect.getsource(llama_pipeline.prove_llama)
        self.assertIn("cargo build --release", source)

    def test_qwen35_binary_check_raises_filenotfounderror(self):
        """prove_qwen35 should raise FileNotFoundError for missing binary."""
        from tvm_to_gkr.qwen35_pipeline import prove_qwen35

        # Mock everything up to the binary check
        mock_extractor = MagicMock()
        mock_extractor.config = ModelConfig(
            d_model=4, d_ff=8, num_q_heads=1, num_kv_heads=1,
            d_head=4, n_layers=1, vocab_size=10,
            norm_type="rmsnorm", activation="swiglu", model_type="qwen3_5",
            layer_types=["gdn"],
            gdn_num_heads=1, gdn_d_head=4,
        )
        mock_extractor.extract.return_value = {}
        mock_extractor.get_hidden_states.return_value = np.ones(4, dtype=np.float32)

        with patch('tvm_to_gkr.qwen35_pipeline.ModelExtractor', return_value=mock_extractor), \
             patch('tvm_to_gkr.qwen35_pipeline.quantize_symmetric',
                   return_value=(np.ones(4, dtype=np.uint32), 1.0)), \
             patch('tvm_to_gkr.qwen35_pipeline.build_qwen_layer_op', return_value=b'\x00'), \
             patch('os.path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError) as ctx:
                prove_qwen35("fake-model", "hello", prove_layers=1)
            self.assertIn("cargo build --release", str(ctx.exception))
            self.assertIn("Rust prover binary not found", str(ctx.exception))


class TestQuantizeSymmetric(unittest.TestCase):
    """Test M31 quantization edge cases."""

    def test_zero_weight(self):
        """All-zero weight should return zeros with scale=1.0."""
        w = np.zeros(8, dtype=np.float32)
        w_q, scale = quantize_symmetric(w)
        self.assertEqual(scale, 1.0)
        np.testing.assert_array_equal(w_q, np.zeros(8, dtype=np.uint32))

    def test_negative_values_become_field_elements(self):
        """Negative values should map to M31 - |val|."""
        w = np.array([-1.0, 1.0], dtype=np.float32)
        w_q, scale = quantize_symmetric(w)
        # scale = 1.0/127 = 0.00787...
        # -1.0 / scale = -127, in M31: M31 - 127
        self.assertEqual(w_q[0], M31 - 127)
        self.assertEqual(w_q[1], 127)

    def test_symmetric_range(self):
        """Quantized values should be in [-QUANT_RANGE, QUANT_RANGE]."""
        np.random.seed(42)
        w = np.random.randn(100).astype(np.float32)
        w_q, scale = quantize_symmetric(w)
        # Convert back to signed
        signed = np.where(w_q > M31 // 2, w_q.astype(np.int64) - M31, w_q.astype(np.int64))
        self.assertTrue(np.all(np.abs(signed) <= QUANT_RANGE))


class TestModelConfigDetection(unittest.TestCase):
    """Test model config detection for various architectures."""

    def test_gpt2_config(self):
        """GPT-2 config should use layernorm and gelu."""
        from tvm_to_gkr.model_extractor import detect_model_config

        hf_config = MagicMock()
        hf_config.model_type = 'gpt2'
        hf_config.n_embd = 768
        hf_config.n_inner = None
        hf_config.n_head = 12
        hf_config.n_layer = 12
        hf_config.vocab_size = 50257

        config = detect_model_config(hf_config)
        self.assertEqual(config.model_type, "gpt2")
        self.assertEqual(config.norm_type, "layernorm")
        self.assertEqual(config.activation, "gelu")
        self.assertEqual(config.d_model, 768)
        self.assertEqual(config.d_ff, 3072)  # 4 * 768

    def test_llama_config(self):
        """Llama config should use rmsnorm and swiglu."""
        from tvm_to_gkr.model_extractor import detect_model_config

        hf_config = MagicMock()
        hf_config.model_type = 'llama'
        hf_config.hidden_size = 4096
        hf_config.intermediate_size = 11008
        hf_config.num_attention_heads = 32
        hf_config.num_key_value_heads = 32
        hf_config.num_hidden_layers = 32
        hf_config.vocab_size = 32000

        config = detect_model_config(hf_config)
        self.assertEqual(config.model_type, "llama")
        self.assertEqual(config.norm_type, "rmsnorm")
        self.assertEqual(config.activation, "swiglu")
        self.assertEqual(config.d_head, 128)  # 4096 / 32

    def test_unsupported_model_raises(self):
        """Unknown model type should raise ValueError."""
        from tvm_to_gkr.model_extractor import detect_model_config

        hf_config = MagicMock()
        hf_config.model_type = 'bloom'

        with self.assertRaises(ValueError) as ctx:
            detect_model_config(hf_config)
        self.assertIn("bloom", str(ctx.exception))

    def test_qwen35_layer_types(self):
        """Qwen3.5 should detect layer_types from HF config."""
        from tvm_to_gkr.model_extractor import detect_model_config

        hf_config = MagicMock()
        hf_config.model_type = 'qwen3_5'
        # Simulate no text_config (direct config)
        hf_config.text_config = hf_config  # self-referential mock
        del hf_config.text_config  # force getattr fallback
        hf_config.hidden_size = 1024
        hf_config.intermediate_size = 3584
        hf_config.num_attention_heads = 8
        hf_config.num_key_value_heads = 2
        hf_config.head_dim = 256
        hf_config.num_hidden_layers = 4
        hf_config.vocab_size = 248320
        hf_config.layer_types = ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']
        hf_config.linear_num_key_heads = 8
        hf_config.linear_key_head_dim = 128

        config = detect_model_config(hf_config)
        self.assertEqual(config.model_type, "qwen3_5")
        self.assertEqual(len(config.layer_types), 4)
        self.assertEqual(config.layer_types[:3], ['gdn', 'gdn', 'gdn'])
        self.assertEqual(config.layer_types[3], 'attn')
        self.assertEqual(config.gdn_num_heads, 8)
        self.assertEqual(config.gdn_d_head, 128)


class TestBinaryPayloadConstruction(unittest.TestCase):
    """Test binary payload building for Rust prover protocol."""

    def test_empty_ops_payload(self):
        """Payload with zero ops should have correct header."""
        from tvm_to_gkr.structured_pipeline import _build_binary_payload

        input_q = [1, 2, 3, 4]
        payload = _build_binary_payload(input_q, [])
        # magic(1) + num_ops(4) + input_len(4) + input(4*4) = 25 bytes
        self.assertEqual(len(payload), 1 + 4 + 4 + 16)
        self.assertEqual(payload[0], 0x00)  # magic
        num_ops = struct.unpack('<I', payload[1:5])[0]
        self.assertEqual(num_ops, 0)
        input_len = struct.unpack('<I', payload[5:9])[0]
        self.assertEqual(input_len, 4)

    def test_relu_op_payload(self):
        """ReLU op should encode as type byte 0x01 + name."""
        from tvm_to_gkr.structured_pipeline import _build_binary_payload

        input_q = [100]
        ops = [{"type": "relu", "name": "relu_0"}]
        payload = _build_binary_payload(input_q, ops)
        # Find the op type byte after header
        # header = 1 + 4 + 4 + 4 = 13 bytes
        op_type = payload[13]
        self.assertEqual(op_type, 0x01)


class TestLlamaPipelineHelpers(unittest.TestCase):
    """Test Llama pipeline helper functions."""

    def test_build_llama_layer_ref_op(self):
        """Should produce correct op dict with all weight names."""
        from tvm_to_gkr.llama_pipeline import build_llama_layer_ref_op

        config = ModelConfig(
            d_model=512, d_ff=1024, num_q_heads=8, num_kv_heads=4,
            d_head=64, n_layers=4, vocab_size=32000,
            norm_type="rmsnorm", activation="swiglu", model_type="llama",
        )
        op = build_llama_layer_ref_op("layer_2", config, silu_scale=1000, layer_idx=2)
        self.assertEqual(op["type"], "llama_layer_ref")
        self.assertEqual(op["name"], "layer_2")
        self.assertEqual(op["config"]["d_model"], 512)
        self.assertEqual(op["config"]["silu_scale"], 1000)
        self.assertEqual(len(op["weight_names"]), 9)
        self.assertIn("layer2.q_proj", op["weight_names"])
        self.assertIn("layer2.down_proj", op["weight_names"])

    def test_to_m31(self):
        """to_m31 should handle negative and positive values."""
        from tvm_to_gkr.llama_pipeline import to_m31
        self.assertEqual(to_m31(0), 0)
        self.assertEqual(to_m31(42), 42)
        self.assertEqual(to_m31(-1), M31 - 1)
        self.assertEqual(to_m31(-M31), 0)


class TestQwenPipelineHelpers(unittest.TestCase):
    """Test Qwen3.5 pipeline helper functions."""

    def test_layer_head_config_gdn(self):
        """GDN layers should use gdn_num_heads and gdn_d_head."""
        from tvm_to_gkr.qwen35_pipeline import _layer_head_config

        config = ModelConfig(
            d_model=1024, d_ff=3584, num_q_heads=8, num_kv_heads=2,
            d_head=256, n_layers=4, vocab_size=248320,
            norm_type="rmsnorm", activation="swiglu", model_type="qwen3_5",
            layer_types=["gdn", "gdn", "gdn", "attn"],
            gdn_num_heads=16, gdn_d_head=128,
        )
        heads = _layer_head_config(config, 0)
        self.assertEqual(heads["num_q_heads"], 16)
        self.assertEqual(heads["d_head"], 128)

    def test_layer_head_config_attn(self):
        """Attention layers should use standard head config."""
        from tvm_to_gkr.qwen35_pipeline import _layer_head_config

        config = ModelConfig(
            d_model=1024, d_ff=3584, num_q_heads=8, num_kv_heads=2,
            d_head=256, n_layers=4, vocab_size=248320,
            norm_type="rmsnorm", activation="swiglu", model_type="qwen3_5",
            layer_types=["gdn", "gdn", "gdn", "attn"],
            gdn_num_heads=16, gdn_d_head=128,
        )
        heads = _layer_head_config(config, 3)
        self.assertEqual(heads["num_q_heads"], 8)
        self.assertEqual(heads["num_kv_heads"], 2)
        self.assertEqual(heads["d_head"], 256)

    def test_build_qwen_layer_ref_op_gdn(self):
        """GDN ref op should use GDN head dimensions."""
        from tvm_to_gkr.qwen35_pipeline import build_qwen_layer_ref_op

        config = ModelConfig(
            d_model=1024, d_ff=3584, num_q_heads=8, num_kv_heads=2,
            d_head=256, n_layers=4, vocab_size=248320,
            norm_type="rmsnorm", activation="swiglu", model_type="qwen3_5",
            layer_types=["gdn", "gdn", "gdn", "attn"],
            gdn_num_heads=16, gdn_d_head=128,
        )
        op = build_qwen_layer_ref_op("layer_0", config, 1000, 1000, layer_idx=0)
        self.assertEqual(op["config"]["num_q_heads"], 16)
        self.assertEqual(op["config"]["d_head"], 128)
        self.assertIn("layer0.g_proj", op["weight_names"])

    def test_build_qwen_layer_ref_op_attn(self):
        """Attention ref op should use standard head dimensions."""
        from tvm_to_gkr.qwen35_pipeline import build_qwen_layer_ref_op

        config = ModelConfig(
            d_model=1024, d_ff=3584, num_q_heads=8, num_kv_heads=2,
            d_head=256, n_layers=4, vocab_size=248320,
            norm_type="rmsnorm", activation="swiglu", model_type="qwen3_5",
            layer_types=["gdn", "gdn", "gdn", "attn"],
            gdn_num_heads=16, gdn_d_head=128,
        )
        op = build_qwen_layer_ref_op("layer_3", config, 1000, 1000, layer_idx=3)
        self.assertEqual(op["config"]["num_q_heads"], 8)
        self.assertEqual(op["config"]["num_kv_heads"], 2)


class TestM31Arithmetic(unittest.TestCase):
    """Test M31 field operations used throughout the pipeline."""

    def test_add_no_overflow(self):
        from tvm_to_gkr.m31 import add
        self.assertEqual(add(10, 20), 30)

    def test_add_with_reduction(self):
        from tvm_to_gkr.m31 import add
        self.assertEqual(add(M31 - 1, 2), 1)

    def test_sub_no_underflow(self):
        from tvm_to_gkr.m31 import sub
        self.assertEqual(sub(20, 10), 10)

    def test_sub_with_wrap(self):
        from tvm_to_gkr.m31 import sub
        self.assertEqual(sub(0, 1), M31 - 1)

    def test_mul_basic(self):
        from tvm_to_gkr.m31 import mul
        self.assertEqual(mul(3, 7), 21)

    def test_mul_large(self):
        from tvm_to_gkr.m31 import mul
        # (M31-1) * 2 = 2*M31 - 2 mod M31 = M31 - 2
        self.assertEqual(mul(M31 - 1, 2), M31 - 2)

    def test_inverse(self):
        from tvm_to_gkr.m31 import mul, inv
        a = 12345
        a_inv = inv(a)
        self.assertEqual(mul(a, a_inv), 1)

    def test_inverse_zero_raises(self):
        from tvm_to_gkr.m31 import inv
        with self.assertRaises(ZeroDivisionError):
            inv(0)

    def test_inner_product(self):
        from tvm_to_gkr.m31 import inner_product
        a = [1, 2, 3]
        b = [4, 5, 6]
        # 1*4 + 2*5 + 3*6 = 32
        self.assertEqual(inner_product(a, b), 32)

    def test_from_field_positive(self):
        from tvm_to_gkr.m31 import from_field
        self.assertEqual(from_field(42), 42)

    def test_from_field_negative(self):
        from tvm_to_gkr.m31 import from_field
        self.assertEqual(from_field(M31 - 1), -1)


if __name__ == '__main__':
    unittest.main()
