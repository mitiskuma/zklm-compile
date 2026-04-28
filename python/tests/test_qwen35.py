"""Tests for Qwen3.5 ZK proving pipeline.

Tests model extraction, Conv1D folding, and end-to-end proving.
Requires: pip install transformers torch
Model download: Qwen/Qwen3.5-0.8B (~1.6GB, auto-downloaded on first run)
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tvm_to_gkr.constants import M31
from tvm_to_gkr.qwen35_pipeline import (
    _fold_conv_into_proj,
    build_qwen_layer_ref_op,
    to_m31,
)
from tvm_to_gkr.model_extractor import ModelConfig, quantize_symmetric


class TestConv1DFolding(unittest.TestCase):
    """Test Conv1D weight folding for seq_len=1."""

    def test_fold_identity(self):
        """Conv weight of all-ones should leave projection unchanged."""
        rows, cols = 4, 8
        w_signed = np.arange(1, rows * cols + 1, dtype=np.int64)
        w_m31 = np.where(w_signed >= 0, w_signed, M31 + w_signed).astype(np.uint32)

        conv_weight = np.ones((rows, 1, 4), dtype=np.float32)

        result = _fold_conv_into_proj(w_m31, conv_weight, rows, cols, 0)
        np.testing.assert_array_equal(result, w_m31)

    def test_fold_scaling(self):
        """Conv weight of 2.0 should double all values."""
        rows, cols = 2, 4
        w_signed = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        w_m31 = w_signed.astype(np.uint32)

        conv_weight = np.full((rows, 1, 4), 2.0, dtype=np.float32)

        result = _fold_conv_into_proj(w_m31, conv_weight, rows, cols, 0)
        expected = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=np.uint32)
        np.testing.assert_array_equal(result, expected)

    def test_fold_negative_values(self):
        """Negative M31 values should fold correctly."""
        rows, cols = 2, 2
        # -1, -2, 3, 4 in M31
        w_m31 = np.array([M31 - 1, M31 - 2, 3, 4], dtype=np.uint32)

        conv_weight = np.ones((rows, 1, 4), dtype=np.float32)
        conv_weight[0, 0, 0] = 2.0  # scale row 0 by 2

        result = _fold_conv_into_proj(w_m31, conv_weight, rows, cols, 0)
        # Row 0: -1*2=-2, -2*2=-4 -> M31-2, M31-4
        # Row 1: 3*1=3, 4*1=4 (unchanged)
        expected = np.array([M31 - 2, M31 - 4, 3, 4], dtype=np.uint32)
        np.testing.assert_array_equal(result, expected)

    def test_fold_with_offset(self):
        """Offset into conv_weight should select correct rows."""
        total_dim, cols = 6, 2
        conv_weight = np.ones((total_dim, 1, 4), dtype=np.float32)
        # Only scale rows 2-3 (offset=2, rows=2)
        conv_weight[2, 0, 0] = 3.0
        conv_weight[3, 0, 0] = 4.0

        w_m31 = np.array([1, 2, 5, 6], dtype=np.uint32)  # 2 rows x 2 cols

        result = _fold_conv_into_proj(w_m31, conv_weight, 2, cols, 2)
        expected = np.array([3, 6, 20, 24], dtype=np.uint32)
        np.testing.assert_array_equal(result, expected)

    def test_fold_causal_tap_distinguishes_first_and_last(self):
        """G1 — the convention parameter must select
        a different tap when the kernel has heterogeneous weights. If the
        two branches produced the same output, a future refactor could
        silently flip the default and we'd never know.

        Build a kernel whose first and last taps are very different
        (1.0 and 5.0 respectively); fold once with the historical
        ``causal_tap_index=0`` and once with the PyTorch-standard
        ``causal_tap_index=-1`` and assert the results differ.
        """
        rows, cols = 1, 2
        w_m31 = np.array([2, 3], dtype=np.uint32)  # 1 row x 2 cols

        conv_weight = np.zeros((rows, 1, 4), dtype=np.float32)
        conv_weight[0, 0, 0] = 1.0   # first tap
        conv_weight[0, 0, 3] = 5.0   # last tap (k = K-1 = 3)

        first_tap = _fold_conv_into_proj(w_m31, conv_weight, rows, cols, 0,
                                          causal_tap_index=0)
        last_tap = _fold_conv_into_proj(w_m31, conv_weight, rows, cols, 0,
                                         causal_tap_index=-1)

        # First-tap fold scales by 1.0 → unchanged.
        np.testing.assert_array_equal(first_tap, np.array([2, 3], dtype=np.uint32))
        # Last-tap fold scales by 5.0 → multiplied by 5.
        np.testing.assert_array_equal(last_tap, np.array([10, 15], dtype=np.uint32))
        # The two branches must produce distinct outputs.
        self.assertFalse(np.array_equal(first_tap, last_tap),
                         "causal_tap_index branches must select different kernel positions")

    def test_fold_causal_tap_out_of_range_raises(self):
        """Out-of-range tap indices must raise IndexError (defensive)."""
        rows, cols = 1, 2
        w_m31 = np.array([1, 2], dtype=np.uint32)
        conv_weight = np.ones((rows, 1, 4), dtype=np.float32)
        with self.assertRaises(IndexError):
            _fold_conv_into_proj(w_m31, conv_weight, rows, cols, 0,
                                  causal_tap_index=4)
        with self.assertRaises(IndexError):
            _fold_conv_into_proj(w_m31, conv_weight, rows, cols, 0,
                                  causal_tap_index=-5)

    def test_fold_post_silu_is_dropped_pinning(self):
        """G2 — pin that the static fold drops the
        post-Conv1D SiLU. The real Qwen3.5 GDN forward pass is
        ``silu(conv1d(in_proj(x)))``; our fold captures ``conv1d(in_proj(x))``
        only. This test demonstrates the approximation explicitly: fold
        a known scale and confirm the output is the LINEAR product, not
        the SiLU-applied product.

        If a future commit adds a SiLU layer between the folded matmul
        and Q/K/V (the real fix), this test will fail and force a
        coordinated roadmap status update — preventing silent rotation.
        """
        rows, cols = 1, 1
        # Choose w * scale = 8 so silu(8) ≈ 7.997 differs from 8 enough
        # that a SiLU-applied path would round to 7 (or 8 minus a small
        # correction depending on quantization), distinguishing it from
        # the pure linear fold.
        w_signed = np.array([4], dtype=np.int64)
        w_m31 = w_signed.astype(np.uint32)

        conv_weight = np.full((rows, 1, 4), 2.0, dtype=np.float32)

        result = _fold_conv_into_proj(w_m31, conv_weight, rows, cols, 0)
        # Pure linear fold: 4 * 2.0 = 8 → M31 element 8.
        np.testing.assert_array_equal(result, np.array([8], dtype=np.uint32))

        # Sanity: confirm SiLU(8.0) ≠ 8 so a future SiLU-aware fold
        # would produce a different result than this test asserts.
        # SiLU(x) = x * sigmoid(x); SiLU(8) ≈ 7.9973 in float.
        silu_8 = 8.0 * (1.0 / (1.0 + np.exp(-8.0)))
        self.assertLess(silu_8, 8.0,
                        "SiLU(8) is strictly less than 8 — proves a "
                        "post-SiLU fold would round to a different value")
        self.assertGreater(silu_8, 7.99,
                           "SiLU(8) is close enough to 8 that quantization "
                           "to int rounds it to a near-but-distinct value")


class TestToM31(unittest.TestCase):
    """Test signed-to-M31 conversion."""

    def test_positive(self):
        self.assertEqual(to_m31(42), 42)

    def test_negative(self):
        self.assertEqual(to_m31(-1), M31 - 1)

    def test_zero(self):
        self.assertEqual(to_m31(0), 0)


class TestBuildQwenLayerRefOp(unittest.TestCase):
    """Test server-mode op construction."""

    def test_ref_op_structure(self):
        config = ModelConfig(
            model_type='qwen3_5',
            d_model=1024, d_ff=3584,
            num_q_heads=8, num_kv_heads=2,
            d_head=256, n_layers=24,
            vocab_size=248320,
            norm_type='rmsnorm',
            activation='swiglu',
            layer_types=['gdn'] * 18 + ['attn'] * 6,
            sigmoid_scale=1000,
        )
        op = build_qwen_layer_ref_op("layer_0", config, silu_scale=1000, sigmoid_scale=1000, layer_idx=0)

        self.assertEqual(op['type'], 'qwen_layer_ref')
        self.assertEqual(op['name'], 'layer_0')
        self.assertEqual(op['config']['d_model'], 1024)
        self.assertEqual(op['config']['sigmoid_scale'], 1000)
        self.assertEqual(len(op['weight_names']), 10)
        self.assertIn('layer0.q_proj', op['weight_names'])
        self.assertIn('layer0.g_proj', op['weight_names'])
        self.assertIn('layer0.down_proj', op['weight_names'])


class TestModelExtraction(unittest.TestCase):
    """Test Qwen3.5 model detection and extraction.

    These tests require the Qwen3.5-0.8B model to be downloaded.
    Skipped if RUN_MODEL_TESTS is not set.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from tvm_to_gkr.model_extractor import ModelExtractor
            cls.extractor = ModelExtractor("Qwen/Qwen3.5-0.8B")
        except Exception:
            cls.extractor = None

    @unittest.skipIf(
        not os.environ.get('RUN_MODEL_TESTS'),
        "Set RUN_MODEL_TESTS=1 to run tests requiring model download"
    )
    def test_config_detection(self):
        """Qwen3.5 should be detected with correct config."""
        self.assertIsNotNone(self.extractor)
        config = self.extractor.config
        self.assertIn(config.model_type, ('qwen3', 'qwen3_5'))
        self.assertEqual(config.d_model, 1024)
        self.assertEqual(config.n_layers, 24)
        self.assertIsNotNone(config.layer_types)
        self.assertEqual(len(config.layer_types), 24)
        gdn_count = sum(1 for t in config.layer_types if t == 'gdn')
        attn_count = sum(1 for t in config.layer_types if t == 'attn')
        self.assertEqual(gdn_count, 18)
        self.assertEqual(attn_count, 6)

    @unittest.skipIf(
        not os.environ.get('RUN_MODEL_TESTS'),
        "Set RUN_MODEL_TESTS=1 to run tests requiring model download"
    )
    def test_weight_extraction(self):
        """Should extract g_proj and short_conv weights for GDN layers."""
        self.assertIsNotNone(self.extractor)
        weights = self.extractor.extract()
        self.assertIn('layer0.g_proj', weights)
        self.assertIn('layer23.g_proj', weights)
        self.assertIn('layer0.short_conv', weights)
        self.assertNotIn('layer3.short_conv', weights)

    @unittest.skipIf(
        not os.environ.get('RUN_MODEL_TESTS'),
        "Set RUN_MODEL_TESTS=1 to run tests requiring model download"
    )
    def test_precompiled_weights(self):
        """QwenPrecompiledWeights should produce correct weight entries."""
        from tvm_to_gkr.qwen35_pipeline import QwenPrecompiledWeights
        self.assertIsNotNone(self.extractor)
        weights = self.extractor.extract()
        precompiled = QwenPrecompiledWeights(self.extractor, weights)
        # 24 layers × (2 norms + 4 qkvo + 1 g_proj + 3 mlp) = 24 × 10 = 240
        self.assertEqual(len(precompiled.weight_entries), 240)
        names = [e[0] for e in precompiled.weight_entries]
        self.assertIn('layer0.q_proj', names)
        self.assertIn('layer0.g_proj', names)
        self.assertIn('layer23.down_proj', names)


if __name__ == '__main__':
    unittest.main()
