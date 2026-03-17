"""Verification logic for ZK-ML proofs.

Extracted from audit.py — provides structured result types and clean API
for checking proof integrity without depending on TVM or the prover codebase.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Conditional torch/transformers imports
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Mersenne-31 prime (inline to avoid cross-package dependency)
_M31 = (1 << 31) - 1

# INT8 symmetric quantization range used by the prover
_QUANT_RANGE = 127


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single audit check."""

    name: str
    passed: bool
    message: str
    details: list[str] = field(default_factory=list)


@dataclass
class CoverageResult:
    """Result of the coverage check."""

    passed: bool
    proved_count: int
    total_count: int
    percentage: float
    proved_ops: list[str]
    passthrough_ops: list[str]
    message: str


@dataclass
class WeightResult:
    """Result of the weight commitment check."""

    passed: bool
    matched: int
    total: int
    message: str
    mismatches: list[str] = field(default_factory=list)


@dataclass
class InputResult:
    """Result of the input commitment check."""

    passed: bool
    computed_hash: str
    proof_hash: str
    message: str


@dataclass
class AuditReport:
    """Full audit report containing all check results."""

    checks: list[CheckResult]

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _quantize_vector(values: np.ndarray, scale: float) -> list[int]:
    """INT16 symmetric quantization matching the prover's scheme."""
    clipped = np.clip(np.round(values / scale), -32768, 32767).astype(np.int64)
    return [(int(v) % _M31) for v in clipped]


def _merkle_root_sha256(field_values: list[int]) -> bytes:
    """Compute SHA-256 blob hash matching the Rust prover's commit_weights.

    Rust uses: H(len_as_u64_le || w[0]_as_u32_le || w[1]_as_u32_le || ...)
    """
    h = hashlib.sha256()
    h.update(struct.pack("<Q", len(field_values)))  # u64 length prefix
    for v in field_values:
        h.update(struct.pack("<I", v & 0xFFFFFFFF))
    return h.digest()


def _require_torch():
    """Raise ImportError if torch is not available."""
    if not HAS_TORCH:
        raise ImportError(
            "Install zk-ml-audit[model] for model-based verification: "
            "pip install 'zk-ml-audit[model]'"
        )


def _get_weight(module):
    """Extract weight matrix from a PyTorch module, transposing Conv1D weights."""
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        Conv1D = None

    w = module.weight.detach().numpy()
    if Conv1D is not None and isinstance(module, Conv1D):
        return w.T
    return w


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_coverage(proof_data: dict) -> CoverageResult:
    """COVERAGE CHECK: report proved vs passthrough ops.

    Examines the proof's coverage data to determine what percentage of
    operations are cryptographically proved vs passed through without proof.
    """
    cov = proof_data.get("coverage")
    if not cov:
        return CoverageResult(
            passed=False,
            proved_count=0,
            total_count=0,
            percentage=0.0,
            proved_ops=[],
            passthrough_ops=[],
            message="No coverage data in proof",
        )

    proved = cov.get("proved", cov.get("proved_ops", []))
    passthrough = cov.get("passthrough", cov.get("passthrough_ops", []))
    proved_count = cov.get("proved_count", len(proved))
    total_count = cov.get("total_count", proved_count + len(passthrough))
    pct = proved_count / total_count * 100 if total_count > 0 else 0.0

    return CoverageResult(
        passed=True,
        proved_count=proved_count,
        total_count=total_count,
        percentage=pct,
        proved_ops=proved,
        passthrough_ops=passthrough,
        message=f"{proved_count}/{total_count} ops proved ({pct:.1f}%)",
    )


def check_weights(proof_data: dict, model_name: str) -> WeightResult:
    """WEIGHT CHECK: quantize model weights, compare Merkle roots to proof.

    Loads the model from HuggingFace, quantizes each weight matrix using
    the same INT16 symmetric scheme as the prover, computes SHA-256
    commitments, and compares them against the proof's weight_commitments.

    Parameters
    ----------
    proof_data : dict
        Parsed proof JSON.
    model_name : str
        HuggingFace model identifier (e.g. "gpt2").

    Returns
    -------
    WeightResult
    """
    _require_torch()

    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    proof_commitments = proof_data.get("weight_commitments", [])
    if not proof_commitments:
        return WeightResult(
            passed=False, matched=0, total=0,
            message="No weight commitments in proof",
        )

    # Build lookup of all weight matrices by op name
    transformer = model.transformer
    weight_by_name: dict[str, np.ndarray] = {}
    for i in range(len(transformer.h)):
        block = transformer.h[i]
        weight_by_name[f"c_attn_{i}"] = _get_weight(block.attn.c_attn)
        weight_by_name[f"c_proj_{i}"] = _get_weight(block.attn.c_proj)
        weight_by_name[f"mlp_fc_{i}"] = _get_weight(block.mlp.c_fc)
        weight_by_name[f"mlp_proj_{i}"] = _get_weight(block.mlp.c_proj)
    weight_by_name["lm_head"] = model.lm_head.weight.detach().numpy()

    # Get list of proved linear ops from coverage.
    # proved_ops is in reverse order (Rust proves backward), but
    # weight_commitments are in forward order (built during forward pass).
    # Reverse to match.
    coverage = proof_data.get("coverage", {})
    proved_ops = coverage.get("proved", coverage.get("proved_ops", []))
    linear_ops = [name for name in reversed(proved_ops) if name in weight_by_name]

    if len(linear_ops) != len(proof_commitments):
        return WeightResult(
            passed=False, matched=0, total=len(proof_commitments),
            message=(
                f"Proved linear ops ({len(linear_ops)}) != "
                f"proof commitments ({len(proof_commitments)})"
            ),
            mismatches=[f"Linear ops: {linear_ops}"],
        )

    mismatches = []
    matched = 0
    for idx, name in enumerate(linear_ops):
        w = weight_by_name[name]
        w_s = np.abs(w).max() / _QUANT_RANGE
        w_q = _quantize_vector(w.flatten(), w_s)
        root_hex = _merkle_root_sha256(w_q).hex()
        if root_hex == proof_commitments[idx]:
            matched += 1
        else:
            mismatches.append(
                f"{name}: MISMATCH (computed {root_hex[:16]}... "
                f"vs proof {proof_commitments[idx][:16]}...)"
            )

    if not mismatches:
        return WeightResult(
            passed=True, matched=matched, total=matched,
            message=f"All {matched} weight commitments match",
        )
    return WeightResult(
        passed=False, matched=matched, total=matched + len(mismatches),
        message=f"{len(mismatches)} weight commitment(s) failed",
        mismatches=mismatches,
    )


def check_input(proof_data: dict, text: str, model_name: str, prove_layers: int = 12) -> InputResult:
    """INPUT CHECK: quantize input, hash, compare to proof.

    Runs the model's forward pass up to the start layer, quantizes the
    hidden state, hashes it, and compares against the proof's input_commitment.

    Parameters
    ----------
    proof_data : dict
        Parsed proof JSON.
    text : str
        The input text that was proved.
    model_name : str
        HuggingFace model identifier (e.g. "gpt2").
    prove_layers : int
        Number of layers that were proved (default: 12, i.e. full model).

    Returns
    -------
    InputResult
    """
    _require_torch()

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    input_ids = tokenizer.encode(text, return_tensors="pt")
    n_layers = len(model.transformer.h)
    start_layer = n_layers - prove_layers

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[start_layer][0, -1, :].numpy()

    x_max = np.abs(hidden).max()
    x_scale = x_max / _QUANT_RANGE if x_max > 0 else 1.0
    x_q = _quantize_vector(hidden, x_scale)

    h = hashlib.sha256()
    h.update(struct.pack("<I", len(x_q)))
    for v in x_q:
        h.update(struct.pack("<I", v % _M31))
    computed_hash = h.hexdigest()

    proof_hash = proof_data.get("input_commitment", "")
    if computed_hash == proof_hash:
        return InputResult(
            passed=True,
            computed_hash=computed_hash,
            proof_hash=proof_hash,
            message=f"Input commitment matches ({computed_hash[:16]}...)",
        )
    return InputResult(
        passed=False,
        computed_hash=computed_hash,
        proof_hash=proof_hash,
        message=f"Mismatch: computed {computed_hash[:16]}... vs proof {proof_hash[:16]}...",
    )


def verify_proof(proof_path: str) -> bool:
    """PROOF CHECK: run Rust standalone verifier on an exported proof file.

    Searches for the Rust binary in several locations:
    1. Next to this package (development layout)
    2. Standard project location (zk/rust/zk_ml_prover)
    3. On PATH

    Returns True if verification passes, False otherwise.
    """
    # Try standard project location relative to this file
    pkg_dir = Path(__file__).parent
    candidates = [
        pkg_dir / ".." / ".." / ".." / ".." / "rust" / "zk_ml_prover" / "target" / "release" / "zk_ml_prover",
        Path("rust") / "zk_ml_prover" / "target" / "release" / "zk_ml_prover",
    ]

    rust_bin = None
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            rust_bin = str(resolved)
            break

    # Fall back to PATH
    if rust_bin is None:
        import shutil
        rust_bin = shutil.which("zk_ml_prover")

    if rust_bin is None:
        raise FileNotFoundError(
            "Rust prover binary not found. Build with: "
            "cd rust/zk_ml_prover && cargo build --release"
        )

    try:
        result = subprocess.run(
            [rust_bin, "--verify", proof_path],
            capture_output=True, timeout=60, text=True,
        )
        return any("PASS" in line for line in result.stdout.strip().split("\n"))
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def audit_full(
    proof_data: dict,
    *,
    model_name: Optional[str] = None,
    input_text: Optional[str] = None,
    proof_path: Optional[str] = None,
    prove_layers: int = 12,
    verbose: bool = False,
) -> AuditReport:
    """Run all applicable audit checks and return a structured report.

    Parameters
    ----------
    proof_data : dict
        Parsed proof JSON.
    model_name : str, optional
        HuggingFace model name for weight/input verification.
    input_text : str, optional
        Input text for input commitment verification.
    proof_path : str, optional
        Path to proof file for standalone Rust verification.
    prove_layers : int
        Number of layers proved (default: 12).
    verbose : bool
        Include extra detail in messages.

    Returns
    -------
    AuditReport
    """
    checks: list[CheckResult] = []

    # 1. COVERAGE CHECK (always available)
    cov = check_coverage(proof_data)
    details = [cov.message]
    if cov.proved_ops:
        ops_str = ", ".join(cov.proved_ops[:10])
        if len(cov.proved_ops) > 10:
            ops_str += "..."
        details.append(f"Proved: {ops_str}")
    if cov.passthrough_ops:
        ops_str = ", ".join(cov.passthrough_ops[:10])
        if len(cov.passthrough_ops) > 10:
            ops_str += "..."
        details.append(f"Passthrough: {ops_str}")
    checks.append(CheckResult(
        name="COVERAGE",
        passed=cov.passed,
        message=cov.message,
        details=details if verbose else [],
    ))

    # 2. WEIGHT CHECK (requires model)
    if model_name and HAS_TORCH:
        wr = check_weights(proof_data, model_name)
        checks.append(CheckResult(
            name="WEIGHTS",
            passed=wr.passed,
            message=wr.message,
            details=wr.mismatches if verbose else [],
        ))

        # 3. INPUT CHECK (requires model + input text)
        if input_text:
            ir = check_input(proof_data, input_text, model_name, prove_layers)
            checks.append(CheckResult(
                name="INPUT",
                passed=ir.passed,
                message=ir.message,
            ))

    # 4. PROOF CHECK (standalone verifier)
    if proof_path:
        try:
            ok = verify_proof(proof_path)
            checks.append(CheckResult(
                name="PROOF",
                passed=ok,
                message="Standalone verification passed" if ok else "Standalone verification failed",
            ))
        except FileNotFoundError as e:
            checks.append(CheckResult(
                name="PROOF",
                passed=False,
                message=str(e),
            ))

    return AuditReport(checks=checks)
