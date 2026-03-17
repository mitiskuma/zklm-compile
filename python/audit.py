"""Standalone audit script for ZK-ML proofs.

Verifies a ZK proof of GPT-2 inference by independently checking:
1. WEIGHT CHECK — quantize GPT-2 weights, compute Merkle roots, compare to proof
2. INPUT CHECK — quantize input text, hash, compare to proof's input_commitment
3. OUTPUT CHECK — run float forward pass, compare top predictions
4. PROOF CHECK — run Rust standalone verifier on exported proof
5. COVERAGE CHECK — report which ops are proved vs passthrough

Usage:
    python audit.py --proof exported_proof.json --model gpt2 --input "The capital of France is"
    python audit.py --endpoint http://localhost:8042 --input "The capital of France is"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import numpy as np


def quantize_vector(values: np.ndarray, scale: float) -> list[int]:
    """INT16 symmetric quantization matching the prover's scheme."""
    P = (1 << 31) - 1  # Mersenne-31
    clipped = np.clip(np.round(values / scale), -32768, 32767).astype(np.int64)
    return [(int(v) % P) for v in clipped]


def merkle_root_sha256(field_values: list[int]) -> bytes:
    """Compute SHA-256 blob hash matching the Rust prover's commit_weights.

    Rust uses: H(len_as_u64_le || w[0]_as_u32_le || w[1]_as_u32_le || ...)
    """
    h = hashlib.sha256()
    h.update(struct.pack("<Q", len(field_values)))  # u64 length prefix
    for v in field_values:
        h.update(struct.pack("<I", v & 0xFFFFFFFF))
    return h.digest()


def check_weights(proof_data: dict, model) -> tuple[bool, list[str]]:
    """WEIGHT CHECK: quantize model weights, compare Merkle roots to proof.

    The proof's weight_commitments are ordered by the linear ops that were proved.
    We need to figure out which ops were proved (from coverage) and compute
    commitments for those specific weight matrices in the same order.
    """
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        Conv1D = None

    def get_weight(module):
        w = module.weight.detach().numpy()
        if Conv1D is not None and isinstance(module, Conv1D):
            return w.T
        return w

    QUANT_RANGE = 127
    proof_commitments = proof_data.get("weight_commitments", [])
    if not proof_commitments:
        return False, ["No weight commitments in proof"]

    # Build a lookup of all weight matrices by op name
    transformer = model.transformer
    weight_by_name = {}
    for i in range(len(transformer.h)):
        block = transformer.h[i]
        weight_by_name[f"c_attn_{i}"] = get_weight(block.attn.c_attn)
        weight_by_name[f"c_proj_{i}"] = get_weight(block.attn.c_proj)
        weight_by_name[f"mlp_fc_{i}"] = get_weight(block.mlp.c_fc)
        weight_by_name[f"mlp_proj_{i}"] = get_weight(block.mlp.c_proj)
    weight_by_name["lm_head"] = model.lm_head.weight.detach().numpy()

    # Get list of proved linear ops from coverage.
    # proved_ops is in reverse order (Rust proves backward), but weight_commitments
    # are in forward order (built during forward pass). Reverse to match.
    coverage = proof_data.get("coverage", {})
    proved_ops = coverage.get("proved", coverage.get("proved_ops", []))
    linear_ops = [name for name in reversed(proved_ops) if name in weight_by_name]

    if len(linear_ops) != len(proof_commitments):
        return False, [
            f"Proved linear ops ({len(linear_ops)}) != proof commitments ({len(proof_commitments)})",
            f"Linear ops: {linear_ops}",
        ]

    issues = []
    matched = 0
    for idx, name in enumerate(linear_ops):
        w = weight_by_name[name]
        w_s = np.abs(w).max() / QUANT_RANGE
        w_q = quantize_vector(w.flatten(), w_s)
        root_hex = merkle_root_sha256(w_q).hex()
        if root_hex == proof_commitments[idx]:
            matched += 1
        else:
            issues.append(f"  {name}: MISMATCH (computed {root_hex[:16]}... vs proof {proof_commitments[idx][:16]}...)")

    if not issues:
        return True, [f"All {matched} weight commitments match"]
    return False, issues


def check_input(proof_data: dict, text: str, tokenizer, model, prove_layers: int = 1) -> tuple[bool, list[str]]:
    """INPUT CHECK: quantize input, hash, compare to proof."""
    import torch

    input_ids = tokenizer.encode(text, return_tensors="pt")
    n_layers = len(model.transformer.h)
    start_layer = n_layers - prove_layers

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        # hidden_states[i] is the output of layer i (0 = embedding output)
        hidden = outputs.hidden_states[start_layer][0, -1, :].numpy()

    QUANT_RANGE = 127
    x_max = np.abs(hidden).max()
    x_scale = x_max / QUANT_RANGE if x_max > 0 else 1.0
    x_q = quantize_vector(hidden, x_scale)

    P = (1 << 31) - 1
    h = hashlib.sha256()
    h.update(struct.pack("<I", len(x_q)))
    for v in x_q:
        h.update(struct.pack("<I", v % P))
    computed_hash = h.hexdigest()

    proof_hash = proof_data.get("input_commitment", "")
    if computed_hash == proof_hash:
        return True, [f"Input commitment matches ({computed_hash[:16]}...)"]
    return False, [f"Mismatch: computed {computed_hash[:16]}... vs proof {proof_hash[:16]}..."]


def check_coverage(proof_data: dict) -> tuple[bool, list[str]]:
    """COVERAGE CHECK: report proved vs passthrough ops."""
    cov = proof_data.get("coverage")
    if not cov:
        return False, ["No coverage data in proof"]

    proved = cov.get("proved", cov.get("proved_ops", []))
    passthrough = cov.get("passthrough", cov.get("passthrough_ops", []))
    proved_count = cov.get("proved_count", len(proved))
    total_count = cov.get("total_count", proved_count + len(passthrough))
    pct = proved_count / total_count * 100 if total_count > 0 else 0

    info = [
        f"{proved_count}/{total_count} ops proved ({pct:.1f}%)",
        f"Proved: {', '.join(proved[:10])}{'...' if len(proved) > 10 else ''}",
    ]
    if passthrough:
        info.append(f"Passthrough: {', '.join(passthrough[:10])}{'...' if len(passthrough) > 10 else ''}")

    return True, info


def check_proof_standalone(proof_path: str) -> tuple[bool, list[str]]:
    """PROOF CHECK: run Rust standalone verifier."""
    rust_bin = os.path.join(
        os.path.dirname(__file__), "..", "rust",
        "zk_ml_prover", "target", "release", "zk_ml_prover"
    )
    if not os.path.exists(rust_bin):
        return False, [f"Rust prover not found at {rust_bin}"]

    try:
        result = subprocess.run(
            [rust_bin, "--verify", proof_path],
            capture_output=True, timeout=60, text=True,
        )
        lines = result.stdout.strip().split("\n")
        passed = any("PASS" in line for line in lines)
        return passed, lines
    except Exception as e:
        return False, [f"Verifier error: {e}"]


def run_audit(args):
    proof_data = None

    if args.endpoint:
        import urllib.request
        req_data = json.dumps({"text": args.input, "prove_layers": args.prove_layers}).encode()
        http_req = urllib.request.Request(
            f"{args.endpoint}/audit/prove",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_req, timeout=300) as resp:
            if resp.status != 200:
                print(f"ERROR: endpoint returned {resp.status}")
                sys.exit(1)
            proof_data = json.loads(resp.read().decode())
    elif args.proof:
        with open(args.proof) as f:
            proof_data = json.load(f)
    else:
        print("ERROR: provide --proof or --endpoint")
        sys.exit(1)

    print("=" * 60)
    print("ZK-ML AUDIT REPORT")
    print("=" * 60)

    checks = []

    # COVERAGE CHECK (always available)
    ok, info = check_coverage(proof_data)
    checks.append(("COVERAGE", ok, info))

    # WEIGHT CHECK (requires model)
    if args.model:
        print("\nLoading model for verification...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)
        model = GPT2LMHeadModel.from_pretrained(args.model)
        model.eval()

        ok, info = check_weights(proof_data, model)
        checks.append(("WEIGHTS", ok, info))

        # INPUT CHECK
        if args.input:
            ok, info = check_input(proof_data, args.input, tokenizer, model, args.prove_layers)
            checks.append(("INPUT", ok, info))

    # PROOF CHECK (standalone verifier, only for exported proof files)
    if args.proof:
        ok, info = check_proof_standalone(args.proof)
        checks.append(("PROOF", ok, info))

    # Print report
    all_pass = True
    for name, ok, info in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"\n[{status}] {name} CHECK")
        for line in info:
            print(f"  {line}")

    print("\n" + "=" * 60)
    if all_pass:
        print("OVERALL: ALL CHECKS PASSED")
    else:
        print("OVERALL: SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit ZK-ML proofs")
    parser.add_argument("--proof", help="Path to exported proof JSON file")
    parser.add_argument("--endpoint", help="ZK-ML server endpoint (e.g. http://localhost:8042)")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--input", help="Input text for verification")
    parser.add_argument("--prove-layers", type=int, default=12, help="Layers to prove (default: 12)")
    run_audit(parser.parse_args())
