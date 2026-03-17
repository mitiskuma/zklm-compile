"""FastAPI backend for ZK-ML proof system.

Serves GPT-2 next-token prediction with ZK proofs via structured sumcheck
over Mersenne-31. Rust prover runs as a persistent subprocess.

Usage:
    cd zk && .venv/bin/python3 python/serve.py

Requirements:
    pip install transformers fastapi uvicorn
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not installed. Install with: pip install transformers")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tvm_to_gkr.structured_pipeline import (
    PrecompiledTransformerWeights,
    RustProverServer,
    prove_rust_transformer_v2,
)
from tvm_to_gkr.llama_pipeline import prove_llama_token, prove_llama_token_server
from tvm_to_gkr.qwen35_pipeline import prove_qwen35_token_server, QwenPrecompiledWeights
from tvm_to_gkr.model_extractor import ModelExtractor
from tvm_to_gkr.structured_pipeline import LlamaPrecompiledWeights

app = FastAPI(title="ZK-ML Proof Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global state -- initialized at startup
gpt2_model = None
gpt2_tokenizer = None
gpt2_precompiled = None
gpt2_prover_server = None      # CPU prover
gpt2_prover_server_gpu = None  # Metal GPU prover (if available)

# On-demand PCS prover server cache: "(backend):(pcs_mode)" -> RustProverServer
prover_servers: dict[str, RustProverServer] = {}

# Lazy-loaded Llama model cache: model_name -> {extractor, weights, tokenizer}
llama_cache: dict = {}

MAX_PROVE_LAYERS = 32
MAX_TEXT_LENGTH = 2048
MAX_TOKENS = 200


# --- Request/Response models ---

class ProofInfo(BaseModel):
    status: str
    constraints: int = 0
    prove_time: float | None = None
    verify_time: float | None = None
    proof_size: int | None = None
    layers_proved: int = 0
    total_layers: int = 0
    ops_proved: int | None = None
    computational_pct: float | None = None
    total_pct: float | None = None


class TokenProb(BaseModel):
    token: str
    token_id: int
    prob: float


class TransformerRequest(BaseModel):
    text: str
    prove_layers: int = 1

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(f"text must be <= {MAX_TEXT_LENGTH} characters")
        return v

    @field_validator("prove_layers")
    @classmethod
    def prove_layers_range(cls, v):
        if v < 1 or v > MAX_PROVE_LAYERS:
            raise ValueError(f"prove_layers must be between 1 and {MAX_PROVE_LAYERS}")
        return v


class TransformerResponse(BaseModel):
    predicted_token: str
    predicted_token_id: int
    top_tokens: list[TokenProb]
    proof: ProofInfo | None = None


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 50
    prove_layers: int = 1
    model: str = "gpt2"
    backend: str = "cpu"
    pcs_mode: str = "default"

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages must not be empty")
        return v

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_range(cls, v):
        return min(max(1, v), MAX_TOKENS)

    @field_validator("prove_layers")
    @classmethod
    def prove_layers_range(cls, v):
        return min(max(1, v), MAX_PROVE_LAYERS)


class VerifyTokensRequest(BaseModel):
    context: str
    tokens: list[str]
    prove_layers: int = 1
    model: str = "gpt2"
    backend: str = "cpu"
    pcs_mode: str = "default"

    @field_validator("context")
    @classmethod
    def context_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("context must not be empty")
        return v

    @field_validator("tokens")
    @classmethod
    def tokens_not_empty(cls, v):
        if not v:
            raise ValueError("tokens must not be empty")
        return v


class AuditProveRequest(BaseModel):
    text: str
    prove_layers: int = 12


# --- Lifecycle ---

@app.on_event("startup")
def startup():
    global gpt2_model, gpt2_tokenizer, gpt2_precompiled, gpt2_prover_server, gpt2_prover_server_gpu

    if not HAS_TRANSFORMERS:
        logger.error("Cannot start without transformers. Install with: pip install transformers")
        return

    logger.info("Loading GPT-2 (124M params)...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()
    logger.info("GPT-2 loaded.")

    logger.info("Pre-compiling transformer weights...")
    gpt2_precompiled = PrecompiledTransformerWeights(gpt2_model)

    logger.info("Starting Rust prover server (CPU)...")
    gpt2_prover_server = RustProverServer(gpt2_precompiled)
    logger.info("CPU prover ready.")

    try:
        logger.info("Starting Rust prover server (Metal GPU)...")
        gpt2_prover_server_gpu = RustProverServer(gpt2_precompiled, gpu=True)
        logger.info("Metal GPU prover ready.")
    except Exception as e:
        logger.warning(f"Metal GPU prover not available: {e}")
        gpt2_prover_server_gpu = None


def _check_ready():
    """Raise 503 if GPT-2 is not loaded."""
    if gpt2_model is None or gpt2_tokenizer is None:
        raise HTTPException(status_code=503, detail="GPT-2 not loaded")


def _get_prover_server(backend: str = "cpu", pcs_mode: str = "default"):
    """Return the appropriate prover server for the requested backend and PCS mode.

    For default PCS mode, returns the pre-started servers.
    For non-default PCS modes, spawns servers on-demand and caches them.
    """
    global prover_servers

    # Fast path: default mode uses the pre-started servers
    if pcs_mode == "default":
        if backend == "metal" and gpt2_prover_server_gpu is not None:
            return gpt2_prover_server_gpu
        return gpt2_prover_server

    # On-demand path: check cache, spawn if needed
    cache_key = f"{backend}:{pcs_mode}"
    if cache_key not in prover_servers:
        gpu = backend == "metal"
        logger.info("Starting on-demand prover server: backend=%s, pcs_mode=%s", backend, pcs_mode)
        prover_servers[cache_key] = RustProverServer(
            gpt2_precompiled, gpu=gpu, pcs_mode=pcs_mode
        )
        logger.info("On-demand prover server ready: %s", cache_key)
    return prover_servers[cache_key]


def _run_proof(prove_input: torch.Tensor, prove_layers: int, verbose: bool = True, backend: str = "cpu", pcs_mode: str = "default") -> ProofInfo:
    """Run the Rust prover and return ProofInfo. Logs errors instead of swallowing them."""
    try:
        server = _get_prover_server(backend, pcs_mode)
        result = prove_rust_transformer_v2(
            gpt2_model, prove_input,
            prove_layers=prove_layers, verbose=verbose,
            precompiled=gpt2_precompiled, server=server,
        )
        coverage = result.get("coverage")
        ops_proved = coverage["proved_count"] if coverage else prove_layers * 5

        return ProofInfo(
            status="VALID" if result["valid"] else "FAILED",
            constraints=ops_proved,
            prove_time=result["prove_time_ms"] / 1000.0,
            verify_time=result["verify_time_ms"] / 1000.0,
            proof_size=result["proof_size_bytes"],
            layers_proved=prove_layers,
            total_layers=12,
            ops_proved=ops_proved,
            computational_pct=coverage.get("computational_pct", 100.0) if coverage else None,
            total_pct=coverage.get("total_pct", 100.0) if coverage else None,
        )
    except Exception as e:
        logger.error("Proof failed: %s\n%s", e, traceback.format_exc())
        return ProofInfo(
            status=f"ERROR: {str(e)[:100]}",
            layers_proved=prove_layers,
            total_layers=12,
        )


def _get_llama(model_name: str) -> dict:
    """Lazy-load and cache a Llama-style or Qwen3.5 model with server-mode prover."""
    if model_name not in llama_cache:
        from transformers import AutoTokenizer
        logger.info("Loading model: %s (first request, may be slow)...", model_name)
        extractor = ModelExtractor(model_name)
        weights = extractor.extract()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        is_qwen = extractor.config.layer_types is not None
        if is_qwen:
            logger.info("Detected Qwen3.5 (GatedDeltaNet hybrid), using QwenPrecompiledWeights...")
            precompiled = QwenPrecompiledWeights(extractor, weights)
        else:
            logger.info("Pre-compiling Llama weights for server mode...")
            precompiled = LlamaPrecompiledWeights(extractor, weights)

        server = RustProverServer(weight_entries=precompiled.weight_entries)
        prove_fn = prove_qwen35_token_server if is_qwen else prove_llama_token_server

        llama_cache[model_name] = {
            'extractor': extractor,
            'weights': weights,
            'tokenizer': tokenizer,
            'config': extractor.config,
            'precompiled': precompiled,
            'server': server,
            'prove_fn': prove_fn,
        }
        logger.info("Model %s loaded and cached with server-mode prover.", model_name)
    return llama_cache[model_name]


def _run_llama_proof(model_name: str, text: str, prove_layers: int) -> ProofInfo:
    """Run Llama/Qwen proof via server-mode prover and return ProofInfo."""
    try:
        cached = _get_llama(model_name)
        prove_fn = cached.get('prove_fn', prove_llama_token_server)
        result = prove_fn(
            cached['server'], cached['extractor'], text,
            prove_layers=prove_layers,
        )
        return ProofInfo(
            status="VALID" if result.get("valid") else "FAILED",
            constraints=result.get("ops_proved", prove_layers),
            prove_time=result.get("prove_time_ms", 0) / 1000.0,
            verify_time=result.get("verify_time_ms", 0) / 1000.0,
            proof_size=result.get("proof_size_bytes", 0),
            layers_proved=prove_layers,
            total_layers=cached['config'].n_layers,
            ops_proved=result.get("ops_proved", prove_layers),
        )
    except Exception as e:
        logger.error("Llama proof failed: %s\n%s", e, traceback.format_exc())
        return ProofInfo(
            status=f"ERROR: {str(e)[:100]}",
            layers_proved=prove_layers,
            total_layers=0,
        )


# --- Endpoints ---

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok" if gpt2_model is not None else "loading",
        "model_loaded": gpt2_model is not None,
        "prover_ready": gpt2_prover_server is not None,
        "gpu_available": gpt2_prover_server_gpu is not None,
    }


@app.get("/system-info")
def system_info():
    """Return system status for the ZK-ML proof system."""
    return {
        "status": "ready" if gpt2_model is not None else "loading",
        "model": {
            "type": "Transformer",
            "name": "GPT-2",
            "params": "124M",
            "d_model": 768,
            "d_ff": 3072,
            "n_layers": 12,
            "vocab_size": 50257,
            "proof_system": "Sumcheck (Mersenne-31)",
            "proved_ops": [
                "LayerNorm (QR perturbation)", "c_attn (QKV proj)",
                "c_proj (attn out)", "c_fc (MLP up)", "GELU (LogUp lookup)",
                "mlp_proj (MLP down)", "LM head (50257x768)",
                "residual add (claim chaining)",
            ],
            "unproved_ops": ["Attention (softmax+dot for seq_len>1)"],
            "server_mode": gpt2_prover_server is not None,
            "precompiled": gpt2_precompiled is not None,
        },
        "benchmarks": {
            "gpt2_12_layers_server": {
                "prove_time_ms": 1191,
                "verify_time_ms": 3.1,
                "proof_size_bytes": 94631,
                "wall_clock_ms": 1265,
                "proved_ops": 60,
                "note": "Server mode (preloaded weights). Includes LN, LM head, residuals.",
            },
        },
    }


@app.post("/prove-transformer", response_model=TransformerResponse)
def prove_transformer(req: TransformerRequest):
    """Run GPT-2 next-token prediction and prove with ZK."""
    _check_ready()

    prove_layers = min(req.prove_layers, MAX_PROVE_LAYERS)
    input_ids = gpt2_tokenizer.encode(req.text, return_tensors="pt")

    with torch.no_grad():
        outputs = gpt2_model(input_ids, output_hidden_states=True)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, 10)
        layer_idx = 12 - prove_layers
        prove_input = outputs.hidden_states[layer_idx][0, -1, :]

    predicted_id = int(top_ids[0].item())
    predicted_token = gpt2_tokenizer.decode([predicted_id])
    top_tokens = [
        TokenProb(
            token=gpt2_tokenizer.decode([int(top_ids[i].item())]),
            token_id=int(top_ids[i].item()),
            prob=float(top_probs[i].item()),
        )
        for i in range(10)
    ]

    proof_info = _run_proof(prove_input, prove_layers)

    return TransformerResponse(
        predicted_token=predicted_token,
        predicted_token_id=predicted_id,
        top_tokens=top_tokens,
        proof=proof_info,
    )


@app.get("/audit/weights")
def audit_weights():
    """Return per-matrix weight commitment roots + quantization scales."""
    _check_ready()
    if gpt2_precompiled is None:
        raise HTTPException(status_code=503, detail="Weights not pre-compiled")

    pc = gpt2_precompiled
    commitments = []
    for i, lw in enumerate(pc.layers):
        for key, label in [
            ("c_attn_w_s", f"c_attn_{i}"),
            ("attn_w_s", f"c_proj_{i}"),
            ("fc1_w_s", f"mlp_fc_{i}"),
            ("fc2_w_s", f"mlp_proj_{i}"),
        ]:
            commitments.append({
                "name": label,
                "quant_scale": float(lw[key]),
                "quant_range": 127,
            })
    commitments.append({
        "name": "lm_head",
        "quant_scale": float(pc.lm_head_w_s),
        "quant_range": 127,
    })

    return {
        "model": "gpt2",
        "source": "https://huggingface.co/gpt2",
        "quantization": "INT8 symmetric (scale = max(|w|) / 127)",
        "field": "Mersenne-31 (p = 2^31 - 1)",
        "commitments": commitments,
    }


@app.post("/audit/prove")
def audit_prove(req: AuditProveRequest):
    """Full audit endpoint: prove + return proof artifact, commitments, coverage."""
    _check_ready()

    prove_layers = min(req.prove_layers, MAX_PROVE_LAYERS)
    input_ids = gpt2_tokenizer.encode(req.text, return_tensors="pt")

    with torch.no_grad():
        outputs = gpt2_model(input_ids, output_hidden_states=True)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, 10)
        layer_idx = 12 - prove_layers
        prove_input = outputs.hidden_states[layer_idx][0, -1, :]

    result = prove_rust_transformer_v2(
        gpt2_model, prove_input,
        prove_layers=prove_layers, verbose=True,
        precompiled=gpt2_precompiled, server=gpt2_prover_server,
    )

    return {
        "valid": result["valid"],
        "predicted_token": gpt2_tokenizer.decode([int(top_ids[0].item())]),
        "predicted_token_id": int(top_ids[0].item()),
        "prove_time_ms": result["prove_time_ms"],
        "verify_time_ms": result["verify_time_ms"],
        "proof_size_bytes": result["proof_size_bytes"],
        "weight_commitments": result.get("weight_commitments", []),
        "input_commitment": result.get("input_commitment", ""),
        "coverage": result.get("coverage"),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """Stream tokens with per-token ZK proofs via SSE. Supports GPT-2 and Llama-style models."""
    is_gpt2 = req.model == "gpt2"

    if is_gpt2:
        _check_ready()

    prove_layers = min(req.prove_layers, MAX_PROVE_LAYERS)

    async def generate():
        user_text = req.messages[-1].get("content", "") if req.messages else ""
        if not user_text.strip():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Empty message'})}\n\n"
            return

        total_prove_ms = 0.0
        total_verify_ms = 0.0
        total_tokens = 0
        all_valid = True

        if is_gpt2:
            # --- GPT-2 path (existing) ---
            context_ids = gpt2_tokenizer.encode(user_text)
            input_ids = torch.tensor([context_ids])

            for i in range(req.max_tokens):
                with torch.no_grad():
                    outputs = gpt2_model(input_ids, output_hidden_states=True)
                    logits = outputs.logits[0, -1, :]
                    next_id = torch.argmax(logits, dim=-1).item()
                    token_str = gpt2_tokenizer.decode([next_id])

                layer_idx = 12 - prove_layers
                h = outputs.hidden_states[layer_idx][0, -1, :]

                proof_data = {"status": "SKIPPED", "prove_time_ms": 0, "verify_time_ms": 0, "proof_size": 0, "ops_proved": 0}
                try:
                    result = prove_rust_transformer_v2(
                        gpt2_model, h, prove_layers=prove_layers,
                        verbose=False, precompiled=gpt2_precompiled,
                        server=_get_prover_server(req.backend, req.pcs_mode),
                    )
                    coverage = result.get("coverage")
                    ops_proved = coverage["proved_count"] if coverage else prove_layers * 5
                    comp_pct = coverage.get("computational_pct", 100.0) if coverage else 100.0
                    total_pct_val = coverage.get("total_pct", 100.0) if coverage else 100.0

                    # Soundness bits based on PCS mode
                    soundness = "124-bit" if req.pcs_mode == "default" else "124+219-bit"

                    proof_data = {
                        "status": "VALID" if result["valid"] else "FAILED",
                        "prove_time_ms": round(result["prove_time_ms"], 1),
                        "verify_time_ms": round(result["verify_time_ms"], 1),
                        "proof_size": result["proof_size_bytes"],
                        "ops_proved": ops_proved,
                        "computational_pct": comp_pct,
                        "total_pct": total_pct_val,
                        "pcs_mode": req.pcs_mode,
                        "soundness": soundness,
                    }
                    total_prove_ms += result["prove_time_ms"]
                    total_verify_ms += result["verify_time_ms"]
                    if not result["valid"]:
                        all_valid = False
                except Exception as e:
                    logger.error("Proof error on token %d: %s", i, e)
                    proof_data = {"status": "ERROR", "prove_time_ms": 0, "verify_time_ms": 0, "proof_size": 0, "ops_proved": 0, "error": str(e)[:100]}
                    all_valid = False

                total_tokens += 1
                yield f"data: {json.dumps({'type': 'token', 'token': token_str, 'token_id': next_id, 'proof': proof_data})}\n\n"

                input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
                if next_id == gpt2_tokenizer.eos_token_id:
                    break
        else:
            # --- Llama/Qwen path ---
            try:
                cached = _get_llama(req.model)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to load model: {str(e)[:200]}'})}\n\n"
                return

            tokenizer = cached['tokenizer']
            model = cached['extractor'].model
            context_ids = tokenizer.encode(user_text, return_tensors="pt")
            input_ids = context_ids
            llama_prove_layers = min(prove_layers, cached['config'].n_layers)

            for i in range(req.max_tokens):
                with torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)
                    logits = outputs.logits[0, -1, :]
                    next_id = torch.argmax(logits, dim=-1).item()
                    token_str = tokenizer.decode([next_id])

                # Build full context text for proving
                full_text = tokenizer.decode(input_ids[0].tolist())

                proof_data = {"status": "SKIPPED", "prove_time_ms": 0, "verify_time_ms": 0, "proof_size": 0, "ops_proved": 0}
                try:
                    prove_fn = cached.get('prove_fn', prove_llama_token_server)
                    result = prove_fn(
                        cached['server'], cached['extractor'], full_text,
                        prove_layers=llama_prove_layers,
                    )
                    proof_data = {
                        "status": "VALID" if result.get("valid") else "FAILED",
                        "prove_time_ms": round(result.get("prove_time_ms", 0), 1),
                        "verify_time_ms": round(result.get("verify_time_ms", 0), 1),
                        "proof_size": result.get("proof_size_bytes", 0),
                        "ops_proved": result.get("ops_proved", llama_prove_layers),
                    }
                    total_prove_ms += result.get("prove_time_ms", 0)
                    total_verify_ms += result.get("verify_time_ms", 0)
                    if not result.get("valid"):
                        all_valid = False
                except Exception as e:
                    logger.error("Llama proof error on token %d: %s", i, e)
                    proof_data = {"status": "ERROR", "prove_time_ms": 0, "verify_time_ms": 0, "proof_size": 0, "ops_proved": 0, "error": str(e)[:100]}
                    all_valid = False

                total_tokens += 1
                yield f"data: {json.dumps({'type': 'token', 'token': token_str, 'token_id': next_id, 'proof': proof_data})}\n\n"

                input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
                eos_id = tokenizer.eos_token_id
                if eos_id is not None and next_id == eos_id:
                    break

        yield f"data: {json.dumps({'type': 'done', 'total_tokens': total_tokens, 'total_prove_time_ms': round(total_prove_ms, 1), 'total_verify_time_ms': round(total_verify_ms, 1), 'all_valid': all_valid})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/verify-tokens")
async def verify_tokens(req: VerifyTokensRequest):
    """Re-run model + ZK proof for a sequence of claimed tokens."""
    is_gpt2 = req.model == "gpt2"

    if is_gpt2:
        _check_ready()

    prove_layers = min(req.prove_layers, MAX_PROVE_LAYERS)

    async def generate():
        total_prove_ms = 0.0
        total_verify_ms = 0.0
        all_valid = True
        all_match = True

        if is_gpt2:
            # --- GPT-2 path ---
            context_ids = gpt2_tokenizer.encode(req.context)
            input_ids = torch.tensor([context_ids])

            for i, claimed_token in enumerate(req.tokens):
                with torch.no_grad():
                    outputs = gpt2_model(input_ids, output_hidden_states=True)
                    logits = outputs.logits[0, -1, :]
                    predicted_id = torch.argmax(logits, dim=-1).item()
                    predicted_token = gpt2_tokenizer.decode([predicted_id])

                claimed_ids = gpt2_tokenizer.encode(claimed_token)
                token_matches = (len(claimed_ids) == 1 and claimed_ids[0] == predicted_id)

                layer_idx = 12 - prove_layers
                h = outputs.hidden_states[layer_idx][0, -1, :]

                proof_data = {"status": "SKIPPED", "prove_time_ms": 0, "verify_time_ms": 0, "proof_size": 0, "ops_proved": 0}
                try:
                    result = prove_rust_transformer_v2(
                        gpt2_model, h, prove_layers=prove_layers,
                        verbose=False, precompiled=gpt2_precompiled,
                        server=_get_prover_server(req.backend, req.pcs_mode),
                    )
                    coverage = result.get("coverage")
                    ops_proved = coverage["proved_count"] if coverage else prove_layers * 5
                    soundness = "124-bit" if req.pcs_mode == "default" else "124+219-bit"
                    proof_data = {
                        "status": "VALID" if result["valid"] else "FAILED",
                        "prove_time_ms": round(result["prove_time_ms"], 1),
                        "verify_time_ms": round(result["verify_time_ms"], 1),
                        "proof_size": result["proof_size_bytes"],
                        "ops_proved": ops_proved,
                        "pcs_mode": req.pcs_mode,
                        "soundness": soundness,
                    }
                    total_prove_ms += result["prove_time_ms"]
                    total_verify_ms += result["verify_time_ms"]
                    if not result["valid"]:
                        all_valid = False
                except Exception as e:
                    logger.error("Proof error on token %d: %s", i, e)
                    proof_data = {"status": "ERROR", "prove_time_ms": 0, "verify_time_ms": 0, "proof_size": 0, "ops_proved": 0}
                    all_valid = False

                if not token_matches:
                    all_match = False

                yield f"data: {json.dumps({'type': 'token', 'index': i, 'claimed_token': claimed_token, 'predicted_token': predicted_token, 'predicted_token_id': predicted_id, 'matches': token_matches, 'proof': proof_data})}\n\n"

                if claimed_ids:
                    input_ids = torch.cat([input_ids, torch.tensor([claimed_ids[:1]])], dim=1)
                else:
                    input_ids = torch.cat([input_ids, torch.tensor([[predicted_id]])], dim=1)
        else:
            # --- Llama/Qwen path ---
            try:
                cached = _get_llama(req.model)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to load model: {str(e)[:200]}'})}\n\n"
                return

            tokenizer = cached['tokenizer']
            model = cached['extractor'].model
            context_ids = tokenizer.encode(req.context, return_tensors="pt")
            input_ids = context_ids
            llama_prove_layers = min(prove_layers, cached['config'].n_layers)

            for i, claimed_token in enumerate(req.tokens):
                with torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)
                    logits = outputs.logits[0, -1, :]
                    predicted_id = torch.argmax(logits, dim=-1).item()
                    predicted_token = tokenizer.decode([predicted_id])

                claimed_ids = tokenizer.encode(claimed_token, add_special_tokens=False)
                token_matches = (len(claimed_ids) == 1 and claimed_ids[0] == predicted_id)

                full_text = tokenizer.decode(input_ids[0].tolist())

                proof_data = {"status": "SKIPPED", "prove_time_ms": 0, "verify_time_ms": 0, "proof_size": 0, "ops_proved": 0}
                try:
                    prove_fn = cached.get('prove_fn', prove_llama_token_server)
                    result = prove_fn(
                        cached['server'], cached['extractor'], full_text,
                        prove_layers=llama_prove_layers,
                    )
                    proof_data = {
                        "status": "VALID" if result.get("valid") else "FAILED",
                        "prove_time_ms": round(result.get("prove_time_ms", 0), 1),
                        "verify_time_ms": round(result.get("verify_time_ms", 0), 1),
                        "proof_size": result.get("proof_size_bytes", 0),
                        "ops_proved": result.get("ops_proved", llama_prove_layers),
                    }
                    total_prove_ms += result.get("prove_time_ms", 0)
                    total_verify_ms += result.get("verify_time_ms", 0)
                    if not result.get("valid"):
                        all_valid = False
                except Exception as e:
                    logger.error("Llama proof error on token %d: %s", i, e)
                    proof_data = {"status": "ERROR", "prove_time_ms": 0, "verify_time_ms": 0, "proof_size": 0, "ops_proved": 0}
                    all_valid = False

                if not token_matches:
                    all_match = False

                yield f"data: {json.dumps({'type': 'token', 'index': i, 'claimed_token': claimed_token, 'predicted_token': predicted_token, 'predicted_token_id': predicted_id, 'matches': token_matches, 'proof': proof_data})}\n\n"

                if claimed_ids:
                    input_ids = torch.cat([input_ids, torch.tensor([claimed_ids[:1]])], dim=1)
                else:
                    input_ids = torch.cat([input_ids, torch.tensor([[predicted_id]])], dim=1)

        yield f"data: {json.dumps({'type': 'done', 'total_tokens': len(req.tokens), 'total_prove_time_ms': round(total_prove_ms, 1), 'total_verify_time_ms': round(total_verify_ms, 1), 'all_valid': all_valid, 'all_match': all_match})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Serve demo UI if it exists
demo_dir = Path(__file__).parent.parent / "demo"
if demo_dir.exists():
    app.mount("/demo", StaticFiles(directory=str(demo_dir), html=True), name="demo")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8042)
