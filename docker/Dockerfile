# ZK-ML Proof Server — Multi-stage Docker build
#
# Builds the Rust structured sumcheck prover (CPU-only, no Metal GPU)
# and packages it with the FastAPI server + PyTorch inference.
#
# Build context must be the repository root:
#   docker build -f docker/Dockerfile -t zk-ml-server .
#   OR: docker compose -f docker/docker-compose.yml up --build

# ============================================================
# Stage 1: Build Rust prover (CPU-only, no Metal)
# ============================================================
FROM rust:1.78-slim AS rust-builder

WORKDIR /build

# Copy only what's needed for the Rust build
COPY rust/zk_ml_prover/ rust/zk_ml_prover/

WORKDIR /build/rust/zk_ml_prover

# Override target-cpu=native from .cargo/config.toml — we need a portable
# binary that runs on any x86_64 (or aarch64) Linux, not just M4 Max.
ENV RUSTFLAGS="-C target-cpu=generic"

# Build without metal_gpu feature (Linux has no Metal framework).
# The build.rs Metal shader compilation is gated behind
# CARGO_FEATURE_METAL_GPU, so it is automatically skipped.
RUN cargo build --release

# ============================================================
# Stage 2: Python runtime with FastAPI server
# ============================================================
FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
COPY docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Place the Rust binary where the Python code expects it.
# All pipeline files resolve the binary relative to __file__:
#   python/tvm_to_gkr/<file>.py -> ../../rust/zk_ml_prover/target/release/zk_ml_prover
# So with Python at /app/python/tvm_to_gkr/, the binary must be at:
#   /app/rust/zk_ml_prover/target/release/zk_ml_prover
RUN mkdir -p /app/rust/zk_ml_prover/target/release
COPY --from=rust-builder /build/rust/zk_ml_prover/target/release/zk_ml_prover \
     /app/rust/zk_ml_prover/target/release/zk_ml_prover

# Also create symlinks for PCS variants (same binary, different names)
# so that pcs_mode="pcs" and "pcs-full" work in the server
RUN ln -s /app/rust/zk_ml_prover/target/release/zk_ml_prover \
          /app/rust/zk_ml_prover/target/release/zk_ml_prover_pcs && \
    ln -s /app/rust/zk_ml_prover/target/release/zk_ml_prover \
          /app/rust/zk_ml_prover/target/release/zk_ml_prover_pcs_full

# Copy Python source
COPY python/ /app/python/

# Copy demo UI (served at /demo by FastAPI if present)
COPY demo/ /app/demo/

# Environment
ENV PYTHONPATH=/app/python
ENV PYTHONUNBUFFERED=1

# HuggingFace cache location (mountable as a volume)
ENV HF_HOME=/models

EXPOSE 8042

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8042/health')" || exit 1

ENTRYPOINT ["python", "/app/python/serve.py"]
