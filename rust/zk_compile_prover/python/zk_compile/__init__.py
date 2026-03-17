"""ZK-Compile: Zero-knowledge proof compiler for ML models."""

from zk_compile.prover import prove, verify, prove_and_verify

__version__ = "0.1.0"
__all__ = ["prove", "verify", "prove_and_verify"]
