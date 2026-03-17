"""ZK-ML-Audit: Lightweight verification for ZK-Compile proofs."""

from zk_ml_audit.verify import (
    check_coverage,
    check_weights,
    check_input,
    verify_proof,
    audit_full,
)

__version__ = "0.1.0"
__all__ = ["check_coverage", "check_weights", "check_input", "verify_proof", "audit_full"]
