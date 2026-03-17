"""CLI entry point for zk-compile.

Provides a user-friendly wrapper around the zk_compile_prover binary.

Usage:
    zk-compile prove circuit.json [-o proof.bin]
    zk-compile verify circuit.json [proof.bin]
    zk-compile prove-and-verify circuit.json
"""

import argparse
import sys

from zk_compile.prover import prove, verify, prove_and_verify


def main():
    parser = argparse.ArgumentParser(
        prog="zk-compile",
        description="ZK-Compile: Generate and verify zero-knowledge proofs for ML circuits",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prove
    p_prove = subparsers.add_parser("prove", help="Generate a Groth16 proof")
    p_prove.add_argument("circuit", help="Path to circuit JSON file")
    p_prove.add_argument(
        "--output", "-o", default="proof.bin",
        help="Output path for proof file (default: proof.bin)",
    )

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify a proof")
    p_verify.add_argument("circuit", help="Path to circuit JSON file")
    p_verify.add_argument(
        "proof", nargs="?", default="proof.bin",
        help="Path to proof file (default: proof.bin)",
    )

    # prove-and-verify
    p_pv = subparsers.add_parser(
        "prove-and-verify", help="Generate and verify in one step",
    )
    p_pv.add_argument("circuit", help="Path to circuit JSON file")

    args = parser.parse_args()

    try:
        if args.command == "prove":
            result = prove(args.circuit, args.output)
            print(
                f"Setup: {result.setup_time_s:.3f}s | "
                f"Prove: {result.prove_time_s:.3f}s"
            )
            print(f"Proof: {result.proof_path}")
            print(f"Verifying key: {result.vk_path}")

        elif args.command == "verify":
            valid = verify(args.circuit, args.proof)
            status = "VALID" if valid else "INVALID"
            print(f"Verification: {status}")
            sys.exit(0 if valid else 1)

        elif args.command == "prove-and-verify":
            result = prove_and_verify(args.circuit)
            parts = [
                f"Setup: {result.setup_time_s:.3f}s",
                f"Prove: {result.prove_time_s:.3f}s",
                f"Verify: {result.verify_time_s:.4f}s",
            ]
            if result.num_constraints is not None:
                parts.append(f"Constraints: {result.num_constraints}")
            if result.proof_size_bytes is not None:
                parts.append(f"Proof: {result.proof_size_bytes} bytes")
            parts.append("VALID" if result.valid else "INVALID")
            print(" | ".join(parts))
            sys.exit(0 if result.valid else 1)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(127)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
