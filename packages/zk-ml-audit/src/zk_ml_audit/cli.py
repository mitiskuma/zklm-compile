"""CLI for ZK-ML proof auditing."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="zk-audit",
        description="Verify and audit ZK-Compile ML inference proofs",
    )
    parser.add_argument("proof", help="Path to proof JSON file")
    parser.add_argument("--model", help="HuggingFace model name (for weight verification)")
    parser.add_argument("--input", help="Input text (for input commitment verification)")
    parser.add_argument("--endpoint", help="Server endpoint URL for remote verification")
    parser.add_argument("--prove-layers", type=int, default=12, help="Layers to prove (default: 12)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Load proof
    with open(args.proof) as f:
        proof_data = json.load(f)

    from zk_ml_audit.verify import audit_full

    report = audit_full(
        proof_data,
        model_name=args.model,
        input_text=args.input,
        proof_path=args.proof,
        prove_layers=args.prove_layers,
        verbose=args.verbose,
    )

    # Print results
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.name}: {check.message}")
        if args.verbose and check.details:
            for detail in check.details:
                print(f"         {detail}")

    print(f"\nOverall: {'ALL PASSED' if report.all_passed else 'SOME FAILED'}")
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
