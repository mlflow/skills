#!/usr/bin/env python3
"""Verify SageMaker Managed MLflow prerequisites: plugin, AWS credentials, ARN connectivity.

Usage:
    python verify_connection.py [<arn>]

The ARN defaults to $MLFLOW_TRACKING_URI. This script validates only -- it never installs the
plugin and never acquires credentials. It exits non-zero with a remediation hint on the first
failing prerequisite.
"""
import os
import sys


def main() -> None:
    arn = sys.argv[1] if len(sys.argv) > 1 else os.getenv("MLFLOW_TRACKING_URI", "")
    if not arn.startswith("arn:aws:sagemaker:"):
        print(
            "Provide a SageMaker MLflow ARN (argument or MLFLOW_TRACKING_URI):\n"
            "  arn:aws:sagemaker:<region>:<account>:mlflow-app/<id>  or\n"
            "  arn:aws:sagemaker:<region>:<account>:mlflow-tracking-server/<name>"
        )
        sys.exit(2)

    # 1. Plugin installed (registers the `arn` tracking scheme; it brings boto3 with it).
    try:
        import sagemaker_mlflow  # noqa: F401
        import boto3

        print("[ok] sagemaker-mlflow plugin installed")
    except ImportError:
        print("[x] sagemaker-mlflow plugin (or its boto3 dependency) not installed  ->  pip install sagemaker-mlflow")
        sys.exit(1)

    # 2. AWS credentials present (validate only, never acquire).
    try:
        identity = boto3.client("sts").get_caller_identity()
        print(f"[ok] AWS credentials present ({identity['Arn']})")
    except Exception as e:  # noqa: BLE001
        print(f"[x] AWS credentials missing or expired: {str(e)[:100]}")
        print("    Configure via env vars, `aws configure`/SSO, or an IAM role, then retry.")
        sys.exit(1)

    # 3. ARN tracking URI connects.
    try:
        import mlflow

        mlflow.set_tracking_uri(arn)
        mlflow.search_experiments(max_results=1)
        print(f"[ok] Connected to {arn}")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "403" in msg or "AccessDenied" in msg:
            print(
                "[x] AccessDenied/403  ->  the AWS principal needs sagemaker-mlflow / sagemaker "
                "permissions on the resource ARN"
            )
        else:
            print(f"[x] Cannot connect to {arn}: {msg[:120]}")
        sys.exit(1)

    print("All SageMaker Managed MLflow prerequisites satisfied.")


if __name__ == "__main__":
    main()
