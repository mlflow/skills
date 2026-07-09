#!/usr/bin/env python3
"""List SageMaker Managed MLflow resources (ARNs) in a region.

Usage:
    python discover_arns.py [<region>]

Region defaults to $AWS_DEFAULT_REGION. Prints the ARN, name, and status for both serverless
mlflow-apps and managed mlflow-tracking-servers so you can pick the tracking URI.
"""
import os
import sys


def main() -> None:
    region = sys.argv[1] if len(sys.argv) > 1 else os.getenv("AWS_DEFAULT_REGION")
    if not region:
        print("Provide a region (argument or AWS_DEFAULT_REGION).")
        sys.exit(2)

    try:
        import boto3
    except ImportError:
        print("boto3 not installed  ->  pip install sagemaker-mlflow")
        sys.exit(1)

    sm = boto3.client("sagemaker", region_name=region)
    found = 0

    try:
        for a in sm.list_mlflow_apps().get("Summaries", []):
            print(f"app              {a['Arn']}  ({a.get('Name')}, {a.get('Status')})")
            found += 1
    except Exception as e:  # noqa: BLE001
        print(f"(list_mlflow_apps unavailable: {str(e)[:80]})")

    try:
        for t in sm.list_mlflow_tracking_servers().get("TrackingServerSummaries", []):
            print(
                f"tracking-server  {t['TrackingServerArn']}  "
                f"({t.get('TrackingServerName')}, {t.get('TrackingServerStatus')})"
            )
            found += 1
    except Exception as e:  # noqa: BLE001
        print(f"(list_mlflow_tracking_servers unavailable: {str(e)[:80]})")

    if not found:
        print(f"No SageMaker MLflow resources found in {region}. Create one, or check the region.")


if __name__ == "__main__":
    main()
