---
name: sagemaker-mlflow
description: Connect to SageMaker Managed MLflow using an mlflow-app or mlflow-tracking-server ARN, validate prerequisites and connectivity, then continue the requested MLflow workflow.
---

# Connect to SageMaker MLflow

## Inputs

Use the supplied ARN or existing `MLFLOW_TRACKING_URI`, AWS region and credential chain.
Both `arn:aws:sagemaker:...:mlflow-app/...` and
`arn:aws:sagemaker:...:mlflow-tracking-server/...` are supported.

## Workflow

1. Check **Python >= 3.10**, **MLflow >= 3.8**, the `sagemaker-mlflow` plugin and resolvable
   AWS credentials (environment, SSO or IAM role). Validate only: do not auto-install
   packages or acquire credentials. If the plugin is missing, give the user
   `pip install sagemaker-mlflow`; if credentials are missing/expired, report that blocker.
2. Reuse the supplied ARN. Only when none is available, run
   `python scripts/discover_arns.py` from this skill's directory in the selected region.
   Select a sole result; for multiple resources, show name/status/ARN and ask which to use.
3. Set `MLFLOW_TRACKING_URI` to the chosen ARN and `AWS_DEFAULT_REGION` to its region.
   The plugin handles SigV4 authentication through the default AWS credential chain;
   never print credentials. Preserve unrelated configuration.

## Verify

Run from this skill's directory:

```bash
python scripts/verify_connection.py "$MLFLOW_TRACKING_URI"
```

It checks plugin, credentials and ARN connectivity, exiting nonzero at the first failure.
Do not proceed until it passes. `UnsupportedModelRegistryStoreURIException` usually
indicates the missing plugin; `AccessDenied`/403 requires checking credential validity,
resource permissions (`sagemaker-mlflow`/`sagemaker`) and ARN region consistency.
Report the observed blocker and remediation rather than claiming a connection.

After successful verification, continue the requested tracing, evaluation, retrieval or
metrics workflow using the same `MLFLOW_TRACKING_URI`; do not repeat setup.
