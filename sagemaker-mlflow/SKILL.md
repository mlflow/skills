---
name: sagemaker-mlflow
description: >
  Connect to SageMaker Managed MLflow (mlflow-app or mlflow-tracking-server ARN) as an
  MLflow backend, then hand off to the other MLflow skills. Triggers on a SageMaker MLflow
  ARN (arn:aws:sagemaker:...:mlflow-app/... or arn:aws:sagemaker:...:mlflow-tracking-server/...)
  or "SageMaker Managed MLflow".
---

# Connect to SageMaker MLflow

Establishes the connection to SageMaker Managed MLflow (the `sagemaker-mlflow` plugin + an
ARN tracking URI + AWS credentials), then hands off to the requested MLflow skill
(instrumenting-with-mlflow-tracing, agent-evaluation, retrieving-mlflow-traces,
querying-mlflow-metrics, etc.), which inherit the connection via `MLFLOW_TRACKING_URI`.

Requires **Python >= 3.10** (mlflow >= 3.8). Validate and report the fix — never
auto-install the plugin or acquire credentials (mirrors the repo's Databricks pattern).

## Step 1: Preconditions
- `python --version` -> must be 3.10+. If older, STOP and tell the user.
- Confirm AWS credentials resolve (env vars, `aws configure` / SSO, or an IAM role). If
  they error (ExpiredToken / no creds), STOP and ask the user to configure them.
  (`scripts/verify_connection.py` also checks this, via boto3.)

## Step 2: Install the plugin
```bash
pip install sagemaker-mlflow
```
Required even if `MLFLOW_TRACKING_URI` is already an ARN — without it,
`mlflow.set_tracking_uri()` fails with `UnsupportedModelRegistryStoreURIException`.

## Step 3: Select the resource ARN
- **If the user already provided an ARN** (or `MLFLOW_TRACKING_URI` is set to
  `arn:aws:sagemaker:...`), use it as-is.
- **Otherwise, discover it** in the user's region:
  ```bash
  python scripts/discover_arns.py
  ```
  If exactly one is returned, use it; if several, list name/status/ARN and ask the user
  which to use. Both `mlflow-app` and `mlflow-tracking-server` ARNs work; the ARN is also
  shown on the resource's detail page in the SageMaker Studio console.

Then export it (the ARN is region-bound — a cross-region ARN will not authenticate):
```bash
export MLFLOW_TRACKING_URI="<arn>"    # mlflow-app/<id> or mlflow-tracking-server/<name>
export AWS_DEFAULT_REGION="<region>"  # AWS credentials resolved via the default chain (SigV4)
```

## Step 4: Verify
```bash
python scripts/verify_connection.py "$MLFLOW_TRACKING_URI"
```
Checks all three prerequisites (plugin / credentials / ARN connectivity) and exits
non-zero with a remediation hint on the first failure. Do not proceed until it passes.

After the connection verifies, hand off to the requested MLflow skill — it inherits the
connected environment via `MLFLOW_TRACKING_URI`.

## Troubleshooting
- `UnsupportedModelRegistryStoreURIException` — the `sagemaker-mlflow` plugin is not
  installed; `pip install sagemaker-mlflow`, then re-run `scripts/verify_connection.py`.
- `AccessDenied` / `403` — credentials are missing/expired, or the principal lacks MLflow
  permissions on the ARN. Check that credentials resolve, grant `sagemaker-mlflow` /
  `sagemaker` permissions, and confirm the ARN region matches `AWS_DEFAULT_REGION`.
