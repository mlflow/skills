# Evaluation environment prerequisites

Read only when configuration is missing or connectivity fails. Reuse onboarding/backend setup already
completed; evaluation does not need a new setup flow.

## Preserve existing configuration

Check `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_ID` without exposing credentials. When both are set, **do
not run `setup_mlflow.py`, create `.env`, replace the backend, or create a new experiment**. Verify access and
use the existing configuration. If one is missing, preserve the other while resolving it. App configuration
can also hold the selected destination; inspect it before autodetection.

Run commands in the project environment (`uv run` for uv-managed projects). Check MLflow >=3.8 and the version
requirements of selected features. If installation is needed, use the project installer and quote version
specifiers, e.g. `uv pip install 'mlflow>=3.8.0'`.

## Fill only missing setup

`scripts/setup_mlflow.py --help` describes explicit URI and experiment options (bind a Databricks profile
through the URI). Inspect them and pass the intended destination; do not accept an arbitrary first
authenticated profile/experiment. The helper prints exports: apply them in the caller's environment, since a
child process cannot export into its parent.

For a custom server or manual configuration:

- Databricks: use `databricks` or the selected `databricks://<profile>` and its existing authentication.
  Validate the intended workspace/profile. If auth is absent, report the required `databricks auth login -p
  <profile>` step; do not start a separate Unity Gateway setup flow or switch to local storage.
- Local/remote OSS: use the selected HTTP(S) tracking server. Start a local server only when that backend is
  intended and no server exists; do not kill unrelated processes to free a port.
- SageMaker ARN: preserve the ARN and use available `mlflow-sagemaker` guidance; its plugin, region, AWS
  credentials, and connection checks are distinct from Databricks setup. Do not reinterpret the ARN as a local
  path.

Find the exact named experiment with `mlflow.get_experiment_by_name(name)` before creating one. Create only if
requested/needed in the intended destination. Databricks workspace experiment names commonly use
`/Users/<email>/<name>`; OSS names need no workspace prefix. Verify the returned experiment ID and preserve it
for subsequent commands.

Persist newly selected values through the application's existing configuration mechanism only when persistence
is part of the task. Avoid duplicate `.env` entries or overwriting caller-supplied variables. Import the app
and confirm it actually loads the intended destination.

## Conditional Unity Catalog Spark setup

Only if UC requires Spark and no notebook session exists, use compatible Databricks Connect with an authenticated
profile/cluster matching the tracking workspace; check builder support (serverless needs different configuration).

```python
import os
from databricks.connect import DatabricksSession
spark = (DatabricksSession.builder.profile(os.environ["DATABRICKS_CONFIG_PROFILE"])
         .clusterId(os.environ["DATABRICKS_CLUSTER_ID"]).getOrCreate())
spark.sql("SELECT 1").collect()
```

## Verify access, then runtime

```python
import os
import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
experiment = mlflow.get_experiment(os.environ["MLFLOW_EXPERIMENT_ID"])
assert experiment is not None, "Configured experiment was not found"
```

The bundled `scripts/validate_environment.py` and `scripts/validate_auth.py` provide diagnostics. Access to an
experiment does not prove trace-write permission or judge access; a present API key does not prove it works.
Distinguish missing package/plugin, invalid profile/credentials, network error, missing experiment, and access
denial. Stop on unresolved failures rather than overwriting configuration.

For live agent evaluation, use `mlflow-tracing` when available and reuse framework
autologging. Run the real application, flush traces, fetch the resulting trace in the intended experiment, and
inspect inputs/outputs and required spans. `scripts/validate_tracing_runtime.py --help` offers runtime
diagnostics; a non-null local trace ID alone is not persisted-trace verification. For precomputed evaluation,
verify the supplied evidence without invoking the agent.

Consult the exact [evaluation quickstart](https://mlflow.org/docs/latest/genai/eval-monitor/quickstart.md) or
selected backend documentation only as needed; use [llms.txt](https://mlflow.org/docs/latest/llms.txt) for
missing/moved pages. Check installed signatures. If documentation is unavailable, use this local minimum plus
introspection and stop when the required operation cannot be verified.
