# Tracing on Databricks (Unity Catalog storage)

On Databricks, store traces in **Unity Catalog Delta tables** by default for governed, production-grade storage. Bind an MLflow experiment to a UC trace location, then instrument code as usual — all traces logged to that experiment land in those tables.

Do not treat `mlflow.set_tracking_uri("databricks")` as sufficient Databricks setup. Without a `UnityCatalog` trace location, traces use legacy workspace experiment storage. Only use that legacy destination when the user explicitly requests it.

This API requires MLflow 3.11.1 or newer. Before editing code, inspect the installed version and look for existing values in application config, environment examples, deployment manifests, or Databricks Asset Bundles.

Required values:

- Catalog name
- Schema name
- SQL warehouse ID

Optional value for Python:

- Table prefix. Omit it to let MLflow use the experiment ID, which avoids naming collisions.

If required values are not available, ask the user for them. Do not invent a production destination or silently omit the UC trace location.

```python
import os
import mlflow
from mlflow.entities.trace_location import UnityCatalog

# Use "databricks" for DEFAULT auth, or "databricks://<profile>" for a named profile.
mlflow.set_tracking_uri("databricks://<DATABRICKS_PROFILE>")
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = "<SQL_WAREHOUSE_ID>"

mlflow.set_experiment(
    experiment_name="<MLFLOW_EXPERIMENT_NAME>",
    trace_location=UnityCatalog(
        catalog_name="<UC_CATALOG_NAME>",
        schema_name="<UC_SCHEMA_NAME>",
    ),
)
```

Add `table_prefix="<UC_TABLE_PREFIX>"` only when the user wants a stable custom prefix. Otherwise MLflow uses the experiment ID.

## TypeScript

Before implementing UC tracing in TypeScript, inspect the installed `mlflow-tracing` typings for an `init()` option named `traceLocation`. Published versions without that option cannot write to a UC trace location. Do not configure only `trackingUri` and `experimentId` and claim that traces use UC; that is legacy experiment storage. Explain the limitation and ask whether the user wants to upgrade when a supporting version is available or use Python instrumentation.

When `traceLocation` is supported, the location must already be provisioned. Provision and link it once with the Python API above, then configure TypeScript with the exact resulting table prefix:

```typescript
import * as mlflow from "mlflow-tracing";

mlflow.init({
  trackingUri: "databricks://<DATABRICKS_PROFILE>",
  experimentId: "<EXPERIMENT_ID>",
  traceLocation: {
    catalogName: "<UC_CATALOG_NAME>",
    schemaName: "<UC_SCHEMA_NAME>",
    tablePrefix: "<UC_TABLE_PREFIX>",
  },
});
```

Unlike Python, TypeScript cannot provision the location or default its table prefix. All three `traceLocation` fields are required.

**`table_prefix`** is the prefix applied to every table storing trace data. MLflow creates four Delta tables from it: `<table_prefix>_otel_spans`, `<table_prefix>_otel_logs`, `<table_prefix>_otel_metrics`, and `<table_prefix>_otel_annotations`.

**Notes**:

- Requires `mlflow>=3.11.1`. Upgrade first if `UnityCatalog` or the `trace_location` argument is unavailable.
- Experiment names on Databricks must be absolute workspace paths (`/Users/<email>/name` or `/Shared/name`). A bare name is rejected. To attach to an existing experiment, use `mlflow.set_experiment(experiment_id="<numeric-id>")`.
- Requires a SQL warehouse (`MLFLOW_TRACING_SQL_WAREHOUSE_ID`) to provision and query the tables.
- Confirm the principal has `USE CATALOG`, `USE SCHEMA`, and `CREATE TABLE` on the destination plus permission to use the SQL warehouse before provisioning.
- A UC trace location is permanent — once bound, an experiment cannot be reassigned to a different UC location.
- Before reusing an existing experiment, inspect `experiment.trace_location`. Bind it when it is unbound and reuse it when it matches the requested UC destination. If it points elsewhere, explain the conflict and use a new experiment name or ask the user which destination to keep; do not catch the error and continue with legacy storage.
- Binding an existing experiment changes the destination for new traces; it does not migrate traces already stored in the legacy experiment backend.
- To create the experiment explicitly, use `mlflow.create_experiment(name=..., trace_location=UnityCatalog(...))`, then `mlflow.set_experiment(experiment_id=...)`.

## Verification

After generating a trace, confirm both the trace and its storage destination. A trace appearing in the Databricks experiment UI proves export succeeded, but does not by itself prove UC storage was configured.

```python
import mlflow
from mlflow.entities.trace_location import UnityCatalog

mlflow.flush_trace_async_logging()
experiment = mlflow.get_experiment_by_name("<MLFLOW_EXPERIMENT_NAME>")
assert experiment is not None
print(experiment.trace_location)
assert isinstance(experiment.trace_location, UnityCatalog)

traces = mlflow.search_traces(locations=[experiment.experiment_id])
assert len(traces) > 0
```

## Link to the verified trace

After verification, give the user a clickable link to a trace from the run you just verified. Prefer this direct route instead of guessing trace-drawer query parameters:

```text
https://<workspace-host>/ml/experiments/<experiment-id>/traces/<trace-id>
```

- Use the workspace host from the authenticated Databricks profile or application configuration. `databricks` and `databricks://<profile>` are tracking URIs, not browser URLs. If the host or experiment ID cannot be discovered, ask the user instead of guessing a link.
- Use the numeric experiment ID configured for the application and linked to its UC trace location, not the experiment name or a catalog/schema/table name. Do not rely on `trace.info.experiment_id` for UC traces: it can be `None`.
- For a UC trace ID such as `trace:/catalog.schema/0123456789abcdef0123456789abcdef`, use only the final `0123456789abcdef0123456789abcdef` component in the browser path. Keep the full qualified ID for SDK/API calls. For legacy workspace trace IDs such as `tr-0123456789abcdef0123456789abcdef`, keep the `tr-` prefix.
- Append `?o=<workspace-id>` after the complete trace path if it is present in an existing workspace URL; do not invent a workspace ID.
- Do not use the OSS MLflow `#/experiments/...` route, a run URL, or a REST API endpoint as a Databricks trace link. A trace URL does not prove UC storage was configured; keep the storage verification above.

The direct trace URL format is documented in the [Databricks tracing FAQ](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/faq).

Docs: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog
