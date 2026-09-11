# Tracing on Databricks (Unity Catalog storage)

On Databricks, store traces in **Unity Catalog Delta tables** by default for governed, production-grade storage. Bind an MLflow experiment to a UC trace location, then instrument code as usual — all traces logged to that experiment land in those tables.

Do not treat `mlflow.set_tracking_uri("databricks")` as sufficient Databricks setup. Without a `UnityCatalog` trace location, traces use legacy workspace experiment storage. Only use that legacy destination when the user explicitly requests it.

Before editing code, look for existing values in application config, environment examples, deployment manifests, or Databricks Asset Bundles. The required values are:

- Catalog name
- Schema name
- Table prefix
- SQL warehouse ID

If they are not available, ask the user for them. Do not invent a production destination or silently omit the UC trace location.

```python
import os
import mlflow
from mlflow.entities.trace_location import UnityCatalog

mlflow.set_tracking_uri("databricks")
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = "<SQL_WAREHOUSE_ID>"

mlflow.set_experiment(
    experiment_name="<MLFLOW_EXPERIMENT_NAME>",
    trace_location=UnityCatalog(
        catalog_name="<UC_CATALOG_NAME>",
        schema_name="<UC_SCHEMA_NAME>",
        table_prefix="<UC_TABLE_PREFIX>",
    ),
)
```

**`table_prefix`** is the prefix applied to every table storing trace data. MLflow creates four Delta tables from it: `<table_prefix>_otel_spans`, `<table_prefix>_otel_logs`, `<table_prefix>_otel_metrics`, and `<table_prefix>_otel_annotations`.

**Notes**:
- Experiment names on Databricks must be absolute workspace paths (`/Users/<email>/name` or `/Shared/name`). A bare name is rejected. To attach to an existing experiment, use `mlflow.set_experiment(experiment_id="<numeric-id>")`.
- Requires a SQL warehouse (`MLFLOW_TRACING_SQL_WAREHOUSE_ID`) to provision and query the tables.
- A UC trace location is permanent — once bound, an experiment cannot be reassigned to a different UC location.
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

Docs: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog
