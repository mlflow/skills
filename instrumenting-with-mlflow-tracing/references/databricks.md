# Tracing on Databricks (Unity Catalog storage)

On Databricks, traces can be stored in **Unity Catalog Delta tables** for governed, production-grade storage. Bind an MLflow experiment to a UC trace location, then instrument code as usual — all traces logged to that experiment land in those tables.

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
- Requires a SQL warehouse (`MLFLOW_TRACING_SQL_WAREHOUSE_ID`) to provision and query the tables.
- A UC trace location is permanent — once bound, an experiment cannot be reassigned to a different UC location.
- To create the experiment explicitly, use `mlflow.create_experiment(name=..., trace_location=UnityCatalog(...))`, then `mlflow.set_experiment(experiment_id=...)`.

Docs: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog
