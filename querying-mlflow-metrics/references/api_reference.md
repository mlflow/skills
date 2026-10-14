# Trace metrics wire reference

The helper posts JSON to `POST /api/3.0/mlflow/traces/metrics`. Check server/version compatibility.

## Views, metrics and dimensions

| View (wire code) | Metric | Typical aggregation | Dimensions |
|---|---|---|---|
| TRACES (1) | `trace_count` | COUNT | `trace_name`, `trace_status` |
| TRACES (1) | `latency` (ms) | AVG, PERCENTILE | same |
| TRACES (1) | `input_tokens`, `output_tokens`, `total_tokens` | SUM, AVG, PERCENTILE | same |
| SPANS (2) | `span_count` | COUNT | `span_name`, `span_type`, `span_status` |
| SPANS (2) | `latency` (ms) | AVG, PERCENTILE | same |
| ASSESSMENTS (3) | `assessment_count` | COUNT | `assessment_name`, `assessment_value` |
| ASSESSMENTS (3) | `assessment_value` | AVG, PERCENTILE for numeric scores | same |

Aggregation codes: COUNT=1, SUM=2, AVG=3, PERCENTILE=4, MIN=5, MAX=6. PERCENTILE requires `percentile_value` (0–100). Helper shorthand `P50`, `P95`, `P99.9` supplies that value; a bare `PERCENTILE` does not. Helper parsing does not validate backend metric/aggregation compatibility.

## Filters and time

Metrics filters use their own grammar, which differs from trace search:

```text
trace.status = "OK"
trace.tag.<key> = "<value>"
trace.metadata.<key> = "<value>"
span.name = "<value>"
span.type = "LLM"
assessment.name = "<value>"
```

`-f` splits on commas; for values containing commas use a client that sends a filter array. `start_time_ms` / `end_time_ms` are epoch milliseconds, while `time_interval_seconds` is seconds (3600 for hourly buckets). Buckets require both bounds; use timezone-aware timestamps.

## Request and response

```json
{
  "experiment_ids": ["1"], "view_type": 1, "metric_name": "total_tokens",
  "aggregations": [{"aggregation_type": 2}],
  "dimensions": ["trace_name"], "filters": ["trace.status = 'OK'"],
  "start_time_ms": 1737244800000, "end_time_ms": 1737331200000,
  "time_interval_seconds": 3600, "max_results": 1000
}
```

The response has `data_points` with `metric_name`, `dimensions` (including `time_bucket` when requested), and `values` keyed by aggregation, plus an optional `next_page_token`. Empty points are not measured zeros. A continuation token means incomplete coverage; `fetch_metrics.py` exposes the token but has no continuation argument. HTTP timeout is 30 seconds; no credentials are attached.
