---
name: querying-mlflow-metrics
description: Query aggregate MLflow trace, span and assessment metrics for latency, tokens, counts and quality trends. Use for operational summaries rather than individual-trace diagnosis.
---

# Query MLflow metrics

## Inputs

Reuse server, experiment IDs and requested window. Select the exact metric, view, aggregation and grouping needed; token counts are not currency cost.

## Workflow

1. Resolve `scripts/fetch_metrics.py` relative to this skill; use `--help` for flags. It accepts HTTP(S), not `databricks://` profiles, and adds no authentication headers. Use an existing supported authenticated client or report the blocker; never request pasted tokens.
2. Select TRACES (default), SPANS or ASSESSMENTS using [the API reference](references/api_reference.md). Use COUNT for entities, SUM for tokens, AVG/P95 for distributions. Do not silently substitute metrics.
3. Query explicit bounds: epoch milliseconds, timezone-qualified ISO timestamps, `now`, or relative values like `-24h`. Buckets are **seconds** and require both bounds.

   ```bash
   python <SKILL_DIR>/scripts/fetch_metrics.py -s <SERVER> -x <EXPERIMENT_ID> \
     -m total_tokens -a SUM -t 3600 --start-time="-24h" --end-time=now -o json
   ```

   Latency: `-m latency -a AVG,P95 -d trace_name`. Numeric quality: `-v ASSESSMENTS -m assessment_value -a AVG -d assessment_name`. Status counts: `-m trace_count -a COUNT -d trace_status`. Keep the same explicit window.

## Verify

Inspect actual `metric_name`, dimensions and values; report exact experiment/window, metric, aggregation, units, filters and coverage. Empty `data_points` is not zero usage; absent token instrumentation can undercount. Quality summaries need per-scorer scales and valid/error/missing denominators: zero/false can be valid, scorer failures are not zero scores, and unrelated scorers should not be averaged.

A `next_page_token` means partial results; the helper exposes it but cannot fetch subsequent pages. Use an existing pagination-capable client or disclose incompleteness. Counts by status are not an error rate: derive ERROR / all statuses only with a complete stated denominator. Distinguish request/auth failures from empty results. Aggregates need representative trace evidence for causal diagnosis.

## Read when needed

[API reference](references/api_reference.md) supplies wire fields, metric/view compatibility and metrics-specific filters. Check backend/client support when the helper is insufficient.
