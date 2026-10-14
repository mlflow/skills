---
name: instrumenting-with-mlflow-tracing
description: Add or repair MLflow tracing in Python or TypeScript applications, including tracing required for evaluation. Covers framework autologging, custom spans, backend setup, and persisted trace verification.
---

# Instrument MLflow tracing

Use when adding or repairing instrumentation. Reuse working tracing when the task is only investigation or evaluation.

## Inputs

Inspect the application entry point, language/framework, installed SDK version, tracking URI, experiment, and deployment identity. Preserve configured destinations and provider choices. Read [Python](references/python.md) or [TypeScript](references/typescript.md) for the selected runtime.

## Workflow

1. For Databricks, read [the UC setup reference](references/databricks.md) **before editing**: validate auth/workspace, reuse an explicitly chosen destination, otherwise resolve schema choice before provisioning. Discover an accessible warehouse automatically. UC is the default; legacy storage requires an explicit user request.
2. Use the matching framework integration before execution. LangGraph uses `mlflow.langchain.autolog()`, not `mlflow.langgraph.autolog()`. Do not add redundant node, tool, or model wrappers to recreate its automatic spans. Manual spans are for uncaptured application work or a deliberate outer application boundary.
3. Capture meaningful root, LLM, retrieval, tool, and routing operations with their actual relevant inputs/outputs. Avoid utility-function noise. Manual spans need explicit inputs/outputs; respect redaction requirements before export.
4. Run one representative application request. Record start/end epoch milliseconds and save its trace ID immediately in the same process (`mlflow.get_last_active_trace_id()` in Python; capture the active span's ID in TypeScript). Do not rerun just to recover an ID.

<a id="verification"></a>

## Verify

Fetch an **actually persisted application trace**, not merely a local span or an old trace in the experiment. Enforce a **60-second wall-clock readback deadline**, including flush, SDK retries, requests, and backoff. Arrange a process deadline before starting; terminate the verification process at expiry. A tool yield or thread-wait timeout does not cancel the call. If cancellation is unavailable, report backend verification as unperformed.

Flush once (`mlflow.flush_trace_async_logging()` / `await mlflow.flushTraces()`), then fetch the saved ID. If absent, make one search in the configured experiment, bounded to the recorded run timestamps, with `max_results=1` and spans included. Every UC search needs a `trace.timestamp_ms` predicate; a result limit is not a scan bound. Python `search_traces()` has no `start_time` keyword.

For Python readback, adapt this recipe to the saved run values and enforce the deadline externally:

```python
mlflow.flush_trace_async_logging()
if trace_id:
    trace = mlflow.get_trace(trace_id)
else:
    traces = mlflow.search_traces(
        locations=[experiment_id],
        filter_string=(f"trace.timestamp_ms >= {run_start_ms} "
                       f"AND trace.timestamp_ms <= {run_end_ms}"),
        max_results=1, return_type="list", include_spans=True,
    )
    trace = traces[0] if traces else None
assert trace is not None, "No persisted trace read back"
assert run_start_ms <= trace.info.request_time <= run_end_ms
assert trace.data.spans, "No spans captured"
```

Inspect the returned expected root/graph, model, retrieval and tool spans and their actual inputs/outputs. Confirm they belong to this application's request, not a scorer. Reuse the returned spans and stop after success. An available `agent-evaluation/scripts/validate_tracing_runtime.py` may replace this check under the same deadline; it is not an additional audit.

Allow at most one retry after a concrete quick fix **within the original budget**. Stop on missing provider keys, auth/permission errors, warehouse timeouts, or `RESOURCE_EXHAUSTED`/429. Do not launch parallel searches, repeated agent runs, direct UC SQL, raw REST, alternate credentials or warehouse fallbacks to force verification. Report the failing operation/error, captured ID, what succeeded, and what remains unverified; request the specific missing prerequisite without asking for secrets. Pause evaluation if tracing remains unverified.

Return the verified trace ID/link and observed spans. For Databricks use the [verified trace URL rules](references/databricks.md#link-to-the-verified-trace), not an experiment-only link. Do not label an unverified link as proof.

## Read when needed

- Deployment, async export, sampling: [production](references/production.md).
- Async/thread context, redaction, caught errors or links: [advanced patterns](references/advanced-patterns.md).
- Cross-service parent propagation: [distributed tracing](references/distributed-tracing.md).
- User ratings tied to traces: [feedback collection](references/feedback-collection.md).
- Unreadable non-chat root previews: [root span previews](references/root-span-previews.md).
- Explaining captured behavior: `debug-agent`; implementing a requested behavior fix: `fix-agent-issue`, carrying the trace evidence. These are optional follow-on tasks.

For other integrations, use the exact [automatic tracing docs](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/automatic.md). Check installed signatures/types against examples. Use [llms.txt](https://mlflow.org/docs/latest/llms.txt) only for a missing/moved topic; if docs are unavailable, use local recipes and installed API introspection, and stop if correctness cannot be established.
