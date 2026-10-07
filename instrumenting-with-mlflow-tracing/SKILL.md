---
name: instrumenting-with-mlflow-tracing
description: Instruments Python and TypeScript code with MLflow Tracing for observability. Must be loaded when setting up tracing as part of any workflow including agent evaluation. Triggers on adding tracing, instrumenting agents/LLM apps, getting started with MLflow tracing, tracing specific frameworks (LangGraph, LangChain, OpenAI, Gemini, DSPy, CrewAI, AutoGen), or when another skill references tracing setup. Examples - "How do I add tracing?", "Instrument my agent", "Trace my LangChain app", "Set up tracing for evaluation"
---

# MLflow Tracing Instrumentation Guide

## Language-Specific Guides

Based on the user's project, load the appropriate guide:

- **Python projects**: Read `references/python.md`
- **TypeScript/JavaScript projects**: Read `references/typescript.md`

If unclear, check for `package.json` (TypeScript) or `requirements.txt`/`pyproject.toml` (Python) in the project.

---

## Databricks: verify auth and use Unity Catalog trace storage by default

When the target is Databricks, **read `references/databricks.md` before editing code** and configure a `UnityCatalog` trace location. Calling only `mlflow.set_tracking_uri("databricks")` and `mlflow.set_experiment(...)` without a trace location uses legacy workspace experiment storage; that does not satisfy a request to send traces to Databricks.

Inspect existing project or environment configuration for candidate destinations and the optional table prefix. Before binding an experiment or provisioning UC resources, follow the schema-selection workflow in `references/databricks.md`: ask the user to choose an existing schema or create a new one unless they have already explicitly chosen the destination. Never select an arbitrary accessible schema. Discover SQL warehouses through the authenticated workspace and select any warehouse the user can access; do not ask the user to choose among usable warehouses. Ask for a warehouse only when no usable candidate can be discovered. Do not silently fall back to legacy workspace trace storage. Use legacy storage only when the user explicitly requests it.

Verify auth and the target workspace before the first run. An expired token, or a default profile pointed at the wrong workspace, drops traces silently at export with no error raised.

```bash
databricks current-user me --profile <name>   # fails if auth is expired, without printing a token
python -c "import mlflow; print(mlflow.get_tracking_uri())"   # confirm databricks or databricks://<name>
```

If auth is expired, run `databricks auth login --profile <name>`. Never print or persist the output of `databricks auth token` in an agent transcript.

---

## What to Trace

**Trace these operations** (high debugging/observability value):

| Operation Type | Examples | Why Trace |
|---------------|----------|-----------|
| **Root operations** | Main entry points, top-level pipelines, workflow steps | End-to-end latency, input/output logging |
| **LLM calls** | Chat completions, embeddings | Token usage, latency, prompt/response inspection |
| **Retrieval** | Vector DB queries, document fetches, search | Relevance debugging, retrieval quality |
| **Tool/function calls** | API calls, database queries, web search | External dependency monitoring, error tracking |
| **Agent decisions** | Routing, planning, tool selection | Understand agent reasoning and choices |
| **External services** | HTTP APIs, file I/O, message queues | Dependency failures, timeout tracking |

**Skip tracing these** (too granular, adds noise):

- Simple data transformations (dict/list manipulation)
- String formatting, parsing, validation
- Configuration loading, environment setup
- Logging or metric emission
- Pure utility functions (math, sorting, filtering)

**Rule of thumb**: Trace operations that are important for debugging and identifying issues in your application.

---

## Verification

After instrumenting the code, run one representative example and attempt a quick verification. **Spend at most 60 seconds of wall-clock time on trace readback**, including flushing, requests, internal SDK retries, and backoff, unless the user asks for deeper troubleshooting.

Before readback starts, arrange a terminal/process deadline. If your tool only yields a running session, track elapsed time and cancel or terminate that verification session at the deadline, even if the MLflow call has not returned. Stop polling and hand off; do not reset the budget for a retry or fallback. A yield timeout or a timeout on waiting for a thread does not stop the underlying call. If cancellation is unavailable, skip optional backend verification and report that limitation.

> **Planning to evaluate your agent?** Tracing must be working before you run `agent-evaluation`. If verification is blocked, report the blocker and pause evaluation.

1. **Run the instrumented code once** — record its start time in epoch milliseconds before executing it, then capture `mlflow.get_last_active_trace_id()` immediately afterward in the same Python process. Save the ID so verification never requires rerunning the agent just to recover it.
2. **Read one trace** — flush pending writes once, then fetch the captured ID. If no ID is available, make one search scoped to the experiment and test run's time range, with `max_results=1`. Every UC search needs a `trace.timestamp_ms` filter, including searches by experiment ID; `max_results` alone does not limit the table scan. `search_traces()` has no `start_time` keyword.

Adapt `run_agent(test_input)` below to the application's entry point. This snippet shows the MLflow operations; the coding agent must enforce the readback deadline through its execution tool. Start the 60-second budget when the `Starting trace readback` marker appears, and preserve the printed trace ID before cancelling a slow verification.

```python
import time

import mlflow

run_start_ms = int(time.time() * 1000)
run_agent(test_input)
trace_id = mlflow.get_last_active_trace_id()
run_end_ms = int(time.time() * 1000)
print(f"Trace ID: {trace_id}", flush=True)

print("Starting trace readback", flush=True)
mlflow.flush_trace_async_logging()
if trace_id:
    trace = mlflow.get_trace(trace_id)
else:
    traces = mlflow.search_traces(
        locations=["<experiment_id>"],
        filter_string=(
            f"trace.timestamp_ms >= {run_start_ms} "
            f"AND trace.timestamp_ms <= {run_end_ms}"
        ),
        max_results=1,
        return_type="list",
        include_spans=True,
    )
    trace = traces[0] if traces else None

assert trace is not None, "Trace readback incomplete; report the error or missing configuration"
assert trace.info.request_time >= run_start_ms, "The trace predates this test run"
assert trace.data.spans, "No spans were captured"
```

3. **Inspect the returned trace once** — check the expected graph/root, LLM, and tool spans and their relevant inputs and outputs. Reuse the returned spans; do not fetch the trace again or launch another audit after this succeeds.

```python
print(f"Trace {trace.info.trace_id} has {len(trace.data.spans)} span(s)")
for span in trace.data.spans:
    print(f"  - {span.name} ({span.span_type})")
```

4. **Report the result** — tell the user how many traces and spans were found and confirm tracing is working. On Databricks, include a clickable link to a verified trace from the run using the URL guidance in `references/databricks.md`; an experiment link alone does not open the trace.

### If verification is slow or blocked

Allow at most one retry after a concrete, quick fix within the same time budget, such as correcting the experiment ID or flushing a missed export. Do not launch parallel searches, repeatedly poll, rerun the application, or try direct UC SQL, raw REST endpoints, or alternate credentials as verification fallbacks. A warehouse timeout, `RESOURCE_EXHAUSTED`/429, missing API key, or authentication/permission error is a reason to stop and hand off.

Use the existing configuration and error output to identify an obvious issue: tracking URI or experiment mismatch, autolog patching warnings, a missing model-provider key, failed authentication, or warehouse availability. Do not broaden this into an infrastructure investigation.

Tell the user what was implemented and what actually succeeded (application run, trace ID captured, spans inspected), what remains unverified, and the exact failing operation/error. Include the captured trace ID or link when available, labeling an unverified link accordingly. Ask for the specific missing prerequisite indicated by the error, such as configuring a named API-key environment variable, completing login, granting access, or providing a usable warehouse. Never ask the user to paste secrets into chat. If the cause is unclear, report the uncertainty and ask for help resolving it. Keep the instrumentation changes; do not claim backend verification succeeded.

For evaluation workflows that require automated validation, use `agent-evaluation/scripts/validate_tracing_runtime.py` as an alternative to the manual workflow under the same timeout and stopping rule. Do not run it as an additional audit after successful verification.

---

## Feedback Collection

Log user feedback on traces for evaluation, debugging, and fine-tuning. Essential for identifying quality issues in production.

See `references/feedback-collection.md` for:
- Recording user ratings and comments with `mlflow.log_feedback()`
- Capturing trace IDs to return to clients
- LLM-as-judge automated evaluation

---

## Reference Documentation

### Production Deployment

See `references/production.md` for:
- Environment variable configuration
- Async logging for low-latency applications
- Sampling configuration (MLFLOW_TRACE_SAMPLING_RATIO)
- Lightweight SDK (`mlflow-tracing`)
- Docker/Kubernetes deployment

### Advanced Patterns

See `references/advanced-patterns.md` for:
- Async function tracing
- Multi-threading with context propagation
- PII redaction with span processors

### Distributed Tracing

See `references/distributed-tracing.md` for:
- Propagating trace context across services
- Client/server header APIs

### Databricks (Unity Catalog storage)

See `references/databricks.md` for the required Databricks default: storing traces in Unity Catalog Delta tables by binding an experiment to a `UnityCatalog` trace location (catalog, schema, table prefix). It also covers tracing Databricks Apps, Model Serving, and external OpenTelemetry clients; verify a trace from the deployed identity.

---

## Next: debug from the traces you just captured

Tracing is now in place. When you move on to debug or improve the agent's behavior, read the spans first. Do not fall back to reading source code and output files alone. The trace shows what each step actually received, produced, and decided, which is the evidence source that pins down where behavior went wrong.

Load the `fix-agent-issue` skill for this. It grounds the diagnosis in the trace, what the agent did, what it should have done, and why, before any code change, and codifies the fix as a regression test so it sticks. Reach for it as soon as you start asking why the agent produced a given output, not only when someone explicitly reports a bug.
