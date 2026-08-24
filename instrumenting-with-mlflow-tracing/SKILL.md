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

### Decide the trace boundary before adding a decorator

Before adding a manual `@mlflow.trace`, classify the operation. Do not decorate per-item validation, parsing, formatting, or utility helpers solely because they call an external service. Trace an external service call when it is a high-value operation worth diagnosing; otherwise, keep high-cardinality details on the existing workflow span. Return aggregate counts and a bounded failure list in that span's output.

With framework autologging enabled, inspect one trace before adding manual decorators. Add manual spans only for high-value operations that autologging misses. For `ThreadPoolExecutor` work, leave utilities untraced. Propagate context only for a child operation worth inspecting.

**Rule of thumb**: Trace operations that are important for debugging and identifying issues in your application.

---

## Verification

After instrumenting the code, **always verify that tracing is working**.

> **Planning to evaluate your agent?** Tracing must be working before you run `agent-evaluation`. Complete verification below first.


1. **Run the instrumented invocation in a new MLflow run** — execute the application or agent so that at least one traced operation fires. Record the run ID and the timestamp immediately before the invocation.
2. **Confirm invocation-scoped traces and topology** — search only the traces from this invocation, not the whole experiment. If the trace is not found, try `mlflow.flush_trace_async_logging()` to flush the background queue.
3. **Verify spans were captured** — confirm every invocation-scoped trace contains the expected spans and topology, not just an empty shell.

```python
import time
import mlflow

expected_root_trace_count = 1
expected_root_span_name = "<root_operation>"
expected_span_names = {"<root_operation>", "<high_value_child_operation>"}

run_start_ms = int(time.time() * 1000)
with mlflow.start_run() as run:
    run_instrumented_application()

mlflow.flush_trace_async_logging()
traces = mlflow.search_traces(
    experiment_ids=["<experiment_id>"],
    run_id=run.info.run_id,
    filter_string=f"trace.timestamp_ms > {run_start_ms}",
    return_type="list",
)
print(f"Found {len(traces)} trace(s)")
assert len(traces) == expected_root_trace_count, (
    f"Expected {expected_root_trace_count} root trace(s), found {len(traces)}"
)

for trace in traces:
    spans = trace.data.spans
    root_spans = [span for span in spans if span.parent_id is None]
    assert len(root_spans) == 1, (
        f"Trace {trace.info.trace_id} has {len(root_spans)} root spans"
    )
    assert root_spans[0].name == expected_root_span_name, (
        f"Trace {trace.info.trace_id} has unexpected root {root_spans[0].name!r}; "
        "a worker or helper must not become a standalone trace"
    )
    span_names = {span.name for span in spans}
    assert expected_span_names <= span_names, (
        f"Trace {trace.info.trace_id} is missing expected spans: "
        f"{expected_span_names - span_names}"
    )
    print(f"Trace {trace.info.trace_id} has {len(spans)} span(s)")
    for span in spans:
        print(f"  - {span.name} ({span.span_type})")
```

4. **Verify the expected trace topology** — state how many application root traces this test run should create, normally one. Inspect every trace returned for the run. A standalone worker or helper trace is a failed verification.

5. **Report the result** — report trace count, topology, and expected span coverage.

### If no traces appear

Check these in order:

- **Verification ran before traces were exported** — trace logging is asynchronous by default, so an in-process `search_traces()` right after the run can return zero before the background queue flushes (up to a few seconds later). Call `mlflow.flush_trace_async_logging()` before searching, as shown above.
- **Tracking URI not set** — is `mlflow.set_tracking_uri(...)` called before the agent run? Without this, traces go to a local `./mlruns` directory instead of the configured server.
- **Autolog warnings** — did `mlflow.autolog()` or framework-specific `mlflow.<framework>.autolog()` raise any warnings during setup? Check stderr for patching failures.
- **Wrong experiment ID** — verify the experiment ID passed to `search_traces()` matches the experiment active when the code ran (`mlflow.get_experiment_by_name(...)` to confirm).
- **Network/auth issues** — can the process reach the tracking server? Check for connection errors or 401/403 responses in logs.

For automated validation, use `agent-evaluation/scripts/validate_tracing_runtime.py`.

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

See `references/databricks.md` for storing traces in Unity Catalog Delta tables by binding an experiment to a `UnityCatalog` trace location (catalog, schema, table prefix).

---

## Next: debug from the traces you just captured

Tracing is now in place. When you move on to debug or improve the agent's behavior, read the spans first. Do not fall back to reading source code and output files alone. The trace shows what each step actually received, produced, and decided, which is the evidence source that pins down where behavior went wrong.

Load the `fix-agent-issue` skill for this. It grounds the diagnosis in the trace, what the agent did, what it should have done, and why, before any code change, and codifies the fix as a regression test so it sticks. Reach for it as soon as you start asking why the agent produced a given output, not only when someone explicitly reports a bug.
