---
name: instrumenting-with-mlflow-tracing
description: Instruments Python and TypeScript code with MLflow Tracing for observability. Must be loaded when setting up tracing as part of any workflow including agent evaluation. Triggers on adding tracing, instrumenting agents/LLM apps, getting started with MLflow tracing, tracing specific frameworks (LangGraph, LangChain, OpenAI, DSPy, CrewAI, AutoGen), or when another skill references tracing setup. Examples - "How do I add tracing?", "Instrument my agent", "Trace my LangChain app", "Set up tracing for evaluation"
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

**Rule of thumb**: Trace operations that are important for debugging and identifying issues in your application.

---

## Verification

After instrumenting the code, **always verify that tracing is working**.

> **Planning to evaluate your agent?** Tracing must be working before you run `agent-evaluation`. Complete verification below first.


1. **Run the instrumented code** — execute the application or agent so that at least one traced operation fires
2. **Confirm traces are logged** — use `mlflow.search_traces()` or `MlflowClient().search_traces()` to check that traces appear in the experiment:

```python
import mlflow

traces = mlflow.search_traces(experiment_ids=["<experiment_id>"])
print(f"Found {len(traces)} trace(s)")
assert len(traces) > 0, "No traces were logged — check tracking URI and experiment settings"
```

3. **Report the result** — tell the user how many traces were found and confirm tracing is working

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
