---
name: instrumenting-with-mlflow-tracing
description: Instruments Python and TypeScript code with MLflow Tracing for observability. Triggers on questions about adding tracing, instrumenting agents/LLM apps, getting started with MLflow tracing, or tracing specific frameworks (LangGraph, LangChain, OpenAI, DSPy, CrewAI, AutoGen). Examples - "How do I add tracing?", "How to instrument my agent?", "How to trace my LangChain app?", "Getting started with MLflow tracing", "Trace my TypeScript app"
---

# MLflow Tracing Instrumentation Guide

## Critical Rules

- **Prefer calling `mlflow.set_tracking_uri()`** when starting from scratch. The local file store (`./mlruns`) is deprecated. Point to an MLflow tracking server instead. For uv-based projects, instruct users to start the server with `uv run mlflow server --port 5000`.
- **Follow the Quick Start in the language-specific guide exactly** — do not skip any setup calls (`set_tracking_uri`, `set_experiment`, `autolog`).

## Checklist

Complete each applicable step in order.

1. [ ] **Detect language** — check for `pyproject.toml`/`requirements.txt` (Python) or `package.json` (TypeScript). Read the matching guide in `references/`.
2. [ ] **Install dependency** — add `mlflow` (Python) or `mlflow-tracing` (TypeScript) to the project's dependency file.
3. [ ] **Configure tracking** — add `set_tracking_uri`, `set_experiment`, and `autolog` calls per the Quick Start.
4. [ ] **Session tracking** — if the app has multi-turn conversations or threads, wrap the entry point with `@mlflow.trace` (so there is an active span) and attach session/user metadata via `update_current_trace()`.
5. [ ] **Feedback collection** — search the codebase for existing feedback mechanisms (e.g., `/feedback` commands, rating endpoints, thumbs up/down). If any exist, read `references/feedback-collection.md` and wire them to `mlflow.log_feedback()`. Capture trace IDs from the active span so feedback links to the correct trace.
6. [ ] **Verify** — confirm the dependency installs and the app starts without errors.

---

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

## Session Tracking

For multi-turn chat applications, attach session/user metadata to traces using `mlflow.update_current_trace()`. This requires an active span — when using autolog, always wrap the entry point function with `@mlflow.trace` so the span is active when `update_current_trace()` is called. See the "User/Session Tracking" section in the language-specific guide for details.

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
