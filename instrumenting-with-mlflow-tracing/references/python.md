# MLflow Tracing - Python Guide

## Contents
- Quick Start
- Instrumentation Methods (AutoLogging, Decorator, Manual Spans)
- User/Session Tracking
- Combining AutoLogging with Custom Tracing
- Common Issues

---

## Quick Start

### Install and Configure

```bash
pip install mlflow>=3.8.0
```

**Check if `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_ID` are already set in the environment.** If both are set, skip the configuration below — MLflow will use them automatically.

Only call these if the environment is NOT pre-configured:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # skip if MLFLOW_TRACKING_URI is set
mlflow.set_experiment("my-agent")                 # skip if MLFLOW_EXPERIMENT_ID is set
```

> **On Databricks: opt into UC trace storage**  
> Traces land in the experiment backend by default (capped at 100,000 per experiment). For production use, bind the experiment to a UC trace location when calling `set_experiment`. See [`references/databricks.md`](references/databricks.md) for the one-liner.

### Enable Tracing

**For supported frameworks** (LangChain, LangGraph, OpenAI, etc.):

```python
mlflow.langchain.autolog()  # or openai, anthropic, litellm, etc.
```

**For custom code:**

```python
from mlflow.entities import SpanType

@mlflow.trace(span_type=SpanType.CHAIN)
def my_function(query: str) -> str:
    # Your code here
    return result
```

---

## Instrumentation Methods

### Method 1: AutoLogging (Recommended for Frameworks)

Zero-code instrumentation for supported libraries. See the [Integrations page](https://mlflow.org/docs/latest/genai/tracing/integrations.md) for the complete list.

```python
import mlflow

# Enable each autolog independently. One missing package will not disable another.
for flavor, fn in [
    ("langchain", mlflow.langchain.autolog),  # LangChain and LangGraph, needs only langchain-core
    ("openai", mlflow.openai.autolog),
    ("anthropic", mlflow.anthropic.autolog),
    ("gemini", mlflow.gemini.autolog),  # Google Gemini (google-genai SDK)
    ("litellm", mlflow.litellm.autolog),
    ("dspy", mlflow.dspy.autolog),
    ("autogen", mlflow.autogen.autolog),
    ("crewai", mlflow.crewai.autolog),
]:
    try:
        fn()
    except Exception as e:
        import warnings
        warnings.warn(f"mlflow.{flavor}.autolog() failed: {e}. Tracing for {flavor} disabled.")
```

> **LangGraph note:** LangGraph is traced via `mlflow.langchain.autolog()` (there is no `mlflow.langgraph.autolog()`). It requires only `langchain-core`, not the full `langchain` package. Without the full `langchain` package installed, the version-compatibility check raises `ModuleNotFoundError: No module named 'langchain'`. The try/except above prevents that error from disabling other autolog calls.

### Method 2: Decorator (Recommended for Custom Code)

**Prefer decorator over manual spans** - it auto-captures function name, inputs, and outputs.

```python
from mlflow.entities import SpanType

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_documents(query: str) -> list[str]:
    return documents

@mlflow.trace(span_type=SpanType.TOOL)
def search_database(sql: str) -> dict:
    return results
```

**Span types**: `LLM`, `CHAIN`, `TOOL`, `AGENT`, `RETRIEVER`, `EMBEDDING`, `RERANKER`, `PARSER`, `UNKNOWN`

### Method 3: Manual Spans (When Decorator Not Possible)

Use only when you can't use a decorator:
- **Tracing code not wrapped in a function** (e.g., script-level code, loop bodies)
- **Dynamic span names** computed at runtime

```python
with mlflow.start_span(name=f"process_{item_id}") as span:
    span.set_inputs({"query": query})  # Must set manually
    result = process(query)
    span.set_outputs({"result": result})  # Must set manually
```

> **Warning:** A `start_span` context manager that never calls `span.set_inputs(...)`
> and `span.set_outputs(...)` produces a span with a name and a duration but no I/O. In
> the trace UI it looks identical to a span that legitimately has none, so a reviewer
> cannot tell instrumentation succeeded. Always set inputs and outputs on a manual span,
> or use the `@mlflow.trace` decorator, which captures both automatically.

### Span content: record full data, not a count

**Always record the actual retrieved content in span inputs and outputs, never a bare count or size summary.** A span that records `{"matches": 20}` instead of the actual documents is useless for debugging.

MLflow's trace store accepts large string attributes. Single attributes over 500K characters have been verified to round-trip correctly, so size is not a reason to truncate. Full content is the default. Any truncation must be a deliberate choice with a comment explaining why.

```python
# Wrong: a count tells you nothing about what the agent retrieved
span.set_outputs({"matches": len(docs)})

# Right: record the content so the trace is debuggable
# docs is the actual list of retrieved records, e.g. [{"text": ..., "id": ...}]
span.set_outputs({"documents": docs})
```

Apply this to all spans that fetch content: retrieval results, search hits, tool outputs, messages, and records.

---

## User/Session Tracking

For multi-turn applications, use standard metadata fields `mlflow.trace.user` and `mlflow.trace.session`.

```python
from fastapi import Request
from mlflow.entities import SpanType

@app.post("/chat")
def handle_chat(request: Request, body: ChatRequest):
    user_id = request.headers.get("X-User-ID", "anonymous")
    session_id = request.headers.get("X-Session-ID", "default")
    return chat(body.message, user_id, session_id)

@mlflow.trace(span_type=SpanType.CHAIN)
def chat(message: str, user_id: str, session_id: str) -> str:
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
        }
    )
    return response
```

**Query traces by user:**

```python
traces = mlflow.search_traces(
    filter_string="metadata.`mlflow.trace.user` = 'user123'"
)
```

---

## Combining AutoLogging with Custom Tracing

```python
import mlflow
from mlflow.entities import SpanType
from langchain_openai import ChatOpenAI

mlflow.langchain.autolog()

@mlflow.trace(name="rag_pipeline", span_type=SpanType.CHAIN)
def rag_query(question: str) -> str:
    docs = retrieve_documents(question)  # Custom function

    llm = ChatOpenAI()  # Auto-traced by autolog
    response = llm.invoke(format_prompt(docs, question))

    return response.content
```

### Adding descriptions to LangGraph node spans

When `mlflow.langchain.autolog()` traces a LangGraph graph, each node becomes a span named after the node function. The span shows inputs and outputs but nothing that explains what the node does. To make the trace readable without opening each node's code, write the node's docstring into the span's `description` attribute:

```python
import mlflow

def web_research(state: dict) -> dict:
    """Search the web for information relevant to the query."""
    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute("description", web_research.__doc__)
    # node logic here
    return state

def enrich_findings(state: dict) -> dict:
    """Cross-reference web results with the internal knowledge base."""
    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute("description", enrich_findings.__doc__)
    # node logic here
    return state
```

The `description` attribute appears in the span's Attributes tab in the MLflow trace viewer. Any string key works. `"description"` is a readable convention.

---

## Common Issues

**Traces not appearing?**
1. Verify tracking URI is correct (`MLFLOW_TRACKING_URI` env var or `mlflow.set_tracking_uri()`)
2. Ensure autolog is called before framework imports
3. Check experiment is configured (`MLFLOW_EXPERIMENT_ID` env var or `mlflow.set_experiment()`)

**Nested spans not connected?**
- Use `@mlflow.trace` or context managers consistently
- For threading, see `advanced-patterns.md`
