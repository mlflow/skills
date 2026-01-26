# MLflow Tracing Integration

Complete guide for integrating MLflow tracing with your agent. This is the authoritative source for all tracing implementation - follow these instructions after completing environment setup in setup-guide.md.

**Prerequisites**: Complete setup-guide.md Steps 1-2 (MLflow install + environment configuration)

## Quick Start

Three steps to integrate tracing:

1. **Enable autolog** - Add `mlflow.<library>.autolog()` BEFORE importing agent code
2. **Decorate entry points** - Add `@mlflow.trace` to agent's main functions
3. **Verify** - Run test query and check trace is captured

**Minimum implementation:**
```python
import mlflow
mlflow.langchain.autolog()  # Before imports

from my_agent import agent

@mlflow.trace
def run_agent(query: str) -> str:
    return agent.run(query)
```

See sections below for detailed instructions and verification steps.

## Documentation Access Protocol

**MANDATORY: Follow the documentation protocol to read MLflow documentation before implementing:**

```bash
# Query llms.txt for tracing documentation
curl https://mlflow.org/docs/latest/llms.txt | grep -A 20 "tracing"
```

Or use WebFetch:
- Start: `https://mlflow.org/docs/latest/llms.txt`
- Query for: "MLflow tracing documentation", "autolog setup", "trace decorators"
- Follow referenced URLs for detailed guides

## Key Rules for Agent Evaluation

1. **Enable Autolog FIRST** - Call `mlflow.{library}.autolog()` before importing agent code
   - Captures internal library calls automatically
   - Supported: `langchain`, `langgraph`, `openai`, `anthropic`, etc.

2. **Add @mlflow.trace to Entry Points** - Decorate agent's main functions
   - Creates top-level span in trace hierarchy
   - Example: `@mlflow.trace` on `run_agent()`, `process_query()`, etc.

3. **Enable Session Tracking for Multi-Turn** - Group conversations by session
   ```python
   trace_id = mlflow.get_last_active_trace_id()
   mlflow.set_trace_tag(trace_id, "session_id", session_id)
   ```

4. **Verify Trace Creation** - Test run should create traces with non-None trace_id
   ```bash
   # Check traces exist
   uv run mlflow traces search --experiment-id $MLFLOW_EXPERIMENT_ID
   ```

5. **Tracing Must Work Before Evaluation** - If traces aren't created, stop and troubleshoot

## Minimal Example

```python
# step 1: Enable autolog BEFORE imports
import mlflow
mlflow.langchain.autolog()  # Or langgraph, openai, etc. Use the documentation protocol to find the integration for different libraries.

# step 2: Import agent code
from my_agent import agent

# step 3: Add @mlflow.trace decorator
@mlflow.trace
def run_agent(query: str, session_id: str = None) -> str:
    """Agent entry point with tracing."""
    result = agent.run(query)

    # step 4 (optional): Track session for multi-turn
    if session_id:
        trace_id = mlflow.get_last_active_trace_id()
        if trace_id:
            mlflow.set_trace_tag(trace_id, "session_id", session_id)

    return result
```

## ⚠️ Critical Verification Checklist

After implementing tracing, verify these requirements **IN ORDER**:

**Quick verification:** Edit and run `scripts/validate_agent_tracing.py` - it checks all items below automatically.

**Manual verification** (if needed):

### 1. Autolog Enabled
```bash
# Find autolog call
grep -r "mlflow.*autolog" .
```
**Expected**: Find autolog() call in initialization file (main.py, __init__.py, app.py, etc.)

### 2. Import Order Correct
**Check**: Autolog call appears BEFORE any agent/library imports in the file
**Expected**: The line with `mlflow.autolog()` comes before any `from your_agent import ...` statements

### 3. Entry Points Decorated
```bash
# Find trace decorators
grep -r "@mlflow.trace" .
```
**Expected**: Find @mlflow.trace on agent's main functions

### 4. Traces Created
```bash
# Run agent with test input
uv run python -c "from my_agent import run_agent; run_agent('test query')"

# Check trace was created
uv run mlflow traces search --experiment-id $MLFLOW_EXPERIMENT_ID --extract-fields info.trace_id
```
**Expected**: Non-empty trace_id returned

### 5. Trace Structure Complete
```bash
# View trace details
uv run mlflow traces get <trace_id>
```
**Expected**:
- Top-level span with your function name
- Child spans showing internal library calls (if autolog enabled)
- Session tags (if multi-turn agent)

**If ANY check fails**: Stop and troubleshoot before proceeding to evaluation.

## Common Issues

**Traces not created**:
- Check autolog is called before imports
- Verify decorator is @mlflow.trace (not @trace or @mlflow.trace_span)
- Ensure MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID are set

**Empty traces** (no child spans):
- Autolog may not support your library version
- Check MLflow docs for supported library versions
- Verify autolog is called before library imports

**Session tracking not working**:
- Verify `trace_id = mlflow.get_last_active_trace_id()` is called inside traced function
- Check `mlflow.set_trace_tag(trace_id, key, value)` has correct parameter order

For detailed troubleshooting, see `troubleshooting.md`.

## Tracing Integration Complete

After completing all steps above, verify:

- [ ] Test run creates traces with non-None trace_id (verified with validate_agent_tracing.py)
- [ ] Traces visible in MLflow UI or via `mlflow traces search`
- [ ] Trace hierarchy includes both @mlflow.trace spans and autolog spans

---

## Trace Analysis Patterns

Once tracing is integrated, use these patterns to analyze trace data for debugging and evaluation.

### Search Traces

```python
import mlflow
import time

# All traces in current experiment
all_traces = mlflow.search_traces()

# Successful traces only
ok_traces = mlflow.search_traces(
    filter_string="attributes.status = 'OK'"
)

# Error traces only
error_traces = mlflow.search_traces(
    filter_string="attributes.status = 'ERROR'"
)

# Recent traces (last hour)
one_hour_ago = int((time.time() - 3600) * 1000)
recent = mlflow.search_traces(
    filter_string=f"attributes.timestamp_ms > {one_hour_ago}"
)

# Slow traces (> 5 seconds)
slow = mlflow.search_traces(
    filter_string="attributes.execution_time_ms > 5000"
)

# By tag
prod_traces = mlflow.search_traces(
    filter_string="tags.environment = 'production'"
)

# By trace name (note backticks for dotted names)
specific_app = mlflow.search_traces(
    filter_string="tags.`mlflow.traceName` = 'my_app_function'"
)
```

### Latency Breakdown by Span Type

```python
from mlflow.entities import Trace, SpanType
from collections import defaultdict

def latency_by_span_type(trace: Trace) -> dict:
    """Break down latency by span type (LLM, TOOL, RETRIEVER, etc.)"""
    spans = trace.search_spans()
    type_latencies = defaultdict(list)

    for span in spans:
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6
        span_type = str(span.span_type) if span.span_type else "UNKNOWN"
        type_latencies[span_type].append(duration_ms)

    results = {}
    for span_type, durations in type_latencies.items():
        results[span_type] = {
            "count": len(durations),
            "total_ms": round(sum(durations), 2),
            "avg_ms": round(sum(durations) / len(durations), 2),
        }
    return results
```

### Find Bottlenecks

```python
def find_bottlenecks(trace: Trace, top_n: int = 5) -> list:
    """Find the slowest spans in a trace."""
    spans = trace.search_spans()
    exclude_patterns = ["forward", "predict", "root"]

    span_timings = []
    for span in spans:
        span_name_lower = span.name.lower()
        if any(p in span_name_lower for p in exclude_patterns):
            continue

        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6
        span_timings.append({
            "name": span.name,
            "span_type": str(span.span_type) if span.span_type else "UNKNOWN",
            "duration_ms": round(duration_ms, 2),
        })

    span_timings.sort(key=lambda x: -x["duration_ms"])
    return span_timings[:top_n]
```

### Analyze Tool Calls

```python
from mlflow.entities import SpanType

def analyze_tool_calls(trace: Trace) -> dict:
    """Analyze tool calls in a trace."""
    spans = trace.search_spans()
    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL]

    tool_calls = []
    for span in tool_spans:
        duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6
        tool_name = span.name.split(".")[-1] if "." in span.name else span.name

        tool_calls.append({
            "tool_name": tool_name,
            "duration_ms": round(duration_ms, 2),
            "inputs": span.inputs,
        })

    return {
        "total_tool_calls": len(tool_calls),
        "calls": tool_calls,
    }
```

### Detect Errors

```python
from mlflow.entities import SpanStatusCode

def detect_errors(trace: Trace) -> dict:
    """Detect error patterns in a trace."""
    spans = trace.search_spans()

    errors = {
        "failed_spans": [],
        "empty_outputs": [],
    }

    for span in spans:
        if span.status and span.status.status_code == SpanStatusCode.ERROR:
            errors["failed_spans"].append({
                "name": span.name,
                "error": span.status.description if span.status.description else "Unknown"
            })

        if span.outputs is None or span.outputs == {} or span.outputs == []:
            errors["empty_outputs"].append({"name": span.name})

    return errors
```

### Filter Syntax Reference

| Syntax Element | Rule |
|----------------|------|
| String values | Use single quotes: `'OK'` NOT `"OK"` |
| Dotted names | Use backticks: `tags.\`mlflow.traceName\`` |
| Prefix | Required for attributes: `attributes.status` |
| Logical operators | `AND` supported, `OR` NOT supported |
| Time values | Use milliseconds since epoch |
