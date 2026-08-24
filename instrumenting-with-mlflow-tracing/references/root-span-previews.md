# Root span previews and compact values

Use this pattern only when the root span's input or output is not already readable in the trace list. MLflow derives a readable preview automatically for OpenAI chat-shaped payloads: a top-level `messages` list, `choices[0].message`, or a Responses-API `input` list. Do not override those previews.

For other shapes, such as an agent state dict, the default preview is a serialized object that can be hard to scan. The UI truncates long strings itself, so keep the useful value intact rather than adding an arbitrary slice in application code.

## Keep the decorator and set a trace-list preview

Use `span.set_inputs()` when you want compact stored inputs. Use `update_current_trace()` when only the trace list needs a readable request or response preview; the decorated function can still return the full state.

```python
import mlflow
from mlflow.entities import SpanType

@mlflow.trace(span_type=SpanType.AGENT)
def run_agent(graph, customer: str, date: str) -> dict:
    span = mlflow.get_current_active_span()
    span.set_inputs({"customer": customer, "date": date})

    state = graph.invoke({"customer": customer, "date": date})
    output_path = state.get("output_path", "")
    preview = state.get("note_text") or ""
    mlflow.update_current_trace(
        request_preview=f"customer={customer}, date={date}",
        response_preview=f"output_path={output_path}, preview={preview}",
    )
    return state
```

## Store compact inputs and outputs on the root span

Keep the decorator when the function must return full state but the root span should store compact inputs or outputs. Explicit `span.set_inputs()` and `span.set_outputs()` values take precedence over the decorator's automatic capture.

```python
from mlflow.entities import SpanType

@mlflow.trace(span_type=SpanType.AGENT)
def run_agent(graph, customer: str, date: str) -> dict:
    span = mlflow.get_current_active_span()
    span.set_inputs({"customer": customer, "date": date})
    state = graph.invoke({"customer": customer, "date": date})
    span.set_outputs(
        {
            "output_path": state.get("output_path", ""),
            "preview": state.get("note_text") or "",
        }
    )
    return state
```
