# Distributed tracing

Use when spans cross service boundaries. Initialize each service with the intended backend and experiment first ([Python](python.md), [Databricks](databricks.md)). Check both deployed identities can export.

## Propagate the active context

On the client, obtain headers inside the active span and send them with the request:

```python
from mlflow.tracing import get_tracing_context_headers_for_http_request

with mlflow.start_span("client-request") as span:
    span.set_inputs(payload)
    response = requests.post(
        service_url, json=payload,
        headers=get_tracing_context_headers_for_http_request(), timeout=30,
    )
    response.raise_for_status()
    span.set_outputs(response.json())
```

On the server, restore headers **before creating the handler span**:

```python
from mlflow.tracing import set_tracing_context_from_http_request_headers

with set_tracing_context_from_http_request_headers(dict(request.headers)):
    with mlflow.start_span("server-handler") as span:
        span.set_inputs(payload)
        result = handle(payload)
        span.set_outputs(result)
```

Adapt request/payload extraction to Flask, FastAPI or the existing server. Keep normal authentication and request handling; trace context is not authorization. Preserve W3C trace context and verify persisted trace IDs and parent IDs connect both services before claiming one connected trace. An absent parent may indicate incomplete export or lost propagation.

For other transports/frameworks, use [distributed tracing docs](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/distributed-tracing.md) and installed signatures. Follow the [bounded readback procedure](../SKILL.md#verification) for setup verification.
