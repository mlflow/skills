# Trace structure and interpretation

Use this reference when inspecting unfamiliar fields. A trace has `info` (identity, timing, location, assessments) and `data.spans` (operations linked by parent IDs). Inspect actual keys first: CLI wire JSON and Python SDK accessors differ, and OTel clients may use GenAI, OpenInference or custom attributes.

## Trace fields

| CLI JSON under `info` | Meaning / pitfall |
|---|---|
| `trace_id` | Preserve the full ID for SDK calls, including `trace:/<location>/<id>` for UC or a legacy `tr-` prefix. |
| `state` | `OK`, `ERROR`, `IN_PROGRESS`, `STATE_UNSPECIFIED`. Successful execution is not proof of answer quality; inspect child statuses separately. |
| `request_time`, `execution_duration` | Epoch start and wall-clock duration in **milliseconds**. Search uses `trace.timestamp_ms` / `trace.execution_time_ms`, not these JSON field names. |
| `request_preview`, `response_preview` | Display summaries may be truncated; inspect stored span I/O for evidence. |
| `client_request_id` | Application-side correlation ID, distinct from trace ID. |
| `trace_location` | Workspace experiment or UC catalog/schema/table prefix. A schema may contain several prefixes. UC `experiment_id` may be absent; use configured experiment identity. |
| `trace_metadata` | Request context: inspect actual keys and version semantics; do not assume every client records them. |
| `tags` | Mutable annotations, including environment/version/prompt provenance. |
| `assessments` | Feedback and expectation records; interpret using the rules below. |

Useful metadata: `mlflow.trace.user`, `mlflow.trace.session`, `mlflow.modelId`, `mlflow.sourceRun`, and JSON-encoded `mlflow.trace.tokenUsage` (`input_tokens`, `output_tokens`, `total_tokens`). Some clients store `mlflow.traceInputs` / `mlflow.traceOutputs`; verify availability and content before using them for extraction. Missing token fields mean unknown usage, not zero.

Useful tags: `mlflow.traceName`, `mlflow.linkedPrompts` (JSON prompt references), `mlflow.trace.sourceScorer` (distinguishes scorer-generated traces from application requests).

## Span fields and raw JSON

| Field / attribute | Interpretation |
|---|---|
| `span_id`, `trace_id`, `parent_span_id` | Build the tree by IDs, not array order. Root normally has null parent; distributed/partial exports may omit parents. |
| `name` | Application function or framework operation name; `mlflow.spanFunctionName` may identify the underlying function. |
| `attributes["mlflow.spanType"]` | SDK `span.span_type`; free-form, so custom types are valid. |
| `start_time_unix_nano`, `end_time_unix_nano` | Epoch **nanoseconds**, sometimes JSON strings. Convert before subtraction; divide by 1e6 for milliseconds. |
| `status.code`, `status.message` | `STATUS_CODE_ERROR` is an error; `STATUS_CODE_UNSET` alone is not. A handled child failure can coexist with successful parent completion. |
| `attributes["mlflow.spanInputs"]`, `attributes["mlflow.spanOutputs"]` | Usually serialized JSON strings; SDK `span.inputs` / `span.outputs` expose decoded values. They are not top-level CLI span fields. |
| `attributes["mlflow.chat.tokenUsage"]` | Recorded per-call token usage; inspect equivalent raw attributes for other OTel clients. |
| `attributes["mlflow.chat.tools"]` | Tools actually offered to the model. |
| `events` | Point-in-time records; exception events carry `exception.type`, `exception.message`, `exception.stacktrace` in attributes. |

To discover an unexpected schema, inspect `jq 'keys'`, `.data.spans[0] | keys`, `.data.spans[0].attributes`, and `.data.spans[0].status` on the saved file. Preserve missing data as a limitation rather than inventing an equivalent field.

Duration is inclusive of children. Parallel child durations can sum beyond parent wall time; do not sum wrapper and child durations as independent latency. Gaps show uninstrumented time, not a proven cause. Repeated operations suggest retries only when inputs, events or code confirm them.

## Assessments

CLI JSON uses `assessment_name`; SDK entities generally expose `name`. In JSON, values live under `feedback.value` or `expectation.value`, and scorer failures under `feedback.error`; inspect the installed representation rather than assuming every SDK object has these nested paths.

| Common field | Interpretation |
|---|---|
| `assessment_id` | Identity; may be absent for an unpersisted local record. |
| `create_time_ms`, `last_update_time_ms` | Assessment timing, distinct from request time. |
| `valid`, `overrides` | Exclude explicitly invalid/overridden records from current judgments; inspect history when relevant. An absent validity field is not automatically false. |
| `source.source_type`, `source.source_id` | HUMAN, LLM_JUDGE or CODE, plus reviewer/scorer identity. Preserve provenance. |
| `rationale` | **Top-level** explanation. Read with value and scorer definition: a label such as `no` has no universal polarity. |
| `span_id` | If set, assessment scope is one span, not the entire trace. |
| `run_id`, `metadata` | Evaluation provenance, scorer context, labeling instructions and source versions where present. |

**Feedback** is a judgment, which may be numeric, boolean, categorical or structured. Zero and false can be genuine results. HUMAN feedback is evidence of a reported experience, not infallible proof of technical cause. Validate automated judgments against content and intended scorer inputs.

**Scorer errors** (timeouts, parsing failures, incompatible trace types) are not application errors or negative quality scores. Report them separately. Inspect error code/message/stack trace and, when recorded, the scorer's own trace ID in metadata. Keys vary: inspect for `mlflow.assessment.scorerTraceId` or the actual emitted equivalent instead of guessing. Other useful metadata includes `mlflow.assessment.sourceRunId`, `mlflow.assessment.judgeCost`, and `mlflow.assessment.onlineScoringSessionId`.

**Expectations** are desired-output/ground-truth labels with source, metadata and optional span scope. Compare the actual value with the appropriate exact, semantic or structural criterion; do not treat the presence of an expectation as a passing score. For requested judge alignment or labeling, use `build-a-scorer` when available.

**Session assessments** are normally attached to the first trace and marked by `mlflow.trace.session` in assessment metadata. They describe the conversation, not necessarily that first turn. If retrieval starts mid-session, the first returned trace may not be the actual first trace.

## What to inspect by operation

| Operation | Evidence to compare |
|---|---|
| LLM / CHAT_MODEL | Actual messages, instructions, history order, model parameters and response. |
| AGENT / TOOL | Offered versus selected tools, arguments, results, handoffs and errors. |
| RETRIEVER / EMBEDDING / RERANKER | Query, actual documents, model/truncation, ranking before and after. |
| CHAIN / PARSER | Data passed between steps, raw response and expected format. |
| MEMORY | History/state loaded, omitted, summarized or overwritten. |
| GUARDRAIL | Data received versus modified/blocked output. |
| Custom types | Infer role from actual names, attributes, I/O and source. |

For additional version-specific fields use the [MLflow entity API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.entities.html) and installed SDK accessors. For non-MLflow events, consult [OTel exception conventions](https://opentelemetry.io/docs/specs/semconv/exceptions/exceptions-spans/). Local inspection remains necessary if documentation is unavailable or describes another version.
