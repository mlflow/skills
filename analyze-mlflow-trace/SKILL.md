---
name: analyzing-mlflow-trace
description: Analyze a single MLflow trace to explain an error, unexpected output, quality issue, or latency. Use an existing trace ID and report evidence without editing application code.
---

# Analyze an MLflow trace

## Inputs

Use the trace ID, configured backend, reported behavior and expected outcome. Reuse a full trace already fetched. For unknown IDs, use available `retrieving-mlflow-traces` guidance or a time-bounded search in the configured experiment.

## Workflow

1. Fetch full JSON once to a file; large piped CLI output can be truncated by the execution tool. `get` needs no `--output` flag. Avoid `--extract-fields` for nested span diagnosis.

   ```bash
   mlflow traces get --trace-id <TRACE_ID> > /tmp/trace.json
   ```

2. Inspect state, assessments and the span tree. This summary preserves parent links and separates trace errors from scoring errors:

   ```bash
   jq '{state: .info.state,
        spans: [.data.spans[] | {span_id, parent_span_id, name, status}],
        error_spans: [.data.spans[] | select(.status.code == "STATUS_CODE_ERROR") | .name],
        assessments: .info.assessments}' /tmp/trace.json
   ```

3. Read feedback **source, value, top-level rationale, validity and scope** before forming a hypothesis. HUMAN feedback describes the reported experience; corroborate it with spans. LLM_JUDGE/CODE feedback may be wrong or misconfigured. `feedback.error` means scoring failed, not that the application failed; exclude errored/overridden judgments from quality conclusions. Genuine zero/false values remain valid. Expectations are desired-output labels, not scores.
4. Follow actual inputs and outputs from root to the first divergence, then downstream effects. Raw MLflow span I/O is usually serialized JSON in `attributes["mlflow.spanInputs"]` / `attributes["mlflow.spanOutputs"]`; parse it before interpreting it. If the root is an empty wrapper, inspect descendants. For other OTel sources or versions, inspect raw keys instead of assuming this schema.
5. Check exception events and status messages. `OK` means execution success, not correct output; `STATUS_CODE_UNSET` is not itself an error. For performance, distinguish enclosing wrappers, overlapping child work and uninstrumented gaps; repeated names alone do not prove retries. Inspect recorded token usage rather than inventing cost or assuming a token spike proves injection.
6. Correlate the decisive span's observed data with relevant source/configuration when available. Function names are clues; autologged names follow framework conventions. Separate observed facts, plausible causes and missing evidence. Do not edit code, create scorers or run an evaluation for diagnosis alone.

## Verify

Report observed versus expected behavior, trace and decisive span IDs, the evidence chain, confidence and the smallest likely remedy. Confirm claims against actual span I/O and feedback semantics, not previews alone. If evidence is missing, say what would resolve the uncertainty; do not promote a hypothesis to a confirmed root cause.

## Read when needed

- Fields, units, CLI versus SDK representation, OTel attributes or assessment interpretation: [trace structure](references/trace-structure.md).
- Bounded batches, recurring failures or component bottlenecks: [cross-trace analysis](references/cross-trace-analysis.md).
- Multi-turn propagation: available `analyzing-mlflow-session` (directory `analyze-mlflow-chat-session`).
- A user-requested implementation: pass diagnosis and trace ID to `fix-agent-issue` when installed; investigation itself stops at evidence and recommendations.
