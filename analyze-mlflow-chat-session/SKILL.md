---
name: analyzing-mlflow-session
description: Analyze a multi-turn MLflow chat session, reconstruct its conversation, and identify the first bad turn and downstream propagation. Use for session history and conversation diagnosis without code changes.
---

# Analyze an MLflow chat session

## Inputs

Reuse the configured backend, experiment, session ID, relevant time window and reported/expected behavior. Session traces share metadata key `mlflow.trace.session`; verify how the application assigns that ID so unrelated conversations are not combined.

## Workflow

1. **Discover the actual schema.** Check installed `mlflow traces search --help`. Search chronologically with an explicit experiment and time bounds; save JSON before parsing it:

   ```bash
   mlflow traces search --experiment-id <EXPERIMENT_ID> \
     --filter-string 'metadata.`mlflow.trace.session` = "<SESSION_ID>" AND trace.timestamp_ms >= <START_MS> AND trace.timestamp_ms <= <END_MS>' \
     --order-by 'timestamp_ms ASC' --max-results 1 --output json > /tmp/first_trace.json
   ```

   Replace placeholders with the incident/session window. Every UC search, even with `max-results 1`, requires a timestamp predicate. If the period is unknown, resolve it before searching. Single outer shell quotes protect backticks from command substitution. Distinguish an empty result from a query failure; do not fetch a null ID.

2. Fetch that first trace's full JSON once: `mlflow traces get --trace-id <ID> > /tmp/trace_detail.json`. Find the root by `parent_span_id == null`. Inspect actual attribute keys and data shapes; MLflow I/O is typically serialized JSON under `mlflow.spanInputs` / `mlflow.spanOutputs`, while external OTel sources may differ. If root I/O is empty, walk descendants to the closest meaningful chat-turn span. Do not assume immediate children suffice.
3. **Interpret assessments.** Session-level assessments normally live on the first session trace and have `mlflow.trace.session` in assessment metadata; other records may apply only to a turn or span. A time-window's first result may be mid-session: disclose missing earlier context or obtain the relevant earlier window. Read source, value and top-level `rationale`. Separate HUMAN reports from automated LLM_JUDGE/CODE judgments. Exclude explicit invalid/overridden judgments and `feedback.error` from quality conclusions, while reporting scorer failures separately. Zero/false scores may be genuine; expectations are labels, not results. Session feedback does not locate a bad turn by itself.
4. **Reconstruct in order.** Search the same bounded session with a suitable page size and `--order-by 'timestamp_ms ASC'`. After discovering available paths, use `--extract-fields` for IDs, time, state, assessments and turn I/O. Some clients expose `info.trace_metadata.\`mlflow.traceInputs\`` / `mlflow.traceOutputs`; do not assume these exist. Nested span extraction can be incomplete: fetch full traces only where required data is missing or a turn needs deeper inspection. Follow continuation tokens, preserving bounds and ordering. A first page is not the whole conversation.
5. **Find the first bad turn**, not just the turn the user reported. Compare actual inputs, outputs and expectations chronologically; trace how an earlier error enters later history, summaries, memory, retrieval or tool arguments. Fetch the decisive turn and necessary neighbors in full. Check actual model messages to establish propagation rather than assuming later turns saw every previous response.
6. Correlate with session assignment, history truncation/summarization, and state read/write code. Read source to confirm the explanation; do not edit the agent, add scorers, or build a regression suite for a diagnosis-only request.

## Verify

Return the reconstructed sequence or concise relevant turns, first divergence with trace/span IDs, downstream propagation, likely cause and confidence. State pagination/window/schema gaps and distinguish application failures from failed scorers. `OK` is execution status, not proof of quality; `STATUS_CODE_UNSET` alone is not an error.

## Read when needed

- Detailed field semantics: if installed, `analyzing-mlflow-trace` (directory `analyze-mlflow-trace`) and its `references/trace-structure.md`. The essential assessment rules above remain usable standalone.
- More filter/API options: [official search guide](https://mlflow.org/docs/latest/genai/tracing/search-traces.md), matched to installed CLI help. If unavailable, use installed help and report unresolved compatibility.
- Existing helpers are optional, resolved relative to this skill: `scripts/discover_schema.sh <EXPERIMENT_ID> <SESSION_ID>` is **not suitable for UC discovery** because it has no time filter; use the bounded workflow above. `scripts/inspect_turn.sh <TRACE_ID>` fetches details but labels all non-OK statuses as errors; inspect actual `STATUS_CODE_ERROR` and do not treat UNSET as failure. Both write fixed filenames, so use a separate working directory to avoid overwriting evidence.
- If implementation is requested, pass the diagnosis and trace IDs to available `fix-agent-issue`.
