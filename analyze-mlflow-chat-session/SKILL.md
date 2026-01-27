---
name: analyzing-mlflow-session
description: Analyzes an MLflow session — a sequence of traces from a multi-turn chat conversation or interaction. Use when the user asks to debug a chat conversation, review session or chat history, find where a multi-turn chat went wrong, or analyze patterns across turns. Triggers on "analyze this session", "what happened in this conversation", "debug session", "review chat history", "where did this chat go wrong", "session traces", "analyze chat", "debug this chat".
---

# Analyzing an MLflow Chat Session

## What is a Session?

A session groups multiple traces that belong to the same chat conversation or user interaction. Each trace in the session represents one turn: the user's input and the system's response. Traces within a session are linked by a shared session ID stored in trace metadata.

The session ID is stored in trace metadata under the key `mlflow.trace.session`. This key contains dots, which affects filter syntax (see below). All traces sharing the same value for this key belong to the same session.

## Reconstructing the Conversation

Reconstructing a session's conversation is a two-step process: discover the input/output schema from one trace, then extract those fields efficiently across all session traces. **Do NOT fetch full traces for every turn** — use `--extract-fields` on the search command instead.

**Step 1: Discover the schema.** Fetch the full JSON for one trace in the session using `mlflow traces get --trace-id <ID>`. Find the root span (the span with no `parent_id`) and examine its `attributes` dict to identify which keys hold the user input and system output. These could be:

- **MLflow standard attributes**: `mlflow.spanInputs` and `mlflow.spanOutputs` (set by the MLflow Python client)
- **Custom attributes**: Application-specific keys set via `@mlflow.trace` or `mlflow.start_span()` with custom attribute logging
- **Third-party OTel attributes**: Keys following GenAI Semantic Conventions, OpenInference, or other instrumentation conventions

The structure of these values also varies by application (e.g., a `query` string, a `messages` array, a dict with multiple fields). Inspect the actual attribute values to understand the format.

**Step 2: Extract across all session traces.** Once you know which attribute keys hold inputs and outputs, search for all traces in the session using `--extract-fields` to pull just those fields:

```bash
mlflow traces search \
  --experiment-id <EXPERIMENT_ID> \
  --filter-string 'metadata.`mlflow.trace.session` = "<SESSION_ID>"' \
  --order-by "timestamp_ms ASC" \
  --extract-fields 'info.trace_id,info.request_time,info.trace_metadata.`mlflow.traceInputs`,info.trace_metadata.`mlflow.traceOutputs`' \
  --output json \
  --max-results 1000
```

The `--extract-fields` example above uses `mlflow.traceInputs`/`mlflow.traceOutputs` from trace metadata — adjust the field paths based on what you discovered in step 1.

**CLI syntax notes:**

- Metadata keys containing dots **must** be escaped with backticks in filter strings and extract-fields: `` metadata.`mlflow.trace.session` ``
- **Shell quoting**: Backticks inside **double quotes** are interpreted by bash as command substitution (e.g., bash will try to run `` `mlflow.trace.session` `` as a command). Always use **single quotes** for the outer string when the value contains backticks. For example: `--filter-string 'metadata.\`mlflow.trace.session\` = "value"'`
- `--max-results` defaults to 100. Always set `--max-results 1000` to avoid missing turns. If exactly 1000 results are returned (meaning more may exist), increase the value.

To inspect a specific turn in detail (e.g., after identifying a problematic turn), fetch its full trace:

```bash
mlflow traces get --trace-id <TRACE_ID>
```

## Analysis Insights

- **Conversation quality often degrades over turns.** Early turns may be correct while later ones fail. When a user reports a bad answer, check whether earlier turns were fine — this narrows the problem to what changed (new context, accumulated errors, context window overflow).
- **Context accumulation is a common failure mode.** Many chat applications pass the full conversation history to the LLM at each turn. As the conversation grows, the context can exceed the model's window, cause truncation, or dilute relevant information. Compare token usage across turns (via `mlflow.trace.tokenUsage` in trace metadata, if set) to detect growing context.
- **Each turn is a full trace with its own span tree.** To understand *why* a specific turn went wrong, analyze that turn's trace the same way you would a single trace — check assessments, examine spans, correlate with code.
- **Earlier turns can poison later ones.** If the system gave a wrong answer in turn 3 and the user didn't correct it, turns 4+ may build on that wrong information. When investigating a failure at turn N, always check turns N-1 and N-2 for earlier errors that propagated.
- **Gaps in timestamps indicate pauses or lost turns.** Sorting by `timestamp_ms` gives chronological order. Large gaps may mean the user left and returned, or that some turns failed silently and weren't recorded.
- **Session-level patterns reveal systemic issues.** If multiple sessions fail at similar turn counts or with similar queries, the problem is likely in the application's context management rather than a one-off issue.

## Codebase Correlation

- **Session ID assignment**: Search the codebase for where `mlflow.trace.session` is set to understand how sessions are created — per user login, per browser tab, per explicit "new conversation" action, etc.
- **Context window management**: Look for how the application constructs the message history passed to the LLM at each turn. Common patterns include sliding window (last N messages), summarization of older turns, or full history. This implementation determines what context the model sees and is a frequent source of multi-turn failures.
- **Memory and state**: Some applications maintain state across turns beyond message history (e.g., extracted entities, user preferences, accumulated tool results). Search for how this state is stored and passed between turns.

## Example: Wrong Answer on Chat Turn 5

A user reports that their chatbot gave an incorrect answer on the 5th message of a chat conversation.

**1. Discover the schema and reconstruct the conversation.**

Fetch the first trace in the session and inspect the root span's attributes to find which keys hold inputs and outputs. In this case, `mlflow.spanInputs` contains the user query and `mlflow.spanOutputs` contains the assistant response. Then search all session traces, extracting those fields in chronological order. Scanning the extracted inputs and outputs confirms that turn 5's response is wrong, and reveals whether earlier turns look correct.

**2. Check if the error originated in an earlier turn.**

Turn 3's response contains a factual error that the user didn't challenge. Turn 4 builds on that incorrect information, and turn 5 compounds it. The root cause is in turn 3, not turn 5.

**3. Analyze the root-cause turn as a single trace.**

Fetch the full trace for turn 3 and analyze it — examine assessments (if any), walk the span tree, check retriever results, and correlate with code. The retriever returned an outdated document, causing the wrong answer.

**4. Recommendations.**

- Fix the retriever's data source to exclude or update outdated documents.
- Add per-turn assessments to detect errors before they propagate across the conversation.
- Consider implementing conversation-level error detection (e.g., checking consistency of answers across turns).
