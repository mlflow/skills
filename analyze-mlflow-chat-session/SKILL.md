---
name: analyzing-mlflow-session
description: Analyzes an MLflow session — a sequence of traces from a multi-turn chat conversation or interaction. Use when the user asks to debug a chat conversation, review session or chat history, find where a multi-turn chat went wrong, or analyze patterns across turns. Triggers on "analyze this session", "what happened in this conversation", "debug session", "review chat history", "where did this chat go wrong", "session traces", "analyze chat", "debug this chat".
---

# Analyzing an MLflow Chat Session

## What is a Session?

A session groups multiple traces that belong to the same chat conversation or user interaction. Each trace in the session represents one turn: the user's input and the system's response. Traces within a session are linked by a shared session ID stored in trace metadata.

The session ID is stored in trace metadata under the key `mlflow.trace.session`. This key contains dots, which affects filter syntax (see below). All traces sharing the same value for this key belong to the same session.

## Fetching Session Traces

```bash
# Get all traces in a session, in chronological order
mlflow traces search \
  --experiment-id <EXPERIMENT_ID> \
  --filter-string "metadata.`mlflow.trace.session` = '<SESSION_ID>'" \
  --order-by "timestamp_ms ASC" \
  --output json \
  --max-results 1000
```

**CLI syntax notes:**

- The metadata key `mlflow.trace.session` contains dots, so it **must** be escaped with backticks in the filter string: `` metadata.`mlflow.trace.session` ``
- In a shell command, backticks inside double quotes are interpreted as command substitution. Ensure proper escaping — e.g., use backslash-escaped backticks within double quotes (`\`mlflow.trace.session\``) or structure quoting carefully.
- Default `--max-results` is 100. Increase for long conversations.

To inspect a specific turn in detail, fetch its full trace:

```bash
mlflow traces get --trace-id <TRACE_ID>
```

## Reconstructing the Conversation

Each trace's inputs and outputs represent one conversational turn. The full (non-truncated) inputs and outputs are available in trace metadata as `mlflow.traceInputs` and `mlflow.traceOutputs` (JSON-encoded strings). The `request_preview` and `response_preview` fields on TraceInfo provide truncated versions for quick scanning.

The input/output structure varies by application. Common patterns:

- **Chat applications**: Inputs contain a `messages` array or a `query` string; outputs contain the assistant's response text.
- **Agent applications**: Inputs contain the user query; outputs contain the agent's final answer, potentially with intermediate tool results.
- **RAG applications**: Inputs contain the query; outputs contain the generated answer and retrieved context.

When the structure isn't obvious, fetch one full trace with `mlflow traces get` and examine the root span's `inputs` and `outputs` to understand the schema.

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

**1. Fetch all session traces and scan the conversation.**

Retrieve all traces for the session ordered by timestamp. Scan `request_preview` and `response_preview` for each turn to reconstruct the conversation. Confirm that turn 5's response is wrong, and check whether earlier turns look correct.

**2. Check if the error originated in an earlier turn.**

Turn 3's response contains a factual error that the user didn't challenge. Turn 4 builds on that incorrect information, and turn 5 compounds it. The root cause is in turn 3, not turn 5.

**3. Analyze the root-cause turn as a single trace.**

Fetch the full trace for turn 3 and analyze it — examine assessments (if any), walk the span tree, check retriever results, and correlate with code. The retriever returned an outdated document, causing the wrong answer.

**4. Recommendations.**

- Fix the retriever's data source to exclude or update outdated documents.
- Add per-turn assessments to detect errors before they propagate across the conversation.
- Consider implementing conversation-level error detection (e.g., checking consistency of answers across turns).
