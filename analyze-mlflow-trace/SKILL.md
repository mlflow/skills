---
name: analyzing-mlflow-trace
description: Analyzes a single MLflow trace to answer a user query about it. Use when the user provides a trace ID and asks to debug, investigate, find issues, root-cause errors, understand behavior, or analyze quality. Triggers on "analyze this trace", "what went wrong with this trace", "debug trace", "investigate trace", "why did this trace fail", "root cause this trace".
---

# Analyzing a Single MLflow Trace

## Trace Structure

A trace captures the full execution of an AI/ML application as a tree of **spans**. Each span represents one operation (LLM call, tool invocation, retrieval step, etc.) and records its inputs, outputs, timing, and status. Traces also carry **assessments** — feedback from humans or LLM judges about quality.

It is recommended to read [references/trace-structure.md](references/trace-structure.md) before analyzing a trace — it covers the complete data model, all fields and types, analysis guidance, and OpenTelemetry compatibility notes.

## Key Commands

```bash
# Fetch full trace (recommended - parse the JSON output directly)
mlflow traces get --trace-id <ID>
```

**Prefer fetching the full trace and parsing the JSON directly** rather than using `--extract-fields`. The `--extract-fields` flag has limited support for nested span data (e.g., span inputs/outputs may return empty objects). Fetch the complete trace once and parse it as needed — traces can be 100KB+ for complex agent executions, which is normal.

## JSON Structure Notes

In the raw trace JSON, span inputs and outputs are stored as serialized strings inside `attributes`, not as top-level span fields. For MLflow-instrumented traces, these are `attributes."mlflow.spanInputs"` and `attributes."mlflow.spanOutputs"`. Traces from third-party OpenTelemetry clients may use different attribute names (e.g., GenAI Semantic Conventions, OpenInference, or custom keys) — check the raw `attributes` dict to find the equivalent fields. Assessment names appear as `assessment_name`, and feedback values are nested under `feedback.value`.

## Analysis Insights

- **`state: OK` does not mean correct output.** It only means no unhandled exception. Check assessments for quality signals, and if none exist, analyze the trace's inputs, outputs, and intermediate span data directly for issues.
- **Assessment `rationale` is the fastest path to understanding an issue** when assessments are present. It often describes the problem in plain language before you need to examine any spans.
- **Assessments tell you *what* went wrong; spans tell you *where*.** If assessments exist, use feedback/expectations to form a hypothesis, then confirm it in the span tree. If no assessments exist, examine span inputs/outputs to identify where the execution diverged from expected behavior.
- **Assessment errors are not trace errors.** If an assessment has an `error` field, it means the scorer or judge that evaluated the trace failed — not that the trace itself has a problem. The trace may be perfectly fine; the assessment's `value` is just unreliable. This can happen when a scorer crashes (e.g., timed out, returned unparseable output) or when a scorer was applied to a trace type it wasn't designed for (e.g., a retrieval relevance scorer applied to a trace with no retrieval steps). The latter is a scorer configuration issue, not a trace issue.
- **Span timing reveals performance issues.** Gaps between parent and child spans indicate overhead; repeated span names suggest retries; compare individual span durations to find bottlenecks.
- **Token usage explains latency and cost.** Look for token usage in trace metadata (e.g., `mlflow.trace.tokenUsage`) or span attributes (e.g., `mlflow.chat.tokenUsage`). Not all clients set these — check the raw `attributes` dict for equivalent fields. Spikes in input tokens may indicate prompt injection or overly large context.

## Codebase Correlation

MLflow Tracing captures inputs, outputs, and metadata from different parts of an application's call stack. By correlating trace contents with the source code, issues can be root-caused more precisely than from the trace alone.

- **Span names map to functions.** Span names typically match the function decorated with `@mlflow.trace` or wrapped in `mlflow.start_span()`. For autologged spans (LangChain, OpenAI, etc.), names follow framework conventions instead (e.g., `ChatOpenAI`, `RetrievalQA`).
- **The span tree mirrors the call stack.** If span A is the parent of span B, then function A called function B.
- **Span inputs/outputs correspond to function parameters/return values.** Comparing them against the code logic reveals whether the function behaved as designed or produced an unexpected result.
- **The trace shows *what happened*; the code shows *why*.** A retriever returning irrelevant results might trace back to a faulty similarity threshold. Incorrect span inputs might reveal wrong model parameters or missing environment variables set in code.

## Example: Investigating a Wrong Answer

A user reports that their customer support agent gave an incorrect answer for the query "What is our refund policy?" There are no assessments on the trace.

**1. Fetch the trace and check high-level signals.**

The trace has `state: OK` — no crash occurred. No assessments are present, so examine the trace's inputs and outputs directly. The `response_preview` says *"Our shipping policy states that orders are delivered within 3-5 business days..."* — this answers a different question than what was asked.

**2. Examine spans to locate the problem.**

The span tree shows:

```
customer_support_agent (AGENT) — OK
├── plan_action (LLM) — OK
│   outputs: {"tool_call": "search_knowledge_base", "args": {"query": "refund policy"}}
├── search_knowledge_base (TOOL) — OK
│   inputs: {"query": "refund policy"}
│   outputs: [{"doc": "Shipping takes 3-5 business days...", "score": 0.82}]
├── generate_response (LLM) — OK
│   inputs: {"messages": [..., {"role": "user", "content": "Context: Shipping takes 3-5 business days..."}]}
│   outputs: {"content": "Our shipping policy states..."}
```

The agent correctly decided to search for "refund policy," but the `search_knowledge_base` tool returned a shipping document. The LLM then faithfully answered using the wrong context. The problem is in the tool's retrieval, not the agent's reasoning or the LLM's generation.

**3. Correlate with the codebase.**

The span `search_knowledge_base` maps to a function in the application code. Investigating reveals the vector index was built from only the shipping FAQ — the refund policy documents were never indexed.

**4. Recommendations.**

- Re-index the knowledge base to include refund policy documents.
- Add a retrieval relevance scorer to detect when retrieved context doesn't match the query topic.
- Consider adding expectation assessments with correct answers for common queries to enable regression testing.
