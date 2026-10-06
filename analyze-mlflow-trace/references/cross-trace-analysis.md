# Compare multiple traces

Use `retrieving-mlflow-traces` to select a bounded sample from one experiment and time window. For Unity Catalog, include a `trace.timestamp_ms` condition so the search does not scan the full table. Then read `trace-structure.md` before interpreting fields. Aggregate trends with `querying-mlflow-metrics`; use this guide to explain which spans caused them.

## Find the failure pattern

Group traces by status, agent version, prompt URI, session, and relevant tags. Inspect assessments and rationales first: a technically successful trace can still have a bad answer, and an assessment error can come from the scorer rather than the agent. Compare failing traces with successful traces for the same input class. Check retrieval results, tool selection, LLM inputs/outputs, retries, and handoffs in execution order.

For multi-agent systems, identify each agent or stage from actual span names and attributes; do not assume a fixed framework schema. Compare component outcomes across traces to locate where errors first diverge. Record representative trace IDs, the affected component, the observed failure, and the expected behavior in a short report. Use `fix-agent-issue` to turn a recurring failure into a regression test.

## Profile performance

For each trace, measure root latency and child span durations. Compare by span type and normalized component name, excluding wrapper spans that merely enclose child work. Account for parallel spans: summing all child durations can exceed wall-clock time. Find repeated calls and gaps between spans to distinguish model latency, tool latency, retries, and orchestration overhead. Examine token usage by LLM span to identify excessive context or output length; use `querying-mlflow-metrics` to confirm prevalence across traffic.

## Preserve findings

When the backend supports trace assessments or tags, attach the diagnosed issue and expected behavior to the relevant trace. Verify that the write succeeded and that a dataset builder can filter the marked traces. If only read access is available, keep the trace IDs and findings in the evaluation report rather than claiming the trace was annotated. Follow `agent-evaluation/references/advanced-evaluation.md` to curate those traces into a regression dataset.
