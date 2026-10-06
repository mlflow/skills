# Scorers for multi-step agents

Use this only after the user has agreed to the criteria in `../SKILL.md`. Prefer a built-in scorer or a small `@scorer` function for each observable failure. Check the installed MLflow scorer signature and required columns before adapting these patterns.

## Component and tool-use checks

- **Tool selection:** compare the requested task with tool calls in the trace. Distinguish a missing required tool call from an appropriate direct answer. Use an `@scorer` code check when the expected tool is explicit in `expectations`; use a judge when correctness depends on semantic intent.
- **Stage accuracy:** score retrieval, planning, tool execution, or response generation only when the corresponding spans are instrumented. Put expected stage behavior in the dataset record. Report which stage failed instead of collapsing all stages into one opaque overall score.
- **Latency:** measure span durations and name the slow component. This is an operational metric; do not ask an LLM judge to infer duration from text. Pair it with a quality scorer so a faster but worse agent does not look like an improvement.
- **Multi-agent workflows:** identify agent or handoff spans and score one criterion per stage. Avoid a scorer factory unless several stages truly share identical logic; a simple named scorer is easier to inspect and register.

Use a class-based scorer only when the same criterion needs persistent configuration or reusable state; for a simple predicate, `@scorer` is clearer. Conditional scoring should return an explicit skipped/not-applicable outcome for requests where the criterion does not apply, rather than counting them as successes. If one execution produces several distinct assessments, give each a stable, unique name so reports and monitoring do not conflate them.

A custom scorer can return a boolean, numeric value, or `Feedback` with rationale, subject to the installed API. Keep assessment names unique when returning multiple feedbacks. Check aggregations supported by the installed version instead of assuming `p50`, `p99`, or `sum` are accepted. Before registering a scorer for production monitoring, test serialization; imported dependencies or complex annotations may need to live inside the scorer function. See `agent-evaluation/references/online-monitoring.md` for the registration and sampling lifecycle.
