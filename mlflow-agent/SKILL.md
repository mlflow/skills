---
name: mlflow-agent
description: Route an explicit broad MLflow request to setup, tracing or evaluation skills. Use when the user has not selected a specific MLflow workflow.
disable-model-invocation: true
---

# Choose an MLflow workflow

Match the user's requested outcome to one owner. Invoke the installed skill using the
host's skill mechanism, or read its `SKILL.md` when direct invocation is unavailable.
If a skill is not installed, say so instead of claiming a handoff. Ask one clarifying
question only when the request cannot be routed from existing context.

| Ownership | Requested outcome | Skill |
|---|---|---|
| Setup | Start or resolve missing backend configuration | `mlflow-onboarding` |
| Setup | Connect a SageMaker MLflow ARN | `sagemaker-mlflow` |
| Tracing | Add or repair instrumentation | `instrumenting-with-mlflow-tracing` |
| Tracing | Retrieve/filter traces | `retrieving-mlflow-traces` |
| Tracing | Explain a specific trace | `analyze-mlflow-trace` |
| Tracing | Explain a multi-turn session | `analyze-mlflow-chat-session` |
| Tracing | Diagnose an agent failure without changing code | `debug-agent` |
| Tracing | Aggregate latency, tokens, costs or quality | `querying-mlflow-metrics` |
| Evaluation | Run evaluation or compare results | `agent-evaluation` |
| Evaluation | Design criteria, build scorers or align judges | `build-a-scorer` |
| Evaluation | Fix or change behavior and verify the change | `fix-agent-issue` |
| All | Look up documentation or APIs | `searching-mlflow-docs` |

Reuse configuration, trace evidence and criteria across handoffs. Working Databricks
configuration needs no new setup. Load only the prerequisites needed for the task:
existing traces can support evaluation/diagnosis without reinstrumentation, and designing
a scorer does not require a full evaluation campaign. Diagnosis stops before edits;
a requested fix proceeds through the evaluation improvement loop.
