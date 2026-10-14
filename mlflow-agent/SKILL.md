---
name: mlflow-agent
description: >
  Master dispatcher for all MLflow workflows. Use this skill when the user wants to do
  anything with MLflow — tracing, evaluating, debugging, or improving an agent.
  Routes to the right MLflow sub-skill automatically.
  Triggers on: "use mlflow", "help with mlflow", "mlflow agent", "add mlflow to my project",
  "trace my agent", "evaluate my agent", or any MLflow task without a specific skill in mind.
disable-model-invocation: true
---

# MLflow Agent

Master dispatcher for MLflow workflows. Reads user intent and invokes the right sub-skill.

## Trigger

Use when the user wants to do anything with MLflow but hasn't specified which skill to use.

## Process

1. Read the user's request and identify intent
2. Map to the appropriate skill:
   - Tracing / instrumentation → `instrumenting-with-mlflow-tracing`
   - Debug or explain an agent failure → `debug-agent`
   - Fix or change an agent's behavior → `fix-agent-issue`
   - Run evaluation / compare results → `agent-evaluation`
   - Design or align criteria / scorers → `build-a-scorer`
   - Debug a trace → `analyze-mlflow-trace`
   - Debug a chat session → `analyze-mlflow-chat-session`
   - Search traces → `retrieving-mlflow-traces`
   - Metrics / costs → `querying-mlflow-metrics`
   - Getting started → `mlflow-onboarding`
   - Docs / API questions → `searching-mlflow-docs`
3. If intent is unclear, ask ONE clarifying question, then dispatch
4. Invoke the matched skill using the Skill tool

## Workflow ownership

- **Setup:** `mlflow-onboarding` and the `sagemaker-mlflow` backend adapter own missing backend configuration.
- **Tracing:** instrumentation, retrieval, trace/session analysis, diagnosis and operational metrics.
- **Evaluation:** evaluation execution, scorer design/alignment (`build-a-scorer`) and verified behavior changes (`fix-agent-issue`).

These groups retain their existing entry points. A diagnosis-only request stays in tracing;
a request to change behavior uses the evaluation improvement loop. Reuse trace findings,
criteria and configuration across handoffs instead of restarting discovery.

## Key Rules

- Never do the work yourself — always dispatch to the appropriate sub-skill
- One clarifying question maximum before dispatching
- If evaluation requires new trace evidence, dispatch tracing first; reuse existing traces when available.
- Do not route an already-instrumented debugging request back to tracing setup
- Reuse working backend and experiment configuration, including configuration supplied by Databricks tooling. Invoke setup only for missing configuration or a requested backend change.
- If the request spans multiple skills, run only the necessary prerequisites before the requested workflow.
