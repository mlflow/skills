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
   - Evaluation / scoring → `agent-evaluation`
   - Debug a trace → `analyze-mlflow-trace`
   - Debug a chat session → `analyze-mlflow-chat-session`
   - Search traces → `retrieving-mlflow-traces`
   - Metrics / costs → `querying-mlflow-metrics`
   - Getting started → `mlflow-onboarding`
   - Docs / API questions → `searching-mlflow-docs`
3. If intent is unclear, ask ONE clarifying question, then dispatch
4. Invoke the matched skill using the Skill tool

## Key Rules

- Never do the work yourself — always dispatch to the appropriate sub-skill
- One clarifying question maximum before dispatching
- If the user says "evaluate AND trace", dispatch tracing first, then evaluation
- If the user's request spans multiple skills, handle them in logical order (setup → instrument → evaluate)
