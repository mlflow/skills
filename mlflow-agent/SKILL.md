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

## Step 1: Identify the workflow

Use the user's request to choose the relevant skill. Ask at most one clarifying question
if the intent is unclear.

## Step 2: Dispatch

- Tracing / instrumentation → `mlflow-tracing`
- Debug or explain an agent failure → `mlflow-trace-analysis`
- Fix or change an agent's behavior → `mlflow-agent-improvement`
- Evaluation / scoring → `mlflow-evaluation`
- Debug a trace → `mlflow-trace-analysis`
- Debug a chat session → `mlflow-trace-analysis`
- Search traces → `mlflow-trace-retrieval`
- Metrics / costs → `mlflow-trace-metrics`
- Getting started → `mlflow-onboarding`
- Docs / API questions → `mlflow-docs`

Invoke the matched skill using the Skill tool.

## Key Rules

- Never do the work yourself — always dispatch to the appropriate sub-skill
- One clarifying question maximum before dispatching
- If the user says "evaluate AND trace", dispatch tracing first, then evaluation
- Do not route an already-instrumented debugging request back to tracing setup
- If the user's request spans multiple skills, handle them in logical order (setup → instrument → evaluate)
