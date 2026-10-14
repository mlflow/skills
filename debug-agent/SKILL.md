---
name: debug-agent
description: Diagnose AI agent failures from MLflow traces without code changes, tests, scorers or evaluation runs. Prefer an existing matching trace; reproduce only when no usable trace exists.
---

# Debug an agent

Implement a fix only when requested.

## Inputs

Reuse reported/expected behavior, tracing configuration, tracking URI, experiment and trace ID or incident time/session. Do not reinstall or reinstrument working tracing.

## Workflow

1. Find a matching existing trace; do not force reproduction of production-only or externally dependent failures. UC searches need incident-time bounds even with small result limits.
2. Only without a usable trace, reproduce once through the normal instrumented entry point when safe and possible, then retrieve the persisted trace. Otherwise state the missing evidence and stop.
3. Read assessments first, then actual span inputs/outputs, errors, model calls, retrieval and tools. Distinguish HUMAN reports from automated judgments and scorer errors; failed scoring does not prove application failure, and OK does not prove quality.
4. Follow the first divergence and downstream effects, including preceding session turns/history where relevant. Correlate decisive trace evidence with source/configuration before claiming a root cause.
5. Log complaints as HUMAN feedback only against an identified trace and when authorized. Do not edit code or create tests, scorers or evaluations for diagnosis alone.

## Verify

Report observed versus expected behavior, trace/span evidence, cause or hypothesis, confidence, gaps and smallest likely remedy. For requested implementation, pass diagnosis and trace ID to `fix-agent-issue` for regression coverage before editing.

## Read when needed

When installed: `retrieving-mlflow-traces` for lookup; `analyzing-mlflow-trace` (directory `analyze-mlflow-trace`) for anatomy; `analyzing-mlflow-session` (directory `analyze-mlflow-chat-session`) for propagation. For missing instrumentation, explain the limitation; use `instrumenting-with-mlflow-tracing` when setup/repair is requested.
