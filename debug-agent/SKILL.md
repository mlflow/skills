---
name: debug-agent
description: Debug, investigate, or explain an AI agent's behavior using MLflow traces and source code. Diagnose only; do not change code or create tests, scorers or evaluation runs. Prefer an existing trace; reproduce only when none is usable.
---

# Debug an agent

Reuse reported/expected behavior, tracing configuration, tracking URI, experiment and trace ID or incident time/session. Do not reinstall or reinstrument working tracing.

## Prerequisites

Before searching for evidence, read [trace retrieval](../retrieving-mlflow-traces/SKILL.md). If that skill is not installed, use the [search guide](https://mlflow.org/docs/latest/genai/tracing/search-traces.md) and installed CLI help.

For detailed span diagnosis, read [trace analysis](../analyze-mlflow-trace/SKILL.md); for earlier-turn propagation, read [session analysis](../analyze-mlflow-chat-session/SKILL.md), when installed.

## Step 1: Find an existing trace

Find a matching existing trace; do not force reproduction of production-only or externally dependent failures. UC searches need incident-time bounds even with small result limits.

## Step 2: Reproduce only if needed

Only without a usable trace, reproduce once through the normal instrumented entry point when safe and possible, then retrieve the persisted trace. Otherwise state the missing evidence and stop.

## Step 3: Read the trace

Read assessments first, then actual span inputs/outputs, errors, model calls, retrieval and tools. Distinguish HUMAN reports from automated judgments and scorer errors; failed scoring does not prove application failure, and OK does not prove quality.

## Step 4: Locate the first divergence

Follow the first divergence and downstream effects, including preceding session turns/history where relevant. Correlate decisive trace evidence with source/configuration before claiming a root cause.

## Step 5: Record authorized feedback

Log complaints as HUMAN feedback only against an identified trace and when authorized. Do not edit code or create tests, scorers or evaluations for diagnosis alone.

## Verification

Report observed versus expected behavior, trace/span evidence, cause or hypothesis, confidence, gaps and smallest likely remedy. For requested implementation, pass diagnosis and trace ID to `fix-agent-issue` for regression coverage before editing.

## References

For missing instrumentation, explain the limitation; read [instrumentation](../instrumenting-with-mlflow-tracing/SKILL.md) when setup/repair is requested and the skill is installed.
