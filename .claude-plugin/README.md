# MLflow Skills — Claude Code Plugin

This plugin packages the MLflow Skills collection for distribution via the Anthropic Claude Code marketplace.

## What's Included

Skills covering the full MLflow agent improvement loop:

| Skill | Purpose |
|-------|---------|
| `mlflow-agent` | Dispatcher — recommended entry point; routes to the right skill automatically |
| `mlflow-onboarding` | Get started with MLflow from scratch |
| `mlflow-docs` | Search MLflow documentation for answers |
| `mlflow-sagemaker` | Connect to SageMaker Managed MLflow |
| `mlflow-tracing` | Instrument Python/TypeScript code with MLflow Tracing |
| `mlflow-trace-retrieval` | Query and filter traces from the MLflow backend |
| `mlflow-trace-analysis` | Analyze traces, agent failures, and multi-turn sessions |
| `mlflow-trace-metrics` | Query token usage, latency, and quality metrics |
| `mlflow-evaluation` | Design scorers and run GenAI evaluation |
| `mlflow-agent-improvement` | Implement and regression-test agent behavior changes |

**1 hook:**

| Hook | Purpose |
|------|---------|
| `hooks/mlflow-suggest-hook.py` | Auto-suggests relevant MLflow skills based on what you're working on |

## Installation

### Via Claude Code Marketplace (recommended)

Search for "MLflow" in the Claude Code marketplace and click Install.

### Manual (clone)

```bash
git clone https://github.com/mlflow/skills
```

Then add the repo path to your Claude Code settings under `Skills > Local paths`.

## Full Documentation

See the [main README](../README.md) for complete usage examples, skill descriptions, and configuration options.
