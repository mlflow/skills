# MLflow Skills — Claude Code Plugin

This plugin packages the MLflow Skills collection for distribution via the Anthropic Claude Code marketplace.

## What's Included

**9 skills** covering the full MLflow agent improvement loop:

| Skill | Purpose |
|-------|---------|
| `mlflow-agent` | Dispatcher — recommended entry point; routes to the right skill automatically |
| `agent-evaluation` | Set up datasets, scorers, and evaluation runs |
| `instrumenting-with-mlflow-tracing` | Instrument Python/TypeScript code with MLflow Tracing |
| `analyze-mlflow-trace` | Debug issues by examining spans and correlating with code |
| `analyze-mlflow-chat-session` | Analyze multi-turn chat session traces |
| `retrieving-mlflow-traces` | Query and filter traces from the MLflow backend |
| `querying-mlflow-metrics` | Query token usage, latency, and quality metrics |
| `mlflow-onboarding` | Get started with MLflow from scratch |
| `searching-mlflow-docs` | Search MLflow documentation for answers |

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
