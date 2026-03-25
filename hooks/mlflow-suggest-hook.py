#!/usr/bin/env python3
"""UserPromptSubmit hook: detects MLflow usage and suggests relevant skill."""
import json
import sys
import re

def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)
    prompt = data.get("prompt", "").lower()

    suggestions = []

    if any(k in prompt for k in ["trace", "tracing", "autolog", "span", "instrument"]):
        suggestions.append("💡 Use the `instrumenting-with-mlflow-tracing` skill to add MLflow tracing.")

    if any(k in prompt for k in ["evaluat", "scorer", "judge", "dataset", "assess", "improve quality"]):
        suggestions.append("💡 Use the `agent-evaluation` skill to evaluate your agent with MLflow.")

    if any(k in prompt for k in ["trace id", "debug trace", "why did", "what went wrong", "analyze trace"]):
        suggestions.append("💡 Use the `analyze-mlflow-trace` skill to debug this trace.")

    if any(k in prompt for k in ["session", "conversation", "chat history", "multi-turn"]):
        suggestions.append("💡 Use the `analyze-mlflow-chat-session` skill to analyze chat sessions.")

    if any(k in prompt for k in ["search traces", "find traces", "filter traces", "get trace"]):
        suggestions.append("💡 Use the `retrieving-mlflow-traces` skill to search/filter traces.")

    if any(k in prompt for k in ["metrics", "token usage", "latency", "cost", "usage trend"]):
        suggestions.append("💡 Use the `querying-mlflow-metrics` skill to fetch aggregated metrics.")

    if any(k in prompt for k in ["get started", "set up mlflow", "onboard", "quickstart"]):
        suggestions.append("💡 Use the `mlflow-onboarding` skill to get started with MLflow.")

    if any(k in prompt for k in ["mlflow docs", "mlflow api", "how to use mlflow"]):
        suggestions.append("💡 Use the `searching-mlflow-docs` skill to search MLflow documentation.")

    if suggestions:
        print("\n".join(suggestions))

if __name__ == "__main__":
    main()
