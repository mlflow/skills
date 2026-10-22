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
        suggestions.append("💡 Use the `mlflow-tracing` skill to add MLflow tracing.")

    if any(k in prompt for k in ["evaluat", "scorer", "judge", "dataset", "assess", "improve quality"]):
        suggestions.append("💡 Use the `mlflow-evaluation` skill to evaluate your agent with MLflow.")

    if any(k in prompt for k in ["trace id", "debug trace", "why did", "what went wrong", "analyze trace"]):
        suggestions.append("💡 Use the `mlflow-trace-analysis` skill to debug this trace.")

    if any(k in prompt for k in ["debug my agent", "debug this agent", "agent failure", "root cause my agent"]):
        suggestions.append("💡 Use the `mlflow-trace-analysis` skill to root-cause the failure from MLflow traces.")

    if any(k in prompt for k in ["session", "conversation", "chat history", "multi-turn"]):
        suggestions.append("💡 Use the `mlflow-trace-analysis` skill to analyze chat sessions.")

    if any(k in prompt for k in ["search traces", "find traces", "filter traces", "get trace"]):
        suggestions.append("💡 Use the `mlflow-trace-retrieval` skill to search/filter traces.")

    if any(k in prompt for k in ["metrics", "token usage", "latency", "cost", "usage trend"]):
        suggestions.append("💡 Use the `mlflow-trace-metrics` skill to fetch aggregated metrics.")

    if any(k in prompt for k in ["get started", "set up mlflow", "onboard", "quickstart"]):
        suggestions.append("💡 Use the `mlflow-onboarding` skill to get started with MLflow.")

    if any(k in prompt for k in ["mlflow docs", "mlflow api", "how to use mlflow"]):
        suggestions.append("💡 Use the `mlflow-docs` skill to search MLflow documentation.")

    if suggestions:
        by_skill = {}
        for suggestion in suggestions:
            skill = re.search(r"`([^`]+)`", suggestion).group(1)
            by_skill.setdefault(skill, suggestion)
        print("\n".join(by_skill.values()))

if __name__ == "__main__":
    main()
