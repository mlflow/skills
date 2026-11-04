#!/usr/bin/env python3
"""UserPromptSubmit hook: detects MLflow usage and suggests relevant skill."""
import json
import sys
import re

def mentions(prompt, patterns):
    # Whole words only, so "traceback", "retrace", and "spanish" do not count as "trace" or "span".
    return re.search(rf"\b(?:{'|'.join(patterns)})\b", prompt) is not None

def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)
    prompt = data.get("prompt", "").lower()

    suggestions = []

    if mentions(prompt, [r"trac(?:e|es|ed|ing)", r"autolog\w*", r"spans?", r"instrument\w*"]):
        suggestions.append("💡 Use the `mlflow-trace-instrumentation` skill to add MLflow tracing.")

    if mentions(prompt, [r"evaluat\w*", r"scorers?", r"judges?", r"datasets?", r"assess\w*", "improve quality"]):
        suggestions.append("💡 Use the `mlflow-evaluation` skill to evaluate your agent with MLflow.")

    if mentions(prompt, [r"trace ids?", r"debug traces?", "why did", "what went wrong", r"analyze traces?"]):
        suggestions.append("💡 Use the `mlflow-trace-analysis` skill to debug this trace.")

    if mentions(prompt, ["debug my agent", "debug this agent", r"agent failures?", "root cause my agent"]):
        suggestions.append("💡 Use the `mlflow-trace-analysis` skill to root-cause the failure from MLflow traces.")

    if mentions(prompt, ["fix my agent", "fix this agent", "fix the agent", "improve my agent"]):
        suggestions.append("💡 Use the `mlflow-agent-improvement` skill to fix the agent and regression-test the change.")

    if mentions(prompt, [r"sessions?", r"conversations?", "chat history", "multi-turn"]):
        suggestions.append("💡 Use the `mlflow-trace-analysis` skill to analyze chat sessions.")

    if mentions(prompt, ["search traces", "find traces", "filter traces", r"get traces?"]):
        suggestions.append("💡 Use the `mlflow-trace-retrieval` skill to search/filter traces.")

    if mentions(prompt, ["metrics", "token usage", "latency", r"costs?", r"usage trends?"]):
        suggestions.append("💡 Use the `mlflow-trace-metrics` skill to fetch aggregated metrics.")

    if mentions(prompt, ["get started", "set up mlflow", r"onboard\w*", "quickstart"]):
        suggestions.append("💡 Use the `mlflow-onboarding` skill to get started with MLflow.")

    if mentions(prompt, ["sagemaker"]):
        suggestions.append("💡 Use the `mlflow-sagemaker` skill to connect to SageMaker Managed MLflow.")

    if mentions(prompt, ["mlflow docs", "mlflow api", "how to use mlflow"]):
        suggestions.append("💡 Use the `mlflow-docs` skill to search MLflow documentation.")

    if suggestions:
        by_skill = {}
        for suggestion in suggestions:
            skill = re.search(r"`([^`]+)`", suggestion).group(1)
            by_skill.setdefault(skill, suggestion)
        # Windows pipes default to the ANSI code page, which cannot encode the emoji.
        sys.stdout.reconfigure(encoding="utf-8")
        print("\n".join(by_skill.values()))

if __name__ == "__main__":
    main()
