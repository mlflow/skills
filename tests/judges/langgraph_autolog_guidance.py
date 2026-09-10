from __future__ import annotations

from typing import Literal

from mlflow.genai.judges import make_judge


def get_judges() -> list:
    return [
        make_judge(
            name="langgraph-autolog-first",
            instructions=(
                "Inspect the agent's work in the {{ trace }} to judge how it added MLflow "
                "tracing to the LangGraph agent in agent.py.\n\n"
                "Answer 'yes' only if BOTH conditions hold:\n"
                "1. The agent enabled framework tracing with a single mlflow.langchain.autolog() "
                "call (autolog covers LangGraph graph execution, nodes, tools, and model calls).\n"
                "2. The agent did NOT add @mlflow.trace decorators or mlflow.start_span() calls to "
                "the graph, its nodes (call_model, call_tool), the tool function, or the model call "
                "for the sole purpose of recreating spans that autolog already produces.\n\n"
                "It is still 'yes' if the agent added a single deliberate application-boundary "
                "decorator or manual span around app-specific work that LangGraph does not capture "
                "on its own (for example a top-level run() wrapper, or a span around a custom "
                "retrieval or post-processing step it wrote). That is allowed and expected.\n\n"
                "Answer 'no' if the agent decorated LangGraph nodes, the tool, or the model call "
                "with @mlflow.trace in addition to autolog, which produces duplicate spans, or if "
                "it never enabled autolog for the framework."
            ),
            feedback_value_type=Literal["yes", "no"],
        ),
    ]
