"""Judges for the tracing skill test."""

from __future__ import annotations

import os
from typing import Literal

from mlflow import MlflowClient
from mlflow.entities import Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import scorer


def get_judges() -> list:
    """Return judges to evaluate Claude Code traces for the tracing test.

    Judges are run on the CC traces after Claude Code finishes. Each judge
    receives a trace and returns a yes/no Feedback.
    """
    eval_exp_id = os.environ["MLFLOW_EXPERIMENT_ID"]

    # LLM judge: check that the tracing skill was invoked
    tracing_skill_invoked = make_judge(
        name="tracing-skill-invoked",
        instructions="Check that the tracing skill is invoked in the {{ trace }}.",
        feedback_value_type=Literal["yes", "no"],
    )

    # LLM judge: check that the agent ran the instrumented code
    agent_ran_code = make_judge(
        name="agent-ran-instrumented-code",
        instructions=(
            "Examine the {{ trace }} and determine whether the agent ran the "
            "application or agent code after adding MLflow tracing instrumentation. "
            "Look for evidence that the agent executed the instrumented program "
            "(e.g., running a CLI command, calling an entry point, executing a script). "
            "Return 'yes' if the agent ran the code after instrumenting it, 'no' otherwise."
        ),
        feedback_value_type=Literal["yes", "no"],
    )

    # Programmatic scorer: check that at least one trace was logged
    # to the evaluation experiment by the agent
    @scorer(name="agent-trace-logged")
    def agent_trace_logged(trace) -> Feedback:
        client = MlflowClient()
        traces = client.search_traces(
            experiment_ids=[eval_exp_id], max_results=1
        )
        if len(traces) > 0:
            return Feedback(
                value="yes",
                rationale=f"Found {len(traces)} trace(s) in experiment {eval_exp_id}",
            )
        return Feedback(
            value="no",
            rationale=f"No traces found in experiment {eval_exp_id}",
        )

    return [tracing_skill_invoked, agent_ran_code, agent_trace_logged]
