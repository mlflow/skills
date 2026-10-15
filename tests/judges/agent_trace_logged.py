from __future__ import annotations

import os

from mlflow import MlflowClient
from mlflow.entities import Feedback
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import scorer


def get_judges() -> list:
    eval_exp_id = os.environ["MLFLOW_EXPERIMENT_ID"]

    @scorer(name="agent-trace-logged")
    def agent_trace_logged(trace) -> Feedback:
        client = MlflowClient()
        traces = client.search_traces(
            experiment_ids=[eval_exp_id], max_results=1
        )
        if traces:
            return Feedback(
                value=CategoricalRating.YES,
                rationale=f"Found {len(traces)} trace(s) in experiment {eval_exp_id}",
            )
        return Feedback(
            value=CategoricalRating.NO,
            rationale=f"No traces found in experiment {eval_exp_id}",
        )

    return [agent_trace_logged]
