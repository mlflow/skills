from __future__ import annotations

import os

from mlflow.entities import Feedback
from mlflow.genai.scorers import list_scorers, scorer


def get_judges() -> list:
    eval_exp_id = os.environ["MLFLOW_EXPERIMENT_ID"]

    @scorer(name="scorer-registered")
    def scorer_registered(trace) -> Feedback:
        scorers = list_scorers(experiment_id=eval_exp_id)
        if scorers:
            names = [s.name for s in scorers]
            return Feedback(value="yes", rationale=f"Found scorers: {names}")
        return Feedback(
            value="no",
            rationale=f"No scorers in experiment {eval_exp_id}",
        )

    return [scorer_registered]
