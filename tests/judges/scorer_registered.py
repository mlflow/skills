"""Judge: check that at least one scorer was registered."""

from __future__ import annotations

import os

from mlflow import MlflowClient
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


def get_judges() -> list:
    """Return a scorer that checks a scorer was registered in the experiment."""
    eval_exp_id = os.environ["MLFLOW_EXPERIMENT_ID"]

    @scorer(name="scorer-registered")
    def scorer_registered(trace) -> Feedback:
        client = MlflowClient()
        runs = client.search_runs(experiment_ids=[eval_exp_id], max_results=1)
        runs_with_metrics = [r for r in runs if r.data.metrics]
        if runs_with_metrics:
            metrics = list(runs_with_metrics[0].data.metrics.keys())
            return Feedback(
                value="yes",
                rationale=f"Found evaluation run with metrics: {metrics}",
            )
        if runs:
            return Feedback(
                value="yes",
                rationale=f"Found {len(runs)} evaluation run(s) in experiment {eval_exp_id}",
            )
        return Feedback(
            value="no",
            rationale=f"No evaluation runs found in experiment {eval_exp_id}",
        )

    return [scorer_registered]
