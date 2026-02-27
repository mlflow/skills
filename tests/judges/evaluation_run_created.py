"""Judge: check that at least one evaluation run was created."""

from __future__ import annotations

import os

from mlflow import MlflowClient
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


def get_judges() -> list:
    """Return a scorer that checks an evaluation run exists in the experiment."""
    eval_exp_id = os.environ["MLFLOW_EXPERIMENT_ID"]

    @scorer(name="evaluation-run-created")
    def evaluation_run_created(trace) -> Feedback:
        client = MlflowClient()
        runs = client.search_runs(experiment_ids=[eval_exp_id], max_results=1)
        if len(runs) >= 1:
            return Feedback(
                value="yes",
                rationale=f"Found {len(runs)} run(s) in experiment {eval_exp_id}",
            )
        return Feedback(
            value="no",
            rationale=f"No runs found in experiment {eval_exp_id}",
        )

    return [evaluation_run_created]
