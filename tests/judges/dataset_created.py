from __future__ import annotations

import os

from mlflow import MlflowClient
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


def get_judges() -> list:
    eval_exp_id = os.environ["MLFLOW_EXPERIMENT_ID"]

    @scorer(name="dataset-created")
    def dataset_created(trace) -> Feedback:
        client = MlflowClient()
        datasets = client.search_datasets(experiment_ids=[eval_exp_id])
        if datasets:
            return Feedback(
                value="yes",
                rationale=f"Found {len(datasets)} dataset(s) in experiment {eval_exp_id}",
            )
        return Feedback(
            value="no",
            rationale=f"No datasets found in experiment {eval_exp_id}",
        )

    return [dataset_created]
