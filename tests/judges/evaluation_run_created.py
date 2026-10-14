from __future__ import annotations

import os

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


def get_judges() -> list:
    eval_exp_id = os.environ["MLFLOW_EXPERIMENT_ID"]
    run_start_ms = int(os.environ["RUN_START_MS"])

    @scorer(name="evaluation-run-created")
    def evaluation_run_created(trace) -> Feedback:
        runs = mlflow.search_runs(
            experiment_ids=[eval_exp_id],
            filter_string=f"attributes.start_time >= {run_start_ms}",
            output_format="list",
        )
        client = MlflowClient()
        for run in runs:
            if run.info.status != "FINISHED":
                continue
            page_token = None
            while True:
                traces = client.search_traces(
                    experiment_ids=[eval_exp_id], run_id=run.info.run_id,
                    page_token=page_token,
                )
                for evaluated_trace in traces:
                    for assessment in evaluated_trace.info.assessments:
                        if (
                            assessment.feedback is not None
                            and assessment.feedback.value is not None
                            and assessment.error is None
                            and assessment.valid
                            and assessment.create_time_ms >= run_start_ms
                        ):
                            return Feedback(
                                value="yes",
                                rationale=f"Run {run.info.run_id} has persisted feedback "
                                f"{assessment.name} on trace {evaluated_trace.info.trace_id}",
                            )
                page_token = traces.token
                if not page_token:
                    break
        return Feedback(
            value="no",
            rationale="No finished run from this attempt has a valid, nonempty score.",
        )

    return [evaluation_run_created]
