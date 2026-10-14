from __future__ import annotations

import os

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import scorer


def get_judges() -> list:
    eval_exp_id = os.environ["MLFLOW_EXPERIMENT_ID"]
    run_start_ms = int(os.environ["RUN_START_MS"])

    @scorer(name="agent-trace-logged")
    def agent_trace_logged(trace) -> Feedback:
        traces = mlflow.search_traces(
            experiment_ids=[eval_exp_id],
            filter_string=f"trace.timestamp_ms >= {run_start_ms}",
            return_type="list",
        )
        for candidate in traces:
            if any(
                span.inputs is not None and span.outputs is not None
                for span in candidate.data.spans
            ):
                return Feedback(
                    value=CategoricalRating.YES,
                    rationale=f"Fresh trace {candidate.info.trace_id} has span inputs/outputs.",
                )
        return Feedback(
            value=CategoricalRating.NO,
            rationale="No fresh trace has persisted span inputs/outputs.",
        )

    return [agent_trace_logged]
