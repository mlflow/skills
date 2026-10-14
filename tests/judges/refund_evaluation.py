"""Check both row-level outcomes for the deterministic refund fixture."""
import os

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


def get_judges():
    experiment_id = os.environ["MLFLOW_EXPERIMENT_ID"]
    start = int(os.environ["RUN_START_MS"])

    @scorer(name="refund-evaluation")
    def refund_evaluation(trace):
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"attributes.start_time >= {start}",
            output_format="list",
        )
        for run in runs:
            if run.info.status != "FINISHED":
                continue
            results = set()
            for candidate in mlflow.search_traces(
                experiment_ids=[experiment_id], run_id=run.info.run_id, return_type="list"
            ):
                for span in candidate.data.spans:
                    inputs = span.inputs
                    if not isinstance(inputs, dict):
                        continue
                    for message, output, score in [
                        ("I need a refund", "sales", False),
                        ("hello", "general", True),
                    ]:
                        if message not in inputs.values() or span.outputs != output:
                            continue
                        for assessment in candidate.info.assessments:
                            if (
                                assessment.feedback is not None
                                and assessment.error is None
                                and assessment.valid
                                and assessment.feedback.value == score
                                and assessment.create_time_ms >= start
                            ):
                                results.add((assessment.name, message))
            if any(
                (name, "I need a refund") in results and (name, "hello") in results
                for name, _ in results
            ):
                return Feedback(value="yes", rationale=f"Both expected scores in {run.info.run_id}")
        return Feedback(value="no", rationale="Missing correct failing/passing refund fixture scores.")

    return [refund_evaluation]
