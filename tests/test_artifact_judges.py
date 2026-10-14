"""Exercise outcome judges against a real local MLflow store, without an LLM.

Run: uv run --with mlflow --with pytest pytest -q tests/test_artifact_judges.py
"""
import time

import mlflow
import pytest
from mlflow.genai.scorers import scorer

from judges.refund_evaluation import get_judges as refund_judges
from judges.agent_trace_logged import get_judges as trace_judges
from judges.evaluation_run_created import get_judges as evaluation_judges


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    experiment_id = mlflow.create_experiment("application", artifact_location=str(tmp_path / "artifacts"))
    experiment = mlflow.set_experiment(experiment_id=experiment_id)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", experiment.experiment_id)
    monkeypatch.setenv("RUN_START_MS", str(int(time.time() * 1000)))
    yield experiment
    mlflow.end_run()
    mlflow.set_tracking_uri(previous_uri)


def verdict(factory):
    mlflow.flush_trace_async_logging()
    return factory()[0](trace=None).value


def test_empty_experiment(experiment):
    assert verdict(trace_judges) == "no"
    assert verdict(evaluation_judges) == "no"


def test_plain_run_with_unrelated_metric(experiment):
    with mlflow.start_run():
        mlflow.log_metric("accuracy", 1)
    assert verdict(evaluation_judges) == "no"


def test_trace_requires_data_and_current_attempt(experiment, monkeypatch):
    with mlflow.start_span(name="empty"):
        pass
    assert verdict(trace_judges) == "no"
    with mlflow.start_span(name="application") as span:
        span.set_inputs({"question": "hello"})
        span.set_outputs({"answer": "hello"})
    assert verdict(trace_judges) == "yes"
    monkeypatch.setenv("RUN_START_MS", str(int(time.time() * 1000) + 1))
    assert verdict(trace_judges) == "no"


@pytest.mark.parametrize("score", [0, False, 1, True])
def test_real_evaluation_accepts_zero_and_false(experiment, monkeypatch, score):
    @scorer
    def quality(outputs):
        return score

    mlflow.genai.evaluate(
        data=[{"inputs": {"question": "hello"}}],
        predict_fn=lambda question: question,
        scorers=[quality],
    )
    assert verdict(evaluation_judges) == "yes"
    assert verdict(trace_judges) == "yes"
    monkeypatch.setenv("RUN_START_MS", str(int(time.time() * 1000) + 1))
    assert verdict(evaluation_judges) == "no"


@pytest.mark.parametrize("fails", [False, True])
def test_skipped_or_failed_scorer_is_not_a_result(experiment, fails):
    @scorer
    def quality(outputs):
        if fails:
            raise ValueError("scoring failed")
        return None

    mlflow.genai.evaluate(
        data=[{"inputs": {"question": "hello"}}],
        predict_fn=lambda question: question,
        scorers=[quality],
    )
    assert verdict(evaluation_judges) == "no"


@pytest.mark.parametrize("mode", ["correct", "reversed", "missing-row", "missing-score"])
def test_refund_fixture_requires_both_correct_scores(experiment, mode):
    @scorer
    def route_correctness(inputs, outputs):
        if mode == "missing-score":
            return None
        expected = "support" if "refund" in inputs["message"] else "general"
        result = outputs == expected
        return not result if mode == "reversed" else result

    data = [{"inputs": {"message": message}} for message in ["I need a refund", "hello"]]
    if mode == "missing-row":
        data.pop()
    mlflow.genai.evaluate(
        data=data,
        predict_fn=lambda message: "sales" if "refund" in message else "general",
        scorers=[route_correctness],
    )
    assert verdict(refund_judges) == ("yes" if mode == "correct" else "no")
