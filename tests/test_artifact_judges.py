"""Exercise outcome judges against a real local MLflow store, without an LLM.

Run: uv run --with mlflow --with pytest pytest -q tests/test_artifact_judges.py
"""
import time

import mlflow
import pytest
from mlflow.genai.scorers import scorer

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


@pytest.mark.parametrize("case", ["trace_retrieval", "trace_analysis", "session_analysis", "diagnosis_only"])
def test_investigation_oracles_require_real_evidence(experiment, tmp_path, monkeypatch, case):
    import json
    from scripts.setup_trace_cases import main as seed
    from judges.trace_behavior import verify_outcome

    project = tmp_path / "project"
    monkeypatch.setenv("PROJECT_DIR", str(project))
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("TEST_CASE", case)
    seed()
    monkeypatch.chdir(project)
    manifest = json.loads((tmp_path / "trace_case_manifest.json").read_text())
    expected = manifest["expected"]
    if case == "trace_retrieval":
        # The unrelated chat session still matches the requested status/tag/time
        # filter. This assertion catches the original two-result false negative.
        assert len(expected["trace_ids"]) == 3
        assert manifest["records"]["unrelated_session"]["trace_id"] in expected["trace_ids"]
    output = project / "outcome.json"
    output.write_text(json.dumps(expected))
    assert verify_outcome()
    for key in ("failing_span_id", "root_cause_span_id"):
        if key in expected:
            import base64
            encoded = dict(expected, **{key: base64.b64encode(bytes.fromhex(expected[key])).decode()})
            output.write_text(json.dumps(encoded))
            assert verify_outcome()
    output.write_text("{}")
    with pytest.raises(AssertionError):
        verify_outcome()
    output.write_text(json.dumps(expected))
    # Even a copied correct answer must fail if the underlying evidence changed.
    trace_id = manifest["records"]["timeout"]["trace_id"]
    mlflow.set_trace_tag(trace_id, "environment", "staging")
    with pytest.raises(AssertionError):
        verify_outcome()


def test_documentation_accepts_complete_cells_but_rejects_invented_code():
    from judges.environment_behavior import copied_documentation_blocks

    source = 'python\n```\nimport mlflow\n\n```\nNext cell\n```python\nmlflow.log_metric("loss", 1)\n```'
    example = 'import mlflow\nmlflow.log_metric("loss", 1)'
    assert copied_documentation_blocks(example, source)
    assert not copied_documentation_blocks(example.replace('"loss"', '"invented"'), source)
    assert not copied_documentation_blocks(example + '\nmlflow.log_metric("extra", 2)', source)
    assert not copied_documentation_blocks('', source)


def test_alignment_pairs_allow_provenance_but_reject_wrong_labels_and_splits(experiment, tmp_path, monkeypatch):
    import json
    from scripts.setup_evaluation_cases import seed_alignment
    from judges.evaluation_behavior import verify_behavior

    project = tmp_path / "project"
    project.mkdir()
    expected = {"case": "alignment"}
    seed_alignment(project, expected, experiment.experiment_id)
    monkeypatch.chdir(project)
    outcome = dict(expected, criterion="total_correct", status="blocked", prerequisites=["embedding model"])
    outcome["pairs"] = [dict(pair, source_id="order-expert") for pair in expected["pairs"]]
    output = project / "outcome.json"
    output.write_text(json.dumps(outcome))
    assert verify_behavior(expected, tmp_path, experiment.experiment_id, 0)
    for key, wrong in [("label", not outcome["pairs"][0]["label"]), ("split", "holdout")]:
        previous = outcome["pairs"][0][key]
        outcome["pairs"][0][key] = wrong
        output.write_text(json.dumps(outcome))
        with pytest.raises(AssertionError):
            verify_behavior(expected, tmp_path, experiment.experiment_id, 0)
        outcome["pairs"][0][key] = previous


def test_port_probe_rejects_active_listener():
    import socket
    from utils import is_port_available

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        assert not is_port_available(port)
    assert is_port_available(port)


@pytest.mark.parametrize("tool,command,output,want", [
    ("Bash", "pip show sagemaker-mlflow", "WARNING: Package(s) not found: sagemaker-mlflow", "yes"),
    ("Bash", "python -c 'import sagemaker_mlflow'", "ModuleNotFoundError: No module named 'sagemaker_mlflow'", "yes"),
    ("Bash", "pip show sagemaker-mlflow", "Name: sagemaker-mlflow\nVersion: 0.2.0", "no"),
    ("Bash", "pip show unrelated", "WARNING: Package(s) not found: unrelated", "no"),
    ("Read", "connection.json", "Package(s) not found: sagemaker-mlflow", "no"),
])
def test_sagemaker_requires_observed_prerequisite_failure(
    experiment, tmp_path, monkeypatch, tool, command, output, want
):
    import json
    from judges.environment_behavior import get_judges

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_CASE", "sagemaker_missing")
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    (tmp_path / "environment_expected.json").write_text(json.dumps({
        "files": {}, "backend": {"tracking_uri": mlflow.get_tracking_uri()},
        "experiment_ids": [e.experiment_id for e in mlflow.MlflowClient().search_experiments()],
    }))
    (tmp_path / "connection.json").write_text(json.dumps({"status": "blocked", "reason": "missing plugin"}))
    with mlflow.start_span(name="participant") as root:
        trace_id = root.trace_id
        mlflow.update_current_trace(metadata={"mlflow.trace.session": "prerequisite-check"})
        with mlflow.start_span(name="tool", attributes={"tool_name": tool}) as span:
            span.set_inputs({"command": command})
            span.set_outputs({"result": output})
    mlflow.flush_trace_async_logging()
    assert get_judges()[0](trace=mlflow.get_trace(trace_id)).value == want
