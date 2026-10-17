"""Exercise outcome judges against a real local MLflow store, without an LLM.

Run: uv run --with mlflow --with pytest pytest -q tests/test_artifact_judges.py
"""
import time

import mlflow
import pytest
from mlflow.genai.scorers import scorer

from judges.evaluation_run_created import get_judges as evaluation_judges


def verdict(factory):
    mlflow.flush_trace_async_logging()
    return factory()[0](trace=None).value


def test_empty_experiment(experiment):
    assert verdict(evaluation_judges) == "no"


def test_plain_run_with_unrelated_metric(experiment):
    with mlflow.start_run():
        mlflow.log_metric("accuracy", 1)
    assert verdict(evaluation_judges) == "no"


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
    from unittest.mock import Mock
    from test_environment_judges import session, bash

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
    assert verify_outcome()
    task = json.loads((project / "task.json").read_text())
    assert "output_fields" not in task and "output_contract" not in task
    participant = session(bash("mlflow traces search --experiment-id 1", " ".join(manifest["snapshots"])))
    response = Mock()
    monkeypatch.setattr("judges.trace_behavior.verify_response", response)
    assert verify_outcome(participant)
    response.assert_called_once_with(participant, manifest["response_requirements"])
    response.side_effect = AssertionError("Incorrect final response")
    with pytest.raises(AssertionError, match="Incorrect final response"):
        verify_outcome(participant)
    # The backend must remain intact even without an answer artifact.
    trace_id = manifest["records"]["timeout"]["trace_id"]
    mlflow.set_trace_tag(trace_id, "environment", "staging")
    with pytest.raises(AssertionError):
        verify_outcome()


@pytest.mark.parametrize("accepted,official", [(True, True), (False, True), (True, False)])
def test_documentation_grades_response_and_retrieved_evidence(experiment, tmp_path, monkeypatch, accepted, official):
    from unittest.mock import Mock
    from judges import environment_behavior

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_CASE", "docs")
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    (tmp_path / "environment_expected.json").write_text("{}")
    grade = Mock(side_effect=None if accepted else AssertionError("Unsupported example"))
    monkeypatch.setattr(environment_behavior, "verify_response", grade)
    url = "https://mlflow.org/docs/latest/ml/tracking/quickstart/"
    example = 'import mlflow\nwith mlflow.start_run():\n    mlflow.log_metric("loss", 0.5)'
    with mlflow.start_span(name="participant") as root:
        trace_id = root.trace_id
        mlflow.update_current_trace(metadata={"mlflow.trace.session": "docs-check"})
        root.set_outputs(f"Example:\n{example}\nSource: {url}")
        with mlflow.start_span(name="tool", attributes={"tool_name": "WebFetch"}) as span:
            span.set_inputs({"url": url if official else "https://example.com/docs"})
            span.set_outputs({"result": 'import mlflow\nmlflow.log_metric("accuracy", 0.9)'})
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id)
    assert environment_behavior.get_judges()[0](trace=trace).value == ("yes" if accepted and official else "no")
    if not official:
        grade.assert_not_called()
        return
    grade.assert_called_once()
    passed_trace, requirements = grade.call_args.args
    assert passed_trace.info.trace_id == trace_id
    assert passed_trace.data.spans[0].outputs == root.outputs
    assert passed_trace.data.spans[1].inputs == {"url": url}
    assert "runnable" in requirements and "MLflow documentation" in requirements
    assert not (tmp_path / "docs.json").exists()


def test_alignment_preserves_native_labels_and_splits(experiment, tmp_path, monkeypatch):
    import csv
    from scripts.setup_evaluation_cases import seed_alignment
    from judges.evaluation_behavior import verify_behavior

    project = tmp_path / "project"
    project.mkdir()
    expected = {"case": "alignment"}
    seed_alignment(project, expected, experiment.experiment_id)
    monkeypatch.chdir(project)
    pairs = [dict(p) for p in expected["pairs"]]

    def export():
        with (project / "labels.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["trace_id", "label", "split"])
            writer.writeheader()
            writer.writerows(pairs)

    export()
    assert verify_behavior(expected, tmp_path, experiment.experiment_id, 0)
    for key, wrong in [("label", not pairs[0]["label"]), ("split", "holdout")]:
        previous = pairs[0][key]
        pairs[0][key] = wrong
        export()
        with pytest.raises(AssertionError):
            verify_behavior(expected, tmp_path, experiment.experiment_id, 0)
        pairs[0][key] = previous


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


@pytest.mark.parametrize("tool,command,output,want,status", [
    ('Bash', 'pip show sagemaker-mlflow', 'WARNING: Package(s) not found: sagemaker-mlflow', 'yes', 'blocked'),
    ('Bash', "python -c 'import sagemaker_mlflow'", "ModuleNotFoundError: No module named 'sagemaker_mlflow'", 'yes', 'blocked'),
    ('Bash', 'pip show sagemaker-mlflow', 'Name: sagemaker-mlflow\nVersion: 0.2.0', 'no', 'blocked'),
    ('Bash', 'pip show sagemaker-mlflow; pip install sagemaker-mlflow', 'Package(s) not found: sagemaker-mlflow', 'no', 'blocked'),
    ('Bash', "pip show sagemaker-mlflow; echo 'pip install sagemaker-mlflow' > instructions.txt", 'Package(s) not found: sagemaker-mlflow', 'yes', 'blocked'),
    ('Bash', 'pip show unrelated', 'WARNING: Package(s) not found: unrelated', 'no', 'blocked'),
    ('Read', 'connection.json', 'Package(s) not found: sagemaker-mlflow', 'no', 'blocked'),
    ('Bash', 'pip show sagemaker-mlflow', 'Package(s) not found: sagemaker-mlflow', 'no', 'ready'),
])
def test_sagemaker_requires_observed_prerequisite_failure(
    experiment, tmp_path, monkeypatch, tool, command, output, want, status
):
    import json
    from judges import environment_behavior

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_CASE", "sagemaker_missing")
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    (tmp_path / "environment_expected.json").write_text(json.dumps({
        "files": {"backend.json": "original configuration"}, "backend": {"tracking_uri": mlflow.get_tracking_uri()},
        "experiment_ids": [e.experiment_id for e in mlflow.MlflowClient().search_experiments()],
    }))
    (tmp_path / "backend.json").write_text("original configuration")
    answer = "The connection is ready." if status == "ready" else "Connection blocked: missing SageMaker plugin."
    graded = []
    def grade(trace, requirements):
        graded.append(trace)
        assert trace.data.spans[0].outputs == answer
        assert "prerequisite failure" in requirements
        assert status != "ready", "Claimed readiness despite missing plugin"
    monkeypatch.setattr(environment_behavior, "verify_response", grade)
    with mlflow.start_span(name="participant") as root:
        trace_id = root.trace_id
        root.set_outputs(answer)
        mlflow.update_current_trace(metadata={"mlflow.trace.session": "prerequisite-check"})
        with mlflow.start_span(name="tool", attributes={"tool_name": tool}) as span:
            span.set_inputs({"command": command})
            span.set_outputs({"result": output})
    mlflow.flush_trace_async_logging()
    assert environment_behavior.get_judges()[0](trace=mlflow.get_trace(trace_id)).value == want
    if want == "yes" or status == "ready":
        assert len(graded) == 1
    else:
        assert not graded
    assert not (tmp_path / "connection.json").exists()


@pytest.mark.parametrize("called_skill,failed,matching_session,want", [
    ("sample-skill", False, True, "yes"),
    ("sample-skill", True, True, "no"),
    ("another-skill", False, True, "no"),
    ("sample-skill", False, False, "no"),
])
def test_skill_use_survives_compaction_without_accepting_wrong_or_failed_calls(
    experiment, tmp_path, monkeypatch, called_skill, failed, matching_session, want
):
    import json
    from judges.skill_usage import get_judges

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("EXPECTED_SKILLS", "sample-skill")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    skill = tmp_path / ".claude/skills/sample-skill/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: sample-skill\ndescription: Example\n---\n")
    logs = tmp_path / "claude-sessions"
    logs.mkdir()
    events = [
        {"message": {"content": [{"type": "tool_use", "id": "call-1", "name": "Skill", "input": {"skill": called_skill}}]}},
        {"message": {"content": [{"type": "tool_result", "tool_use_id": "call-1", "is_error": failed, "content": "Launching skill"}]}},
    ]
    (logs / (("current" if matching_session else "unrelated") + ".jsonl")).write_text(
        "\n".join(json.dumps(event) for event in events)
    )
    with mlflow.start_span(name="compacted-participant") as root:
        trace_id = root.trace_id
        mlflow.update_current_trace(metadata={"mlflow.trace.session": "current"})
    mlflow.flush_trace_async_logging()
    assert get_judges()[0](trace=mlflow.get_trace(trace_id)).value == want


def test_optional_inventory_timeout_does_not_abort_participant(tmp_path, monkeypatch):
    import json
    import subprocess
    from types import SimpleNamespace
    import test_skill

    def inventory(command, **kwargs):
        if command[1:3] == ["mcp", "list"]:
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, 0, stdout="available")

    executions = []
    def participant(command, **kwargs):
        executions.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(test_skill, "run_command", inventory)
    monkeypatch.setattr(test_skill.subprocess, "run", participant)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
    state = SimpleNamespace(work_dir=tmp_path, full_project_dir=tmp_path, repo_root=tmp_path,
                            experiment_id="1")
    config = SimpleNamespace(prompt="test", timeout_seconds=1, allowed_tools="Skill")
    assert test_skill.run_claude_code(config, state)
    assert len(executions) == 1 and executions[0][:2] == ["claude", "-p"]
    recorded = json.loads((tmp_path / "environment.json").read_text())
    assert recorded["mcp"]["error"] == "Inventory command timed out after 30 seconds"


@pytest.mark.parametrize("required,wanted", [(None, "primary,helper"), (["primary"], "primary")])
def test_required_skills_default_and_verifier_only_override(tmp_path, monkeypatch, required, wanted):
    import json
    from types import SimpleNamespace
    import test_skill
    from config import TestConfig, RuntimeState

    config = TestConfig(name="test", project_dir="project", setup_script="setup.py",
                        judges=[], skills=["primary", "helper"], prompt="Evaluate my app.",
                        required_skills=required)
    state = RuntimeState(work_dir=tmp_path, full_project_dir=tmp_path, repo_root=tmp_path)
    observed = {}
    def run(command, **kwargs):
        observed.update(kwargs["env"])
        return SimpleNamespace(stdout=json.dumps([{
            "pass": True, "scorer": "test", "trace_id": "test", "value": "yes"
        }]), stderr="")
    monkeypatch.delenv("EXPECTED_SKILLS", raising=False)
    monkeypatch.setattr(test_skill, "run_command", run)
    monkeypatch.setattr(test_skill.time, "sleep", lambda _: None)
    assert test_skill.verify_judges(config, state)
    assert observed["EXPECTED_SKILLS"] == wanted
    assert "EXPECTED_SKILLS" not in test_skill.claude_env()


def test_local_port_is_selected_but_external_server_needs_none(tmp_path):
    from config import TestConfig, RuntimeState
    from infra import check_prerequisites
    from utils import is_port_available

    (tmp_path / "setup.py").touch()
    config = TestConfig(name="test", project_dir="project", setup_script="setup.py",
                        judges=[], skills=[], prompt="Evaluate my app.")
    assert config.mlflow_port is None
    assert check_prerequisites(config, RuntimeState(repo_root=tmp_path))
    assert config.mlflow_port > 0 and is_port_available(config.mlflow_port)
    config.mlflow_port = None
    config.tracking_uri = "http://example.test:5000"
    state = RuntimeState(repo_root=tmp_path)
    assert check_prerequisites(config, state)
    assert state.use_external_server and config.mlflow_port is None
