"""Local regression controls for environment and investigation judges; no live agents."""
import json
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest
from mlflow.entities import SpanStatusCode
from mlflow.genai.scorers import scorer

from judges.environment_behavior import get_judges, verify_prerequisite, verify_training_data
from judges.trace_behavior import verify_outcome, verify_retrieval
from scripts.setup_trace_cases import main as seed_traces


def session(*events):
    return SimpleNamespace(
        info=SimpleNamespace(trace_metadata={"mlflow.trace.session": "participant"}),
        data=SimpleNamespace(spans=[SimpleNamespace(
            attributes={"tool_name": tool}, inputs=inputs, outputs=output,
            status=SimpleNamespace(status_code=SpanStatusCode.ERROR if failed else SpanStatusCode.OK),
        ) for tool, inputs, output, failed in events]),
    )


def bash(command, output, failed=False):
    return ("Bash", {"command": command}, output, failed)


@pytest.mark.parametrize("native", [True, False])
def test_dispatcher_requires_native_scorer(experiment, tmp_path, monkeypatch, native):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("TEST_CASE", "dispatcher")
    (tmp_path / "environment_expected.json").write_text("{}")
    (tmp_path / "scorers.py").write_text(
        "from mlflow.genai.scorers import scorer\n" + ("@scorer\n" if native else "")
        + "def answer_present(outputs):\n"
        "    return isinstance(outputs.get('answer'), str) and bool(outputs['answer'].strip())\n"
    )
    assert get_judges()[0](trace=session()).value == ("yes" if native else "no")


@pytest.mark.parametrize("command,passes", [
    ("pip show sagemaker-mlflow", True),
    ("/usr/bin/python3 -m pip show sagemaker-mlflow", True),
    ("uv pip show sagemaker-mlflow", True),
    ("/usr/bin/env /opt/bin/uv run python -m pip show sagemaker-mlflow", True),
    ("python -c 'import sagemaker_mlflow'", True),
    ("bash -lc 'python3 -c \"import sagemaker_mlflow\"'", True),
    ("echo 'No module named sagemaker_mlflow'", False),
    ("python -c 'print(\"import sagemaker_mlflow\")'", False),
    ("pip show unrelated", False),
    ("pip show sagemaker-mlflow; bash -lc 'pip install sagemaker-mlflow'", False),
    ("pip show sagemaker-mlflow; uv pip install sagemaker-mlflow", False),
    ("pip show sagemaker-mlflow; pip --disable-pip-version-check install sagemaker-mlflow", False),
    ("pip show sagemaker-mlflow; /usr/bin/python3 -m pip install sagemaker-mlflow", False),
    ("pip show sagemaker-mlflow; sh -c 'aws sso login'", False),
    ("pip show sagemaker-mlflow; echo 'pip install sagemaker-mlflow' > instructions.txt", True),
])
def test_prerequisite_commands(tmp_path, monkeypatch, command, passes):
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    trace = session(bash(command, "No module named sagemaker_mlflow; Package(s) not found: sagemaker-mlflow"))
    if passes:
        verify_prerequisite(trace, "sagemaker_missing", "unused")
    else:
        with pytest.raises(AssertionError):
            verify_prerequisite(trace, "sagemaker_missing", "unused")


@pytest.mark.parametrize("command,passes", [
    ("curl --max-time 2 http://127.0.0.1:9/health", True),
    ("/usr/bin/curl -sS http://127.0.0.1:9", True),
    ("curl http://127.0.0.1:8", False),
    ("echo 'http://127.0.0.1:9 Connection refused'", False),
    ("python -c 'print(\"requests.get(http://127.0.0.1:9) Connection refused\")'", False),
])
def test_backend_probe_is_tied_to_target(tmp_path, monkeypatch, command, passes):
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    trace = session(bash(command, "Connection refused", failed=True))
    if passes:
        verify_prerequisite(trace, "unavailable_backend", "http://127.0.0.1:9")
    else:
        with pytest.raises(AssertionError):
            verify_prerequisite(trace, "unavailable_backend", "http://127.0.0.1:9")


def test_python_prerequisite_scripts_and_heredoc(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    script = tmp_path / "scripts/verify_connection.py"
    script.parent.mkdir()
    script.write_text("import boto3\nboto3.client('sts').get_caller_identity()\n")
    for command in ["python scripts/verify_connection.py arn:example", f"python {script} arn:example",
                    "python - <<'PY'\nimport boto3\nboto3.client('sts').get_caller_identity()\nPY"]:
        verify_prerequisite(session(bash(command, "NoCredentialsError")), "sagemaker_missing", "unused")


def test_bundled_sagemaker_probe_output(tmp_path, monkeypatch, capsys):
    import builtins
    import runpy
    import sys

    script = Path(__file__).resolve().parents[2] / "mlflow-sagemaker/scripts/verify_connection.py"
    original = builtins.__import__
    def missing_plugin(name, *args, **kwargs):
        if name == "sagemaker_mlflow":
            raise ImportError(name)
        return original(name, *args, **kwargs)
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [str(script), "arn:aws:sagemaker:us-west-2:000000000000:mlflow-app/example"])
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", missing_plugin)
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 1
    output = capsys.readouterr().out
    verify_prerequisite(session(bash(f"python {script}", output)), "sagemaker_missing", "unused")
    with pytest.raises(AssertionError):
        verify_prerequisite(session(bash(f"echo '{output}'", output)), "sagemaker_missing", "unused")


@pytest.mark.parametrize("changed", [False, True])
def test_classic_ml_checks_actual_fit_data(tmp_path, monkeypatch, changed):
    monkeypatch.chdir(tmp_path)
    data = {"x": [[1], [2], [3], [4]], "y": [3, 5, 7, 9]}
    (tmp_path / "train.py").write_text(
        "from sklearn.linear_model import LinearRegression\n"
        f"x = {data['x']!r}\ny = {data['y']!r}\n"
        + ("x = [[5], [6], [7], [8]]\ny = [11, 13, 15, 17]\n" if changed else "")
        + "LinearRegression().fit(x, y)\n"
    )
    if changed:
        with pytest.raises(AssertionError, match="Training data"):
            verify_training_data(data)
    else:
        verify_training_data(data)


@pytest.mark.parametrize("command,passes", [
    ("mlflow traces get --trace-id tr-example", True),
    ("/opt/venv/bin/mlflow traces get -t tr-example", True),
    ("uv run mlflow --tracking-uri http://localhost:5000 traces search --experiment-id 3", True),
    ("python -m mlflow traces search --experiment-id 3", True),
    ("bash -lc 'mlflow traces get --trace-id tr-example'", True),
    ("python -c 'import mlflow; print(mlflow.get_trace(\"tr-example\"))'", True),
    ("curl -sS http://localhost:5000/api/3.0/mlflow/traces/tr-example", True),
    ("echo 'mlflow traces get --trace-id tr-example'", False),
    ("cat ../trace_case_manifest.json", False),
    ("mlflow traces get --help", False),
    ("python -c 'print(\"mlflow.get_trace(tr-example)\")'", False),
])
def test_retrieval_requires_query_and_result(tmp_path, monkeypatch, command, passes):
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    manifest = {"case": "trace_analysis", "expected": {"trace_id": "tr-example"}}
    trace = session(bash(command, '{"trace_id": "tr-example"}'))
    if passes:
        verify_retrieval(trace, manifest)
        with pytest.raises(AssertionError):
            verify_retrieval(session(bash(command, "unrelated output")), manifest)
        with pytest.raises(AssertionError):
            verify_retrieval(session(bash(command, "tr-example", failed=True)), manifest)
    else:
        with pytest.raises(AssertionError):
            verify_retrieval(trace, manifest)


def test_retrieval_script_and_redirect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    Path("fetch.py").write_text("import mlflow\nprint(mlflow.search_traces())\n")
    manifest = {"case": "session_analysis", "expected": {"ordered_trace_ids": ["tr-first", "tr-next"]}}
    verify_retrieval(session(bash("python ./fetch.py", "tr-first tr-next")), manifest)
    verify_retrieval(session(
        bash("mlflow traces search --experiment-id 1 > traces.json", ""),
        ("Read", {"file_path": str(tmp_path / "traces.json")}, "tr-first tr-next", False),
    ), manifest)
    with pytest.raises(AssertionError):
        verify_retrieval(session(bash("python ./fetch.py", "tr-first")), manifest)


def test_retrieval_uses_matching_compacted_session(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    logs = tmp_path / "claude-sessions"
    logs.mkdir()
    events = [
        {"message": {"content": [{"type": "tool_use", "id": "query", "name": "Bash",
                                  "input": {"command": "mlflow traces get --trace-id tr-example"}}]}},
        {"message": {"content": [{"type": "tool_result", "tool_use_id": "query",
                                  "content": "tr-example", "is_error": False}]}},
    ]
    path = logs / "participant.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events))
    manifest = {"case": "trace_analysis", "expected": {"trace_id": "tr-example"}}
    verify_retrieval(session(), manifest)
    path.rename(logs / "unrelated.jsonl")
    with pytest.raises(AssertionError):
        verify_retrieval(session(), manifest)


@pytest.mark.parametrize("output", [
    '{"data_points": [{"values": {"SUM": 123}}]}',
    "SUM\n---\n123.00", "123", "1.23e2",
])
def test_metrics_requires_native_result_or_trace_readback(tmp_path, monkeypatch, output):
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    manifest = {"case": "trace_metrics", "expected": {"total_tokens": 123},
                "snapshots": {"tr-example": {}}}
    verify_retrieval(session(bash("curl http://localhost:5000/api/3.0/mlflow/traces/metrics", output)), manifest)
    verify_retrieval(session(bash("mlflow traces search --experiment-id 1", "tr-example")), manifest)
    with pytest.raises(AssertionError):
        verify_retrieval(session(bash("echo 'SUM 123'", "SUM 123")), manifest)


def test_metrics_skill_script_and_sdk_aggregation(tmp_path, monkeypatch):
    import shutil
    script = Path(__file__).resolve().parents[2] / "mlflow-trace-metrics/scripts/fetch_metrics.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    Path("scripts").mkdir()
    shutil.copy(script, "scripts/fetch_metrics.py")
    manifest = {"case": "trace_metrics", "expected": {"total_tokens": 123, "trace_count": 3},
                "snapshots": {"tr-example": {}}}
    verify_retrieval(session(bash(
        "python ./scripts/fetch_metrics.py -s http://localhost:5000 -x 1 -m total_tokens -a SUM",
        "SUM\n---\n123.00",
    )), manifest)
    verify_retrieval(session(bash(
        "python -c 'import mlflow; rows = mlflow.search_traces(return_type=\"list\"); print(len(rows))'",
        '{"trace_count": 3, "total_tokens": 123}',
    )), manifest)
    verify_retrieval(session(
        bash("curl http://localhost:5000/api/3.0/mlflow/traces/metrics --output=metrics.json", ""),
        ("Read", {"file_path": str(tmp_path / "metrics.json")}, '{"SUM": 123}', False),
    ), manifest)


@pytest.mark.parametrize("addition", ["none", "test", "scorer_file", "run", "scorer"])
def test_diagnosis_preserves_native_and_recursive_state(experiment, tmp_path, monkeypatch, addition):
    project = tmp_path / "project"
    for key, value in {"PROJECT_DIR": project, "WORK_DIR": tmp_path, "TEST_WORK_DIR": tmp_path,
                       "TEST_CASE": "diagnosis_only"}.items():
        monkeypatch.setenv(key, str(value))
    seed_traces()
    monkeypatch.chdir(project)
    manifest = json.loads((tmp_path / "trace_case_manifest.json").read_text())
    trace_id = manifest["expected"]["trace_id"]
    participant = session(bash(f"mlflow traces get --trace-id {trace_id}", trace_id))
    assert verify_outcome()
    verify_retrieval(participant, manifest)
    if addition in {"test", "scorer_file"}:
        path = Path("tests/test_order.py" if addition == "test" else "quality/scorers.py")
        path.parent.mkdir()
        path.write_text("# unintended diagnosis artifact\n")
    elif addition == "run":
        with mlflow.start_run():
            mlflow.log_metric("quality", 1)
    elif addition == "scorer":
        monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
        @scorer
        def quality(outputs):
            return True
        quality.register(experiment_id=experiment.experiment_id)
    if addition == "none":
        assert verify_outcome()
        with pytest.raises(AssertionError, match="retrieval"):
            verify_retrieval(session(bash("cat ../trace_case_manifest.json", trace_id)), manifest)
    else:
        with pytest.raises(AssertionError, match="Diagnosis"):
            verify_outcome()


def test_classic_ml_setup_and_judge(experiment, tmp_path, monkeypatch):
    import runpy
    import sys
    from utils import run_command

    setup = Path(__file__).parents[1] / "scripts/setup_environment_cases.py"
    project = tmp_path / "project"
    for key, value in {"PROJECT_DIR": project, "WORK_DIR": tmp_path, "TEST_WORK_DIR": tmp_path,
                       "TEST_CASE": "classic_ml", "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri()}.items():
        monkeypatch.setenv(key, str(value))
    runpy.run_path(str(setup))
    monkeypatch.chdir(project)
    source = Path("train.py").read_text()
    Path("train.py").write_text(
        source + "\nimport mlflow, mlflow.sklearn\n"
        "with mlflow.start_run():\n"
        "    mlflow.log_param('fit_intercept', True)\n"
        "    mlflow.log_metric('mse', mean_squared_error(y, model.predict(x)))\n"
        "    mlflow.sklearn.log_model(model, name='model', pip_requirements=[])\n"
    )
    run_command([sys.executable, "train.py"], timeout=60)
    result = get_judges()[0](trace=session())
    assert result.value == "yes", result.rationale


@pytest.mark.parametrize("missing,passes", [
    ("sagemaker-mlflow", True),
    ("boto3, mlflow, sagemaker-mlflow", True),
    ("sagemaker-mlflow, boto3", True),
    ("boto3, mlflow", False),
    ("sagemaker-mlflow-extra", False),
])
def test_multi_package_prerequisite_diagnostic(tmp_path, monkeypatch, missing, passes):
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    output = f"WARNING: Package(s) not found: {missing}"
    trace = session(bash("pip show sagemaker-mlflow mlflow boto3", output))
    if passes:
        verify_prerequisite(trace, "sagemaker_missing", "unused")
    else:
        with pytest.raises(AssertionError):
            verify_prerequisite(trace, "sagemaker_missing", "unused")
    with pytest.raises(AssertionError):
        verify_prerequisite(session(bash(f"echo '{output}'", output)), "sagemaker_missing", "unused")


@pytest.mark.parametrize("path,passes", [("/tmp/trace.json", True), ("/tmp/other.json", False)])
def test_retrieval_redirect_inspected_with_python(tmp_path, monkeypatch, path, passes):
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    manifest = {"case": "trace_analysis", "expected": {"trace_id": "tr-example"}}
    trace = session(
        bash("mlflow traces get --trace-id tr-example > /tmp/trace.json", ""),
        bash(f"""python -I -c "import json; print(json.load(open('{path}')))" """, "tr-example"),
    )
    if passes:
        verify_retrieval(trace, manifest)
    else:
        with pytest.raises(AssertionError):
            verify_retrieval(trace, manifest)


@pytest.mark.parametrize("reader", ["jq .", "cat"])
@pytest.mark.parametrize("path,passes", [("/tmp/traces.json", True), ("/tmp/other.json", False)])
def test_retrieval_redirect_inspected_with_shell_reader(tmp_path, monkeypatch, reader, path, passes):
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    manifest = {"case": "trace_analysis", "expected": {"trace_id": "tr-example"}}
    trace = session(bash("mlflow traces search --experiment-id 1 > /tmp/traces.json", ""),
                    bash(f"{reader} {path}", "tr-example"))
    if passes:
        verify_retrieval(trace, manifest)
    else:
        with pytest.raises(AssertionError):
            verify_retrieval(trace, manifest)


@pytest.mark.parametrize("value,error,passes", [
    ("yes", None, True), ("no", None, False), (None, None, False),
    ("yes", "provider failed", False),
])
def test_response_verdict(monkeypatch, value, error, passes):
    from judges.response import verify_response

    trace = SimpleNamespace(data=SimpleNamespace(spans=[
        SimpleNamespace(parent_id=None, outputs="Connection failed: missing credentials.")]))

    def make_judge(**kwargs):
        assert "{{ outputs }}" in kwargs["instructions"]
        assert "Explain the failed connection" in kwargs["instructions"]

        def judge(**args):
            assert args["outputs"] == trace.data.spans[0].outputs
            return SimpleNamespace(value=value, error=error, rationale="test assessment")

        return judge

    monkeypatch.setattr("judges.response.make_judge", make_judge)
    if passes:
        verify_response(trace, "Explain the failed connection")
    else:
        with pytest.raises(AssertionError):
            verify_response(trace, "Explain the failed connection")
