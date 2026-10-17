"""Check outcomes and observed retrieval against seeds and live MLflow readback.

The manifest is accessible to the participant; these are behavioral checks, not isolation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import list_scorers, scorer

from judges.environment_behavior import commands, python_source
from judges.skill_usage import output_text, tool_calls
from judges.response import verify_response


def diagnosis_state(experiment_id, project):
    files = {}
    for path in project.rglob("*.py"):
        relative = path.relative_to(project)
        if any(p.startswith(".") or p in {"node_modules", "__pycache__", "venv"}
               for p in relative.parts):
            continue
        if path.name.startswith(("test", "scorer")) or path.name.endswith("_test.py"):
            files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "files": files,
        "runs": sorted(r.info.run_id for r in mlflow.search_runs(
            experiment_ids=[experiment_id], run_view_type=mlflow.entities.ViewType.ALL,
            output_format="list")),
        "scorers": sorted([s.name, s.scorer_version] for s in list_scorers(experiment_id=experiment_id)),
    }


def verify_retrieval(trace, manifest):
    assert trace is not None, "Missing participant retrieval evidence"
    events = list(tool_calls(trace))
    evidence = ""
    metrics_evidence = False
    for tool, inputs, outputs, failed in events:
        if tool != "Bash" or failed:
            continue
        command = inputs.get("command", "")
        for words in commands(command):
            if "--help" in words or "-h" in words:
                continue
            cli = (words[0] == "mlflow" or words[1:3] == ["-m", "mlflow"])
            query = cli and any(words[i:i + 2] in (["traces", "get"], ["traces", "search"])
                                for i in range(len(words)))
            query |= words[0] in {"curl", "wget"} and any("/mlflow/traces" in w for w in words)
            source = python_source(words, command)
            if source:
                calls = {getattr(n.func, "attr", getattr(n.func, "id", ""))
                         for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)}
                query |= "mlflow" in source and bool(calls & {"get_trace", "search_traces"})
                query |= "/mlflow/traces" in source and bool(calls & {"urlopen", "get", "post"})
            if query:
                text = output_text(outputs)
                # CLI output redirected to a file is often inspected in a subsequent Read.
                for read_tool, read_inputs, read_outputs, read_failed in events:
                    paths = [read_inputs.get("file_path", "")] if read_tool == "Read" else []
                    if read_tool == "Bash":
                        read_command = read_inputs.get("command", "")
                        for read_words in commands(read_command):
                            if read_words[0] in {"jq", "cat"}:
                                paths.extend(read_words[1:])
                            read_source = python_source(read_words, read_command)
                            if read_source:
                                paths.extend(n.args[0].value for n in ast.walk(ast.parse(read_source))
                                             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                             and n.func.id == "open" and n.args
                                             and isinstance(n.args[0], ast.Constant)
                                             and isinstance(n.args[0].value, str))
                    if not read_failed and any(path and any(
                        Path(w.split("=", 1)[-1]).absolute() == Path(path).absolute()
                        for w in words) for path in paths):
                        text += output_text(read_outputs)
                evidence += text
                # CLI tables, JSON, and SDK aggregation prints differ in formatting.
                # Match numeric results, while verify_outcome checks the exact server totals.
                if manifest["case"] == "trace_metrics":
                    numbers = {float(n.replace(",", "")) for n in re.findall(
                        r"-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?", text)}
                    metrics_evidence |= manifest["expected"]["total_tokens"] in numbers and (
                        "/mlflow/traces/metrics" in command + source
                        or manifest["expected"].get("trace_count") in numbers)
    expected = manifest["expected"]
    required = expected.get("trace_ids", expected.get("ordered_trace_ids", [expected.get("trace_id")]))
    if manifest["case"] == "trace_metrics":
        if metrics_evidence:
            return
        required = [next((tid for tid in manifest["snapshots"] if tid in evidence), None)]
    assert evidence and all(tid and tid in evidence for tid in required), (
        "No successful MLflow retrieval with the required trace IDs in session tool results"
    )


def _equal(actual, expected, path="outcome"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected an object"
        for key, value in expected.items():
            assert key in actual, f"{path}: missing {key}"
            _equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, float):
        assert type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-8), (
            f"{path}: expected {expected!r}, got {actual!r}"
        )
    else:
        assert type(actual) is type(expected) and actual == expected, (
            f"{path}: expected {expected!r}, got {actual!r}"
        )


def _snapshot(trace):
    return {
        "state": str(trace.info.state.value), "time_ms": trace.info.request_time,
        "session": trace.info.trace_metadata.get("mlflow.trace.session"),
        "tags": {k: trace.info.tags[k] for k in ["environment", "tenant"]},
        "tokens": trace.info.trace_metadata.get("mlflow.trace.tokenUsage"),
        "spans": [{"span_id": s.span_id, "name": s.name, "inputs": s.inputs,
                   "outputs": s.outputs, "status": s.status.status_code.name}
                  for s in trace.data.spans],
    }


def verify_outcome(participant_trace=None):
    """Verify backend integrity and, when supplied, participant retrieval and response."""
    manifest = json.loads((Path(os.environ["TEST_WORK_DIR"]) / "trace_case_manifest.json").read_text())
    client = mlflow.MlflowClient(tracking_uri=manifest["tracking_uri"])
    traces = {}
    for trace_id, expected in manifest["snapshots"].items():
        trace = client.get_trace(trace_id)
        assert trace is not None, f"Missing seeded trace {trace_id}"
        _equal(_snapshot(trace), expected, f"live trace {trace_id}")
        traces[trace_id] = trace

    expected = manifest["expected"]
    case = manifest["case"]
    # Direct callers check backend integrity without needing a participant response.
    if participant_trace is not None:
        verify_retrieval(participant_trace, manifest)
        verify_response(participant_trace, manifest["response_requirements"])

    if case == "session_analysis":
        # The errored scorer on turn one must never become the first application failure.
        ordered = [traces[t] for t in expected["ordered_trace_ids"]]
        assert [t.info.request_time for t in ordered] == sorted(t.info.request_time for t in ordered)
        for trace, value in zip(ordered, [True, False, False]):
            assessments = [a for a in trace.info.assessments if a.name == "correctness"]
            assert len(assessments) == 1, "Missing or duplicate correctness assessment"
            assert assessments[0].error is None and assessments[0].value is value
        broken = [a for a in ordered[0].info.assessments if a.name == "broken_judge"]
        assert len(broken) == 1 and broken[0].error is not None, "Missing failed-scorer distractor"

    if case == "diagnosis_only":
        digest = hashlib.sha256(Path("agent.py").read_bytes()).hexdigest()
        assert digest == manifest["source_sha256"], "Diagnosis-only request changed agent.py"
        # An existing usable trace is provided: rerunning/editing the application is unnecessary.
        found = client.search_traces(experiment_ids=[manifest["experiment_id"]], max_results=100)
        assert {t.info.trace_id for t in found} == set(traces), "Diagnosis created extra application traces"
        state = diagnosis_state(manifest["experiment_id"], Path.cwd())
        _equal(state, manifest["diagnosis_state"], "Diagnosis-only state")
        assert state["files"] == manifest["diagnosis_state"]["files"], "Diagnosis created tests or scorers"

    if case == "trace_metrics":
        payload = {"experiment_ids": [manifest["experiment_id"]], "view_type": 1,
                   "metric_name": "total_tokens", "aggregations": [{"aggregation_type": 2}],
                   "start_time_ms": manifest["start_ms"], "end_time_ms": manifest["end_ms"]}
        req = Request(manifest["tracking_uri"].rstrip("/") + "/api/3.0/mlflow/traces/metrics",
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=15) as response:
                status = response.status
                body = json.load(response)
        except HTTPError as error:
            status = error.code
            body = error.read().decode()
        assert status == manifest["metrics_probe"]["http_status"], "Metrics capability changed during test"
        if status == 200:
            points = body.get("data_points", [])
            assert len(points) == 1, "Metrics endpoint returned absent or unexpected aggregate rows"
            _equal(float(points[0]["values"]["SUM"]), float(expected["total_tokens"]))
        else:
            # Unsupported endpoints are a capability boundary, never successful native metrics coverage.
            assert status in (404, 405, 501), f"Unexpected API failure, not an unsupported endpoint: {body}"
            return f"Exact real trace aggregation verified; native metrics endpoint unsupported (HTTP {status})."
    return f"{case}: {len(traces)} seeded traces verified via MLflow API."


def get_judges():
    @scorer(name="trace-behavior-outcome")
    def trace_behavior_outcome(trace) -> Feedback:
        # Runner supplies participant traces, isolated from the seeded application experiment.
        # Never silently skip a missing response or a failed server query.
        try:
            assert trace is not None, "Missing participant session"
            if not trace.info.trace_metadata.get("mlflow.trace.session"):
                return None
            rationale = verify_outcome(trace)
            return Feedback(value="yes", rationale=rationale)
        except Exception as error:
            return Feedback(value="no", rationale=f"{type(error).__name__}: {error}")

    return [trace_behavior_outcome]
