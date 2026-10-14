"""Exact behavioral outcomes checked against private seeds and live MLflow readback."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


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


def verify_outcome():
    """Raises on missing/incorrect answers or missing/modified server evidence."""
    manifest = json.loads((Path(os.environ["TEST_WORK_DIR"]) / "trace_case_manifest.json").read_text())
    outcome = json.loads(Path("outcome.json").read_text())
    client = mlflow.MlflowClient(tracking_uri=manifest["tracking_uri"])
    traces = {}
    for trace_id, expected in manifest["snapshots"].items():
        trace = client.get_trace(trace_id)
        assert trace is not None, f"Missing seeded trace {trace_id}"
        _equal(_snapshot(trace), expected, f"live trace {trace_id}")
        traces[trace_id] = trace

    expected = manifest["expected"]
    case = manifest["case"]
    if case == "trace_retrieval":
        ids = outcome.get("trace_ids")
        assert isinstance(ids, list) and len(ids) == len(set(ids)), "Missing IDs or duplicate rows"
        outcome = {**outcome, "trace_ids": sorted(ids)}
        expected = {**expected, "trace_ids": sorted(expected["trace_ids"])}
    _equal(outcome, expected)

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
        assert not list(Path.cwd().glob("test*.py")), "Diagnosis-only request created tests"

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
        _equal(outcome.get("metrics_endpoint_http_status"), status)
        if status == 200:
            _equal(outcome.get("aggregation_method"), "metrics_endpoint")
            points = body.get("data_points", [])
            assert len(points) == 1, "Metrics endpoint returned absent or unexpected aggregate rows"
            _equal(float(points[0]["values"]["SUM"]), float(expected["total_tokens"]))
        else:
            # Unsupported endpoints are a capability boundary, never successful native metrics coverage.
            assert status in (404, 405, 501), f"Unexpected API failure, not an unsupported endpoint: {body}"
            _equal(outcome.get("aggregation_method"), "search_traces_fallback")
            return f"Exact real trace aggregation verified; native metrics endpoint unsupported (HTTP {status})."
    return f"{case}: exact outcome and {len(traces)} seeded traces verified via MLflow API."


def get_judges():
    @scorer(name="trace-behavior-outcome")
    def trace_behavior_outcome(trace) -> Feedback:
        # Runner supplies participant traces, isolated from the seeded application experiment.
        # Never silently skip a missing outcome or a failed server query.
        try:
            rationale = verify_outcome()
            return Feedback(value="yes", rationale=rationale)
        except Exception as error:
            return Feedback(value="no", rationale=f"{type(error).__name__}: {error}")

    return [trace_behavior_outcome]
