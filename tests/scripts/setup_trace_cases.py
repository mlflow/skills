#!/usr/bin/env python3
"""Seed real MLflow traces. The answer manifest is separate, but participant-accessible."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import mlflow
from mlflow.entities import AssessmentError, SpanStatus, SpanStatusCode

# Setup scripts run directly, with only scripts/ on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judges.trace_behavior import diagnosis_state

SOURCE = '''import mlflow

TIMEOUT_MS = 5
SERVICE_LATENCY_MS = 40

@mlflow.trace(span_type="TOOL")
def lookup_order(order_id):
    span = mlflow.get_current_active_span()
    span.set_inputs({"order_id": order_id, "timeout_ms": TIMEOUT_MS,
                     "service_latency_ms": SERVICE_LATENCY_MS})
    if TIMEOUT_MS < SERVICE_LATENCY_MS:
        raise TimeoutError(f"lookup_order deadline exceeded: {TIMEOUT_MS}ms < {SERVICE_LATENCY_MS}ms")
    return {"order_id": order_id, "status": "shipped"}

@mlflow.trace(span_type="AGENT")
def support_agent(order_id):
    return lookup_order(order_id)
'''


def metrics_probe(uri, experiment_id, start_ms, end_ms):
    payload = {
        "experiment_ids": [experiment_id], "view_type": 1,
        "metric_name": "total_tokens", "aggregations": [{"aggregation_type": 2}],
        "start_time_ms": start_ms, "end_time_ms": end_ms,
    }
    request = Request(uri.rstrip("/") + "/api/3.0/mlflow/traces/metrics",
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=15) as response:
            return {"http_status": response.status, "body": json.load(response)}
    except HTTPError as error:
        return {"http_status": error.code, "body": error.read().decode()}


def main():
    project = Path(os.environ["PROJECT_DIR"])
    work = Path(os.environ["WORK_DIR"])
    case = os.environ["TEST_CASE"]
    assert case in {"trace_retrieval", "trace_analysis", "session_analysis",
                    "trace_metrics", "diagnosis_only"}, case
    project.mkdir(parents=True, exist_ok=True)
    experiment_id = os.environ["MLFLOW_EXPERIMENT_ID"]
    mlflow.set_experiment(experiment_id=experiment_id)
    suffix = uuid.uuid4().hex[:10]
    session = "policy-" + suffix
    order_id = "order-" + suffix
    document_id = "policy-legacy-" + suffix
    records = {}

    def seed(key, *, session_id="other", environment="production", error=False,
             inputs=None, outputs=None, child_name="generate_answer", child_inputs=None,
             child_outputs=None, tokens=11):
        with mlflow.start_span(name="support_agent", span_type="AGENT") as root:
            mlflow.update_current_trace(
                metadata={"mlflow.trace.session": session_id},
                tags={"environment": environment, "tenant": suffix},
            )
            root.set_inputs(inputs or {"query": "Check order"})
            root.set_outputs(outputs or {"answer": "shipped"})
            with mlflow.start_span(name=child_name, span_type="TOOL") as child:
                child.set_inputs(child_inputs or {"order_id": order_id})
                child.set_outputs(child_outputs or {"status": "shipped"})
                if error:
                    child.set_status(SpanStatus(SpanStatusCode.ERROR,
                        "lookup_order deadline exceeded: 5ms < 40ms"))
                    root.set_status(SpanStatus(SpanStatusCode.ERROR, "tool call failed"))
            with mlflow.start_span(name="format_answer", span_type="LLM") as llm:
                llm.set_inputs({"context": "order service response"})
                llm.set_outputs({"text": "Unable to look up order" if error else "Done"})
                llm.set_attribute("mlflow.chat.tokenUsage", {
                    "input_tokens": tokens + int(suffix[:2], 16), "output_tokens": 7,
                    "total_tokens": tokens + int(suffix[:2], 16) + 7,
                })
        mlflow.flush_trace_async_logging()
        trace = mlflow.get_trace(root.trace_id)
        assert trace is not None and len(trace.data.spans) == 3
        records[key] = {"trace_id": root.trace_id, "child_span_id": child.span_id,
                        "time_ms": trace.info.request_time}
        time.sleep(0.003)  # Ensure distinct millisecond ordering for chat turns.
        return root.trace_id

    # Matching status/tag but outside the requested incident window.
    seed("old", error=True, child_name="lookup_order")
    start_ms = int(time.time() * 1000)
    timeout_inputs = {"order_id": order_id, "timeout_ms": 5, "service_latency_ms": 40}
    target = seed("timeout", error=True, child_name="lookup_order",
                  child_inputs=timeout_inputs, child_outputs={"error": "deadline exceeded"})
    seed("second_timeout", error=True, child_name="lookup_order", tokens=29)
    seed("staging_error", environment="staging", error=True, tokens=41)
    seed("production_ok", tokens=53)
    first = seed("turn1", session_id=session, inputs={"query": "What is the refund window?"},
                 outputs={"answer": "30 days"}, child_name="retrieve_policy",
                 child_outputs={"document_id": "current-" + suffix, "refund_days": 30})
    second = seed("turn2", session_id=session, inputs={"query": "Does that include gifts?"},
                  outputs={"answer": "Gifts may be refunded within 90 days"},
                  child_name="retrieve_policy", child_inputs={"query": "gift refund window"},
                  child_outputs={"document_id": document_id, "version": "retired",
                                 "refund_days": 90}, tokens=67)
    third = seed("turn3", session_id=session, inputs={"query": "So my 60 day old gift qualifies?"},
                 outputs={"answer": "Yes, it is within 90 days"}, child_name="reuse_history",
                 child_inputs={"prior_answer": "Gifts may be refunded within 90 days"},
                 child_outputs={"refund_days": 90}, tokens=79)
    seed("unrelated_session", session_id=session + "-unrelated", error=True, tokens=997)
    for trace_id, value, rationale in [
        (first, True, "Correct: current policy is 30 days including gifts."),
        (second, False, "Retired policy was used; current policy is 30 days including gifts."),
        (third, False, "Repeats the previous turn's incorrect 90-day policy."),
    ]:
        mlflow.log_feedback(trace_id=trace_id, name="correctness", value=value, rationale=rationale)
    mlflow.log_feedback(trace_id=first, name="session_coherence", value=False,
                        rationale="The later gift-policy answer contradicts current policy.",
                        metadata={"mlflow.trace.session": session})
    mlflow.log_feedback(trace_id=first, name="broken_judge",
                        error=AssessmentError("JUDGE_TIMEOUT", "Evaluator timed out; not an application failure"))
    end_ms = int(time.time() * 1000) + 1

    # Seed snapshots detect fabricated IDs and changed evidence; they are not a security boundary.
    snapshots = {}
    for row in records.values():
        trace = mlflow.get_trace(row["trace_id"])
        snapshots[trace.info.trace_id] = {
            "state": str(trace.info.state.value), "time_ms": trace.info.request_time,
            "session": trace.info.trace_metadata.get("mlflow.trace.session"),
            "tags": {k: trace.info.tags[k] for k in ["environment", "tenant"]},
            "tokens": trace.info.trace_metadata.get("mlflow.trace.tokenUsage"),
            "spans": [{"span_id": s.span_id, "name": s.name, "inputs": s.inputs,
                       "outputs": s.outputs, "status": s.status.status_code.name}
                      for s in trace.data.spans],
        }
    assert all(s["tokens"] for s in snapshots.values()), "Server did not persist token usage"
    expected = {
        "trace_retrieval": {"trace_ids": [trace_id for trace_id, snapshot in snapshots.items()
            if snapshot["state"] == "ERROR"
            and snapshot["tags"] == {"environment": "production", "tenant": suffix}
            and start_ms <= snapshot["time_ms"] <= end_ms]},
        "trace_analysis": {"trace_id": target, "failing_span_id": records["timeout"]["child_span_id"],
                           "root_cause": "tool_timeout", **timeout_inputs},
        "session_analysis": {"session_id": session, "ordered_trace_ids": [first, second, third],
                             "first_bad_trace_id": second, "propagated_trace_ids": [third],
                             "root_cause_span_id": records["turn2"]["child_span_id"],
                             "document_id": document_id, "observed_refund_days": 90,
                             "expected_refund_days": 30, "ignored_assessment_names": ["broken_judge"]},
    }
    expected["diagnosis_only"] = dict(expected["trace_analysis"])
    expected["diagnosis_only"]["source_file"] = "agent.py"
    in_window = [s for s in snapshots.values() if start_ms <= s["time_ms"] <= end_ms]
    usage = [json.loads(s["tokens"]) if isinstance(s["tokens"], str) else s["tokens"] for s in in_window]
    expected["trace_metrics"] = {
        "trace_count": len(in_window),
        "error_count": sum(s["state"] == "ERROR" for s in in_window),
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "total_tokens": sum(u["total_tokens"] for u in usage),
    }
    expected["trace_metrics"]["avg_total_tokens"] = expected["trace_metrics"]["total_tokens"] / len(in_window)
    diagnosis = (
        f"Explain that trace {target} failed in lookup_order because its 5 ms timeout was "
        "shorter than the 40 ms service latency. Identify the tool timeout as the cause, "
        "rather than blaming answer generation. Accept equivalent wording; no span ID is required."
    )
    response_requirements = {
        "trace_retrieval": (
            "Report exactly these failed production trace IDs for the requested tenant and time window, "
            f"with no missing or extra IDs: {expected['trace_retrieval']['trace_ids']}. Order does not matter."
        ),
        "trace_analysis": diagnosis,
        "diagnosis_only": diagnosis + " Locate the timeout setting in agent.py; explain without claiming to fix it.",
        "session_analysis": (
            f"Explain the conversation's progression: {first} correctly used the 30-day policy; "
            f"{second} was the first bad turn, retrieving retired document {document_id} and answering "
            f"90 days instead of 30; {third} propagated that mistake by reusing the earlier answer. "
            "Distinguish the broken_judge evaluator failure from the actual application error. "
            "Identify the relevant trace IDs exactly; accept equivalent explanations and any format."
        ),
        "trace_metrics": (
            f"Report {expected['trace_metrics']['total_tokens']} total tokens and "
            f"{expected['trace_metrics']['error_count']} errors in the requested time window. "
            f"If reporting a breakdown or averages, it must agree with {expected['trace_metrics']}. "
            "No HTTP status, method label, or particular output format is required."
        ),
    }
    (project / "agent.py").write_text(SOURCE)
    task = {"case": case, "start_time_ms": start_ms, "end_time_ms": end_ms,
            "tenant": suffix, "session_id": session, "trace_id": target,
            "current_policy": "Refund window is 30 days for all items including gifts."}
    (project / "task.json").write_text(json.dumps(task, indent=2))
    manifest = {"case": case, "experiment_id": experiment_id, "tracking_uri": mlflow.get_tracking_uri(),
                "start_ms": start_ms, "end_ms": end_ms, "records": records,
                "snapshots": snapshots, "expected": expected[case],
                "response_requirements": response_requirements[case],
                "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest()}
    if case == "diagnosis_only":
        manifest["diagnosis_state"] = diagnosis_state(experiment_id, project)
    if case == "trace_metrics":
        manifest["metrics_probe"] = metrics_probe(mlflow.get_tracking_uri(), experiment_id, start_ms, end_ms)
        print("Real metrics endpoint HTTP status:", manifest["metrics_probe"]["http_status"])
    (work / "trace_case_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Seeded {len(records)} real traces for {case} in experiment {experiment_id}")


if __name__ == "__main__":
    main()
