from __future__ import annotations

import importlib
import os
from typing import Literal

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import scorer


def _is_claude_session_trace(trace) -> bool:
    metadata = trace.info.trace_metadata or {}
    return bool(metadata.get("mlflow.trace.session"))


def _routing_results() -> list[tuple[int, str, str]]:
    run_start_ms = int(os.environ["RUN_START_MS"])
    traces = mlflow.search_traces(
        locations=[os.environ["CC_EXPERIMENT_ID"]],
        filter_string=f"trace.timestamp_ms > {run_start_ms}",
        return_type="list",
    )
    results = []
    for trace in traces:
        for span in trace.data.spans:
            if span.name != "route_request":
                continue
            message = (span.inputs or {}).get("message", "")
            results.append((trace.info.request_time, message, span.outputs))
    return sorted(results)


def get_judges() -> list:
    trace_first_judge = make_judge(
        name="debugged-from-mlflow-trace-before-editing",
        instructions=(
            "Inspect the agent's ordered work in {{ trace }}. Answer 'yes' only if all "
            "of these conditions hold: (1) it loaded the fix-agent-issue skill; "
            "(2) it reproduced the reported refund-routing failure through the existing "
            "instrumented application; (3) it retrieved and inspected the matching MLflow "
            "trace before its first source-code edit; and (4) its diagnosis used evidence "
            "from that trace, including the failing input and output. Do not infer actions "
            "that are not visible in the trace. Explain the evidence and ordering in the "
            "rationale."
        ),
        feedback_value_type=Literal["yes", "no"],
        generate_rationale_first=True,
    )

    @scorer(name="debugged-from-mlflow-trace-before-editing")
    def debugged_from_mlflow_trace_before_editing(trace) -> Feedback | None:
        # Application runs also emit traces into this experiment. Judge only the
        # Claude session trace, which contains the ordered debugging workflow.
        if not _is_claude_session_trace(trace):
            return None
        return trace_first_judge(trace=trace)

    @scorer(name="mlflow-traces-confirm-routing-fix")
    def mlflow_traces_confirm_routing_fix(trace) -> Feedback | None:
        if not _is_claude_session_trace(trace):
            return None

        results = _routing_results()
        refund_outputs = [
            output for _, message, output in results if "refund" in message.lower()
        ]
        saw_failure = "sales" in refund_outputs
        saw_fixed_after_failure = (
            saw_failure
            and "support" in refund_outputs[refund_outputs.index("sales") + 1 :]
        )
        passed = saw_failure and saw_fixed_after_failure
        return Feedback(
            value="yes" if passed else "no",
            rationale=f"Ordered refund trace outputs: {refund_outputs}",
        )

    @scorer(name="agent-routing-is-correct-after-fix")
    def agent_routing_is_correct_after_fix(trace) -> Feedback | None:
        if not _is_claude_session_trace(trace):
            return None

        agent = importlib.import_module("agent")
        refund = agent.route_request("I need a refund")
        non_refund = agent.route_request("Where is my order?")
        passed = refund == "support" and non_refund == "general"
        return Feedback(
            value="yes" if passed else "no",
            rationale=f"refund={refund!r}, non_refund={non_refund!r}",
        )

    return [
        debugged_from_mlflow_trace_before_editing,
        mlflow_traces_confirm_routing_fix,
        agent_routing_is_correct_after_fix,
    ]
