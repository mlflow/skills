import json

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

from judges.skill_usage import tool_calls


def get_judges():
    @scorer(name="default-trace-location-invoked")
    def default_trace_location_invoked(trace):
        if not trace.info.trace_metadata.get("mlflow.trace.session"):
            return None
        calls = [
            tool for tool, inputs, _, failed in tool_calls(trace)
            if not failed and "get_default_trace_location(" in json.dumps(inputs)
        ]
        return Feedback(
            value="yes" if calls else "no",
            rationale=f"Called via {calls}" if calls else "get_default_trace_location() was never called",
        )

    return [default_trace_location_invoked]
