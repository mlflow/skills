from __future__ import annotations

import json
import os

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


def _trace_text(trace) -> str:
    if hasattr(trace, "to_dict"):
        return json.dumps(trace.to_dict(), default=str)
    return str(trace)


def get_judges() -> list:
    @scorer(name="agent-debugged-from-trace")
    def agent_debugged_from_trace(trace) -> Feedback:
        traces = mlflow.search_traces(
            locations=[os.environ["CC_EXPERIMENT_ID"]],
            return_type="list",
        )
        text = "\n".join(_trace_text(item) for item in traces)
        required = {
            "skill loaded": "fix-agent-issue" in text,
            "failure reproduced": "python agent.py" in text,
            "trace retrieved": "search_traces" in text,
            "trace inspected": "newest trace" in text and "input:" in text and "output:" in text,
            "source edited": 'return \\"support\\"' in text or 'return "support"' in text,
            "fix verified": "pytest" in text and "passed" in text,
        }
        passed = all(required.values())
        return Feedback(
            value="yes" if passed else "no",
            rationale=", ".join(
                f"{name}={'yes' if value else 'no'}" for name, value in required.items()
            ),
        )

    return [agent_debugged_from_trace]
