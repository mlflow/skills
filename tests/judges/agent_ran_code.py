from __future__ import annotations

from typing import Literal

from mlflow.genai.judges import make_judge


def get_judges() -> list:
    return [
        make_judge(
            name="agent-ran-instrumented-code",
            instructions=(
                "Examine the {{ trace }} and determine whether the agent ran the "
                "application or agent code after adding MLflow tracing instrumentation. "
                "Look for evidence that the agent executed the instrumented program "
                "(e.g., running a CLI command, calling an entry point, executing a script). "
                "Return 'yes' if the agent ran the code after instrumenting it, 'no' otherwise."
            ),
            feedback_value_type=Literal["yes", "no"],
        ),
    ]
