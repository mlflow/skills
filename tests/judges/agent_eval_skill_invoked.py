"""Judge: check that the agent-evaluation skill was invoked."""

from __future__ import annotations

from typing import Literal

from mlflow.genai.judges import make_judge


def get_judges() -> list:
    """Return a judge that checks the agent-evaluation skill was invoked."""
    return [
        make_judge(
            name="agent-eval-skill-invoked",
            instructions="Check that the agent-evaluation skill is invoked in the {{ trace }}.",
            feedback_value_type=Literal["yes", "no"],
        ),
    ]
