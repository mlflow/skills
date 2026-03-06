from __future__ import annotations

from typing import Literal

from mlflow.genai.judges import make_judge


def get_judges() -> list:
    return [
        make_judge(
            name="agent-eval-skill-invoked",
            instructions=(
                "Check that the agent-evaluation skill was loaded and invoked in the {{ trace }}. "
                "The skill is loaded when the agent reads its SKILL.md file from .claude/skills/agent-evaluation/. "
                "Answer 'yes' only if you see evidence that the skill was loaded."
            ),
            feedback_value_type=Literal["yes", "no"],
        ),
    ]
