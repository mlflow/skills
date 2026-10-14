import json
import os
from pathlib import Path

import yaml
from mlflow.entities import Feedback, SpanStatusCode
from mlflow.genai.scorers import scorer


def get_judges():
    required = os.environ["EXPECTED_SKILLS"].split(",")

    @scorer(name="required-skills-loaded")
    def required_skills_loaded(trace):
        if not trace.info.trace_metadata.get("mlflow.trace.session"):
            return None
        missing = []
        for directory in required:
            content = (Path(".claude/skills") / directory / "SKILL.md").read_text()
            name = yaml.safe_load(content.split("---", 2)[1])["name"]
            loaded = False
            for span in trace.data.spans:
                tool = span.attributes.get("tool_name", "")
                inputs = json.dumps(span.inputs or {})
                outputs = str(span.outputs or "")
                if (tool == "Skill" and any(f'"{candidate}"' in inputs for candidate in (name, directory))
                        and span.status.status_code != SpanStatusCode.ERROR
                        and "Unknown skill" not in outputs):
                    loaded = True
                if tool in {"Read", "Bash"} and f"{directory}/SKILL.md" in inputs:
                    loaded |= f"name: {name}" in outputs and "description:" in outputs
            if not loaded:
                missing.append(directory)
        return Feedback(
            value="no" if missing else "yes",
            rationale=f"Missing skill reads: {missing}" if missing else f"Loaded: {required}",
        )

    return [required_skills_loaded]
