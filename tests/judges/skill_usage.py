import json
import os
import re
from pathlib import Path

import yaml
from mlflow.entities import Feedback, SpanStatusCode
from mlflow.genai.scorers import scorer



def tool_calls(trace):
    for span in trace.data.spans:
        yield (span.attributes.get("tool_name", ""), span.inputs or {},
               span.outputs or "", span.status.status_code == SpanStatusCode.ERROR)
    # Claude compaction can omit earlier tools from its exported MLflow trace.
    # Use only the matching session in this run's isolated home or saved logs.
    session = trace.info.trace_metadata.get("mlflow.trace.session", "")
    if not session or Path(session).name != session:
        return
    roots = [Path(os.environ["TEST_WORK_DIR"]) / "claude-sessions"]
    if os.environ.get("CLAUDE_CONFIG_DIR"):
        roots.append(Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects")
    pending = {}
    for root in roots:
        for path in root.rglob(session + ".jsonl"):
            for line in path.read_text().splitlines():
                try:
                    content = json.loads(line).get("message", {}).get("content", [])
                except ValueError:  # A live transcript may end with a partial line.
                    continue
                if not isinstance(content, list):
                    continue
                for item in content:
                    if item.get("type") == "tool_use":
                        pending[item["id"]] = item
                    elif item.get("type") == "tool_result" and item.get("tool_use_id") in pending:
                        call = pending.pop(item["tool_use_id"])
                        yield (call["name"], call.get("input", {}), item.get("content", ""),
                               item.get("is_error", False))


def output_text(value):
    """Unwrap both exported span outputs and native transcript content blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(output_text(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(output_text(v) for v in value)
    return ""


def read_lines(text):
    # Read/cat -n add line numbers; whitespace and chunk boundaries are immaterial.
    return {
        " ".join(re.sub(r"^\s*\d+(?:→|\t|[|:])", "", line).split())
        for line in text.splitlines() if line.strip()
    } - {""}


def get_judges():
    required = os.environ["EXPECTED_SKILLS"].split(",")

    @scorer(name="required-skills-loaded")
    def required_skills_loaded(trace):
        if not trace.info.trace_metadata.get("mlflow.trace.session"):
            return None
        calls = list(tool_calls(trace))
        missing = []
        for directory in required:
            content = (Path(".claude/skills") / directory / "SKILL.md").read_text()
            name = yaml.safe_load(content.split("---", 2)[1])["name"]
            loaded = False
            observed = set()
            for tool, inputs, outputs, failed in calls:
                if failed:
                    continue
                text = output_text(outputs)
                if (tool == "Skill" and inputs.get("skill", "").split(":")[-1] in {name, directory}
                        and "Unknown skill" not in text):
                    loaded = True
                if tool in {"Read", "Bash"} and f"{directory}/SKILL.md" in json.dumps(inputs):
                    observed.update(read_lines(text))
            if not loaded and not read_lines(content) <= observed:
                missing.append(directory)
        return Feedback(
            value="no" if missing else "yes",
            rationale=f"Missing full skill reads: {missing}" if missing else f"Loaded: {required}",
        )

    return [required_skills_loaded]
