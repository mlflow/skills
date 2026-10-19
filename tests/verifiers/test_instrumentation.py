"""Local regressions; no Claude or paid model calls.

uv run --with mlflow --with pytest --with langgraph --with langchain-core pytest -q tests/verifiers/test_instrumentation.py
"""
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import mlflow
import pytest

from judges.instrumentation_behavior import get_judges
from judges.skill_usage import get_judges as skill_judges
from scripts.setup_langgraph_autolog_guidance import AGENT_SOURCE


@pytest.mark.parametrize("tool,style,want", [
    ("Bash", "header", "no"),
    ("Read", "header", "no"),
    ("Bash", "full", "yes"),
    ("Read", "chunks", "yes"),
    ("Bash", "chunks", "yes"),
    ("Read", "failed_chunk", "no"),
    ("Read", "wrong_path", "no"),
    ("Skill", "decoy_argument", "no"),
])
def test_full_skill_read(experiment, tmp_path, monkeypatch, tool, style, want):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("EXPECTED_SKILLS", "sample")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    path = Path(".claude/skills/sample/SKILL.md")
    path.parent.mkdir(parents=True)
    content = "---\nname: sample\ndescription: Example\n---\n\n# Instructions\nRead all of this.\nVerify the result.\n"
    path.write_text(content)
    lines = content.splitlines()
    chunks = ["\n".join(lines[:4])]
    if style != "header":
        chunks.append("\n".join(f"{i + 5}→{line}" for i, line in enumerate(lines[4:])))
    if style == "full":
        chunks = [content]
    with mlflow.start_span(name="participant") as root:
        trace_id = root.trace_id
        mlflow.update_current_trace(metadata={"mlflow.trace.session": "current"})
        for i, chunk in enumerate(chunks):
            with mlflow.start_span(name="read", attributes={"tool_name": tool}) as span:
                target = str(path) if style != "wrong_path" else "other/SKILL.md"
                if tool == "Skill":
                    span.set_inputs({"skill": "unrelated", "args": "sample"})
                else:
                    span.set_inputs({"file_path": target} if tool == "Read" else {"command": f"cat {target}"})
                span.set_outputs({"content": [{"type": "text", "text": chunk}]})
                if style == "failed_chunk" and i == 1:
                    span.set_status("ERROR")
    mlflow.flush_trace_async_logging()
    assert skill_judges()[0](trace=mlflow.get_trace(trace_id)).value == want


@pytest.mark.parametrize("instrumented,positional,want", [
    (False, False, "no"), (True, False, "yes"), (True, True, "yes"),
])
def test_entrypoint_requires_fresh_traces_and_runs_once(
    experiment, tmp_path, monkeypatch, instrumented, positional, want
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_CASE", "python")
    monkeypatch.delenv("REPO_URL", raising=False)
    # The original grader accepted these synthetic spans without running the app.
    if not instrumented:
        for message, output in [("REFUND", "support"), ("Hello", "general")]:
            with mlflow.start_span(name="unrelated") as span:
                span.set_inputs({"message": message})
                span.set_outputs(output)
    source = '''from pathlib import Path
import mlflow
with Path("executions").open("a") as counter:
    counter.write("run\\n")
def route_request(message):
    return "support" if "refund" in message.lower() else "general"
'''
    if instrumented and not positional:
        source += 'route_request = mlflow.trace(route_request)\n'
    if instrumented and positional:
        # The TypeScript SDK serializes positional arguments this way in the same store.
        source += '''for message in ("REFUND", "Hello"):
    with mlflow.start_span(name="route_request") as span:
        span.set_inputs({"args": [message]})
        span.set_outputs(route_request(message))
'''
    else:
        source += 'print(route_request("REFUND"))\nprint(route_request("Hello"))\n'
    Path("agent.py").write_text(source)
    judge = get_judges()[0]
    assert judge(trace=None).value == want
    assert judge(trace=None).value == want
    assert Path("executions").read_text() == "run\n"


@pytest.mark.parametrize("change,want", [
    ("autolog", "yes"), ("helper", "yes"), ("custom_block", "yes"),
    ("node_decorator", "no"), ("model_wrapper", "no"),
    ("node_wrapper", "no"), ("no_model", "no"),
])
def test_langgraph_real_execution(experiment, tmp_path, monkeypatch, change, want):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_CASE", "langgraph")
    monkeypatch.delenv("REPO_URL", raising=False)
    source = 'import mlflow\nmlflow.langchain.autolog()\n' + AGENT_SOURCE
    if change == "helper":
        source = source.replace('def lookup_weather(', '@mlflow.trace\ndef lookup_weather(')
    elif change == "custom_block":
        source = source.replace('    reading = lookup_weather("San Francisco")',
                                '    with mlflow.start_span(name="weather_lookup"):\n'
                                '        reading = lookup_weather("San Francisco")')
    elif change == "node_decorator":
        source = source.replace('def call_tool(', '@mlflow.trace\ndef call_tool(')
    elif change == "model_wrapper":
        source = source.replace('    response = llm.invoke(state["messages"])',
                                '    with mlflow.start_span(name="manual_model"):\n'
                                '        response = llm.invoke(state["messages"])')
    elif change == "node_wrapper":
        source = source.replace('    reading = lookup_weather("San Francisco")\n'
                                '    return {"messages": [HumanMessage(content=reading)]}',
                                '    with mlflow.start_span(name="manual_tool"):\n'
                                '        reading = lookup_weather("San Francisco")\n'
                                '        return {"messages": [HumanMessage(content=reading)]}')
    elif change == "no_model":
        source = source.replace('response = llm.invoke(state["messages"])',
                                'response = HumanMessage(content="It is 72F and sunny in San Francisco.")')
    Path("agent.py").write_text(source)
    # A participant smoke run alone must not make duplicate/missing-model cases pass.
    subprocess.run([sys.executable, "agent.py"], check=True, capture_output=True, timeout=30,
                   env={**os.environ, "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri()})
    feedback = get_judges()[0](trace=None)
    assert feedback.value == want, feedback.rationale


def test_external_application_and_bounded_clone(experiment, tmp_path, monkeypatch):
    setup = Path(__file__).parents[1] / "scripts/setup_tracing_test.py"
    repo = tmp_path / "original"
    repo.mkdir()
    (repo / "app.py").write_text('''import mlflow
@mlflow.trace
def answer(question):
    return "External application: " + question
print(answer("hello"))
''')
    for command in (["git", "init", "-q"], ["git", "add", "."],
                    ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"]):
        subprocess.run(command, cwd=repo, check=True, timeout=10)
    project = tmp_path / "clone"
    monkeypatch.setenv("PROJECT_DIR", str(project))
    monkeypatch.setenv("REPO_URL", repo.as_uri())
    monkeypatch.setenv("TEST_CASE", "python")
    real_run = subprocess.run
    calls = []

    def record_run(*args, **kwargs):
        calls.append((args, kwargs))
        return real_run(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", record_run)
        runpy.run_path(str(setup))
    assert "--depth" in calls[0][0][0]
    assert 0 < calls[0][1]["timeout"] <= 120
    assert (project / ".git/shallow").exists()
    monkeypatch.chdir(project)
    Path("tracing-command.json").write_text(json.dumps([sys.executable, "app.py"]))
    assert get_judges()[0](trace=None).value == "yes"
    Path("app.py").write_text('print("no tracing")\n')
    assert get_judges()[0](trace=None).value == "no"
