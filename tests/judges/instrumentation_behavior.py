import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlflow
from mlflow.entities import Feedback, SpanStatusCode
from mlflow.genai.scorers import scorer


def redundant_node_spans(tree):
    """Reject wrapping framework nodes/model calls, not custom helper work."""
    aliases = {"mlflow": "mlflow"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({a.asname or a.name: a.name for a in node.names})
        elif isinstance(node, ast.ImportFrom) and node.module == "mlflow":
            aliases.update({a.asname or a.name: f"mlflow.{a.name}" for a in node.names})

    def is_tracing(node):
        name = ast.unparse(node.func if isinstance(node, ast.Call) else node)
        root, dot, rest = name.partition(".")
        return aliases.get(root, root) + dot + rest in {"mlflow.trace", "mlflow.start_span"}

    nodes = {
        call.args[1].id for call in ast.walk(tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_node" and len(call.args) > 1
        and isinstance(call.args[1], ast.Name)
    }
    redundant = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in nodes:
            continue
        if any(is_tracing(d) for d in node.decorator_list):
            redundant.append(node.name)
        for block in ast.walk(node):
            if isinstance(block, (ast.With, ast.AsyncWith)) and any(
                is_tracing(item.context_expr) for item in block.items
            ):
                # A whole-node wrapper or a wrapper around the native model call
                # duplicates autolog. A span around custom helper work does not.
                if any(isinstance(n, ast.Return) or (
                    isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in {"invoke", "ainvoke", "stream", "astream"}
                ) for n in ast.walk(block)):
                    redundant.append(node.name)
    return redundant


def application_command(case):
    if os.environ.get("REPO_URL"):
        command = json.loads(Path("tracing-command.json").read_text())
        if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
            raise ValueError("tracing-command.json must contain a nonempty argv array")
        return command
    if case == "typescript":
        for runner in ("tsx", "ts-node"):
            executable = Path("node_modules/.bin") / runner
            if executable.exists():
                return [str(executable.resolve()), "agent.ts"]
        return ["node", "agent.ts"]
    python = Path(".venv/bin/python")
    return [str(python.absolute()) if python.exists() else sys.executable, "agent.py"]


def verify_instrumentation():
    experiment_id = os.environ["MLFLOW_EXPERIMENT_ID"]
    case = os.environ["TEST_CASE"]
    # Snapshot before executing the entrypoint: participant smoke tests or unrelated
    # synthetic spans must not satisfy verification. Run once, before scorer threads.
    mlflow.flush_trace_async_logging()
    start_ms = int(time.time() * 1000)
    search = dict(experiment_ids=[experiment_id],
                  filter_string=f"trace.timestamp_ms >= {start_ms}", return_type="list")
    existing = {t.info.trace_id for t in mlflow.search_traces(**search)}
    result = subprocess.run(
        application_command(case), check=False, capture_output=True, text=True, timeout=120,
        env={**os.environ, "MLFLOW_TRACKING_URI": mlflow.get_tracking_uri(),
             "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING": "false"},
    )
    if result.returncode:
        return False, f"Application exited {result.returncode}: {result.stderr[-2000:]}"
    traces = [t for t in mlflow.search_traces(**search) if t.info.trace_id not in existing]
    spans = [s for t in traces for s in t.data.spans]
    if not spans or any(s.status.status_code == SpanStatusCode.ERROR for s in spans):
        return False, "Application produced no fresh successful traces"
    if os.environ.get("REPO_URL"):
        passed = any(s.inputs is not None and s.outputs is not None for s in spans)
        return passed, f"External application produced {len(traces)} fresh traces with I/O={passed}"
    if case == "langgraph":
        tree = ast.parse(Path("agent.py").read_text())
        autolog_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "autolog"]
        redundant = redundant_node_spans(tree)
        graphs = [t for t in traces if any(s.name == "LangGraph" for s in t.data.spans)]
        passed = len(autolog_calls) == 1 and not redundant and bool(graphs)
        for graph in graphs:
            names = [s.name for s in graph.data.spans]
            passed &= names.count("model") == 1 and names.count("tool") == 1
            passed &= any(s.name == "FakeListChatModel" and s.span_type in {"LLM", "CHAT_MODEL"}
                          and s.inputs and "72F" in str(s.outputs) for s in graph.data.spans)
        return bool(passed), (f"autolog calls={len(autolog_calls)}, redundant nodes={redundant}, "
                              f"fresh graphs={len(graphs)}; spans={[(s.name, s.span_type) for s in spans]}")
    outcomes = set()
    for span in spans:
        for message in ("REFUND", "Hello"):
            if (span.inputs in ({"message": message}, {"args": [message]})
                    and span.outputs in ("support", "general")):
                outcomes.add((message, span.outputs))
    return {("REFUND", "support"), ("Hello", "general")} <= outcomes, f"Fresh application I/O: {sorted(outcomes)}"


def get_judges():
    try:
        passed, rationale = verify_instrumentation()
    except (OSError, ValueError, SyntaxError, subprocess.TimeoutExpired) as error:
        passed, rationale = False, f"Application verification failed: {error}"

    @scorer(name="instrumentation-behavior")
    def instrumentation_behavior(trace):
        return Feedback(value="yes" if passed else "no", rationale=rationale)

    return [instrumentation_behavior]
