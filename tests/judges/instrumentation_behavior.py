import ast
import json
import os
from pathlib import Path

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


def get_judges():
    experiment_id = os.environ["MLFLOW_EXPERIMENT_ID"]
    start_ms = os.environ["RUN_START_MS"]
    @scorer(name="instrumentation-behavior")
    def instrumentation_behavior(trace):
        traces = mlflow.search_traces(
            experiment_ids=[experiment_id],
            filter_string=f"trace.timestamp_ms >= {start_ms}", return_type="list",
        )
        if os.environ["TEST_CASE"] == "langgraph":
            tree = ast.parse(Path("agent.py").read_text())
            autolog_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Attribute) and n.func.attr == "autolog"]
            manual_nodes = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                            and n.name in {"call_model", "call_tool", "lookup_weather"}
                            and any("trace" in ast.unparse(d) for d in n.decorator_list)]
            graphs = [t for t in traces if any(s.name == "LangGraph" for s in t.data.spans)]
            passed = len(autolog_calls) == 1 and not manual_nodes and bool(graphs)
            for graph in graphs:
                names = [s.name for s in graph.data.spans]
                passed &= names.count("model") == 1 and names.count("tool") == 1
                passed &= any("72F" in str(s.outputs) for s in graph.data.spans)
            rationale = f"autolog calls={len(autolog_calls)}, manually traced nodes={manual_nodes}, persisted graphs={len(graphs)}"
        else:
            outcomes = set()
            for candidate in traces:
                for span in candidate.data.spans:
                    if span.outputs in ("support", "general"):
                        text = json.dumps(span.inputs)
                        for message in ["REFUND", "Hello"]:
                            if message in text:
                                outcomes.add((message, span.outputs))
            passed = {("REFUND", "support"), ("Hello", "general")} <= outcomes
            rationale = f"Persisted application input/output pairs: {sorted(outcomes)}"
        return Feedback(value="yes" if passed else "no", rationale=rationale)

    return [instrumentation_behavior]
