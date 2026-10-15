import ast
import html
import importlib.util
import json
import os
import re
import shlex
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import Scorer, scorer

from judges.skill_usage import tool_calls
from utils import run_command


def commands(command):
    """Recognize ordinary commands and shell -c wrappers, not shell semantics."""
    # ponytail: explicit command forms only; extend for observed workflows, not a shell interpreter.
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    words = []
    for word in [*lexer, ";"]:
        if word and all(c in ";&|\n" for c in word):
            while words and ("=" in words[0] or Path(words[0]).name in {"env", "uv", "run"}):
                words.pop(0)
            if words:
                words[0] = Path(words[0]).name
                if words[0] in {"bash", "sh", "zsh"}:
                    for i, arg in enumerate(words[1:], 1):
                        if arg.startswith("-") and "c" in arg and i + 1 < len(words):
                            yield from commands(words[i + 1])
                            break
                else:
                    yield words
            words = []
        else:
            words.append(word)


def python_source(words, command):
    if not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", words[0]):
        return ""
    if "-c" in words:
        return words[words.index("-c") + 1]
    for arg in words[1:]:
        if arg.endswith(".py") and Path(arg).is_file():
            return Path(arg).read_text()
    heredoc = re.search(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\1(?:\n|$)", command, re.S)
    return heredoc[2] if heredoc else ""


def forbidden_setup(words):
    pip = words[0] in {"pip", "pip3"} or words[1:3] == ["-m", "pip"]
    return (pip and "install" in words or words[0] == "aws" and (
        "configure" in words or "sso" in words and "login" in words))


def verify_prerequisite(trace, case, uri):
    observed = False
    for tool, inputs, outputs, _ in tool_calls(trace):
        if tool != "Bash":
            continue
        command = inputs.get("command", "")
        for words in commands(command):
            if case != "unavailable_backend":
                assert not forbidden_setup(words), "Attempted package installation or credential acquisition"
            source = python_source(words, command)
            nodes = list(ast.walk(ast.parse(source))) if source else []
            calls = {getattr(n.func, "attr", getattr(n.func, "id", ""))
                     for n in nodes if isinstance(n, ast.Call)}
            if case == "unavailable_backend":
                target = uri in " ".join(words) or uri in source or "backend.json" in source
                probe = words[0] in {"curl", "wget"} or bool(
                    calls & {"urlopen", "get", "post", "search_experiments", "get_experiment"})
                observed |= target and probe and bool(re.search(
                    r"ConnectionError|Connection refused|Failed to establish|Max retries|Failed to connect",
                    str(outputs), re.I))
            else:
                probe = ("show" in words and "sagemaker-mlflow" in words
                         and (words[0] in {"pip", "pip3"} or words[1:3] == ["-m", "pip"]))
                probe |= any(isinstance(n, ast.Import) and any(
                    a.name in {"sagemaker_mlflow", "boto3"} for a in n.names)
                    or isinstance(n, ast.ImportFrom) and n.module in {"sagemaker_mlflow", "boto3"}
                    for n in nodes)
                probe |= words[:3] == ["aws", "sts", "get-caller-identity"]
                observed |= probe and bool(re.search(
                    r"ModuleNotFoundError|No module named|Unable to locate credentials|NoCredentialsError|Package\(s\) not found: sagemaker-mlflow",
                    str(outputs)))
    assert observed, "No observed failing prerequisite probe for the configured backend"


def verify_training_data(expected):
    result = run_command([sys.executable, "-c", '''
import json, runpy, sys
import numpy as np
from sklearn.linear_model import LinearRegression
expected = json.loads(sys.argv[1])
original = LinearRegression.fit
calls = []
def fit(self, X, y, *args, **kwargs):
    np.testing.assert_array_equal(X, expected["x"])
    np.testing.assert_array_equal(y, expected["y"])
    calls.append(True)
    return original(self, X, y, *args, **kwargs)
LinearRegression.fit = fit
runpy.run_path("train.py", run_name="__main__")
assert calls, "train.py never fitted the model"
''', json.dumps(expected)], check=False, timeout=90)
    assert result.returncode == 0, f"Training data verification failed: {result.stderr[-2000:]}"


def copied_documentation_blocks(snippet, source):
    remaining = "\n".join(line for line in snippet.splitlines() if line.strip())
    for block in re.findall(r"```[^\n]*\n(.*?)```", html.unescape(source), re.S):
        block = "\n".join(line for line in block.splitlines() if line.strip())
        if block and (remaining == block or remaining.startswith(block + "\n")):
            remaining = remaining[len(block):].lstrip("\n")
    return bool(snippet.strip()) and not remaining


def get_judges():
    experiment_id = os.environ["MLFLOW_EXPERIMENT_ID"]
    case = os.environ["TEST_CASE"]
    expected = json.loads((Path(os.environ["TEST_WORK_DIR"]) / "environment_expected.json").read_text())

    @scorer(name="environment-behavior")
    def environment_behavior(trace):
        if not trace.info.trace_metadata.get("mlflow.trace.session"):
            return None
        try:
            if case == "dispatcher":
                spec = importlib.util.spec_from_file_location("candidate_scorers", "scorers.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                assert isinstance(module.answer_present, Scorer), "answer_present must be a native MLflow Scorer"
                for output, want in [({"answer": "hello"}, True), ({"answer": ""}, False),
                                     ({"answer": "  "}, False), ({}, False), ({"answer": None}, False)]:
                    result = module.answer_present(outputs=output)
                    value = result.value if isinstance(result, Feedback) else result
                    assert type(value) is bool and value == want, f"Wrong score for {output}: {value}"
            elif case == "docs":
                result = json.loads(Path("docs.json").read_text())
                parsed = urlparse(result["url"])
                assert parsed.scheme == "https" and parsed.hostname == "mlflow.org"
                assert parsed.path.startswith("/docs/") and parsed.path.endswith(".md")
                with urlopen(result["url"], timeout=20) as response:
                    source = response.read().decode()
                snippet = result["code"].strip()
                assert "mlflow.log_metric" in snippet
                ast.parse(snippet)
                # Quickstarts split runnable examples across notebook cells; the
                # Markdown renderer inserts blank lines. Require complete source
                # blocks in order, allowing concatenation and blank-line changes.
                assert copied_documentation_blocks(snippet, source), "Example includes undocumented code"
                assert any(result["url"] in str(s.inputs) and "log_metric" in str(s.outputs) for s in trace.data.spans), "No source retrieval"
            else:
                for name, content in expected["files"].items():
                    assert Path(name).read_text() == content, f"Unexpected change to {name}"
                # Only the local test server is exclusive to this participant.
                # Other users may create experiments in a shared Databricks workspace.
                if not expected["backend"]["tracking_uri"].startswith("databricks"):
                    now = {e.experiment_id for e in mlflow.MlflowClient().search_experiments()}
                    assert now == set(expected["experiment_ids"]), "Created an unnecessary experiment"
                if case == "classic_ml":
                    import numpy as np
                    runs = mlflow.search_runs(experiment_ids=[experiment_id],
                        filter_string=f"attributes.start_time >= {os.environ['RUN_START_MS']}",
                        output_format="list")
                    candidates = [r for r in runs if r.info.status == "FINISHED"
                                  and abs(r.data.metrics.get("mse", -1)) < 1e-10
                                  and r.data.params.get("fit_intercept", "").lower() == "true"]
                    assert candidates, "No finished training run with exact metric and parameter"
                    models = mlflow.search_logged_models(experiment_ids=[experiment_id], output_format="list")
                    verified = False
                    for model in models:
                        if model.source_run_id in {r.info.run_id for r in candidates}:
                            loaded = mlflow.sklearn.load_model(model.model_uri)
                            np.testing.assert_allclose(loaded.predict([[5], [9]]), [11, 19])
                            verified = True
                    assert verified, "No loadable logged sklearn model with correct held-out predictions"
                    verify_training_data(expected["training_data"])
                elif case == "existing_backend":
                    traces = mlflow.search_traces(
                        experiment_ids=[experiment_id],
                        filter_string=f"trace.timestamp_ms >= {os.environ['RUN_START_MS']}", return_type="list",
                    )
                    assert any(s.inputs == {"name": "Ada"} and s.outputs == "hello:Ada"
                               for t in traces for s in t.data.spans), "No persisted application smoke trace"
                else:
                    result = json.loads(Path("connection.json").read_text())
                    assert result["status"] in {"blocked", "unavailable"}, "Claimed readiness despite unavailable prerequisite"
                    assert result.get("reason"), "Missing prerequisite failure explanation"
                    verify_prerequisite(trace, case, expected["backend"]["tracking_uri"])
            return Feedback(value="yes", rationale=f"Verified {case} against independent outputs and preserved configuration")
        except Exception as error:
            return Feedback(value="no", rationale=f"{type(error).__name__}: {error}")

    return [environment_behavior]
