import ast
import html
import importlib.util
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


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
                assert snippet in html.unescape(source), "Example was not copied from the cited source"
                assert any(result["url"] in str(s.inputs) and "log_metric" in str(s.outputs) for s in trace.data.spans), "No source retrieval"
            else:
                for name, content in expected["files"].items():
                    assert Path(name).read_text() == content, f"Unexpected change to {name}"
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
                elif case == "existing_backend":
                    traces = mlflow.search_traces(
                        experiment_ids=[experiment_id],
                        filter_string=f"trace.timestamp_ms >= {os.environ['RUN_START_MS']}", return_type="list",
                    )
                    assert any(s.inputs == {"name": "Ada"} and s.outputs == "hello:Ada"
                               for t in traces for s in t.data.spans), "No persisted application smoke trace"
                else:
                    result = json.loads(Path("connection.json").read_text())
                    assert result["status"] == "blocked", "Claimed readiness despite unavailable prerequisite"
                    outputs = "\n".join(str(s.outputs) for s in trace.data.spans if s.attributes.get("tool_name") == "Bash")
                    if case == "unavailable_backend":
                        assert re.search(r"ConnectionError|Connection refused|Failed to establish|Max retries", outputs, re.I)
                    else:
                        assert any(
                            re.search(r"sagemaker_mlflow|verify_connection.py|boto3", str(s.inputs))
                            and re.search(r"ModuleNotFoundError|No module named|Unable to locate credentials|NoCredentialsError", str(s.outputs))
                            for s in trace.data.spans if s.attributes.get("tool_name") == "Bash"
                        ), "No observed failing SageMaker prerequisite check"
                        commands = "\n".join(str(s.inputs) for s in trace.data.spans if s.attributes.get("tool_name") == "Bash")
                        assert not re.search(r"(?:pip|uv pip) install.*sagemaker|aws (?:configure|sso login)", commands)
            return Feedback(value="yes", rationale=f"Verified {case} against independent outputs and preserved configuration")
        except Exception as error:
            return Feedback(value="no", rationale=f"{type(error).__name__}: {error}")

    return [environment_behavior]
