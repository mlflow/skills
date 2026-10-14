import json
import os
from pathlib import Path

import mlflow

project = Path(os.environ["PROJECT_DIR"])
project.mkdir(parents=True, exist_ok=True)
case = os.environ["TEST_CASE"]
uri = os.environ["MLFLOW_TRACKING_URI"]
experiment = os.environ["MLFLOW_EXPERIMENT_ID"]
if case == "unavailable_backend":
    uri = "http://127.0.0.1:9"
backend = {"tracking_uri": uri, "experiment_id": experiment}
(project / "backend.json").write_text(json.dumps(backend))
(project / ".env").write_text(
    f"MLFLOW_TRACKING_URI={uri}\nMLFLOW_EXPERIMENT_ID={experiment}\nKEEP_EXISTING_VALUE=yes\n"
)
source = '''import json
import mlflow

@mlflow.trace
def greet(name):
    return "hello:" + name

if __name__ == "__main__":
    config = json.load(open("backend.json"))
    mlflow.set_tracking_uri(config["tracking_uri"])
    mlflow.set_experiment(experiment_id=config["experiment_id"])
    print(greet("Ada"))
    mlflow.flush_trace_async_logging()
'''
(project / "agent.py").write_text(source)
manifest = {
    "backend": backend,
    "files": {name: (project / name).read_text() for name in ["backend.json", ".env", "agent.py"]},
    "experiment_ids": [e.experiment_id for e in mlflow.MlflowClient().search_experiments()],
}
(Path(os.environ["WORK_DIR"]) / "environment_expected.json").write_text(json.dumps(manifest))

if case == "classic_ml":
    (project / "agent.py").unlink()
    manifest["files"].pop("agent.py")
    (project / "train.py").write_text(
        "from sklearn.linear_model import LinearRegression\n"
        "from sklearn.metrics import mean_squared_error\n"
        "x = [[1], [2], [3], [4]]\ny = [3, 5, 7, 9]\n"
        "model = LinearRegression().fit(x, y)\n"
        "print(mean_squared_error(y, model.predict(x)))\n"
    )
    (Path(os.environ["WORK_DIR"]) / "environment_expected.json").write_text(json.dumps(manifest))
