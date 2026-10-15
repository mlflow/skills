import time

import mlflow
import pytest


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    experiment_id = mlflow.create_experiment("application", artifact_location=str(tmp_path / "artifacts"))
    experiment = mlflow.set_experiment(experiment_id=experiment_id)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", experiment.experiment_id)
    monkeypatch.setenv("RUN_START_MS", str(int(time.time() * 1000)))
    yield experiment
    mlflow.end_run()
    mlflow.set_tracking_uri(previous_uri)
