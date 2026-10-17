"""Negative controls for the active evaluation oracle, using native MLflow state."""
import importlib.util
import json
from pathlib import Path

import mlflow
import pytest
from mlflow.genai.datasets import create_dataset, get_dataset
from mlflow.genai.scorers import list_scorers

from judges.evaluation_behavior import verify_behavior
from scripts.setup_evaluation_cases import main as seed


def setup_case(case, experiment, tmp_path, monkeypatch):
    project = tmp_path / "project"
    for key, value in {"PROJECT_DIR": project, "WORK_DIR": tmp_path,
                       "TEST_WORK_DIR": tmp_path, "EVALUATION_CASE": case,
                       "MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS": "true"}.items():
        monkeypatch.setenv(key, str(value))
    seed()
    monkeypatch.chdir(project)
    expected = json.loads((tmp_path / "evaluation_expected.json").read_text())
    return project, expected


def load_predictor():
    spec = importlib.util.spec_from_file_location("candidate_app", "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predict


def test_full_evaluation_requires_an_earlier_scored_dry_run(experiment, tmp_path, monkeypatch):
    project, expected = setup_case("nested", experiment, tmp_path, monkeypatch)
    dataset = create_dataset(name="orders", experiment_id=experiment.experiment_id)
    dataset.merge_records(json.loads(Path("orders.json").read_text()))
    rows = dataset.to_df()
    scorers = list_scorers(experiment_id=experiment.experiment_id)
    predict = load_predictor()
    monkeypatch.chdir(tmp_path)  # Caller working directory need not be the app directory.
    mlflow.genai.evaluate(data=rows, predict_fn=predict, scorers=scorers)
    monkeypatch.chdir(project)
    with pytest.raises(AssertionError, match="three-row dry run"):
        verify_behavior(expected, tmp_path, experiment.experiment_id, 0)
    mlflow.genai.evaluate(data=rows.iloc[:3], predict_fn=predict, scorers=scorers)
    # A dry run after the full run does not satisfy the gate.
    with pytest.raises(AssertionError, match="three-row dry run"):
        verify_behavior(expected, tmp_path, experiment.experiment_id, 0)
    mlflow.genai.evaluate(data=rows, predict_fn=predict, scorers=scorers)
    assert verify_behavior(expected, tmp_path, experiment.experiment_id, 0)


@pytest.mark.parametrize("batch_size,passes", [(2, True), (3, True), (12, False)])
def test_failed_preflight_rejects_full_dataset(experiment, tmp_path, monkeypatch, batch_size, passes):
    _, expected = setup_case("dry_run", experiment, tmp_path, monkeypatch)
    rows = get_dataset(dataset_id=expected["dataset_id"]).to_df().to_dict("records")
    rows.sort(key=lambda row: int(row["inputs"]["request"]["id"].split("-")[-1]))
    mlflow.genai.evaluate(data=rows[:batch_size], predict_fn=load_predictor(),
                          scorers=list_scorers(experiment_id=experiment.experiment_id))
    if passes:
        assert verify_behavior(expected, tmp_path, experiment.experiment_id, 0)
    else:
        with pytest.raises(AssertionError, match="Dry run made"):
            verify_behavior(expected, tmp_path, experiment.experiment_id, 0)


@pytest.mark.parametrize("constant_currency", [True, False])
def test_suite_requires_expectation_relative_scores(experiment, tmp_path, monkeypatch, constant_currency):
    _, expected = setup_case("suite", experiment, tmp_path, monkeypatch)
    currency = "'USD'" if constant_currency else "expectations['currency']"
    Path("quality_suite.py").write_text('''from mlflow.genai.scorers import scorer
@scorer
def total_correct(outputs, expectations):
    return isinstance(outputs, dict) and outputs.get('total') == expectations['total']
@scorer
def currency_correct(outputs, expectations):
    return isinstance(outputs, dict) and outputs.get('currency') == ''' + currency + '''
@scorer
def task_completed(outputs):
    return isinstance(outputs, dict) and outputs.get('status') == 'complete'
def get_scorers():
    return [total_correct, currency_correct, task_completed]
''')
    if constant_currency:
        with pytest.raises(AssertionError, match="Wrong held-out currency_correct"):
            verify_behavior(expected, tmp_path, experiment.experiment_id, 0)
    else:
        assert verify_behavior(expected, tmp_path, experiment.experiment_id, 0)


@pytest.mark.parametrize("contaminate", [False, True])
def test_memalign_requires_saved_native_training_evidence(experiment, tmp_path, monkeypatch, contaminate):
    from mlflow.genai.judges import make_judge
    from mlflow.genai.judges.optimizers.memalign.optimizer import MemoryAugmentedJudge

    monkeypatch.setenv("TEST_JUDGE_MODEL", "openai:/unused-offline-control")
    monkeypatch.setenv("TEST_EMBEDDING_MODEL", "openai:/unused-offline-control")
    _, expected = setup_case("alignment_live", experiment, tmp_path, monkeypatch)
    Path("alignment.py").write_text("def align_judge(base, traces):\n    raise RuntimeError('must validate saved evidence first')\n")
    if contaminate:
        base = make_judge(name="total_correct", model=expected["model"],
                          instructions="Given {{ outputs }}, is total 20?", feedback_value_type=bool)
        base.register(name="base_total_correct", experiment_id=experiment.experiment_id)
        saved = MemoryAugmentedJudge(base_judge=base, _defer_init=True)
        saved._episodic_trace_ids = [p["trace_id"] for p in expected["pairs"]]
        saved.register(experiment_id=experiment.experiment_id)
    with pytest.raises(AssertionError, match="training split" if contaminate else "Missing registered"):
        verify_behavior(expected, tmp_path, experiment.experiment_id, 0)
