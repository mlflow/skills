"""Behavior checks against real MLflow state and private seeded ground truth."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.datasets import get_dataset, search_datasets
from mlflow.genai.scorers import list_scorers, scorer


def _binary(value):
    if isinstance(value, bool):
        return value
    if value in (0, 1, "yes", "no"):
        return value in (1, "yes")
    raise AssertionError(f"Non-aggregating or missing feedback: {value!r}")


def _roots(trace):
    return [s for s in trace.data.spans if s.parent_id is None]


def _feedback(trace, name, start):
    found = [
        a
        for a in trace.info.assessments
        if a.name == name and a.feedback is not None and a.create_time_ms >= start
    ]
    assert found, f"Missing {name} feedback on {trace.info.trace_id}"
    assert all(a.error is None and a.valid for a in found), (
        f"Failed/invalid {name} assessment"
    )
    values = {_binary(a.feedback.value) for a in found}
    assert len(values) == 1, f"Conflicting {name} feedback"
    return values.pop()


def _runs(exp, start):
    return mlflow.search_runs(
        experiment_ids=[exp],
        filter_string=f"attributes.start_time >= {start}",
        output_format="list",
    )


def _verify_suite(expected, work):
    assert Path("quality_suite.py").is_file(), "Missing runnable quality_suite.py"
    # A separate process avoids nested evaluation in the harness's scorer threads.
    # Pass inputs/expectations, not the verifier's expected verdicts, to candidate code.
    code = """import importlib.util, json, mlflow, sys
spec = importlib.util.spec_from_file_location("participant_quality_suite", "quality_suite.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
scorers = module.get_scorers()
assert len(scorers) == 3
expected = json.load(open(sys.argv[1]))
names = expected.get("names", ["total_correct", "currency_correct", "task_completed"])
assert {s.name for s in scorers} == set(names)
if expected["case"] == "model_suite":
    from mlflow.genai.scorers import Guidelines, RelevanceToQuery
    from mlflow.genai.judges import Judge
    by_name = {s.name: s for s in scorers}
    assert isinstance(by_name["relevance_to_query"], RelevanceToQuery)
    assert isinstance(by_name["supported_resolution"], (Judge, Guidelines))
    assert all(by_name[n].model == expected["model"] for n in names[1:])
    for text, wanted in [("hello", True), ("", False), ("   ", False), (None, False)]:
        result = by_name["nonempty"](outputs=text)
        value = getattr(result, "value", result)
        assert value in ((True, "yes") if wanted else (False, "no"))
from mlflow.genai.scorers import Scorer
assert all(isinstance(s, Scorer) for s in scorers)
rows = json.load(open(sys.argv[1]))["heldout"]
experiment = mlflow.create_experiment("heldout-suite-" + __import__("uuid").uuid4().hex)
mlflow.set_experiment(experiment_id=experiment)
result = mlflow.genai.evaluate(data=rows, scorers=scorers)
json.dump({"run_id": result.run_id, "experiment_id": experiment}, open(sys.argv[2], "w"))
"""
    result_path = work / "suite_validation_run.json"
    input_path = work / "suite_validation_input.json"
    input_path.write_text(json.dumps({
        "case": expected["case"],
        "names": expected.get("names", ["total_correct", "currency_correct", "task_completed"]),
        "model": expected.get("model"),
        "heldout": [{k: v for k, v in row.items() if k != "verdicts"} for row in expected["heldout"]],
    }))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(input_path),
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert result.returncode == 0, (
        f"Held-out suite execution failed: {result.stderr[-2200:]}"
    )
    info = json.loads(result_path.read_text())
    run = mlflow.get_run(info["run_id"])
    assert run.info.status == "FINISHED", "Held-out native evaluation did not finish"
    traces = mlflow.search_traces(
        experiment_ids=[info["experiment_id"]],
        run_id=info["run_id"],
        return_type="list",
    )
    names = expected.get(
        "names", ["total_correct", "currency_correct", "task_completed"]
    )
    remaining = list(expected["heldout"])
    for trace in traces:
        for root in _roots(trace):
            match = next(
                (
                    r
                    for r in remaining
                    if root.inputs == r["inputs"] and root.outputs == r["outputs"]
                ),
                None,
            )
            if match is None:
                continue
            for name, verdict in zip(names, match["verdicts"]):
                assert _feedback(trace, name, 0) == verdict, (
                    f"Wrong held-out {name}: {root.outputs!r}"
                )
            remaining.remove(match)
    assert not remaining, f"Missing held-out rows: {len(remaining)}"
    for index, name in enumerate(names):
        mean = sum(row["verdicts"][index] for row in expected["heldout"]) / len(
            expected["heldout"]
        )
        assert abs(run.data.metrics.get(f"{name}/mean", -1) - mean) < 1e-9, (
            f"Wrong/missing {name} mean"
        )
    return "Participant suite ran on all held-out rows; exact feedback and aggregate metrics matched."


def _verify_alignment_data(expected, exp):
    datasets = search_datasets(experiment_ids=[exp])
    assert len(datasets) == 1 and datasets[0].dataset_id == expected["dataset_id"]
    rows = get_dataset(dataset_id=expected["dataset_id"]).to_df().to_dict("records")
    assert len(rows) == len(expected["pairs"])
    for pair in expected["pairs"]:
        assert any(
            r["inputs"] == {"trace_id": pair["trace_id"]}
            and r["expectations"] == {"split": pair["split"]}
            for r in rows
        )
        trace = mlflow.get_trace(pair["trace_id"])
        human = [
            a
            for a in trace.info.assessments
            if a.name == "total_correct"
            and str(a.source.source_type) == "HUMAN"
            and a.source.source_id == "order-expert"
        ]
        assert len(human) == 1 and human[0].error is None and human[0].valid
        assert human[0].feedback.value is pair["label"], (
            "Pair does not match real human feedback"
        )


def verify_behavior(expected, work, exp, start):
    case = expected["case"]
    if case == "alignment_live":
        import importlib.util
        from mlflow.entities import AssessmentSource
        from mlflow.genai.judges import make_judge
        from mlflow.genai.judges.optimizers.memalign.optimizer import MemoryAugmentedJudge

        # Check the participant's saved native result before exercising its helper
        # on new data. Verifier-created training alone cannot prove the smoke run.
        _verify_alignment_data(expected, exp)
        registered = {s.name: s for s in list_scorers(experiment_id=exp)}
        assert set(registered) == {"base_total_correct", "total_correct"}, (
            "Missing registered base/aligned judges from the participant's smoke run"
        )
        saved = registered["total_correct"]
        assert isinstance(saved, MemoryAugmentedJudge), "Saved scorer is not a native MemAlign judge"
        memory = saved.model_dump()["memory_augmented_judge_data"]
        train_ids = {p["trace_id"] for p in expected["pairs"] if p["split"] == "train"}
        assert set(memory["episodic_trace_ids"]) == train_ids, "Wrong alignment training split"
        assert saved.model == registered["base_total_correct"].model == expected["model"]
        holdouts = [mlflow.get_trace(p["trace_id"]) for p in expected["pairs"] if p["split"] == "holdout"]
        compared = False
        for run in _runs(exp, start):
            if run.info.status != "FINISHED":
                continue
            evaluated = mlflow.search_traces(experiment_ids=[exp], run_id=run.info.run_id, return_type="list")
            matched = set()
            for trace in evaluated:
                for index, historical in enumerate(holdouts):
                    original = _roots(historical)[0]
                    if any(root.inputs == original.inputs and root.outputs == original.outputs
                           for root in _roots(trace)):
                        for name in registered:
                            _feedback(trace, name, start)
                        matched.add(index)
            compared |= len(matched) == len(holdouts)
        assert compared, "Missing native base/aligned comparison on the heldout rows"

        spec = importlib.util.spec_from_file_location("candidate_alignment", "alignment.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Private verification traces are newly generated, with unrelated criterion
        # feedback to catch accidental mixing. Holdout traces are never passed in.
        training = []
        for i, total in enumerate([20, 25, 20, 19]):
            with mlflow.start_span(name="alignment-verification") as span:
                span.set_inputs({"order_id": f"private-{i}"})
                span.set_outputs({"total": total})
            mlflow.flush_trace_async_logging()
            for name, label in [("total_correct", total == 20), ("tone", total != 20)]:
                mlflow.log_feedback(trace_id=span.trace_id, name=name, value=label,
                    rationale="The contracted order total is exactly 20.",
                    source=AssessmentSource(source_type="HUMAN", source_id="verifier"))
            training.append(mlflow.get_trace(span.trace_id))
        base = make_judge(name="total_correct", model=expected["model"], feedback_value_type=bool,
            instructions="Given {{ inputs }} and {{ outputs }}, is the order total exactly 20?")
        original = base.instructions
        aligned = module.align_judge(base, training)
        assert isinstance(aligned, MemoryAugmentedJudge), "No actual MemAlign result"
        assert base.instructions == original, "Base judge was mutated"
        assert aligned.model == expected["model"]
        comparison = []
        for i, total in enumerate([20, 200, 0, 20]):
            kwargs = {"inputs": {"order_id": f"unseen-{i}"}, "outputs": {"total": total}}
            values = []
            for judge in [base, aligned]:
                result = judge(**kwargs)
                assert result.error is None and result.value is not None
                values.append(_binary(result.value))
            assert values[1] == (total == 20), "Aligned judge disagrees with held-out expert label"
            comparison.append({"inputs": kwargs["inputs"], "expected": total == 20,
                               "base": values[0], "aligned": values[1]})
        (work / "alignment-verification.json").write_text(json.dumps(comparison, indent=2))
        return "Native MemAlign result executed on four private holdouts; base preserved and exact expert agreement checked."
    if case == "alignment":
        outcome = json.loads(Path("outcome.json").read_text())
        assert outcome.get("criterion") == "total_correct"
        assert outcome.get("dataset_id") == expected["dataset_id"]
        assert outcome.get("status") == "blocked" and outcome.get("prerequisites")
        pairs = outcome.get("pairs", [])
        assert len(pairs) == len(expected["pairs"]), (
            "Missing or duplicate alignment pairs"
        )
        assert all(type(p["label"]) is bool for p in pairs), "Human labels must be boolean"
        required_pairs = [{key: p[key] for key in ("trace_id", "label", "split")} for p in pairs]
        assert sorted(required_pairs, key=lambda p: p["trace_id"]) == sorted(
            expected["pairs"], key=lambda p: p["trace_id"]
        ), "Wrong criterion labels or holdout contamination"
        _verify_alignment_data(expected, exp)
        return "All exact criterion/human-label pairs and disjoint splits match real MLflow records; optimization explicitly blocked."
    if case in {"suite", "model_suite"}:
        return _verify_suite(expected, work)
    runs = _runs(exp, start)
    assert (
        hashlib.sha256(Path("app.py").read_bytes()).hexdigest()
        == expected["app_sha256"]
    ), "Predictor was modified"
    datasets = search_datasets(experiment_ids=[exp])
    assert len(datasets) == 1, f"Expected one native dataset, found {len(datasets)}"
    dataset = datasets[0]
    if "dataset_id" in expected:
        assert dataset.dataset_id == expected["dataset_id"], "Existing dataset replaced"
    # Read records through the public dataframe API (to_dict metadata alone has no rows).
    rows = get_dataset(dataset_id=dataset.dataset_id).to_df().to_dict("records")
    assert len(rows) == len(expected["records"]), "Dataset rows missing or added"
    for row in expected["records"]:
        assert any(all(r.get(k) == v for k, v in row.items()) for r in rows), (
            "Seeded dataset changed"
        )
    registered = list_scorers(experiment_id=exp)
    assert [s.name for s in registered] == ["exact_total"], (
        "Registered scorer was replaced or extra scorers added"
    )
    # A renamed/re-registered constant scorer must not satisfy reuse.
    for s in registered:
        assert _binary(s(outputs={"total": 7}, expectations={"total": 7})) is True
        assert _binary(s(outputs={"total": 8}, expectations={"total": 7})) is False
    calls_path = Path("predict_calls.jsonl")
    calls = (
        [json.loads(line) for line in calls_path.read_text().splitlines()]
        if calls_path.exists()
        else []
    )
    traces = mlflow.search_traces(
        experiment_ids=[exp],
        filter_string=f"trace.timestamp_ms >= {start}",
        return_type="list",
    )
    if case == "precomputed":
        assert not calls, "Precomputed evaluation reran predictor"
    if case == "dry_run":
        assert 3 <= len(calls) <= 4, (
            f"Dry run made {len(calls)} calls (3 rows plus optional validation call allowed)"
        )
        assert {json.dumps(c, sort_keys=True) for c in calls} == {
            json.dumps(r["inputs"], sort_keys=True) for r in expected["records"][:3]
        }, "Expected only the first three dry-run inputs; full evaluation started or dry run incomplete"
        assert any(str(t.info.state).upper().endswith("ERROR") for t in traces) or any(
            r.info.status == "FAILED" for r in runs
        ), "No real failed trace/evaluation run"
        for t in traces:
            if any(root.inputs == expected["records"][0]["inputs"] for root in _roots(t)):
                continue  # The cached first row succeeds; backend-dependent rows must not.
            assert not any(
                a.feedback
                and a.feedback.value in (True, 1, "yes")
                and a.name == "exact_total"
                for a in t.info.assessments
            ), "Invented successful scores"
        outcome = json.loads(Path("outcome.json").read_text())
        assert (
            outcome.get("status") == "blocked"
            and outcome.get("full_evaluation_started") is False
        )
        assert outcome.get("reason"), "Missing blocker explanation"
        return "Exactly three dry-run inputs attempted; backend-dependent rows failed; no full evaluation."
    if case == "nested":
        assert all(any(c == r["inputs"] for c in calls) for r in expected["records"]), (
            "Not all nested inputs reached predictor"
        )
    # Validate bounded and complete runs independently, including their ordering.
    dry_runs, full_runs = [], []
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        evaluated = mlflow.search_traces(
            experiment_ids=[exp], run_id=run.info.run_id, return_type="list"
        )
        matched = set()
        for t in evaluated:
            for root in _roots(t):
                for i, record in enumerate(expected["records"]):
                    if root.inputs != record["inputs"]:
                        continue
                    output = record.get(
                        "outputs", {"total": record["expectations"]["total"] + (i >= 4)}
                    )
                    assert root.outputs == output, f"Wrong output persisted for row {i}"
                    assert _feedback(t, "exact_total", start) == (i < 4), (
                        f"Wrong score for row {i}"
                    )
                    matched.add(i)
        if len(matched) == 3:
            assert abs(run.data.metrics.get("exact_total/mean", -1) - sum(i < 4 for i in matched) / 3) < 1e-9
            dry_runs.append(run)
        if len(matched) == len(expected["records"]):
            assert abs(run.data.metrics.get("exact_total/mean", -1) - 4 / 6) < 1e-9, (
                "Wrong/missing aggregate score"
            )
            full_runs.append(run)
    assert full_runs, "No single finished evaluation contains every required row and score"
    if case == "nested":
        assert any(d.info.end_time <= f.info.start_time for d in dry_runs for f in full_runs), (
            "Missing separately scored three-row dry run before full evaluation"
        )
    return "Complete native evaluation persisted all six outputs and exact scores; required dry-run ordering verified."


def get_judges() -> list:
    work = Path(os.environ["TEST_WORK_DIR"])
    expected = json.loads((work / "evaluation_expected.json").read_text())
    exp = os.environ["MLFLOW_EXPERIMENT_ID"]
    start = int(os.environ["RUN_START_MS"])
    # Validate once before the harness changes the active experiment or enters
    # parallel scorer workers; return native feedback on each participant trace.
    try:
        rationale = verify_behavior(expected, work, exp, start)
        value = "yes"
    except Exception as error:
        value, rationale = "no", f"{type(error).__name__}: {error}"

    @scorer(name="evaluation-behavior")
    def evaluation_behavior(trace) -> Feedback:
        return Feedback(value=value, rationale=rationale)

    return [evaluation_behavior]
