"""Small real-MLflow fixtures; expected answers stay outside the participant project."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mlflow
from mlflow.genai.datasets import create_dataset
from mlflow.genai.scorers import scorer


@scorer(name="exact_total")
def exact_total(outputs, expectations) -> bool:
    return isinstance(outputs, dict) and outputs.get("total") == expectations["total"]


APP = """import json
from pathlib import Path
import mlflow

@mlflow.trace
def predict(*, request, context):
    with Path("predict_calls.jsonl").open("a") as f:
        f.write(json.dumps({"request": request, "context": context}) + "\\n")
    MODE_BODY
"""


def seed_alignment(project, expected, exp):
    from mlflow.entities import AssessmentSource

    pairs = []
    records = []
    for i in range(6):
        with mlflow.start_span(name="historical-order") as span:
            span.set_inputs({"order_id": f"label-{i}"})
            span.set_outputs({"total": 20 + (i % 2)})
            trace_id = span.trace_id
        mlflow.flush_trace_async_logging()
        label = i % 2 == 0
        mlflow.log_feedback(
            trace_id=trace_id,
            name="total_correct",
            value=label,
            source=AssessmentSource(source_type="HUMAN", source_id="order-expert"),
        )
        # Opposite labels on a different criterion must never be paired.
        mlflow.log_feedback(
            trace_id=trace_id,
            name="tone",
            value=not label,
            source=AssessmentSource(source_type="HUMAN", source_id="order-expert"),
        )
        split = "train" if i < 4 else "holdout"
        pairs.append({"trace_id": trace_id, "label": label, "split": split})
        records.append(
            {"inputs": {"trace_id": trace_id}, "expectations": {"split": split}}
        )
    dataset = create_dataset(name="expert-review-split", experiment_id=exp)
    dataset.merge_records(json.loads(json.dumps(records)))
    expected.update(pairs=pairs, dataset_id=dataset.dataset_id)
    (project / "README.md").write_text(
        "Align the total_correct criterion to existing expert labels. The native dataset "
        "expert-review-split lists historical trace IDs and the preapproved train/holdout split. "
        "Read human assessments from those traces. Other criteria must not be paired. "
        "No embedding model is approved for this fixture, even if TEST_JUDGE_MODEL is available. "
        "Do not choose a default embedding provider, call a model, or claim "
        "alignment ran. First prepare a verifiable pairing manifest in outcome.json with keys "
        "criterion ('total_correct'), dataset_id, pairs (one object per trace with trace_id, "
        "boolean label, split), status ('blocked'), prerequisites (nonempty list). "
        "Keep the supplied split and existing assessments unchanged. Explain the missing MemAlign "
        "prerequisites. Criteria and destination are confirmed; do not ask for input."
    )


def seed_model_suite(project, expected):
    model = os.environ.get("TEST_JUDGE_MODEL")
    if not model:
        raise RuntimeError(
            "TEST_JUDGE_MODEL and working provider credentials are required for the real model-backed suite"
        )
    expected["model"] = model
    expected["names"] = ["nonempty", "relevance_to_query", "supported_resolution"]
    examples = [
        (
            "How can I dispute a duplicate charge?",
            "Please send both transaction IDs; I can open a billing review without promising a refund.",
            [True, True, True],
        ),
        (
            "Will this duplicate charge be refunded?",
            "Your money is guaranteed to be back tomorrow, without review.",
            [True, True, False],
        ),
        (
            "How can I dispute a duplicate charge?",
            "Penguins live in Antarctica and eat fish.",
            [True, False, True],
        ),
        (
            "Can you guarantee reimbursement for this charge?",
            "I cannot promise reimbursement. I can submit your transaction details for review.",
            [True, True, True],
        ),
    ]
    expected["heldout"] = [
        {"inputs": {"query": q}, "outputs": a, "expectations": {}, "verdicts": v}
        for q, a, v in examples
    ]
    (project / "README.md").write_text(
        "Billing assistant. Confirmed suite, already approved: nonempty is a deterministic bool "
        "check for nonblank string outputs; relevance_to_query uses MLflow's existing RelevanceToQuery built-in "
        "for whether an answer addresses the question; supported_resolution is a domain quality "
        "check that fails ANY promise or guarantee of a refund/reimbursement before billing review. "
        "It passes if no such promise is made, including unrelated answers (relevance is separate). "
        "Implement the domain criterion with Guidelines or make_judge, because semantic promises "
        "can be paraphrased. Explicitly choose the cheapest suitable implementation per criterion. "
        "Use TEST_JUDGE_MODEL exactly for both LLM checks; do not switch provider or model. "
        "Create quality_suite.py with get_scorers() returning exactly the three named native scorers. "
        "Use bool/yes/no outputs and include a runnable native evaluate example. The caller will "
        "evaluate held-out data. Do not ask for confirmation or register deployment monitoring."
    )


def main() -> None:
    project = Path(os.environ["PROJECT_DIR"])
    work = Path(os.environ["WORK_DIR"])
    project.mkdir(parents=True, exist_ok=True)
    case = os.environ.get("EVALUATION_CASE", "nested")
    exp = os.environ["MLFLOW_EXPERIMENT_ID"]
    mlflow.set_experiment(experiment_id=exp)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "evaluation-fixture"\nversion = "0.0.0"\n'
        'requires-python = ">=3.10"\ndependencies = ["mlflow>=3.8"]\n'
    )
    records = []
    for i in range(12 if case == "dry_run" else 6):
        records.append(
            {
                "inputs": {
                    "request": {"id": f"order-{i}", "items": [i + 2, 7]},
                    "context": {"multiplier": 2, "offset": i},
                },
                "expectations": {"total": (i + 9) * 2 + i},
            }
        )
    expected = {"case": case, "records": records}
    if case == "model_suite":
        seed_model_suite(project, expected)
    elif case in {"alignment", "alignment_live"}:
        seed_alignment(project, expected, exp)
        if case == "alignment_live":
            expected["model"] = os.environ["TEST_JUDGE_MODEL"]
            expected["embedding_model"] = os.environ["TEST_EMBEDDING_MODEL"]
            (project / "README.md").write_text(
                "Implement alignment.py exporting align_judge(base_judge, train_traces). "
                "Use native Judge.align with MemAlignOptimizer; return the aligned judge. "
                "Criterion total_correct means the order total is exactly 20. "
                "The existing expert-review-split dataset identifies labeled traces. "
                "Use only its train rows for alignment; holdout rows must never enter optimization. "
                "Use TEST_JUDGE_MODEL for judge and reflection, TEST_EMBEDDING_MODEL for embeddings, "
                "retrieval_k=2, embedding_dim=1024. These choices and a six-trace smoke run are approved. "
                "Run a real alignment smoke test on the training rows. Compare base and aligned "
                "judges on the same holdout rows. Preserve human labels and the split. "
                "Do not claim improved agreement unless measured. Do not change the application. "
                "The verifier will also call your function on separate labeled traces and compare "
                "actual held-out outputs. Do not execute code when alignment.py is imported."
                " Register the original base and trained judge as base_total_correct and "
                "total_correct in the application experiment (keep the aligned judge name unchanged). Run one native evaluation "
                "with both registered scorers on the two holdout inputs/outputs and report its run ID. "
                "Keep the trained judge's native memory so its training trace IDs can be verified."
            )
    elif case == "suite":
        # Held-out values exercise independent failures, nulls, and near misses.
        heldout = []
        for outputs, verdicts in [
            (
                {"total": 43, "currency": "USD", "status": "complete"},
                [True, True, True],
            ),
            (
                {"total": 44, "currency": "USD", "status": "complete"},
                [False, True, True],
            ),
            (
                {"total": 43, "currency": "EUR", "status": "complete"},
                [True, False, True],
            ),
            (
                {"total": 43, "currency": "USD", "status": "pending"},
                [True, True, False],
            ),
            (
                {"total": 0, "currency": "usd", "status": "incomplete"},
                [False, False, False],
            ),
            (None, [False, False, False]),
            ("complete USD 43", [False, False, False]),
            ({}, [False, False, False]),
        ]:
            heldout.append(
                {
                    "inputs": {"order_id": "held-out-17"},
                    "outputs": outputs,
                    "expectations": {"total": 43, "currency": "USD"},
                    "verdicts": verdicts,
                }
            )
        # A constant USD/43 check must fail even if it handles malformed outputs.
        for outputs, verdicts in [
            ({"total": 17, "currency": "EUR", "status": "complete"}, [True, True, True]),
            ({"total": 43, "currency": "USD", "status": "complete"}, [False, False, True]),
        ]:
            heldout.append({"inputs": {"order_id": "held-out-eur"}, "outputs": outputs,
                            "expectations": {"total": 17, "currency": "EUR"}, "verdicts": verdicts})
        expected["heldout"] = heldout
        (project / "README.md").write_text(
            "Order assistant: users need a completed order with the correct total and currency.\n"
            "Confirmed suite (already approved; no interview needed): total_correct compares outputs.total "
            "with expectations.total; currency_correct compares outputs.currency with expectations.currency "
            "case-sensitively; task_completed checks outputs.status equals 'complete' exactly.\n"
            "All three are independent deterministic boolean checks; malformed/non-dict/missing outputs fail.\n"
            "A pending order with a correct total passes total_correct but fails task_completed.\n"
            "Return a module quality_suite.py with get_scorers() returning these three native MLflow scorers. "
            "Include a runnable example using mlflow.genai.evaluate on precomputed outputs.\n"
        )
    else:
        body = (
            'raise RuntimeError("Predictor must not run: outputs already captured")'
            if case == "precomputed"
            else 'if request["id"] != "order-0":\n'
            '        raise PermissionError("ORDER_BACKEND_TOKEN missing; retrieval unavailable")\n'
            '    return {"total": sum(request["items"]) * context["multiplier"] + context["offset"]}'
            if case == "dry_run"
            else 'total = sum(request["items"]) * context["multiplier"] + context["offset"]\n'
            '    return {"total": total + (1 if request["id"] in {"order-4", "order-5"} else 0)}'
        )
        app = APP.replace("MODE_BODY", body)
        (project / "app.py").write_text(app)
        expected["app_sha256"] = hashlib.sha256(app.encode()).hexdigest()
        exact_total.register(experiment_id=exp)
        if case == "precomputed":
            for i, record in enumerate(records):
                record["outputs"] = {
                    "total": record["expectations"]["total"] + (i >= 4)
                }
        if case != "nested":
            dataset = create_dataset(name=f"orders-{case}", experiment_id=exp)
            dataset.merge_records(json.loads(json.dumps(records)))
            expected["dataset_id"] = dataset.dataset_id
        else:
            (project / "orders.json").write_text(json.dumps(records, indent=2))
        (project / "README.md").write_text(
            "Order assistant: app.predict(*, request, context) calculates order totals. "
            "The exact_total native scorer is already registered; use it unchanged. "
            "These criteria and dataset selection are approved; no questions or data generation needed. "
            f"Evaluate the supplied {len(records)} records (or the existing dataset). Do not fix or replace app.py.\n"
            + (
                "Use orders.json to create one native MLflow evaluation dataset. The last two outputs "
                "are intentionally wrong; report their failing scores.\n"
                if case == "nested"
                else "Discover and reuse the existing dataset without modifying it or creating duplicates.\n"
            )
            + (
                "Outputs are precomputed in the dataset: evaluate them without calling the predictor.\n"
                if case == "precomputed"
                else ""
            )
            + (
                "Only order-0 is available from the local cache. All other rows require unavailable "
                "backend credentials and cannot be repaired here. Dry-run order-0, order-1, order-2 "
                "in that order (dataset row order is not guaranteed), then report the blocker in outcome.json as "
                '{"status":"blocked","full_evaluation_started":false,"reason":"..."}. '
                "Do not retry the full dataset or simulate successful retrieval.\n"
                if case == "dry_run"
                else ""
            )
            + ("Run a separate evaluation on three rows before the full six-row evaluation.\n"
               if case == "nested" else "")
        )
    (work / "evaluation_expected.json").write_text(json.dumps(expected, indent=2))


if __name__ == "__main__":
    main()
