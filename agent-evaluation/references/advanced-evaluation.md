# Evaluation datasets, comparisons, and continuous checks

Use this guide after the basic evaluation workflow in `../SKILL.md`. Check the installed MLflow API and the current GenAI documentation before running examples: dataset and trace fields vary across MLflow versions and tracking backends.

## Build a representative dataset from production traces

1. Search the intended experiment in a bounded time window for real requests, including successes, failures, slow requests, different users, and multi-turn sessions. For Unity Catalog traces, include `trace.timestamp_ms` bounds in the filter even when also filtering by experiment or tag; a result limit alone does not bound the scan. Use `retrieving-mlflow-traces` for the installed CLI's filter syntax. Inspect candidate traces and their assessments before choosing them; an `OK` execution status says nothing about answer quality.
2. Mark selected traces with a project-specific tag such as `eval_candidate=regression`, `eval_candidate=edge_case`, or `eval_candidate=happy_path`. Record why each case matters. Do not indiscriminately evaluate every production request; remove sensitive data and obtain permission to reuse it.
3. Discover existing GenAI datasets before creating one. Convert each selected trace into a record with `inputs` matching the agent's `predict_fn` parameters. Add `expectations` only when ground truth is known; preserve outputs only when assessing the original response rather than rerunning the agent. Use `dataset.merge_records(...)` from `dataset-preparation.md` and retain a trace ID or source tag for provenance.
4. Include normal, edge, adversarial, out-of-scope, multi-turn, and error-recovery cases. Keep known failures as a regression subset. Review the final records in the MLflow dataset UI before a full run.

On Databricks, a dataset may be backed by a Unity Catalog table. Confirm the chosen `catalog.schema` and permissions before creating it; see `dataset-preparation.md`. When converting a trace search DataFrame, inspect its actual columns: some versions expose `request`/`response`, whereas `merge_records()` expects records with `inputs` and optionally `outputs`/`expectations`. Do not rename columns by assumption or silently store a stringified trace as an input.

## Compare agent or prompt versions

Run the **same dataset and scorer definitions** against each candidate, in named MLflow runs. Record the code revision, model, prompt URI or version, dataset version, and judge version with each run. Compare per-scorer metrics, but also join row-level results by stable input or dataset record ID to find `pass → fail` regressions. Inspect the corresponding traces and rationales; an aggregate mean can improve while a critical user journey degrades.

For an A/B prompt test, change only the prompt version if possible. Keep the agent, model, dataset, and scorer suite fixed. Define the acceptance rule before looking at results (for example, no safety regression and improved task success), and use an aligned judge or human review for subjective criteria. A lower score from a newly aligned judge is not evidence that the agent regressed; compare candidates under the **same** judge version.

For precomputed responses, evaluate the `inputs` and `outputs` already in the records without a `predict_fn`. Use a `predict_fn` when the goal is to rerun the current agent. Do not mix these modes within one comparison.

## Continuous evaluation

Keep a small, versioned regression dataset of failures and a larger representative dataset for scheduled checks. In CI, first validate auth and tracing, then run a small dry run with the installed scorer suite. Fail on an explicit, stable quality threshold or a critical regression, and attach the MLflow run URL plus failing trace IDs to the CI result. Reserve expensive LLM-judge suites and broad production samples for scheduled runs; pin judge/model and dataset versions so comparisons remain meaningful. Use `throughput-guide.md` to set worker concurrency instead of writing a custom evaluation loop.

## Troubleshooting and API checks

- Use `mlflow.genai.evaluate()`, with nested `inputs`; `predict_fn` receives unpacked input keys. See `troubleshooting.md` for the common wrong shapes.
- Check required columns before selecting a scorer. `Correctness` needs ground truth and `RetrievalGroundedness` needs a `RETRIEVER` span.
- Inspect actual run metrics and result DataFrame columns before computing pass rates or joining runs. Do not assume every scorer returns a numeric metric or that all runs contain the same columns.
