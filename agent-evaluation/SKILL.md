---
name: agent-evaluation
description: Run MLflow GenAI evaluation on an existing agent, dataset, precomputed outputs, or traces. Discover and reuse datasets and registered scorers, validate a small run, compare results, and report quality failures. Use build-a-scorer to design criteria and fix-agent-issue to implement behavior changes.
allowed-tools: Read, Write, Bash, Grep, Glob, WebFetch
---

# Agent evaluation

- Establish purpose, important outcomes, known failures, data/trace selection, and agreed scorers from the request, project, and configuration. Ask only about unresolved decisions.
- Reuse explicit criteria and implementations. For missing/vague criteria, use `build-a-scorer` if available; otherwise define atomic criteria and the cheapest suitable native scorers before execution.

- Use `mlflow.genai.datasets`, built-ins / `@scorer` / `make_judge`, and `mlflow.genai.evaluate()`; no parallel evaluation framework.
- Accept existing records/DataFrames; persist new reusable datasets through the GenAI dataset API.

## Step 1: Reuse setup and select execution mode

- Preserve `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID`, provider credentials, and configured judge model. If
  both MLflow variables are set, skip setup: no `setup_mlflow.py`, replacement backend, new experiment, or
  `.env` override. For missing configuration, read [setup-guide.md](references/setup-guide.md).
- Use the project environment (`uv run` for uv-managed projects). Check MLflow >=3.8 and installed signatures
  for the APIs needed. Resolve bundled script paths relative to this skill, not the application directory.
  Keep stderr separate when capturing structured stdout.
- **Rerun mode:** inspect the real agent entry point and its configuration. Use
  `instrumenting-with-mlflow-tracing` when available for instrumentation; reuse working tracing. Run a
  representative invocation, flush and fetch its persisted trace in the intended experiment. If tracing is
  missing/broken, repair and verify it before evaluation; do not add duplicate framework spans.
- **Precomputed mode:** evaluate the supplied `inputs` and `outputs`, or existing traces, **without
  `predict_fn`**. Do not invoke the agent, change stored records, or create a replacement dataset. Check
  scorer data requirements against existing evidence; a final-output scorer does not require new
  instrumentation.

## Step 2: Reuse and register scorers

- List registered scorers: `mlflow scorers list -x "$MLFLOW_EXPERIMENT_ID"`. Reuse the suite, models, and definitions; add only missing agreed scorers.
- Read [scorers.md](references/scorers.md) for model selection, registration, and retrieval. Register new scorers before evaluation; verify them in the intended experiment.

- Check installed `required_columns` and trace requirements: `Correctness` needs ground truth; retrieval scorers need `RETRIEVER`, not just `TOOL`, spans.
- Aggregate booleans, numbers, or `"yes"`/`"no"`. Stored `"pass"`/`"fail"` scores may disappear from aggregates.
- For custom scorers or missing scores, read [scorers-constraints.md](references/scorers-constraints.md).

## Step 3: Discover and bind data

- Run `scripts/list_datasets.py` before creating data. Inspect candidates; reuse a suitable or explicitly selected dataset. Discovery failure does not mean none exists.
- For UC, use the intended fully qualified table and backend-supported discovery. Ask only if selection is ambiguous.

Read [dataset-preparation.md](references/dataset-preparation.md) when creating, adapting, or synthesizing
records. Inspect actual record keys and the predictor signature:

```python
# MLflow calls predict_fn(**record["inputs"]). Nested values stay nested.
record = {"inputs": {"query": "Where is order 42?", "context": {"order_id": 42}}}
def predict(query, context):
    return agent(query=query, context=context)  # bind project config here as needed
```

- Accept `inputs` as a wrapper argument only if the dataset has that input key.
- Put ground truth under `expectations`, e.g. `expectations.expected_response`. Preserve multi-key inputs and source IDs.
- Size data to the task/coverage; no mandatory 100-question expansion or synthesis dependency.

For requested corpus synthesis, use the [RAGAS recipe](references/dataset-preparation.md#ragas-corpus-synthesis).

## Step 4: Dry run: required before a full evaluation

Evaluate **three representative records**, or the user's smaller explicit budget, with the full run's mode, wrapper, and scorers. Use one pilot, not separate sanity/dry runs. Inspect every row:

- Did the predictor finish, required tools/retrieval execute, and usable output persist? Tools need not run
  for greetings or other cases that do not require them.
- Did every applicable scorer return a valid assessment without an error? Check both row-level
  values/rationales and aggregate inclusion; a successful trace status or run alone is insufficient.
- **Zero, `False`, and `"no"` are valid quality failures.** Investigate suspicious uniform scores against
  actual responses and known-good/bad cases; never require nonzero scores or change the rubric to manufacture
  passes.

- **Stop before the full run** on auth/tool failures, failed prediction, empty rerun responses, or missing/error-only scores. Diagnose, repair within scope, and repeat the pilot.
- Preserve broken precomputed outputs as evidence. Distinguish scored negatives from scorer/infrastructure failures.
- Report blockers if repair is unavailable. Never invent credentials or switch providers to retry.

## Step 5: Execute and inspect

- Reruns: `mlflow.genai.evaluate(data=selected_data, predict_fn=predict, scorers=registered_scorers)`. Precomputed: omit `predict_fn`.
- Optional rerun template: `scripts/run_evaluation_template.py`. Inspect/adapt its wrapper, dataset, experiment, and result-table accessor before running.

- Estimate duration from pilot throughput. For expensive runs, read [throughput-guide.md](references/throughput-guide.md); honor resource limits and host-supported execution methods.
- Wait for completion. Keep exit status, run ID/URL, elapsed time, and row-level artifact path; launch alone is not success.

## Verification

- Verify a fresh run in the intended experiment, with row-level results for every selected record and applicable scorer.
- Preserve outputs, trace IDs, values, rationales, and assessment errors.
- Report evaluated, failed, missing, skipped, and errored counts; do not hide missing rows in pass-rate denominators. Metric-only runs and empty score tables are not success.

- Summarize criteria separately; link representative failures to evidence and state small-sample limits. Optional [report outline](assets/evaluation_report_template.md).
- `scripts/analyze_results.py` accepts CSV (`<scorer>/value`, `<scorer>/rationale`) or legacy JSON. Check summaries against raw typed scores/errors; generated grades/recommendations do not establish root cause.
- Requested behavior fixes: hand results to `fix-agent-issue`.

## References

- Infrastructure, schema, or scorer failure: [troubleshooting.md](references/troubleshooting.md).
- Production-trace curation, version comparisons, or continuous evaluation:
  [advanced-evaluation.md](references/advanced-evaluation.md).
- Databricks production scoring:
  [databricks-online-monitoring.md](references/databricks-online-monitoring.md); registration alone does not
  start monitoring.
- Judge disagreement, domain-expert labels, Review App, or MemAlign: `build-a-scorer` and its [alignment
  reference](../build-a-scorer/references/judge-alignment.md), when installed.
- Registered-prompt optimization / GEPA: [prompt-optimization.md](references/prompt-optimization.md).

- API details: [evaluation quickstart](https://mlflow.org/docs/latest/genai/eval-monitor/quickstart.md) or the reference's exact topic link. Check installed-version compatibility.
- Missing/moved page: [llms.txt](https://mlflow.org/docs/latest/llms.txt).
- Docs/skill unavailable: use local instructions and installed API introspection; stop and explain unresolved correctness. Do not silently substitute legacy APIs, backend, or provider.
