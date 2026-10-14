---
name: agent-evaluation
description: Run MLflow GenAI evaluation on an existing agent, dataset, precomputed outputs, or traces. Discover and reuse datasets and registered scorers, validate a small run, compare results, and report quality failures. Use build-a-scorer to design criteria and fix-agent-issue to implement behavior changes.
allowed-tools: Read, Write, Bash, Grep, Glob, WebFetch
---

# Agent evaluation

## Inputs

Establish the agent's purpose, important outcomes, known failures, dataset or trace selection, and agreed
scorer suite from the request, project, and existing configuration. Ask only for unresolved decisions; do not
repeat an answered interview. If criteria are vague or absent, use `build-a-scorer` when available; otherwise
establish atomic criteria and the cheapest suitable native scorers before execution. Reuse explicit criteria
and implementations without another design cycle.

Use `mlflow.genai.datasets`, native built-ins / `@scorer` / `make_judge`, and `mlflow.genai.evaluate()`. Do
not build a parallel evaluation framework. Existing records and DataFrames are valid evaluation inputs;
persist new reusable datasets through the GenAI dataset API.

## Workflow

### 1. Reuse setup and select execution mode

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

### 2. Reuse and register scorers

List the experiment's registered scorers (`mlflow scorers list -x "$MLFLOW_EXPERIMENT_ID"`) and use the
existing suite. Add only missing agreed scorers; preserve their model and definition. Read
[scorers.md](references/scorers.md) for model selection, registration, and retrieval. Register new scorers
before evaluation and verify they appear in the intended experiment.

Check each scorer's installed `required_columns` and trace requirements. `Correctness` needs ground truth;
retrieval scorers need `RETRIEVER` spans, not merely `TOOL` spans. Use booleans, numeric values, or
`"yes"`/`"no"` for aggregation. `"pass"`/`"fail"` can be stored yet disappear from aggregate metrics. Read
[scorers-constraints.md](references/scorers-constraints.md) when implementing a custom scorer or diagnosing
missing scores.

### 3. Discover and bind data

Run `scripts/list_datasets.py` before creating a dataset; inspect the relevant candidates and reuse a suitable
or explicitly selected one. A discovery error is not evidence that no dataset exists. For Unity Catalog, use
the intended fully qualified table and backend-supported discovery. Clarify only ambiguous selection.

Read [dataset-preparation.md](references/dataset-preparation.md) when creating, adapting, or synthesizing
records. Inspect actual record keys and the predictor signature:

```python
# MLflow calls predict_fn(**record["inputs"]). Nested values stay nested.
record = {"inputs": {"query": "Where is order 42?", "context": {"order_id": 42}}}
def predict(query, context):
    return agent(query=query, context=context)  # bind project config here as needed
```

A wrapper taking `inputs` is wrong unless the dataset itself has an input key named `inputs`. Put ground truth
under `expectations`, such as `expectations.expected_response`. Preserve multi-key inputs and source IDs. Size
the dataset to the requested task and coverage; do not mandate a 100-question expansion or synthesis
dependency.

### 4. Dry run: required before a full evaluation

Evaluate **three representative records**, or the user's smaller explicit budget, using the exact mode,
wrapper, and scorers planned for the full run. This single pilot replaces separate sanity and dry-run
campaigns. Inspect every row:

- Did the predictor finish, required tools/retrieval execute, and usable output persist? Tools need not run
  for greetings or other cases that do not require them.
- Did every applicable scorer return a valid assessment without an error? Check both row-level
  values/rationales and aggregate inclusion; a successful trace status or run alone is insufficient.
- **Zero, `False`, and `"no"` are valid quality failures.** Investigate suspicious uniform scores against
  actual responses and known-good/bad cases; never require nonzero scores or change the rubric to manufacture
  passes.

**Stop before the full run** on auth/tool failures, failed prediction, empty responses in rerun mode, or
missing/error-only scorer results. Diagnose the cause, repair within scope, and repeat the pilot. Preserve
broken precomputed outputs as evidence rather than repairing their data; distinguish genuine scored negatives
from scorer/infrastructure failures. Report a blocked outcome honestly if repair is unavailable. Do not retry
by inventing credentials or switching providers.

### 5. Execute and inspect

Use `mlflow.genai.evaluate(data=selected_data, predict_fn=predict, scorers=registered_scorers)` for reruns;
omit `predict_fn` for precomputed data. `scripts/run_evaluation_template.py` is a starting point for rerun
scripts: inspect/adapt the generated wrapper, selected dataset, experiment, and result-table accessor before
running it.

Estimate duration from measured pilot throughput. For expensive runs read
[throughput-guide.md](references/throughput-guide.md), honor resource limits, and use the execution method the
host supports. Wait for actual completion; retain exit status, run ID/URL, elapsed time, and row-level
artifact path. A launched process is not a completed evaluation.

## Verify and report

Verify a fresh evaluation run in the intended experiment and actual row-level results for every selected
record and applicable scorer. Preserve outputs, trace IDs, values, rationales, and assessment errors. Report
evaluated, failed, missing, skipped, and errored counts explicitly; do not hide missing rows in a pass-rate
denominator. A metric-only run or empty score table is not success.

Summarize each criterion separately and link representative failures to their evidence.
`scripts/analyze_results.py` accepts result CSV (`<scorer>/value`, `<scorer>/rationale`) or legacy JSON, but
verify its summary against raw typed scores and errors; generated grades or recommendations are not root-cause
evidence. Use the optional [report outline](assets/evaluation_report_template.md). State small-sample limits.
For requested behavior fixes, hand the results to `fix-agent-issue`.

## Read when needed

- Infrastructure, schema, or scorer failure: [troubleshooting.md](references/troubleshooting.md).
- Production-trace curation, version comparisons, or continuous evaluation:
  [advanced-evaluation.md](references/advanced-evaluation.md).
- Databricks production scoring:
  [databricks-online-monitoring.md](references/databricks-online-monitoring.md); registration alone does not
  start monitoring.
- Judge disagreement, domain-expert labels, Review App, or MemAlign: `build-a-scorer` and its [alignment
  reference](../build-a-scorer/references/judge-alignment.md), when installed.
- Registered-prompt optimization / GEPA: [prompt-optimization.md](references/prompt-optimization.md).

For API details, start with the [evaluation
quickstart](https://mlflow.org/docs/latest/genai/eval-monitor/quickstart.md) or a reference's exact topic
link. Check installed-version compatibility before copying examples. Use
[llms.txt](https://mlflow.org/docs/latest/llms.txt) only for a missing/moved page. If docs or a sibling skill
are unavailable, use this skill's local contract and installed API introspection; stop and explain when
correctness cannot be established. Do not silently substitute legacy APIs, backend, or provider.
