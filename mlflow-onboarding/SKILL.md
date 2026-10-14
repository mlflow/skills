---
name: mlflow-onboarding
description: Set up or connect MLflow for GenAI tracing or traditional ML training. Use for initial integration or missing backend configuration; reuse an already configured environment.
---

# Set up MLflow

## Inputs

Use the project, requested workflow, and supplied tracking URI/experiment. Inspect the
existing MLflow configuration before creating anything. Reuse working Databricks-provided
configuration, including the profile, credentials and experiment; no second setup flow
is needed. Change the backend only when requested. Never print credentials.

## Workflow

1. **Identify the use case.** Agent/LLM calls imply GenAI; model training implies classic
   ML. If an experiment is supplied, inspect its `mlflow.experimentKind` tag:
   `genai_development` or `custom_model_development`. The MLflow repository itself
   contains both and is not evidence of user intent. Ask only if these signals and the
   request are inconclusive.
2. **Verify the selected backend.** Check the configured experiment using the installed
   MLflow client. Resolve missing configuration; report authentication or connectivity
   blockers without silently switching to a local server or creating another experiment.
   For a SageMaker ARN, use `sagemaker-mlflow` when installed.
3. **Integrate the requested path.** Reuse existing instrumentation/training setup.
   A request to implement the integration authorizes those edits; a request for an
   explanation or tutorial does not. Load only the relevant guide below, checking
   compatibility with the installed MLflow version.

| Path | Integration | Guide |
|---|---|---|
| GenAI | Enable the framework's autologging once before calls. Use manual spans only for otherwise unobserved work. | [Tracing quickstart](https://mlflow.org/docs/latest/genai/tracing/quickstart/) |
| Classic ML | Enable the selected framework's autologging before training, or log parameters, metrics and model artifacts explicitly. | [Tracking quickstart](https://mlflow.org/docs/latest/ml/tracking/quickstart/) |
| Deep learning | Log training parameters, metrics, checkpoints and the model. | [Deep learning](https://mlflow.org/docs/latest/ml/getting-started/deep-learning/) |

For GenAI instrumentation, use `instrumenting-with-mlflow-tracing` if installed; otherwise
follow the quickstart and verification contract below. Do not require an LLM API key for
an introductory example: `@mlflow.trace` around a deterministic mock, with child spans
for retrieval and response generation, demonstrates persisted inputs/outputs without a
provider call. Clearly label mock outputs.

## Verify

**GenAI:** execute the app once, flush tracing, and fetch the captured trace ID. If the ID
is unavailable, search the selected experiment with a `trace.timestamp_ms` lower bound
from the run start and `max_results=1`. Confirm expected spans and inputs/outputs.
Enforce a **60-second wall-clock deadline** on readback, including SDK retries; cancel
when it expires. Retry at most once after a concrete quick fix within the same budget.
Missing API keys, authentication, permissions or unavailable/slow SQL warehouses are
blockers: report the observed error, what was verified, what was not, and the needed
user action. Do not rerun indefinitely or try SQL, raw REST or alternate authentication.

**Classic ML:** run training and read back the specific new run from the selected
experiment. Confirm logged parameters/metrics and the model artifact. When model
logging is requested, reload the model and verify predictions on a small held-out input;
a run ID alone does not establish a working model.

Return the experiment and trace/run ID, the code/configuration changed, and the observed
result. Do not claim verification from a successful API call without reading its artifacts.

## Read when needed

- Quality evaluation: [evaluation quickstart](https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/).
- Versioned prompts: [prompt registry](https://mlflow.org/docs/latest/genai/prompt-registry/).
  On Databricks, use a three-part `catalog.schema.prompt_name` and enable the Prompt
  Registry preview (account admin → Previews) when required by the workspace. Registration
  returns a `prompts:/.../version` URI and a workspace UI link. If the active experiment's
  Prompts tab is empty on older clients, repair its two-part location tag:
  `mlflow.set_experiment_tag("mlflow.promptRegistryLocation", "catalog.schema")`.
- Missing or moved documentation: look up the topic in
  [the official index](https://mlflow.org/docs/latest/llms.txt), then fetch only the relevant
  page. If unavailable, use installed API signatures and the local recipe; stop when
  correctness cannot be established rather than inventing a replacement API.
