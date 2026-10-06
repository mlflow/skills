# Tracing deployed Databricks agents

Use `databricks.md` first to choose and verify the experiment's Unity Catalog trace location. Then configure the deployment so the same tracking URI and numeric experiment ID are available at runtime. Never treat a successful local trace as proof that the deployed identity can write to UC.

## Databricks Apps

Provide the app's service identity with access to the selected experiment, UC schema, and SQL warehouse. Set the MLflow tracking URI, experiment ID, and any required tracing configuration as app environment variables or resources; avoid embedding a personal token. Initialize tracing in the app process before the request handler creates the first span. Call a test route, flush asynchronous logging where possible, and verify the resulting trace in the intended experiment and UC location.

## Model Serving

Instrument the model's `predict` or agent entry point with supported MLflow autologging or `@mlflow.trace` before deployment. Confirm the serving endpoint's identity can access the same destination. Invoke the deployed endpoint once and check for a trace with root inputs/outputs and meaningful child spans. Do not assume calling an endpoint from a locally traced client captures the endpoint's internal spans.

## External OpenTelemetry clients

Check the installed MLflow and Databricks documentation for the supported OTLP ingestion endpoint, authentication, and resource attributes. Configure the exporter to that endpoint rather than an arbitrary workspace URL. Preserve trace and span IDs across services with standard W3C trace context. Emit a test span and inspect raw attributes: third-party OTel clients may use GenAI semantic conventions rather than MLflow's own field names. A trace appearing in the UI still requires the separate UC storage check in `databricks.md`.

For each deployment, test a representative LLM call, retrieval operation (`RETRIEVER` span when using retrieval scorers), tool call, and failure. Verify span type, parent-child relationships, timing, token fields when available, and an actual trace link. Follow `../SKILL.md`'s 60-second readback deadline and stop after a concrete blocker. If any deployment cannot reach the configured backend, report the failing identity and operation; do not silently switch to legacy workspace trace storage.
