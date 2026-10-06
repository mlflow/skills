# Tracing deployed Databricks agents

Use `databricks.md` first to choose and verify the experiment's Unity Catalog trace location. Then configure the deployment so the same tracking URI and numeric experiment ID are available at runtime. Never treat a successful local trace as proof that the deployed identity can write to UC.

## Databricks Apps

Provide the app's service identity with access to the selected experiment, UC schema, and SQL warehouse. Set the MLflow tracking URI, experiment ID, and any required tracing configuration as app environment variables or resources; avoid embedding a personal token. Initialize tracing in the app process before the request handler creates the first span.

## Model Serving

Instrument the model's `predict` or agent entry point with supported MLflow autologging or `@mlflow.trace` before deployment. Confirm the serving endpoint's identity can access the same destination. Calling an endpoint from a locally traced client does not establish that the endpoint's internal spans are captured.

## External OpenTelemetry clients

Check the installed MLflow and Databricks documentation for the supported OTLP ingestion endpoint, authentication, and resource attributes. Configure the exporter to that endpoint rather than an arbitrary workspace URL. Preserve trace and span IDs across services with standard W3C trace context. Third-party OTel clients may use GenAI semantic conventions rather than MLflow's own field names.

For each deployment, run one representative request and inspect its root and child spans. When the request uses retrieval or tools, verify those spans (including `RETRIEVER` when a retrieval scorer needs it), timing, token fields when available, and the actual trace destination. Follow `../SKILL.md`'s 60-second readback deadline and stop after a concrete blocker. If the deployment cannot reach the configured backend, report the failing identity and operation; do not silently switch to legacy workspace trace storage.
