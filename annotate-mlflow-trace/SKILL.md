---
name: annotate-mlflow-trace
description: Tags traces and logs human feedback on them with the MLflow CLI. Use when the user asks to tag a trace, label a trace, rate a trace, leave feedback on a session, give a thumbs up or thumbs down, or mark a coding session for later evaluation. Triggers on "tag trace", "annotate trace", "label trace", "feedback on trace", "rate trace", "thumbs up", "thumbs down".
---

# Annotate MLflow Trace

Label and rate traces in place so sessions can be filtered and evaluated later, without switching
to the MLflow UI. Annotating a coding session is the step that turns raw traces into an evaluation
dataset: tag what the session was, log whether the outcome was good, then use
`agent-evaluation` on the traces you marked.

All operations use the `mlflow traces` CLI.

## Annotation Commands

| Goal | Command |
|------|---------|
| Set a tag | `mlflow traces set-tag --trace-id <id> --key <key> --value <value>` |
| Remove a tag | `mlflow traces delete-tag --trace-id <id> --key <key>` |
| Log human feedback | `mlflow traces log-feedback --trace-id <id> --name <name> --value <value> --source-type HUMAN` |
| Log ground truth | `mlflow traces log-expectation --trace-id <id> --name <name> --value <value>` |
| Read an assessment | `mlflow traces get-assessment --trace-id <id> --assessment-id <id>` |
| Correct an assessment | `mlflow traces update-assessment --trace-id <id> --assessment-id <id> --value <value>` |

Tags describe what a trace **is** and are the right tool for filtering. Feedback records a
judgement about how it **went** and is stored as an assessment with a source, so it survives
review and can be corrected later.

## Workflow

1. **Check CLI usage** (required): `mlflow traces set-tag --help` and `mlflow traces log-feedback --help`
2. **Identify the trace** to annotate
3. **Tag it, log feedback, or both**
4. **Confirm** with `mlflow traces get --trace-id <id>`

## Prerequisite: Check CLI Usage

```bash
mlflow traces set-tag --help
mlflow traces log-feedback --help
```

Always run this first to get accurate flags for the installed MLflow version.

## Identifying the Trace

When the user says "this session" or "the last run", find the trace before annotating it.
`--extract-fields` keeps the output to what you need.

```bash
# Most recent traces in the experiment
mlflow traces search --experiment-id 1 --order-by "timestamp_ms DESC" --max-results 5 \
    --extract-fields "info.trace_id,info.state,info.request_time"
```

For Claude Code sessions traced by `mlflow.claude_code`, scope the search to the current project.
MLflow 3.14+ records the session's working directory, so a shared experiment does not surface
another project's traces:

```bash
mlflow traces search --experiment-id 1 \
    --filter-string "metadata.\`mlflow.trace.working_directory\` = '$(pwd)'" \
    --order-by "timestamp_ms DESC" --max-results 5 \
    --extract-fields "info.trace_id"
```

On older MLflow versions that metadata is absent. Fall back to the most recent trace, and
confirm with the user before annotating if the experiment is shared.

Claude Code sessions log a `claude_code_conversation` trace alongside a short `env_snapshot`
trace. The conversation trace is the one to annotate:

```bash
mlflow traces search --experiment-id 1 \
    --filter-string "trace.name = 'claude_code_conversation'" \
    --order-by "timestamp_ms DESC" --max-results 1 \
    --extract-fields "info.trace_id"
```

## Tagging

```bash
# Label a session
mlflow traces set-tag --trace-id tr-abc123 --key quality --value good

# Tags are one key at a time; repeat the command for more
mlflow traces set-tag --trace-id tr-abc123 --key sprint --value 42
mlflow traces set-tag --trace-id tr-abc123 --key reviewer --value alice

# Remove a tag
mlflow traces delete-tag --trace-id tr-abc123 --key sprint
```

Find tagged traces later with the `retrieving-mlflow-traces` skill:

```bash
mlflow traces search --experiment-id 1 --filter-string "tag.quality = 'good'"
```

## Logging Feedback

Feedback is a human assessment. Pass `--source-type HUMAN` so it is distinguishable from
LLM-judge and automated scores during evaluation.

```bash
# Thumbs up
mlflow traces log-feedback --trace-id tr-abc123 --name thumbs_up --value true \
    --source-type HUMAN --source-id alice@example.com

# Rating with a justification
mlflow traces log-feedback --trace-id tr-abc123 --name quality --value good \
    --source-type HUMAN --source-id alice@example.com \
    --rationale "Completed the refactor correctly"

# Numeric score
mlflow traces log-feedback --trace-id tr-abc123 --name accuracy --value 0.9 \
    --source-type HUMAN
```

`--value` accepts a number, a string, a boolean, or JSON for structured values.

The command prints the new assessment ID. Keep it to read or correct the assessment later:

```bash
mlflow traces update-assessment --trace-id tr-abc123 \
    --assessment-id a-df258c2c --value bad --rationale "Missed an edge case on retry"
```

### Make feedback searchable

Assessments are not filterable the way tags are. When feedback should be findable in bulk, set a
companion tag alongside it:

```bash
mlflow traces log-feedback --trace-id tr-abc123 --name thumbs_up --value true --source-type HUMAN
mlflow traces set-tag --trace-id tr-abc123 --key has_feedback --value true
```

```bash
mlflow traces search --experiment-id 1 --filter-string "tag.has_feedback = 'true'"
```

## Verifying

```bash
mlflow traces get --trace-id tr-abc123 \
    --extract-fields "info.tags,info.assessments.*.feedback.value,info.assessments.*.source.source_type"
```

## Notes

- Set `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_ID` in the environment, or pass
  `--experiment-id` explicitly. `mlflow traces search` requires one of the two.
- Ground truth labels go through `log-expectation`, not `log-feedback`. Use it when recording
  what the answer should have been rather than how good the answer was.
- The `mlflow traces` CLI does not emit traces of its own. In a Claude Code session traced by
  `mlflow.claude_code`, the turn that runs these commands is still logged as a conversation trace
  by the `Stop` hook.
