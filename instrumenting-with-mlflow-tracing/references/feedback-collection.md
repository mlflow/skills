# Feedback collection

Use when implementing authorized user ratings/comments tied to a concrete trace. Capture the active trace ID inside the request (`mlflow.get_current_active_span().trace_id` in Python; `mlflow.getCurrentActiveSpan()?.traceId` in TypeScript) and return/store it for later feedback. Do not confuse it with a run ID.

## Record the judgment and source

Check the installed SDK supports these public APIs:

```python
import mlflow
from mlflow.entities import AssessmentSource

mlflow.log_feedback(
    trace_id=trace_id, name="user_rating", value=rating,
    source=AssessmentSource(source_type="HUMAN", source_id=reviewer_id),
)
```

```typescript
import * as mlflow from "mlflow-tracing";
await mlflow.logFeedback({
  traceId, name: "user_rating", value: rating,
  source: { sourceType: "HUMAN", sourceId: reviewerId },
});
```

Values may be numeric ratings, text comments or boolean judgments. Preserve actual source identity and meaning: HUMAN for user/manual review, LLM_JUDGE for automated model judgments, CODE for programmatic checks. Do not label generated feedback as human. A zero/false value is not a logging failure; a scorer error is not a negative judgment.

Expected-output labels are expectations, not feedback judgments. For requested scorer alignment or Review App labeling use `build-a-scorer` when available. For feedback field/source compatibility use the [entity API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.entities.html) and installed TypeScript typings. Verify the stored assessment belongs to the intended trace and retains value/source; recording user complaints during diagnosis still requires authorization.
