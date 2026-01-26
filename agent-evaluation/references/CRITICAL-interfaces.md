# Critical MLflow 3 GenAI Interfaces

**Purpose**: Implementation details NOT available in MLflow official documentation.
**For general API reference**: Use documentation protocol (llms.txt)
**For common mistakes**: See GOTCHAS.md

---

## Table of Contents

- [Data Schema](#data-schema)
- [Custom Scorer Return Types](#custom-scorer-return-types)
- [Judges API (Low-level)](#judges-api-low-level)
- [Trace Search Filter Syntax](#trace-search-filter-syntax)
- [Trace Object Access](#trace-object-access)
- [SpanType Constants](#spantype-constants)
- [Production Monitoring API](#production-monitoring-api)

---

## Data Schema

### Evaluation Dataset Record

```python
# CORRECT format
record = {
    "inputs": {                    # REQUIRED - passed to predict_fn
        "customer_name": "Acme",
        "query": "What is X?"
    },
    "outputs": {                   # OPTIONAL - pre-computed outputs
        "response": "X is..."
    },
    "expectations": {              # OPTIONAL - ground truth for scorers
        "expected_facts": ["fact1", "fact2"],
        "expected_response": "X is...",
        "guidelines": ["Must be concise"]
    }
}
```

**CRITICAL Schema Rules**:
- `inputs` is REQUIRED - contains what's passed to your app
- `outputs` is OPTIONAL - if provided, predict_fn is skipped
- `expectations` is OPTIONAL - used by Correctness, ExpectationsGuidelines

---

## Custom Scorer Return Types

### Function-based Scorer (Decorator)

```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

@scorer
def my_scorer(
    inputs: dict,          # From data record
    outputs: dict,         # App outputs or pre-computed
    expectations: dict,    # From data record (optional)
    trace: Trace = None    # Full MLflow Trace object (optional)
) -> Feedback | bool | int | float | str | list[Feedback]:
    """Custom scorer implementation"""

    # Return options:
    # 1. Simple value (metric name = function name)
    return True

    # 2. Feedback object with custom name
    return Feedback(
        name="custom_metric",
        value="yes",  # or "no", True/False, int, float
        rationale="Explanation of score"
    )

    # 3. Multiple feedbacks
    return [
        Feedback(name="metric_1", value=True),
        Feedback(name="metric_2", value=0.85)
    ]
```

### Class-based Scorer

```python
from mlflow.genai.scorers import Scorer
from mlflow.entities import Feedback
from typing import Optional

class MyScorer(Scorer):
    name: str = "my_scorer"  # REQUIRED
    threshold: int = 50      # Custom fields allowed (Pydantic)

    def __call__(
        self,
        outputs: str,
        inputs: dict = None,
        expectations: dict = None,
        trace = None
    ) -> Feedback:
        if len(outputs) > self.threshold:
            return Feedback(value=True, rationale="Meets length requirement")
        return Feedback(value=False, rationale="Too short")

# Usage
my_scorer = MyScorer(threshold=100)
```

---

## Judges API (Low-level)

### Import Path
```python
from mlflow.genai.judges import (
    meets_guidelines,
    is_correct,
    is_safe,
    is_context_relevant,
    is_grounded,
    make_judge,
)
```

### meets_guidelines()
```python
from mlflow.genai.judges import meets_guidelines

feedback = meets_guidelines(
    name="my_check",                    # Optional display name
    guidelines="Must be professional",   # str or List[str]
    context={                           # Dict with data to evaluate
        "request": "user question",
        "response": "app response",
        "retrieved_documents": [...]     # Can include any keys
    },
    model="databricks:/endpoint"        # Optional custom model
)
# Returns: Feedback(value="yes"|"no", rationale="...")
```

### is_correct()
```python
from mlflow.genai.judges import is_correct

feedback = is_correct(
    request="What is MLflow?",
    response="MLflow is an open-source platform...",
    expected_facts=["MLflow is open-source"],  # OR expected_response
    model="databricks:/endpoint"               # Optional
)
```

### make_judge() - Custom LLM Judge
```python
from mlflow.genai.judges import make_judge

issue_judge = make_judge(
    name="issue_resolution",
    instructions="""
    Evaluate if the customer's issue was resolved.
    User's messages: {{ inputs }}
    Agent's responses: {{ outputs }}

    Rate and respond with exactly one of:
    - 'fully_resolved'
    - 'partially_resolved'
    - 'needs_follow_up'
    """,
    model="databricks:/databricks-gpt-5-mini"  # Optional
)

# Use in evaluation
results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_app,
    scorers=[issue_judge]
)
```

### Trace-based Judge (with {{ trace }})
```python
# Including {{ trace }} in instructions enables trace exploration
tool_judge = make_judge(
    name="tool_correctness",
    instructions="""
    Analyze the execution {{ trace }} to determine if appropriate tools were called.
    Respond with true or false.
    """,
    model="databricks:/databricks-gpt-5-mini"  # REQUIRED for trace judges
)
```

---

## Trace Search Filter Syntax

### Common Filters
```python
import mlflow

traces_df = mlflow.search_traces(
    filter_string="attributes.status = 'OK'",
    order_by=["attributes.timestamp_ms DESC"],
    max_results=100,
    run_id="optional-run-id"  # Filter to specific evaluation run
)

# Common filters:
# "attributes.status = 'OK'" or "attributes.status = 'ERROR'"
# "attributes.timestamp_ms > {milliseconds}"
# "attributes.execution_time_ms > 5000"
# "tags.environment = 'production'"
# "tags.`mlflow.traceName` = 'my_function'"
```

### Filter Syntax Rules

| Syntax Element | Rule |
|----------------|------|
| String values | Use single quotes: `'OK'` NOT `"OK"` |
| Dotted names | Use backticks: `tags.\`mlflow.traceName\`` |
| Prefix | Required for attributes: `attributes.status` |
| Logical operators | `AND` supported, `OR` NOT supported |
| Time values | Use milliseconds since epoch |

---

## Trace Object Access

```python
from mlflow.entities import Trace, SpanType

@scorer
def trace_scorer(trace: Trace) -> Feedback:
    # Search spans by type
    llm_spans = trace.search_spans(span_type=SpanType.CHAT_MODEL)
    retriever_spans = trace.search_spans(span_type=SpanType.RETRIEVER)

    # Access span data
    for span in llm_spans:
        duration = (span.end_time_ns - span.start_time_ns) / 1e9
        inputs = span.inputs
        outputs = span.outputs
```

---

## SpanType Constants

```python
from mlflow.entities import SpanType

SpanType.CHAT_MODEL      # LLM calls
SpanType.RETRIEVER       # RAG retrieval
SpanType.TOOL            # Tool/function calls
SpanType.AGENT           # Agent execution
SpanType.CHAIN           # Chain execution
```

### Feedback Values
```python
# LLM judges typically return:
"yes" | "no"     # For pass/fail assessments

# Custom scorers can return:
True | False     # Boolean
0.0 - 1.0        # Float scores
int              # Integer scores
str              # Categorical values
```

---

## Production Monitoring API

### Register and Start Scorer
```python
from mlflow.genai.scorers import Safety, Guidelines, ScorerSamplingConfig

# Register scorer to experiment
safety = Safety().register(name="safety_monitor")

# Start monitoring with sample rate
safety = safety.start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.5)  # 50% of traces
)
```

### ScorerSamplingConfig Options
```python
ScorerSamplingConfig(
    sample_rate=0.5,  # Sample 50% of traces (0.0 to 1.0)
)
```

### Manage Scorers
```python
from mlflow.genai.scorers import list_scorers, get_scorer, delete_scorer

# List all registered scorers
scorers = list_scorers()

# Get specific scorer
my_scorer = get_scorer(name="safety_monitor")

# Update sample rate
my_scorer = my_scorer.update(
    sampling_config=ScorerSamplingConfig(sample_rate=0.8)
)

# Stop monitoring (keeps registration)
my_scorer = my_scorer.stop()

# Delete entirely
delete_scorer(name="safety_monitor")
```
