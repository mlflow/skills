# MLflow Evaluation Scorers Guide

Complete guide for selecting and creating scorers to evaluate agent quality.

## Table of Contents

1. [Understanding Scorers](#understanding-scorers)
2. [Built-in Scorers](#built-in-scorers)
3. [Custom Scorer Design](#custom-scorer-design)
4. [Scorer Registration](#scorer-registration)
5. [Testing Scorers](#testing-scorers)

## Understanding Scorers

### What are Scorers?

Scorers (also called "judges" or "LLM-as-a-judge") are evaluation criteria that assess the quality of agent responses. They:

- Take as input single-turn inputs&outputs, or multi-turn conversations, or traces
- Apply quality criteria (relevance, accuracy, completeness, etc.)
- Return a score or pass/fail judgment
- Can be built-in (provided by MLflow) or custom (defined by you)

### Types of Scorers

**1. Reference-Free Scorers**

- Don't require ground truth or expected outputs
- Judge quality based on their inputs alone
- Examples: Relevance, Completeness, Clarity
- **Easiest to use** - work with any dataset

**2. Ground-Truth Scorers**

- Require expectations in the dataset
- Compare agent response to ground truth
- Examples: Factual Accuracy, Answer Correctness
- Require datasets with `expectations` field

### LLM-as-a-Judge Pattern

Modern scorers use an LLM to judge quality:

1. Scorer receives information on agent's execution (input&output, or trace, or conversation)
2. LLM is given evaluation instructions
3. LLM judges whether criteria is met
4. Returns a structured assessment (pass/fail or numeric) along with a rationale

## Built-in Scorers

MLflow provides several built-in scorers for common evaluation criteria.

### Discovering Built-in Scorers

**IMPORTANT: Use the documentation protocol to discover built-in scorers.**

Do NOT use `mlflow scorers list -b` - it may be incomplete or unavailable in some environments. Instead:

1. Query MLflow documentation via llms.txt:
   ```
   WebFetch https://mlflow.org/docs/latest/llms.txt with prompt:
   "What built-in LLM judges or scorers are available in MLflow for evaluating GenAI agents?"
   ```

2. Read scorer documentation pages referenced in llms.txt to understand:
   - Scorer names and how to import them
   - What each scorer evaluates
   - Required inputs (trace structure, expected_response, etc.)
   - When to use each scorer

3. Verify scorer availability by attempting import:
   ```python
   from mlflow.genai.scorers import Correctness, RelevanceToQuery
   ```

### Checking Registered Scorers

List scorers registered in your experiment:

```bash
uv run mlflow scorers list -x $MLFLOW_EXPERIMENT_ID
```

Output shows:
- Scorer names
- Whether they're built-in or custom
- Registration details

**IMPORTANT: if there are registered scorers in the experiment then they must be used for evaluation.**

### Understanding Built-in Scorers

After querying the documentation, you'll typically find scorers in these categories:

**Reference-free scorers** (judge without ground truth):

- Relevance, Completeness, Coherence, Clarity
- Use for: All agents, no expected outputs needed

**Ground-truth scorers** (require expected outputs):

- Answer Correctness, Faithfulness, Accuracy
- Use for: When you have known correct answers in dataset

**Context-based scorers** (require context/documents):

- Groundedness, Citation Quality
- Use for: RAG systems, knowledge base agents

### Important: Trace Structure Assumptions

**CRITICAL**: Built-in scorers make assumptions about trace structure.

Before using a built-in scorer:

1. **Read its documentation** to understand required inputs
2. **Check trace structure** matches expectations
3. **Verify it works** with a test trace before full evaluation

**Example issue**:

- Scorer expects `context` field in trace
- Your agent doesn't provide `context`
- Scorer fails or returns null

**Solution**:

- Read scorer docs carefully
- Test on single trace first
- Create custom scorer if built-in doesn't match your structure

### Using Built-in Scorers

After discovering scorers via documentation, register them to your experiment:

```python
import os
from mlflow.genai.scorers import Correctness, RelevanceToQuery

# Note: Import exact class names from documentation
# Common mistake: trying to import "Relevance" when it's actually "RelevanceToQuery"

# Register built-in scorer to experiment
scorer = Correctness()
scorer.register(experiment_id=os.getenv("MLFLOW_EXPERIMENT_ID"))
```

**Benefits of registration**:

- Shows up in `mlflow scorers list -x <id>`
- Keeps all evaluation criteria in one place
- Makes it clear what scorers are being used for the experiment

## Custom Scorer Design

Create custom scorers when:

- Built-in scorers don't match your criteria
- You need domain-specific evaluation
- Your agent has unique requirements
- Trace structure doesn't match built-in assumptions

## MLflow Judge Constraints

⚠️ **The MLflow CLI has specific requirements for custom scorers.**

Before creating custom scorers, read the complete constraints guide:

- See `references/scorers-constraints.md` for detailed requirements

**Key constraints:**

1. `{{trace}}` variable cannot be mixed with `{{inputs}}` or `{{outputs}}`
2. CLI requires "yes"/"no" return values (not "pass"/"fail")
3. Instructions must include at least one template variable

---

### Design Process

**Step 1: Define Quality Criterion Clearly**

What specific aspect of quality are you judging?

**Examples**:

- "Response uses appropriate tools for the query"
- "Response is factually accurate based on available data"
- "Response follows the expected format"
- "Response appropriately handles ambiguous queries"

**Step 2: Determine Required Inputs**

What information does the scorer need?

**Common inputs**:

- `query`: The user's question
- `response`: The agent's answer
- `trace`: Full trace with tool calls, LLM calls, etc.
- `context`: Retrieved documents or context (if applicable)

**Step 3: Write Evaluation Instructions**

Clear instructions for the LLM judge:

```
You are evaluating whether an agent used appropriate tools for a query.

Given:
- Query: {query}
- Response: {response}
- Trace: {trace} (contains tool calls)

Criteria: The agent should use tools when needed (e.g., search for factual queries)
and should not use tools unnecessarily (e.g., for greetings).

Evaluate whether appropriate tools were used. Return "yes" if tools were used
appropriately, "no" if not.
```

**Step 4: Choose Output Format**

Use yes/no format as required by the CLI (see CRITICAL CONSTRAINTS above).

### Example Custom Scorers

**Example 1: Tool Usage Appropriateness**

```
Scorer Name: ToolUsageAppropriate

Definition: Judges whether the agent used appropriate tools for the query.

Instructions:
You are evaluating tool usage by an AI agent.

Given a query and trace showing tool calls, determine if:
1. Tools were used when needed (factual questions, searches, lookups)
2. Tools were NOT used unnecessarily (greetings, simple questions)
3. The RIGHT tools were chosen for the task

Return "yes" if tool usage was appropriate, "no" if not.

Variables: query, response, trace
Output: yes/no
```

**Example 2: Factual Accuracy**

```
Scorer Name: FactualAccuracy

Definition: Judges whether the response is factually accurate.

Instructions:
You are evaluating the factual accuracy of an AI agent's response.

Review the agent's response and determine if the information provided is
factually correct based on the context and your knowledge.

Return "yes" if the response is factually accurate, "no" if it contains
incorrect information or makes unsupported claims.

Variables: query, response, context (optional)
Output: yes/no
```

## Scorer Registration

### Check CLI Help First

Run `--help` to verify parameter names:

```bash
uv run mlflow scorers register-llm-judge --help
```

### Correct CLI Parameters

### Registration Example - All Requirements Met

```bash
# ✅ CORRECT - Has variable, uses yes/no, correct parameters
uv run mlflow scorers register-llm-judge \
  -n "RelevanceCheck" \
  -d "Checks if response addresses the query" \
  -i "Given the response {{ outputs }}, determine if it directly addresses the query. Return 'yes' if relevant, 'no' if not."
```

```bash
# ✅ CORRECT - Uses {{trace}} only (no other variables), yes/no, correct parameters
uv run mlflow scorers register-llm-judge \
  -n "ToolUsageCheck" \
  -d "Evaluates tool selection quality" \
  -i "Examine the trace {{ trace }}. Did the agent use appropriate tools for the query? Return 'yes' if appropriate, 'no' if not."
```

### Using make_judge() Function

**Programmatic registration** for advanced use cases:

```python
from mlflow.genai.judges import make_judge
from typing import Literal

scorer = make_judge(
    name="ToolUsageAppropriate",
    description="Judges whether appropriate tools were used",
    instructions="""
    You are evaluating tool usage by an AI agent.

    Given a trace: {{ trace }}

    Determine if appropriate tools were used.
    Return "yes" if tool usage was appropriate, "no" if not.
    """,
    feedback_value_type=Literal["yes", "no"],
)

# Register the scorer
registered_scorer = scorer.register(experiment_id="your_experiment_id")
```

**When to use make_judge()**:

- `mlflow scorers register-llm-judge` fails with an obscure error
- Need programmatic control
- Integration with existing code

**Important**: The `make_judge()` API follows the same constraints documented in the CRITICAL CONSTRAINTS section above. Use `Literal["yes", "no"]` for `feedback_value_type` for binary scorers.

### Best Practices

**1. Use default model** unless you have specific needs:

- Default is usually sufficient and cost-effective
- Specify model only for specialized evaluation

**2. Register both built-in and custom scorers for version control and team collaboration**

**3. Test before full evaluation**:

- Test on single trace first
- Verify output format is correct
- Check that instructions are clear

**4. Version your scorers**:

- Include version in name if criteria change: `ToolUsageAppropriate_v2`
- Document what changed between versions

## Testing Scorers

**Always test a scorer** before using it on your full evaluation dataset.

### Quick Single-Trace Test

```bash
# Get a sample trace ID (from previous agent run)
export TRACE_ID="<sample_trace_id>"

# Test scorer on single trace
uv run mlflow traces evaluate \
  --output json \
  --scorers ToolUsageAppropriate \
  --trace-ids $TRACE_ID
```

### Verify Scorer Behavior

**Check 1: No Errors**

- Scorer executes without errors
- No null or empty outputs

**Check 2: Output Format**

- For yes/no: Returns "yes" or "no"
- For numeric: Returns number in expected range

**Check 3: Makes Sense**

- Review the trace
- Manually judge if scorer output is reasonable
- If scorer is wrong, refine instructions

**Check 4: Trace Coverage**

- Test on diverse traces (different query types)
- Ensure scorer handles all cases
- Check edge cases

### Iteration Workflow

1. **Register scorer** with initial instructions
2. **Test on single trace** with known expected outcome
3. **Review output** - does it match your judgment?
4. **If wrong**: Refine instructions, re-register, test again
5. **Test on diverse traces** (3-5 different types)
6. **Deploy to full evaluation** once confident

### Example Test Session

```bash
# Test ToolUsageAppropriate scorer

# Test 1: Query that should use tools (expect: yes)
uv run mlflow traces evaluate \
  --scorers ToolUsageAppropriate \
  --trace-ids <trace_with_tool_use> \
  --output json

# Test 2: Greeting that shouldn't use tools (expect: yes)
uv run mlflow traces evaluate \
  --scorers ToolUsageAppropriate \
  --trace-ids <trace_greeting> \
  --output json

# Test 3: Query that should use tools but didn't (expect: no)
uv run mlflow traces evaluate \
  --scorers ToolUsageAppropriate \
  --trace-ids <trace_missing_tools> \
  --output json
```

Review each output to verify scorer behaves as expected.

---

## Advanced Scorer Patterns

### Pattern: Custom Scorer with Multiple Metrics

Return multiple metrics from a single scorer.

```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

@scorer
def comprehensive_check(inputs, outputs):
    """Return multiple metrics from one scorer."""
    response = str(outputs.get("response", ""))
    query = inputs.get("query", "")

    feedbacks = []

    # Check 1: Response exists
    feedbacks.append(Feedback(
        name="has_response",
        value=len(response) > 0,
        rationale="Response is present" if response else "No response"
    ))

    # Check 2: Word count
    word_count = len(response.split())
    feedbacks.append(Feedback(
        name="word_count",
        value=word_count,
        rationale=f"Response contains {word_count} words"
    ))

    # Check 3: Query terms in response
    query_terms = set(query.lower().split())
    response_terms = set(response.lower().split())
    overlap = len(query_terms & response_terms) / len(query_terms) if query_terms else 0
    feedbacks.append(Feedback(
        name="query_coverage",
        value=round(overlap, 2),
        rationale=f"{overlap*100:.0f}% of query terms found in response"
    ))

    return feedbacks
```

### Pattern: Trace-Based Scorer

Analyze execution details from the trace.

```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback, Trace, SpanType

@scorer
def llm_latency_check(trace: Trace) -> Feedback:
    """Check if LLM response time is acceptable."""

    # Find LLM spans in trace
    llm_spans = trace.search_spans(span_type=SpanType.CHAT_MODEL)

    if not llm_spans:
        return Feedback(
            value="no",
            rationale="No LLM calls found in trace"
        )

    # Calculate total LLM time
    total_llm_time = 0
    for span in llm_spans:
        duration = (span.end_time_ns - span.start_time_ns) / 1e9
        total_llm_time += duration

    max_acceptable = 5.0  # seconds

    if total_llm_time <= max_acceptable:
        return Feedback(
            value="yes",
            rationale=f"LLM latency {total_llm_time:.2f}s within {max_acceptable}s limit"
        )
    else:
        return Feedback(
            value="no",
            rationale=f"LLM latency {total_llm_time:.2f}s exceeds {max_acceptable}s limit"
        )
```

### Pattern: Tool Selection Accuracy Scorer

Check if the correct tools were called during agent execution.

```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback, Trace, SpanType
from typing import Dict, Any

@scorer
def tool_selection_accuracy(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    expectations: Dict[str, Any],
    trace: Trace
) -> Feedback:
    """Check if the correct tools were called."""

    expected_tools = expectations.get("expected_tools", [])

    if not expected_tools:
        return Feedback(
            name="tool_selection_accuracy",
            value="skip",
            rationale="No expected_tools in expectations"
        )

    # Get actual tool calls from TOOL spans
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    actual_tools = {span.name for span in tool_spans}

    # Normalize names (handle fully qualified names like "catalog.schema.func")
    def normalize(name: str) -> str:
        return name.split(".")[-1] if "." in name else name

    expected_normalized = {normalize(t) for t in expected_tools}
    actual_normalized = {normalize(t) for t in actual_tools}

    # Check if all expected tools were called
    missing = expected_normalized - actual_normalized

    all_expected_called = len(missing) == 0

    rationale = f"Expected: {list(expected_normalized)}, Actual: {list(actual_normalized)}"
    if missing:
        rationale += f" | Missing: {list(missing)}"

    return Feedback(
        name="tool_selection_accuracy",
        value="yes" if all_expected_called else "no",
        rationale=rationale
    )
```

### Pattern: Stage Latency Scorer (Multiple Metrics)

Measure latency per pipeline stage and identify bottlenecks.

```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback, Trace
from typing import List

@scorer
def stage_latency_scorer(trace: Trace) -> List[Feedback]:
    """Measure latency for each pipeline stage."""

    feedbacks = []
    all_spans = trace.search_spans()

    # Total trace time
    root_spans = [s for s in all_spans if s.parent_id is None]
    if root_spans:
        root = root_spans[0]
        total_ms = (root.end_time_ns - root.start_time_ns) / 1e6
        feedbacks.append(Feedback(
            name="total_latency_ms",
            value=round(total_ms, 2),
            rationale=f"Total execution time: {total_ms:.2f}ms"
        ))

    # Per-stage latency (customize patterns for your pipeline)
    stage_patterns = ["classifier", "rewriter", "executor", "retriever"]
    stage_times = {}

    for span in all_spans:
        span_name_lower = span.name.lower()
        for pattern in stage_patterns:
            if pattern in span_name_lower:
                duration_ms = (span.end_time_ns - span.start_time_ns) / 1e6
                stage_times[pattern] = stage_times.get(pattern, 0) + duration_ms
                break

    for stage, time_ms in stage_times.items():
        feedbacks.append(Feedback(
            name=f"{stage}_latency_ms",
            value=round(time_ms, 2),
            rationale=f"Stage '{stage}' took {time_ms:.2f}ms"
        ))

    return feedbacks
```

### Pattern: Scorer with Aggregations

Use for numeric scorers that need aggregate statistics.

```python
from mlflow.genai.scorers import scorer

@scorer(aggregations=["mean", "min", "max", "median", "p90"])
def response_latency(outputs) -> float:
    """Return response generation time."""
    return outputs.get("latency_ms", 0) / 1000.0  # Convert to seconds

# Valid aggregations: min, max, mean, median, variance, p90
# NOTE: p50, p99, sum are NOT valid - use median instead of p50
```

### Pattern: Conditional Scoring Based on Input

Apply different guidelines based on query type.

```python
from mlflow.genai.scorers import scorer, Guidelines

@scorer
def conditional_scorer(inputs, outputs):
    """Apply different guidelines based on query type."""

    query = inputs.get("query", "").lower()

    if "technical" in query or "how to" in query:
        # Technical queries need detailed responses
        judge = Guidelines(
            name="technical_quality",
            guidelines=[
                "Response must include step-by-step instructions",
                "Response must include code examples where relevant"
            ]
        )
    elif "price" in query or "cost" in query:
        # Pricing queries need specific info
        judge = Guidelines(
            name="pricing_quality",
            guidelines=[
                "Response must include specific pricing information",
                "Response must mention any conditions or limitations"
            ]
        )
    else:
        # General queries
        judge = Guidelines(
            name="general_quality",
            guidelines=[
                "Response must directly address the question",
                "Response must be clear and concise"
            ]
        )

    return judge(inputs=inputs, outputs=outputs)
```

---

**For troubleshooting scorer issues**, see `references/troubleshooting.md`
