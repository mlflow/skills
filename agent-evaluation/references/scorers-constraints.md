# MLflow Judge Constraints & Requirements

Critical constraints when using `mlflow scorers register-llm-judge` CLI command.

## Table of Contents

1. [Constraint 1: {{trace}} Variable is Mutually Exclusive](#constraint-1-trace-variable-is-mutually-exclusive)
2. [Constraint 2: CLI Requires "yes"/"no" Return Values](#constraint-2-cli-requires-yesno-return-values)
3. [Constraint 3: Instructions Must Include Template Variable](#constraint-3-instructions-must-include-template-variable)

## Overview

The MLflow CLI for registering LLM judges has specific requirements. Follow these constraints to avoid registration errors.

## Constraint 1: {{trace}} Variable is Mutually Exclusive

If you use `{{trace}}` in your instructions, it MUST be the ONLY variable.

**Cannot mix {{trace}} with:**

- ❌ `{{inputs}}`
- ❌ `{{outputs}}`

**Example - Correct:**

```bash
uv run mlflow scorers register-llm-judge \
  -n "ToolUsage" \
  -i "Evaluate the trace: {{ trace }}. Did the agent use appropriate tools? Return yes or no."
```

**Example - Wrong:**

```bash
uv run mlflow scorers register-llm-judge \
  -n "ToolUsage" \
  -i "Given query {{ inputs }} and trace {{ trace }}, evaluate tools used."  # ❌ Cannot mix!
```

**Why this constraint exists:**

The `{{trace}}` variable contains everything:

- Input parameters (same as {{inputs}})
- Output responses (same as {{outputs}})
- All intermediate steps
- Tool calls
- LLM interactions

Since it includes inputs and outputs already, MLflow doesn't allow redundant variables.

**When to use {{trace}} vs {{inputs}}/{{outputs}}:**

Use `{{trace}}` when evaluating:

- ✅ Tool selection/usage
- ✅ Execution flow
- ✅ Intermediate reasoning
- ✅ Multi-step processes

Use `{{inputs}}`/`{{outputs}}` when evaluating:

- ✅ Final input/output quality only
- ✅ Response relevance
- ✅ Answer correctness

## Constraint 2: Scorers MUST Return "yes"/"no" — "pass"/"fail" Silently Drops Results

🚨 **CRITICAL: "pass"/"fail" return values cause SILENT DATA LOSS — use "yes"/"no" instead**

### Return Values (CRITICAL)

**These work correctly:**

- `"yes"` / `"no"` — ✅ Required. Correctly cast to numeric scores and included in `results.metrics`.
- `True` / `False` (boolean) — ✅ Works. Cast to 1.0 / 0.0.
- Numeric values (e.g., `0.0`–`1.0`) — ✅ Works. Used as-is.

**These cause silent data loss:**

- `"pass"` / `"fail"` — ❌ BROKEN. `_cast_assessment_value_to_float` maps these to `None`, which **excludes the scorer entirely from `results.metrics`**. No error is raised — results simply disappear.
- `"true"` / `"false"` (strings) — ❌ BROKEN. Same silent-drop behavior.
- `"passed"` / `"failed"` — ❌ BROKEN. Same silent-drop behavior.
- `"1"` / `"0"` (strings) — ⚠️ Unreliable. Do not use.

### Why "pass"/"fail" Silently Drops Results

MLflow's internal `_cast_assessment_value_to_float` function only recognizes `"yes"` and `"no"` as valid string return values. Any unrecognized string (including `"pass"` and `"fail"`) is cast to `None`. A `None` assessment value is then **excluded from aggregate metrics** — the scorer appears to run but its scores are never counted. No error or warning is emitted.

**Example - Correct (results appear in metrics):**

```bash
uv run mlflow scorers register-llm-judge \
  -n "QualityCheck" \
  -i "Evaluate if {{ outputs }} is high quality. Return 'yes' if high quality, 'no' if not."
```

**Example - BROKEN (results silently dropped from metrics):**

```bash
uv run mlflow scorers register-llm-judge \
  -n "QualityCheck" \
  -i "Evaluate if {{ outputs }} is high quality. Return 'pass' if good, 'fail' if bad."  # ❌ Silent data loss!
```

**Why "yes"/"no"?**

MLflow's built-in judges use the binary yes/no format. The internal casting logic is built around this convention — only `"yes"` and `"no"` are guaranteed to produce numeric scores that appear in `results.metrics`.

## Constraint 3: Instructions Must Include Template Variable

Instructions must contain at least one template variable:

- `{{ inputs }}` - Evaluation inputs
- `{{ outputs }}` - Agent outputs
- `{{ trace }}` - Complete execution trace

The above can be combined with optional variables:
- `{{ expectations }}` - Ground truth (optional)

**Example - Wrong (no variables):**

```bash
-i "Evaluate the quality. Return yes or no."  # ❌ Missing variable!
```

**Example - Correct:**

```bash
-i "Evaluate if {{ outputs }} is high quality. Return yes or no."  # ✅ Has variable
```

**Remember**: If using `{{ trace }}`, it must be the ONLY variable (see Constraint 1).

## Registration Example - All Constraints Met

```bash
# ✅ Correct - has variable, uses yes/no, correct parameters
uv run mlflow scorers register-llm-judge \
  -n "RelevanceCheck" \
  -d "Checks if response addresses the query" \
  -i "Given the response {{ outputs }}, determine if it directly addresses the query. Return 'yes' if relevant, 'no' if not."
```

```bash
# ✅ Correct - uses {{trace}} only (no other variables), yes/no, correct parameters
uv run mlflow scorers register-llm-judge \
  -n "ToolUsageCheck" \
  -d "Evaluates tool selection quality" \
  -i "Examine the trace {{ trace }}. Did the agent use appropriate tools for the query? Return 'yes' if appropriate, 'no' if not."
```

## Common Mistakes

1. **Mixing {{trace}} with {{inputs}} or {{outputs}}**

   - Error: "Cannot use trace variable with other variables"
   - Fix: Use only {{trace}} or only {{inputs}}/{{outputs}}

2. **Using "pass"/"fail" instead of "yes"/"no"**

   - Result: `_cast_assessment_value_to_float` maps "pass"/"fail" to `None` → scorer is **silently excluded from `results.metrics`**. No error is raised.
   - Fix: Always use `"yes"`/`"no"` (strings), booleans, or numeric values

3. **Missing template variables**

   - Error: "Instructions must contain at least one variable"
   - Fix: Include {{ outputs }}, {{ inputs }}, or {{ trace }}

4. **Wrong parameter names**
   - Check CLI help first: `mlflow scorers register-llm-judge --help`
   - Common correct parameters: `-n` (name), `-i` (instructions), `-d` (description)
