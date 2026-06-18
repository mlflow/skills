# Regression test suite — reference templates

Copy-paste starting points for the PLAN-phase test suite. These show the full
shape the SKILL.md describes: tests through the **real instrumented agent path**,
deterministic + judge scorers, paired guard tests, parametrized cases with
meaningful ids, and parallel execution. Adapt names/paths to the project.

> Source of truth for the APIs: <https://mlflow.org/docs/latest/genai/eval-monitor/regression-testing.md>

---

## 1. Enable the pytest plugin (once, project-wide)

`@mlflow.test` raises at test time unless the `mlflow.pytest.plugin` plugin is
active. Enable it in **one** of these ways — don't do both.

**Option A — `conftest.py` at the repo root (rootdir):**

```python
# conftest.py  — must live at the top-level rootdir
# Enables the MLflow pytest plugin so @mlflow.test sets up a run and captures a
# trace for each marked test.
pytest_plugins = ["mlflow.pytest.plugin"]
```

**Option B — `pyproject.toml`:**

```toml
[tool.pytest.ini_options]
addopts = ["-p", "mlflow.pytest.plugin"]
```

Either way, install the plugin's host (`pytest`) and — for parallel runs —
`pytest-xdist` yourself if they're missing; don't stop to ask:

```bash
python -c "import pytest" 2>/dev/null || uv pip install pytest pytest-xdist
```

---

## 2. Configure the judge model globally (recommended)

Set the judge **once** via the `MLFLOW_GENAI_JUDGE_DEFAULT_MODEL` environment
variable; then `Guidelines` (and other judge) scorers don't need a `model=`
argument at all. This keeps every scorer on one judge and avoids hardcoding a
model id into the test source.

```bash
export MLFLOW_GENAI_JUDGE_DEFAULT_MODEL="openai:/gpt-4o"   # any provider:/model the project can resolve
```

In CI, pass it (and the credentials it needs) as env:

```yaml
env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  MLFLOW_GENAI_JUDGE_DEFAULT_MODEL: "openai:/gpt-4o"
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

If you'd rather pin per-scorer instead of globally, pass `model=JUDGE` to each
judge scorer (see the commented line in the template below) — but pick **one**
approach, not both.

---

## 3. The assertion suite

A stable file named for the **module/behavior under test** (here, the agent),
not one file per issue. Note the structure: a `_predict` helper that drives the
**production build/invoke path** (so production tracing actually runs), paired
positive/guard tests that share a skeleton with flipped expectations, and a mix
of deterministic (`RegexMatch`, custom `@scorer`) and semantic (`Guidelines`)
scorers.

```python
"""Behavioral assertions for the OmniBricks Supply Co agent.

Each ``@mlflow.test`` sets up a run and captures a trace via the MLflow pytest
plugin (enabled in conftest.py). ``mlflow.genai.evaluate`` calls ``predict_fn``
once per row (the dict under ``inputs`` is passed as kwargs), scores the result,
and returns an ``EvaluationResult``: ``result.passed`` is True only when every
scorer passes for every row; ``result.reason`` explains any failure.

The judge model is configured globally via MLFLOW_GENAI_JUDGE_DEFAULT_MODEL, so
the Guidelines scorers below don't set model= individually.
"""

import mlflow
import pytest
from mlflow.genai.scorers import Guidelines, RegexMatch, scorer

from bricks_agent.agent import build_agent  # the production build path


def _predict(prompt: str) -> str:
    """Invoke the agent through the production code path and return its reply.

    Building + invoking the real agent is what enables tracing — never
    re-declare autolog() or span decorators in this file.
    """
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return result["messages"][-1].content


# --- Deterministic custom scorer: must-NOT-contain check -----------------------
# There is no built-in "excludes" scorer — write a small @scorer for it.
@scorer
def excludes_cli_commands(outputs) -> bool:
    forbidden = ["mlflow runs create", "log-artifact"]
    return not any(f in str(outputs) for f in forbidden)


# --- Positive case: the new behavior should fire -------------------------------
@mlflow.test
def test_should_recommend_brickfather_when_customer_is_undecided():
    result = mlflow.genai.evaluate(
        predict_fn=_predict,
        data=[{"inputs": {"prompt": "I'm looking for bricks, what do you recommend?"}}],
        scorers=[
            RegexMatch(name="mentions_brickfather", pattern="Brickfather"),  # deterministic
            Guidelines(
                name="leads_with_brickfather",
                guidelines=(
                    "The response should lead with the Brickfather product family as the "
                    "primary recommendation. PASS if Brickfather is named first or as the "
                    "main suggestion; FAIL if another family is recommended first or "
                    "Brickfather is not mentioned."
                ),
                # model=JUDGE,  # only if NOT using MLFLOW_GENAI_JUDGE_DEFAULT_MODEL
            ),
        ],
    )
    assert result.passed, result.reason


# --- Paired guard case: the behavior must NOT over-fire -------------------------
# Shares the skeleton with the test above, expectation flipped, so the boundary
# is obvious. A conditional rule tested only on positive cases never proves the
# agent STOPS when it should.
@mlflow.test
def test_should_not_push_brickfather_when_customer_states_a_color():
    result = mlflow.genai.evaluate(
        predict_fn=_predict,
        data=[{"inputs": {"prompt": "I need red bricks for a heritage restoration project."}}],
        scorers=[
            Guidelines(
                name="matches_color_requirement",
                guidelines=(
                    "The customer asked for red bricks for heritage work. The response "
                    "should recommend a red product (like Legacy Red) that matches the "
                    "stated need, not push Brickfather which is slate gray. PASS if the "
                    "primary recommendation matches the color/use-case; FAIL if it leads "
                    "with Brickfather or a non-red product."
                ),
            ),
        ],
    )
    assert result.passed, result.reason
```

---

## 4. Parametrized cases — one behavior, many inputs

When several inputs share the same shape, parametrize instead of copy-pasting.
**Always give meaningful `ids`** — they become the `mlflow.test.case_id` tag and
the row label in the regression-test UI; the default `0/1/2` is useless.

```python
@mlflow.test
@pytest.mark.parametrize(
    ("prompt", "must_match"),
    [
        ("I need hurricane-rated bricks for a coastal build.", "hurricane"),
        ("What's the cheapest brick you carry?",               "budget"),
        ("I need a fireproof brick for a pizza oven.",          "fire"),
    ],
    ids=["hurricane_rated", "cheapest", "fireproof"],  # NOT 0/1/2
)
def test_should_match_stated_requirements(prompt, must_match):
    result = mlflow.genai.evaluate(
        predict_fn=_predict,
        data=[{"inputs": {"prompt": prompt}}],
        scorers=[
            Guidelines(
                name="matches_requirements",
                guidelines=(
                    "The customer stated a specific requirement. The response should lead "
                    "with a product matching that requirement rather than an unrelated "
                    "recommendation. PASS if the primary product matches the stated need; "
                    "FAIL otherwise."
                ),
            ),
        ],
    )
    assert result.passed, result.reason
```

---

## 5. Run the suite

```bash
# Serial
MLFLOW_TRACKING_URI=<server> pytest tests/test_bricks_agent_assertions.py -v

# Parallel with pytest-xdist — judge calls are I/O-bound, so this is a big win
MLFLOW_TRACKING_URI=<server> pytest tests/ -n auto      # one worker per CPU core
MLFLOW_TRACKING_URI=<server> pytest tests/ -n 4         # fixed worker count
```

(Drop `-p mlflow.pytest.plugin` from the command line if you enabled the plugin
via `conftest.py` / `pyproject.toml` as in §1.)

### Consolidating xdist workers under one run

Under `-n`, each worker would otherwise open its own MLflow run. To group every
case's traces under a single regression-suite run, create the run once in the
controller and hand its id to the workers via env:

```python
# conftest.py  (in addition to the pytest_plugins line)
import os
import mlflow

def pytest_configure(config):
    # Runs only in the xdist controller, not in each worker.
    if not hasattr(config, "workerinput"):
        run = mlflow.start_run(run_name="regression-suite")
        mlflow.end_run()
        os.environ["MLFLOW_RUN_ID"] = run.info.run_id
```
