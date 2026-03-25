# Evaluation Throughput Guide

How to get faster results when running `mlflow.genai.evaluate()` on large datasets.

## When to Think About Throughput

| Dataset size | Sequential estimate | Action |
|---|---|---|
| <50 questions | ~5–15 min | Default settings are fine |
| 50–200 questions | ~15–60 min | Tune `MLFLOW_GENAI_EVAL_MAX_WORKERS` |
| 200+ questions | 1–3+ hours | Tune workers + consider dataset splitting |

These are rough estimates assuming a Sonnet-class judge and one scorer per question. Opus-class judges take ~2× longer.

## How MLflow Parallelizes Evaluation

`mlflow.genai.evaluate()` uses background threadpools internally — it does **not** expose `max_workers` as a function parameter. Parallelism is controlled entirely via environment variables.

Two levels of concurrency:

```
Data-level:   N questions run in parallel (MLFLOW_GENAI_EVAL_MAX_WORKERS)
Scorer-level: K scorers run in parallel per question (MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS)
```

Total concurrent LLM calls ≈ `MAX_WORKERS × min(MAX_SCORER_WORKERS, num_scorers)`

**Important:** `mlflow.genai.evaluate()` is **not thread-safe**. Do not call it from multiple threads in the same process. Use separate processes instead (see Dataset Splitting below).

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_GENAI_EVAL_MAX_WORKERS` | 10 | Parallel data items evaluated simultaneously |
| `MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS` | 10 | Parallel scorers per data item |
| `MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT` | 300 | Seconds before async predict_fn times out |
| `MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION` | (unset) | Skip the N+1 validation call if your predict_fn already generates traces |
| `MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING` | (unset) | Set to `true` to trace scorer execution (useful for debugging, adds overhead) |

### Recommended Starting Configuration

```bash
# Balanced — good for most LLM APIs with standard rate limits
export MLFLOW_GENAI_EVAL_MAX_WORKERS=10
export MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS=5

# Skip N+1 validation call if predict_fn already produces traces (saves ~N extra LLM calls)
export MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION=true
```

### If You're Hitting Rate Limits

Reduce workers to lower total concurrent API calls:

```bash
# Conservative — for free-tier API keys or strict rate limits
export MLFLOW_GENAI_EVAL_MAX_WORKERS=3
export MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS=2
# Peak concurrent calls: 3 × 2 = 6
```

### If You Have High Rate Limits

```bash
# Aggressive — for managed APIs with high rate limits (e.g., Databricks-hosted models)
export MLFLOW_GENAI_EVAL_MAX_WORKERS=20
export MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS=10
```

## Using Async predict_fn

If your agent supports async, use an async predict function. MLflow detects it automatically and runs it more efficiently:

```python
async def predict(query: str) -> str:
    # MLflow auto-detects async and wraps appropriately
    return await your_agent.arun(query)

results = mlflow.genai.evaluate(
    data=dataset.df,
    predict_fn=predict,  # async function — no changes needed to evaluate() call
    scorers=scorers,
)
```

Increase the timeout for slow agents:

```bash
export MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT=600  # 10 minutes
```

## Dataset Splitting (200+ Questions)

For very large datasets, split into shards and run each shard in a **separate process**. Since `evaluate()` is not thread-safe, use `subprocess` or separate shell scripts — not threads.

```python
import math
import subprocess
import pandas as pd
import mlflow

dataset = mlflow.genai.datasets.get_dataset(name="large-eval-1000q")
df = dataset.df
shard_size = 100
num_shards = math.ceil(len(df) / shard_size)

# Save shards to disk for subprocess consumption
for i in range(num_shards):
    shard = df.iloc[i * shard_size : (i + 1) * shard_size]
    shard.to_parquet(f"/tmp/shard_{i}.parquet")
```

Then run each shard as a separate script invocation. A shard script might look like:

```python
# run_shard.py
import sys
import pandas as pd
import mlflow
from your_agent import predict

shard_path = sys.argv[1]
df = pd.read_parquet(shard_path)

scorers = [...]  # same scorers as full eval

results = mlflow.genai.evaluate(
    data=df,
    predict_fn=predict,
    scorers=scorers,
)
results.tables["eval_results"].to_csv(shard_path.replace(".parquet", "_results.csv"))
```

Run shards in parallel via background processes:

```bash
for i in $(seq 0 9); do
  MLFLOW_GENAI_EVAL_MAX_WORKERS=10 \
  uv run python run_shard.py /tmp/shard_${i}.parquet > /tmp/shard_${i}.log 2>&1 &
done
wait  # wait for all background jobs
```

Then concatenate results:

```python
import glob
import pandas as pd

all_results = pd.concat([
    pd.read_csv(f) for f in sorted(glob.glob("/tmp/shard_*_results.csv"))
])
all_results.to_csv("evaluation_results_full.csv", index=False)
```

> **Note:** Each shard runs in its own process and logs independently to the same MLflow experiment. Runs will appear as separate entries — aggregate them in analysis.

## Quick Wins Checklist

Before running a large eval, check these:

- [ ] `MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION=true` — eliminates the extra N agent calls for validation
- [ ] `MLFLOW_GENAI_EVAL_MAX_WORKERS` set to match your API rate limit headroom
- [ ] Judge model is Sonnet-class (not Opus) unless you need Opus quality — 2× faster
- [ ] Scorers are registered, not inline — registered scorers run more reliably
- [ ] Agent itself is not doing unnecessary work (extra retrieval calls, full document fetching)
