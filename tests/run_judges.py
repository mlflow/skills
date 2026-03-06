#!/usr/bin/env python3
"""
Subprocess entry point for running judges on CC traces.

Called by test_skill.verify_judges() to avoid in-process hangs with LLM API calls.
Dynamically imports each judges module, calls get_judges(), and runs
mlflow.genai.evaluate() on the CC traces.

Arguments (via environment variables):
    JUDGE_PATHS: JSON list of file paths to judge modules
    CC_EXPERIMENT_ID: Claude Code tracing experiment ID
    MLFLOW_EXPERIMENT_ID: Evaluation experiment ID
    RUN_START_MS: Timestamp (ms) to filter traces created after run start
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import os
import sys

import mlflow

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# Pre-import modules that scorer threads will need. Without this, concurrent
# lazy imports of litellm/openai from scorer threads deadlock on Python's
# import lock (see deadlock-thread-dump-analysis.md).
import litellm  # noqa: F401
import mlflow.server.jobs.utils  # noqa: F401

judge_paths = json.loads(os.environ["JUDGE_PATHS"])
cc_experiment_id = os.environ["CC_EXPERIMENT_ID"]
eval_experiment_id = os.environ["MLFLOW_EXPERIMENT_ID"]
run_start_ms = int(os.environ["RUN_START_MS"])

# Load judges from all configured modules
judges = []
for i, module_path in enumerate(judge_paths):
    spec = importlib.util.spec_from_file_location(f"judges_{i}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    judges.extend(module.get_judges())

log.info(f"Loaded {len(judges)} judge(s)")
if not judges:
    print(json.dumps({"error": "No judges returned by get_judges()"}))
    sys.exit(0)

# Get traces created after the main run started.
# First check the CC tracing experiment, then fall back to the eval experiment
# (some skills log traces to the eval experiment rather than CC tracing).
filter_str = f"trace.timestamp_ms > {run_start_ms}"
trace_df = mlflow.search_traces(
    experiment_ids=[cc_experiment_id],
    filter_string=filter_str,
)
log.info(f"Found {len(trace_df)} trace(s) after run start")
if trace_df.empty:
    log.info("No CC traces found, checking eval experiment...")
    trace_df = mlflow.search_traces(
        experiment_ids=[eval_experiment_id],
        filter_string=filter_str,
    )
    log.info(f"Found {len(trace_df)} trace(s) in eval experiment")
if trace_df.empty:
    print(json.dumps({"error": "No traces found after run start"}))
    sys.exit(0)

mlflow.set_experiment(experiment_id=cc_experiment_id)

names = [s.name for s in judges]
log.info(f"Running judges: {names}")
with contextlib.redirect_stdout(sys.stderr):
    eval_result = mlflow.genai.evaluate(
        data=trace_df,
        scorers=judges,
    )

log.info("Evaluation complete")

# Collect results. Column format is "{scorer_name}/value".
results = []
result_df = eval_result.result_df
for _, row in result_df.iterrows():
    trace_id = row.get("trace_id", "unknown")
    for judge in judges:
        val_col = f"{judge.name}/value"
        rat_col = f"{judge.name}/rationale"
        value = row.get(val_col)
        if value is not None:
            results.append({
                "scorer": judge.name,
                "trace_id": trace_id,
                "value": str(value),
                "rationale": str(row.get(rat_col, "")),
                "pass": str(value).lower() == "yes",
            })

print(json.dumps(results))
