# Plan: Add MLflow Tracing for Claude Code in Test Script

## Status: COMPLETED AND VERIFIED

## Summary
Modified `tests/test_agent_evaluation.py` to create a second MLflow experiment named `claude-code-skill-<timestamp>` and enable MLflow tracing for the headless Claude Code execution using the `mlflow autolog claude` command.

## Changes Made

### 1. Added new config fields in `Config` dataclass (lines 77-79)
- `cc_tracing_experiment_name: Optional[str] = None`
- `cc_tracing_experiment_id: Optional[str] = None`

### 2. Created new function `setup_claude_code_tracing(config: Config) -> bool` (lines 338-404)
This function:
1. Generates experiment name: `claude-code-skill-<timestamp>` (with workspace path prefix for Databricks)
2. Creates the experiment via `mlflow.create_experiment()`
3. Stores the experiment ID in config
4. Runs `mlflow autolog claude <project_dir>` with:
   - `-u <tracking_uri>` for MLflow tracking URI
   - `-e <experiment_id>` for the new experiment
5. Returns success/failure

### 3. Called the new function in `setup_phase()` (lines 534-536)
After the evaluation experiment is created, calls `setup_claude_code_tracing(config)` before returning.

### 4. Modified `test_claude_headless()` to verify tracing works (lines 580-611)
After running the "Say hello world" test query:
1. Waits 5 seconds for trace to be flushed
2. Queries MLflow for traces in the `claude-code-skill-<timestamp>` experiment
3. Verifies at least one trace was captured
4. Logs the trace info for debugging
5. Fails the test if no traces are found (indicates tracing not working)

This ensures tracing is validated early before the longer evaluation run.

### 5. Updated cleanup logging (line 204)
Logs the Claude Code tracing experiment ID alongside the evaluation experiment ID.

## Files Modified
- `tests/test_agent_evaluation.py`

## Verification
After implementation:
1. Run `python tests/test_agent_evaluation.py`
2. The test will now fail early (during headless test) if Claude Code tracing doesn't work
3. Check MLflow UI for two experiments:
   - `agent-eval-test-<timestamp>` - contains evaluation traces
   - `claude-code-skill-<timestamp>` - contains Claude Code CLI traces

## Dependencies
- MLflow >= 3.4 (for `mlflow autolog claude` command)

## Phase 2: Remove uv dependency (2026-02-06)

**Status: COMPLETED**

Removed all `uv` usage from `tests/test_agent_evaluation.py`. The test now assumes mlflow is already installed in the environment.

Changes:
- Removed `uv` prerequisite check from `check_prerequisites()`
- Removed `uv sync` and `uv add mlflow` steps from `setup_phase()`
- Removed `uv add databricks-agents databricks-connect` step
- Replaced all `uv run python` commands with `python` (mlflow server, experiment creation, autolog, verification)
- Removed venv PATH prepending (no longer needed)

No settings.json patching needed — experiment separation works via `get_env_var()` in hooks vs `os.getenv()` in eval scripts.
