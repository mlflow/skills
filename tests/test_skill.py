#!/usr/bin/env python3
"""
YAML-driven test runner for Claude Code skills.

This script:
1. Parses a YAML config describing the test case
2. Checks prerequisites (claude, git, skill dirs)
3. Creates a work directory and starts MLflow (if needed)
4. Creates experiments (eval + CC tracing)
5. Runs a setup script (clone repo, register judges, etc.)
6. Copies skills into the project's .claude/skills/
7. Configures Claude Code tracing (mlflow autolog claude)
8. Tests Claude Code headless mode
9. Runs Claude Code with the configured prompt
10. Discovers all registered judges and runs them on CC traces
11. Cleans up

Usage:
    python tests/test_skill.py tests/configs/agent_evaluation.yaml
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from config import (
    TestConfig,
    RuntimeState,
    EXIT_SUCCESS,
    EXIT_SETUP_FAILED,
    EXIT_EXECUTION_FAILED,
    EXIT_VERIFICATION_FAILED,
    load_config,
)
from utils import log, log_section, run_command, claude_env
from infra import (
    check_prerequisites,
    setup_infrastructure,
    run_setup_script,
    install_skills,
    setup_claude_code_tracing,
    test_claude_headless,
    cleanup,
)


def run_claude_code(config: TestConfig, state: RuntimeState) -> bool:
    log_section("Running Claude Code")

    state.log_file = state.work_dir / "claude_output.log"

    # Record start time so verify_judges can filter out pre-existing traces
    state.run_start_timestamp_ms = int(time.time() * 1000)

    log.info("Executing Claude Code...")
    log.info(f"Prompt: {config.prompt}")
    log.info(f"Timeout: {config.timeout_seconds} seconds")
    log.info(f"Log file: {state.log_file}")
    log.info(
        f"MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI', 'not set')}"
    )
    log.info(
        f"MLFLOW_EXPERIMENT_ID: {os.environ.get('MLFLOW_EXPERIMENT_ID', 'not set')}"
    )

    try:
        with open(state.log_file, "w") as f:
            result = subprocess.run(
                [
                    "claude",
                    "-p",
                    config.prompt,
                    "--dangerously-skip-permissions",
                    "--allowedTools",
                    config.allowed_tools,
                ],
                cwd=state.full_project_dir,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=config.timeout_seconds,
                stdin=subprocess.DEVNULL,
                env=claude_env(),
            )
            exit_code = result.returncode
    except subprocess.TimeoutExpired:
        log.info(
            f"Claude Code execution timed out after {config.timeout_seconds} seconds"
        )
        log.info("Will still verify if artifacts were created before timeout")
        return True
    except Exception as e:
        log.error(f"Claude Code execution failed: {e}")
        return False

    if exit_code != 0:
        log.error(f"Claude Code exited with code: {exit_code}")
        log.error(f"Check log file for details: {state.log_file}")
        return False

    log.info("Claude Code execution completed")
    return True


def verify_judges(config: TestConfig, state: RuntimeState) -> bool:
    log_section("Verification Phase: Running Judges")

    log.info("Waiting for traces to flush...")
    time.sleep(10)

    judge_paths = [str(state.repo_root / j) for j in config.judges]
    for p in judge_paths:
        log.info(f"Loading judges from: {p}")

    # Run evaluation in a subprocess to avoid in-process hangs with LLM API calls.
    run_judges_script = str(Path(__file__).parent / "run_judges.py")
    env = {
        "JUDGE_PATHS": json.dumps(judge_paths),
        "CC_EXPERIMENT_ID": state.cc_tracing_experiment_id,
        "MLFLOW_EXPERIMENT_ID": state.experiment_id,
        "RUN_START_MS": str(state.run_start_timestamp_ms),
    }

    try:
        result = run_command(
            ["python", run_judges_script],
            cwd=state.full_project_dir,
            check=False,
            timeout=config.verification_timeout,
            env=env,
        )
        output = result.stdout.strip()
        stderr_output = result.stderr.strip()
        if stderr_output:
            for line in stderr_output.splitlines():
                log.info(line)
    except subprocess.TimeoutExpired as e:
        log.error(f"Verification timed out after {config.verification_timeout} seconds")
        if e.stderr:
            for line in e.stderr.decode(errors="replace").splitlines():
                log.info(f"  (timeout) {line}")
        return False
    except Exception as e:
        log.error(f"Verification script failed: {e}")
        return False

    if not output:
        log.error("Verification script produced no output")
        return False

    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse verification output: {e}")
        log.error(f"Raw output: {output}")
        return False

    # Handle error case (no judges or no traces)
    if isinstance(data, dict) and "error" in data:
        log.error(data["error"])
        return False

    # Report results
    log.info("Judge Results:")
    print()

    all_passed = True
    for entry in data:
        passed = entry["pass"]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {entry['scorer']} on trace {entry['trace_id']}: {entry['value']}")
        if entry.get("rationale"):
            print(f"          Rationale: {entry['rationale']}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        log.info("All judge checks passed")
    else:
        log.error("Some judge checks failed")
    return all_passed


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config.yaml> [KEY=VALUE ...]", file=sys.stderr)
        return EXIT_SETUP_FAILED

    yaml_path = sys.argv[1]

    config = load_config(yaml_path)

    # Apply CLI environment variable overrides
    for arg in sys.argv[2:]:
        if "=" not in arg:
            print(f"Invalid override (expected KEY=VALUE): {arg}", file=sys.stderr)
            return EXIT_SETUP_FAILED
        key, _, value = arg.partition("=")
        config.environment[key] = value
    state = RuntimeState()

    # Determine repo root (parent of tests/ directory)
    script_dir = Path(__file__).parent.resolve()
    state.repo_root = script_dir.parent

    # Inject user-defined environment variables into the process
    os.environ.update(config.environment)

    # Register cleanup handler
    atexit.register(cleanup, config, state)

    log_section(f"Skill Test: {config.name}")
    log.info(f"Starting test at {datetime.now()}")
    log.info(f"Config file: {yaml_path}")
    if config.tracking_uri:
        log.info(f"External MLflow URI: {config.tracking_uri}")

    # Phase 1: Check prerequisites
    if not check_prerequisites(config, state):
        log.error("Prerequisites check failed")
        return EXIT_SETUP_FAILED

    # Phase 2: Setup infrastructure (work dir, MLflow, experiments)
    if not setup_infrastructure(config, state):
        log.error("Infrastructure setup failed")
        return EXIT_SETUP_FAILED

    # Phase 3: Run setup script (clone repo, register judges, etc.)
    if not run_setup_script(config, state):
        log.error("Setup script failed")
        return EXIT_SETUP_FAILED

    # Phase 4: Install skills
    if not install_skills(config, state):
        log.error("Skill installation failed")
        return EXIT_SETUP_FAILED

    # Phase 5: Configure Claude Code tracing
    if not setup_claude_code_tracing(config, state):
        log.error("Claude Code tracing setup failed")
        return EXIT_SETUP_FAILED

    # Phase 6: Test Claude Code headless mode
    if not test_claude_headless(config, state):
        log.error("Claude Code headless mode test failed")
        return EXIT_SETUP_FAILED

    # Phase 7: Run Claude Code with prompt
    if not run_claude_code(config, state):
        log.error("Claude Code execution failed")
        log.error(f"Check log file: {state.log_file}")
        return EXIT_EXECUTION_FAILED

    # Phase 8: Verify by running judges on traces
    if not verify_judges(config, state):
        log.error("Judge verification failed")
        return EXIT_VERIFICATION_FAILED

    log_section("Test Completed Successfully")
    log.info(f"Evaluation experiment ID: {state.experiment_id}")
    log.info("All registered judges passed on all traces")

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
