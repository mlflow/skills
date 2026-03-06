"""Test environment setup and teardown."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from config import TestConfig, RuntimeState
from utils import (
    log_section,
    run_command,
    claude_env,
    is_port_available,
)

log = logging.getLogger(__name__)


def _is_databricks_uri(uri: str) -> bool:
    """Return True for 'databricks' or 'databricks://<profile>'."""
    return uri == "databricks" or uri.startswith("databricks://")


def check_prerequisites(config: TestConfig, state: RuntimeState) -> bool:
    log_section("Checking Prerequisites")

    # Check skill directories exist
    for skill_name in config.skills:
        skill_dir = state.repo_root / skill_name
        if not skill_dir.exists():
            log.error(f"Skill directory not found: {skill_dir}")
            return False
        if not (skill_dir / "SKILL.md").exists():
            log.error(f"SKILL.md not found in: {skill_dir}")
            return False
        log.info(f"Skill directory found: {skill_dir}")

    # Check setup script exists
    setup_script = state.repo_root / config.setup_script
    if not setup_script.exists():
        log.error(f"Setup script not found: {setup_script}")
        return False
    log.info(f"Setup script found: {setup_script}")

    # Check judges modules exist
    for judge_path in config.judges:
        judges_module = state.repo_root / judge_path
        if not judges_module.exists():
            log.error(f"Judges module not found: {judges_module}")
            return False
        log.info(f"Judges module found: {judges_module}")

    # Check external server or port availability
    if config.tracking_uri:
        state.use_external_server = True
        log.info(f"Using external MLflow server: {config.tracking_uri}")
    else:
        if not is_port_available(config.mlflow_port):
            log.error(f"Port {config.mlflow_port} is already in use")
            log.error("Set mlflow_port in YAML to use a different port")
            return False
        log.info(f"Port {config.mlflow_port} is available")

    return True


def start_mlflow_server(config: TestConfig, state: RuntimeState) -> bool:
    log_section("Starting Local MLflow Server")

    mlflow_data_dir = state.work_dir / "mlflow-data"
    mlflow_data_dir.mkdir(parents=True, exist_ok=True)
    (mlflow_data_dir / "artifacts").mkdir(exist_ok=True)

    backend_store = f"sqlite:///{mlflow_data_dir}/mlflow.db"
    artifact_root = str(mlflow_data_dir / "artifacts")

    log.info(f"Backend store: {backend_store}")
    log.info(f"Artifact root: {artifact_root}")
    log.info(f"Starting server on port {config.mlflow_port}...")

    # Start MLflow server in background with its own session so it doesn't
    # get killed by signals sent to the test runner's process group.
    log_file = state.work_dir / "mlflow-server.log"
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            [
                "python",
                "-m",
                "mlflow",
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                str(config.mlflow_port),
                "--backend-store-uri",
                backend_store,
                "--default-artifact-root",
                artifact_root,
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=state.full_project_dir,
            start_new_session=True,
        )

    state.mlflow_server_pid = process.pid

    # Wait for server to be ready
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            result = subprocess.run(
                ["curl", "-s", f"http://127.0.0.1:{config.mlflow_port}/health"],
                capture_output=True,
                timeout=2,
            )
            if result.returncode == 0:
                break
        except Exception:
            pass

        # Check if process died
        if process.poll() is not None:
            log.error("MLflow server process died")
            log.error(f"Check log: {log_file}")
            with open(log_file) as f:
                print(f.read(), file=sys.stderr)
            return False

        time.sleep(1)
    else:
        log.error("MLflow server failed to start (timeout)")
        log.error(f"Check log: {log_file}")
        with open(log_file) as f:
            print(f.read(), file=sys.stderr)
        return False

    log.info(f"MLflow server started (PID: {state.mlflow_server_pid})")

    tracking_uri = f"http://127.0.0.1:{config.mlflow_port}"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    log.info(f"MLFLOW_TRACKING_URI set to: {tracking_uri}")
    return True


def setup_infrastructure(config: TestConfig, state: RuntimeState) -> bool:
    log_section("Setup Infrastructure")

    # Create working directory
    config.test_runs_dir.mkdir(exist_ok=True)
    state.work_dir = Path(
        tempfile.mkdtemp(prefix=f"{config.name}-", dir=config.test_runs_dir)
    )
    state.full_project_dir = state.work_dir / config.project_dir
    log.info(f"Created working directory: {state.work_dir}")

    # Start local MLflow server or use external one
    if state.use_external_server:
        log_section("Using External MLflow Server")
        os.environ["MLFLOW_TRACKING_URI"] = config.tracking_uri
        log.info(f"MLFLOW_TRACKING_URI set to: {config.tracking_uri}")
    else:
        # Create project dir early so MLflow server can use it as cwd.
        # The setup script will populate it with actual content.
        state.full_project_dir.mkdir(parents=True, exist_ok=True)
        if not start_mlflow_server(config, state):
            return False

    # Create evaluation experiment
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_experiment_name = f"{config.name}-{timestamp}"

    if state.use_external_server and _is_databricks_uri(config.tracking_uri):
        db_profile = config.tracking_uri.replace("databricks://", "")
        cmd = ["databricks", "current-user", "me"]
        if db_profile:
            cmd += ["-p", db_profile]
            log.info(f"Detecting Databricks workspace user for profile: {db_profile}")
        else:
            log.info("Detecting Databricks workspace user (default profile)")
        try:
            result = run_command(cmd)
            user_data = json.loads(result.stdout)
            db_user = user_data.get("userName", "")
        except Exception:
            db_user = ""

        if not db_user:
            log.error(
                "Failed to get Databricks user. "
                "Make sure 'databricks auth login' has been run."
            )
            return False

        experiment_name = f"/Users/{db_user}/{base_experiment_name}"
        log.info("Using Databricks workspace path for experiment")
    else:
        experiment_name = base_experiment_name

    log.info(f"Creating evaluation experiment: {experiment_name}")

    try:
        state.experiment_id = str(mlflow.create_experiment(experiment_name))
        log.info(f"Created evaluation experiment with ID: {state.experiment_id}")
    except Exception as e:
        log.error(f"Failed to create experiment: {e}")
        return False

    os.environ["MLFLOW_EXPERIMENT_ID"] = state.experiment_id

    # Create Claude Code tracing experiment
    cc_base_name = f"claude-code-skill-{timestamp}"
    if state.use_external_server and _is_databricks_uri(config.tracking_uri):
        user_prefix = "/".join(experiment_name.split("/")[:-1])
        cc_tracing_experiment_name = f"{user_prefix}/{cc_base_name}"
    else:
        cc_tracing_experiment_name = cc_base_name

    log.info(
        f"Creating Claude Code tracing experiment: {cc_tracing_experiment_name}"
    )
    try:
        state.cc_tracing_experiment_id = str(
            mlflow.create_experiment(cc_tracing_experiment_name)
        )
        log.info(
            f"Created tracing experiment with ID: {state.cc_tracing_experiment_id}"
        )
    except Exception as e:
        log.error(f"Failed to create tracing experiment: {e}")
        return False

    return True


def run_setup_script(config: TestConfig, state: RuntimeState) -> bool:
    log_section("Running Setup Script")

    setup_script = state.repo_root / config.setup_script
    log.info(f"Executing: {setup_script}")

    env = {
        "WORK_DIR": str(state.work_dir),
        "PROJECT_DIR": str(state.full_project_dir),
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
        "MLFLOW_EXPERIMENT_ID": state.experiment_id,
        "CC_EXPERIMENT_ID": state.cc_tracing_experiment_id,
        "REPO_ROOT": str(state.repo_root),
    }

    # Merge user-defined environment from YAML config
    env.update(config.environment)

    try:
        run_command(
            ["python", str(setup_script)],
            cwd=state.work_dir,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        log.error(f"Setup script failed: {e}")
        if e.stderr:
            log.error(f"stderr: {e.stderr}")
        return False

    if not state.full_project_dir.exists():
        log.error(
            f"Setup script did not create project directory: {state.full_project_dir}"
        )
        return False

    log.info("Setup script completed")
    return True


def install_skills(config: TestConfig, state: RuntimeState) -> bool:
    log_section("Installing Skills")

    skills_dir = state.full_project_dir / ".claude" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    for skill_name in config.skills:
        src = state.repo_root / skill_name
        dst = skills_dir / skill_name
        shutil.copytree(src, dst)
        log.info(f"Installed skill: {skill_name} -> {dst}")

    return True


def setup_claude_code_tracing(config: TestConfig, state: RuntimeState) -> bool:
    log_section("Setting Up Claude Code Tracing")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")

    try:
        cmd = [
            "mlflow",
            "autolog",
            "claude",
            str(state.full_project_dir),
            "-u",
            tracking_uri,
            "-e",
            state.cc_tracing_experiment_id,
        ]
        run_command(cmd, cwd=state.full_project_dir)
        log.info("MLflow autolog configured for Claude Code")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to configure Claude Code tracing: {e}")
        return False

    return True


def test_claude_headless(config: TestConfig, state: RuntimeState) -> bool:
    log_section("Testing Claude Code Headless Mode")

    log.info("Running simple Claude Code test query...")
    try:
        result = subprocess.run(
            ["claude", "-p", "Say hello world", "--allowedTools", ""],
            cwd=state.full_project_dir,
            capture_output=True,
            text=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
            env=claude_env(),
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        log.error("Claude Code headless mode test timed out")
        return False
    except Exception as e:
        log.error(f"Claude Code headless mode test failed: {e}")
        return False

    if not output:
        log.error("Claude Code headless mode produced no output")
        return False

    if "error" in output.lower():
        log.error(f"Claude Code headless mode returned an error: {output}")
        return False

    log.info(f"Claude Code responded: {output[:100]}...")

    # Verify Claude Code tracing worked
    log.info("Verifying Claude Code tracing captured the test query...")
    mlflow.flush_trace_async_logging()

    client = MlflowClient()
    traces = client.search_traces(experiment_ids=[state.cc_tracing_experiment_id])
    if not traces:
        log.error("Claude Code tracing verification failed - no traces found")
        return False

    log.info(f"Found {len(traces)} trace(s), first ID: {traces[0].info.request_id}")
    return True


def cleanup(config: TestConfig, state: RuntimeState) -> None:
    log_section("Cleanup")

    # Copy Claude session logs to work directory
    if state.work_dir and state.work_dir.exists() and state.full_project_dir:
        project_path_encoded = str(state.full_project_dir).replace("/", "-")
        session_dir = Path.home() / ".claude" / "projects" / project_path_encoded

        if session_dir.exists():
            dest_dir = state.work_dir / "claude-sessions"
            try:
                shutil.copytree(session_dir, dest_dir, dirs_exist_ok=True)
                log.info(f"Claude session logs copied to: {dest_dir}")
            except Exception as e:
                log.error(f"Failed to copy session logs: {e}")

    # Stop MLflow server if we started one
    if state.mlflow_server_pid and not state.use_external_server:
        log.info(f"Stopping MLflow server (PID: {state.mlflow_server_pid})...")
        try:
            # Kill the entire process group since server runs in its own session
            os.killpg(os.getpgid(state.mlflow_server_pid), signal.SIGTERM)
            for _ in range(10):
                try:
                    os.kill(state.mlflow_server_pid, 0)
                    time.sleep(0.5)
                except OSError:
                    break
            log.info("MLflow server stopped")
        except OSError:
            pass

    # Remove or keep working directory
    if state.work_dir and state.work_dir.exists():
        if not config.keep_workdir:
            log.info(f"Removing working directory: {state.work_dir}")
            shutil.rmtree(state.work_dir)
        else:
            log.info(f"Keeping working directory: {state.work_dir}")
            log.info(f"  Claude Code output log: {state.log_file or 'N/A'}")
            log.info(f"  Claude session logs: {state.work_dir}/claude-sessions/")
            if state.use_external_server:
                log.info(f"  MLflow tracking URI: {config.tracking_uri}")
            else:
                log.info(f"  MLflow data: {state.work_dir}/mlflow-data")
            log.info(f"  Evaluation experiment ID: {state.experiment_id or 'N/A'}")
            log.info(
                f"  Claude Code tracing experiment ID: "
                f"{state.cc_tracing_experiment_id or 'N/A'}"
            )
