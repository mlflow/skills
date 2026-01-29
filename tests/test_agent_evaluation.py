#!/usr/bin/env python3
"""
Test script for the agent-evaluation skill.

This script:
1. Clones a fresh checkout of mlflow-agent
2. Starts a local MLflow tracking server (SQLite backend)
3. Installs the agent-evaluation skill in .claude/skills/
4. Tests Claude Code headless mode with a simple query
5. Runs Claude Code evaluation (skill should be auto-discovered)
6. Verifies that evaluation artifacts were created
7. Copies Claude session logs for inspection
8. Cleans up (stops server, removes temp files unless --keep-workdir)

Usage:
    python tests/test_agent_evaluation.py [OPTIONS] [extra_prompt]

Options:
    --skill-dir PATH         Path to agent-evaluation skill
    --timeout SECONDS        Claude Code execution timeout (default: 900)
    --mlflow-port PORT       Port for local MLflow server (default: 5000)
    --keep-workdir           Keep the working directory after completion
    --tracking-uri URI       Use external MLflow server instead of local

Environment variables:
    OPENAI_API_KEY           Passed to Claude Code for MLflow's default judge
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# Exit codes
EXIT_SUCCESS = 0
EXIT_SETUP_FAILED = 1
EXIT_EXECUTION_FAILED = 2
EXIT_VERIFICATION_FAILED = 3


@dataclass
class Config:
    """Test configuration."""

    skill_dir: Path
    timeout_seconds: int = 900
    mlflow_port: int = 5000
    keep_workdir: bool = True
    extra_prompt: str = ""
    tracking_uri: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Runtime state
    work_dir: Optional[Path] = None
    experiment_name: Optional[str] = None
    experiment_id: Optional[str] = None
    log_file: Optional[Path] = None
    mlflow_server_pid: Optional[int] = None
    use_external_server: bool = False

    # Constants
    mlflow_agent_repo: str = "https://github.com/alkispoly-db/mlflow-agent"


class Logger:
    """Simple logger with colored output."""

    @staticmethod
    def info(msg: str) -> None:
        print(f"[INFO] {msg}")

    @staticmethod
    def error(msg: str) -> None:
        print(f"[ERROR] {msg}", file=sys.stderr)

    @staticmethod
    def success(msg: str) -> None:
        print(f"[SUCCESS] {msg}")

    @staticmethod
    def section(msg: str) -> None:
        print()
        print("=" * 40)
        print(msg)
        print("=" * 40)


log = Logger()


def run_command(
    cmd: list[str],
    cwd: Optional[Path] = None,
    capture_output: bool = True,
    check: bool = True,
    timeout: Optional[int] = None,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture_output,
        text=True,
        check=check,
        timeout=timeout,
        env=merged_env,
    )


def command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(cmd) is not None


def is_port_available(port: int) -> bool:
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def cleanup(config: Config) -> None:
    """Cleanup function called on exit."""
    log.section("Cleanup")

    # Copy Claude session logs to work directory
    if config.work_dir and config.work_dir.exists():
        claude_projects_dir = Path.home() / ".claude" / "projects"
        project_path_encoded = str(config.work_dir / "mlflow-agent").replace("/", "-")
        session_dir = claude_projects_dir / project_path_encoded

        if session_dir.exists():
            log.info("Copying Claude session logs...")
            dest_dir = config.work_dir / "claude-sessions"
            dest_dir.mkdir(exist_ok=True)
            try:
                for item in session_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dest_dir)
                    else:
                        shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)
                log.info(f"Claude session logs copied to: {dest_dir}")
            except Exception as e:
                log.error(f"Failed to copy session logs: {e}")

    # Stop MLflow server if we started one
    if config.mlflow_server_pid and not config.use_external_server:
        log.info(f"Stopping MLflow server (PID: {config.mlflow_server_pid})...")
        try:
            os.kill(config.mlflow_server_pid, signal.SIGTERM)
            # Wait for process to terminate
            for _ in range(10):
                try:
                    os.kill(config.mlflow_server_pid, 0)
                    time.sleep(0.5)
                except OSError:
                    break
            log.info("MLflow server stopped")
        except OSError:
            pass

    # Remove or keep working directory
    if config.work_dir and config.work_dir.exists():
        if not config.keep_workdir:
            log.info(f"Removing working directory: {config.work_dir}")
            shutil.rmtree(config.work_dir)
        else:
            log.info(f"Keeping working directory: {config.work_dir}")
            log.info(f"  Claude Code output log: {config.log_file or 'N/A'}")
            log.info(f"  Claude session logs: {config.work_dir}/claude-sessions/")
            if config.use_external_server:
                log.info(f"  MLflow tracking URI: {config.tracking_uri}")
            else:
                log.info(f"  MLflow data: {config.work_dir}/mlflow-data")
            log.info(f"  Evaluation experiment ID: {config.experiment_id or 'N/A'}")


def check_prerequisites(config: Config) -> bool:
    """Check that all prerequisites are met."""
    log.section("Checking Prerequisites")

    missing_deps = []

    # Check required commands
    if not command_exists("claude"):
        missing_deps.append("claude (Claude Code CLI)")
    if not command_exists("git"):
        missing_deps.append("git")
    if not command_exists("uv"):
        missing_deps.append("uv (Python package manager)")

    if missing_deps:
        log.error("Missing required commands:")
        for dep in missing_deps:
            log.error(f"  - {dep}")
        return False

    log.info("All required commands available: claude, git, uv")

    # Check skill directory
    if not config.skill_dir.exists():
        log.error(f"Skill directory not found: {config.skill_dir}")
        return False

    if not (config.skill_dir / "SKILL.md").exists():
        log.error(f"SKILL.md not found in: {config.skill_dir}")
        return False

    log.info(f"Skill directory found: {config.skill_dir}")

    # Check external server or port availability
    if config.tracking_uri:
        config.use_external_server = True
        log.info(f"Using external MLflow server: {config.tracking_uri}")
    else:
        if not is_port_available(config.mlflow_port):
            log.error(f"Port {config.mlflow_port} is already in use")
            log.error("Set --mlflow-port to use a different port")
            return False
        log.info(f"Port {config.mlflow_port} is available")

    log.success("All prerequisites satisfied")
    return True


def start_mlflow_server(config: Config) -> bool:
    """Start a local MLflow server."""
    log.section("Starting Local MLflow Server")

    mlflow_data_dir = config.work_dir / "mlflow-data"
    mlflow_data_dir.mkdir(parents=True, exist_ok=True)
    (mlflow_data_dir / "artifacts").mkdir(exist_ok=True)

    backend_store = f"sqlite:///{mlflow_data_dir}/mlflow.db"
    artifact_root = str(mlflow_data_dir / "artifacts")

    log.info(f"Backend store: {backend_store}")
    log.info(f"Artifact root: {artifact_root}")
    log.info(f"Starting server on port {config.mlflow_port}...")

    # Start MLflow server in background
    log_file = config.work_dir / "mlflow-server.log"
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            [
                "uv",
                "run",
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
            cwd=config.work_dir / "mlflow-agent",
        )

    config.mlflow_server_pid = process.pid

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
        except (subprocess.TimeoutExpired, Exception):
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

    log.info(f"MLflow server started (PID: {config.mlflow_server_pid})")

    # Set tracking URI
    tracking_uri = f"http://127.0.0.1:{config.mlflow_port}"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    log.info(f"MLFLOW_TRACKING_URI set to: {tracking_uri}")

    log.success("MLflow server is ready")
    return True


def setup_phase(config: Config) -> bool:
    """Set up the test environment."""
    log.section("Setup Phase")

    # Create working directory
    repo_root = config.skill_dir.parent
    test_runs_dir = repo_root / "test-runs"
    test_runs_dir.mkdir(exist_ok=True)
    config.work_dir = Path(tempfile.mkdtemp(prefix="agent-eval-test-", dir=test_runs_dir))
    log.info(f"Created working directory: {config.work_dir}")

    # Clone mlflow-agent repository
    log.info("Cloning mlflow-agent repository...")
    try:
        run_command(
            ["git", "clone", "--depth", "1", config.mlflow_agent_repo, str(config.work_dir / "mlflow-agent")]
        )
        log.info("Repository cloned successfully")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to clone repository: {e}")
        return False

    # Copy skill to .claude/skills directory
    log.info("Setting up agent-evaluation skill in .claude/skills/...")
    skills_dir = config.work_dir / "mlflow-agent" / ".claude" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config.skill_dir, skills_dir / "agent-evaluation")
    log.info(f"Skill installed at: {skills_dir / 'agent-evaluation'}")

    project_dir = config.work_dir / "mlflow-agent"

    # Install dependencies
    log.info("Installing Python dependencies with uv sync...")
    try:
        run_command(["uv", "sync"], cwd=project_dir)
        log.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to install dependencies: {e}")
        return False

    # Add mlflow package
    log.info("Adding mlflow package...")
    try:
        run_command(["uv", "add", "mlflow"], cwd=project_dir)
        log.info("MLflow package added")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to add mlflow package: {e}")
        return False

    # Start local MLflow server or use external one
    if config.use_external_server:
        log.section("Using External MLflow Server")
        os.environ["MLFLOW_TRACKING_URI"] = config.tracking_uri
        log.info(f"MLFLOW_TRACKING_URI set to: {config.tracking_uri}")

        # Install Databricks packages if needed
        if config.tracking_uri.startswith("databricks://"):
            log.info("Installing Databricks packages for Unity Catalog dataset support...")
            try:
                run_command(["uv", "add", "databricks-agents", "databricks-connect"], cwd=project_dir)
                log.info("Databricks packages installed (databricks-agents, databricks-connect)")
            except subprocess.CalledProcessError:
                log.error("Failed to install Databricks packages")
                log.error("Dataset operations may fail without databricks-agents and databricks-connect")
                # Don't fail - evaluation can still work with local DataFrames

        log.success("External MLflow server configured")
    else:
        if not start_mlflow_server(config):
            return False

    # Create test experiment
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_experiment_name = f"agent-eval-test-{timestamp}"

    # For Databricks, experiment names must be absolute workspace paths
    if config.use_external_server and config.tracking_uri.startswith("databricks://"):
        db_profile = config.tracking_uri.replace("databricks://", "")
        log.info(f"Detecting Databricks workspace user for profile: {db_profile}")

        try:
            result = run_command(["databricks", "current-user", "me", "-p", db_profile])
            user_data = json.loads(result.stdout)
            db_user = user_data.get("userName", "")
        except Exception:
            db_user = ""

        if not db_user:
            log.error(f"Failed to get Databricks user. Make sure 'databricks auth login -p {db_profile}' has been run.")
            return False

        config.experiment_name = f"/Users/{db_user}/{base_experiment_name}"
        log.info("Using Databricks workspace path for experiment")
    else:
        config.experiment_name = base_experiment_name

    log.info(f"Creating test experiment: {config.experiment_name}")

    # Create experiment
    try:
        result = run_command(
            [
                "uv",
                "run",
                "python",
                "-c",
                f"import mlflow; print(mlflow.create_experiment('{config.experiment_name}'))",
            ],
            cwd=project_dir,
        )
        config.experiment_id = result.stdout.strip()

        if not config.experiment_id or "Error" in config.experiment_id or "Traceback" in config.experiment_id:
            log.error(f"Failed to create test experiment: {config.experiment_id}")
            return False

        log.info(f"Created evaluation experiment with ID: {config.experiment_id}")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to create experiment: {e}")
        return False

    # Set experiment ID
    os.environ["MLFLOW_EXPERIMENT_ID"] = config.experiment_id

    # Set OpenAI API key if provided
    if config.openai_api_key:
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
        log.info("OPENAI_API_KEY set (for MLflow scorers)")

    log.success("Setup phase completed")
    return True


def test_claude_headless(config: Config) -> bool:
    """Test that Claude Code headless mode works."""
    log.section("Testing Claude Code Headless Mode")

    project_dir = config.work_dir / "mlflow-agent"

    log.info("Running simple Claude Code test query...")
    env = os.environ.copy()
    try:
        result = subprocess.run(
            ["claude", "-p", "Say hello world", "--allowedTools", ""],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
            env=env,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        log.error("Claude Code headless mode test timed out")
        return False
    except Exception as e:
        log.error(f"Claude Code headless mode test failed: {e}")
        return False

    if not output:
        log.error("Claude Code headless mode test failed - no output")
        return False

    if "error" in output.lower() and "Error" in output:
        log.error("Claude Code headless mode test failed")
        log.error(f"Output: {output}")
        return False

    log.info(f"Claude Code responded: {output[:100]}...")
    log.success("Claude Code headless mode is working")
    return True


def run_claude_code(config: Config) -> bool:
    """Run Claude Code with the evaluation prompt."""
    log.section("Running Claude Code Evaluation")

    config.log_file = config.work_dir / "claude_output.log"
    project_dir = config.work_dir / "mlflow-agent"

    log.info("Changing to mlflow-agent directory...")

    # Build the prompt
    base_prompt = "Evaluate the output quality of my agent. Do not ask for input."
    full_prompt = base_prompt
    if config.extra_prompt:
        full_prompt = f"{base_prompt} {config.extra_prompt}"
        log.info(f"Extra prompt: {config.extra_prompt}")

    log.info("Executing Claude Code with agent-evaluation skill...")
    log.info(f"Prompt: {full_prompt}")
    log.info(f"Timeout: {config.timeout_seconds} seconds")
    log.info(f"Log file: {config.log_file}")
    log.info(f"MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI', 'not set')}")
    log.info(f"MLFLOW_EXPERIMENT_ID: {os.environ.get('MLFLOW_EXPERIMENT_ID', 'not set')}")

    # Run Claude Code with explicit environment
    env = os.environ.copy()
    try:
        with open(config.log_file, "w") as f:
            result = subprocess.run(
                [
                    "claude",
                    "-p",
                    full_prompt,
                    "--dangerously-skip-permissions",
                    "--allowedTools",
                    "Bash,Read,Write,Edit,Grep,Glob,WebFetch",
                ],
                cwd=project_dir,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=config.timeout_seconds,
                stdin=subprocess.DEVNULL,
                env=env,
            )
            exit_code = result.returncode
    except subprocess.TimeoutExpired:
        log.info(f"Claude Code execution timed out after {config.timeout_seconds} seconds")
        log.info("Will still verify if artifacts were created before timeout")
        # Don't return - continue to verification
        return True
    except Exception as e:
        log.error(f"Claude Code execution failed: {e}")
        return False

    if exit_code != 0:
        log.error(f"Claude Code exited with code: {exit_code}")
        log.error(f"Check log file for details: {config.log_file}")
        return False

    log.success("Claude Code execution completed")
    return True


def verify_results(config: Config) -> bool:
    """Verify that evaluation artifacts were created."""
    log.section("Verification Phase")

    project_dir = config.work_dir / "mlflow-agent"

    log.info("Running verification checks...")

    verification_code = f"""
import sys
import json
import subprocess
from mlflow import MlflowClient

experiment_id = '{config.experiment_id}'
client = MlflowClient()

results = {{
    'datasets': {{'found': 0, 'pass': False}},
    'scorers': {{'found': 0, 'pass': False}},
    'traces': {{'found': 0, 'with_assessments': 0, 'pass': False}}
}}

# Check datasets
try:
    datasets = client.search_datasets(experiment_ids=[experiment_id])
    results['datasets']['found'] = len(datasets)
    results['datasets']['pass'] = len(datasets) >= 1
except Exception as e:
    print(f'Warning: Error checking datasets: {{e}}', file=sys.stderr)

# Check traces
try:
    traces = client.search_traces(experiment_ids=[experiment_id])
    results['traces']['found'] = len(traces)
    results['traces']['pass'] = len(traces) >= 1
except Exception as e:
    print(f'Warning: Error checking traces: {{e}}', file=sys.stderr)

# Check scorers
try:
    scorer_result = subprocess.run(
        ['uv', 'run', 'python', '-m', 'mlflow', 'scorers', 'list', '-x', experiment_id],
        capture_output=True,
        text=True
    )
    lines = [l for l in scorer_result.stdout.strip().split('\\n') if l and not l.startswith('---') and 'Name' not in l]
    results['scorers']['found'] = len(lines)
    results['scorers']['pass'] = len(lines) >= 1
except Exception as e:
    print(f'Warning: Error checking scorers: {{e}}', file=sys.stderr)

print(json.dumps(results))
"""

    try:
        result = run_command(
            ["uv", "run", "python", "-c", verification_code],
            cwd=project_dir,
            check=False,
        )
        verification_result = result.stdout.strip()
    except Exception as e:
        log.error(f"Verification script failed: {e}")
        return False

    if not verification_result:
        log.error("Verification script failed to produce output")
        return False

    try:
        results = json.loads(verification_result)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse verification results: {e}")
        log.error(f"Raw output: {verification_result}")
        return False

    log.info("Verification Results:")
    print()

    all_pass = True

    # Datasets check
    datasets = results.get("datasets", {})
    if datasets.get("pass"):
        print(f"  [PASS] Datasets: {datasets.get('found', 0)} created")
    else:
        print(f"  [FAIL] Datasets: {datasets.get('found', 0)} created (expected >= 1)")
        all_pass = False

    # Scorers check
    scorers = results.get("scorers", {})
    if scorers.get("pass"):
        print(f"  [PASS] Scorers: {scorers.get('found', 0)} registered")
    else:
        print(f"  [FAIL] Scorers: {scorers.get('found', 0)} registered (expected >= 1)")
        all_pass = False

    # Traces check
    traces = results.get("traces", {})
    if traces.get("pass"):
        print(f"  [PASS] Evaluation runs: {traces.get('found', 0)} traces created")
    else:
        print(f"  [FAIL] Evaluation runs: {traces.get('found', 0)} traces created (expected >= 1)")
        all_pass = False

    print()

    if all_pass:
        log.success("All verification checks passed")
        return True
    else:
        log.error("Some verification checks failed")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test script for the agent-evaluation skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Determine default skill directory
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent
    default_skill_dir = repo_root / "agent-evaluation"

    parser.add_argument(
        "extra_prompt",
        nargs="?",
        default="",
        help="Optional text to append to the Claude Code prompt",
    )
    parser.add_argument(
        "--skill-dir",
        type=Path,
        default=default_skill_dir,
        help=f"Path to agent-evaluation skill (default: {default_skill_dir})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        dest="timeout_seconds",
        help="Claude Code execution timeout in seconds (default: 900)",
    )
    parser.add_argument(
        "--mlflow-port",
        type=int,
        default=5000,
        help="Port for local MLflow server (default: 5000)",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        default=True,
        help="Keep the working directory after completion (default: True)",
    )
    parser.add_argument(
        "--no-keep-workdir",
        action="store_false",
        dest="keep_workdir",
        help="Remove the working directory after completion",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="Use external MLflow server instead of local (can also set MLFLOW_TRACKING_URI env var)",
    )

    args = parser.parse_args()

    config = Config(
        skill_dir=args.skill_dir.resolve(),
        timeout_seconds=args.timeout_seconds,
        mlflow_port=args.mlflow_port,
        keep_workdir=args.keep_workdir,
        extra_prompt=args.extra_prompt,
        tracking_uri=args.tracking_uri,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Register cleanup handler
    atexit.register(cleanup, config)

    log.section("Agent Evaluation Skill Test")
    log.info(f"Starting test at {datetime.now()}")
    if config.extra_prompt:
        log.info(f"Extra prompt argument: {config.extra_prompt}")
    if config.tracking_uri:
        log.info(f"External MLflow URI: {config.tracking_uri}")
    if config.openai_api_key:
        log.info("OPENAI_API_KEY: provided (for MLflow scorers)")

    # Phase 1: Check prerequisites
    if not check_prerequisites(config):
        log.error("Prerequisites check failed")
        return EXIT_SETUP_FAILED

    # Phase 2: Setup
    if not setup_phase(config):
        log.error("Setup phase failed")
        return EXIT_SETUP_FAILED

    # Phase 3: Test Claude Code headless mode
    if not test_claude_headless(config):
        log.error("Claude Code headless mode test failed")
        return EXIT_SETUP_FAILED

    # Phase 4: Run Claude Code evaluation
    if not run_claude_code(config):
        log.error("Claude Code execution failed")
        log.error(f"Check log file: {config.log_file}")
        return EXIT_EXECUTION_FAILED

    # Phase 5: Verify results
    if not verify_results(config):
        log.error("Verification failed")
        return EXIT_VERIFICATION_FAILED

    log.section("Test Completed Successfully")
    log.info(f"Experiment: {config.experiment_name} (ID: {config.experiment_id})")
    log.info("All evaluation artifacts were created as expected")

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
