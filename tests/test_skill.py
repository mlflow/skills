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

import mlflow
from mlflow import MlflowClient

try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML is required for the test runner. "
        "Install it with: pip install pyyaml"
    )


# Exit codes
EXIT_SUCCESS = 0
EXIT_SETUP_FAILED = 1
EXIT_EXECUTION_FAILED = 2
EXIT_VERIFICATION_FAILED = 3

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def log_section(msg: str) -> None:
    print()
    log.info("=" * 40)
    log.info(msg)
    log.info("=" * 40)


@dataclass
class TestConfig:
    name: str
    project_dir: str
    setup_script: str
    judges: list[str]
    skills: list[str]
    prompt: str
    timeout_seconds: int = 900
    allowed_tools: str = "Bash,Read,Write,Edit,Grep,Glob,WebFetch"
    mlflow_port: int = 5000
    tracking_uri: Optional[str] = None
    test_runs_dir: Path = field(default_factory=lambda: Path("/tmp"))
    keep_workdir: bool = True
    environment: dict[str, str] = field(default_factory=dict)


@dataclass
class RuntimeState:
    work_dir: Optional[Path] = None
    full_project_dir: Optional[Path] = None
    experiment_id: Optional[str] = None
    log_file: Optional[Path] = None
    mlflow_server_pid: Optional[int] = None
    use_external_server: bool = False
    cc_tracing_experiment_id: Optional[str] = None
    repo_root: Optional[Path] = None
    run_start_timestamp_ms: Optional[int] = None


def run_command(
    cmd: list[str],
    cwd: Optional[Path] = None,
    capture_output: bool = True,
    check: bool = True,
    timeout: Optional[int] = None,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
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
    return shutil.which(cmd) is not None


def claude_env() -> dict[str, str]:
    """Return a copy of os.environ without CLAUDECODE to avoid nested-session conflicts."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    return env


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def load_config(yaml_path: str) -> TestConfig:
    path = Path(yaml_path)
    if not path.exists():
        log.error(f"Config file not found: {yaml_path}")
        sys.exit(EXIT_SETUP_FAILED)

    with open(path) as f:
        data = yaml.safe_load(f)

    if "test_runs_dir" in data:
        data["test_runs_dir"] = Path(data["test_runs_dir"])

    env = data.get("environment")
    if env:
        data["environment"] = {k: str(v) for k, v in env.items()}

    return TestConfig(**data)


def check_prerequisites(config: TestConfig, state: RuntimeState) -> bool:
    log_section("Checking Prerequisites")

    missing_deps = []

    if not command_exists("claude"):
        missing_deps.append("claude (Claude Code CLI)")
    if not command_exists("git"):
        missing_deps.append("git")
    if missing_deps:
        log.error("Missing required commands:")
        for dep in missing_deps:
            log.error(f"  - {dep}")
        return False

    log.info("All required commands available: claude, git")

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

    if state.use_external_server and config.tracking_uri.startswith("databricks://"):
        db_profile = config.tracking_uri.replace("databricks://", "")
        log.info(f"Detecting Databricks workspace user for profile: {db_profile}")
        try:
            result = run_command(["databricks", "current-user", "me", "-p", db_profile])
            user_data = json.loads(result.stdout)
            db_user = user_data.get("userName", "")
        except Exception:
            db_user = ""

        if not db_user:
            log.error(
                f"Failed to get Databricks user. "
                f"Make sure 'databricks auth login -p {db_profile}' has been run."
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
    if state.use_external_server and config.tracking_uri.startswith("databricks://"):
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

    if "Error" in output:
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
    # The subprocess dynamically imports each judges module, calls get_judges(),
    # and runs mlflow.genai.evaluate() on the CC traces.
    judge_paths_repr = repr(judge_paths)
    verification_code = f"""
import contextlib
import importlib.util
import json
import sys

import mlflow

# Pre-import modules that scorer threads will need. Without this, concurrent
# lazy imports of litellm/openai from scorer threads deadlock on Python's
# import lock (see deadlock-thread-dump-analysis.md).
import litellm  # noqa: F401
import mlflow.server.jobs.utils  # noqa: F401

# Load judges from all configured modules
judges = []
for i, module_path in enumerate({judge_paths_repr}):
    spec = importlib.util.spec_from_file_location(f"judges_{{i}}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    judges.extend(module.get_judges())

sys.stderr.write(f'Loaded {{len(judges)}} judge(s)\\n')
sys.stderr.flush()
if not judges:
    print(json.dumps({{"error": "No judges returned by get_judges()"}}))
    sys.exit(0)

# Get traces created after the main run started.
# First check the CC tracing experiment, then fall back to the eval experiment
# (some skills log traces to the eval experiment rather than CC tracing).
cc_experiment_id = '{state.cc_tracing_experiment_id}'
eval_experiment_id = '{state.experiment_id}'
run_start_ms = {state.run_start_timestamp_ms}
filter_str = f"trace.timestamp_ms > {{run_start_ms}}"
trace_df = mlflow.search_traces(
    experiment_ids=[cc_experiment_id],
    filter_string=filter_str,
)
sys.stderr.write(f'Found {{len(trace_df)}} trace(s) after run start\\n')
sys.stderr.flush()
if trace_df.empty:
    sys.stderr.write('No CC traces found, checking eval experiment...\\n')
    trace_df = mlflow.search_traces(
        experiment_ids=[eval_experiment_id],
        filter_string=filter_str,
    )
    sys.stderr.write(f'Found {{len(trace_df)}} trace(s) in eval experiment\\n')
    sys.stderr.flush()
if trace_df.empty:
    print(json.dumps({{"error": "No traces found after run start"}}))
    sys.exit(0)

mlflow.set_experiment(experiment_id=cc_experiment_id)

names = [s.name for s in judges]
sys.stderr.write(f'Running judges: {{names}}\\n')
sys.stderr.flush()
with contextlib.redirect_stdout(sys.stderr):
    eval_result = mlflow.genai.evaluate(
        data=trace_df,
        scorers=judges,
    )

sys.stderr.write('Evaluation complete\\n')
sys.stderr.flush()

# Collect results. Column format is "{{scorer_name}}/value".
results = []
result_df = eval_result.result_df
for _, row in result_df.iterrows():
    trace_id = row.get("trace_id", "unknown")
    for judge in judges:
        val_col = f"{{judge.name}}/value"
        rat_col = f"{{judge.name}}/rationale"
        value = row.get(val_col)
        if value is not None:
            results.append({{
                "scorer": judge.name,
                "trace_id": trace_id,
                "value": str(value),
                "rationale": str(row.get(rat_col, "")),
                "pass": str(value).lower() == "yes",
            }})

print(json.dumps(results))
"""

    try:
        result = run_command(
            ["python", "-c", verification_code],
            cwd=state.full_project_dir,
            check=False,
            timeout=300,
        )
        output = result.stdout.strip()
        stderr_output = result.stderr.strip()
        if stderr_output:
            for line in stderr_output.splitlines():
                log.info(line)
    except subprocess.TimeoutExpired as e:
        log.error("Verification timed out after 300 seconds")
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
