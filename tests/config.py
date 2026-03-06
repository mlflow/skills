from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

log = logging.getLogger(__name__)


@dataclass
class TestConfig:
    name: str
    project_dir: str
    setup_script: str
    judges: list[str]
    skills: list[str]
    prompt: str
    timeout_seconds: int = 900
    verification_timeout: int = 300
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
