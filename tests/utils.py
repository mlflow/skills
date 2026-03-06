"""Generic helpers (no MLflow dependency)."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
from pathlib import Path
from typing import Optional

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def log_section(msg: str) -> None:
    print()
    log.info("=" * 40)
    log.info(msg)
    log.info("=" * 40)


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
