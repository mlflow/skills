#!/usr/bin/env python3
"""Scaffold an already-instrumented agent with a deliberate routing bug."""

from __future__ import annotations

import os
from pathlib import Path


SOURCE = '''\
import mlflow


@mlflow.trace
def route_request(message: str) -> str:
    if "refund" in message.lower():
        return "sales"  # Bug: refund requests belong to support.
    return "general"


if __name__ == "__main__":
    print(route_request("I need a refund"))
    mlflow.flush_trace_async_logging()
'''


def main() -> None:
    project_dir = Path(os.environ["PROJECT_DIR"])
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "agent.py").write_text(SOURCE, encoding="utf-8")
    (project_dir / "README.md").write_text(
        "Refund requests should route to support. MLflow tracing is already enabled.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
