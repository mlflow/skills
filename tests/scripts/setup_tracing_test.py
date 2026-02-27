#!/usr/bin/env python3
"""Setup for tracing test: clone repo."""

from __future__ import annotations

import os
import subprocess


def main() -> None:
    repo_url = os.environ["REPO_URL"]
    project_dir = os.environ["PROJECT_DIR"]

    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, project_dir],
        check=True,
    )


if __name__ == "__main__":
    main()
