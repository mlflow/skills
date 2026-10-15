#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def main() -> None:
    # Keep the original external-project path; no model/config substitutions.
    repo_url = os.environ.get("REPO_URL", "")
    if repo_url:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, os.environ["PROJECT_DIR"]],
            check=True,
            timeout=120,
        )
        (Path(os.environ["WORK_DIR"]) / "evaluation_expected.json").write_text(
            json.dumps({"case": "external"})
        )
    else:
        from setup_evaluation_cases import main as setup

        setup()


if __name__ == "__main__":
    main()
