#!/usr/bin/env python3
"""Prepare the original external-repository evaluation scenario."""
import os
import subprocess

repo_url = os.environ.get("REPO_URL")
if not repo_url:
    raise ValueError("agent_evaluation requires REPO_URL; evaluation_nested is the local fixture")
subprocess.run(
    ["git", "clone", "--depth", "1", repo_url, os.environ["PROJECT_DIR"]],
    check=True, timeout=120,
)
