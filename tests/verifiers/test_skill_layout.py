import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_plugin_registers_every_prefixed_skill_once():
    registered = json.loads((ROOT / ".claude-plugin/plugin.json").read_text())["skills"]
    skill_files = sorted(ROOT.glob("*/SKILL.md"))
    discovered = sorted(f"./{path.parent.name}" for path in skill_files)

    assert sorted(registered) == discovered
    assert len(registered) == len(set(registered))
    assert all(path.parent.name.startswith("mlflow-") for path in skill_files)
    for skill_file in skill_files:
        frontmatter = re.match(r"^---\n(.*?)\n---", skill_file.read_text(), re.DOTALL)
        assert frontmatter
        assert yaml.safe_load(frontmatter.group(1))["name"] == skill_file.parent.name

    catalog = (ROOT / ".claude-plugin/README.md").read_text().split(
        "| Skill | Purpose |", 1
    )[1].split("**1 hook:**", 1)[0]
    assert set(re.findall(r"^\| `([^`]+)` \|", catalog, re.MULTILINE)) == {
        entry.removeprefix("./") for entry in registered
    }


def test_scenarios_reference_existing_skills_without_duplicates():
    registered = {
        entry.removeprefix("./")
        for entry in json.loads((ROOT / ".claude-plugin/plugin.json").read_text())["skills"]
    }
    for path in (ROOT / "tests/configs").rglob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        skills = config.get("skills", [])
        required = config.get("required_skills", [])

        assert len(skills) == len(set(skills)), path
        assert set(required) <= set(skills), path
        assert set(skills) <= registered, path


def test_routers_only_suggest_registered_skills_and_preserve_folds():
    registered = {
        entry.removeprefix("./")
        for entry in json.loads((ROOT / ".claude-plugin/plugin.json").read_text())["skills"]
    }
    dispatcher = (ROOT / "mlflow-agent/SKILL.md").read_text()
    hook = (ROOT / "hooks/mlflow-suggest-hook.py").read_text()

    assert set(re.findall(r"→ `([^`]+)`", dispatcher)) <= registered
    assert set(re.findall(r"Use the `([^`]+)` skill", hook)) <= registered
    assert "references/session-analysis.md" in (
        ROOT / "mlflow-trace-analysis/SKILL.md"
    ).read_text()
    evaluation = (ROOT / "mlflow-evaluation/SKILL.md").read_text()
    assert "references/scorer-design.md" in evaluation
    assert "references/judge-alignment.md" in evaluation


def test_relative_markdown_links_resolve():
    for path in ROOT.rglob("*.md"):
        for target in re.findall(r"\[[^]]*\]\(([^)#]+)(?:#[^)]*)?\)", path.read_text()):
            if "://" not in target and not target.startswith("/"):
                assert (path.parent / target).resolve().exists(), (path, target)


def test_hook_deduplicates_folded_trace_analysis_suggestion():
    result = subprocess.run(
        [sys.executable, ROOT / "hooks/mlflow-suggest-hook.py"],
        input=json.dumps({"prompt": "Why did this multi-turn conversation trace fail?"}),
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.count("`mlflow-trace-analysis`") == 1
