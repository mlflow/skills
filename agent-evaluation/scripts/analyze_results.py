"""
Analyze MLflow evaluation results and generate actionable insights.

Supports two formats:
1. CSV output from `mlflow.genai.evaluate()` (primary format)
   Columns: trace_id, request_id, inputs, outputs, {scorer_name}/value,
            {scorer_name}/rationale, ...
2. JSON output from legacy `mlflow traces evaluate` CLI (backward compat)

Usage:
    # Primary: CSV from mlflow.genai.evaluate()
    python scripts/analyze_results.py evaluation_results.csv

    # Legacy: JSON from mlflow traces evaluate
    python scripts/analyze_results.py evaluation_results.json

    # With custom output file
    python scripts/analyze_results.py evaluation_results.csv --output report.md
    python scripts/analyze_results.py evaluation_results.csv --results-path evaluation_results.csv
"""

import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


class EvaluationLoadError(Exception):
    """Raised when an evaluation results file cannot be loaded or parsed."""


# ---------------------------------------------------------------------------
# CSV format (mlflow.genai.evaluate output)
# ---------------------------------------------------------------------------

def load_csv_results(csv_file: str) -> dict[str, list[dict]]:
    """Load evaluation results from CSV file produced by mlflow.genai.evaluate().

    The CSV has one row per trace. Scorer columns follow the pattern:
        {scorer_name}/value      — "yes" / "no" (or True/False variants)
        {scorer_name}/rationale  — free-text explanation

    Additional columns (trace_id, request_id, inputs, outputs, ...) are ignored
    for scoring purposes but `inputs` is used to extract a display query string.

    Returns:
        Dictionary mapping scorer names to list of result dicts.
        Each result dict: {query, trace_id, passed, rationale}
    """
    try:
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        raise EvaluationLoadError(f"File not found: {csv_file}")

    if not rows:
        raise EvaluationLoadError(f"CSV file is empty: {csv_file}")

    fieldnames = rows[0].keys()

    # Identify scorer names from columns ending in "/value"
    scorer_names = []
    for col in fieldnames:
        if col.endswith("/value"):
            scorer_names.append(col[: -len("/value")])

    if not scorer_names:
        raise EvaluationLoadError(
            f"No scorer columns found in CSV (expected columns like '{{scorer}}/value')."
            f" Available columns: {list(fieldnames)}"
        )

    scorer_results: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        trace_id = row.get("trace_id", row.get("request_id", "unknown"))

        # Best-effort: extract a human-readable query from the inputs column
        query = _extract_query_from_cell(row.get("inputs", ""))

        for scorer_name in scorer_names:
            value_col = f"{scorer_name}/value"
            rationale_col = f"{scorer_name}/rationale"

            raw_value = row.get(value_col, "")
            rationale = row.get(rationale_col, "")

            # Skip rows where the scorer produced no result
            if raw_value is None or str(raw_value).strip() == "":
                continue

            passed = _parse_bool_value(raw_value)

            scorer_results[scorer_name].append(
                {
                    "query": query,
                    "trace_id": trace_id,
                    "passed": passed,
                    "rationale": rationale,
                }
            )

    return scorer_results


def _extract_query_from_cell(cell_value: str) -> str:
    """Extract a human-readable query string from an inputs cell.

    The inputs column may be a JSON string like '{"query": "..."}' or
    '{"question": "..."}', or a plain string.
    """
    if not cell_value:
        return "unknown"
    cell_str = str(cell_value).strip()
    # Try JSON parse
    try:
        obj = json.loads(cell_str)
        if isinstance(obj, dict):
            return obj.get("query", obj.get("question", cell_str[:120]))
    except (json.JSONDecodeError, ValueError):
        pass
    return cell_str[:120]


def _parse_bool_value(raw: Any) -> bool:
    """Convert a scorer value cell to True (pass) / False (fail).

    Handles:
    - String "yes" / "no" (MLflow judge output)
    - String "true" / "false"
    - Python bool True / False
    - Numeric 1 / 0
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    s = str(raw).strip().lower()
    return s in {"yes", "true", "1"}


# ---------------------------------------------------------------------------
# JSON format (legacy mlflow traces evaluate CLI output)
# ---------------------------------------------------------------------------

def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def load_json_results(json_file: str) -> dict[str, list[dict]]:
    """Load evaluation results from legacy JSON file (mlflow traces evaluate).

    Parses the structure:
    [{
        "trace_id": "tr-...",
        "inputs": {"query": "..."},
        "assessments": [
            {"name": "scorer", "result": "yes/no", "rationale": "...", "error": null}
        ]
    }]

    Returns:
        Dictionary mapping scorer names to list of result dicts.
    """
    try:
        with open(json_file) as f:
            content = f.read()
    except FileNotFoundError:
        raise EvaluationLoadError(f"File not found: {json_file}")

    content = strip_ansi_codes(content)

    json_start = content.find("[")
    if json_start == -1:
        raise EvaluationLoadError("No JSON array found in file")

    json_content = content[json_start:]
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError as e:
        raise EvaluationLoadError(
            f"Invalid JSON: {e}. First 100 chars after '[': {json_content[:100]}"
        )

    if not isinstance(data, list):
        raise EvaluationLoadError(f"Expected JSON array, got {type(data).__name__}")

    scorer_results: dict[str, list[dict]] = defaultdict(list)

    for trace_result in data:
        trace_id = trace_result.get("trace_id", "unknown")
        inputs = trace_result.get("inputs", {})
        query = inputs.get("query", inputs.get("question", "unknown"))

        for assessment in trace_result.get("assessments", []):
            scorer_name = assessment.get("name", "unknown")
            result = assessment.get("result", "fail")
            rationale = assessment.get("rationale", "")
            error = assessment.get("error")

            if error:
                print(f"  ⚠ Warning: Scorer {scorer_name} had error for trace {trace_id}: {error}")
                continue

            passed = _parse_bool_value(result)

            scorer_results[scorer_name].append(
                {"query": query, "trace_id": trace_id, "passed": passed, "rationale": rationale}
            )

    return scorer_results


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def load_evaluation_results(input_file: str) -> dict[str, list[dict]]:
    """Detect file format and load results accordingly.

    Detection logic:
    - `.csv` extension → CSV format (mlflow.genai.evaluate)
    - `.json` extension → legacy JSON format
    - Unknown extension → try CSV first, fall back to JSON
    """
    suffix = Path(input_file).suffix.lower()
    if suffix == ".csv":
        return load_csv_results(input_file)
    elif suffix == ".json":
        return load_json_results(input_file)
    else:
        # Try CSV first (preferred modern format); if it fails, try JSON.
        # If both fail, raise the original CSV error so the message is meaningful.
        try:
            return load_csv_results(input_file)
        except EvaluationLoadError as csv_err:
            try:
                return load_json_results(input_file)
            except EvaluationLoadError:
                raise csv_err


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def calculate_pass_rates(scorer_results: dict[str, list[dict]]) -> dict[str, dict]:
    """Calculate pass rates for each scorer.

    Returns:
        Dictionary mapping scorer names to {pass_rate, passed, total, grade, emoji}
    """
    pass_rates = {}

    for scorer_name, results in scorer_results.items():
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        pass_rate = (passed / total * 100) if total > 0 else 0

        if pass_rate >= 90:
            grade, emoji = "A", "✓✓"
        elif pass_rate >= 80:
            grade, emoji = "B", "✓"
        elif pass_rate >= 70:
            grade, emoji = "C", "⚠"
        elif pass_rate >= 60:
            grade, emoji = "D", "⚠⚠"
        else:
            grade, emoji = "F", "✗"

        pass_rates[scorer_name] = {
            "pass_rate": pass_rate,
            "passed": passed,
            "total": total,
            "grade": grade,
            "emoji": emoji,
        }

    return pass_rates


def detect_failure_patterns(scorer_results: dict[str, list[dict]]) -> list[dict]:
    """Detect patterns in failed queries.

    Returns:
        List of pattern dicts: {name, queries, priority, description}
    """
    patterns = []
    failures_by_query: dict[str, list[dict]] = defaultdict(list)

    for scorer_name, results in scorer_results.items():
        for result in results:
            if not result["passed"]:
                failures_by_query[result["query"]].append(
                    {
                        "scorer": scorer_name,
                        "rationale": result["rationale"],
                        "trace_id": result["trace_id"],
                    }
                )

    # Multi-failure queries: failing 3+ scorers
    multi_failures = [
        {"query": q, "scorers": [f["scorer"] for f in failures], "count": len(failures)}
        for q, failures in failures_by_query.items()
        if len(failures) >= 3
    ]

    if multi_failures:
        patterns.append(
            {
                "name": "Multi-Failure Queries",
                "description": "Queries failing 3 or more scorers — need comprehensive fixes",
                "queries": multi_failures,
                "priority": "CRITICAL",
            }
        )

    return patterns


def generate_recommendations(
    pass_rates: dict[str, dict], patterns: list[dict]
) -> list[dict]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []

    for scorer_name, metrics in pass_rates.items():
        if metrics["pass_rate"] < 80:
            recommendations.append(
                {
                    "title": f"Improve {scorer_name} performance",
                    "issue": f"Only {metrics['pass_rate']:.1f}% pass rate ({metrics['passed']}/{metrics['total']})",
                    "impact": "Will improve overall evaluation quality",
                    "effort": "Medium",
                    "priority": "HIGH" if metrics["pass_rate"] < 70 else "MEDIUM",
                }
            )

    for pattern in patterns:
        if pattern["priority"] == "CRITICAL":
            recommendations.append(
                {
                    "title": f"Fix {pattern['name'].lower()}",
                    "issue": f"{len(pattern['queries'])} queries failing multiple scorers",
                    "impact": "Critical for baseline quality",
                    "effort": "High",
                    "priority": "CRITICAL",
                }
            )
        elif len(pattern["queries"]) >= 3:
            recommendations.append(
                {
                    "title": f"Address {pattern['name'].lower()}",
                    "issue": pattern["description"],
                    "impact": f"Affects {len(pattern['queries'])} queries",
                    "effort": "Medium",
                    "priority": "HIGH",
                }
            )

    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))

    return recommendations


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    scorer_results: dict[str, list[dict]],
    pass_rates: dict[str, dict],
    patterns: list[dict],
    recommendations: list[dict],
    output_file: str,
) -> None:
    """Generate markdown evaluation report."""
    total_queries = max(len(v) for v in scorer_results.values()) if scorer_results else 0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Agent Evaluation Results Analysis",
        "",
        f"**Generated**: {timestamp}",
        f"**Dataset**: {total_queries} queries evaluated",
        f"**Scorers**: {len(scorer_results)} ({', '.join(scorer_results.keys())})",
        "",
        "## Overall Pass Rates",
        "",
    ]

    for scorer_name, metrics in pass_rates.items():
        lines.append(
            f"  {scorer_name:30} {metrics['pass_rate']:5.1f}%"
            f" ({metrics['passed']}/{metrics['total']}) {metrics['emoji']}"
        )

    lines.extend(["", ""])

    avg_pass_rate = (
        sum(m["pass_rate"] for m in pass_rates.values()) / len(pass_rates) if pass_rates else 0
    )
    lines.append(f"**Average Pass Rate**: {avg_pass_rate:.1f}%")
    lines.extend(["", ""])

    if patterns:
        lines.extend(["## Failure Patterns Detected", ""])
        for i, pattern in enumerate(patterns, 1):
            lines.extend(
                [
                    f"### {i}. {pattern['name']} [{pattern['priority']}]",
                    "",
                    f"**Description**: {pattern['description']}",
                    "",
                    f"**Affected Queries**: {len(pattern['queries'])}",
                    "",
                ]
            )
            for query_info in pattern["queries"][:5]:
                q = query_info["query"]
                lines.append(
                    f'- **Query**: "{q[:100]}{"..." if len(q) > 100 else ""}"'
                )
                lines.append(f"  - Failed scorers: {', '.join(query_info['scorers'])}")
                lines.append("")
            if len(pattern["queries"]) > 5:
                lines.append(f"  _(+{len(pattern['queries']) - 5} more queries)_")
                lines.append("")
            lines.append("")

    if recommendations:
        lines.extend(["## Recommendations", ""])
        for i, rec in enumerate(recommendations, 1):
            lines.extend(
                [
                    f"### {i}. {rec['title']} [{rec['priority']}]",
                    "",
                    f"- **Issue**: {rec['issue']}",
                    f"- **Expected Impact**: {rec['impact']}",
                    f"- **Effort**: {rec['effort']}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Next Steps",
            "",
            "1. Address CRITICAL and HIGH priority recommendations first",
            "2. Re-run evaluation after implementing fixes",
            "3. Compare results to measure improvement",
            "4. Consider expanding dataset to cover identified gaps",
            "",
            "---",
            "",
            f"**Report Generated**: {timestamp}",
            "**Evaluation Framework**: MLflow Agent Evaluation",
            "",
        ]
    )

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"\n✓ Report saved to: {output_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main analysis workflow."""
    print("=" * 60)
    print("MLflow Evaluation Results Analysis")
    print("=" * 60)
    print()

    # Parse arguments
    # Supports both positional and --results-path for the input file
    args = sys.argv[1:]
    if not args:
        print(
            "Usage: python scripts/analyze_results.py <evaluation_results.csv> [--output report.md]"
        )
        print("       python scripts/analyze_results.py --results-path evaluation_results.csv")
        sys.exit(1)

    input_file = None
    output_file = "evaluation_report.md"

    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            output_file = args[i + 1]
            i += 2
        elif args[i] == "--results-path" and i + 1 < len(args):
            input_file = args[i + 1]
            i += 2
        elif not args[i].startswith("--"):
            input_file = args[i]
            i += 1
        else:
            i += 1

    if input_file is None:
        print("✗ No input file specified")
        sys.exit(1)

    # Detect format and load
    suffix = Path(input_file).suffix.lower()
    fmt = "CSV (mlflow.genai.evaluate)" if suffix == ".csv" else (
        "JSON (legacy mlflow traces evaluate)" if suffix == ".json" else "auto-detect"
    )
    print(f"Loading evaluation results from: {input_file}")
    print(f"Format: {fmt}")
    try:
        scorer_results = load_evaluation_results(input_file)
    except EvaluationLoadError as e:
        print(f"✗ {e}")
        sys.exit(1)

    if not scorer_results:
        print("✗ No scorer results found")
        sys.exit(1)

    print(f"✓ Loaded results for {len(scorer_results)} scorer(s)")
    print()

    # Calculate pass rates
    print("Calculating pass rates...")
    pass_rates = calculate_pass_rates(scorer_results)

    print("\nOverall Pass Rates:")
    for scorer_name, metrics in pass_rates.items():
        print(
            f"  {scorer_name:30} {metrics['pass_rate']:5.1f}%"
            f" ({metrics['passed']}/{metrics['total']}) {metrics['emoji']}"
        )
    print()

    # Detect patterns
    print("Detecting failure patterns...")
    patterns = detect_failure_patterns(scorer_results)

    if patterns:
        print(f"✓ Found {len(patterns)} pattern(s)")
        for pattern in patterns:
            print(
                f"  - {pattern['name']}: {len(pattern['queries'])} queries [{pattern['priority']}]"
            )
    else:
        print("  No significant patterns detected")
    print()

    # Generate recommendations
    print("Generating recommendations...")
    recommendations = generate_recommendations(pass_rates, patterns)
    print(f"✓ Generated {len(recommendations)} recommendation(s)")
    print()

    # Generate report
    print("Generating markdown report...")
    generate_report(scorer_results, pass_rates, patterns, recommendations, output_file)
    print()

    print("=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    print()
    print(f"Review the report at: {output_file}")
    print()


if __name__ == "__main__":
    main()
