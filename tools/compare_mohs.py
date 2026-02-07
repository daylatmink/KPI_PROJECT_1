#!/usr/bin/env python3
"""
Compare MOHS Pareto summaries.
Reads projects/<PROJECT>/mohs_score.json and prints a compact table.
"""

import argparse
import json
from pathlib import Path


def load_score(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MOHS Pareto solutions.")
    parser.add_argument("--project", default="ZOOKEEPER", help="Project key")
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Max Pareto solutions to display (default 20)",
    )
    args = parser.parse_args()

    score_path = Path("projects") / args.project / "mohs_score.json"
    if not score_path.is_file():
        raise FileNotFoundError(score_path)

    score = load_score(score_path)
    summaries = score.get("pareto_summaries", [])
    if not summaries:
        print("No pareto_summaries found in", score_path)
        return 0

    rows = summaries[: max(1, args.top)]
    headers = [
        "label",
        "global_total_score",
        "total_project_cost_usd",
        "makespan_hours",
        "utilization",
    ]

    def fmt(val, key):
        if val is None:
            return ""
        if key in ("global_total_score", "utilization"):
            return f"{float(val):.4f}"
        if key in ("total_project_cost_usd", "makespan_hours"):
            return f"{float(val):.2f}"
        return str(val)

    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(fmt(row.get(h), h)))

    line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep = "-+-".join("-" * col_widths[h] for h in headers)
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(fmt(row.get(h), h).ljust(col_widths[h]) for h in headers))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
