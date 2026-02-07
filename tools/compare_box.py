#!/usr/bin/env python3
"""
Print boxed comparison tables for makespan and cost vs a baseline algorithm.
Reads score JSONs from projects/<PROJECT>/.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


ALGO_FILES = {
    "HS": "hs_score.json",
    "IHS": "ihs_score.json",
    "GHS": "ghs_score.json",
    "MOHS": "mohs_score.json",
    "GREEDY": "greedy_score.json",
}


def load_score(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (value - baseline) / baseline * 100.0


def build_rows(
    scores: Dict[str, Dict],
    baseline_algo: str,
    metric_key: str,
    display_name: str,
) -> List[Tuple[str, float, float]]:
    rows = []
    baseline_val = scores[baseline_algo][metric_key]
    for algo, score in scores.items():
        val = score[metric_key]
        rows.append((algo, float(val), pct_change(float(val), float(baseline_val))))
    return rows


def box_table(title: str, metric_label: str, rows: List[Tuple[str, float, float]], baseline_algo: str) -> str:
    col_algo = "ALGO"
    col_val = metric_label
    col_pct = f"% vs {baseline_algo}"

    algo_width = max(len(col_algo), max(len(r[0]) for r in rows))
    val_width = max(len(col_val), max(len(f"{r[1]:.2f}") for r in rows))
    pct_width = max(len(col_pct), max(len(f"{r[2]:+.1f}%") for r in rows))

    use_unicode = (sys.stdout.encoding or "").lower().startswith("utf")

    if use_unicode:
        def line(l, m, r):
            return l + "═" * (algo_width + 2) + m + "═" * (val_width + 2) + m + "═" * (pct_width + 2) + r
        corners = ("╔", "╦", "╗", "╠", "╬", "╣", "╚", "╩", "╝")
        sep = "│"
    else:
        def line(l, m, r):
            return l + "-" * (algo_width + 2) + m + "-" * (val_width + 2) + m + "-" * (pct_width + 2) + r
        corners = ("+", "+", "+", "+", "+", "+", "+", "+", "+")
        sep = "|"

    out = []
    out.append(line(corners[0], corners[1], corners[2]))
    title_cell = f" {title} "
    total_width = algo_width + val_width + pct_width + 8
    out.append(f"{sep}{title_cell.center(total_width)}{sep}")
    out.append(line(corners[3], corners[4], corners[5]))
    out.append(
        f"{sep} {col_algo.ljust(algo_width)} {sep} {col_val.rjust(val_width)} {sep} {col_pct.rjust(pct_width)} {sep}"
    )
    out.append(line(corners[3], corners[4], corners[5]))
    for algo, val, pct in rows:
        out.append(
            f"{sep} {algo.ljust(algo_width)} {sep} {val:>{val_width}.2f} {sep} {pct:>{pct_width}.1f}% {sep}"
        )
    out.append(line(corners[6], corners[7], corners[8]))
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Boxed comparison vs baseline.")
    parser.add_argument("--project", default="ZOOKEEPER", help="Project key")
    parser.add_argument("--baseline", default="GREEDY", help="Baseline algo (e.g., GREEDY)")
    args = parser.parse_args()

    project_dir = Path("projects") / args.project
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project folder not found: {project_dir}")

    scores: Dict[str, Dict] = {}
    for algo, fname in ALGO_FILES.items():
        path = project_dir / fname
        if path.is_file():
            scores[algo] = load_score(path)

    if args.baseline not in scores:
        raise ValueError(f"Baseline {args.baseline} score not found in {project_dir}")

    rows_makespan = build_rows(scores, args.baseline, "makespan_hours", "MAKESPAN (hours)")
    rows_cost = build_rows(scores, args.baseline, "total_project_cost_usd", "COST (USD)")

    print(box_table("TOTAL EFFECT - VS BASELINE", "MAKESPAN (hours)", rows_makespan, args.baseline))
    print()
    print(box_table("TOTAL EFFECT - VS BASELINE", "COST (USD)", rows_cost, args.baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
