import argparse
import json
import os
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_PROJECT = "ZOOKEEPER"
ALGO_FILES = {
    "HS": ("hs_assignment.csv", "hs_score.json"),
    "IHS": ("ihs_assignment.csv", "ihs_score.json"),
    "GHS": ("ghs_assignment.csv", "ghs_score.json"),
    "GREEDY": ("greedy_assignment.csv", "greedy_score.json"),
    "MOHS": ("mohs_assignment.csv", "mohs_score.json"),
}


def load_score(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_metrics(score: Dict) -> Dict[str, float]:
    keys = [
        # Objective-focused metrics only
        "global_total_score",
        "total_project_cost_usd",
        "makespan_hours",
        "utilization",
    ]
    out = {}
    for k in keys:
        if k in score:
            out[k] = score[k]
    return out


def load_assignment(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def compare(project_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    dist_rows = []
    for algo, (assign_name, score_name) in ALGO_FILES.items():
        assign_path = os.path.join(project_dir, assign_name)
        score_path = os.path.join(project_dir, score_name)
        if not os.path.isfile(score_path):
            continue
        score = load_score(score_path)
        metrics = pick_metrics(score)
        metrics["algo"] = algo
        rows.append(metrics)

        if os.path.isfile(assign_path):
            df = load_assignment(assign_path)
            if "Assigned_To" in df.columns:
                dist = df["Assigned_To"].value_counts().describe()
                dist_rows.append(
                    {
                        "algo": algo,
                        "assignee_count": df["Assigned_To"].nunique(),
                        "tasks_per_assignee_mean": dist["mean"],
                        "tasks_per_assignee_min": dist["min"],
                        "tasks_per_assignee_max": dist["max"],
                    }
                )

    metrics_df = pd.DataFrame(rows)
    dist_df = pd.DataFrame(dist_rows)
    return metrics_df, dist_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare algorithm outputs for a project."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Project key")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output CSV path for summary table",
    )
    args = parser.parse_args()

    project_dir = os.path.join("projects", args.project)
    if not os.path.isdir(project_dir):
        raise FileNotFoundError(f"Project folder not found: {project_dir}")

    metrics_df, dist_df = compare(project_dir)
    if metrics_df.empty:
        print("No score files found in", project_dir)
        return

    metrics_df = metrics_df.sort_values("algo")
    print("== Summary metrics ==")
    print(metrics_df.to_string(index=False))

    if not dist_df.empty:
        dist_df = dist_df.sort_values("algo")
        print("\n== Assignee distribution ==")
        print(dist_df.to_string(index=False))

    if args.out:
        out_path = os.path.abspath(args.out)
        metrics_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("\nSaved summary CSV:", out_path)


if __name__ == "__main__":
    main()
