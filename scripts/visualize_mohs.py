import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates


def load_pareto_points(score_path):
    with open(score_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fronts = data.get("pareto_fronts", [])
    rows = []
    for front in fronts:
        level = front.get("Topo_Level")
        batch = front.get("Batch_Index")
        for obj in front.get("pareto_solutions", []):
            row = {
                "Topo_Level": level,
                "Batch_Index": batch,
                "skill_matching": obj.get("skill_matching", 0.0),
                "workload_balance": obj.get("workload_balance", 0.0),
                "priority_respect": obj.get("priority_respect", 0.0),
                "skill_development": obj.get("skill_development", 0.0),
                "total": obj.get("total", 0.0),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_scatter(df, out_dir):
    if df.empty:
        print("No Pareto points to plot.")
        return

    out_path = os.path.join(out_dir, "mohs_pareto_scatter.png")

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        df["skill_matching"],
        df["workload_balance"],
        c=df["priority_respect"],
        s=12,
        cmap="viridis",
        alpha=0.8,
    )
    plt.colorbar(sc, label="priority_respect")
    plt.xlabel("skill_matching")
    plt.ylabel("workload_balance")
    plt.title("MOHS Pareto Scatter (color = priority_respect)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved:", out_path)


def plot_parallel(df, out_dir):
    if df.empty:
        print("No Pareto points to plot.")
        return

    out_path = os.path.join(out_dir, "mohs_parallel_coordinates.png")
    plot_df = df.copy()
    plot_df["group"] = (
        plot_df["Topo_Level"].astype(str) + "-" + plot_df["Batch_Index"].astype(str)
    )

    # Limit groups to avoid extreme clutter
    max_groups = 20
    groups = plot_df["group"].unique().tolist()
    if len(groups) > max_groups:
        keep = groups[:max_groups]
        plot_df = plot_df[plot_df["group"].isin(keep)]

    cols = [
        "skill_matching",
        "workload_balance",
        "priority_respect",
        "skill_development",
        "total",
    ]

    plt.figure(figsize=(9, 5))
    parallel_coordinates(
        plot_df[cols + ["group"]],
        class_column="group",
        colormap=plt.cm.tab20,
        alpha=0.4,
    )
    plt.title("MOHS Parallel Coordinates (subset of batches)")
    plt.xlabel("Objectives")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved:", out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MOHS Pareto fronts.")
    parser.add_argument(
        "--score",
        default="projects/ZOOKEEPER/mohs_score.json",
        help="Path to mohs_score.json",
    )
    parser.add_argument(
        "--out-dir",
        default="projects/ZOOKEEPER/mohs_plots",
        help="Output directory for plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = load_pareto_points(args.score)
    plot_scatter(df, out_dir)
    plot_parallel(df, out_dir)


if __name__ == "__main__":
    main()
