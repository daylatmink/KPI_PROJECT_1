#!/usr/bin/env python3
"""
Render a Gantt chart directly from assignment CSV (Start_Hour/End_Hour).
No dependency links required.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def render_gantt(
    assignment_path: Path,
    output_path: Path,
    title: str,
    dpi: int,
    max_tasks: int,
    seed: int,
    show_labels: bool,
    show_ylabels: bool,
    label_fontsize: int,
    ytick_fontsize: int,
    fig_width: float,
    row_height: float,
    bar_height: float,
    line_width: float,
):
    df = pd.read_csv(assignment_path)
    required = {"Task_ID", "Start_Hour", "End_Hour", "Assigned_To", "Priority"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    df["Duration"] = df["End_Hour"] - df["Start_Hour"]
    df = df.sort_values(["Assigned_To", "Start_Hour", "Task_ID"]).reset_index(drop=True)

    if max_tasks and len(df) > max_tasks:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(df), size=max_tasks, replace=False)
        df = df.iloc[sorted(sample_idx)].reset_index(drop=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assignees = sorted(df["Assigned_To"].dropna().unique())
    assignee_idx = {a: i for i, a in enumerate(assignees)}
    row_count = len(assignees)
    fig_height = max(10, min(120, row_count * row_height))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    priority_colors = {
        "Blocker": (0.6, 0.0, 0.0, 0.9),
        "Critical": (0.85, 0.1, 0.1, 0.9),
        "Major": (0.95, 0.5, 0.1, 0.9),
        "Minor": (0.98, 0.8, 0.2, 0.9),
        "Trivial": (0.2, 0.7, 0.2, 0.9),
    }
    fallback_color = (0.6, 0.6, 0.6, 0.8)

    for i, row in df.iterrows():
        assignee = row["Assigned_To"]
        y = assignee_idx.get(assignee, 0)
        color = priority_colors.get(row.get("Priority"), fallback_color)
        ax.barh(
            y,
            row["Duration"],
            left=row["Start_Hour"],
            height=bar_height,
            color=color,
            edgecolor="black",
            linewidth=line_width,
            alpha=0.85,
        )
        if show_labels:
            label = f"{row['Task_ID']}:{assignee}"
            ax.text(
                row["Start_Hour"] + (row["Duration"] / 2 if row["Duration"] else 0),
                y,
                label,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                weight="bold",
            )

    if show_ylabels:
        ax.set_yticks(np.arange(len(assignees)))
        ax.set_yticklabels([str(a) for a in assignees], fontsize=ytick_fontsize)
    else:
        ax.set_yticks([])
    ax.set_xlabel("Time (Hours)", fontsize=11, weight="bold")
    ax.set_title(title, fontsize=13, weight="bold", pad=16)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    if priority_colors:
        import matplotlib.patches as mpatches
        legend_items = [
            mpatches.Patch(color=priority_colors[p], label=p)
            for p in ["Blocker", "Critical", "Major", "Minor", "Trivial"]
            if p in df["Priority"].unique()
        ]
        if legend_items:
            ax.legend(handles=legend_items, loc="upper right", fontsize=8, frameon=False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description="Render Gantt from assignment CSV (Start_Hour/End_Hour)."
    )
    parser.add_argument("--assignment", required=True, help="Path to assignment CSV")
    parser.add_argument(
        "--output",
        default=None,
        help="Output file (png/pdf). Default: <assignment_dir>/gantt_<algo>.png",
    )
    parser.add_argument("--title", default=None, help="Chart title")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-tasks", type=int, default=0, help="Sample N tasks if too large (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Draw Task_ID and Assigned_To on each bar.",
    )
    parser.add_argument(
        "--show-ylabels",
        action="store_true",
        help="Show Task_ID on Y axis (dense).",
    )
    parser.add_argument("--label-fontsize", type=int, default=6)
    parser.add_argument("--ytick-fontsize", type=int, default=6)
    parser.add_argument("--fig-width", type=float, default=6.0)
    parser.add_argument("--row-height", type=float, default=0.18)
    parser.add_argument("--bar-height", type=float, default=0.6)
    parser.add_argument("--line-width", type=float, default=0.2)
    args = parser.parse_args()
    show_labels = False if "--show-labels" not in sys.argv else args.show_labels
    show_ylabels = True if "--show-ylabels" not in sys.argv else args.show_ylabels

    assignment_path = Path(args.assignment)
    if not assignment_path.is_file():
        raise FileNotFoundError(assignment_path)

    if args.output:
        output_path = Path(args.output)
    else:
        stem = assignment_path.stem.replace("_assignment", "")
        output_path = assignment_path.parent / f"gantt_{stem}.png"

    title = args.title or f"Gantt Chart - {assignment_path.stem}"

    render_gantt(
        assignment_path=assignment_path,
        output_path=output_path,
        title=title,
        dpi=args.dpi,
        max_tasks=args.max_tasks,
        seed=args.seed,
        show_labels=show_labels,
        show_ylabels=show_ylabels,
        label_fontsize=args.label_fontsize,
        ytick_fontsize=args.ytick_fontsize,
        fig_width=args.fig_width,
        row_height=args.row_height,
        bar_height=args.bar_height,
        line_width=args.line_width,
    )
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
