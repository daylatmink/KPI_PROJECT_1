import os
import argparse
import pandas as pd

# Paths can be changed if needed (or overridden via CLI args).
TASKS_FILE = r"projects/ZOOKEEPER/logical_tasks.csv"
MAPPING_FILE = r"projects/ZOOKEEPER/issue_to_task_mapping.csv"
TAGGED_ISSUES_FILE = r"data/interim/all_issues_tagged.csv"
OUTPUT_FILE = r"projects/ZOOKEEPER/logical_tasks_tagged.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign Task_Tag for logical tasks by majority of issue tags.",
    )
    parser.add_argument(
        "--tasks",
        default=TASKS_FILE,
        help="Path to 01_logical_tasks.csv",
    )
    parser.add_argument(
        "--mapping",
        default=MAPPING_FILE,
        help="Path to 01_issue_to_task_mapping.csv",
    )
    parser.add_argument(
        "--issues-tagged",
        default=TAGGED_ISSUES_FILE,
        help="Tagged issues with Task_Tag column",
    )
    parser.add_argument(
        "--project-key",
        default="ZOOKEEPER",
        help="Project key filter (optional).",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help="Output path for 02_logical_tasks_tagged.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tasks_path = os.path.abspath(args.tasks)
    mapping_path = os.path.abspath(args.mapping)
    issues_tagged_path = os.path.abspath(args.issues_tagged)
    output_path = os.path.abspath(args.output)

    for p in [tasks_path, mapping_path, issues_tagged_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")

    tasks = pd.read_csv(tasks_path)
    required_task_cols = ["task_id"]
    missing = [c for c in required_task_cols if c not in tasks.columns]
    if missing:
        raise ValueError(f"Tasks file missing columns: {missing}")

    mapping = pd.read_csv(mapping_path)
    required_map_cols = ["issue_key", "task_id"]
    missing = [c for c in required_map_cols if c not in mapping.columns]
    if missing:
        raise ValueError(f"Mapping file missing columns: {missing}")

    issues = pd.read_csv(issues_tagged_path)
    if "Task_Tag" not in issues.columns:
        raise ValueError("Tagged issues file missing column 'Task_Tag'")
    if "issue_key" not in issues.columns:
        raise ValueError("Tagged issues file missing column 'issue_key'")

    if args.project_key and "project_key" in issues.columns:
        issues = issues[issues["project_key"] == args.project_key]

    merged = mapping.merge(
        issues[["issue_key", "Task_Tag"]],
        on="issue_key",
        how="inner",
    )
    merged["task_id"] = merged["task_id"].astype(str)

    def majority_tag(series):
        counts = series.value_counts()
        if counts.empty:
            return ""
        top = counts.max()
        winners = sorted(counts[counts == top].index.astype(str))
        return winners[0]

    task_to_tag = (
        merged.groupby("task_id")["Task_Tag"]
        .apply(majority_tag)
        .to_dict()
    )

    tasks = tasks.copy()
    tasks["task_id"] = tasks["task_id"].astype(str)
    tasks["Task_Tag"] = tasks["task_id"].map(task_to_tag).fillna("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tasks.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("Saved:", output_path)
    print("Label counts:\n", tasks["Task_Tag"].value_counts())


if __name__ == "__main__":
    main()
