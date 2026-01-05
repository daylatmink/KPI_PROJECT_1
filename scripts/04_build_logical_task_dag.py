import os
import argparse
from collections import defaultdict

import pandas as pd


DEFAULT_TASKS = r"projects/ZOOKEEPER/logical_tasks_tagged.csv"
DEFAULT_EDGES = r"projects/ZOOKEEPER/issue_dag_edges.csv"
DEFAULT_OUT_NODES = r"projects/ZOOKEEPER/logical_dag_nodes.csv"
DEFAULT_OUT_EDGES = r"projects/ZOOKEEPER/logical_dag_edges.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build logical-task DAG from issue-level edges.",
    )
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--edges", default=DEFAULT_EDGES)
    parser.add_argument("--out-nodes", default=DEFAULT_OUT_NODES)
    parser.add_argument("--out-edges", default=DEFAULT_OUT_EDGES)
    return parser.parse_args()


def build_issue_to_task_map(tasks_df: pd.DataFrame) -> dict:
    issue_to_task = {}
    for _, row in tasks_df.iterrows():
        task_id = str(row["task_id"])
        keys_raw = str(row.get("issue_keys") or "").strip()
        if not keys_raw:
            continue
        for key in keys_raw.split("|"):
            key = key.strip()
            if key:
                issue_to_task[key] = task_id
    return issue_to_task


def main():
    args = parse_args()

    tasks_path = os.path.abspath(args.tasks)
    edges_path = os.path.abspath(args.edges)
    out_nodes = os.path.abspath(args.out_nodes)
    out_edges = os.path.abspath(args.out_edges)

    for p in [tasks_path, edges_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")

    tasks = pd.read_csv(tasks_path)
    required_task_cols = [
        "task_id",
        "project_key",
        "issue_count",
        "issue_keys",
        "representative_summary",
        "main_priority",
        "main_issuetype",
        "Task_Tag",
    ]
    missing = [c for c in required_task_cols if c not in tasks.columns]
    if missing:
        raise ValueError(f"Tasks file missing columns: {missing}")

    # Drop tasks with empty issue list so edges only reference valid tasks.
    tasks = tasks.copy()
    tasks["issue_keys"] = tasks["issue_keys"].fillna("").astype(str)
    tasks = tasks[tasks["issue_keys"].str.strip() != ""].copy()

    edges = pd.read_csv(edges_path)
    required_edge_cols = ["from_issue_key", "to_issue_key"]
    missing = [c for c in required_edge_cols if c not in edges.columns]
    if missing:
        raise ValueError(f"Edges file missing columns: {missing}")

    issue_to_task = build_issue_to_task_map(tasks)
    valid_issues = set(issue_to_task.keys())
    edges = edges[
        edges["from_issue_key"].astype(str).isin(valid_issues)
        & edges["to_issue_key"].astype(str).isin(valid_issues)
    ].copy()

    edge_map = {}
    indeg = defaultdict(int)
    outdeg = defaultdict(int)

    skipped_missing = 0
    skipped_same_task = 0

    for _, row in edges.iterrows():
        u = str(row["from_issue_key"])
        v = str(row["to_issue_key"])
        tu = issue_to_task.get(u)
        tv = issue_to_task.get(v)

        if not tu or not tv:
            skipped_missing += 1
            continue
        if tu == tv:
            skipped_same_task += 1
            continue

        key = (tu, tv)
        if key not in edge_map:
            edge_map[key] = {
                "from_task_id": tu,
                "to_task_id": tv,
                "edge_count": 0,
                "link_types": set(),
                "example_from_issue": u,
                "example_to_issue": v,
            }
        edge_map[key]["edge_count"] += 1
        lt = row.get("link_type")
        if isinstance(lt, str) and lt.strip():
            edge_map[key]["link_types"].add(lt.strip())

    for (tu, tv), info in edge_map.items():
        outdeg[tu] += 1
        indeg[tv] += 1

    edges_out = []
    for info in edge_map.values():
        link_types = sorted(info["link_types"])
        edges_out.append(
            {
                "from_task_id": info["from_task_id"],
                "to_task_id": info["to_task_id"],
                "edge_count": info["edge_count"],
                "link_types": ";".join(link_types),
                "example_from_issue": info["example_from_issue"],
                "example_to_issue": info["example_to_issue"],
            }
        )

    edges_out = pd.DataFrame(edges_out)
    if not edges_out.empty:
        edges_out = edges_out.sort_values(by=["from_task_id", "to_task_id"])

    nodes_out = tasks.copy()
    nodes_out["indegree"] = nodes_out["task_id"].astype(str).map(indeg).fillna(0).astype(int)
    nodes_out["outdegree"] = nodes_out["task_id"].astype(str).map(outdeg).fillna(0).astype(int)
    nodes_out["is_root"] = (nodes_out["indegree"] == 0).astype(int)
    nodes_out["is_leaf"] = (nodes_out["outdegree"] == 0).astype(int)

    os.makedirs(os.path.dirname(out_nodes), exist_ok=True)
    os.makedirs(os.path.dirname(out_edges), exist_ok=True)

    nodes_out.to_csv(out_nodes, index=False, encoding="utf-8-sig")
    edges_out.to_csv(out_edges, index=False, encoding="utf-8-sig")

    print("Saved nodes:", out_nodes)
    print("Saved edges:", out_edges)
    print("Skipped edges (missing task):", skipped_missing)
    print("Skipped edges (same task):", skipped_same_task)


if __name__ == "__main__":
    main()
