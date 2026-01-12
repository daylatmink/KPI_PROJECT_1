import os
import argparse
from datetime import datetime
from heapq import heappush, heappop
from collections import defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_NODES = r"projects/ZOOKEEPER/logical_dag_nodes.csv"
DEFAULT_EDGES = r"projects/ZOOKEEPER/logical_dag_edges.csv"
DEFAULT_OUTPUT = r"projects/ZOOKEEPER/logical_topo.csv"

# Duration estimation (method 5: clamp issue_count*k then apply priority multiplier)
DURATION_HOURS_PER_ISSUE = 2.0
DURATION_MIN_HOURS = 1.0
DURATION_MAX_HOURS = 40.0
PRIORITY_MULTIPLIER = {
    "Blocker": 1.5,
    "Critical": 1.3,
    "Major": 1.1,
    "Minor": 1.0,
    "Trivial": 0.8,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Topological sort for logical tasks with duration-aware tie-break.",
    )
    parser.add_argument("--nodes", default=DEFAULT_NODES)
    parser.add_argument("--edges", default=DEFAULT_EDGES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tie-break",
        choices=["shortest", "longest"],
        default="shortest",
        help="When multiple tasks are available, pick shortest or longest duration first.",
    )
    return parser.parse_args()


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def compute_duration_hours(issue_count, priority) -> float:
    try:
        count = float(issue_count)
    except Exception:
        count = 0.0

    base = count * DURATION_HOURS_PER_ISSUE
    base = clamp(base, DURATION_MIN_HOURS, DURATION_MAX_HOURS)

    mult = PRIORITY_MULTIPLIER.get(str(priority).strip(), 1.0)
    return base * mult


def main():
    args = parse_args()

    nodes_path = os.path.abspath(args.nodes)
    edges_path = os.path.abspath(args.edges)
    output_path = os.path.abspath(args.output)
    
    # Tạo thư mục output nếu chưa tồn tại
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for p in [nodes_path, edges_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    required_node_cols = ["task_id", "issue_count", "main_priority"]
    missing = [c for c in required_node_cols if c not in nodes.columns]
    if missing:
        raise ValueError(f"Nodes file missing columns: {missing}")

    required_edge_cols = ["from_task_id", "to_task_id"]
    missing = [c for c in required_edge_cols if c not in edges.columns]
    if missing:
        raise ValueError(f"Edges file missing columns: {missing}")

    nodes = nodes.copy()
    nodes["task_id"] = nodes["task_id"].astype(str)
    nodes["duration_hours"] = nodes.apply(
        lambda r: compute_duration_hours(r["issue_count"], r["main_priority"]),
        axis=1,
    )

    adj = defaultdict(list)
    indeg = defaultdict(int)
    for _, row in edges.iterrows():
        u = str(row["from_task_id"])
        v = str(row["to_task_id"])
        adj[u].append(v)
        indeg[v] += 1

    # ensure all nodes are present in indeg
    for tid in nodes["task_id"]:
        _ = indeg[tid]

    # priority queue by duration with stable tie-break on task_id
    heap = []
    for _, row in nodes.iterrows():
        tid = row["task_id"]
        if indeg[tid] == 0:
            dur = float(row["duration_hours"])
            key = dur if args.tie_break == "shortest" else -dur
            heappush(heap, (key, tid))

    order = []
    topo_level = {}
    next_level = 1
    while heap:
            # process one "batch" of ready tasks to mark topo_level
        batch = []
        while heap:
            batch.append(heappop(heap)[1])

        for tid in batch:
            order.append(tid)
            topo_level[tid] = next_level
            for v in adj.get(tid, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    dur = float(nodes.loc[nodes["task_id"] == v, "duration_hours"].iloc[0])
                    key = dur if args.tie_break == "shortest" else -dur
                    heappush(heap, (key, v))

        next_level += 1
    if len(order) != len(nodes):
        remaining = [tid for tid in nodes["task_id"] if tid not in set(order)]
        print("Warning: graph has cycles or disconnected nodes. Remaining:", len(remaining))

    nodes["topo_level"] = nodes["task_id"].map(topo_level).fillna(0).astype(int)
    nodes["topo_order"] = nodes["task_id"].map({tid: i + 1 for i, tid in enumerate(order)})
    out = nodes.sort_values("topo_order")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
