import csv
import os
import argparse
from collections import defaultdict, deque

PROJECT_KEY = "ZOOKEEPER"

ISSUES_IN = r"data/raw/all_issues.csv"
LINKS_IN = r"projects/ZOOKEEPER/issue_links.csv"

DAG_NODES_OUT = r"projects/ZOOKEEPER/issue_dag_nodes.csv"
DAG_EDGES_OUT = r"projects/ZOOKEEPER/issue_dag_edges.csv"


def load_issues(path: str, project_key: str | None):
    """
    Đọc 00_issues_2.csv thành dict:
    issues[issue_key] = {...thông tin...}
    """
    issues = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["issue_key"]
            if project_key and row.get("project_key") != project_key:
                continue
            issues[key] = row
    return issues


def is_dependency_link(row: dict) -> bool:
    """
    Chỉ coi các link có chứa 'block' là dependency thật sự.
    (Blocks / is blocked by)
    """
    lt = (row.get("link_type") or "").lower()
    raw = (row.get("raw_link_type") or "").lower()
    if "block" in lt or "block" in raw:
        return True
    return False


def load_dependency_edges(path: str, existing_issues: set[str]):
    """
    Đọc 00_issue_links.csv → edges (from_issue_key, to_issue_key, link_type, raw_link_type)
    Chỉ giữ các edge:
    - là dependency (blocks / is blocked by)
    - cả 2 đầu from/to tồn tại trong danh sách issues
    """
    edges = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not is_dependency_link(row):
                continue

            u = row["from_issue_key"]
            v = row["to_issue_key"]

            if u not in existing_issues or v not in existing_issues:
                continue

            edges.append(
                {
                    "from_issue_key": u,
                    "to_issue_key": v,
                    "link_type": row.get("link_type"),
                    "raw_link_type": row.get("raw_link_type"),
                    "direction": row.get("direction"),
                }
            )
    return edges


def build_graph(issues: dict, edges: list[dict]):
    """
    Xây graph có hướng:
    - adj[u] = list các v sao cho u → v
    Đồng thời tính indegree/outdegree cơ bản.
    """
    nodes = set(issues.keys())
    adj = defaultdict(list)
    indeg = defaultdict(int)
    outdeg = defaultdict(int)

    for e in edges:
        u = e["from_issue_key"]
        v = e["to_issue_key"]
        if u not in nodes or v not in nodes:
            continue
        adj[u].append(v)
        outdeg[u] += 1
        indeg[v] += 1

    # đảm bảo mọi node đều có key trong indeg/outdeg
    for n in nodes:
        _ = indeg[n]
        _ = outdeg[n]

    return nodes, adj, indeg, outdeg


def compute_weak_components(nodes: set[str], adj: dict[str, list[str]]):
    """
    Tính weakly connected component cho graph có hướng:
    - Xem cạnh u→v như vô hướng.
    - DFS/BFS để gán component_id.
    """
    undirected = defaultdict(set)
    for u, vs in adj.items():
        for v in vs:
            undirected[u].add(v)
            undirected[v].add(u)

    comp_id = {}
    cur_id = 0
    for n in nodes:
        if n in comp_id:
            continue
        cur_id += 1
        # BFS
        q = deque([n])
        comp_id[n] = cur_id
        while q:
            x = q.popleft()
            for y in undirected[x]:
                if y not in comp_id:
                    comp_id[y] = cur_id
                    q.append(y)

    # node lẻ (không có cạnh nào) vẫn là component riêng
    return comp_id


def export_dag_nodes(issues: dict, nodes, indeg, outdeg, comp_id, out_path: str):
    """
    Ghi file DAG_NODES_OUT với:
    - issue_key, summary, group_key, group_domain, group_worktype, ...
    - indegree, outdegree, component_id, is_root, is_leaf
    """
    fieldnames = [
        "issue_key",
        "summary",
        "group_key",
        "group_domain",
        "group_release",
        "group_worktype",
        "group_priority",
        "group_has_pr",
        "dev_has_activity",
        "indegree",
        "outdegree",
        "component_id",
        "is_root",
        "is_leaf",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in sorted(nodes):
            row = issues.get(k, {})
            indegree = indeg.get(k, 0)
            outdegree = outdeg.get(k, 0)
            writer.writerow(
                {
                    "issue_key": k,
                    "summary": row.get("summary"),
                    "group_key": row.get("group_key"),
                    "group_domain": row.get("group_domain"),
                    "group_release": row.get("group_release"),
                    "group_worktype": row.get("group_worktype"),
                    "group_priority": row.get("group_priority"),
                    "group_has_pr": row.get("group_has_pr"),
                    "dev_has_activity": row.get("dev_has_activity"),
                    "indegree": indegree,
                    "outdegree": outdegree,
                    "component_id": comp_id.get(k),
                    "is_root": 1 if indegree == 0 else 0,
                    "is_leaf": 1 if outdegree == 0 else 0,
                }
            )


def export_dag_edges(issues: dict, edges: list[dict], out_path: str):
    """
    Ghi file DAG_EDGES_OUT, enrich thêm:
    - from_summary, to_summary
    - same_group, same_component (component ở đây là 'primary_component')
    - same_release
    """
    fieldnames = [
        "from_issue_key",
        "to_issue_key",
        "link_type",
        "raw_link_type",
        "direction",
        "from_summary",
        "to_summary",
        "same_group",
        "same_component",
        "same_release",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in edges:
            u = e["from_issue_key"]
            v = e["to_issue_key"]
            iu = issues.get(u, {})
            iv = issues.get(v, {})

            same_group = 1 if iu.get("group_key") and iu.get("group_key") == iv.get("group_key") else 0
            same_component = 1 if iu.get("primary_component") and iu.get("primary_component") == iv.get("primary_component") else 0
            same_release = 1 if iu.get("primary_fixversion") and iu.get("primary_fixversion") == iv.get("primary_fixversion") else 0

            writer.writerow(
                {
                    "from_issue_key": u,
                    "to_issue_key": v,
                    "link_type": e.get("link_type"),
                    "raw_link_type": e.get("raw_link_type"),
                    "direction": e.get("direction"),
                    "from_summary": iu.get("summary"),
                    "to_summary": iv.get("summary"),
                    "same_group": same_group,
                    "same_component": same_component,
                    "same_release": same_release,
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build issue DAG for a project.",
    )
    parser.add_argument("--issues", default=ISSUES_IN)
    parser.add_argument("--links", default=LINKS_IN)
    parser.add_argument("--project-key", default=PROJECT_KEY)
    parser.add_argument("--out-nodes", default=DAG_NODES_OUT)
    parser.add_argument("--out-edges", default=DAG_EDGES_OUT)
    return parser.parse_args()


def main():
    args = parse_args()

    issues_path = os.path.abspath(args.issues)
    links_path = os.path.abspath(args.links)
    out_nodes = os.path.abspath(args.out_nodes)
    out_edges = os.path.abspath(args.out_edges)

    # 1) Load issues
    issues = load_issues(issues_path, project_key=args.project_key)
    existing_issues = set(issues.keys())
    print(f"Loaded {len(existing_issues)} issues from {issues_path}")

    # 2) Load dependency edges (blocks)
    edges = load_dependency_edges(links_path, existing_issues)
    print(f"Loaded {len(edges)} dependency edges from {links_path}")

    # 3) Build graph
    nodes, adj, indeg, outdeg = build_graph(issues, edges)
    print(f"Graph has {len(nodes)} nodes")

    # 4) Compute weak components
    comp_id = compute_weak_components(nodes, adj)
    print(f"Found {len(set(comp_id.values()))} weakly connected components")

    # 5) Export node-level DAG info
    export_dag_nodes(issues, nodes, indeg, outdeg, comp_id, out_nodes)
    print(f"Exported DAG nodes to {out_nodes}")

    # 6) Export enriched edges
    export_dag_edges(issues, edges, out_edges)
    print(f"Exported DAG edges to {out_edges}")


if __name__ == "__main__":
    main()
