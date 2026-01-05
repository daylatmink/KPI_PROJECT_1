import csv
import re
import argparse
from collections import Counter
from datetime import datetime

# M ?Tt A-t stopword ti ??ng Anh c?? b ??n ?` ?? lA?m s ??ch summary
STOPWORDS = {
    "the","and","for","with","this","that","from","into","onto","about","your",
    "you","are","our","not","can","have","has","will","shall","should","could",
    "would","there","their","them","they","been","being","was","were","than",
    "some","any","all","but","when","what","why","how","where","who","whose",
    "also","such","more","less","very","just","only","then","else","over",
    "may","might","must","does","did","doing","done","on","in","at","of","to",
    "by","or","an","a","is","it","its","as","be","we","i"
}

def normalize_summary(text: str) -> set:
    """Convert summary -> t ?-p token ?`A? l ??c stopword, ch ?_ th?? ??ng."""
    if not text:
        return set()
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if len(t) >= 3 and t not in STOPWORDS]
    return set(tokens)

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union

def parse_components(value: str):
    """
    mongodata.py xuat components dang "server;documentation;client" hoac "".
    Tra ve list cac component.
    """
    if not value:
        return []
    return [c.strip() for c in re.split(r"[;|]", value) if c.strip()]

def parse_datetime(value: str):
    """
    Tr?? ??ng created/updated trong CSV lA? string gi ?`ng:
    2021-11-06T08:09:59.000+0000
    Ta parse ph ?n yyyy-mm-ddThh:mm:ss.
    """
    if not value:
        return None
    try:
        return datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None

class LogicalTask:
    """?? ??i di ??n cho 1 logical task gom nhi ??u issue liA?n quan."""

    def __init__(self, task_id: int, first_issue: dict):
        self.task_id = task_id
        self.issue_keys = [first_issue["issue_key"]]
        self.jira_ids = [first_issue["jira_id"]]
        self.project_key = first_issue["project_key"]

        self.components = parse_components(first_issue.get("components", ""))
        self.issuetypes = [first_issue.get("issuetype_name") or ""]
        self.priorities = [first_issue.get("priority_name") or ""]
        self.statuses = [first_issue.get("status_name") or ""]
        self.summaries = [first_issue.get("summary") or ""]

        # union token c ?a t ??t c ?? issue trong task
        self.summary_tokens = normalize_summary(first_issue.get("summary") or "")

        self.created_dates = []
        dt = parse_datetime(first_issue.get("created") or "")
        if dt:
            self.created_dates.append(dt)

        self.updated_dates = []
        dt = parse_datetime(first_issue.get("updated") or "")
        if dt:
            self.updated_dates.append(dt)

    def add_issue(self, issue: dict):
        """ThA?m 1 issue vA?o logical task hi ??n t ??i."""
        self.issue_keys.append(issue["issue_key"])
        self.jira_ids.append(issue["jira_id"])

        comp = parse_components(issue.get("components", ""))
        if comp:
            self.components.extend(comp)

        self.issuetypes.append(issue.get("issuetype_name") or "")
        self.priorities.append(issue.get("priority_name") or "")
        self.statuses.append(issue.get("status_name") or "")
        self.summaries.append(issue.get("summary") or "")

        self.summary_tokens |= normalize_summary(issue.get("summary") or "")

        dt = parse_datetime(issue.get("created") or "")
        if dt:
            self.created_dates.append(dt)
        dt = parse_datetime(issue.get("updated") or "")
        if dt:
            self.updated_dates.append(dt)

    # ------ CA?c thu ?Tc tA-nh ?` ??i di ??n cho task ------

    def component_signature(self):
        """Component ph ? bi ??n nh ??t (t ?`i ?`a 3) trong task."""
        if not self.components:
            return ""
        cnt = Counter(self.components)
        return "|".join([c for c, _ in cnt.most_common(3)])

    def main_issuetype(self):
        cnt = Counter([t for t in self.issuetypes if t])
        return cnt.most_common(1)[0][0] if cnt else ""

    def main_priority(self):
        cnt = Counter([p for p in self.priorities if p])
        return cnt.most_common(1)[0][0] if cnt else ""

    def representative_summary(self):
        """Ch ??n summary dA?i nh ??t lA?m mA' t ?? ?` ??i di ??n cho task."""
        if not self.summaries:
            return ""
        return max(self.summaries, key=lambda s: len(s or ""))

    def first_created(self):
        return min(self.created_dates).isoformat() if self.created_dates else ""

    def last_updated(self):
        return max(self.updated_dates).isoformat() if self.updated_dates else ""


    # ------ Similarity ?` ?? g ?Tp issue vA?o task ------

    def similarity(self, tokens: set, component_sig: str, issuetype: str) -> float:
        """
        Score = Jaccard(summary tokens) + bonus n ??u cA1ng component/issuetype.
        """
        base = jaccard(self.summary_tokens, tokens)
        bonus = 0.0
        if component_sig and self.component_signature():
            if component_sig == self.component_signature():
                bonus += 0.15
        if issuetype and issuetype == self.main_issuetype():
            bonus += 0.05
        return base + bonus


def issue_duration_days(created_dt, resolution_dt, updated_dt):
    if created_dt is None:
        return None
    end = resolution_dt or updated_dt or created_dt
    return (end - created_dt).total_seconds() / 86400


def group_issues(
    issues_rows,
    sim_threshold: float = 0.45,
    issue_max_days=None,
):
    """
    Gom cac issue thanh logical task:
    - Duyet theo thu tu created tang dan.
    - Moi issue: tim task co similarity lon nhat.
      + Neu score >= sim_threshold -> gan vao task do.
      + Nguoc lai -> tao task moi.
    """
    def created_key(row):
        dt = parse_datetime(row.get("created") or "")
        return dt or datetime.max

    rows_sorted = sorted(issues_rows, key=created_key)

    tasks = []
    for row in rows_sorted:
        created_dt = parse_datetime(row.get("created") or "")
        resolution_dt = parse_datetime(row.get("resolutiondate") or "")
        updated_dt = parse_datetime(row.get("updated") or "")
        if issue_max_days is not None:
            dur_days = issue_duration_days(created_dt, resolution_dt, updated_dt)
            if dur_days is not None and dur_days > issue_max_days:
                continue

        tokens = normalize_summary(row.get("summary") or "")
        component_sig = "|".join(parse_components(row.get("components", ""))[:3])
        issuetype = row.get("issuetype_name") or ""

        best_task = None
        best_score = 0.0

        for t in tasks:
            score = t.similarity(tokens, component_sig, issuetype)
            if score > best_score:
                best_score = score
                best_task = t

        if best_task is None or best_score < sim_threshold:
            # Tao logical task moi
            new_task = LogicalTask(len(tasks) + 1, row)
            tasks.append(new_task)
        else:
            # Gom vao task tot nhat
            best_task.add_issue(row)

    return tasks


def read_issues_csv(path, project_key=None):
    issues = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # b ?? cA?c dA?ng rA?c n ??u thi ??u issue_key
            if not row.get("issue_key"):
                continue
            if project_key and row.get("project_key") != project_key:
                continue
            issues.append(row)
    return issues


def write_tasks(tasks, tasks_out, mapping_out):
    # 1) B ??ng t ?ng h ??p logical_task
    with open(tasks_out, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "task_id",
            "project_key",
            "component_signature",
            "main_issuetype",
            "main_priority",
            "issue_count",
            "issue_keys",
            "jira_ids",
            "representative_summary",
            "first_created",
            "last_updated",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in tasks:
            writer.writerow({
                "task_id": t.task_id,
                "project_key": t.project_key,
                "component_signature": t.component_signature(),
                "main_issuetype": t.main_issuetype(),
                "main_priority": t.main_priority(),
                "issue_count": len(t.issue_keys),
                "issue_keys": "|".join(t.issue_keys),
                "jira_ids": "|".join(t.jira_ids),
                "representative_summary": t.representative_summary(),
                "first_created": t.first_created(),
                "last_updated": t.last_updated(),
            })

    # 2) Mapping issue -> task
    with open(mapping_out, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["issue_key", "jira_id", "task_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in tasks:
            for key, jid in zip(t.issue_keys, t.jira_ids):
                writer.writerow({
                    "issue_key": key,
                    "jira_id": jid,
                    "task_id": t.task_id,
                })


def main():
    parser = argparse.ArgumentParser(
        description="Group JIRA issues (mongodata.py output) into logical tasks."
    )
    parser.add_argument(
        "--issues_csv",
        default="data/raw/all_issues.csv",
        help="CSV dau vao (issues) tu kho all-projects.",
    )
    parser.add_argument(
        "--project_key",
        default="ZOOKEEPER",
        help="Project key de loc issues (vd: ZOOKEEPER).",
    )
    parser.add_argument(
        "--tasks_out",
        default="projects/ZOOKEEPER/logical_tasks.csv",
        help="File CSV output ch ?ca thA'ng tin cA?c logical task.",
    )
    parser.add_argument(
        "--mapping_out",
        default="projects/ZOOKEEPER/issue_to_task_mapping.csv",
        help="File CSV output mapping issue -> task.",
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.45,
        help="Ng?? ??ng similarity ?` ?? g ?Tp issue vA?o task (0-1)."
    )
    parser.add_argument(
        "--issue_max_days",
        type=float,
        default=180,
        help="Loai bo issue co do dai (ngay) vuot qua nguong; <=0 de bo.",
    )
    args = parser.parse_args()

    issues = read_issues_csv(args.issues_csv, project_key=args.project_key)
    issue_max_days = args.issue_max_days if args.issue_max_days and args.issue_max_days > 0 else None
    tasks = group_issues(
        issues,
        sim_threshold=args.sim_threshold,
        issue_max_days=issue_max_days,
    )
    write_tasks(tasks, args.tasks_out, args.mapping_out)


if __name__ == "__main__":
    main()
