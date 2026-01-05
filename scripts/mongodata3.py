from pymongo import MongoClient
from datetime import datetime
import json
import csv

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "JiraReposAnon"
COLLECTION_NAME = "Apache"

PROJECT_KEY = "ZOOKEEPER"
ISSUES_OUT = r"projects/ZOOKEEPER/issues.csv"
STATUS_EVENTS_OUT = r"projects/ZOOKEEPER/status_events.csv"
ISSUE_LINKS_OUT = r"projects/ZOOKEEPER/issue_links.csv"


def parse_jira_datetime(s: str | None) -> str | None:
    """
    JIRA datetime ví dụ: '2021-11-27T07:51:54.000+0000'
    Trả về chuỗi ISO 'YYYY-MM-DDTHH:MM:SS+ZZZZ' cho dễ xử lý sau.
    Nếu không parse được thì trả lại nguyên string.
    """
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f%z")
        return dt.isoformat()
    except Exception:
        return s


def parse_devstatus_field(raw: str | None) -> dict:
    """
    Parse customfield_12314020 (Jira DevStatus plugin).
    Trả về dict gồm các count + flag dev_has_activity.
    Nếu parse lỗi thì trả về toàn 0.
    """
    default = {
        "dev_pullrequest_count": 0,
        "dev_build_count": 0,
        "dev_review_count": 0,
        "dev_deploy_count": 0,
        "dev_repo_count": 0,
        "dev_branch_count": 0,
        "dev_has_activity": 0,
    }
    if not raw or not isinstance(raw, str):
        return default

    marker = "devSummaryJson="
    idx = raw.find(marker)
    if idx == -1:
        return default

    # Lấy đoạn sau "devSummaryJson="
    s = raw[idx + len(marker):].strip()

    # Tìm từ '{' đầu tiên đến '}' cuối cùng
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return default
    s = s[start:end + 1]

    # Nếu còn bọc trong dấu quote, bỏ đi
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    # Thử unescape các dấu \" (trường hợp copy y nguyên từ JIRA)
    s_unescaped = s.replace('\\"', '"')

    dev_json = None
    for candidate in (s_unescaped, s):
        try:
            dev_json = json.loads(candidate)
            break
        except Exception:
            continue

    if dev_json is None:
        return default

    cached = dev_json.get("cachedValue", {})
    summary = cached.get("summary", {})

    def get_count(path: list[str]) -> int:
        cur = summary
        for key in path:
            if not isinstance(cur, dict):
                return 0
            cur = cur.get(key)
        if isinstance(cur, (int, float)):
            return int(cur)
        return 0

    pr = get_count(["pullrequest", "overall", "count"])
    build = get_count(["build", "overall", "count"])
    review = get_count(["review", "overall", "count"])
    deploy = get_count(["deployment-environment", "overall", "count"])
    repo = get_count(["repository", "overall", "count"])
    branch = get_count(["branch", "overall", "count"])

    total = pr + build + review + deploy + repo + branch

    return {
        "dev_pullrequest_count": pr,
        "dev_build_count": build,
        "dev_review_count": review,
        "dev_deploy_count": deploy,
        "dev_repo_count": repo,
        "dev_branch_count": branch,
        "dev_has_activity": 1 if total > 0 else 0,
    }


def build_group_features(issue_flat: dict) -> dict:
    """
    Sinh ra các trường nhóm logic cho bài toán phân công:
    - primary_component, primary_fixversion
    - group_domain, group_release, group_worktype, group_priority
    - group_has_pr, group_dev_activity_bucket, group_key
    """
    components = issue_flat.get("components") or []
    fix_versions = issue_flat.get("fixVersions") or []
    priority_name = (issue_flat.get("priority_name") or "").lower()
    issuetype_name = (issue_flat.get("issuetype_name") or "").lower()
    summary = (issue_flat.get("summary") or "").lower()

    # --- primary component / fixVersion ---
    primary_component = components[0] if components else None
    primary_fixversion = fix_versions[0] if fix_versions else None

    # --- domain: DOC / SERVER / CLIENT / OTHER ---
    comp_lc = (primary_component or "").lower()
    domain = "OTHER"
    if any(k in comp_lc for k in ["doc", "documentation", "readme", "site", "wiki"]) or \
       any(k in summary for k in ["doc", "readme", "documentation"]):
        domain = "DOC"
    elif any(k in comp_lc for k in ["server", "quorum", "leader", "follower"]):
        domain = "SERVER"
    elif any(k in comp_lc for k in ["client", "cli", "shell"]):
        domain = "CLIENT"

    # --- release: lấy fixVersion đầu tiên hoặc UNPLANNED ---
    release = primary_fixversion or "UNPLANNED"

    # --- worktype từ issuetype ---
    if "bug" in issuetype_name:
        worktype = "BUG"
    elif "improvement" in issuetype_name or "new feature" in issuetype_name or "story" in issuetype_name:
        worktype = "FEATURE"
    elif "task" in issuetype_name:
        worktype = "TASK"
    else:
        worktype = "OTHER"

    # --- priority bucket ---
    if any(k in priority_name for k in ["blocker", "critical"]):
        prio_bucket = "HIGH"
    elif any(k in priority_name for k in ["major", "medium"]):
        prio_bucket = "MEDIUM"
    elif any(k in priority_name for k in ["minor", "trivial", "low"]):
        prio_bucket = "LOW"
    else:
        prio_bucket = "UNKNOWN"

    # --- PR / Dev activity ---
    has_pr_label = issue_flat.get("has_pr_label") or 0
    has_pr_weblink = issue_flat.get("has_pr_weblink") or 0
    dev_pr = issue_flat.get("dev_pullrequest_count") or 0
    has_pr_any = 1 if (has_pr_label or has_pr_weblink or dev_pr > 0) else 0

    dev_total = sum(
        (issue_flat.get(k) or 0)
        for k in [
            "dev_pullrequest_count",
            "dev_build_count",
            "dev_review_count",
            "dev_deploy_count",
            "dev_repo_count",
            "dev_branch_count",
        ]
    )
    if dev_total == 0:
        dev_bucket = "NONE"
    elif dev_total <= 3:
        dev_bucket = "LOW"
    elif dev_total <= 10:
        dev_bucket = "MEDIUM"
    else:
        dev_bucket = "HIGH"

    group_key = "|".join([
        domain,
        release,
        worktype,
        prio_bucket,
        f"PR{has_pr_any}",
    ])

    return {
        "primary_component": primary_component,
        "primary_fixversion": primary_fixversion,
        "group_domain": domain,
        "group_release": release,
        "group_worktype": worktype,
        "group_priority": prio_bucket,
        "group_has_pr": has_pr_any,
        "group_dev_activity_bucket": dev_bucket,
        "group_key": group_key,
    }


def flatten_issue_minimal(doc: dict) -> dict:
    """
    Nhận 1 document JIRA từ Mongo, trả về dict gọn chỉ
    giữ field cần dùng cho phân tích & tối ưu phân công.
    """
    fields = doc.get("fields", {}) or {}

    project = fields.get("project") or {}
    status = fields.get("status") or {}
    resolution = fields.get("resolution") or {}
    priority = fields.get("priority") or {}
    issuetype = fields.get("issuetype") or {}
    assignee = fields.get("assignee") or {}
    creator = fields.get("creator") or {}
    reporter = fields.get("reporter") or {}

    components = fields.get("components") or []
    fix_versions = fields.get("fixVersions") or []
    labels = fields.get("labels") or []

    issuelinks = fields.get("issuelinks") or []
    changelog = doc.get("changelog") or {}
    histories = changelog.get("histories") or []

    # --- parse DevStatus field ---
    devstatus_raw = fields.get("customfield_12314020")
    dev_metrics = parse_devstatus_field(devstatus_raw)

    # --- flag PR từ label ---
    has_pr_label = "pull-request-available" in labels

    # --- flag PR từ RemoteIssueLink trong changelog ---
    has_pr_weblink = False
    for h in histories:
        for item in h.get("items", []):
            if item.get("field") == "RemoteIssueLink":
                text = (item.get("toString") or "") + " " + (item.get("fromString") or "")
                if "Pull Request" in text or "PR-" in text:
                    has_pr_weblink = True
                    break
        if has_pr_weblink:
            break

    out = {
        # ID & key
        "mongo_id": str(doc.get("_id")),
        "jira_id": doc.get("id"),
        "issue_key": doc.get("key"),

        # Project
        "project_key": project.get("key"),
        "project_name": project.get("name"),

        # Time
        "created": parse_jira_datetime(fields.get("created")),
        "resolutiondate": parse_jira_datetime(fields.get("resolutiondate")),
        "updated": parse_jira_datetime(fields.get("updated")),

        "timespent": fields.get("timespent"),
        "aggregatetimespent": fields.get("aggregatetimespent"),
        "timeoriginalestimate": fields.get("timeoriginalestimate"),

        # Status & type
        "status_id": status.get("id"),
        "status_name": status.get("name"),
        "resolution_name": resolution.get("name") if resolution else None,
        "priority_name": priority.get("name"),
        "issuetype_name": issuetype.get("name"),

        # Classification
        "components": [c.get("name") for c in components],
        "fixVersions": [v.get("name") for v in fix_versions],
        "labels": labels,

        # People
        "assignee_key": assignee.get("key") if assignee else None,
        "assignee_name": assignee.get("displayName") if assignee else None,

        "creator_key": creator.get("key") if creator else None,
        "creator_name": creator.get("displayName") if creator else None,

        "reporter_key": reporter.get("key") if reporter else None,
        "reporter_name": reporter.get("displayName") if reporter else None,

        # Short content
        "summary": fields.get("summary"),

        # Links / DAG info
        "issuelinks": issuelinks,
        "changelog_histories": histories,

        # KPI DevStatus & flags PR
        "has_pr_label": int(has_pr_label),
        "has_pr_weblink": int(has_pr_weblink),
    }

    # Gộp thêm các metric DevStatus
    out.update(dev_metrics)

    # Gộp thêm các trường nhóm logic
    out.update(build_group_features(out))

    return out


def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]

    # Chỉ lấy issue thuộc 1 project
    query = {"fields.project.key": PROJECT_KEY}

    # Projection: chỉ field cần thiết để nhẹ data
    projection = {
        "id": 1,
        "key": 1,
        "fields.project": 1,
        "fields.created": 1,
        "fields.resolutiondate": 1,
        "fields.updated": 1,
        "fields.timespent": 1,
        "fields.aggregatetimespent": 1,
        "fields.timeoriginalestimate": 1,
        "fields.status": 1,
        "fields.resolution": 1,
        "fields.priority": 1,
        "fields.issuetype": 1,
        "fields.components": 1,
        "fields.fixVersions": 1,
        "fields.labels": 1,
        "fields.assignee": 1,
        "fields.creator": 1,
        "fields.reporter": 1,
        "fields.summary": 1,
        "fields.issuelinks": 1,
        "fields.customfield_12314020": 1,  # DevStatus
        "changelog.histories": 1,
    }

    cursor = col.find(query, projection)

    with open(ISSUES_OUT, "w", newline="", encoding="utf-8") as f_issues, \
         open(STATUS_EVENTS_OUT, "w", newline="", encoding="utf-8") as f_status, \
         open(ISSUE_LINKS_OUT, "w", newline="", encoding="utf-8") as f_links:
        # Writer cho b §œng ISSUES
        issues_writer = csv.DictWriter(
            f_issues,
            fieldnames=[
                "jira_id",
                "issue_key",
                "project_key",
                "created",
                "resolutiondate",
                "updated",
                "components",
                "issuetype_name",
                "priority_name",
                "assignee_key",
                "assignee_name",
                "creator_key",
                "summary",
            ],
        )
        issues_writer.writeheader()

        # Writer cho bảng STATUS_EVENTS (đổi status)
        status_writer = csv.DictWriter(
            f_status,
            fieldnames=[
                "issue_key",
                "event_time",
                "from_status",
                "to_status",
            ],
        )
        status_writer.writeheader()

        # NEW: Writer cho bảng ISSUE_LINKS (edges cho DAG)
        links_writer = csv.DictWriter(
            f_links,
            fieldnames=[
                "from_issue_key",
                "to_issue_key",
                "link_type",
                "raw_link_type",
                "direction",
            ],
        )
        links_writer.writeheader()

        total_issue = 0
        total_events = 0
        total_links = 0

        for raw_doc in cursor:
            doc = flatten_issue_minimal(raw_doc)

            # ---- Ghi bảng ISSUES ----
            issues_writer.writerow({
                "jira_id": doc.get("jira_id"),
                "issue_key": doc.get("issue_key"),
                "project_key": doc.get("project_key"),
                "created": doc.get("created"),
                "resolutiondate": doc.get("resolutiondate"),
                "updated": doc.get("updated"),
                "components": ";".join(doc.get("components") or []),
                "issuetype_name": doc.get("issuetype_name"),
                "priority_name": doc.get("priority_name"),
                "assignee_key": doc.get("assignee_key"),
                "assignee_name": doc.get("assignee_name"),
                "creator_key": doc.get("creator_key"),
                "summary": doc.get("summary"),
            })
            total_issue += 1

            # ---- Ghi bảng STATUS_EVENTS từ changelog ----
            histories = doc.get("changelog_histories") or []
            for h in histories:
                h_time_raw = h.get("created")
                h_time = parse_jira_datetime(h_time_raw)
                for item in h.get("items", []):
                    if item.get("field") == "status":
                        status_writer.writerow({
                            "issue_key": doc.get("issue_key"),
                            "event_time": h_time,
                            "from_status": item.get("fromString"),
                            "to_status": item.get("toString"),
                        })
                        total_events += 1

            # ---- Ghi bảng ISSUE_LINKS từ issuelinks ----
            links = doc.get("issuelinks") or []
            self_key = doc.get("issue_key")
            for link in links:
                ltype = link.get("type") or {}
                raw_name = ltype.get("name")
                inward_desc = ltype.get("inward")  # ví dụ "is blocked by"
                outward_desc = ltype.get("outward")  # ví dụ "blocks"

                inward_issue = link.get("inwardIssue")
                if inward_issue:
                    other_key = inward_issue.get("key")
                    links_writer.writerow({
                        "from_issue_key": other_key,
                        "to_issue_key": self_key,
                        "link_type": inward_desc or raw_name,
                        "raw_link_type": raw_name,
                        "direction": "inward",
                    })
                    total_links += 1

                outward_issue = link.get("outwardIssue")
                if outward_issue:
                    other_key = outward_issue.get("key")
                    links_writer.writerow({
                        "from_issue_key": self_key,
                        "to_issue_key": other_key,
                        "link_type": outward_desc or raw_name,
                        "raw_link_type": raw_name,
                        "direction": "outward",
                    })
                    total_links += 1

    client.close()
    print(f"OK. Exported {total_issue} issues of project {PROJECT_KEY} to {ISSUES_OUT}")
    print(f"OK. Exported {total_events} status events to {STATUS_EVENTS_OUT}")
    print(f"OK. Exported {total_links} issue links to {ISSUE_LINKS_OUT}")
if __name__ == "__main__":
    main()
