from pymongo import MongoClient
import json
import csv
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "JiraReposAnon"
COLLECTION_NAME = "Apache"

PROJECT_KEY = "ZOOKEEPER"
ISSUES_OUT = rf"{PROJECT_KEY}_issues.csv"
STATUS_EVENTS_OUT = rf"{PROJECT_KEY}_status_events.csv"
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


def flatten_issue_minimal(doc: dict) -> dict:
    """
    Nhận 1 document JIRA từ Mongo, trả về dict gọn chỉ giữ field cần dùng.
    (Không dùng lại file JSONL nữa, đọc trực tiếp từ Mongo.)
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
    }

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
        "changelog.histories": 1,
    }

    cursor = col.find(query, projection)

    # Mở file output
    with open(ISSUES_OUT, "w", newline="", encoding="utf-8") as f_issues, \
         open(STATUS_EVENTS_OUT, "w", newline="", encoding="utf-8") as f_status:

        # Writer cho bảng ISSUES
        issues_writer = csv.DictWriter(
            f_issues,
            fieldnames=[
                "mongo_id",
                "jira_id",
                "issue_key",
                "project_key",
                "project_name",
                "created",
                "resolutiondate",
                "updated",
                "timespent",
                "aggregatetimespent",
                "timeoriginalestimate",
                "status_id",
                "status_name",
                "resolution_name",
                "priority_name",
                "issuetype_name",
                "assignee_key",
                "assignee_name",
                "creator_key",
                "creator_name",
                "reporter_key",
                "reporter_name",
                "components",
                "fixVersions",
                "labels",
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

        total_issue = 0
        total_events = 0

        for raw_doc in cursor:
            doc = flatten_issue_minimal(raw_doc)

            # ---- Ghi bảng ISSUES ----
            issues_writer.writerow({
                "mongo_id": doc.get("mongo_id"),
                "jira_id": doc.get("jira_id"),
                "issue_key": doc.get("issue_key"),
                "project_key": doc.get("project_key"),
                "project_name": doc.get("project_name"),
                "created": doc.get("created"),
                "resolutiondate": doc.get("resolutiondate"),
                "updated": doc.get("updated"),
                "timespent": doc.get("timespent"),
                "aggregatetimespent": doc.get("aggregatetimespent"),
                "timeoriginalestimate": doc.get("timeoriginalestimate"),
                "status_id": doc.get("status_id"),
                "status_name": doc.get("status_name"),
                "resolution_name": doc.get("resolution_name"),
                "priority_name": doc.get("priority_name"),
                "issuetype_name": doc.get("issuetype_name"),
                "assignee_key": doc.get("assignee_key"),
                "assignee_name": doc.get("assignee_name"),
                "creator_key": doc.get("creator_key"),
                "creator_name": doc.get("creator_name"),
                "reporter_key": doc.get("reporter_key"),
                "reporter_name": doc.get("reporter_name"),
                "components": ";".join(doc.get("components") or []),
                "fixVersions": ";".join(doc.get("fixVersions") or []),
                "labels": ";".join(doc.get("labels") or []),
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

    client.close()
    print(f"✅ Exported {total_issue} issues of project {PROJECT_KEY} to {ISSUES_OUT}")
    print(f"✅ Exported {total_events} status events to {STATUS_EVENTS_OUT}")


if __name__ == "__main__":
    main()