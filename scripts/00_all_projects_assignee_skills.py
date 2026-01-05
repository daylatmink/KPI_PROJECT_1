import os
import re
import csv
import argparse
from datetime import datetime
from typing import Optional

import pandas as pd
from pymongo import MongoClient

# Defaults (override via CLI)
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "JiraReposAnon"
COLLECTION_NAME = "Apache"
OUT_DIR = r"data"

ASSIGNEE_CODE_PATTERN = re.compile(r"^A-\d+$")
UUID_PATTERN = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
CODE_NUM_PATTERN = re.compile(r"(\d+)$")


def parse_jira_datetime(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f%z")
        return dt.isoformat()
    except Exception:
        return s


def extract_uid(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    m = UUID_PATTERN.search(value)
    if m:
        return m.group(1).strip().lower()
    v = value.strip().lower()
    return v if v else None


def extract_code_number(code) -> Optional[int]:
    if not isinstance(code, str):
        code = str(code)
    m = CODE_NUM_PATTERN.search(code)
    if not m:
        return None
    return int(m.group(1))


def infer_job_label(summary, issue_type):
    label_keywords = {
        "Build & Release Engineering": [
            "build", "release", "maven", "gradle", "pom.xml", "ci", "jenkins", "pipeline",
            "artifact", "jar", "plugin", "upgrade", "bump", "dependency", "version",
            "archiver", "plexus", "packaging", "invoker", "assembly",
        ],
        "Testing & Verification": [
            "test", "tests", "junit", "integrationtest", "flaky", "failsafe", "surefire",
            "coverage", "mock",
        ],
        "Documentation": ["doc", "docs", "documentation", "guide", "reference", "javadoc", "usage", "readme"],
        "Code Refactoring & Cleanup": ["refactor", "cleanup", "re-factor", "rename", "restructure", "tidy", "warnings"],
        "Design & API Evolution": ["api", "interface", "schema", "contract", "client", "server", "signature", "package"],
        "Performance Optimization": ["performance", "latency", "throughput", "optimiz", "speed", "slow", "faster", "memory", "cpu"],
    }

    text_raw = "" if pd.isna(summary) else str(summary)
    type_raw = "" if pd.isna(issue_type) else str(issue_type)
    text = text_raw.lower()
    itype = type_raw.lower()

    def has_any(keys):
        return any(k in text for k in keys)

    # Bug first
    if "bug" in itype or "fix" in text:
        if has_any(label_keywords["Documentation"]):
            return "Documentation"
        if has_any(label_keywords["Testing & Verification"]):
            return "Testing & Verification"
        return "Bug fixing / Maintenance"

    # Docs
    if has_any(label_keywords["Documentation"]):
        return "Documentation"

    # Build/Release (includes upgrades)
    if has_any(label_keywords["Build & Release Engineering"]):
        return "Build & Release Engineering"

    # Testing
    if has_any(label_keywords["Testing & Verification"]) or "test" in text:
        return "Testing & Verification"

    # Refactor/Cleanup
    if has_any(label_keywords["Code Refactoring & Cleanup"]):
        return "Code Refactoring & Cleanup"

    # Performance
    if has_any(label_keywords["Performance Optimization"]):
        return "Performance Optimization"

    # Design/API
    if has_any(label_keywords["Design & API Evolution"]):
        return "Design & API Evolution"

    # Issue-type based fallback
    if any(k in itype for k in ["improvement", "new feature", "wish", "brainstorming"]):
        return "Feature / Improvement Implementation"
    if itype == "task":
        return "Project / General Task"

    return "Other"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def export_all_issues(out_path, mongo_uri, db_name, collection_name):
    ensure_dir(os.path.dirname(out_path))

    client = MongoClient(mongo_uri)
    db = client[db_name]
    col = db[collection_name]

    projection = {
        "id": 1,
        "key": 1,
        "fields.project": 1,
        "fields.created": 1,
        "fields.resolutiondate": 1,
        "fields.updated": 1,
        "fields.issuetype": 1,
        "fields.priority": 1,
        "fields.assignee": 1,
        "fields.creator": 1,
        "fields.reporter": 1,
        "fields.summary": 1,
    }

    cursor = col.find({}, projection)

    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "jira_id",
                "issue_key",
                "project_key",
                "project_name",
                "created",
                "resolutiondate",
                "updated",
                "issuetype_name",
                "priority_name",
                "assignee_key",
                "assignee_name",
                "creator_key",
                "creator_name",
                "reporter_key",
                "reporter_name",
                "summary",
            ],
        )
        writer.writeheader()

        total = 0
        for doc in cursor:
            fields = doc.get("fields", {}) or {}
            project = fields.get("project") or {}
            issuetype = fields.get("issuetype") or {}
            priority = fields.get("priority") or {}
            assignee = fields.get("assignee") or {}
            creator = fields.get("creator") or {}
            reporter = fields.get("reporter") or {}

            writer.writerow({
                "jira_id": doc.get("id"),
                "issue_key": doc.get("key"),
                "project_key": project.get("key"),
                "project_name": project.get("name"),
                "created": parse_jira_datetime(fields.get("created")),
                "resolutiondate": parse_jira_datetime(fields.get("resolutiondate")),
                "updated": parse_jira_datetime(fields.get("updated")),
                "issuetype_name": issuetype.get("name"),
                "priority_name": priority.get("name"),
                "assignee_key": assignee.get("key") if assignee else None,
                "assignee_name": assignee.get("displayName") if assignee else None,
                "creator_key": creator.get("key") if creator else None,
                "creator_name": creator.get("displayName") if creator else None,
                "reporter_key": reporter.get("key") if reporter else None,
                "reporter_name": reporter.get("displayName") if reporter else None,
                "summary": fields.get("summary"),
            })
            total += 1

    client.close()
    print("Exported issues:", total)


def load_or_create_mapping(mapping_path):
    if os.path.isfile(mapping_path):
        mapping = pd.read_csv(mapping_path)
    else:
        mapping = pd.DataFrame(columns=["assignee_id", "assignee_code", "user_uid"])
    if "user_uid" not in mapping.columns:
        mapping["user_uid"] = mapping["assignee_id"]
    return mapping


def build_mapping_from_issues(issues_df, mapping_path):
    mapping = load_or_create_mapping(mapping_path)
    existing_uids = set(
        str(u).strip().lower() for u in mapping["user_uid"].dropna()
    )

    uid_cols = []
    for col in ["assignee_key", "creator_key", "reporter_key"]:
        if col in issues_df.columns:
            uid_col = f"{col}_uid"
            issues_df[uid_col] = issues_df[col].apply(extract_uid)
            uid_cols.append(uid_col)

    all_uids = pd.unique(
        pd.concat([issues_df[c] for c in uid_cols], ignore_index=True)
    )
    all_uids = [u for u in all_uids if isinstance(u, str)]
    missing_uids = [u for u in all_uids if u not in existing_uids]

    existing_nums = [
        extract_code_number(c) for c in mapping["assignee_code"]
        if extract_code_number(c) is not None
    ]
    start = (max(existing_nums) + 1) if existing_nums else 1

    new_rows = []
    current = start
    for uid in missing_uids:
        new_rows.append({
            "assignee_id": uid,
            "assignee_code": f"A-{current}",
            "user_uid": uid,
        })
        current += 1

    if new_rows:
        mapping = pd.concat([mapping, pd.DataFrame(new_rows)], ignore_index=True)

    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    print("Mapping updated:", mapping_path, "new:", len(new_rows))

    return mapping


def map_group(value, uid_to_code, code_set):
    if pd.isna(value):
        return ""
    v = str(value).strip()
    if not v:
        return ""
    if ASSIGNEE_CODE_PATTERN.match(v) and v in code_set:
        return v
    uid = extract_uid(v)
    if not uid:
        return ""
    return uid_to_code.get(uid, "")


def tag_and_build_profiles(issues_path, mapping_path, tagged_path, profile_path, tag_count_path):
    issues = pd.read_csv(issues_path)
    mapping = build_mapping_from_issues(issues, mapping_path)

    uid_to_code = dict(zip(mapping["user_uid"].astype(str), mapping["assignee_code"].astype(str)))
    code_set = set(uid_to_code.values())

    issues["assignee_group"] = issues["assignee_key"].apply(lambda v: map_group(v, uid_to_code, code_set))
    issues["creator_group"] = issues["creator_key"].apply(lambda v: map_group(v, uid_to_code, code_set))
    issues["reporter_group"] = issues["reporter_key"].apply(lambda v: map_group(v, uid_to_code, code_set))

    assignee = issues["assignee_group"].replace("", pd.NA)
    creator = issues["creator_group"].replace("", pd.NA)
    reporter = issues["reporter_group"].replace("", pd.NA)
    issues["Assignee"] = assignee.fillna(creator).fillna(reporter).fillna("")

    issues["Task_Tag"] = [
        infer_job_label(summary, issue_type)
        for summary, issue_type in zip(issues["summary"], issues["issuetype_name"])
    ]

    issues.to_csv(tagged_path, index=False, encoding="utf-8-sig")
    print("Tagged issues saved:", tagged_path)

    df = issues.copy()
    df["Assignee"] = df["Assignee"].astype(str)
    df["Task_Tag"] = df["Task_Tag"].astype(str)
    df = df[(df["Assignee"].str.strip() != "") & (df["Task_Tag"].str.strip() != "")]

    df["Task_Tag"] = df["Task_Tag"].str.split(";")
    df_exploded = df.explode("Task_Tag")
    df_exploded["Task_Tag"] = df_exploded["Task_Tag"].astype(str).str.strip()
    df_exploded = df_exploded[df_exploded["Task_Tag"] != ""]

    assignee_tag_counts = (
        df_exploded
        .groupby(["Assignee", "Task_Tag"])
        .size()
        .reset_index(name="task_count")
        .sort_values(["Assignee", "task_count"], ascending=[True, False])
    )

    total_tasks = (
        df.groupby("Assignee")
          .size()
          .reset_index(name="total_tasks")
    )

    num_skills = (
        assignee_tag_counts.groupby("Assignee")["Task_Tag"]
        .nunique()
        .reset_index(name="num_skills")
    )

    idx = assignee_tag_counts.groupby("Assignee")["task_count"].idxmax()
    main_skill = (
        assignee_tag_counts.loc[idx, ["Assignee", "Task_Tag"]]
        .rename(columns={"Task_Tag": "main_skill_tag"})
    )

    pivot = (
        assignee_tag_counts
        .pivot(index="Assignee", columns="Task_Tag", values="task_count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    assignee_skill_profile = (
        total_tasks
        .merge(num_skills, on="Assignee", how="left")
        .merge(main_skill, on="Assignee", how="left")
        .merge(pivot, on="Assignee", how="left")
    )

    ensure_dir(os.path.dirname(profile_path))
    assignee_tag_counts.to_csv(tag_count_path, index=False, encoding="utf-8-sig")
    assignee_skill_profile.to_csv(profile_path, index=False, encoding="utf-8-sig")

    print("Assignee tag counts saved:", tag_count_path)
    print("Assignee skill profile saved:", profile_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export all project issues from Mongo and build assignee skill profiles."
    )
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    parser.add_argument("--db", default=DB_NAME)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--out-dir", default=OUT_DIR, help="Base output directory (default: data).")
    parser.add_argument("--mapping-file", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.abspath(args.out_dir)
    raw_dir = os.path.join(base_dir, "raw")
    interim_dir = os.path.join(base_dir, "interim")
    ensure_dir(raw_dir)
    ensure_dir(interim_dir)

    issues_raw = os.path.join(raw_dir, "all_issues.csv")
    issues_tagged = os.path.join(interim_dir, "all_issues_tagged.csv")
    mapping_file = args.mapping_file or os.path.join(interim_dir, "assignee_mapping.csv")
    tag_count_file = os.path.join(interim_dir, "assignee_tag_count.csv")
    skill_profile_file = os.path.join(interim_dir, "assignee_skill_profile.csv")

    export_all_issues(issues_raw, args.mongo_uri, args.db, args.collection)
    tag_and_build_profiles(
        issues_raw,
        mapping_file,
        issues_tagged,
        skill_profile_file,
        tag_count_file,
    )


if __name__ == "__main__":
    main()
