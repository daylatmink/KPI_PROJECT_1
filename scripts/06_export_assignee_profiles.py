import os
import argparse
import pandas as pd
from pathlib import Path

# Default paths (override via CLI args)
ISSUES_FILE = r"data/interim/all_issues_tagged.csv"
MAPPING_FILE = r"data/interim/assignee_mapping.csv"
SKILL_PROFILE_FILE = r"data/interim/assignee_skill_profile.csv"
COST_PROFILE_FILE = r"data/interim/assignee_cost_profile.csv"
OUTPUT_FILE = r"projects/ZOOKEEPER/assignees.csv"

# Column names
ISSUE_ASSIGNEE_COL = "assignee_group"
ISSUE_PROJECT_COL = "project_key"
SKILL_ASSIGNEE_COL = "Assignee"
MAPPING_CODE_COL = "assignee_code"
MAPPING_ID_COL = "assignee_id"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export assignee profiles for a project with skill scores and hourly costs.",
    )
    parser.add_argument("--issues", default=ISSUES_FILE, help="Issues CSV for the project.")
    parser.add_argument("--mapping", default=MAPPING_FILE, help="Assignee mapping CSV.")
    parser.add_argument("--skills", default=SKILL_PROFILE_FILE, help="Assignee skill profile CSV.")
    parser.add_argument("--costs", default=COST_PROFILE_FILE, help="Assignee cost profile CSV.")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output CSV path.")
    parser.add_argument("--assignee-col", default=ISSUE_ASSIGNEE_COL, help="Assignee column in issues CSV.")
    parser.add_argument("--project-key", default="ZOOKEEPER", help="Project key filter (optional).")
    return parser.parse_args()


def build_skill_lists(row, skill_cols):
    skills = []
    scores = []
    for col in skill_cols:
        val = row.get(col)
        if pd.isna(val):
            continue
        try:
            num = float(val)
        except Exception:
            continue
        if num > 0:
            skills.append(col)
            if num.is_integer():
                scores.append(str(int(num)))
            else:
                scores.append(str(num))
    return ";".join(skills), ";".join(scores)


def main():
    args = parse_args()

    issues_path = os.path.abspath(args.issues)
    mapping_path = os.path.abspath(args.mapping)
    skills_path = os.path.abspath(args.skills)
    costs_path = os.path.abspath(args.costs)
    output_path = os.path.abspath(args.output)
    
    # Tạo thư mục output nếu chưa tồn tại
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for p in [issues_path, mapping_path, skills_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")

    issues = pd.read_csv(issues_path)
    assignee_col = args.assignee_col
    if assignee_col not in issues.columns:
        raise ValueError(f"Missing column '{assignee_col}' in {issues_path}")

    mapping = pd.read_csv(mapping_path)
    if MAPPING_CODE_COL not in mapping.columns or MAPPING_ID_COL not in mapping.columns:
        raise ValueError(f"Mapping file must include {MAPPING_CODE_COL} and {MAPPING_ID_COL}")
    code_to_id = dict(zip(mapping[MAPPING_CODE_COL], mapping[MAPPING_ID_COL]))

    if args.project_key and ISSUE_PROJECT_COL in issues.columns:
        issues = issues[issues[ISSUE_PROJECT_COL] == args.project_key]

    assignee_codes = (
        issues[assignee_col]
        .dropna()
        .astype(str)
        .map(lambda x: x.strip())
    )
    assignee_codes = sorted({c for c in assignee_codes if c})

    skills_df = pd.read_csv(skills_path)
    if SKILL_ASSIGNEE_COL not in skills_df.columns:
        raise ValueError(f"Skill profile file missing column '{SKILL_ASSIGNEE_COL}'")

    # Load cost profile if it exists
    cost_df = None
    if os.path.isfile(costs_path):
        cost_df = pd.read_csv(costs_path)
        if "Assignee" not in cost_df.columns or "hourly_cost_usd" not in cost_df.columns:
            print(f"Warning: Cost profile missing required columns. Skipping cost data.")
            cost_df = None
    else:
        print(f"Warning: Cost profile file not found at {costs_path}. Skipping cost data.")

    ignore_cols = {SKILL_ASSIGNEE_COL, "total_tasks", "num_skills", "main_skill_tag"}
    skill_cols = [c for c in skills_df.columns if c not in ignore_cols]

    skill_lookup = {}
    for _, row in skills_df.iterrows():
        code = str(row.get(SKILL_ASSIGNEE_COL, "")).strip()
        if not code:
            continue
        skills, scores = build_skill_lists(row, skill_cols)
        skill_lookup[code] = (skills, scores)

    # Build cost lookup if cost_df is available
    cost_lookup = {}
    if cost_df is not None:
        for _, row in cost_df.iterrows():
            code = str(row.get("Assignee", "")).strip()
            if not code:
                continue
            hourly_cost = row.get("hourly_cost_usd", None)
            seniority = row.get("seniority_level", None)
            if pd.notna(hourly_cost) and pd.notna(seniority):
                cost_lookup[code] = {
                    "hourly_cost_usd": float(hourly_cost),
                    "seniority_level": seniority
                }

    rows = []
    for code in assignee_codes:
        assignee_id = code_to_id.get(code, "")
        skills, scores = skill_lookup.get(code, ("", ""))
        row_data = {
            "assignee_code": code,
            "assignee_id": assignee_id,
            "skills": skills,
            "skill_scores": scores,
        }
        
        # Add cost information if available
        if code in cost_lookup:
            row_data["hourly_cost_usd"] = cost_lookup[code]["hourly_cost_usd"]
            row_data["seniority_level"] = cost_lookup[code]["seniority_level"]
        
        rows.append(row_data)

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
