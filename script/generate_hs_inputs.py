import os
import stat
import pandas as pd

# =============================================================================
# Config (update if needed)
# =============================================================================
PROJECT_KEY = "MACR"
INPUT_ISSUES = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\data\issues_mapped_tagged.csv"
OUTPUT_ISSUES = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\hs\hs_input\macr_issues.csv"
OUTPUT_SKILLS = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\hs\hs_input\macr_skills.csv"

# Priority -> numeric score
PRIORITY_WEIGHT = {
    "blocker": 5,
    "critical": 4,
    "major": 3,
    "minor": 2,
    "trivial": 1,
}


def ensure_writable(path: str):
    """Clear read-only bit if file exists."""
    if os.path.isfile(path):
        try:
            os.chmod(path, stat.S_IWRITE)
        except Exception:
            pass


def safe_write_csv(df: pd.DataFrame, path: str):
    """Write CSV, fall back to .tmp if the target is locked."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ensure_writable(path)
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path, None
    except PermissionError as e:
        alt = path + ".tmp"
        df.to_csv(alt, index=False, encoding="utf-8-sig")
        return alt, e


def priority_score(val) -> float:
    if pd.isna(val):
        return 1.0
    return float(PRIORITY_WEIGHT.get(str(val).strip().lower(), 1))


def skill_level_from_count(cnt: int) -> int:
    if cnt <= 2:
        return 1
    if cnt <= 5:
        return 2
    if cnt <= 10:
        return 3
    if cnt <= 20:
        return 4
    return 5


def main():
    if not os.path.isfile(INPUT_ISSUES):
        raise FileNotFoundError(f"Missing input file: {INPUT_ISSUES}")

    df = pd.read_csv(INPUT_ISSUES, encoding="utf-8-sig")

    # Filter by project if column exists
    if "Project" in df.columns:
        df = df[df["Project"].astype(str) == PROJECT_KEY]

    # Ensure Task_Tag exists
    if "Task_Tag" not in df.columns:
        raise ValueError("Input issues missing column Task_Tag")

    # Persist issues as-is (single tag string) to hs_input
    issues_path, issues_err = safe_write_csv(df, OUTPUT_ISSUES)

    # Choose assignee id column
    assignee_col = "Assignee_ID" if "Assignee_ID" in df.columns else "Assignee"
    if assignee_col not in df.columns:
        raise ValueError("Input issues missing Assignee/Assignee_ID column")

    # Prepare for skill aggregation: split multi-tag cells
    tags_split = df["Task_Tag"].fillna("").astype(str).str.split(";")
    df_exploded = df.copy()
    df_exploded["Task_Tag"] = tags_split
    df_exploded = df_exploded.explode("Task_Tag")
    df_exploded["Task_Tag"] = df_exploded["Task_Tag"].astype(str).str.strip()
    df_exploded = df_exploded[df_exploded["Task_Tag"] != ""]

    # Priority numeric
    df_exploded["priority_score"] = df_exploded.get("Priority", 1).apply(priority_score)

    agg = (
        df_exploded
        .groupby([assignee_col, "Task_Tag"])
        .agg(
            Issue_Count=("Task_Tag", "size"),
            Avg_Priority=("priority_score", "mean"),
        )
        .reset_index()
    )

    agg["Skill_Level"] = agg["Issue_Count"].apply(skill_level_from_count)

    # Reorder/rename to match hs_input format
    agg = agg.rename(columns={assignee_col: "Assignee_ID"})
    agg["Avg_Priority"] = agg["Avg_Priority"].round(2)
    agg = agg[["Assignee_ID", "Task_Tag", "Issue_Count", "Avg_Priority", "Skill_Level"]]

    skills_path, skills_err = safe_write_csv(agg, OUTPUT_SKILLS)

    if issues_err:
        print(f"Permission denied writing {OUTPUT_ISSUES}; wrote to {issues_path} instead ({issues_err})")
    else:
        print(f"Saved issues to {issues_path}")

    if skills_err:
        print(f"Permission denied writing {OUTPUT_SKILLS}; wrote to {skills_path} instead ({skills_err})")
    else:
        print(f"Saved skills to {skills_path}")


if __name__ == "__main__":
    main()
