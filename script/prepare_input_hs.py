import pandas as pd
import os

# ============================
# 0. C?u h?nh c? th? s?a d?
# ============================

# File issues ?? c? c?t Assignee (A-1, A-2, ...) v? Task_Tag
INPUT_FILE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\data\issues_mapped_tagged.csv"

# N?u None -> t? sinh: <input>_employee_skill_profile.csv
OUTPUT_FILE = None

# ============================
# 1. ??c file
# ============================

input_path = os.path.abspath(INPUT_FILE)
if not os.path.isfile(input_path):
    raise FileNotFoundError(f"Kh?ng t?m th?y file input: {input_path}")

df = pd.read_csv(input_path)

for col in ["Assignee", "Task_Tag"]:
    if col not in df.columns:
        raise ValueError(f"Thi?u c?t b?t bu?c: '{col}' trong file {input_path}")

df = df.copy()
df = df.dropna(subset=["Assignee", "Task_Tag"])
df["Assignee"] = df["Assignee"].astype(str)
df["Task_Tag"] = df["Task_Tag"].astype(str)

df = df[(df["Assignee"].str.strip() != "") & (df["Task_Tag"].str.strip() != "")]

# ============================
# 2. T?ch nhi?u tag (n?u c?)
# ============================

df["Task_Tag"] = df["Task_Tag"].str.split(";")
df_exploded = df.explode("Task_Tag")

df_exploded["Task_Tag"] = df_exploded["Task_Tag"].astype(str).str.strip()
df_exploded = df_exploded[~df_exploded["Task_Tag"].isin(["", "nan", "None"])]

# ============================
# 3. T?nh ?i?m (c? tr?ng s? theo Priority) cho t?ng SKILL / ASSIGNEE
# ============================

# Tr?ng s? ?? kh? d?a tr?n Priority (c? th? ?i?u ch?nh)
PRIORITY_WEIGHT = {
    "blocker": 5,
    "critical": 4,
    "major": 3,
    "minor": 2,
    "trivial": 1,
}

def priority_score(priority) -> int:
    if pd.isna(priority):
        return 1
    return PRIORITY_WEIGHT.get(str(priority).strip().lower(), 1)

# N?u thi?u c?t Priority s? m?c ??nh 1
if "Priority" in df_exploded.columns:
    df_exploded["priority_weight"] = df_exploded["Priority"].apply(priority_score)
else:
    df_exploded["priority_weight"] = 1

agg = (
    df_exploded
    .groupby(["Assignee", "Task_Tag"])
    .agg(
        issue_count=("Task_Tag", "size"),
        weighted_score=("priority_weight", "sum"),
    )
    .reset_index()
)

# ============================
# 4. MAP weighted_score -> skill_level
# ============================

def score_skill(score: int) -> int:
    if score <= 2:
        return 1
    elif score <= 5:
        return 2
    elif score <= 12:
        return 3
    elif score <= 25:
        return 4
    else:
        return 5

agg["skill_level"] = agg["weighted_score"].apply(score_skill)

# ??i t?n c?t cho ??ng format profile
result = agg.rename(columns={
    "Assignee": "assignee_code",
    "Task_Tag": "Skill_Tag"
})

result = result.sort_values(
    by=["assignee_code", "issue_count"],
    ascending=[True, False]
).reset_index(drop=True)

# ============================
# 5. L?u file
# ============================

if OUTPUT_FILE is None:
    base, ext = os.path.splitext(input_path)
    out_path = base + "_employee_skill_profile.csv"
else:
    out_path = os.path.abspath(OUTPUT_FILE)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
result.to_csv(out_path, index=False, encoding="utf-8-sig")
print("Saved employee_skill_profile:", out_path)
