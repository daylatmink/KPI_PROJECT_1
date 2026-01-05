import pandas as pd
import os
import re

# ============================
# 0. CẤU HÌNH ĐƯỜNG DẪN (CHỈ SỬA Ở ĐÂY)
# ============================

# Đường dẫn tới file issues_mapped.csv
# Ví dụ: r"D:\data\HBASE_issues_mapped.csv"  hoặc  "HBASE_issues_mapped.csv"
INPUT_FILE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\data\issues_mapped.csv"

# Nếu để None → tự sinh tên: <input>_tagged.csv cùng thư mục với INPUT_FILE
# Nếu muốn chỉ định cụ thể thì sửa thành chuỗi đường dẫn, vd:
# OUTPUT_FILE = r"D:\data\HBASE_issues_tagged.csv"
OUTPUT_FILE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\data\issues_mapped_tagged.csv"

# ============================
# 1. ĐỌC FILE
# ============================

input_path = os.path.abspath(INPUT_FILE)

if not os.path.isfile(input_path):
    raise FileNotFoundError(f"Không tìm thấy file input: {input_path}")

df = pd.read_csv(input_path)

# ============================
# 2. GÁN NHÃN LOẠI CÔNG VIỆC (Job_Label)
# ============================

LABEL_KEYWORDS = {
    "Build & Release Engineering": [
        "build", "release", "maven", "gradle", "pom.xml", "ci", "jenkins", "pipeline",
        "artifact", "jar", "plugin", "upgrade", "bump", "dependency", "version",
        "archiver", "plexus", "packaging", "invoker", "assembly"
    ],
    "Testing & Verification": [
        "test", "tests", "junit", "integrationtest", "flaky", "failsafe", "surefire",
        "coverage", "mock"
    ],
    "Documentation": ["doc", "docs", "documentation", "guide", "reference", "javadoc", "usage", "readme"],
    "Code Refactoring & Cleanup": ["refactor", "cleanup", "re-factor", "rename", "restructure", "tidy", "warnings"],
    "Design & API Evolution": ["api", "interface", "schema", "contract", "client", "server", "signature", "package"],
    "Performance Optimization": ["performance", "latency", "throughput", "optimiz", "speed", "slow", "faster", "memory", "cpu"],
}

def infer_job_label(summary, issue_type):
    text_raw = "" if pd.isna(summary) else str(summary)
    type_raw = "" if pd.isna(issue_type) else str(issue_type)
    text = text_raw.lower()
    itype = type_raw.lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))

    def has_any(keys):
        return any(k in text for k in keys)

    def has_token(keys):
        return any(k in tokens for k in keys)

    # Bug first
    if "bug" in itype or "fix" in text:
        if has_any(LABEL_KEYWORDS["Documentation"]):
            return "Documentation"
        if has_any(LABEL_KEYWORDS["Testing & Verification"]):
            return "Testing & Verification"
        return "Bug fixing / Maintenance"

    # Docs
    if has_any(LABEL_KEYWORDS["Documentation"]):
        return "Documentation"

    # Build/Release (includes upgrades)
    if has_any(LABEL_KEYWORDS["Build & Release Engineering"]):
        return "Build & Release Engineering"

    # Testing
    if has_any(LABEL_KEYWORDS["Testing & Verification"]) or has_token(["test", "tests"]):
        return "Testing & Verification"

    # Refactor/Cleanup
    if has_any(LABEL_KEYWORDS["Code Refactoring & Cleanup"]):
        return "Code Refactoring & Cleanup"

    # Performance
    if has_any(LABEL_KEYWORDS["Performance Optimization"]):
        return "Performance Optimization"

    # Design/API
    if has_any(LABEL_KEYWORDS["Design & API Evolution"]):
        return "Design & API Evolution"

    # Issue-type based fallback
    if any(k in itype for k in ["improvement", "new feature", "wish", "brainstorming"]):
        return "Feature / Improvement Implementation"
    if itype == "task":
        return "Project / General Task"

    return "Other"



df["Task_Tag"] = [
    infer_job_label(s, t)
    for s, t in zip(df.get("Summary", ""), df.get("IssueType", ""))
]

# ============================
# 4. LƯU LẠI
# ============================

if OUTPUT_FILE is None:
    base, ext = os.path.splitext(input_path)
    out_path = base + "_tagged" + ext
else:
    out_path = os.path.abspath(OUTPUT_FILE)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print("Saved:", out_path)
