import pandas as pd
import os

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

def infer_job_label(summary, issue_type):
    text = str(summary).lower() if pd.notna(summary) else ""
    itype = str(issue_type).lower() if pd.notna(issue_type) else ""

    # Ưu tiên TEST
    if issue_type == "Test" or "test" in text:
        return "Testing & Verification"

    # BUG
    if "bug" in itype:
        if "test" in text or "flaky" in text:
            return "Testing & Verification"
        if any(k in text for k in ["doc", "javadoc", "documentation", "reference guide"]):
            return "Documentation"
        return "Bug fixing / Maintenance"

    # Improvement / New Feature / Wish / Brainstorming
    if any(k in itype for k in ["improvement", "new feature", "wish", "brainstorming"]):
        if any(k in text for k in ["design", "api", "interface", "schema", "client", "server"]):
            return "Design & API Evolution"
        if any(k in text for k in ["performance", "latency", "throughput", "optimiz"]):
            return "Performance Optimization"
        if any(k in text for k in ["test", "junit", "integrationtest", "flaky"]):
            return "Testing & Verification"
        if any(k in text for k in ["refactor", "cleanup", "re-factor"]):
            return "Code Refactoring & Cleanup"
        if any(k in text for k in ["release", "build", "maven", "jenkins", "pom.xml", "ci"]):
            return "Build & Release Engineering"
        return "Feature / Improvement Implementation"

    # Task chung
    if itype == "task":
        if any(k in text for k in ["doc", "documentation", "javadoc", "reference guide"]):
            return "Documentation"
        if any(k in text for k in ["release", "build", "maven", "jenkins", "pom.xml", "ci"]):
            return "Build & Release Engineering"
        if any(k in text for k in ["test", "junit", "integrationtest", "flaky"]):
            return "Testing & Verification"
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
print("Đã lưu:", out_path)
