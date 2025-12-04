import pandas as pd
import os

# ============================
# 0. CẤU HÌNH ĐƯỜNG DẪN (SỬA Ở ĐÂY)
# ============================

# File issues đã có cột Assignee và Task_Tag
# Ví dụ: r"D:\data\HBASE_issues_tagged.csv" hoặc "HBASE_issues_tagged.csv"
INPUT_FILE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\data\issues_mapped_tagged.csv"

# Nếu để None → tự sinh tên: <input>_employee_skill_profile.csv cùng thư mục với INPUT_FILE
# Nếu muốn đặt tên cụ thể thì sửa thành chuỗi đường dẫn, vd:
# OUTPUT_FILE = r"D:\data\employee_skill_profile_new.csv"
OUTPUT_FILE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\data\assignee_skill_profile.csv"

# ============================
# 1. ĐỌC FILE
# ============================

input_path = os.path.abspath(INPUT_FILE)

if not os.path.isfile(input_path):
    raise FileNotFoundError(f"Không tìm thấy file input: {input_path}")

df = pd.read_csv(input_path)

# Kiểm tra cột cần thiết
for col in ["Assignee", "Task_Tag"]:
    if col not in df.columns:
        raise ValueError(f"Thiếu cột bắt buộc: '{col}' trong file {input_path}")

df = df.copy()
df["Assignee"] = df["Assignee"].astype(str)
df["Task_Tag"] = df["Task_Tag"].astype(str)

# Bỏ dòng trống
df = df[(df["Assignee"].str.strip() != "") & (df["Task_Tag"].str.strip() != "")]

# ============================
# 2. TÁCH NHIỀU TAG (NẾU CÓ)
# ============================

# Nếu 1 task có nhiều tag: "Bug fixing; Testing" → tách thành nhiều dòng
df["Task_Tag"] = df["Task_Tag"].str.split(";")
df_exploded = df.explode("Task_Tag")

# Làm sạch khoảng trắng
df_exploded["Task_Tag"] = df_exploded["Task_Tag"].astype(str).str.strip()
df_exploded = df_exploded[df_exploded["Task_Tag"] != ""]

# ============================
# 3. ĐẾM SỐ TASK MỖI SKILL / ASSIGNEE
# ============================

agg = (
    df_exploded
    .groupby(["Assignee", "Task_Tag"])
    .size()
    .reset_index(name="issue_count")
)

# ============================
# 4. MAP issue_count -> skill_level (1–5)
# ============================

def score_skill(cnt: int) -> int:
    if cnt <= 2:
        return 1
    elif cnt <= 5:
        return 2
    elif cnt <= 10:
        return 3
    elif cnt <= 20:
        return 4
    else:
        return 5

agg["skill_level"] = agg["issue_count"].apply(score_skill)

# Đổi tên cột cho giống file employee_skill_profile.csv
result = agg.rename(columns={
    "Assignee": "assignee_code",
    "Task_Tag": "Skill_Tag"
})

# Sắp xếp cho dễ nhìn
result = result.sort_values(
    by=["assignee_code", "issue_count"],
    ascending=[True, False]
).reset_index(drop=True)

# ============================
# 5. LƯU FILE
# ============================

if OUTPUT_FILE is None:
    base, ext = os.path.splitext(input_path)
    out_path = base + "_employee_skill_profile.csv"
else:
    out_path = os.path.abspath(OUTPUT_FILE)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
result.to_csv(out_path, index=False, encoding="utf-8-sig")

print("Đã lưu employee_skill_profile:", out_path)
