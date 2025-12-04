import pandas as pd
import os

# ============================
# 0. CẤU HÌNH ĐƯỜNG DẪN (SỬA Ở ĐÂY)
# ============================

# File issues đã gán Task_Tag
# Ví dụ: r"D:\data\HBASE_issues_tagged.csv" hoặc "HBASE_issues_tagged.csv"
INPUT_FILE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\data\issues_mapped_tagged.csv"

# Nếu để None: tự sinh tên từ INPUT_FILE
ASSIGNEE_TAG_COUNTS_FILE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\data\assignee_tag_count.csv"      # file dạng dài
ASSIGNEE_SKILL_PROFILE_FILE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\data\assignee_skill_profile.csv"   # file dạng rộng


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

# Bỏ các dòng thiếu Assignee hoặc Task_Tag
df = df.copy()
df["Assignee"] = df["Assignee"].astype(str)
df["Task_Tag"] = df["Task_Tag"].astype(str)

df = df[(df["Assignee"].str.strip() != "") & (df["Task_Tag"].str.strip() != "")]

# ============================
# 2. TÁCH NHIỀU TAG (NẾU CÓ)
# ============================

# Nếu 1 task có nhiều tag, dạng "Bug fixing / Maintenance; Testing & Verification"
# → split ra thành nhiều dòng
df["Task_Tag"] = df["Task_Tag"].str.split(";")
df_exploded = df.explode("Task_Tag")

# Làm sạch khoảng trắng
df_exploded["Task_Tag"] = df_exploded["Task_Tag"].astype(str).str.strip()
df_exploded = df_exploded[df_exploded["Task_Tag"] != ""]

# ============================
# 3. BẢNG DẠNG DÀI: assignee_tag_counts
# ============================

assignee_tag_counts = (
    df_exploded
    .groupby(["Assignee", "Task_Tag"])
    .size()
    .reset_index(name="task_count")
    .sort_values(["Assignee", "task_count"], ascending=[True, False])
)

# ============================
# 4. BẢNG DẠNG RỘNG: assignee_skill_profile
# ============================

# Tổng số task (trước khi explode) cho mỗi assignee
total_tasks = (
    df.groupby("Assignee")
      .size()
      .reset_index(name="total_tasks")
)

# Số tag khác nhau mỗi assignee
num_skills = (
    assignee_tag_counts.groupby("Assignee")["Task_Tag"]
    .nunique()
    .reset_index(name="num_skills")
)

# Tag xuất hiện nhiều nhất mỗi assignee
idx = assignee_tag_counts.groupby("Assignee")["task_count"].idxmax()
main_skill = (
    assignee_tag_counts.loc[idx, ["Assignee", "Task_Tag"]]
    .rename(columns={"Task_Tag": "main_skill_tag"})
)

# Pivot: mỗi tag là một cột (số lượng task thuộc tag đó)
pivot = (
    assignee_tag_counts
    .pivot(index="Assignee", columns="Task_Tag", values="task_count")
    .fillna(0)
    .astype(int)
)

pivot = pivot.reset_index()

# Gộp các info lại thành skill profile
assignee_skill_profile = (
    total_tasks
    .merge(num_skills, on="Assignee", how="left")
    .merge(main_skill, on="Assignee", how="left")
    .merge(pivot, on="Assignee", how="left")
)

# ============================
# 5. LƯU FILE
# ============================

base, ext = os.path.splitext(input_path)

if ASSIGNEE_TAG_COUNTS_FILE is None:
    tag_counts_path = base + "_assignee_tag_counts" + ext
else:
    tag_counts_path = os.path.abspath(ASSIGNEE_TAG_COUNTS_FILE)

if ASSIGNEE_SKILL_PROFILE_FILE is None:
    skill_profile_path = base + "_assignee_skill_profile" + ext
else:
    skill_profile_path = os.path.abspath(ASSIGNEE_SKILL_PROFILE_FILE)

os.makedirs(os.path.dirname(tag_counts_path), exist_ok=True)
os.makedirs(os.path.dirname(skill_profile_path), exist_ok=True)

assignee_tag_counts.to_csv(tag_counts_path, index=False, encoding="utf-8-sig")
assignee_skill_profile.to_csv(skill_profile_path, index=False, encoding="utf-8-sig")

print("Đã lưu bảng tag-count:", tag_counts_path)
print("Đã lưu bảng skill profile:", skill_profile_path)
