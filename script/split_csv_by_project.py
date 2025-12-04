import pandas as pd
import os

# 1. Đọc file CSV
input_file = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\data\issues_mapped.csv"
df = pd.read_csv(input_file)

# 2. Tạo thư mục output nếu chưa có
output_dir = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\split_data"
os.makedirs(output_dir, exist_ok=True)

# 3. Lấy danh sách các project duy nhất
projects = df['Project'].unique()

print(f"Tìm thấy {len(projects)} project:")

# 4. Tách dữ liệu theo project
for project in projects:
    # Lọc dữ liệu cho project hiện tại
    project_df = df[df['Project'] == project]
    
    # Đường dẫn file output
    output_path = os.path.join(output_dir, f"{project}_issues.csv")
    
    # Ghi file CSV
    project_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    count = len(project_df)
    print(f"  • {project}: {count} bản ghi → {output_path}")

print(f"\nHoàn tất! Đã tạo {len(projects)} file CSV")