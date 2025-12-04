import pandas as pd
import re
import os

# Sử dụng đường dẫn đầy đủ
csv_file = r'C:\Users\ADMIN\PycharmProjects\PythonProject4\data\apache_issues_flat.csv'
output_dir = os.path.dirname(csv_file)

# Kiểm tra file tồn tại
if not os.path.exists(csv_file):
    print(f"❌ Lỗi: File không tồn tại: {csv_file}")
    exit()

# Đọc file
df = pd.read_csv(csv_file)

# Sử dụng cột 'Assignee' (chữ hoa)
col = 'Assignee'

# Xử lý giá trị rỗng và NaN
df[col] = df[col].fillna('UNASSIGNED').astype(str)

# Trích xuất ID từ định dạng <<|author_displayName|ID|>>
def extract_assignee_id(value):
    match = re.search(r'\|([a-f0-9\-]+)\|', value)
    if match:
        return match.group(1)
    return value if value != 'UNASSIGNED' else 'UNASSIGNED'

df['assignee_id'] = df[col].apply(extract_assignee_id)

unique = list(dict.fromkeys(df['assignee_id']))
mapping = {uid: f"A-{i+1}" for i, uid in enumerate(unique)}

df['assignee_code'] = df['assignee_id'].map(mapping)

# ====== PHẦN BẠN MUỐN SỬA ======
# Dùng mã thay cho cột Assignee gốc
df[col] = df['assignee_code']

# Xóa cột phụ
df = df.drop(columns=['assignee_id', 'assignee_code'])
# =================================

# mapping & agg vẫn giữ nguyên
mapping_df = pd.DataFrame(list(mapping.items()),
                          columns=['assignee_id', 'assignee_code'])
mapping_df.to_csv('assignee_mapping.csv', index=False, encoding='utf-8')

agg = (df.groupby(col)
         .size()
         .reset_index(name='issue_count')
         .sort_values('issue_count', ascending=False))
agg.to_csv('assignee_agg_counts.csv', index=False, encoding='utf-8')

df.to_csv('issues_mapped.csv', index=False, encoding='utf-8')


print(f"✓ Đã tạo 3 file:")
print(f"  1. assignee_mapping.csv - Bảng mapping ID -> Mã")
print(f"  2. assignee_agg_counts.csv - Thống kê số issue theo assignee")
print(f"  3. HBASE_issues_mapped.csv - File CSV đã thêm cột mã")