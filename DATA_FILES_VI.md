# Hướng dẫn sử dụng bộ dữ liệu Mongo (file bị gitignore)

Tài liệu này mô tả cách sử dụng các file được trích xuất từ MongoDB và đang bị `.gitignore`. Sau khi clone repo, bạn tải bộ dữ liệu về và giải nén, sau đó đặt đúng vào thư mục tương ứng.

## 1) Danh sách file và vị trí cần đặt

Đặt các file vào đúng đường dẫn trong repo:

- `data/raw/all_issues.csv`
- `data/raw/all_issue_links.csv` (nếu dùng export tổng theo Cách B)
- `data/interim/all_issues_tagged.csv`
- `data/interim/assignee_mapping.csv`
- `data/interim/assignee_tag_count.csv`
- `data/interim/assignee_skill_profile.csv`
- `data/interim/assignee_cost_profile.csv` (tùy chọn, bổ sung chi phí/level)

Lưu ý:
- `assignee_cost_profile.csv` là tùy chọn. Nếu không có, pipeline vẫn chạy, chỉ thiếu thông tin chi phí.
- `assignee_skill_profile.csv` là bắt buộc cho bước tạo assignees.

## 2) Cách sử dụng sau khi tải về

1) Giải nén bộ dữ liệu vào thư mục gốc repo (cung cấp thư mục `data/`).
2) Kiểm tra file có đúng đường dẫn như mục (1).
3) Chạy pipeline:
   - Lần đầu (gồm Step 0): `python scripts/run_pipeline.py --project-key ZOOKEEPER --with-step0`
   - Nếu đã có dữ liệu Step 0: `python scripts/run_pipeline.py --project-key ZOOKEEPER`

## 3) Tạo lại từ MongoDB (khi không có file)

Nếu bạn không có bộ dữ liệu:

- Tạo toàn bộ dữ liệu Step 0:
  ```bash
  python scripts/00_all_projects_assignee_skills.py
  ```
- Export issue links:
  - Cách A (theo từng project):
    ```bash
    python tools/mongodata3.py
    ```
  - Cách B (export tổng, sau đó tách theo project):
    ```bash
    python tools/export_all_issue_links.py --mongo-uri mongodb://localhost:27017 --db JiraReposAnon --collection Apache --out data/raw/all_issue_links.csv
    python tools/extract_issue_links.py --input data/raw/all_issue_links.csv --project-key YOUR_PROJECT
    ```

