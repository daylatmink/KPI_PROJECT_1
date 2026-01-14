# JIRA Task Assignment Pipeline (HS/IHS/GHS/MOHS)

This repo contains a pipeline to build logical tasks from JIRA data and assign
tasks to assignees using HS/IHS/GHS/MOHS. This README is written for a new
environment and a first-time user.

## 1) Requirements

- Python 3.10+ (tested with 3.10+)
- MongoDB with JIRA data (if you need to export issues/links)
- Python packages:
  - pandas
  - numpy
  - pymongo
  - matplotlib

Install packages:
```bash
pip install pandas numpy pymongo matplotlib
```

## 1.1) Get Raw Data into MongoDB (Zenodo)

If you are starting from the raw Zenodo dataset:

1) Download and extract the dataset from Zenodo.  
2) Start MongoDB:
```bash
mongod
```
3) Import a JSON collection:
```bash
mongoimport --db zenodo_dataset --collection COLLECTION_NAME --file FILE.json --jsonArray
```
If the file is JSONL:
```bash
mongoimport --db zenodo_dataset --collection COLLECTION_NAME --file FILE.jsonl
```
4) Verify the data:
```bash
mongosh
use zenodo_dataset
show collections
db.COLLECTION_NAME.findOne()
```

## 2) Repo Layout (important paths)

```
PythonProject4/
  data/
    raw/
      all_issues.csv
    interim/
      all_issues_tagged.csv
      assignee_mapping.csv
      assignee_tag_count.csv
      assignee_skill_profile.csv
      assignee_cost_profile.csv
  projects/
    ZOOKEEPER/
      issue_links.csv
      logical_tasks.csv
      logical_tasks_tagged.csv
      issue_dag_nodes.csv
      issue_dag_edges.csv
      logical_dag_nodes.csv
      logical_dag_edges.csv
      logical_topo.csv
      assignees.csv
      hs_assignment.csv
      hs_score.json
      ihs_assignment.csv
      ihs_score.json
      ghs_assignment.csv
      ghs_score.json
      mohs_assignment.csv
      mohs_score.json
      hs_plots/
      ihs_plots/
      ghs_plots/
      mohs_plots/
  scripts/
    00_all_projects_assignee_skills.py
    01_group_tasks.py
    02_tag_logical_tasks.py
    03_build_issue_dag.py
    04_build_logical_task_dag.py
    05_topo_sort_logical_tasks.py
    06_export_assignee_profiles.py
    06b_assign_cost_to_assignees.py
    07_hs_topo_assign.py
    07_ihs_topo_assign.py
    07_ghs_topo_assign.py
    07_mohs_topo_assign.py
    run_pipeline.py
  tools/               # helper scripts (not part of pipeline)
```

## 3) Data Inputs

The pipeline expects these inputs:

1) Global data (Step 0):
   - Generated from MongoDB by `scripts/00_all_projects_assignee_skills.py`
   - Outputs:
     - `data/raw/all_issues.csv`
     - `data/interim/all_issues_tagged.csv`
     - `data/interim/assignee_mapping.csv`
     - `data/interim/assignee_tag_count.csv`
     - `data/interim/assignee_skill_profile.csv`

2) Project issue links:
   - `projects/<PROJECT>/issue_links.csv`
   - You can export this from MongoDB with:
     - `tools/mongodata3.py` (edit PROJECT_KEY and output paths in that file)

Note:
- If the global data files are missing, regenerate them via Step 0.

## 4) Quick Start (Recommended)

Run full pipeline for a project:

```bash
# Menu mode (no args)
python scripts/run_pipeline.py

# First-time run (includes Step 0)
python scripts/run_pipeline.py --project-key ZOOKEEPER --with-step0

# Next runs (skip Step 0 if data already exists)
python scripts/run_pipeline.py --project-key ZOOKEEPER
```

Optional flags:
```bash
python scripts/run_pipeline.py --project-key ZOOKEEPER --skip-mohs
python scripts/run_pipeline.py --project-key ZOOKEEPER --only-assignment
python scripts/run_pipeline.py --project-key ZOOKEEPER --verbose
```

## 5) Manual Run (Step-by-step)

Step 0 (global data, run once unless data changes):
```bash
python scripts/00_all_projects_assignee_skills.py
```

Steps 1-7 (per project):
```bash
python scripts/01_group_tasks.py --project_key ZOOKEEPER
python scripts/02_tag_logical_tasks.py --project-key ZOOKEEPER
python scripts/03_build_issue_dag.py --project-key ZOOKEEPER
python scripts/04_build_logical_task_dag.py --project-key ZOOKEEPER
python scripts/05_topo_sort_logical_tasks.py --project-key ZOOKEEPER
python scripts/06_export_assignee_profiles.py --project-key ZOOKEEPER
python scripts/06b_assign_cost_to_assignees.py
python scripts/07_hs_topo_assign.py --topo projects/ZOOKEEPER/logical_topo.csv --assignees projects/ZOOKEEPER/assignees.csv
python scripts/07_ihs_topo_assign.py --topo projects/ZOOKEEPER/logical_topo.csv --assignees projects/ZOOKEEPER/assignees.csv
python scripts/07_ghs_topo_assign.py --topo projects/ZOOKEEPER/logical_topo.csv --assignees projects/ZOOKEEPER/assignees.csv
python scripts/07_mohs_topo_assign.py --topo projects/ZOOKEEPER/logical_topo.csv --assignees projects/ZOOKEEPER/assignees.csv
```

## 6) Running on a New Project

1) Export issue links for the project:
   Option A (direct from MongoDB, per project):
   - Open `tools/mongodata3.py` and set `PROJECT_KEY` + output paths
   - Run:
     ```bash
     python tools/mongodata3.py
     ```
   Option B (export once, reuse without MongoDB queries):
   ```bash
   python tools/export_all_issue_links.py --mongo-uri mongodb://localhost:27017 --db JiraReposAnon --collection Apache --out data/raw/all_issue_links.csv
   python tools/extract_issue_links.py --input data/raw/all_issue_links.csv --project-key YOUR_PROJECT
   ```

2) Run the pipeline:
   ```bash
   python scripts/run_pipeline.py --project-key YOUR_PROJECT
   ```

## 6.1) Rebuilding Global Data (when files are missing)

If `data/interim/all_issues_tagged.csv` (and other Step 0 outputs) are missing:
```bash
python scripts/00_all_projects_assignee_skills.py
```

This will query MongoDB and regenerate:
- `data/raw/all_issues.csv`
- `data/interim/all_issues_tagged.csv`
- `data/interim/assignee_mapping.csv`
- `data/interim/assignee_tag_count.csv`
- `data/interim/assignee_skill_profile.csv`

## 7) Outputs

For each project, you will get:
- `logical_tasks.csv`, `logical_tasks_tagged.csv`
- DAG files: `issue_dag_*.csv`, `logical_dag_*.csv`
- `logical_topo.csv`
- `assignees.csv`
- Assignments + scores for each algorithm:
  - `hs_assignment.csv`, `hs_score.json`
  - `ihs_assignment.csv`, `ihs_score.json`
  - `ghs_assignment.csv`, `ghs_score.json`
  - `mohs_assignment.csv`, `mohs_score.json`

## 8) Tools (Optional)

Helper scripts are in `tools/` and are not part of the pipeline:
- `tools/mongodata3.py`: export project issues/links from MongoDB
- `tools/export_all_issue_links.py`: export all issue links to one CSV
- `tools/extract_issue_links.py`: filter project links from a global links CSV/JSONL
- `tools/compare_algorithms.py`: summarize algorithm metrics
- `tools/visualize_gantt.py`: create Gantt charts (legacy, uses issue_links)
- `tools/render_gantt_from_assignment.py`: create Gantt charts from assignment (Start_Hour/End_Hour)
- `tools/visualize_mohs.py`: Pareto plots for MOHS

Example (dense label style is now default):
```bash
python tools/render_gantt_from_assignment.py --assignment projects/WODEN/ihs_assignment.csv --label-fontsize 6 --ytick-fontsize 6
```

## 9) Troubleshooting

- `ModuleNotFoundError: pandas` -> run `pip install pandas numpy pymongo matplotlib`
- `ConnectionRefused` to MongoDB -> check MongoDB is running and connection string
- Missing `issue_links.csv` -> run `tools/mongodata3.py`
- Empty output files -> verify input data exists in `data/` and `projects/<PROJECT>/`

---

# Hướng dẫn tiếng Việt

Tài liệu này mô tả quy trình tạo logical tasks từ dữ liệu JIRA và gán task cho assignee bằng HS/IHS/GHS/MOHS. Các bước dưới đây dành cho người dùng mới clone repo.


## 1) Yêu cầu

- Python 3.10+ (đã test 3.10+)
- MongoDB có dữ liệu JIRA (nếu cần export issues/links) nếu muốn import từ dataset gốc
- Các gói Python:
  - pandas
  - numpy
  - pymongo
  - matplotlib

Cài đặt gói:
```bash
pip install pandas numpy pymongo matplotlib
```

## 1.1) Nạp dữ liệu gốc vào MongoDB (Zenodo) (có thể bỏ qua bước này nếu download trực tiếp các file data mẫu)

Nếu bắt đầu từ dataset Zenodo:

1) Tải về và giải nén dataset.  
2) Chạy MongoDB:
```bash
mongod
```
3) Import JSON:
```bash
mongoimport --db zenodo_dataset --collection COLLECTION_NAME --file FILE.json --jsonArray
```
Nếu file là JSONL:
```bash
mongoimport --db zenodo_dataset --collection COLLECTION_NAME --file FILE.jsonl
```
4) Kiểm tra:
```bash
mongosh
use zenodo_dataset
show collections
db.COLLECTION_NAME.findOne()
```

## 2) Cấu trúc thư mục quan trọng

Xem mục "Repo Layout" ở phần tiếng Anh phía trên để biết vị trí `data/`, `projects/`, `scripts/`, `tools/`.

## 3) Dữ liệu đầu vào

Pipeline cần các đầu vào:

1) Dữ liệu toàn cục (Step 0):
   - Tạo từ MongoDB bởi `scripts/00_all_projects_assignee_skills.py`
   - Tạo ra:
     - `data/raw/all_issues.csv`
     - `data/interim/all_issues_tagged.csv`
     - `data/interim/assignee_mapping.csv`
     - `data/interim/assignee_tag_count.csv`
     - `data/interim/assignee_skill_profile.csv`
     - `data/interim/assignee_cost_profile.csv`

2) Issue links theo project:
   - `projects/<PROJECT>/issue_links.csv`
   - Có thể export từ MongoDB qua `tools/mongodata3.py` (sửa PROJECT_KEY và đường dẫn output)

## 4) Chạy nhanh (khuyến dùng)

Chạy toàn bộ pipeline cho một project:

```bash
# Chế độ menu (không tham số)
python scripts/run_pipeline.py

# Lần đầu (bao gồm Step 0)
python scripts/run_pipeline.py --project-key ZOOKEEPER --with-step0

# Các lần sau (bỏ Step 0 nếu đã có dữ liệu)
python scripts/run_pipeline.py --project-key ZOOKEEPER
```

Tùy chọn:
```bash
python scripts/run_pipeline.py --project-key ZOOKEEPER --skip-mohs
python scripts/run_pipeline.py --project-key ZOOKEEPER --only-assignment
python scripts/run_pipeline.py --project-key ZOOKEEPER --verbose
```

## 5) Chạy từng bước (thủ công)

Step 0 (dữ liệu toàn cục, chạy 1 lần khi cần):
```bash
python scripts/00_all_projects_assignee_skills.py
```

Steps 1-7 (mỗi project):
```bash
python scripts/01_group_tasks.py --project_key ZOOKEEPER
python scripts/02_tag_logical_tasks.py --project-key ZOOKEEPER
python scripts/03_build_issue_dag.py --project-key ZOOKEEPER
python scripts/04_build_logical_task_dag.py --project-key ZOOKEEPER
python scripts/05_topo_sort_logical_tasks.py --project-key ZOOKEEPER
python scripts/06_export_assignee_profiles.py --project-key ZOOKEEPER
python scripts/06b_assign_cost_to_assignees.py
python scripts/07_hs_topo_assign.py --topo projects/ZOOKEEPER/logical_topo.csv --assignees projects/ZOOKEEPER/assignees.csv
python scripts/07_ihs_topo_assign.py --topo projects/ZOOKEEPER/logical_topo.csv --assignees projects/ZOOKEEPER/assignees.csv
python scripts/07_ghs_topo_assign.py --topo projects/ZOOKEEPER/logical_topo.csv --assignees projects/ZOOKEEPER/assignees.csv
python scripts/07_mohs_topo_assign.py --topo projects/ZOOKEEPER/logical_topo.csv --assignees projects/ZOOKEEPER/assignees.csv
```

## 6) Chạy cho project mới

1) Export issue links:
   - Cách A (từ MongoDB, theo từng project):
     - Mở `tools/mongodata3.py`, sửa `PROJECT_KEY` và đường dẫn output
     - Chạy:
       ```bash
       python tools/mongodata3.py
       ```
   - Cách B (export 1 lần, dùng lại):
     ```bash
     python tools/export_all_issue_links.py --mongo-uri mongodb://localhost:27017 --db JiraReposAnon --collection Apache --out data/raw/all_issue_links.csv
     python tools/extract_issue_links.py --input data/raw/all_issue_links.csv --project-key YOUR_PROJECT
     ```

2) Chạy pipeline:
   ```bash
   python scripts/run_pipeline.py --project-key YOUR_PROJECT
   ```

## 6.1) Tạo lại dữ liệu toàn cục (khi thiếu file)

Nếu thiếu các file Step 0:
```bash
python scripts/00_all_projects_assignee_skills.py
```

File sẽ được tạo lại:
- `data/raw/all_issues.csv`
- `data/interim/all_issues_tagged.csv`
- `data/interim/assignee_mapping.csv`
- `data/interim/assignee_tag_count.csv`
- `data/interim/assignee_skill_profile.csv`
- `data/interim/assignee_cost_profile.csv`

## 7) Outputs

Sau khi chạy, mỗi project sẽ có:
- `logical_tasks.csv`, `logical_tasks_tagged.csv`
- Cac file DAG: `issue_dag_*.csv`, `logical_dag_*.csv`
- `logical_topo.csv`
- `assignees.csv`
- Assignment + score:
  - `hs_assignment.csv`, `hs_score.json`
  - `ihs_assignment.csv`, `ihs_score.json`
  - `ghs_assignment.csv`, `ghs_score.json`
  - `mohs_assignment.csv`, `mohs_score.json`

## 8) Công cụ (tùy chọn)

Công cụ trong `tools/` (không nằm trong pipeline chính):
- `tools/mongodata3.py`: export issues/links tu MongoDB
- `tools/export_all_issue_links.py`: export toan bo issue links
- `tools/extract_issue_links.py`: loc issue links theo project
- `tools/compare_algorithms.py`: tong hop metrics
- `tools/visualize_gantt.py`: tao Gantt (legacy)
- `tools/render_gantt_from_assignment.py`: tao Gantt tu assignment
- `tools/visualize_mohs.py`: bieu do Pareto cho MOHS

## 9) Khắc phục lỗi

- `ModuleNotFoundError: pandas` -> chạy `pip install pandas numpy pymongo matplotlib`
- Không kết nối được MongoDB -> kiểm tra MongoDB đang chạy và connection string
- Thiếu `issue_links.csv` -> chạy `tools/mongodata3.py`
- File output trống -> kiểm tra input trong `data/` và `projects/<PROJECT>/`
