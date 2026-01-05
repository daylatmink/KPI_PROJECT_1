# Pipeline xử lý JIRA + gán assignee (HS/IHS/GHS/MOHS)

Repo này chứa pipeline xử lý issue JIRA và các thuật toán gán task (HS, IHS, GHS, MOHS)
theo DAG/topo để phân công assignee.

## Cấu trúc thư mục

```
configs/                 Cấu hình (tuỳ chọn)
data/
  raw/                  Dữ liệu thô (toàn bộ project)
  interim/              Dữ liệu trung gian (toàn bộ project)
  final/                Dự phòng output cuối
logs/                   Log (tuỳ chọn)
projects/
  ZOOKEEPER/            Dữ liệu riêng theo project
scripts/                Pipeline + thuật toán gán
```

## Luồng xử lý tổng quan (bước 0–7)

**Bước 0 (global export + skill):**
- Script: `scripts/00_all_projects_assignee_skills.py`
- Output:
  - `data/raw/all_issues.csv`
  - `data/interim/all_issues_tagged.csv`
  - `data/interim/assignee_mapping.csv`
  - `data/interim/assignee_tag_count.csv`
  - `data/interim/assignee_skill_profile.csv`

**Bước 1–7 (theo project):**
1) Gom issue thành logical task  
   Script: `scripts/01_group_tasks.py`  
   Output: `projects/ZOOKEEPER/logical_tasks.csv`, `projects/ZOOKEEPER/issue_to_task_mapping.csv`
2) Gán `Task_Tag` theo đa số issue trong task  
   Script: `scripts/02_tag_logical_tasks.py`  
   Output: `projects/ZOOKEEPER/logical_tasks_tagged.csv`
3) Xây DAG issue  
   Script: `scripts/03_build_issue_dag.py`  
   Input: `projects/ZOOKEEPER/issue_links.csv`  
   Output: `projects/ZOOKEEPER/issue_dag_nodes.csv`, `projects/ZOOKEEPER/issue_dag_edges.csv`
4) Xây DAG logical task  
   Script: `scripts/04_build_logical_task_dag.py`  
   Output: `projects/ZOOKEEPER/logical_dag_nodes.csv`, `projects/ZOOKEEPER/logical_dag_edges.csv`
5) Topo sort + duration  
   Script: `scripts/05_topo_sort_logical_tasks.py`  
   Output: `projects/ZOOKEEPER/logical_topo.csv` (có `topo_level` + `topo_order`)
6) Xuất assignee cho project  
   Script: `scripts/06_export_assignee_profiles.py`  
   Output: `projects/ZOOKEEPER/assignees.csv`
7) Gán task (HS family)  
   Scripts:
   - `scripts/07_hs_topo_assign.py`  -> `projects/ZOOKEEPER/hs_assignment.csv`, `hs_score.json`
   - `scripts/07_ihs_topo_assign.py` -> `projects/ZOOKEEPER/ihs_assignment.csv`, `ihs_score.json`
   - `scripts/07_ghs_topo_assign.py` -> `projects/ZOOKEEPER/ghs_assignment.csv`, `ghs_score.json`
   - `scripts/07_mohs_topo_assign.py` -> `projects/ZOOKEEPER/mohs_assignment.csv`, `mohs_score.json`

## Cách chạy

Yêu cầu:
- Python 3.10+
- Thư viện: `pandas`, `numpy`, `pymongo`, `matplotlib`
- MongoDB cấu hình trong `scripts/mongodata3.py`

Chạy bước 0 (global):
```
python scripts/00_all_projects_assignee_skills.py
```

Chạy bước 1–7 (project):
```
python scripts/01_group_tasks.py
python scripts/02_tag_logical_tasks.py
python scripts/03_build_issue_dag.py
python scripts/04_build_logical_task_dag.py
python scripts/05_topo_sort_logical_tasks.py
python scripts/06_export_assignee_profiles.py
python scripts/07_hs_topo_assign.py
```

## Tuỳ chọn khi đổi project

Để chạy project khác, bạn cần:
- Tạo thư mục: `projects/<PROJECT_KEY>/`
- Xuất links cho project đó bằng `scripts/mongodata3.py` (set `PROJECT_KEY`)
- Chạy lại bước 1–7 với output trỏ vào `projects/<PROJECT_KEY>/`

Các script đã có option để đổi project/key:
```
python scripts/01_group_tasks.py --project_key HBASE
python scripts/02_tag_logical_tasks.py --project-key HBASE
python scripts/03_build_issue_dag.py --project-key HBASE --links projects/HBASE/issue_links.csv
python scripts/06_export_assignee_profiles.py --project-key HBASE --output projects/HBASE/assignees.csv
```

Gợi ý: copy `projects/ZOOKEEPER/` thành `projects/HBASE/` để có sẵn cấu trúc file.

## Thuật toán gán task

- **HS**: phiên bản cơ bản
- **IHS**: PAR, BW thay đổi theo iteration
- **GHS**: kéo theo best harmony (không dùng BW)
- **MOHS**: đa mục tiêu, Pareto archive

## Visualize

HS/IHS/GHS plots:
- `projects/ZOOKEEPER/hs_plots/`
- `projects/ZOOKEEPER/ihs_plots/`
- `projects/ZOOKEEPER/ghs_plots/`

MOHS plots:
```
python scripts/visualize_mohs.py
```
Output:
- `projects/ZOOKEEPER/mohs_plots/mohs_pareto_scatter.png`
- `projects/ZOOKEEPER/mohs_plots/mohs_parallel_coordinates.png`

## Ghi chú

- `scripts/mongodata3.py` xuất `projects/ZOOKEEPER/issue_links.csv`.
- Bước 7 dùng `topo_level` để gom batch theo DAG.
