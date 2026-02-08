# JIRA Task Assignment Pipeline (HS/IHS/GHS/MOHS)

Simple usage guide: install once, prepare data, run only the needed steps.

## 1) First-time setup

Requirements:
- Python 3.10+
- MongoDB only if you need to export issues/links

Install packages (first time only):
```bash
pip install pandas numpy pymongo matplotlib
```

## 2) Data you need (minimal)

Global data (Step 0, run once unless data changes):
- Creates:
  - `data/raw/all_issues.csv`
  - `data/interim/all_issues_tagged.csv`
  - `data/interim/assignee_mapping.csv`
  - `data/interim/assignee_tag_count.csv`
  - `data/interim/assignee_skill_profile.csv`
  - `data/interim/assignee_cost_profile.csv`

Project data:
- `projects/<PROJECT>/issue_links.csv`
- `projects/<PROJECT>/issues.csv` (optional but useful)

## 3) Create project inputs

Issue links per project:
```bash
# Export all links once (global file)
python tools/export_all_issue_links.py --mongo-uri mongodb://localhost:27017 --db JiraReposAnon --collection Apache --out data/raw/all_issue_links.csv
# Filter per project
python tools/extract_issue_links.py --input data/raw/all_issue_links.csv --project-key YOUR_PROJECT
```

Issues per project from all_issues.csv:
```bash
python tools/extract_raw_project.py YOUR_PROJECT
```

## 4) Run the pipeline (minimal commands)

First time (includes Step 0):
```bash
python scripts/run_pipeline.py --project-key YOUR_PROJECT --with-step0
```

Next runs (skip Step 0):
```bash
python scripts/run_pipeline.py --project-key YOUR_PROJECT
```

Only assignment step (Step 7):
```bash
python scripts/run_pipeline.py --project-key YOUR_PROJECT --only-assignment
```

Optional HS/IHS/GHS objectives:
```bash
python scripts/run_pipeline.py --project-key YOUR_PROJECT --only-assignment \
  --hs-objective primary --ihs-objective primary --ghs-objective primary
```

## 5) Outputs

For each project:
- `projects/<PROJECT>/logical_topo.csv`
- `projects/<PROJECT>/assignees.csv`
- `projects/<PROJECT>/hs_assignment.csv`, `hs_score.json`
- `projects/<PROJECT>/ihs_assignment.csv`, `ihs_score.json`
- `projects/<PROJECT>/ghs_assignment.csv`, `ghs_score.json`
- `projects/<PROJECT>/mohs_assignment.csv`, `mohs_score.json`

## 6) Optional tools

```bash
python tools/compare_algorithms.py --project YOUR_PROJECT
python tools/compare_mohs.py --project YOUR_PROJECT
python tools/render_gantt_from_assignment.py --assignment projects/YOUR_PROJECT/hs_assignment.csv
```

## 7) Troubleshooting

- Missing packages: `pip install pandas numpy pymongo matplotlib`
- Missing `issue_links.csv`: run export + extract links
- Missing global data: run Step 0

---

# Huong dan ngan gon (VI, ASCII)

## Cai dat lan dau
```bash
pip install pandas numpy pymongo matplotlib
```

## Tao du lieu can thiet
```bash
# Step 0
python scripts/00_all_projects_assignee_skills.py

# Issue links theo project
python tools/export_all_issue_links.py --mongo-uri mongodb://localhost:27017 --db JiraReposAnon --collection Apache --out data/raw/all_issue_links.csv
python tools/extract_issue_links.py --input data/raw/all_issue_links.csv --project-key YOUR_PROJECT

# Issues theo project (tu all_issues.csv)
python tools/extract_raw_project.py YOUR_PROJECT
```

## Chay pipeline
```bash
python scripts/run_pipeline.py --project-key YOUR_PROJECT --with-step0
python scripts/run_pipeline.py --project-key YOUR_PROJECT
python scripts/run_pipeline.py --project-key YOUR_PROJECT --only-assignment
```
