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
- `data/interim/all_issues_tagged.csv` is in `.gitignore`, so new users must regenerate it via Step 0.

## 4) Quick Start (Recommended)

Run full pipeline for a project:

```bash
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
