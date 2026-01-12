import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from cost_utils import load_assignee_costs


DEFAULT_TOPO = r"projects/ZOOKEEPER/logical_topo.csv"
DEFAULT_ASSIGNEES = r"projects/ZOOKEEPER/assignees.csv"
DEFAULT_TASK_EDGES = r"projects/ZOOKEEPER/logical_dag_edges.csv"

DEFAULT_OUTPUT_ASSIGNMENT = r"projects/ZOOKEEPER/hs_assignment.csv"
DEFAULT_OUTPUT_SCORE = r"projects/ZOOKEEPER/hs_score.json"

TASK_ID_COL = "task_id"
TASK_TAG_COL = "Task_Tag"
PRIORITY_COL = "main_priority"
SUMMARY_COL = "representative_summary"
LEVEL_COL = "topo_level"

ASSIGNEE_CODE_COL = "assignee_code"
ASSIGNEE_SKILLS_COL = "skills"
ASSIGNEE_SCORES_COL = "skill_scores"
ASSIGNEE_COST_COL = "hourly_cost_usd"


@dataclass
class HSConfig:
    harmony_memory_size: int = 15
    hmcr: float = 0.85
    par: float = 0.15
    num_iterations: int = 200
    seed: int = 42


@dataclass
class ObjectiveWeights:
    skill_matching: float = 0.60
    workload_balance: float = 0.20
    priority_respect: float = 0.15
    skill_development: float = 0.05


@dataclass
class PenaltyFactors:
    skill_mismatch: float = 0.30
    skill_gap: float = 10
    priority_ignore: float = 0.25


# Duration estimation (same as topo_sort: clamp issue_count*k then apply priority multiplier)
DURATION_HOURS_PER_ISSUE = 2.0
DURATION_MIN_HOURS = 1.0
DURATION_MAX_HOURS = 40.0
PRIORITY_MULTIPLIER = {
    "Blocker": 1.5,
    "Critical": 1.3,
    "Major": 1.1,
    "Minor": 1.0,
    "Trivial": 0.8,
}


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def compute_duration_fallback(issue_count, priority) -> float:
    try:
        count = float(issue_count)
    except Exception:
        count = 0.0
    base = count * DURATION_HOURS_PER_ISSUE
    base = clamp(base, DURATION_MIN_HOURS, DURATION_MAX_HOURS)
    mult = PRIORITY_MULTIPLIER.get(str(priority).strip(), 1.0)
    return base * mult


def score_skill(cnt: int) -> int:
    if cnt <= 2:
        return 1
    if cnt <= 5:
        return 2
    if cnt <= 10:
        return 3
    if cnt <= 20:
        return 4
    return 5


def priority_to_level(p: str) -> int:
    p = str(p).strip()
    mapping = {
        "Blocker": 5,
        "Critical": 4,
        "Major": 3,
        "Minor": 2,
        "Trivial": 1,
    }
    return mapping.get(p, 2)


class BatchContext:
    def __init__(self, tasks_df: pd.DataFrame, emp_skills: Dict[str, Dict[str, int]], 
                 emp_costs: Dict[str, float] = None):
        self.tasks_df = tasks_df
        self.emp_skills = emp_skills
        self.emp_costs = emp_costs or {}
        self.task_skills: Dict[str, List[Tuple[str, int]]] = {}
        self.task_info: Dict[str, Dict] = {}
        self._parse_tasks()

    def _parse_tasks(self):
        for _, row in self.tasks_df.iterrows():
            tid = str(row[TASK_ID_COL])
            tag = str(row[TASK_TAG_COL]).strip()
            pr = priority_to_level(row[PRIORITY_COL])
            self.task_skills[tid] = [(tag, pr)]
            self.task_info[tid] = {
                "Task_ID": tid,
                "Summary": row[SUMMARY_COL],
                "Priority": row[PRIORITY_COL],
                "Task_Tag": row[TASK_TAG_COL],
                "Duration_Hours": row["duration_hours"],
                "Topo_Level": row[LEVEL_COL],
            }

    def get_tasks(self) -> List[str]:
        return list(self.task_skills.keys())

    def get_task_info(self, tid: str) -> Dict:
        return self.task_info[tid]

    def get_emp_skills(self, emp: str) -> Dict[str, int]:
        return self.emp_skills.get(emp, {})


class ObjectiveCalculator:
    def __init__(self, ctx: BatchContext):
        self.data = ctx
        self.weights = ObjectiveWeights()
        self.penalties = PenaltyFactors()

    def score(self, assign: Dict[str, str]) -> Tuple[float, Dict]:
        s1 = self._skill_matching(assign)
        s2 = self._workload_balance(assign)
        s3 = self._priority_respect(assign)
        s4 = self._skill_dev(assign)

        total = (
            s1 * self.weights.skill_matching
            + s2 * self.weights.workload_balance
            + s3 * self.weights.priority_respect
            + s4 * self.weights.skill_development
        )

        return total, {
            "skill_matching": s1,
            "workload_balance": s2,
            "priority_respect": s3,
            "skill_development": s4,
            "total": total,
        }

    def _skill_matching(self, assign):
        penalty = 0
        for tid, emp in assign.items():
            reqs = self.data.task_skills[tid]
            emp_sk = self.data.get_emp_skills(emp)
            for tag, need in reqs:
                have = emp_sk.get(tag, 0)
                if have == 0:
                    penalty += self.penalties.skill_mismatch
                else:
                    gap = max(0, need - have)
                    penalty += gap * self.penalties.skill_gap / 100.0

        avg_penalty = penalty / len(assign)
        return max(0, 1 - avg_penalty)

    def _workload_balance(self, assign):
        counts = {}
        for _, emp in assign.items():
            counts[emp] = counts.get(emp, 0) + 1
        vals = list(counts.values())
        std = np.std(vals)
        max_std = len(assign) / 2
        return 1 / (1 + std / max_std)

    def _priority_respect(self, assign):
        weights = {
            "Blocker": 1.0,
            "Critical": 0.8,
            "Major": 0.5,
            "Minor": 0.2,
            "Trivial": 0.1,
        }
        reward = 0
        for tid, emp in assign.items():
            info = self.data.get_task_info(tid)
            w = weights.get(info["Priority"], 0.5)
            emp_skill_avg = np.mean(list(self.data.get_emp_skills(emp).values()))
            if np.isnan(emp_skill_avg):
                emp_skill_avg = 1
            if w >= 0.5:
                reward += w * (emp_skill_avg / 5)
            else:
                reward += 0.3
        return min(1, reward / len(assign))

    def _skill_dev(self, assign):
        diversity = {}
        for tid, emp in assign.items():
            tag, _ = self.data.task_skills[tid][0]
            diversity.setdefault(emp, set()).add(tag)
        vals = [len(v) for v in diversity.values()]
        avg = np.mean(vals)
        return min(1, avg / 10)


class HarmonySearchBatch:
    def __init__(
        self,
        tasks: List[str],
        employees: List[str],
        calc: ObjectiveCalculator,
        seed_offset: int = 0,
        log_every: int = 50,
    ):
        self.tasks = tasks
        self.emps = employees
        self.calc = calc
        self.cfg = HSConfig()
        np.random.seed(self.cfg.seed + seed_offset)
        self.log_every = int(log_every) if log_every else 0

        self.hm = []
        self.best = None
        self.best_score = -1

    def random_assign(self):
        picks = np.random.choice(self.emps, size=len(self.tasks), replace=True)
        return {tid: emp for tid, emp in zip(self.tasks, picks)}

    def run(self):
        history = []
        for _ in range(self.cfg.harmony_memory_size):
            h = self.random_assign()
            s, _ = self.calc.score(h)
            self.hm.append((h, s))
            if s > self.best_score:
                self.best_score = s
                self.best = h

        self.hm.sort(key=lambda x: x[1], reverse=True)
        _, best_details = self.calc.score(self.best)
        best_score = self.best_score

        for i in range(1, self.cfg.num_iterations + 1):
            new = {}
            for tid in self.tasks:
                if np.random.rand() < self.cfg.hmcr:
                    h, _ = self.hm[np.random.randint(len(self.hm))]
                    val = h[tid]
                    if np.random.rand() < self.cfg.par:
                        # Pitch adjustment - pick random assignee (allow replacement)
                        val = np.random.choice(self.emps)
                else:
                    # Random choice - pick random assignee (allow replacement)
                    val = np.random.choice(self.emps)
                new[tid] = val

            s, _ = self.calc.score(new)

            if s > self.hm[-1][1]:
                self.hm[-1] = (new, s)
                self.hm.sort(key=lambda x: x[1], reverse=True)

            if s > self.best_score:
                self.best_score = s
                self.best = new
                _, best_details = self.calc.score(self.best)
                best_score = self.best_score

            if self.log_every and i % self.log_every == 0:
                print(f"  HS iter {i}/{self.cfg.num_iterations} best={self.best_score:.4f}")

            history.append(
                {
                    "iteration": i,
                    "skill_matching": best_details["skill_matching"],
                    "workload_balance": best_details["workload_balance"],
                    "priority_respect": best_details["priority_respect"],
                    "skill_development": best_details["skill_development"],
                    "total": best_details["total"],
                    "current_score": s,
                    "best_score": best_score,
                }
            )

        return self.best, self.best_score, history


def parse_assignee_skills(path: str) -> Dict[str, Dict[str, int]]:
    df = pd.read_csv(path)
    for col in [ASSIGNEE_CODE_COL, ASSIGNEE_SKILLS_COL, ASSIGNEE_SCORES_COL]:
        if col not in df.columns:
            raise ValueError(f"Assignees file missing column: {col}")

    emp_skills: Dict[str, Dict[str, int]] = {}
    for _, row in df.iterrows():
        emp = str(row[ASSIGNEE_CODE_COL]).strip()
        if not emp:
            continue
        emp_skills.setdefault(emp, {})
        skills_raw = str(row.get(ASSIGNEE_SKILLS_COL, "")).strip()
        scores_raw = str(row.get(ASSIGNEE_SCORES_COL, "")).strip()

        if skills_raw in ("", "nan", "None"):
            continue

        skills = [s.strip() for s in skills_raw.split(";") if s.strip()]
        scores = [s.strip() for s in scores_raw.split(";") if s.strip()]
        if len(scores) < len(skills):
            scores += ["0"] * (len(skills) - len(scores))

        for skill, score in zip(skills, scores):
            try:
                cnt = int(float(score))
            except Exception:
                cnt = 0
            level = score_skill(cnt)
            emp_skills[emp][skill] = max(level, emp_skills[emp].get(skill, 0))

    return emp_skills


def parse_args():
    parser = argparse.ArgumentParser(
        description="HS assignment by topo levels (one assignee per task per batch).",
    )
    parser.add_argument("--topo", default=DEFAULT_TOPO)
    parser.add_argument("--assignees", default=DEFAULT_ASSIGNEES)
    parser.add_argument(
        "--task-edges",
        default=DEFAULT_TASK_EDGES,
        help="Logical task DAG edges (from_task_id,to_task_id).",
    )
    parser.add_argument("--output-assignment", default=DEFAULT_OUTPUT_ASSIGNMENT)
    parser.add_argument("--output-score", default=DEFAULT_OUTPUT_SCORE)
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print progress every N iterations (0 to disable).",
    )
    parser.add_argument(
        "--tie-break",
        choices=["shortest", "longest"],
        default="shortest",
        help="Within each level, sort tasks by duration before batching.",
    )
    parser.add_argument(
        "--plot-dir",
        default="projects/ZOOKEEPER/hs_plots",
        help="Directory to save objective plots per batch.",
    )
    return parser.parse_args()


def load_task_edges(edges_path: str) -> dict[str, list[str]]:
    if not edges_path or not os.path.isfile(edges_path):
        return {}
    edges_df = pd.read_csv(edges_path)
    required = {"from_task_id", "to_task_id"}
    if not required.issubset(edges_df.columns):
        return {}
    preds: dict[str, list[str]] = {}
    for _, row in edges_df.iterrows():
        u = str(row["from_task_id"])
        v = str(row["to_task_id"])
        preds.setdefault(v, []).append(u)
    return preds


def schedule_tasks(rows: list[dict], preds: dict[str, list[str]]):
    if not rows:
        return {}, {}
    has_topo_order = "Topo_Order" in rows[0]

    def sort_key(r):
        if has_topo_order:
            return (int(r.get("Topo_Order", 0)), int(r.get("Topo_Level", 0)), str(r.get("Task_ID")))
        return (int(r.get("Topo_Level", 0)), str(r.get("Task_ID")))

    ordered = sorted(rows, key=sort_key)
    assignee_available: dict[str, float] = {}
    task_end: dict[str, float] = {}
    task_start: dict[str, float] = {}

    for row in ordered:
        tid = str(row["Task_ID"])
        duration = float(row.get("Duration_Hours", 0.0))
        emp = row.get("Assigned_To", "")
        dep_ends = [task_end.get(p, 0.0) for p in preds.get(tid, [])]
        deps_ready = max(dep_ends) if dep_ends else 0.0
        avail = assignee_available.get(emp, 0.0)
        start = max(deps_ready, avail)
        end = start + duration
        task_start[tid] = start
        task_end[tid] = end
        assignee_available[emp] = end

    return task_start, task_end


def main():
    args = parse_args()

    topo_path = os.path.abspath(args.topo)
    assignees_path = os.path.abspath(args.assignees)
    task_edges_path = os.path.abspath(args.task_edges)

    for p in [topo_path, assignees_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File not found: {p}")

    topo = pd.read_csv(topo_path)
    required_cols = [TASK_ID_COL, TASK_TAG_COL, PRIORITY_COL, SUMMARY_COL]
    missing = [c for c in required_cols if c not in topo.columns]
    if missing:
        raise ValueError(f"Topo file missing columns: {missing}")
    if LEVEL_COL not in topo.columns:
        for alt in ["topo_order", "topoLevel", "level", "topo_rank"]:
            if alt in topo.columns:
                topo[LEVEL_COL] = topo[alt]
                break
    if LEVEL_COL not in topo.columns:
        raise ValueError(f"Topo file missing columns: ['{LEVEL_COL}']")

    topo = topo.copy()
    if "duration_hours" not in topo.columns:
        if "issue_count" not in topo.columns:
            raise ValueError("Topo file missing duration_hours and issue_count columns.")
        topo["duration_hours"] = topo.apply(
            lambda r: compute_duration_fallback(r["issue_count"], r[PRIORITY_COL]),
            axis=1,
        )
    else:
        topo["duration_hours"] = pd.to_numeric(topo["duration_hours"], errors="coerce").fillna(0.0)

    topo_order_col = "topo_order" if "topo_order" in topo.columns else None

    if not os.path.isfile(task_edges_path):
        task_edges_path = str(Path(topo_path).with_name("logical_dag_edges.csv"))

    emp_skills = parse_assignee_skills(assignees_path)
    emp_costs = load_assignee_costs(assignees_path)
    employees = sorted(emp_skills.keys())
    if not employees:
        raise ValueError("No employees found in assignees file.")

    plot_dir = os.path.abspath(args.plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    all_rows = []
    level_solutions = {}  # Store best solution per level
    current_time = 0.0

    # STEP 1: Run HS per level (not per batch)
    print("\n=== STEP 1: Running HS per Topo Level ===")
    for level in sorted(topo[LEVEL_COL].dropna().unique()):
        level_tasks = topo[topo[LEVEL_COL] == level].copy()
        asc = args.tie_break == "shortest"
        level_tasks = level_tasks.sort_values("duration_hours", ascending=asc)

        tasks_list = list(level_tasks[TASK_ID_COL].astype(str))
        if not tasks_list:
            continue

        print(f"\nLevel {int(level)}: {len(tasks_list)} tasks")

        # Run HS on entire level (all tasks together)
        ctx = BatchContext(level_tasks, emp_skills, emp_costs)
        calc = ObjectiveCalculator(ctx)
        hs = HarmonySearchBatch(
            ctx.get_tasks(),
            employees,
            calc,
            seed_offset=int(level) * 1000,
            log_every=args.log_every,
        )

        best, best_score, history = hs.run()
        total, details = calc.score(best)
        
        level_solutions[level] = {
            "assignment": best,
            "score_details": details,
            "total_score": total,
            "tasks_df": level_tasks,
            "history": history,
        }
        
        print(f"  Best score for level: {total:.4f}")
        print(f"  Details: skill={details['skill_matching']:.3f}, balance={details['workload_balance']:.3f}, "
              f"priority={details['priority_respect']:.3f}, dev={details['skill_development']:.3f}")

        # Plot history for this level
        if history:
            try:
                import matplotlib.pyplot as plt

                hist_df = pd.DataFrame(history)
                plt.figure(figsize=(8, 4))
                for col in [
                    "skill_matching",
                    "workload_balance",
                    "priority_respect",
                    "skill_development",
                    "total",
                ]:
                    plt.plot(hist_df["iteration"], hist_df[col], label=col)

                plt.title(f"HS Objectives - Level {int(level)}")
                plt.xlabel("Iteration")
                plt.ylabel("Score")
                plt.legend(loc="best", fontsize=8)
                plt.tight_layout()
                out_name = f"hs_objectives_L{int(level)}.png"
                plt.savefig(os.path.join(plot_dir, out_name), dpi=150)
                plt.close()

                plt.figure(figsize=(8, 4))
                plt.plot(hist_df["iteration"], hist_df["current_score"], label="current_score", alpha=0.6)
                plt.plot(hist_df["iteration"], hist_df["best_score"], label="best_score", linewidth=2)
                if len(hist_df) >= 10:
                    hist_df["current_ma10"] = hist_df["current_score"].rolling(10).mean()
                    plt.plot(hist_df["iteration"], hist_df["current_ma10"], label="current_ma10", linewidth=2)
                plt.title(f"HS Scores - Level {int(level)}")
                plt.xlabel("Iteration")
                plt.ylabel("Score")
                plt.legend(loc="best", fontsize=8)
                plt.tight_layout()
                out_name = f"hs_scores_L{int(level)}.png"
                plt.savefig(os.path.join(plot_dir, out_name), dpi=150)
                plt.close()
            except Exception as exc:
                print("Warning: plot skipped:", exc)

    # STEP 2: Combine all level solutions into global assignment
    print("\n=== STEP 2: Combining Level Solutions into Global Assignment ===")
    all_rows = []
    level_times = {}  # Track time window per level
    current_time = 0.0
    
    for level in sorted(level_solutions.keys()):
        level_sol = level_solutions[level]
        level_tasks = level_sol["tasks_df"]
        best_assignment = level_sol["assignment"]
        
        # Calculate time for this level
        level_duration = float(level_tasks["duration_hours"].max()) if not level_tasks.empty else 0.0
        level_start = current_time
        level_end = current_time + level_duration
        level_times[level] = {"start": level_start, "end": level_end}
        
        # Add tasks to all_rows with assignments
        for tid, emp in best_assignment.items():
            # Find task info
            task_row = level_tasks[level_tasks[TASK_ID_COL].astype(str) == str(tid)]
            if task_row.empty:
                continue
            info_dict = task_row.iloc[0].to_dict()
            
            match = 1 if emp_skills.get(emp, {}).get(info_dict.get(TASK_TAG_COL, ""), 0) > 0 else 0
            row = {
                "Task_ID": tid,
                "Topo_Level": int(level),
                "Summary": info_dict.get(SUMMARY_COL, ""),
                "Task_Tag": info_dict.get(TASK_TAG_COL, ""),
                "Priority": info_dict.get(PRIORITY_COL, ""),
                "Duration_Hours": float(info_dict.get("duration_hours", 0)),
                "Assigned_To": emp,
                "Start_Hour": level_start,
                "End_Hour": level_end,
                "Skill_Match": match,
            }
            if topo_order_col:
                row["Topo_Order"] = int(info_dict.get(topo_order_col, 0))
            all_rows.append(row)
        
        current_time = level_end
    
    print(f"Total tasks assigned: {len(all_rows)}")

    # STEP 2.5: Compute task schedule using dependencies + assignee availability
    preds = load_task_edges(task_edges_path)
    task_start, task_end = schedule_tasks(all_rows, preds)
    if task_start and task_end:
        for row in all_rows:
            tid = str(row["Task_ID"])
            if tid in task_start and tid in task_end:
                row["Start_Hour"] = task_start[tid]
                row["End_Hour"] = task_end[tid]

    os.makedirs(os.path.dirname(args.output_assignment), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_score), exist_ok=True)

    pd.DataFrame(all_rows).to_csv(args.output_assignment, index=False, encoding="utf-8-sig")

    df_rows = pd.DataFrame(all_rows)
    total_work_hours = float(df_rows["Duration_Hours"].sum()) if not df_rows.empty else 0.0
    skill_match_rate = float(df_rows["Skill_Match"].mean()) if not df_rows.empty else 0.0
    
    makespan_hours = float(df_rows["End_Hour"].max()) if not df_rows.empty else 0.0
    utilization = (
        total_work_hours / (len(employees) * makespan_hours)
        if makespan_hours > 0 and employees
        else 0.0
    )
    
    if not df_rows.empty:
        task_durations = df_rows["Duration_Hours"].astype(float)
        avg_task_duration = float(task_durations.mean())
        p50_task_duration = float(np.percentile(task_durations, 50))
        p90_task_duration = float(np.percentile(task_durations, 90))

        # Per-level duration stats
        level_stats = (
            df_rows.groupby(["Topo_Level"])
            .agg(end_hour=("End_Hour", "max"), start_hour=("Start_Hour", "min"))
        )
        level_durations = (level_stats["end_hour"] - level_stats["start_hour"]).values
        avg_level_duration = float(np.mean(level_durations)) if len(level_durations) else 0.0
        p90_level_duration = float(np.percentile(level_durations, 90)) if len(level_durations) else 0.0
    else:
        avg_task_duration = 0.0
        p50_task_duration = 0.0
        p90_task_duration = 0.0
        avg_level_duration = 0.0
        p90_level_duration = 0.0

    total_idle_hours = (len(employees) * makespan_hours) - total_work_hours if makespan_hours > 0 else 0.0
    idle_ratio = 1 - utilization if utilization > 0 else 0.0
    
    # Calculate global cost KPIs
    total_project_cost = sum(row.get("Duration_Hours", 0) * emp_costs.get(row.get("Assigned_To"), 50.0) for row in all_rows)
    
    # Calculate global quality scores
    global_details = {
        "skill_matching": 0.0,
        "workload_balance": 0.0,
        "priority_respect": 0.0,
        "skill_development": 0.0,
        "cost_optimization": 0.0,
        "cost_efficiency": 0.0,
    }
    
    # Average scores from each level (weighted by number of tasks)
    for level, level_sol in level_solutions.items():
        level_detail = level_sol["score_details"]
        level_task_count = len(level_sol["tasks_df"])
        weight = level_task_count / len(all_rows) if len(all_rows) > 0 else 0
        
        for key in global_details.keys():
            if key in level_detail:
                global_details[key] += level_detail[key] * weight
    
    global_total_score = (
        global_details["skill_matching"] * 0.60 +
        global_details["workload_balance"] * 0.20 +
        global_details["priority_respect"] * 0.15 +
        global_details["skill_development"] * 0.05
    )
    
    # Level scores for output
    level_scores = []
    for level in sorted(level_solutions.keys()):
        level_sol = level_solutions[level]
        level_scores.append({
            "Topo_Level": int(level),
            "num_tasks": len(level_sol["tasks_df"]),
            **level_sol["score_details"],
        })
    
    summary = {
        "total_tasks": len(all_rows),
        "num_levels": len(level_solutions),
        "makespan_hours": makespan_hours,
        "total_work_hours": total_work_hours,
        "avg_task_duration_hours": avg_task_duration,
        "p50_task_duration_hours": p50_task_duration,
        "p90_task_duration_hours": p90_task_duration,
        "avg_level_duration_hours": avg_level_duration,
        "p90_level_duration_hours": p90_level_duration,
        "total_idle_hours": total_idle_hours,
        "idle_ratio": idle_ratio,
        "utilization": utilization,
        "skill_match_rate": skill_match_rate,
        # Global Quality Scores (weighted average across levels)
        "global_skill_matching": float(global_details["skill_matching"]),
        "global_workload_balance": float(global_details["workload_balance"]),
        "global_priority_respect": float(global_details["priority_respect"]),
        "global_skill_development": float(global_details["skill_development"]),
        "global_cost_optimization": float(global_details["cost_optimization"]),
        "global_cost_efficiency": float(global_details["cost_efficiency"]),
        "global_total_score": float(global_total_score),
        # Cost KPIs
        "total_project_cost_usd": round(total_project_cost, 2),
        "cost_per_task_usd": round(total_project_cost / len(all_rows), 2) if all_rows else 0,
        "cost_per_hour_usd": round(total_project_cost / total_work_hours, 2) if total_work_hours > 0 else 0,
        "efficiency_score_per_1000_usd": round(global_total_score / (total_project_cost / 1000), 4) if total_project_cost > 0 else 0,
        # Per-level breakdown
        "level_scores": level_scores,
    }
    
    with open(args.output_score, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== STEP 3: Global Results ===")
    print(f"Assignment saved -> {args.output_assignment}")
    print(f"Score saved -> {args.output_score}")
    print(f"\nGlobal Quality Scores:")
    print(f"  Skill Matching: {global_details['skill_matching']:.4f}")
    print(f"  Workload Balance: {global_details['workload_balance']:.4f}")
    print(f"  Priority Respect: {global_details['priority_respect']:.4f}")
    print(f"  Skill Development: {global_details['skill_development']:.4f}")
    print(f"  Total Score: {global_total_score:.4f}")
    print(f"\nGlobal Cost KPIs:")
    print(f"  Total Cost: ${total_project_cost:,.2f}")
    print(f"  Cost per Task: ${total_project_cost / len(all_rows):.2f}")
    print(f"  Cost per Hour: ${total_project_cost / total_work_hours:.2f}")
    print(f"  Utilization: {utilization:.2%}")


if __name__ == "__main__":
    main()
