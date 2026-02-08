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

DEFAULT_OUTPUT_ASSIGNMENT = r"projects/ZOOKEEPER/greedy_assignment.csv"
DEFAULT_OUTPUT_SCORE = r"projects/ZOOKEEPER/greedy_score.json"

TASK_ID_COL = "task_id"
TASK_TAG_COL = "Task_Tag"
PRIORITY_COL = "main_priority"
SUMMARY_COL = "representative_summary"
LEVEL_COL = "topo_level"

ASSIGNEE_CODE_COL = "assignee_code"
ASSIGNEE_SKILLS_COL = "skills"
ASSIGNEE_SCORES_COL = "skill_scores"


@dataclass
class ObjectiveWeights:
    skill_matching: float = 0.25
    workload_balance: float = 0.15
    priority_respect: float = 0.10
    skill_development: float = 0.05
    cost_optimization: float = 0.20
    makespan_score: float = 0.15
    utilization_score: float = 0.10


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
PRIORITY_DURATION_HOURS = {
    "critical": 32.0,
    "high": 16.0,
    "major": 16.0,
    "medium": 8.0,
    "minor": 8.0,
    "low": 4.0,
    "trivial": 4.0,
    "blocker": 32.0,
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
    def __init__(
        self,
        tasks_df: pd.DataFrame,
        emp_skills: Dict[str, Dict[str, int]],
        emp_costs: Dict[str, float] = None,
    ):
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
            if "topo_order" in self.tasks_df.columns:
                self.task_info[tid]["Topo_Order"] = row["topo_order"]

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

    def evaluate(self, assign: Dict[str, str]) -> Dict:
        s1 = self._skill_matching(assign)
        s2 = self._workload_balance(assign)
        s3 = self._priority_respect(assign)
        s4 = self._skill_dev(assign)
        total_cost, makespan = self._assignment_cost_makespan(assign)
        utilization = self._utilization_ratio(assign, makespan)
        primary_score = float(np.mean([s2, s3, s4])) if assign else 0.0

        return {
            "skill_matching": s1,
            "workload_balance": s2,
            "priority_respect": s3,
            "skill_development": s4,
            "primary_score": primary_score,
            "total_cost_usd": total_cost,
            "makespan_hours": makespan,
            "utilization_ratio": utilization,
        }

    def _skill_matching(self, assign):
        penalty = 0.0
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

        avg_penalty = penalty / len(assign) if assign else 0.0
        return max(0.0, 1.0 - avg_penalty)

    def _workload_balance(self, assign):
        totals = {}
        for _, emp in assign.items():
            totals.setdefault(emp, 0.0)
        for tid, emp in assign.items():
            info = self.data.get_task_info(tid)
            duration = info.get("Duration_Hours", 0)
            try:
                duration = float(duration)
            except Exception:
                duration = 0.0
            if duration <= 0:
                pr = str(info.get("Priority", "")).strip().lower()
                duration = PRIORITY_DURATION_HOURS.get(pr, 8.0)
            totals[emp] += duration
        vals = list(totals.values())
        if not vals:
            return 1.0
        mean = float(np.mean(vals))
        if mean <= 0:
            return 1.0
        std = float(np.std(vals))
        cv = std / mean
        return 1 / (1 + cv)

    def _priority_respect(self, assign):
        weights = {
            "Blocker": 1.0,
            "Critical": 0.8,
            "Major": 0.5,
            "Minor": 0.2,
            "Trivial": 0.1,
        }
        reward = 0.0
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
        return min(1.0, reward / len(assign)) if assign else 0.0

    def _skill_dev(self, assign):
        diversity = {}
        for tid, emp in assign.items():
            tag, _ = self.data.task_skills[tid][0]
            diversity.setdefault(emp, set()).add(tag)
        vals = [len(v) for v in diversity.values()]
        avg = np.mean(vals) if vals else 0.0
        return min(1.0, avg / 10)

    def _assignment_cost_makespan(self, assign: Dict[str, str]) -> Tuple[float, float]:
        total_cost = 0.0
        per_emp = {}
        for tid, emp in assign.items():
            info = self.data.get_task_info(tid)
            duration = float(info.get("Duration_Hours", 0) or 0)
            per_emp[emp] = per_emp.get(emp, 0.0) + duration
            total_cost += duration * self.data.emp_costs.get(emp, 50.0)
        makespan = max(per_emp.values()) if per_emp else 0.0
        return total_cost, makespan

    def _utilization_ratio(self, assign: Dict[str, str], makespan: float) -> float:
        if not assign or makespan <= 0:
            return 0.0
        total_hours = 0.0
        for tid in assign:
            duration = float(self.data.get_task_info(tid).get("Duration_Hours", 0) or 0)
            total_hours += duration
        total_emps = max(1, len(self.data.emp_skills))
        return total_hours / (total_emps * makespan)

    def _cost_optimization(self, assign):
        default_rate = 50.0
        total_hours = 0.0
        total_cost = 0.0
        for tid, emp in assign.items():
            info = self.data.get_task_info(tid)
            duration = float(info.get("Duration_Hours", 0) or 0)
            rate = self.data.emp_costs.get(emp, default_rate)
            total_hours += duration
            total_cost += duration * rate

        if total_hours <= 0:
            return 1.0

        avg_rate = total_cost / total_hours
        if self.data.emp_costs:
            min_rate = min(self.data.emp_costs.values())
            max_rate = max(self.data.emp_costs.values())
        else:
            min_rate = default_rate
            max_rate = default_rate

        if max_rate > min_rate:
            score = 1.0 - ((avg_rate - min_rate) / (max_rate - min_rate))
        else:
            score = 1.0
        return max(0.0, min(1.0, score))

    def _makespan_score(self, assign):
        if not assign:
            return 1.0
        per_emp = {}
        total_hours = 0.0
        for tid, emp in assign.items():
            duration = float(self.data.get_task_info(tid).get("Duration_Hours", 0) or 0)
            per_emp[emp] = per_emp.get(emp, 0.0) + duration
            total_hours += duration
        if not per_emp or total_hours <= 0:
            return 1.0
        makespan = max(per_emp.values())
        ideal = total_hours / max(1, len(per_emp))
        return max(0.0, min(1.0, ideal / makespan if makespan > 0 else 1.0))

    def _utilization_score(self, assign):
        if not assign:
            return 1.0
        per_emp = {}
        total_hours = 0.0
        for tid, emp in assign.items():
            duration = float(self.data.get_task_info(tid).get("Duration_Hours", 0) or 0)
            per_emp[emp] = per_emp.get(emp, 0.0) + duration
            total_hours += duration
        if not per_emp or total_hours <= 0:
            return 1.0
        makespan = max(per_emp.values())
        total_emps = max(1, len(self.data.emp_skills))
        return max(0.0, min(1.0, total_hours / (total_emps * makespan) if makespan > 0 else 1.0))


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
            emp_skills[emp][skill] = level

    return emp_skills


def parse_args():
    parser = argparse.ArgumentParser(
        description="Greedy assignment baseline by topo levels (fast heuristic).",
    )
    parser.add_argument("--topo", default=DEFAULT_TOPO)
    parser.add_argument("--assignees", default=DEFAULT_ASSIGNEES)
    parser.add_argument("--task-edges", default=DEFAULT_TASK_EDGES)
    parser.add_argument("--output-assignment", default=DEFAULT_OUTPUT_ASSIGNMENT)
    parser.add_argument("--output-score", default=DEFAULT_OUTPUT_SCORE)
    parser.add_argument(
        "--tie-break",
        choices=["shortest", "longest"],
        default="shortest",
        help="Within each level, sort tasks by duration.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Candidate assignees to consider per task (fast heuristic).",
    )
    parser.add_argument(
        "--load-weight",
        type=float,
        default=0.20,
        help="Weight for load balancing term in heuristic.",
    )
    parser.add_argument(
        "--pick-mode",
        choices=["least-load", "most-load", "first", "random"],
        default="random",
        help="Greedy pick strategy among eligible assignees.",
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


def greedy_assign(ctx: BatchContext, tasks_list: List[str], employees: List[str]) -> Dict[str, str]:
    raise NotImplementedError


def greedy_fast_assign(
    ctx: BatchContext,
    tasks_list: List[str],
    employees: List[str],
    top_k: int,
    load_weight: float,
    pick_mode: str,
) -> Dict[str, str]:
    assignment: Dict[str, str] = {}
    loads = {emp: 0.0 for emp in employees}

    for tid in tasks_list:
        info = ctx.get_task_info(tid)
        tag = str(info.get("Task_Tag", "")).strip()
        need = priority_to_level(info.get("Priority", ""))
        duration = float(info.get("Duration_Hours", 0) or 0)

        chosen = None
        chosen_load = None
        eligible = []
        for emp in employees:
            have = ctx.get_emp_skills(emp).get(tag, 0)
            if have < need:
                continue
            eligible.append(emp)

        if not eligible:
            raise ValueError(f"No eligible assignees for task {tid} with tag {tag} need {need}")

        if pick_mode == "first":
            chosen = eligible[0]
        elif pick_mode == "random":
            chosen = np.random.choice(eligible)
        else:
            for emp in eligible:
                load = loads[emp]
                if chosen_load is None:
                    chosen = emp
                    chosen_load = load
                    continue
                if pick_mode == "most-load":
                    if load > chosen_load:
                        chosen = emp
                        chosen_load = load
                else:
                    if load < chosen_load:
                        chosen = emp
                        chosen_load = load

        if chosen is None:
            raise ValueError(f"No eligible assignees for task {tid} with tag {tag} need {need}")

        assignment[tid] = chosen
        loads[chosen] += duration

    return assignment


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
        raise ValueError(f"Missing columns in topo file: {missing}")

    if "duration_hours" not in topo.columns:
        topo["duration_hours"] = topo.apply(
            lambda r: compute_duration_fallback(r.get("issue_count", 0), r.get(PRIORITY_COL)),
            axis=1,
        )

    topo_order_col = "topo_order" if "topo_order" in topo.columns else None

    emp_skills = parse_assignee_skills(assignees_path)
    emp_costs = load_assignee_costs(assignees_path)
    employees = sorted(emp_skills.keys())
    if not employees:
        raise ValueError("No employees found in assignees file.")

    all_rows = []
    level_solutions = {}
    current_time = 0.0

    print("\n=== STEP 1: Greedy per Topo Level ===")
    for level in sorted(topo[LEVEL_COL].dropna().unique()):
        level_tasks = topo[topo[LEVEL_COL] == level].copy()
        asc = args.tie_break == "shortest"
        level_tasks = level_tasks.sort_values("duration_hours", ascending=asc)

        tasks_list = list(level_tasks[TASK_ID_COL].astype(str))
        if not tasks_list:
            continue

        print(f"\nLevel {int(level)}: {len(tasks_list)} tasks")

        ctx = BatchContext(level_tasks, emp_skills, emp_costs)
        assignment = greedy_fast_assign(
            ctx,
            tasks_list,
            employees,
            top_k=args.top_k,
            load_weight=args.load_weight,
            pick_mode=args.pick_mode,
        )
        details = ObjectiveCalculator(ctx).evaluate(assignment)

        level_solutions[level] = {
            "assignment": assignment,
            "score_details": details,
            "total_score": details["primary_score"],
            "tasks_df": level_tasks,
        }

        print(f"  Score for level: {details['primary_score']:.4f}")
        print(
            f"  Details: skill={details['skill_matching']:.3f}, "
            f"balance={details['workload_balance']:.3f}, "
            f"priority={details['priority_respect']:.3f}, "
            f"dev={details['skill_development']:.3f}, "
            f"cost_usd={details['total_cost_usd']:.2f}, "
            f"makespan_h={details['makespan_hours']:.2f}, "
            f"util={details['utilization_ratio']:.3f}"
        )

    # STEP 2: Combine all level solutions into global assignment
    print("\n=== STEP 2: Combining Level Solutions into Global Assignment ===")
    level_times = {}
    for level in sorted(level_solutions.keys()):
        level_sol = level_solutions[level]
        level_tasks = level_sol["tasks_df"]
        best_assignment = level_sol["assignment"]

        level_duration = float(level_tasks["duration_hours"].max()) if not level_tasks.empty else 0.0
        level_start = current_time
        level_end = current_time + level_duration
        level_times[level] = {"start": level_start, "end": level_end}

        for tid, emp in best_assignment.items():
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

    default_rate = 50.0
    total_project_cost = sum(
        row.get("Duration_Hours", 0) * emp_costs.get(row.get("Assigned_To"), default_rate)
        for row in all_rows
    )
    global_details = {
        "skill_matching": 0.0,
        "workload_balance": 0.0,
        "priority_respect": 0.0,
        "skill_development": 0.0,
    }

    for level, level_sol in level_solutions.items():
        level_detail = level_sol["score_details"]
        level_task_count = len(level_sol["tasks_df"])
        weight = level_task_count / len(all_rows) if len(all_rows) > 0 else 0

        for key in (
            "skill_matching",
            "workload_balance",
            "priority_respect",
            "skill_development",
        ):
            if key in level_detail:
                global_details[key] += level_detail[key] * weight

    global_total_score = float(
        np.mean(
            [
                global_details["workload_balance"],
                global_details["priority_respect"],
                global_details["skill_development"],
            ]
        )
    )

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
        "global_skill_matching": float(global_details["skill_matching"]),
        "global_workload_balance": float(global_details["workload_balance"]),
        "global_priority_respect": float(global_details["priority_respect"]),
        "global_skill_development": float(global_details["skill_development"]),
        "global_total_score": float(global_total_score),
        "total_project_cost_usd": round(total_project_cost, 2),
        "cost_per_task_usd": round(total_project_cost / len(all_rows), 2) if all_rows else 0,
        "cost_per_hour_usd": round(total_project_cost / total_work_hours, 2) if total_work_hours > 0 else 0,
        "efficiency_score_per_1000_usd": round(global_total_score / (total_project_cost / 1000), 4) if total_project_cost > 0 else 0,
        "level_scores": level_scores,
    }

    with open(args.output_score, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== STEP 3: Global Results ===")
    print(f"Assignment saved -> {args.output_assignment}")
    print(f"Score saved -> {args.output_score}")
    print("\nGlobal Quality Scores:")
    print(f"  Skill Matching: {global_details['skill_matching']:.4f}")
    print(f"  Workload Balance: {global_details['workload_balance']:.4f}")
    print(f"  Priority Respect: {global_details['priority_respect']:.4f}")
    print(f"  Skill Development: {global_details['skill_development']:.4f}")
    print(f"  Total Score: {global_total_score:.4f}")
    print("\nGlobal Cost KPIs:")
    print(f"  Total Cost: ${total_project_cost:,.2f}")
    print(f"  Cost per Task: ${total_project_cost / len(all_rows):.2f}")
    print(f"  Cost per Hour: ${total_project_cost / total_work_hours:.2f}")
    print(f"  Utilization: {utilization:.2%}")


if __name__ == "__main__":
    main()
