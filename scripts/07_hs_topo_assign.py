import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_TOPO = r"projects/ZOOKEEPER/logical_topo.csv"
DEFAULT_ASSIGNEES = r"projects/ZOOKEEPER/assignees.csv"

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
    def __init__(self, tasks_df: pd.DataFrame, emp_skills: Dict[str, Dict[str, int]]):
        self.tasks_df = tasks_df
        self.emp_skills = emp_skills
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
        picks = np.random.choice(self.emps, size=len(self.tasks), replace=False)
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
            used = set()
            for tid in self.tasks:
                if np.random.rand() < self.cfg.hmcr:
                    h, _ = self.hm[np.random.randint(len(self.hm))]
                    val = h[tid]
                    if val in used or np.random.rand() < self.cfg.par:
                        candidates = [e for e in self.emps if e not in used]
                        val = np.random.choice(candidates)
                else:
                    candidates = [e for e in self.emps if e not in used]
                    val = np.random.choice(candidates)
                new[tid] = val
                used.add(val)

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


def main():
    args = parse_args()

    topo_path = os.path.abspath(args.topo)
    assignees_path = os.path.abspath(args.assignees)

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

    emp_skills = parse_assignee_skills(assignees_path)
    employees = sorted(emp_skills.keys())
    if not employees:
        raise ValueError("No employees found in assignees file.")

    plot_dir = os.path.abspath(args.plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    all_rows = []
    score_rows = []
    current_time = 0.0

    for level in sorted(topo[LEVEL_COL].dropna().unique()):
        level_tasks = topo[topo[LEVEL_COL] == level].copy()
        asc = args.tie_break == "shortest"
        level_tasks = level_tasks.sort_values("duration_hours", ascending=asc)

        tasks_list = list(level_tasks[TASK_ID_COL].astype(str))
        if not tasks_list:
            continue

        batch_size = len(employees)
        level_start_time = current_time
        for i in range(0, len(tasks_list), batch_size):
            batch_ids = tasks_list[i : i + batch_size]
            batch_df = level_tasks[level_tasks[TASK_ID_COL].astype(str).isin(batch_ids)]

            ctx = BatchContext(batch_df, emp_skills)
            calc = ObjectiveCalculator(ctx)
            hs = HarmonySearchBatch(
                ctx.get_tasks(),
                employees,
                calc,
                seed_offset=int(level) * 1000 + i,
                log_every=args.log_every,
            )

            best, _, history = hs.run()
            total, details = calc.score(best)

            wave_duration = float(batch_df["duration_hours"].max()) if not batch_df.empty else 0.0
            wave_start = level_start_time
            wave_end = level_start_time + wave_duration

            for tid, emp in best.items():
                info = ctx.get_task_info(tid)
                match = 1 if emp_skills.get(emp, {}).get(info["Task_Tag"], 0) > 0 else 0
                all_rows.append(
                    {
                        "Task_ID": tid,
                        "Topo_Level": info["Topo_Level"],
                        "Batch_Index": (i // batch_size) + 1,
                        "Summary": info["Summary"],
                        "Task_Tag": info["Task_Tag"],
                        "Priority": info["Priority"],
                        "Duration_Hours": info["Duration_Hours"],
                        "Assigned_To": emp,
                        "Start_Hour": wave_start,
                        "End_Hour": wave_end,
                        "Skill_Match": match,
                    }
                )

            score_rows.append(
                {
                    "Topo_Level": int(level),
                    "Batch_Index": (i // batch_size) + 1,
                    **details,
                }
            )
            level_start_time = wave_end

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

                    plt.title(f"HS Objectives - Level {int(level)} Batch {(i // batch_size) + 1}")
                    plt.xlabel("Iteration")
                    plt.ylabel("Score")
                    plt.legend(loc="best", fontsize=8)
                    plt.tight_layout()
                    out_name = f"hs_objectives_L{int(level)}_B{(i // batch_size) + 1}.png"
                    plt.savefig(os.path.join(plot_dir, out_name), dpi=150)
                    plt.close()

                    plt.figure(figsize=(8, 4))
                    plt.plot(hist_df["iteration"], hist_df["current_score"], label="current_score", alpha=0.6)
                    plt.plot(hist_df["iteration"], hist_df["best_score"], label="best_score", linewidth=2)
                    if len(hist_df) >= 10:
                        hist_df["current_ma10"] = hist_df["current_score"].rolling(10).mean()
                        plt.plot(hist_df["iteration"], hist_df["current_ma10"], label="current_ma10", linewidth=2)
                    plt.title(f"HS Scores - Level {int(level)} Batch {(i // batch_size) + 1}")
                    plt.xlabel("Iteration")
                    plt.ylabel("Score")
                    plt.legend(loc="best", fontsize=8)
                    plt.tight_layout()
                    out_name = f"hs_scores_L{int(level)}_B{(i // batch_size) + 1}.png"
                    plt.savefig(os.path.join(plot_dir, out_name), dpi=150)
                    plt.close()
                except Exception as exc:
                    print("Warning: plot skipped:", exc)

        current_time = level_start_time

    os.makedirs(os.path.dirname(args.output_assignment), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_score), exist_ok=True)

    pd.DataFrame(all_rows).to_csv(args.output_assignment, index=False, encoding="utf-8-sig")

    df_rows = pd.DataFrame(all_rows)
    total_work_hours = float(df_rows["Duration_Hours"].sum()) if not df_rows.empty else 0.0
    skill_match_rate = float(df_rows["Skill_Match"].mean()) if not df_rows.empty else 0.0
    utilization = (
        total_work_hours / (len(employees) * current_time)
        if current_time > 0 and employees
        else 0.0
    )
    if not df_rows.empty:
        task_durations = df_rows["Duration_Hours"].astype(float)
        avg_task_duration = float(task_durations.mean())
        p50_task_duration = float(np.percentile(task_durations, 50))
        p90_task_duration = float(np.percentile(task_durations, 90))

        batch_stats = (
            df_rows.groupby(["Topo_Level", "Batch_Index"])
            .agg(end_hour=("End_Hour", "max"), start_hour=("Start_Hour", "min"))
        )
        batch_durations = (batch_stats["end_hour"] - batch_stats["start_hour"]).values
        avg_batch_duration = float(np.mean(batch_durations)) if len(batch_durations) else 0.0
        p90_batch_duration = float(np.percentile(batch_durations, 90)) if len(batch_durations) else 0.0
    else:
        avg_task_duration = 0.0
        p50_task_duration = 0.0
        p90_task_duration = 0.0
        avg_batch_duration = 0.0
        p90_batch_duration = 0.0

    total_idle_hours = (len(employees) * current_time) - total_work_hours if current_time > 0 else 0.0
    idle_ratio = 1 - utilization if utilization > 0 else 0.0
    summary = {
        "total_tasks": len(all_rows),
        "levels": int(topo[LEVEL_COL].max()) if not topo.empty else 0,
        "batches": len(score_rows),
        "makespan_hours": current_time,
        "total_work_hours": total_work_hours,
        "avg_task_duration_hours": avg_task_duration,
        "p50_task_duration_hours": p50_task_duration,
        "p90_task_duration_hours": p90_task_duration,
        "avg_batch_duration_hours": avg_batch_duration,
        "p90_batch_duration_hours": p90_batch_duration,
        "total_idle_hours": total_idle_hours,
        "idle_ratio": idle_ratio,
        "utilization": utilization,
        "skill_match_rate": skill_match_rate,
        "avg_total_score": float(np.mean([s["total"] for s in score_rows])) if score_rows else 0.0,
        "batch_scores": score_rows,
    }
    with open(args.output_score, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Assignment saved ->", args.output_assignment)
    print("Score saved ->", args.output_score)


if __name__ == "__main__":
    main()
