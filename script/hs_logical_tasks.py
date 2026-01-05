import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# 0. CONFIG (override via CLI)
# =============================================================================

TASKS_FILE = r"zookeeper/logical_tasks_tagged.csv"
ASSIGNEES_FILE = r"zookeeper/ZOOKEEPER_assignees.csv"

OUTPUT_ASSIGNMENT = r"zookeeper/ZOOKEEPER_hs_assignment.csv"
OUTPUT_SCORE = r"zookeeper/ZOOKEEPER_hs_score.csv"

TASK_ID_COL = "task_id"
TASK_TAG_COL = "Task_Tag"
PRIORITY_COL = "main_priority"
SUMMARY_COL = "representative_summary"

ASSIGNEE_CODE_COL = "assignee_code"
ASSIGNEE_SKILLS_COL = "skills"
ASSIGNEE_SCORES_COL = "skill_scores"


# =============================================================================
# 1. CONFIG STRUCTS
# =============================================================================

@dataclass
class HSConfig:
    harmony_memory_size: int = 15
    hmcr: float = 0.85
    par: float = 0.15
    num_iterations: int = 300
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


# =============================================================================
# 2. DATA LOADER
# =============================================================================

class DataLoaderLogicalTasks:
    def __init__(self, tasks_file: str, assignees_file: str):
        self.tasks = pd.read_csv(tasks_file)
        self.assignees = pd.read_csv(assignees_file)

        # task_id -> [(tag, required_level)]
        self.task_skills: Dict[str, List[Tuple[str, int]]] = {}
        # assignee_code -> { tag: level }
        self.emp_skills: Dict[str, Dict[str, int]] = {}
        self.emp_list: List[str] = []

    def load(self):
        for col in [TASK_ID_COL, TASK_TAG_COL, PRIORITY_COL, SUMMARY_COL]:
            if col not in self.tasks.columns:
                raise ValueError(f"Tasks file missing column: {col}")

        for col in [ASSIGNEE_CODE_COL, ASSIGNEE_SKILLS_COL, ASSIGNEE_SCORES_COL]:
            if col not in self.assignees.columns:
                raise ValueError(f"Assignees file missing column: {col}")

        self._parse_task_skill()
        self._parse_employee_skill()

    def _priority_to_level(self, p: str) -> int:
        p = str(p).strip()
        mapping = {
            "Blocker": 5,
            "Critical": 4,
            "Major": 3,
            "Minor": 2,
            "Trivial": 1,
        }
        return mapping.get(p, 2)

    def _parse_task_skill(self):
        for _, row in self.tasks.iterrows():
            tid = str(row[TASK_ID_COL])
            tag = str(row[TASK_TAG_COL]).strip()
            pr = self._priority_to_level(row[PRIORITY_COL])
            self.task_skills[tid] = [(tag, pr)]

    def _parse_employee_skill(self):
        for _, row in self.assignees.iterrows():
            emp = str(row[ASSIGNEE_CODE_COL]).strip()
            if not emp:
                continue
            self.emp_list.append(emp)
            self.emp_skills.setdefault(emp, {})

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
                self.emp_skills[emp][skill] = max(
                    level,
                    self.emp_skills[emp].get(skill, 0),
                )

        self.emp_list = sorted(set(self.emp_list))

    def get_tasks(self) -> List[str]:
        return [str(x) for x in self.tasks[TASK_ID_COL]]

    def get_employees(self) -> List[str]:
        return self.emp_list

    def get_task_info(self, tid: str) -> Dict:
        row = self.tasks[self.tasks[TASK_ID_COL] == int(tid)].iloc[0]
        return {
            "Task_ID": tid,
            "Summary": row[SUMMARY_COL],
            "Priority": row[PRIORITY_COL],
            "Task_Tag": row[TASK_TAG_COL],
        }

    def get_emp_skills(self, emp: str) -> Dict[str, int]:
        return self.emp_skills.get(emp, {})


# =============================================================================
# 3. OBJECTIVE CALCULATOR
# =============================================================================

class ObjectiveCalculator:
    def __init__(self, loader: DataLoaderLogicalTasks):
        self.data = loader
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


# =============================================================================
# 4. HARMONY SEARCH
# =============================================================================

class HarmonySearch:
    def __init__(self, loader: DataLoaderLogicalTasks, calc: ObjectiveCalculator):
        self.data = loader
        self.calc = calc
        self.cfg = HSConfig()

        np.random.seed(self.cfg.seed)

        self.tasks = loader.get_tasks()
        self.emps = loader.get_employees()

        self.hm = []  # [(assign, score)]
        self.best = None
        self.best_score = -1

    def random_assign(self):
        return {tid: np.random.choice(self.emps) for tid in self.tasks}

    def run(self):
        for _ in range(self.cfg.harmony_memory_size):
            h = self.random_assign()
            s, _ = self.calc.score(h)
            self.hm.append((h, s))
            if s > self.best_score:
                self.best_score = s
                self.best = h

        self.hm.sort(key=lambda x: x[1], reverse=True)

        for it in range(self.cfg.num_iterations):
            new = {}
            for tid in self.tasks:
                if np.random.rand() < self.cfg.hmcr:
                    h, _ = self.hm[np.random.randint(len(self.hm))]
                    val = h[tid]
                    if np.random.rand() < self.cfg.par:
                        val = np.random.choice(self.emps)
                    new[tid] = val
                else:
                    new[tid] = np.random.choice(self.emps)

            s, _ = self.calc.score(new)

            if s > self.hm[-1][1]:
                self.hm[-1] = (new, s)
                self.hm.sort(key=lambda x: x[1], reverse=True)

            if s > self.best_score:
                self.best_score = s
                self.best = new

            if (it + 1) % 50 == 0:
                print(f"Iter {it + 1}: best={self.best_score:.4f}")

        print("\nHS Done. Best =", self.best_score)
        return self.best, self.best_score


# =============================================================================
# 5. SAVE OUTPUT
# =============================================================================

def save_assignment(assign: Dict[str, str], loader: DataLoaderLogicalTasks, out_path: str):
    rows = []
    for tid, emp in assign.items():
        info = loader.get_task_info(tid)
        rows.append(
            {
                "Task_ID": tid,
                "Summary": info["Summary"],
                "Task_Tag": info["Task_Tag"],
                "Priority": info["Priority"],
                "Assigned_To": emp,
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def save_score(score, details, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Harmony Search assignment using logical_tasks_tagged + assignees.",
    )
    parser.add_argument("--tasks", default=TASKS_FILE)
    parser.add_argument("--assignees", default=ASSIGNEES_FILE)
    parser.add_argument("--output-assignment", default=OUTPUT_ASSIGNMENT)
    parser.add_argument("--output-score", default=OUTPUT_SCORE)
    return parser.parse_args()


def main():
    args = parse_args()

    loader = DataLoaderLogicalTasks(args.tasks, args.assignees)
    loader.load()

    calc = ObjectiveCalculator(loader)
    hs = HarmonySearch(loader, calc)

    best, _ = hs.run()
    total, details = calc.score(best)

    os.makedirs(os.path.dirname(args.output_assignment), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_score), exist_ok=True)

    save_assignment(best, loader, args.output_assignment)
    save_score(total, details, args.output_score)

    print("\nAssignment saved ->", args.output_assignment)
    print("Score saved ->", args.output_score)


if __name__ == "__main__":
    main()
