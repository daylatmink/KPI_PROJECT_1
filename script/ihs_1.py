"""
Improved Harmony Search (IHS) – Version 2
Compatible with new data format:

ISSUES file columns:
    Task_ID, Project, Summary, Assignee, Priority, IssueType, Estimated_Seconds,
    Spent_Seconds, Created, ResolutionDate, Status, Depends_On, Task_Tag, Assignee_ID

SKILLS file columns:
    Assignee_ID, Task_Tag, Issue_Count, Avg_Priority, Skill_Level

No SWE_Area, no assignee_code, no capacity constraint.
Automatically detects correct column names.
"""

import os
import json
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple


# =============================================================================
# 0. CONFIG  ------------------------------------------------------------------
# =============================================================================

ISSUES_FILE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\hs\hs_input\macr_issues.csv"
SKILLS_FILE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\hs\hs_input\macr_skills.csv"

OUTPUT_ASSIGN = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\hs\hs_output\macr_hs_assignment_i.csv"
OUTPUT_SCORE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\hs\hs_output\macr_hs_score_i.csv"

TASK_ID_COL = "Task_ID"
TASK_TAG_COL = "Task_Tag"
ASSIGNEE_COL = "Assignee"
PRIORITY_COL = "Priority"


# =============================================================================
# 1. CONFIG STRUCTS -----------------------------------------------------------
# =============================================================================

@dataclass
class IHSConfig:
    harmony_memory_size: int = 20
    hmcr: float = 0.88
    par_min: float = 0.10
    par_max: float = 0.95
    bw_min: float = 0.05
    bw_max: float = 0.30
    num_iterations: int = 400
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


# =============================================================================
# 2. DATA LOADER --------------------------------------------------------------
# =============================================================================

class DataLoaderV2:

    def __init__(self):
        self.issues = pd.read_csv(ISSUES_FILE)
        self.skills = pd.read_csv(SKILLS_FILE)

        self.task_skill_map = {}
        self.emp_skills = {}

        self.skill_assignee_col = None
        self.skill_tag_col = None

    # -------------------------------------------------------------------------
    def load(self):
        print("\n[DataLoader] Loading...")

        # Must contain basic columns in issues -----------------
        for col in [TASK_ID_COL, TASK_TAG_COL, ASSIGNEE_COL, PRIORITY_COL]:
            if col not in self.issues.columns:
                raise ValueError(f"Issues file missing column: {col}")

        # Detect assignee col in skills ------------------------
        if "assignee_code" in self.skills.columns:
            self.skill_assignee_col = "assignee_code"
        elif "Assignee_ID" in self.skills.columns:
            self.skill_assignee_col = "Assignee_ID"
        elif "Assignee" in self.skills.columns:
            self.skill_assignee_col = "Assignee"
        else:
            raise ValueError(
                "Skills file must contain one of: assignee_code / Assignee_ID / Assignee"
            )

        # Detect tag column in skills --------------------------
        if "Skill_Tag" in self.skills.columns:
            self.skill_tag_col = "Skill_Tag"
        elif "Task_Tag" in self.skills.columns:
            self.skill_tag_col = "Task_Tag"
        else:
            raise ValueError("Skills file must contain Skill_Tag or Task_Tag")

        if "Skill_Level" not in self.skills.columns:
            raise ValueError("Skills file missing column: Skill_Level")

        # Parse data -------------------------------------------
        self._parse_task_skills()
        self._parse_employee_skills()

        print(f"  ✓ Loaded {len(self.task_skill_map)} tasks")
        print(f"  ✓ Loaded {len(self.emp_skills)} employees")

    # -------------------------------------------------------------------------
    def _priority_to_level(self, p: str):
        mapping = {
            "Blocker": 5,
            "Critical": 4,
            "Major": 3,
            "Minor": 2,
            "Trivial": 1
        }
        return mapping.get(str(p).strip(), 2)

    # -------------------------------------------------------------------------
    def _parse_task_skills(self):
        for _, row in self.issues.iterrows():
            tid = str(row[TASK_ID_COL])
            tag = str(row[TASK_TAG_COL]).strip()
            req = self._priority_to_level(row[PRIORITY_COL])
            self.task_skill_map[tid] = [(tag, req)]

    # -------------------------------------------------------------------------
    def _parse_employee_skills(self):
        for _, row in self.skills.iterrows():
            emp = str(row[self.skill_assignee_col])
            tag = str(row[self.skill_tag_col])
            lv = int(row["Skill_Level"])

            if emp not in self.emp_skills:
                self.emp_skills[emp] = {}

            self.emp_skills[emp][tag] = max(
                lv,
                self.emp_skills[emp].get(tag, 0)
            )

    # -------------------------------------------------------------------------
    def get_tasks(self):
        return [str(x) for x in self.issues[TASK_ID_COL]]

    def get_employees(self):
        return sorted(self.emp_skills.keys())

    def get_task_info(self, tid: str):
        row = self.issues[self.issues[TASK_ID_COL] == tid].iloc[0]
        return {
            "Task_ID": tid,
            "Summary": row["Summary"],
            "Priority": row[PRIORITY_COL],
            "Task_Tag": row[TASK_TAG_COL]
        }

    def get_emp_skills(self, emp):
        return self.emp_skills.get(emp, {})


# =============================================================================
# 3. OBJECTIVES ---------------------------------------------------------------
# =============================================================================

class ObjectiveCalculatorV2:

    def __init__(self, loader: DataLoaderV2):
        self.data = loader
        self.weights = ObjectiveWeights()
        self.penalties = PenaltyFactors()

    # -------------------------------------------------------------------------
    def score(self, assign: Dict[str, str]):
        s1 = self._skill_matching(assign)
        s2 = self._workload_balance(assign)
        s3 = self._priority_respect(assign)
        s4 = self._skill_dev(assign)

        total = (
            s1 * 0.60 +
            s2 * 0.20 +
            s3 * 0.15 +
            s4 * 0.05
        )

        return total, {
            "skill_matching": s1,
            "workload_balance": s2,
            "priority_respect": s3,
            "skill_development": s4,
            "total_score": total
        }

    # -------------------------------------------------------------------------
    def _skill_matching(self, assign):
        penalty = 0
        N = len(assign)

        for tid, emp in assign.items():
            reqs = self.data.task_skill_map[tid]
            emp_sk = self.data.get_emp_skills(emp)

            for tag, need in reqs:
                have = emp_sk.get(tag, 0)
                if have == 0:
                    penalty += self.penalties.skill_mismatch
                else:
                    gap = max(0, need - have)
                    penalty += gap * self.penalties.skill_gap / 100.0

        avg_penalty = penalty / N
        return max(0, 1 - avg_penalty)

    # -------------------------------------------------------------------------
    def _workload_balance(self, assign):
        counts = {}
        for _, emp in assign.items():
            counts[emp] = counts.get(emp, 0) + 1

        vals = list(counts.values())
        std = np.std(vals)
        max_std = len(assign) / 2

        return 1 / (1 + std / max_std)

    # -------------------------------------------------------------------------
    def _priority_respect(self, assign):
        wmap = {
            "Blocker": 1.0, "Critical": 0.8,
            "Major": 0.5, "Minor": 0.2,
            "Trivial": 0.1
        }

        total = 0
        N = len(assign)

        for tid, emp in assign.items():
            info = self.data.get_task_info(tid)
            w = wmap.get(info["Priority"], 0.5)
            emp_lv = np.mean(list(self.data.get_emp_skills(emp).values())) or 1

            if w >= 0.5:
                total += w * (emp_lv / 5)
            else:
                total += 0.3

        return min(1, total / N)

    # -------------------------------------------------------------------------
    def _skill_dev(self, assign):
        sets = {}
        for tid, emp in assign.items():
            tag,_ = self.data.task_skill_map[tid][0]
            sets.setdefault(emp, set()).add(tag)

        vals = [len(v) for v in sets.values()]
        avg = np.mean(vals)
        return min(1, avg / 10)


# =============================================================================
# 4. IMPROVED HARMONY SEARCH --------------------------------------------------
# =============================================================================

class IHS:

    def __init__(self, loader: DataLoaderV2, calc: ObjectiveCalculatorV2):
        self.data = loader
        self.calc = calc
        self.cfg = IHSConfig()

        np.random.seed(self.cfg.seed)

        self.tasks = loader.get_tasks()
        self.emps = loader.get_employees()

        self.hm = []
        self.best_assign = None
        self.best_score = -1

        self._coef_bw = math.log(self.cfg.bw_min / self.cfg.bw_max) / self.cfg.num_iterations

    # -------------------------------------------------------------------------
    def random_harmony(self):
        return {tid: np.random.choice(self.emps) for tid in self.tasks}

    # -------------------------------------------------------------------------
    def neighbors(self, emp, bw):
        base = set(self.data.get_emp_skills(emp).keys())

        sims = []
        for e in self.emps:
            skills = set(self.data.get_emp_skills(e).keys())
            inter = len(base & skills)
            uni = len(base | skills)
            sim = inter / uni if uni > 0 else 0
            sims.append((e, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        k = max(1, int(len(self.emps) * bw))
        return [x[0] for x in sims[:k]]

    # -------------------------------------------------------------------------
    def run(self):
        print("\n[IHS] Initializing Harmony Memory...")

        # Init HM
        for _ in range(self.cfg.harmony_memory_size):
            h = self.random_harmony()
            s,_ = self.calc.score(h)
            self.hm.append((h,s))
            if s > self.best_score:
                self.best_score = s
                self.best_assign = h.copy()

        self.hm.sort(key=lambda x: x[1], reverse=True)

        print(f"  ✓ Initial best score: {self.best_score:.4f}")

        # Improve
        print("\n[IHS] Improving...")
        for it in range(self.cfg.num_iterations):

            par = self.cfg.par_min + (self.cfg.par_max - self.cfg.par_min) * (it/self.cfg.num_iterations)
            bw = self.cfg.bw_max * math.exp(self._coef_bw * it)

            new = {}

            for tid in self.tasks:
                if np.random.rand() < self.cfg.hmcr:
                    # memory
                    idx = np.random.randint(len(self.hm))
                    emp = self.hm[idx][0][tid]

                    if np.random.rand() < par:
                        emp = np.random.choice(self.neighbors(emp, bw))
                else:
                    emp = np.random.choice(self.emps)

                new[tid] = emp

            s,_ = self.calc.score(new)

            if s > self.hm[-1][1]:
                self.hm[-1] = (new,s)
                self.hm.sort(key=lambda x:x[1], reverse=True)

            if s > self.best_score:
                self.best_score = s
                self.best_assign = new.copy()

            if (it+1) % 50 == 0:
                print(f"  Iter {it+1}/{self.cfg.num_iterations} – Best: {self.best_score:.4f}")

        print("\n[IHS] Done.")
        print("  Best Score =", self.best_score)
        return self.best_assign, self.best_score


# =============================================================================
# 5. OUTPUT -------------------------------------------------------------------
# =============================================================================

def save_assignment(assign, loader):
    rows = []
    for tid, emp in assign.items():
        info = loader.get_task_info(tid)
        rows.append({
            "Task_ID": tid,
            "Summary": info["Summary"],
            "Task_Tag": info["Task_Tag"],
            "Priority": info["Priority"],
            "Assigned_To": emp
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_ASSIGN, index=False, encoding="utf-8-sig")
    print(f"✓ Assignment saved → {OUTPUT_ASSIGN}")


def save_score(score, details):
    with open(OUTPUT_SCORE, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"✓ Score saved → {OUTPUT_SCORE}")


# =============================================================================
# 6. MAIN ---------------------------------------------------------------------
# =============================================================================

def main():
    loader = DataLoaderV2()
    loader.load()

    calc = ObjectiveCalculatorV2(loader)
    ihs = IHS(loader, calc)

    best, score = ihs.run()
    _, details = calc.score(best)

    save_assignment(best, loader)
    save_score(score, details)

    print("\nFINISHED.")


if __name__ == "__main__":
    main()
