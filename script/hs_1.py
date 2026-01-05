import pandas as pd
import numpy as np
import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

# =============================================================================
# 0. CONFIG – sửa nếu cần
# =============================================================================

ISSUES_FILE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\hs\hs_input\macr_issues.csv"
SKILLS_FILE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\hs\hs_input\macr_skills.csv"

OUTPUT_ASSIGNMENT = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\hs\hs_output\macr_hs_assignment.csv"
OUTPUT_SCORE = r"C:\Users\ADMIN\PycharmProjects\PythonProject4\hs\hs_output\macr_hs_score.csv"

TASK_ID_COL = "Task_ID"
TASK_TAG_COL = "Task_Tag"
PRIORITY_COL = "Priority"
ASSIGNEE_COL = "Assignee_ID"

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

# =============================================================================
# 2. DATA LOADER (fix column names)
# =============================================================================

class DataLoaderV2:

    def __init__(self):
        self.issues = pd.read_csv(ISSUES_FILE)
        self.skills = pd.read_csv(SKILLS_FILE)

        # task_id → [(tag, required_level)]
        self.task_skills: Dict[str, List[Tuple[str,int]]] = {}

        # assignee → { tag: level }
        self.emp_skills: Dict[str, Dict[str,int]] = {}

    # -----------------------------
    def load(self):
        print("\n[DataLoader] Loading...")

        # Alias Assignee -> Assignee_ID if missing
        if ASSIGNEE_COL not in self.issues.columns and "Assignee" in self.issues.columns:
            self.issues[ASSIGNEE_COL] = self.issues["Assignee"]

        # Validate minimal columns
        for col in [TASK_ID_COL, TASK_TAG_COL, PRIORITY_COL, ASSIGNEE_COL]:
            if col not in self.issues.columns:
                raise ValueError(f"Issues file missing column: {col}")

        for col in [ASSIGNEE_COL, TASK_TAG_COL, "Skill_Level"]:
            if col not in self.skills.columns:
                raise ValueError(f"Skills file missing column: {col}")

        self._parse_task_skill()
        self._parse_employee_skill()

        print(f"  Tasks parsed: {len(self.task_skills)}")
        print(f"  Employees parsed: {len(self.emp_skills)}")

    # -----------------------------
    def _priority_to_level(self, p: str) -> int:
        p = str(p).strip()
        mapping = {
            "Blocker": 5,
            "Critical": 4,
            "Major": 3,
            "Minor": 2,
            "Trivial": 1
        }
        return mapping.get(p, 2)

    # -----------------------------
    def _parse_task_skill(self):
        for _, row in self.issues.iterrows():
            tid = str(row[TASK_ID_COL])
            tag = str(row[TASK_TAG_COL]).strip()
            pr = self._priority_to_level(row[PRIORITY_COL])
            self.task_skills[tid] = [(tag, pr)]

    # -----------------------------
    def _parse_employee_skill(self):
        for _, row in self.skills.iterrows():
            emp = str(row[ASSIGNEE_COL])
            tag = str(row[TASK_TAG_COL]).strip()
            level = int(row["Skill_Level"])

            if emp not in self.emp_skills:
                self.emp_skills[emp] = {}

            self.emp_skills[emp][tag] = max(
                level,
                self.emp_skills[emp].get(tag, 0)
            )

    # -----------------------------
    def get_tasks(self) -> List[str]:
        return [str(x) for x in self.issues[TASK_ID_COL]]

    def get_employees(self) -> List[str]:
        return sorted(self.emp_skills.keys())

    def get_task_info(self, tid: str) -> Dict:
        row = self.issues[self.issues[TASK_ID_COL] == tid].iloc[0]
        return {
            "Task_ID": tid,
            "Summary": row["Summary"],
            "Priority": row[PRIORITY_COL],
            "Task_Tag": row[TASK_TAG_COL]
        }

    def get_emp_skills(self, emp: str) -> Dict[str,int]:
        return self.emp_skills.get(emp, {})

# =============================================================================
# 3. OBJECTIVE CALCULATOR
# =============================================================================

class ObjectiveCalculatorV2:

    def __init__(self, loader: DataLoaderV2):
        self.data = loader
        self.weights = ObjectiveWeights()
        self.penalties = PenaltyFactors()

    # MAIN
    def score(self, assign: Dict[str,str]) -> Tuple[float,Dict]:
        s1 = self._skill_matching(assign)
        s2 = self._workload_balance(assign)
        s3 = self._priority_respect(assign)
        s4 = self._skill_dev(assign)

        total = (
            s1*self.weights.skill_matching +
            s2*self.weights.workload_balance +
            s3*self.weights.priority_respect +
            s4*self.weights.skill_development
        )

        return total, {
            "skill_matching": s1,
            "workload_balance": s2,
            "priority_respect": s3,
            "skill_development": s4,
            "total": total
        }

    # Obj1
    def _skill_matching(self, assign):
        penalty = 0
        for tid, emp in assign.items():
            reqs = self.data.task_skills[tid]
            emp_sk = self.data.get_emp_skills(emp)

            for tag, need in reqs:
                have = emp_sk.get(tag,0)
                if have == 0:
                    penalty += self.penalties.skill_mismatch
                else:
                    gap = max(0, need - have)
                    penalty += gap * self.penalties.skill_gap / 100.0

        avg_penalty = penalty / len(assign)
        return max(0, 1 - avg_penalty)

    # Obj2
    def _workload_balance(self, assign):
        counts = {}
        for _, emp in assign.items():
            counts[emp] = counts.get(emp, 0) + 1

        vals = list(counts.values())
        std = np.std(vals)
        max_std = len(assign)/2
        return 1 / (1 + std/max_std)

    # Obj3
    def _priority_respect(self, assign):
        weights = {
            "Blocker":1.0,"Critical":0.8,"Major":0.5,"Minor":0.2,"Trivial":0.1
        }

        reward = 0
        for tid, emp in assign.items():
            info = self.data.get_task_info(tid)
            w = weights.get(info["Priority"],0.5)
            emp_skill_avg = np.mean(list(self.data.get_emp_skills(emp).values()))
            if np.isnan(emp_skill_avg):
                emp_skill_avg = 1

            if w >= 0.5:
               reward += w * (emp_skill_avg/5)
            else:
               reward += 0.3

        return min(1, reward/len(assign))

    # Obj4
    def _skill_dev(self, assign):
        diversity = {}
        for tid, emp in assign.items():
            tag,_ = self.data.task_skills[tid][0]
            diversity.setdefault(emp,set()).add(tag)

        vals = [len(v) for v in diversity.values()]
        avg = np.mean(vals)
        return min(1, avg/10)

# =============================================================================
# 4. HARMONY SEARCH
# =============================================================================

class HarmonySearch:

    def __init__(self, loader: DataLoaderV2, calc: ObjectiveCalculatorV2):
        self.data = loader
        self.calc = calc
        self.cfg = HSConfig()

        np.random.seed(self.cfg.seed)

        self.tasks = loader.get_tasks()
        self.emps = loader.get_employees()

        self.hm = []  # [(assign,score)]
        self.best = None
        self.best_score = -1

    def random_assign(self):
        return {
            tid: np.random.choice(self.emps)
            for tid in self.tasks
        }

    def run(self):
        # Init HM
        for _ in range(self.cfg.harmony_memory_size):
            h = self.random_assign()
            s,_ = self.calc.score(h)
            self.hm.append((h,s))
            if s > self.best_score:
                self.best_score = s
                self.best = h

        self.hm.sort(key=lambda x:x[1], reverse=True)

        # Improve
        for it in range(self.cfg.num_iterations):
            new = {}
            for tid in self.tasks:
                if np.random.rand() < self.cfg.hmcr:
                    # pick from memory
                    h,_ = self.hm[np.random.randint(len(self.hm))]
                    val = h[tid]
                    if np.random.rand() < self.cfg.par:
                        val = np.random.choice(self.emps)
                    new[tid] = val
                else:
                    new[tid] = np.random.choice(self.emps)

            s,_ = self.calc.score(new)

            # replace worst
            if s > self.hm[-1][1]:
                self.hm[-1] = (new,s)
                self.hm.sort(key=lambda x:x[1], reverse=True)

            if s > self.best_score:
                self.best_score = s
                self.best = new

            if (it+1) % 50 == 0:
                print(f"Iter {it+1}: best={self.best_score:.4f}")

        print("\nHS Done. Best =", self.best_score)
        return self.best, self.best_score

# =============================================================================
# 5. SAVE OUTPUT
# =============================================================================

def save_assignment(assign:Dict[str,str], loader:DataLoaderV2):
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
    pd.DataFrame(rows).to_csv(OUTPUT_ASSIGNMENT, index=False, encoding="utf-8-sig")

def save_score(score, details):
    with open(OUTPUT_SCORE,"w",encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

# =============================================================================
# MAIN
# =============================================================================

def main():
    loader = DataLoaderV2()
    loader.load()

    calc = ObjectiveCalculatorV2(loader)
    hs = HarmonySearch(loader, calc)

    best, score = hs.run()
    total,details = calc.score(best)

    save_assignment(best, loader)
    save_score(total, details)

    print("\nAssignment saved ->", OUTPUT_ASSIGNMENT)
    print("Score saved ->", OUTPUT_SCORE)

if __name__ == "__main__":
    main()
