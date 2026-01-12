#!/usr/bin/env python3
"""
Utility script to update all 07_*.py assignment scripts with cost optimization support.
This script adds:
1. emp_costs parameter to BatchContext
2. emp_costs loading in main()
3. emp_costs passing to BatchContext initialization
4. cost_optimization scoring method
5. cost_optimization to score() method
"""

import os
import re
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
SCRIPTS_TO_UPDATE = [
    "07_ihs_topo_assign.py",
    "07_ghs_topo_assign.py",
    "07_mohs_topo_assign.py"
]

COST_OPTIMIZATION_METHOD = '''    def _cost_optimization(self, assign):
        """
        Calculate cost optimization score.
        Prefer assignments that distribute cost efficiently across employees.
        Higher score = lower total cost + better workload distribution
        """
        if not self.data.emp_costs:
            return 1.0  # No cost data, neutral score
        
        # Calculate total cost of this assignment
        total_cost = 0
        emp_costs = {}
        
        for tid, emp in assign.items():
            info = self.data.get_task_info(tid)
            duration = info.get("Duration_Hours", 1.0)
            hourly_rate = self.data.emp_costs.get(emp, 50.0)  # Default rate if not found
            task_cost = hourly_rate * duration
            total_cost += task_cost
            emp_costs[emp] = emp_costs.get(emp, 0) + task_cost
        
        # Calculate cost balance (lower is better)
        # We want to balance costs across employees, not just minimize total
        if not emp_costs:
            return 1.0
        
        cost_values = list(emp_costs.values())
        cost_mean = np.mean(cost_values)
        cost_std = np.std(cost_values)
        
        # Normalize: std / mean gives us coefficient of variation
        if cost_mean > 0:
            cost_ratio = cost_std / cost_mean
            # Lower ratio = better balance, score should be higher
            balance_score = 1.0 / (1.0 + cost_ratio)
        else:
            balance_score = 0.5
        
        return balance_score
'''

def update_batch_context(content):
    """Add emp_costs parameter to BatchContext __init__"""
    # Update __init__ signature
    pattern = r'(class BatchContext:)\s*(def __init__\(self, tasks_df: pd\.DataFrame, emp_skills: Dict\[str, Dict\[str, int\]\]\):)'
    replacement = r'\1\n    def __init__(self, tasks_df: pd.DataFrame, emp_skills: Dict[str, Dict[str, int]], \n                 emp_costs: Dict[str, float] = None):'
    content = re.sub(pattern, replacement, content)
    
    # Add emp_costs field initialization
    pattern = r'(self\.emp_skills = emp_skills)\n(        self\.task_skills)'
    replacement = r'\1\n        self.emp_costs = emp_costs or {}\n\2'
    content = re.sub(pattern, replacement, content)
    
    return content

def update_score_method(content):
    """Update score() method to include cost_optimization"""
    # Find score method and update it
    pattern = r'(def score\(self, assign: Dict\[str, str\]\) -> Tuple\[float, Dict\]:)\s*(s1 = self\._skill_matching\(assign\))\s*(s2 = self\._workload_balance\(assign\))\s*(s3 = self\._priority_respect\(assign\))\s*(s4 = self\._skill_dev\(assign\))'
    replacement = r'\1\n        \2\n        \3\n        \4\n        \5\n        s5 = self._cost_optimization(assign)'
    content = re.sub(pattern, replacement, content)
    
    # Update total calculation
    pattern = r'(total = \(\s*s1 \* self\.weights\.skill_matching\s*\+ s2 \* self\.weights\.workload_balance\s*\+ s3 \* self\.weights\.priority_respect\s*\+ s4 \* self\.weights\.skill_development)\s*(\))'
    replacement = r'\1\n            + s5 * self.weights.cost_optimization\n\2'
    content = re.sub(pattern, replacement, content)
    
    # Update return statement dict
    pattern = r'(return total, \{\s*"skill_matching": s1,\s*"workload_balance": s2,\s*"priority_respect": s3,\s*"skill_development": s4,)\s*("total": total,)'
    replacement = r'\1\n            "cost_optimization": s5,\n            \2'
    content = re.sub(pattern, replacement, content)
    
    return content

def add_cost_optimization_method(content):
    """Add _cost_optimization method to ObjectiveCalculator"""
    # Find the _skill_dev method and add _cost_optimization after it
    pattern = r'(    def _skill_dev\(self, assign\):.*?return min\(1, avg / 10\))'
    if re.search(pattern, content, re.DOTALL):
        replacement = r'\1\n\n' + COST_OPTIMIZATION_METHOD
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    return content

def update_main_function(content):
    """Update main() to load costs and pass to BatchContext"""
    # Add emp_costs loading after emp_skills
    pattern = r'(emp_skills = parse_assignee_skills\(assignees_path\))'
    replacement = r'\1\n    emp_costs = load_assignee_costs(assignees_path)'
    content = re.sub(pattern, replacement, content)
    
    # Update BatchContext instantiation
    pattern = r'(ctx = BatchContext\(batch_df, emp_skills)\)'
    replacement = r'\1, emp_costs)'
    content = re.sub(pattern, replacement, content)
    
    return content

def process_file(file_path):
    """Process a single script file"""
    print(f"Processing {file_path.name}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip if already updated
    if '_cost_optimization' in content and 'emp_costs = load_assignee_costs' in content:
        print(f"  ✓ Already updated")
        return
    
    # Apply updates
    content = update_batch_context(content)
    content = update_score_method(content)
    content = add_cost_optimization_method(content)
    content = update_main_function(content)
    
    # Save updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ Updated successfully")

def main():
    for script_name in SCRIPTS_TO_UPDATE:
        script_path = SCRIPTS_DIR / script_name
        if script_path.exists():
            process_file(script_path)
        else:
            print(f"⚠ {script_name} not found")
    
    print("\n✅ All scripts updated!")

if __name__ == "__main__":
    main()
