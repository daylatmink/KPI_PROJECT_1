"""
Utility functions for handling assignee costs in assignment algorithms.
"""

import os
import pandas as pd
from typing import Dict, Optional


def load_assignee_costs(assignees_path: str) -> Dict[str, float]:
    """
    Load hourly costs for assignees from assignees CSV file.
    
    Args:
        assignees_path: Path to assignees CSV (must have 'assignee_code' and 'hourly_cost_usd' columns)
    
    Returns:
        Dictionary mapping assignee_code -> hourly_cost_usd
    """
    if not os.path.isfile(assignees_path):
        print(f"Warning: Assignees file not found: {assignees_path}")
        return {}
    
    df = pd.read_csv(assignees_path)
    
    # Check if hourly_cost_usd column exists
    if "hourly_cost_usd" not in df.columns:
        print(f"Warning: hourly_cost_usd column not found in {assignees_path}")
        return {}
    
    if "assignee_code" not in df.columns:
        print(f"Warning: assignee_code column not found in {assignees_path}")
        return {}
    
    costs = {}
    for _, row in df.iterrows():
        code = str(row.get("assignee_code", "")).strip()
        cost = row.get("hourly_cost_usd")
        
        if code and pd.notna(cost):
            try:
                costs[code] = float(cost)
            except (ValueError, TypeError):
                pass
    
    return costs


def get_total_cost(assignment: Dict[str, str], assignee_costs: Dict[str, float], 
                   task_durations: Dict[str, float]) -> float:
    """
    Calculate total cost of an assignment.
    
    Args:
        assignment: Dictionary mapping task_id -> assignee_code
        assignee_costs: Dictionary mapping assignee_code -> hourly_cost_usd
        task_durations: Dictionary mapping task_id -> duration_hours
    
    Returns:
        Total cost in USD
    """
    total = 0.0
    for task_id, assignee_code in assignment.items():
        if assignee_code in assignee_costs and task_id in task_durations:
            total += assignee_costs[assignee_code] * task_durations[task_id]
    return total


def get_assignee_total_cost(assignee_code: str, assigned_tasks: list, 
                           assignee_costs: Dict[str, float], 
                           task_durations: Dict[str, float]) -> float:
    """
    Calculate total cost for an assignee based on assigned tasks.
    
    Args:
        assignee_code: Code of the assignee
        assigned_tasks: List of task IDs assigned to this assignee
        assignee_costs: Dictionary mapping assignee_code -> hourly_cost_usd
        task_durations: Dictionary mapping task_id -> duration_hours
    
    Returns:
        Total cost in USD for this assignee
    """
    cost = assignee_costs.get(assignee_code, 0.0)
    duration = sum(task_durations.get(task_id, 0.0) for task_id in assigned_tasks)
    return cost * duration
