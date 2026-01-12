"""
Script to assign hourly costs to assignees based on their skill profile.
Cost is determined by:
1. Skill complexity weights
2. Number of skills (breadth of expertise)
3. Experience level (number of tasks completed)
"""

import pandas as pd
import os
from pathlib import Path

# Configuration for skill weights (complexity level)
SKILL_WEIGHTS = {
    'Design & API Evolution': 1.5,           # Highest complexity
    'Build & Release Engineering': 1.4,      # High complexity
    'Performance Optimization': 1.3,         # High complexity
    'Code Refactoring & Cleanup': 1.2,       # Medium-high complexity
    'Feature / Improvement Implementation': 1.1,  # Medium complexity
    'Testing & Verification': 1.0,           # Medium complexity (baseline)
    'Bug fixing / Maintenance': 0.9,         # Medium-low complexity
    'Documentation': 0.8,                    # Low complexity
    'Project / General Task': 0.7,           # Low complexity
    'Other': 0.5,                            # Minimal complexity
}

# Base hourly rates (USD) by seniority
BASE_RATES = {
    'junior': 25,      # Entry level: 0-2 skills
    'mid': 40,         # Mid level: 3-5 skills
    'senior': 60,      # Senior: 6-8 skills
    'expert': 85,      # Expert: 9+ skills
}

# Experience multiplier based on total tasks
def get_experience_multiplier(total_tasks):
    """
    Calculate experience multiplier based on number of tasks completed.
    More tasks = higher rate (specialization in that domain)
    """
    if total_tasks < 5:
        return 0.8  # Junior
    elif total_tasks < 15:
        return 0.95  # Mid-junior
    elif total_tasks < 50:
        return 1.1   # Mid-senior
    elif total_tasks < 100:
        return 1.3   # Senior
    else:
        return 1.5   # Expert


def get_seniority_level(num_skills):
    """Determine seniority level based on number of skills"""
    if num_skills <= 2:
        return 'junior'
    elif num_skills <= 5:
        return 'mid'
    elif num_skills <= 8:
        return 'senior'
    else:
        return 'expert'


def calculate_assignee_cost(row):
    """
    Calculate hourly cost for an assignee.
    
    Parameters:
    - row: pandas Series with columns: Assignee, num_skills, total_tasks, and skill columns
    
    Returns:
    - hourly_cost: float
    """
    num_skills = row['num_skills']
    total_tasks = row['total_tasks']
    
    # Get base rate by seniority level
    seniority = get_seniority_level(num_skills)
    base_rate = BASE_RATES[seniority]
    
    # Calculate skill complexity bonus
    # Sum up weights for skills where assignee has done at least 1 task
    skill_bonus = 0
    skill_columns = [col for col in row.index if col in SKILL_WEIGHTS]
    
    for skill in skill_columns:
        if row[skill] > 0:  # If assignee has done at least 1 task in this skill
            skill_bonus += SKILL_WEIGHTS[skill]
    
    # Normalize skill bonus (divide by number of skills to get average weight)
    if num_skills > 0:
        avg_skill_weight = skill_bonus / num_skills
    else:
        avg_skill_weight = 0.5
    
    # Calculate experience multiplier
    exp_multiplier = get_experience_multiplier(total_tasks)
    
    # Final hourly cost
    hourly_cost = base_rate * avg_skill_weight * exp_multiplier
    
    # Round to nearest 0.5
    hourly_cost = round(hourly_cost * 2) / 2
    
    return hourly_cost


def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    skill_profile_path = project_root / 'data' / 'interim' / 'assignee_skill_profile.csv'
    output_path = project_root / 'data' / 'interim' / 'assignee_cost_profile.csv'
    
    print(f"Reading skill profile from: {skill_profile_path}")
    
    # Read skill profile
    df = pd.read_csv(skill_profile_path)
    
    print(f"Loaded {len(df)} assignees")
    
    # Calculate hourly cost for each assignee
    df['hourly_cost_usd'] = df.apply(calculate_assignee_cost, axis=1)
    
    # Create seniority level column for reference
    df['seniority_level'] = df['num_skills'].apply(get_seniority_level)
    
    # Create experience level column for reference
    df['experience_level'] = df['total_tasks'].apply(
        lambda x: 'entry' if x < 5 else 'junior' if x < 15 else 'mid' if x < 50 else 'senior' if x < 100 else 'expert'
    )
    
    # Select relevant columns for output
    output_columns = [
        'Assignee',
        'total_tasks',
        'num_skills',
        'seniority_level',
        'experience_level',
        'hourly_cost_usd',
        'main_skill_tag'
    ]
    
    # Add skill columns
    skill_columns = [col for col in df.columns if col in SKILL_WEIGHTS]
    output_columns.extend(skill_columns)
    
    df_output = df[output_columns].copy()
    
    # Save to CSV
    print(f"Saving cost profile to: {output_path}")
    df_output.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ASSIGNEE COST PROFILE SUMMARY")
    print("="*60)
    print(f"\nTotal assignees: {len(df_output)}")
    print(f"\nCost Statistics (USD/hour):")
    print(f"  Min: ${df_output['hourly_cost_usd'].min():.2f}")
    print(f"  Max: ${df_output['hourly_cost_usd'].max():.2f}")
    print(f"  Mean: ${df_output['hourly_cost_usd'].mean():.2f}")
    print(f"  Median: ${df_output['hourly_cost_usd'].median():.2f}")
    
    print(f"\nCost by Seniority Level:")
    for level in ['junior', 'mid', 'senior', 'expert']:
        mask = df_output['seniority_level'] == level
        if mask.any():
            costs = df_output[mask]['hourly_cost_usd']
            print(f"  {level.upper():10s}: {len(costs):3d} assignees, "
                  f"Avg: ${costs.mean():.2f}, Range: ${costs.min():.2f} - ${costs.max():.2f}")
    
    print(f"\nTop 10 Most Expensive Assignees:")
    top_10 = df_output.nlargest(10, 'hourly_cost_usd')[['Assignee', 'hourly_cost_usd', 'seniority_level', 'total_tasks', 'num_skills']]
    for idx, row in top_10.iterrows():
        print(f"  {row['Assignee']:15s}: ${row['hourly_cost_usd']:6.2f}/hr ({row['seniority_level']:6s}, {row['total_tasks']:3.0f} tasks, {row['num_skills']:1.0f} skills)")
    
    print(f"\nTop 10 Cheapest Assignees:")
    bottom_10 = df_output.nsmallest(10, 'hourly_cost_usd')[['Assignee', 'hourly_cost_usd', 'seniority_level', 'total_tasks', 'num_skills']]
    for idx, row in bottom_10.iterrows():
        print(f"  {row['Assignee']:15s}: ${row['hourly_cost_usd']:6.2f}/hr ({row['seniority_level']:6s}, {row['total_tasks']:3.0f} tasks, {row['num_skills']:1.0f} skills)")
    
    print("\n" + "="*60)
    print(f"✓ Cost profile saved to: {output_path}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
