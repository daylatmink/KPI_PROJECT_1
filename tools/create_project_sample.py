#!/usr/bin/env python
"""
Create a subset of project data for visualization.
Selects tasks from the largest topo level to create a manageable dataset.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse


def create_project_subset(assignment_file, links_file, output_dir, target_size=150):
    """
    Create a subset of project with approximately target_size tasks.
    Strategy: Select top N assignees by task count, keep their tasks.
    """
    
    # Load data
    assignment_df = pd.read_csv(assignment_file)
    links_df = pd.read_csv(links_file)
    
    print(f"Original dataset: {len(assignment_df)} tasks")
    
    # Calculate assignee task counts
    assignee_counts = assignment_df['Assigned_To'].value_counts()
    
    # Select assignees to reach target size
    selected_assignees = []
    task_count = 0
    
    for assignee, count in assignee_counts.items():
        selected_assignees.append(assignee)
        task_count += count
        if task_count >= target_size:
            break
    
    # Filter assignment data
    subset_assignment = assignment_df[
        assignment_df['Assigned_To'].isin(selected_assignees)
    ].copy()
    
    # Get relevant tasks
    relevant_task_ids = set(subset_assignment['Task_ID'].astype(str))
    
    # Filter links - only keep links between relevant tasks
    subset_links = links_df[
        (links_df['from_issue_key'].str.replace('ZOOKEEPER-', '', regex=False).isin(relevant_task_ids)) &
        (links_df['to_issue_key'].str.replace('ZOOKEEPER-', '', regex=False).isin(relevant_task_ids))
    ].copy()
    
    # Save subset files
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    assignment_out = output_path / 'hs_assignment_sample.csv'
    links_out = output_path / 'issue_links_sample.csv'
    
    subset_assignment.to_csv(assignment_out, index=False)
    subset_links.to_csv(links_out, index=False)
    
    print(f"\n✓ Subset created:")
    print(f"  Tasks: {len(subset_assignment)} (target was {target_size})")
    print(f"  Assignees: {len(selected_assignees)}")
    print(f"  Dependencies: {len(subset_links)}")
    print(f"\n✓ Saved:")
    print(f"  {assignment_out}")
    print(f"  {links_out}")
    
    # Print selected assignees
    print(f"\n✓ Selected assignees ({len(selected_assignees)}):")
    for assignee in selected_assignees[:15]:
        count = len(subset_assignment[subset_assignment['Assigned_To'] == assignee])
        print(f"  {assignee}: {count} tasks")
    if len(selected_assignees) > 15:
        print(f"  ... and {len(selected_assignees) - 15} more")
    
    return str(assignment_out), str(links_out)


def analyze_project_samples():
    """Analyze all projects and show sample options."""
    
    projects = {
        'ZOOKEEPER': {
            'assignment': 'projects/ZOOKEEPER/hs_assignment.csv',
            'links': 'projects/ZOOKEEPER/issue_links.csv'
        },
        'MAPREDUCE': {
            'assignment': 'projects/MAPREDUCE/hs_assignment.csv',
            'links': 'projects/MAPREDUCE/issue_links.csv'
        },
        'OFBIZ': {
            'assignment': 'projects/OFBIZ/hs_assignment.csv',
            'links': 'projects/OFBIZ/issue_links.csv'
        }
    }
    
    print("\n" + "="*70)
    print("PROJECT ANALYSIS FOR GANTT VISUALIZATION".center(70))
    print("="*70 + "\n")
    
    for proj_name, files in projects.items():
        try:
            assignment_df = pd.read_csv(files['assignment'])
            links_df = pd.read_csv(files['links'])
            
            # Count by level
            level_counts = assignment_df.groupby('Topo_Level').size()
            assignee_counts = assignment_df.groupby('Assigned_To').size()
            
            print(f"📊 {proj_name}:")
            print(f"   Total Tasks: {len(assignment_df)}")
            print(f"   Total Assignees: {len(assignee_counts)}")
            print(f"   Dependencies: {len(links_df)}")
            print(f"   Topo Levels: {len(level_counts)}")
            
            for level, count in level_counts.items():
                print(f"     Level {level}: {count} tasks")
            
            print(f"   Top 5 Assignees:")
            top_5 = assignee_counts.nlargest(5)
            for assignee, count in top_5.items():
                print(f"     {assignee}: {count} tasks")
            print()
            
        except FileNotFoundError:
            print(f"⚠️  {proj_name}: Files not found\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create project subset for visualization'
    )
    parser.add_argument('--assignment', help='Assignment CSV file path')
    parser.add_argument('--links', help='Issue links CSV file path')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--size', type=int, default=150, help='Target subset size')
    parser.add_argument('--analyze-only', action='store_true', 
                       help='Just analyze projects without creating subset')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_project_samples()
    elif args.assignment and args.links:
        create_project_subset(args.assignment, args.links, args.output_dir, args.size)
    else:
        print("Please provide --assignment and --links, or use --analyze-only")
