#!/usr/bin/env python
"""
Generate Gantt chart visualization for task assignments.
Shows task scheduling with dependencies, colored by assignee.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import json
import argparse
from pathlib import Path


def load_data(assignment_file, issue_links_file):
    """Load assignment and dependency data."""
    assignment_df = pd.read_csv(assignment_file)
    links_df = pd.read_csv(issue_links_file)
    return assignment_df, links_df


def calculate_task_schedule(assignment_df, links_df):
    """
    Calculate start/end times for tasks respecting dependencies.
    Uses topological ordering and critical path for scheduling.
    """
    # Create task lookup
    tasks = {}
    for _, row in assignment_df.iterrows():
        task_id = str(row['Task_ID'])
        tasks[task_id] = {
            'id': task_id,
            'summary': row['Summary'][:50],  # Truncate summary
            'duration': row['Duration_Hours'],
            'assignee': row['Assigned_To'],
            'priority': row['Priority'],
            'tag': row['Task_Tag'],
            'topo_level': row['Topo_Level'],
            'dependencies': [],
            'start': 0,
            'end': row['Duration_Hours']
        }
    
    # Build dependency graph (only blocking dependencies)
    for _, row in links_df.iterrows():
        from_key = str(row['from_issue_key'].replace('ZOOKEEPER-', ''))
        to_key = str(row['to_issue_key'].replace('ZOOKEEPER-', ''))
        link_type = row['link_type'].lower()
        
        # Only consider blocking relationships
        if from_key in tasks and to_key in tasks:
            if 'block' in link_type or 'parent' in link_type or 'subtask' in link_type:
                if from_key not in tasks[to_key]['dependencies']:
                    tasks[to_key]['dependencies'].append(from_key)
    
    # Calculate critical path using level-by-level scheduling
    # Group by topo_level and assign times sequentially
    level_end_times = {}
    
    for task_id, task in tasks.items():
        level = task['topo_level']
        
        # Base time: end of previous level + task duration
        if level in level_end_times:
            base_start = level_end_times[level]
        else:
            base_start = 0
        
        # Check dependencies within same level
        max_dep_end = 0
        for dep_id in task['dependencies']:
            if dep_id in tasks:
                dep_task = tasks[dep_id]
                if dep_task['topo_level'] == level:
                    max_dep_end = max(max_dep_end, dep_task['end'])
        
        task['start'] = max(base_start, max_dep_end)
        task['end'] = task['start'] + task['duration']
        
        # Update level end time
        level_end_times[level] = max(level_end_times.get(level, 0), task['end'])
    
    return tasks


def create_gantt_chart(tasks, output_file=None, max_tasks=200):
    """
    Create Gantt chart visualization.
    
    Parameters:
    - tasks: dict of task information
    - output_file: path to save chart (PNG/PDF)
    - max_tasks: max tasks to display (will sample if exceeded)
    """
    
    # Prepare data for plotting
    task_list = list(tasks.values())
    
    # Sort by start time, then by assignee
    task_list.sort(key=lambda x: (x['start'], x['assignee']))
    
    # Sample if too many tasks
    if len(task_list) > max_tasks:
        # Keep critical path tasks + sample others
        critical_tasks = [t for t in task_list if t['dependencies']]
        other_tasks = [t for t in task_list if not t['dependencies']]
        
        sample_indices = np.random.choice(len(other_tasks), 
                                         size=max_tasks - len(critical_tasks),
                                         replace=False)
        sampled_others = [other_tasks[i] for i in sorted(sample_indices)]
        task_list = critical_tasks + sampled_others
        task_list.sort(key=lambda x: (x['start'], x['assignee']))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(10, len(task_list) * 0.15)))
    
    # Define colors for assignees
    assignees = sorted(set(t['assignee'] for t in task_list))
    colors = plt.cm.tab20(np.linspace(0, 1, len(assignees)))
    assignee_colors = {assignee: colors[i] for i, assignee in enumerate(assignees)}
    
    # Plot bars
    y_pos = np.arange(len(task_list))
    
    for i, task in enumerate(task_list):
        # Bar for task duration
        ax.barh(i, 
                task['duration'], 
                left=task['start'],
                height=0.6,
                color=assignee_colors[task['assignee']],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8)
        
        # Add task label (ID, priority)
        label = f"#{task['id']} ({task['priority'][:3]})"
        ax.text(task['start'] + task['duration']/2, 
                i, 
                label,
                ha='center', 
                va='center',
                fontsize=7,
                weight='bold')
    
    # Add dependency arrows
    for task in task_list:
        if task['dependencies']:
            task_idx = task_list.index(task)
            for dep_id in task['dependencies']:
                if dep_id in tasks:
                    dep_task = tasks[dep_id]
                    try:
                        dep_idx = task_list.index(dep_task)
                        
                        # Draw arrow from dependency to task
                        ax.annotate('',
                                   xy=(task['start'], task_idx),
                                   xytext=(dep_task['end'], dep_idx),
                                   arrowprops=dict(arrowstyle='->', 
                                                  lw=0.5,
                                                  color='gray',
                                                  alpha=0.5,
                                                  connectionstyle="arc3,rad=0.1"))
                    except ValueError:
                        pass  # Dependency not in display list
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{t['id']}" for t in task_list], fontsize=8)
    ax.set_xlabel('Time (Hours)', fontsize=11, weight='bold')
    ax.set_ylabel('Tasks', fontsize=11, weight='bold')
    ax.set_title(f'Project Gantt Chart - {len(task_list)} Tasks (Colored by Assignee)', 
                 fontsize=13, weight='bold', pad=20)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add legend for assignees
    legend_patches = [mpatches.Patch(color=assignee_colors[assignee], 
                                     label=assignee, 
                                     alpha=0.8)
                     for assignee in assignees]
    ax.legend(handles=legend_patches, 
             loc='upper left', 
             bbox_to_anchor=(1.01, 1),
             fontsize=9,
             ncol=1)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Gantt chart saved: {output_file}")
    
    plt.show()


def create_gantt_by_assignee(tasks, output_file=None):
    """
    Create Gantt chart grouped and colored by assignee.
    """
    
    # Group tasks by assignee
    assignee_groups = {}
    for task_id, task in tasks.items():
        assignee = task['assignee']
        if assignee not in assignee_groups:
            assignee_groups[assignee] = []
        assignee_groups[assignee].append(task)
    
    # Sort assignees
    assignees = sorted(assignee_groups.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    priority_colors = {
        'Blocker': '#FF0000',
        'Critical': '#FF6600',
        'Major': '#FFCC00',
        'Minor': '#99CCFF',
        'Trivial': '#CCCCCC'
    }
    
    y_pos = 0
    y_labels = []
    y_ticks = []
    assignee_lines = []
    
    # Plot for each assignee
    for assignee in assignees:
        assignee_lines.append(y_pos)
        
        # Sort tasks by start time
        tasks_sorted = sorted(assignee_groups[assignee], 
                            key=lambda x: x['start'])
        
        for task in tasks_sorted:
            # Color by priority
            priority = task['priority'].split('/')[0]  # Get first part if composite
            color = priority_colors.get(priority, '#CCCCCC')
            
            # Bar
            ax.barh(y_pos, 
                   task['duration'],
                   left=task['start'],
                   height=0.8,
                   color=color,
                   edgecolor='black',
                   linewidth=0.5,
                   alpha=0.7)
            
            # Label
            label = f"#{task['id']}"
            ax.text(task['start'] + task['duration']/2,
                   y_pos,
                   label,
                   ha='center',
                   va='center',
                   fontsize=7,
                   weight='bold')
            
            y_pos += 1
        
        y_ticks.append((y_pos - 0.5 - len(assignee_groups[assignee])/2, assignee))
    
    # Formatting
    ax.set_ylim(-1, y_pos)
    ax.set_xlabel('Time (Hours)', fontsize=11, weight='bold')
    ax.set_title('Project Gantt Chart - Grouped by Assignee (Colored by Priority)', 
                fontsize=13, weight='bold', pad=20)
    
    # Set y-axis labels
    ax.set_yticks([tick[0] for tick in y_ticks])
    ax.set_yticklabels([tick[1] for tick in y_ticks], fontsize=9)
    
    # Add vertical lines between assignees
    for line_pos in assignee_lines[1:]:
        ax.axhline(y=line_pos - 0.5, color='gray', linestyle='-', linewidth=2, alpha=0.5)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add legend for priority
    legend_patches = [mpatches.Patch(color=priority_colors[priority], 
                                    label=priority,
                                    alpha=0.7)
                     for priority in ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']]
    ax.legend(handles=legend_patches, 
             loc='upper right',
             fontsize=9)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Gantt chart (by assignee) saved: {output_file}")
    
    plt.show()


def create_timeline_summary(tasks, output_file=None):
    """
    Create a summary timeline chart showing project milestones.
    """
    
    # Calculate timeline statistics
    all_tasks = list(tasks.values())
    project_start = min(t['start'] for t in all_tasks)
    project_end = max(t['end'] for t in all_tasks)
    
    # Group by topo level and calculate stats
    level_stats = {}
    for task in all_tasks:
        level = task['topo_level']
        if level not in level_stats:
            level_stats[level] = {
                'start': task['start'],
                'end': task['end'],
                'duration': 0,
                'task_count': 0,
                'tasks': []
            }
        level_stats[level]['start'] = min(level_stats[level]['start'], task['start'])
        level_stats[level]['end'] = max(level_stats[level]['end'], task['end'])
        level_stats[level]['task_count'] += 1
        level_stats[level]['tasks'].append(task)
    
    # Calculate duration
    for level_data in level_stats.values():
        level_data['duration'] = level_data['end'] - level_data['start']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot level timeline
    levels = sorted(level_stats.keys())
    colors_level = plt.cm.viridis(np.linspace(0, 1, len(levels)))
    
    for i, level in enumerate(levels):
        stats = level_stats[level]
        ax.barh(i,
               stats['duration'],
               left=stats['start'],
               height=0.6,
               color=colors_level[i],
               edgecolor='black',
               linewidth=1.5,
               alpha=0.8)
        
        # Add text label
        label_text = f"Level {level}\n{stats['task_count']} tasks"
        ax.text(stats['start'] + stats['duration']/2,
               i,
               label_text,
               ha='center',
               va='center',
               fontsize=10,
               weight='bold',
               color='white')
    
    # Formatting
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([f"Level {level}" for level in levels], fontsize=10)
    ax.set_xlabel('Time (Hours)', fontsize=11, weight='bold')
    ax.set_title(f'Project Timeline Summary (Total Duration: {project_end:.1f}h)', 
                fontsize=13, weight='bold', pad=20)
    
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Timeline summary saved: {output_file}")
    
    plt.show()


def print_statistics(tasks):
    """Print project statistics."""
    all_tasks = list(tasks.values())
    
    print("\n" + "="*70)
    print("PROJECT GANTT STATISTICS".center(70))
    print("="*70)
    
    print(f"\n✓ Total Tasks: {len(all_tasks)}")
    print(f"✓ Total Duration: {max(t['end'] for t in all_tasks):.1f} hours")
    print(f"✓ Number of Assignees: {len(set(t['assignee'] for t in all_tasks))}")
    
    # Group by level
    level_stats = {}
    for task in all_tasks:
        level = task['topo_level']
        if level not in level_stats:
            level_stats[level] = []
        level_stats[level].append(task)
    
    print(f"\n--- By Topo Level ---")
    for level in sorted(level_stats.keys()):
        tasks_in_level = level_stats[level]
        total_hours = sum(t['duration'] for t in tasks_in_level)
        print(f"  Level {level}: {len(tasks_in_level)} tasks, {total_hours:.1f} total hours")
    
    # Top assignees
    assignee_load = {}
    for task in all_tasks:
        assignee = task['assignee']
        if assignee not in assignee_load:
            assignee_load[assignee] = {'count': 0, 'hours': 0}
        assignee_load[assignee]['count'] += 1
        assignee_load[assignee]['hours'] += task['duration']
    
    print(f"\n--- Top 10 Assignees by Task Count ---")
    for assignee, stats in sorted(assignee_load.items(), 
                                  key=lambda x: x[1]['count'], 
                                  reverse=True)[:10]:
        print(f"  {assignee}: {stats['count']} tasks, {stats['hours']:.1f} hours")
    
    # Critical path
    critical_tasks = [t for t in all_tasks if t['dependencies']]
    print(f"\n✓ Critical Path Tasks (with dependencies): {len(critical_tasks)}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Gantt chart visualization for project assignments'
    )
    parser.add_argument('--assignment', required=True, help='Assignment CSV file path')
    parser.add_argument('--links', required=True, help='Issue links CSV file path')
    parser.add_argument('--output-dir', default='output', help='Output directory for charts')
    parser.add_argument('--max-tasks', type=int, default=200, 
                       help='Max tasks to display in detail chart')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    assignment_df, links_df = load_data(args.assignment, args.links)
    
    # Calculate schedule
    print("Calculating task schedule...")
    tasks = calculate_task_schedule(assignment_df, links_df)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Print statistics
    print_statistics(tasks)
    
    # Generate charts
    print("Generating Gantt charts...")
    
    # Chart 1: Detailed chart with dependencies
    chart1_path = output_dir / 'gantt_detailed.png'
    create_gantt_chart(tasks, output_file=str(chart1_path), max_tasks=args.max_tasks)
    
    # Chart 2: Grouped by assignee
    chart2_path = output_dir / 'gantt_by_assignee.png'
    create_gantt_by_assignee(tasks, output_file=str(chart2_path))
    
    # Chart 3: Timeline summary
    chart3_path = output_dir / 'gantt_timeline_summary.png'
    create_timeline_summary(tasks, output_file=str(chart3_path))
    
    print(f"\n✓ All charts generated in: {output_dir}")


if __name__ == '__main__':
    main()
