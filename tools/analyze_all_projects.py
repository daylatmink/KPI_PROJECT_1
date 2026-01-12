#!/usr/bin/env python
"""
Query MongoDB to get task count by project and find suitable projects for testing.
"""

import pandas as pd
from pymongo import MongoClient
from pathlib import Path
import json


def get_projects_from_mongodb():
    """Query MongoDB to get all projects and their task counts."""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['jira_db']  # Adjust database name if needed
        issues_collection = db['issues']  # Adjust collection name if needed
        
        # Aggregate task count by project
        pipeline = [
            {
                '$group': {
                    '_id': '$project_key',
                    'task_count': {'$sum': 1}
                }
            },
            {
                '$sort': {'task_count': -1}
            }
        ]
        
        results = list(issues_collection.aggregate(pipeline))
        
        data = []
        for result in results:
            data.append({
                'Project': result['_id'],
                'Tasks': result['task_count']
            })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        print("Trying alternative method...")
        return None


def get_projects_from_file(chunk_size=10000):
    """Read all_issues.csv in chunks to avoid memory overload."""
    try:
        print("Reading all_issues.csv in chunks...")
        
        project_counts = {}
        
        # Read CSV in chunks
        for i, chunk in enumerate(pd.read_csv(
            'data/raw/all_issues.csv',
            chunksize=chunk_size,
            usecols=['project_key']
        )):
            if i % 5 == 0:
                print(f"  Processed {(i+1) * chunk_size} rows...")
            
            # Count by project in this chunk
            project_counts.update(
                chunk['project_key'].value_counts().to_dict()
            )
        
        # Aggregate counts
        for chunk in pd.read_csv(
            'data/raw/all_issues.csv',
            chunksize=chunk_size,
            usecols=['project_key']
        ):
            chunk_counts = chunk['project_key'].value_counts()
            for project, count in chunk_counts.items():
                project_counts[project] = project_counts.get(project, 0) + count
        
        data = [
            {'Project': project, 'Tasks': count}
            for project, count in sorted(
                project_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None


def main():
    print("\n" + "="*70)
    print("ANALYZING ALL PROJECTS FROM DATA SOURCE".center(70))
    print("="*70 + "\n")
    
    # Try MongoDB first
    print("Attempting to connect to MongoDB...")
    df = get_projects_from_mongodb()
    
    # Fallback to file reading
    if df is None or df.empty:
        print("Querying from CSV file...")
        df = get_projects_from_file()
    
    if df is None or df.empty:
        print("❌ Could not retrieve project data")
        return
    
    # Display results
    print("\n" + "-"*70)
    print("ALL PROJECTS BY TASK COUNT".center(70))
    print("-"*70 + "\n")
    
    print(df.to_string(index=False))
    
    # Analyze by size category
    print("\n\n" + "-"*70)
    print("PROJECTS BY SIZE CATEGORY".center(70))
    print("-"*70 + "\n")
    
    tiny = df[df['Tasks'] < 100]
    small = df[(df['Tasks'] >= 100) & (df['Tasks'] < 500)]
    medium = df[(df['Tasks'] >= 500) & (df['Tasks'] < 2000)]
    large = df[(df['Tasks'] >= 2000) & (df['Tasks'] < 5000)]
    huge = df[df['Tasks'] >= 5000]
    
    categories = [
        ('🟢 TINY (< 100 tasks)', tiny),
        ('🟢 SMALL (100-500 tasks)', small),
        ('🟡 MEDIUM (500-2000 tasks)', medium),
        ('🟠 LARGE (2000-5000 tasks)', large),
        ('🔴 HUGE (> 5000 tasks)', huge)
    ]
    
    for label, group in categories:
        if not group.empty:
            print(f"\n{label}:")
            for _, row in group.iterrows():
                print(f"  • {row['Project']}: {row['Tasks']} tasks")
        else:
            print(f"\n{label}: None")
    
    # Recommend for testing
    print("\n\n" + "-"*70)
    print("RECOMMENDATION FOR GANTT TESTING".center(70))
    print("-"*70 + "\n")
    
    if not medium.empty:
        print("✓ MEDIUM SIZE (500-2000 tasks) - BEST FOR VISUALIZATION:")
        for _, row in medium.iterrows():
            print(f"  → {row['Project']}: {row['Tasks']} tasks")
    elif not small.empty:
        print("✓ SMALL SIZE (100-500 tasks) - GOOD FOR VISUALIZATION:")
        for _, row in small.iterrows():
            print(f"  → {row['Project']}: {row['Tasks']} tasks")
    elif not tiny.empty:
        print("✓ TINY SIZE (< 100 tasks) - PERFECT FOR VISUALIZATION:")
        for _, row in tiny.iterrows():
            print(f"  → {row['Project']}: {row['Tasks']} tasks")
    
    # Save to file
    output_file = Path('output/all_projects_statistics.csv')
    df.to_csv(output_file, index=False)
    print(f"\n\n✓ Complete statistics saved to: {output_file}")


if __name__ == '__main__':
    main()
