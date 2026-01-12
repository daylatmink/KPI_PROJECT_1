#!/usr/bin/env python
"""
Extract and process WODEN project from MongoDB.
"""

import pandas as pd
from pymongo import MongoClient
from pathlib import Path
import json


def extract_project_from_mongodb(project_key='WODEN'):
    """Extract project data from MongoDB."""
    
    print(f"Connecting to MongoDB...")
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['jira_db']
        issues_coll = db['issues']
        
        # Get all issues for this project
        print(f"Extracting {project_key} issues...")
        issues = list(issues_coll.find({'project_key': project_key}))
        print(f"  Found {len(issues)} issues")
        
        if not issues:
            return None, None, None
        
        # Convert to dataframe
        df_issues = pd.DataFrame(issues)
        
        # Get links if available
        links_coll = db['issue_links']
        links = list(links_coll.find({
            '$or': [
                {'from_issue_key': {'$regex': f'^{project_key}-'}},
                {'to_issue_key': {'$regex': f'^{project_key}-'}}
            ]
        }))
        print(f"  Found {len(links)} links")
        
        df_links = pd.DataFrame(links) if links else pd.DataFrame()
        
        return df_issues, df_links, len(issues)
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


def create_project_structure(project_key, df_issues):
    """Create minimal project structure for testing."""
    
    project_dir = Path(f'projects/{project_key}')
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create issues.csv
    issues_file = project_dir / 'issues.csv'
    df_issues.to_csv(issues_file, index=False)
    print(f"  ✓ {issues_file}")
    
    # Create issue_links.csv (dummy)
    links_file = project_dir / 'issue_links.csv'
    pd.DataFrame({
        'from_issue_key': [],
        'to_issue_key': [],
        'link_type': [],
        'raw_link_type': [],
        'direction': []
    }).to_csv(links_file, index=False)
    print(f"  ✓ {links_file}")
    
    # Create assignees.csv (unique from issues)
    assignees = df_issues['assignee'].dropna().unique() if 'assignee' in df_issues.columns else []
    assignees_df = pd.DataFrame({
        'assignee': assignees,
        'name': [f"Team Member {i+1}" for i in range(len(assignees))]
    })
    assignees_file = project_dir / 'assignees.csv'
    assignees_df.to_csv(assignees_file, index=False)
    print(f"  ✓ {assignees_file} ({len(assignees)} assignees)")
    
    return project_dir


def main():
    print("\n" + "="*70)
    print("EXTRACTING WODEN PROJECT FROM MONGODB".center(70))
    print("="*70 + "\n")
    
    # Extract from MongoDB
    df_issues, df_links, count = extract_project_from_mongodb('WODEN')
    
    if df_issues is None:
        print("❌ Failed to extract WODEN project")
        return
    
    # Create project structure
    print(f"\nCreating project structure for WODEN ({count} tasks)...")
    project_dir = create_project_structure('WODEN', df_issues)
    
    print(f"\n✓ WODEN project extracted successfully!")
    print(f"  Location: {project_dir}")
    
    # Summary
    if 'priority' in df_issues.columns:
        priority_counts = df_issues['priority'].value_counts()
        print(f"\n  Priority distribution:")
        for priority, count in priority_counts.items():
            print(f"    {priority}: {count}")
    
    if 'status' in df_issues.columns:
        status_counts = df_issues['status'].value_counts()
        print(f"\n  Status distribution:")
        for status, count in status_counts.items():
            print(f"    {status}: {count}")


if __name__ == '__main__':
    main()
