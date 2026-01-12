#!/usr/bin/env python
"""
Extract raw project data from all_issues.csv
"""

import pandas as pd
from pathlib import Path
import sys


def extract_project_data(project_key='WODEN', chunk_size=10000):
    """Extract project data from all_issues.csv"""
    
    print(f"Extracting {project_key} from all_issues.csv...")
    print("(This may take a few moments...)\n")
    
    project_dir = Path(f'projects/{project_key}')
    project_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = project_dir / 'issues.csv'
    
    # Read in chunks and filter
    chunks = []
    total_rows = 0
    matched_rows = 0
    
    for i, chunk in enumerate(pd.read_csv('data/raw/all_issues.csv', chunksize=chunk_size)):
        total_rows += len(chunk)
        
        # Filter for project
        project_data = chunk[chunk['project_key'] == project_key]
        
        if len(project_data) > 0:
            chunks.append(project_data)
            matched_rows += len(project_data)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {total_rows:,} rows, found {matched_rows} {project_key} issues...")
    
    if not chunks:
        print(f"❌ No issues found for {project_key}")
        return False
    
    # Combine and save
    df = pd.concat(chunks, ignore_index=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Extracted {len(df)} {project_key} issues")
    print(f"  Saved to: {output_file}\n")
    
    # Print statistics
    if 'priority' in df.columns:
        print("Priority distribution:")
        for priority, count in df['priority'].value_counts().items():
            print(f"  {priority}: {count}")
    
    if 'status' in df.columns:
        print("\nStatus distribution:")
        for status, count in df['status'].value_counts().items():
            print(f"  {status}: {count}")
    
    if 'assignee' in df.columns:
        print(f"\nTotal assignees: {df['assignee'].nunique()}")
    
    return True


if __name__ == '__main__':
    project = sys.argv[1] if len(sys.argv) > 1 else 'WODEN'
    extract_project_data(project)
