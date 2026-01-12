#!/usr/bin/env python3
"""
Display cost and effectiveness KPIs for all algorithms
"""
import json

files = {
    'HS': 'projects/ZOOKEEPER/hs_score.json',
    'IHS': 'projects/ZOOKEEPER/ihs_score.json',
    'GHS': 'projects/ZOOKEEPER/ghs_score.json',
    'MOHS': 'projects/ZOOKEEPER/mohs_score.json'
}

print('=' * 90)
print('COMPARISON OF 4 ALGORITHMS - Cost and Effectiveness KPIs')
print('=' * 90)

for algo_name, file_path in files.items():
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print()
    print(algo_name + ' Algorithm:')
    print('-' * 90)
    print('  Effectiveness Metrics:')
    print('    - Average Effectiveness Score:', f"{data.get('avg_total_score', 0):.4f}", '(0-1 scale, higher is better)')
    print('    - Utilization:', f"{data.get('utilization', 0):.2%}")
    print('    - Skill Match Rate:', f"{data.get('skill_match_rate', 0):.2%}")
    print()
    print('  Cost Metrics (NEW KPIs):')
    print('    - Total Project Cost:', f"${data.get('total_project_cost_usd', 0):,.2f}")
    print('    - Cost per Task:', f"${data.get('cost_per_task_usd', 0):.2f}")
    print('    - Cost per Hour:', f"${data.get('cost_per_hour_usd', 0):.2f}")
    print()
    print('  Cost-Effectiveness Metric:')
    efficiency = data.get('efficiency_score_per_1000_usd', 0)
    print('    - Efficiency Score per 1000 USD:', f"{efficiency:.4f}")
    print('      (How much effectiveness you get per 1000 dollars spent)')

print()
print('=' * 90)
print('HOW TO USE THESE KPIs:')
print('=' * 90)
print('1. CHOOSE ALGORITHM BASED ON YOUR PRIORITY:')
print('   - If you want BEST QUALITY: Pick algorithm with highest Effectiveness Score')
print('   - If you want LOWEST COST: Pick algorithm with lowest Total Project Cost')
print('   - If you want BEST VALUE: Pick algorithm with highest Efficiency Score/1000 USD')
print()
print('2. BALANCE COST vs EFFECTIVENESS:')
print('   - Look at: Effectiveness Score vs Total Project Cost')
print('   - The efficiency_score_per_1000_usd already does this calculation for you')
print('   - Higher number = Better quality for the money spent')
print()
print('3. ALL KPIs ARE NOW REFERENCE METRICS (not part of scoring):')
print('   - The algorithms are scored on: Skill Match, Priority, Workload Balance, Skill Dev')
print('   - Cost metrics are TRACKED SEPARATELY for decision-making')
print('=' * 90)
