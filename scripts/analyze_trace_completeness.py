#!/usr/bin/env python3
"""
Analyze trace CSV files for completeness - check if all start events have completion events.
"""

import csv
import sys
from collections import defaultdict

def analyze_trace(csv_file):
    """Analyze a trace CSV file for completeness."""
    start_events = []
    completion_events = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('complete') == 'COMPLETE_ALL':
                completion_events.append(row)
            else:
                start_events.append(row)
    
    # Group by function name
    start_by_func = defaultdict(list)
    comp_by_func = defaultdict(list)
    
    for event in start_events:
        fname = (event.get('fname') or '').strip()
        if fname:
            start_by_func[fname].append(event)

    for event in completion_events:
        fname = (event.get('fname') or '').strip()
        token = (event.get('token') or '').strip()
        if fname and token:
            comp_by_func[fname].append((token, event))
    
    print("="*70)
    print(f"TRACE COMPLETENESS ANALYSIS: {csv_file}")
    print("="*70)
    print(f"\nTotal start events: {len(start_events):,}")
    print(f"Total completion events: {len(completion_events):,}")
    unmatched_total = len(start_events) - len(completion_events)
    print(f"Unmatched start events: {unmatched_total:,}")
    
    if unmatched_total > 0:
        print(f"\n⚠️  WARNING: Trace appears INCOMPLETE!")
        print(f"   {unmatched_total} operations started but never completed.")
        print(f"   This suggests tracing was stopped before all operations finished.")
    else:
        print("\n✓ Trace appears COMPLETE - all start events have completions.")
    
    print("\n" + "="*70)
    print("BREAKDOWN BY FUNCTION")
    print("="*70)
    
    unmatched_functions = []
    for fname in sorted(start_by_func.keys()):
        starts = len(start_by_func[fname])
        comps = len(comp_by_func[fname])
        unmatched = starts - comps
        status = "⚠️  INCOMPLETE" if unmatched > 0 else "✓ Complete"
        print(f"\n{fname}:")
        print(f"  Start events: {starts:,}")
        print(f"  Completion events: {comps:,}")
        print(f"  Status: {status}")
        if unmatched > 0:
            print(f"  Missing completions: {unmatched}")
            unmatched_functions.append((fname, unmatched))
    
    # Show last few unmatched events
    if unmatched_total > 0:
        print("\n" + "="*70)
        print("LAST 10 START EVENTS (likely unmatched)")
        print("="*70)
        for i, event in enumerate(start_events[-10:], 1):
            fname = event.get('fname') or ''
            timestamp = event.get('timestamp') or ''
            grid = event.get('grid') or ''
            block = event.get('block') or ''
            print(f"{i:2d}. {fname:25s} at {timestamp:15s} grid={grid:10s} block={block}")

        print("\n" + "="*70)
        print("LAST 10 COMPLETION EVENTS")
        print("="*70)
        for i, event in enumerate(completion_events[-10:], 1):
            fname = event.get('fname') or ''
            token = event.get('token') or ''
            timestamp = event.get('timestamp') or ''
            latency = event.get('latency') or ''
            print(f"{i:2d}. {fname:25s} token={token:6s} latency={latency:8s}ms at {timestamp}")
    
    # Calculate completion percentage
    if len(start_events) > 0:
        completion_rate = (len(completion_events) / len(start_events)) * 100
        print(f"\n" + "="*70)
        print(f"COMPLETION RATE: {completion_rate:.2f}%")
        print("="*70)
    
    return unmatched_total > 0, unmatched_functions

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_trace_completeness.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    is_incomplete, unmatched = analyze_trace(csv_file)
    
    if is_incomplete:
        print("\n" + "="*70)
        print("RECOMMENDATIONS:")
        print("="*70)
        print("1. Ensure tracing continues until all CUDA operations complete")
        print("2. Wait for vLLM request to fully finish before stopping trace")
        print("3. Check if tracing script stops prematurely")
        print("4. Consider adding a delay before stopping trace")
        sys.exit(1)
    else:
        sys.exit(0)

