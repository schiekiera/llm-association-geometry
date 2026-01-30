#!/usr/bin/env python3
"""
Extract statistics from forced choice .out files.

Extracts the following statistics from Round 1:
- Initial generation: compliant/total, noncompliant
- After deterministic repair: remaining noncompliant
- After sampled retries: compliant/total, percentage, sampled_retries_used
"""

import re
from pathlib import Path
import pandas as pd
from datetime import datetime

def extract_stats_from_file(file_path):
    """Extract statistics from a single .out file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        stats = {
            'file': file_path.name,
            'model_dir': file_path.parent.name,
            'initial_compliant': None,
            'initial_total': None,
            'initial_noncompliant': None,
            'repair_remaining_noncompliant': None,
            'final_compliant': None,
            'final_total': None,
            'final_percentage': None,
            'sampled_retries_used': None,
        }
        
        # Pattern for initial generation
        initial_pattern = r'\[Round 1\] Initial generation: compliant=(\d+)/(\d+) noncompliant=(\d+)'
        initial_match = re.search(initial_pattern, content)
        if initial_match:
            stats['initial_compliant'] = int(initial_match.group(1))
            stats['initial_total'] = int(initial_match.group(2))
            stats['initial_noncompliant'] = int(initial_match.group(3))
        
        # Pattern for deterministic repair
        repair_pattern = r'\[Round 1\] After deterministic repair: remaining noncompliant=(\d+)'
        repair_match = re.search(repair_pattern, content)
        if repair_match:
            stats['repair_remaining_noncompliant'] = int(repair_match.group(1))
        
        # Pattern for sampled retries
        retries_pattern = r'\[Round 1\] After sampled retries: compliant=(\d+)/(\d+) \((\d+\.\d+)%\), sampled_retries_used=(\d+)'
        retries_match = re.search(retries_pattern, content)
        if retries_match:
            stats['final_compliant'] = int(retries_match.group(1))
            stats['final_total'] = int(retries_match.group(2))
            stats['final_percentage'] = float(retries_match.group(3))
            stats['sampled_retries_used'] = int(retries_match.group(4))
        
        return stats
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    """Extract statistics from all .out files in forced choice subdirectories."""
    base_dir = Path("logs/02_get_behavioral_associations/01_forced_choice")
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    all_stats = []
    
    # Get all subdirectories (one level only)
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            print(f"Processing directory: {subdir.name}")
            
            # Find .out files in this subdirectory (not recursive)
            out_files = list(subdir.glob("*.out"))
            
            for out_file in out_files:
                print(f"  Processing file: {out_file.name}")
                stats = extract_stats_from_file(out_file)
                if stats:
                    all_stats.append(stats)
    
    if not all_stats:
        print("No statistics extracted.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_stats)
    
    # Calculate derived statistics
    df['initial_compliance_rate'] = (df['initial_compliant'] / df['initial_total'] * 100).round(1)
    df['repair_effectiveness'] = ((df['initial_noncompliant'] - df['repair_remaining_noncompliant']) / df['initial_noncompliant'] * 100).round(1)
    df['retry_effectiveness'] = ((df['repair_remaining_noncompliant'] - (df['final_total'] - df['final_compliant'])) / df['repair_remaining_noncompliant'] * 100).round(1)
    
    # Sort by model directory
    df = df.sort_values('model_dir')
    
    # Display results
    print("\n" + "="*80)
    print("FORCED CHOICE GENERATION STATISTICS")
    print("="*80)
    
    for _, row in df.iterrows():
        print(f"\nModel: {row['model_dir']}")
        print(f"File: {row['file']}")
        print(f"Initial Generation:")
        print(f"  - Compliant: {row['initial_compliant']:,}/{row['initial_total']:,} ({row['initial_compliance_rate']:.1f}%)")
        print(f"  - Noncompliant: {row['initial_noncompliant']:,}")
        print(f"Deterministic Repair:")
        print(f"  - Remaining noncompliant: {row['repair_remaining_noncompliant']:,}")
        print(f"  - Repair effectiveness: {row['repair_effectiveness']:.1f}%")
        print(f"Sampled Retries:")
        print(f"  - Final compliant: {row['final_compliant']:,}/{row['final_total']:,} ({row['final_percentage']:.1f}%)")
        print(f"  - Retries used: {row['sampled_retries_used']:,}")
        print(f"  - Retry effectiveness: {row['retry_effectiveness']:.1f}%")
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/02_behavioral_associations/01_forced_choice/summary/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"forced_choice_generation_stats_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved detailed statistics to: {output_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    summary_cols = ['initial_compliance_rate', 'repair_effectiveness', 'retry_effectiveness', 'final_percentage']
    summary = df[['model_dir'] + summary_cols].set_index('model_dir')
    
    print(summary.to_string(float_format="%.1f"))
    
    # Save summary
    summary_file = output_dir / f"forced_choice_generation_summary_{timestamp}.csv"
    summary.to_csv(summary_file)
    print(f"\nSaved summary statistics to: {summary_file}")

if __name__ == "__main__":
    main()