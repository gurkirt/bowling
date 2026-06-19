#!/usr/bin/env python3
"""
analyze_ablation_results.py - Analyze and compare ablation study results
"""

import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiment_results(exp_dir):
    """Load results from an experiment directory"""
    results = {
        'name': exp_dir.name,
        'best_val_acc': None,
        'best_val_f1': None,
        'final_metrics': None,
    }
    
    # Try to load metrics.json if it exists
    metrics_file = exp_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            results.update(metrics)
    
    return results

def analyze_ablation_studies():
    """Analyze all ablation study experiments"""
    trainings_dir = Path('trainings')
    
    if not trainings_dir.exists():
        print(f"Error: {trainings_dir} directory not found")
        return
    
    # Find all exp* directories
    exp_dirs = sorted([d for d in trainings_dir.iterdir() if d.is_dir() and d.name.startswith('exp')])
    
    if not exp_dirs:
        print("No experiment directories found")
        return
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS ANALYSIS")
    print(f"{'='*80}\n")
    
    results_list = []
    for exp_dir in exp_dirs:
        results = load_experiment_results(exp_dir)
        results_list.append(results)
        
        print(f"Experiment: {results['name']}")
        if results['best_val_acc']:
            print(f"  Best Val Accuracy: {results['best_val_acc']:.4f}")
        if results['best_val_f1']:
            print(f"  Best Val F1:       {results['best_val_f1']:.4f}")
        print()
    
    # Create comparison summary
    print(f"{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}\n")
    
    if results_list:
        df = pd.DataFrame(results_list)
        print(df.to_string(index=False))
        print()
        
        # Identify best experiment
        if 'best_val_acc' in df.columns:
            best_idx = df['best_val_acc'].idxmax()
            print(f"\n✓ Best by Validation Accuracy: {results_list[best_idx]['name']}")
        
        if 'best_val_f1' in df.columns:
            best_idx = df['best_val_f1'].idxmax()
            print(f"✓ Best by F1 Score: {results_list[best_idx]['name']}\n")

def monitor_experiment_progress():
    """Monitor real-time progress of experiments"""
    log_dir = Path('ablation_logs')
    
    if not log_dir.exists():
        print(f"Log directory {log_dir} not found")
        return
    
    log_files = sorted(log_dir.glob('exp*.log'))
    
    if not log_files:
        print("No experiment logs found yet")
        return
    
    print(f"\n{'='*80}")
    print("EXPERIMENT PROGRESS")
    print(f"{'='*80}\n")
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                # Show last few relevant lines
                recent_lines = [l.strip() for l in lines[-10:] if l.strip()]
                print(f"\n{log_file.name}:")
                for line in recent_lines[-3:]:
                    print(f"  {line}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'progress':
            monitor_experiment_progress()
        else:
            print("Usage: python analyze_ablation_results.py [progress]")
    else:
        analyze_ablation_studies()
