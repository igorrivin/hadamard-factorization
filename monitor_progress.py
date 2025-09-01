#!/usr/bin/env python3
"""
Monitor progress of F_3 computation by checking saved files.
Run this in a separate Colab cell or notebook.
"""

import os
import time
import numpy as np
from datetime import datetime, timedelta

def check_progress():
    """Check progress by looking at saved files."""
    print(f"\n{'='*60}")
    print(f"F_3 Computation Progress - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check for rank-2 checkpoint
    if os.path.exists('f3_rank2_checkpoint.npy'):
        rank2_data = np.load('f3_rank2_checkpoint.npy')
        n_rank2 = len(rank2_data)
        print(f"✓ Rank-2 matrices found: {n_rank2:,}")
        print(f"  Expected: ~811,200")
        print(f"  Accuracy: {n_rank2/811200*100:.1f}%")
        
        # Estimate file size for product computation
        n_products = n_rank2 * (n_rank2 + 1) // 2
        print(f"\n  Products to compute: {n_products:,}")
        print(f"  (That's {n_products/1e9:.1f} billion products!)")
    else:
        print("⏳ Still searching for rank-2 matrices...")
    
    # Check for other output files
    files_to_check = [
        'f3_rank2_matrices_temp.npy',
        'f3_product_map_temp.pkl',
        'f3_rank2_matrices_safe.npy',
        'f3_product_map_safe.pkl',
        'f3_results_safe.pkl'
    ]
    
    print("\nOther files:")
    for fname in files_to_check:
        if os.path.exists(fname):
            size_mb = os.path.getsize(fname) / (1024 * 1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(fname))
            age = datetime.now() - mtime
            print(f"  {fname}: {size_mb:.1f} MB (updated {age} ago)")
    
    # Memory usage (if available)
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"\nMemory usage: {mem_info.rss / (1024**3):.1f} GB")
    except:
        pass
    
    # GPU usage (if available)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
            print(f"GPU usage: {gpu_util}% ({mem_used} MB / {mem_total} MB)")
    except:
        pass

def monitor_loop(interval_minutes=10):
    """Monitor progress every N minutes."""
    print("Starting progress monitor...")
    print(f"Will check every {interval_minutes} minutes")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            check_progress()
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

if __name__ == "__main__":
    # Single check
    check_progress()
    
    # Or run continuous monitoring:
    # monitor_loop(interval_minutes=10)