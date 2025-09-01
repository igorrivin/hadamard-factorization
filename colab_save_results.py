#!/usr/bin/env python3
"""
Save F_3 computation results to persistent storage in Colab.
Run this in a cell to set up automatic saving.
"""

import os
import shutil
from google.colab import drive
from datetime import datetime

def setup_google_drive_saving():
    """Mount Google Drive and create results directory."""
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'/content/drive/MyDrive/F3_Hadamard_Results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"✓ Results will be saved to: {results_dir}")
    return results_dir

def save_checkpoint_to_drive(local_file, drive_dir):
    """Copy a file to Google Drive."""
    if os.path.exists(local_file):
        dest_path = os.path.join(drive_dir, os.path.basename(local_file))
        shutil.copy2(local_file, dest_path)
        print(f"✓ Saved {local_file} to Drive")
        return True
    return False

def setup_auto_save(drive_dir, check_interval=300):
    """Set up automatic saving every N seconds."""
    import time
    import threading
    
    files_to_monitor = [
        'f3_rank2_checkpoint.npy',
        'f3_rank2_matrices_temp.npy',
        'f3_product_map_temp.pkl',
        'f3_rank2_matrices_safe.npy',
        'f3_product_map_safe.pkl',
        'f3_results_safe.pkl',
        # Add any other output files
    ]
    
    def auto_save_loop():
        while True:
            time.sleep(check_interval)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Auto-saving to Drive...")
            saved_count = 0
            for fname in files_to_monitor:
                if save_checkpoint_to_drive(fname, drive_dir):
                    saved_count += 1
            if saved_count > 0:
                print(f"✓ Saved {saved_count} files to Drive")
    
    # Start background thread
    save_thread = threading.Thread(target=auto_save_loop, daemon=True)
    save_thread.start()
    print(f"✓ Auto-save enabled (every {check_interval//60} minutes)")

# Alternative: Save directly to Drive in your main code
def modify_code_for_direct_saving():
    """
    Add these lines to f3_superfast.py to save directly to Drive:
    """
    example_code = '''
# At the beginning of main():
from google.colab import drive
drive.mount('/content/drive')
results_dir = '/content/drive/MyDrive/F3_Results'
os.makedirs(results_dir, exist_ok=True)

# Replace save lines like:
np.save('f3_rank2_checkpoint.npy', rank2_ints)

# With:
np.save(os.path.join(results_dir, 'f3_rank2_checkpoint.npy'), rank2_ints)
'''
    return example_code

# Quick setup script
if __name__ == "__main__":
    # Option 1: Set up Google Drive saving
    print("Setting up Google Drive saving...")
    drive_dir = setup_google_drive_saving()
    
    # Option 2: Enable auto-save
    print("\nEnabling auto-save...")
    setup_auto_save(drive_dir, check_interval=600)  # Every 10 minutes
    
    print("\n✓ All set! Your results will be automatically saved to Google Drive")
    print(f"✓ Check your Drive folder: {drive_dir}")
    
    # Also show how to manually save
    print("\nTo manually save a file anytime, run:")
    print(f"save_checkpoint_to_drive('filename.npy', '{drive_dir}')")