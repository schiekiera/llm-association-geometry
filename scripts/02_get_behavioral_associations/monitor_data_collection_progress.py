import os
import glob
import re
import pandas as pd
from datetime import datetime
import time

# =============================================================================
# Configuration
# =============================================================================

LOG_BASE = "logs/02_get_behavioral_associations"

# Models and their log patterns
# Type: 'FC' (Forced Choice) or 'FA' (Free Association)
TASKS = [
    # Forced Choice
    {"type": "FC", "model": "essential", "subfolder": "01_forced_choice/essential", "pattern": "essential_hier_sim-*.out"},
    {"type": "FC", "model": "falcon", "subfolder": "01_forced_choice/falcon", "pattern": "falcon_hier_sim-*.out"},
    {"type": "FC", "model": "gemma", "subfolder": "01_forced_choice/gemma", "pattern": "gemma_hier_sim-*.out"},
    {"type": "FC", "model": "llama", "subfolder": "01_forced_choice/llama", "pattern": "llama_hier_sim-*.out"},
    {"type": "FC", "model": "mistral", "subfolder": "01_forced_choice/mistral", "pattern": "mistral_hier_sim-*.out"},
    {"type": "FC", "model": "mistral_nemo", "subfolder": "01_forced_choice/mistral_nemo", "pattern": "mistral_nemo_hier_sim-*.out"},
    {"type": "FC", "model": "phi4", "subfolder": "01_forced_choice/phi4", "pattern": "phi4_hier_sim-*.out"},
    {"type": "FC", "model": "qwen", "subfolder": "01_forced_choice/qwen", "pattern": "qwen_hier_sim-*.out"},

    # Free Association
    {"type": "FA", "model": "falcon", "subfolder": "02_free_associations/falcon", "pattern": "falcon_FA-*.out"},
    {"type": "FA", "model": "gemma", "subfolder": "02_free_associations/gemma", "pattern": "gemma_free_associations-*.out"},
    {"type": "FA", "model": "llama", "subfolder": "02_free_associations/llama", "pattern": "llama_FA-*.out"},
    {"type": "FA", "model": "mistral", "subfolder": "02_free_associations/mistral", "pattern": "mistral_FA-*.out"},
    {"type": "FA", "model": "mistral_nemo", "subfolder": "02_free_associations/mistral_nemo", "pattern": "mist_nemo_FA-*.out"},
    {"type": "FA", "model": "phi4", "subfolder": "02_free_associations/phi4", "pattern": "phi4_FA-*.out"},
    {"type": "FA", "model": "qwen", "subfolder": "02_free_associations/qwen", "pattern": "qwen_free_associations-*.out"},
    {"type": "FA", "model": "rnj", "subfolder": "02_free_associations/rnj", "pattern": "rnj_FA-*.out"},
]

def get_latest_log(subfolder, pattern):
    search_path = os.path.join(LOG_BASE, subfolder, pattern)
    files = glob.glob(search_path)
    if not files:
        return None, None
    
    # Sort by modification time
    latest_file = max(files, key=os.path.getmtime)
    
    # Check for corresponding .err file
    err_file = latest_file.replace(".out", ".err")
    if not os.path.exists(err_file):
        err_file = None
        
    return latest_file, err_file

def parse_fc_log(out_file, err_file):
    status = "Unknown"
    progress = "0%"
    details = ""
    job_id = os.path.basename(out_file).split("-")[-1].replace(".out", "")

    try:
        # Check output file content
        with open(out_file, 'r') as f:
            content = f.read()
            
        if "SLURM job completed at:" in content:
            status = "Completed"
            progress = "100%"
        elif "Wrote final pools summary:" in content:
            status = "Completed"
            progress = "100%"
        elif "Traceback (most recent call last):" in content:
            status = "Failed"
        else:
            status = "Running"
            
            # Try to get round info
            round_matches = re.findall(r"\[Round (\d+)\]", content)
            if round_matches:
                current_round = round_matches[-1]
                details = f"Round {current_round}"
                
        # Check error file for tqdm progress if running
        if status == "Running" and err_file:
            try:
                with open(err_file, 'r') as f:
                    err_lines = f.readlines()
                    # Look for tqdm percentages in last few lines
                    for line in reversed(err_lines[-20:]):
                        match = re.search(r"(\d+)%", line)
                        if match:
                            progress = f"{match.group(1)}%"
                            break
            except Exception:
                pass

    except Exception as e:
        details = f"Error reading log: {str(e)}"

    return job_id, status, progress, details

def parse_fa_log(out_file, err_file):
    status = "Unknown"
    progress = "0%"
    details = ""
    job_id = os.path.basename(out_file).split("-")[-1].replace(".out", "")
    
    total_runs = 114 # Default range(N_RUNS) -> 114 runs. May vary by script, but good estimate.
    # Actually, let's try to infer total runs if printed
    
    try:
        with open(out_file, 'r') as f:
            content = f.read()

        if "SLURM job completed at:" in content:
            status = "Completed"
            progress = "100%"
        elif "Traceback (most recent call last):" in content:
            status = "Failed"
        else:
            status = "Running"
            
            # Find last run line: "--- Run 35/126 (seed=...)"
            run_matches = re.findall(r"--- Run (\d+)/(\d+)", content)
            if run_matches:
                current_run, total_runs_log = run_matches[-1]
                current_run = int(current_run)
                total_runs_log = int(total_runs_log)
                
                # Approximate progress based on run number
                # We are at start of 'current_run'
                base_progress = (current_run - 1) / total_runs_log
                
                progress_val = base_progress * 100
                progress = f"{progress_val:.1f}%"
                details = f"Run {current_run}/{total_runs_log}"
            else:
                 details = "Initializing..."

    except Exception as e:
        details = f"Error reading log: {str(e)}"

    return job_id, status, progress, details

def main():
    print(f"Monitoring Data Collection Progress")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 100)
    print(f"{'Type':<5} {'Model':<15} {'Job ID':<10} {'Status':<12} {'Progress':<10} {'Details':<30} {'Last Update'}")
    print("-" * 100)

    results = []

    for task in TASKS:
        out_file, err_file = get_latest_log(task['subfolder'], task['pattern'])
        
        if not out_file:
            results.append({
                "Type": task['type'],
                "Model": task['model'],
                "Job ID": "N/A",
                "Status": "No Log",
                "Progress": "-",
                "Details": "-",
                "Last Update": "-"
            })
            continue

        last_update = datetime.fromtimestamp(os.path.getmtime(out_file)).strftime('%H:%M:%S')
        
        if task['type'] == 'FC':
            job_id, status, progress, details = parse_fc_log(out_file, err_file)
        else:
            job_id, status, progress, details = parse_fa_log(out_file, err_file)
            
        results.append({
            "Type": task['type'],
            "Model": task['model'],
            "Job ID": job_id,
            "Status": status,
            "Progress": progress,
            "Details": details,
            "Last Update": last_update
        })

    # Sort by Type then Model
    results.sort(key=lambda x: (x['Type'], x['Model']))

    for r in results:
        print(f"{r['Type']:<5} {r['Model']:<15} {r['Job ID']:<10} {r['Status']:<12} {r['Progress']:<10} {r['Details']:<30} {r['Last Update']}")

    print("-" * 100)

if __name__ == "__main__":
    main()
