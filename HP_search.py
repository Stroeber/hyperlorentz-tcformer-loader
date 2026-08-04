import os
import subprocess
import itertools
import pandas as pd
import numpy as np
import zipfile
import json
import io
from pathlib import Path

# ==========================================
# 1. Flexible Configuration Grid
# ==========================================
BASE_FOLDER = "./results"
DATASET = "mamem"
MODEL_NAME = "BaselineDeviationModelIdEmbedHeadLora"
MAX_GRACE_COUNT = 3
match DATASET:
    case 'mamem':
        NUM_SUBJECTS = 11
    case 'bci': 
        NUM_SUBJECTS = 9
    case 'bcicha':
        NUM_SUBJECTS = 16
RUN_IDENTIFIERS = [10, 11, 12, 13, 14]

# Define your hyperparameter search space here.
# You can add, remove, or change keys freely.
HP_GRID = {
    'inception': {
        'bs': [64, 128], #32, 
        'lr': [1e-3, 1e-4], #1e-2, 
        'dist': ['AE', 'COV'], #, 'KL'
        'wd': [1e-2, 1e-3], 
        'filter_len': [40], #, 60
        # 'run_id': RUN_IDENTIFIERS
    },
    'BaselineDeviationModelIdEmbedHeadLora':{
        'bs': [64, 128],
        'lr': [1e-2, 1e-3, 1e-4],
        'dist': ['AE', 'COV'],# 'KL'],
        'wd': [1e-2, 1e-3],
        'learn_decoder': [0, 1],
        'scheduler': [1],# 0],
        # 'run_id': RUN_IDENTIFIERS
    }
}

TRACKED_METRICS = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']
BCICHA_SUB_ID = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]


# ==========================================
# 2. Path & Configuration String Helper
# ==========================================

def add_default_params(hp):
    if MODEL_NAME == 'BaselineDeviationModelIdEmbedHeadLora':
        if DATASET == 'mamem':
            hp['filter_len'] = 0
            hp['windows'] = 1
            hp['dropout'] = 0
            hp['lora_lr'] = 1e-5
            hp['pre_processor'] = True
            hp['pre_encoder'] = True
            hp['cutfill'] = True
            hp['learn_predecoder'] = False
            hp['learn_lora'] = True
            hp['debug'] = False
        if DATASET == 'bci':
            pass #Don't have params, also need to train recon model first
            # hp['filter_len'] = 0
            # hp['windows'] =  
            # hp['dropout'] =
            # hp['lora_lr'] =
            # hp['pre_processor'] =
            # hp['pre_encoder'] =
            # hp['cutfill'] =
            # hp['learn_predecoder'] =
            # hp['learn_lora'] =
            # hp['debug'] = False
        if DATASET == 'bcicha':
            hp['filter_len'] = 0
            hp['windows'] = 1
            hp['dropout'] = 0
            hp['lora_lr'] = 1e-1
            hp['pre_processor'] = True
            hp['pre_encoder'] = True
            hp['cutfill'] = True
            hp['learn_predecoder'] = False
            hp['learn_lora'] = True
            hp['debug'] = False
    elif MODEL_NAME == 'inception':
        hp['learn_decoder'] = False
        hp['scheduler'] = True
        hp['windows'] = 0
        hp['dropout'] = 0.0
        hp['lora_lr'] = 0.0
        hp['pre_processor'] = False
        hp['pre_encoder'] = False
        hp['cutfill'] = False
        hp['learn_predecoder'] = False
        hp['learn_lora'] = False
        hp['debug'] = False
    return hp
    

def get_hp_config_string(hp):
    """
    Generates the exact folder name based on custom naming convention.
    """
    #cache_path = WindowsPath('locked_pools/mamem/inception/sub_1_pool_bs32_lr0.01_distAE_wd0.01_filterlen40_learndecoderFalse_schedulerTrue_windows0_dropout0.0_loralr0.0_preprocessorFalse_preencoderFalse_cutfillFalse_learnpredecoderFalse_learnloraFalse.json'),
# DATASET = 'mamem',
# MODEL_NAME = 'inception'
    return (f"bs{hp['bs']}_lr{hp['lr']}_dist{hp['dist']}_wd{hp['wd']}_fl{hp['filter_len']}_dec{hp['learn_decoder']}_"
            f"sd{hp['scheduler']}_win{hp['windows']}_dp{hp['dropout']}_llr{hp['lora_lr']}_proc{hp['pre_processor']}_"
            f"enc{hp['pre_encoder']}_cut{hp['cutfill']}_predec{hp['learn_predecoder']}_lora{hp['learn_lora']}")
    # return (f"win{hp['windows']}_bs{hp['bs']}_lr{hp['lr']}_wd{hp['wd']}_"
    #         f"dp{hp['dropout']}_llr{hp['lora_lr']}_proc{hp['pre_processor']}_"
    #         f"enc{hp['pre_encoder']}_cut{hp['cutfill']}_dec{hp['learn_decoder']}_"
    #         f"predec{hp['learn_predecoder']}_lora{hp['learn_lora']}_id{hp['id']}")

def get_base_hp_identifier(hp):
    """
    Generates a unique identifier for a configuration EXCLUDING the run identifier (id).
    Used to lock and cache the subject pool.
    """
    hp_copy = hp.copy()
    hp_copy.pop('run_id', None)
    return "_".join(f"{k.replace('_', '')}{v}" for k, v in hp_copy.items())

# ==========================================
# 3. Zip Extraction & Metric Loading
# ==========================================
def extract_metrics_from_csv(hp, sub_list, run_id, del_results = False):
    """
    Opens the {hp_config} folder and extracts tracked metrics from run_{run_id}.csv
    """
    hp_config = get_hp_config_string(hp)
    
    sub_list_str = '_'.join(str(x) for x in sub_list)
    
    subject_str = f"subject_{sub_list_str}"
    
    results_path = os.path.join(BASE_FOLDER, DATASET, MODEL_NAME, hp_config, subject_str)##############################
    target_csv = os.path.join(results_path, f"run_{run_id}.csv")
    
    if not os.path.exists(results_path):
        print(f"    [Warning] Results folder not found at {results_path}")
        return None
        
    try:      
        with open(target_csv) as f:
            df = pd.read_csv(f)
            # Fetch row with lowest validation loss
            final_metrics = {}
            for metric in TRACKED_METRICS:
                if metric in df.columns:
                    final_metrics[metric] = float(df[metric].iloc[df['val_loss'].idxmin()])
                    # final_metrics[metric] = float(df[metric].iloc[-1])
        if del_results:
              os.remove(target_csv)
              os.rmdir(results_path)  
              os.rmdir(os.path.join(BASE_FOLDER, DATASET, MODEL_NAME, subject_str))
        return final_metrics
    except Exception as e:
        print(f"    [Error] Failed to read metrics from file: {e}")
        return None

# ==========================================
# 4. Command Execution Logic
# ==========================================
def run_training_command(hp, sub_list, early_stopping, run_id):
    """
    Dynamically constructs CLI flags from the HP dict and calls main.py
    """
    sub_list_str = '_'.join(str(x) for x in sub_list)
    
    # Base command python call
    cmd = ["python", "main.py", "--sub", sub_list_str, "--run_id", str(run_id), "--es", str(early_stopping), "--dataset", DATASET, "--model", MODEL_NAME]
    
    # Automatically append all grid configurations as flags (e.g., --bs 32 --lr 0.001)
    for key, value in hp.items():
        cmd.extend([f"--{key}", str(value)])
        
    print(f"  Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# ==========================================
# 5. Greedy Subject Pooling & Locking System
# ==========================================
def load_distance_matrix(metric):
    matrix_path = os.path.join('distances', f'{DATASET}_{metric}_averaged_matrix.csv')
    matrix = pd.read_csv(matrix_path, index_col=0)
    return matrix

def get_or_create_subject_pool(hp, sub, base_run_id):
    """
    Looks for a locked subject pool file. If it doesn't exist, runs the greedy search
    using the first seed (base_run_id) and saves the layout.
    """
    
    cache_path = Path("locked_pools", DATASET, MODEL_NAME, f"sub_{sub}_pool_{get_hp_config_string(hp)}.json")
    absolute_cache_path = cache_path.resolve()
    absolute_cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If the pool is locked/cached, load it immediately
    if absolute_cache_path.exists():
        with open(absolute_cache_path, "r") as f:
            locked_pool = json.load(f)
        print(f"  [Cache Hit] Loaded locked pool for subject_{sub}: {locked_pool}")
        return locked_pool

    print(f"  [Cache Miss] Starting greedy pooling search for subject_{sub}...")
    distances = load_distance_matrix(hp['dist'])
    if DATASET == 'bcicha':
        sorted_neighbor_indices = distances.iloc[BCICHA_SUB_ID.index(sub), :].argsort().tolist()
    else:
        sorted_neighbor_indices = distances.iloc[sub - 1, :].argsort().tolist()
    
    # Setup baseline configuration copy explicitly for the pooling seed
    pooling_hp = hp.copy()
    pooling_hp['run_id'] = base_run_id
    
    best_sub_list = [sub]
    
    # Baseline run
    run_training_command(pooling_hp, best_sub_list, early_stopping=False, run_id=base_run_id)
    metrics = extract_metrics_from_csv(pooling_hp, best_sub_list, run_id=base_run_id)
    best_loss = metrics['val_loss'] if metrics and not np.isnan(metrics['val_loss']) else 1e5
    
    grace_count = 0
    
    for neighbor_idx in sorted_neighbor_indices:
        if DATASET == 'bchicha':
            neighbor_sub = BCICHA_SUB_ID[neighbor_idx]
        else:
            neighbor_sub = neighbor_idx + 1
        if neighbor_sub == sub:
            continue
            

        candidate_list = best_sub_list + [neighbor_sub]
        
        # Test candidate
        run_training_command(pooling_hp, candidate_list, early_stopping=False, run_id=base_run_id)
        metrics = extract_metrics_from_csv(pooling_hp, candidate_list, run_id=base_run_id)
        new_loss = metrics['val_loss'] if metrics and not np.isnan(metrics['val_loss']) else 1e5
        
        if new_loss < best_loss:
            print(f'[pooling] Found new pool {candidate_list} with {round(new_loss, 5)} loss (vs {best_loss})')
            best_loss = new_loss
            best_sub_list = candidate_list
            grace_count = 0
        else:
            print(f'[pooling] Pool {candidate_list} was not better than {best_sub_list} ({round(new_loss, 5)} / {round(best_loss, 5)})')
            grace_count += 1
            
        if grace_count >= MAX_GRACE_COUNT:
            print(f'[pooling] Could not find better pool for sub {sub} after {MAX_GRACE_COUNT} tries')
            break
            
    # Save/Lock the discovered pool layout
    with open(absolute_cache_path, "w") as f:
        json.dump(best_sub_list, f)

    print(f"  [Locked] Saved subject pool {best_sub_list} to {cache_path}")
    
    return best_sub_list

# ==========================================
# 6. Main Execution Core
# ==========================================
def main():
    # Separate the seeds from operational HPs
    identifiers = RUN_IDENTIFIERS
    hp_keys = [k for k in HP_GRID[MODEL_NAME].keys() if k != 'it']
    hp_values = [HP_GRID[MODEL_NAME][k] for k in hp_keys]
    
    # Generate all non-seed combinations
    base_combinations = [dict(zip(hp_keys, v)) for v in itertools.product(*hp_values)]
    raw_results = []
    
    for hp_comb_idx, base_hp in enumerate(base_combinations):
        print(f"\n==================================================")
        print(f"Evaluating Base HP Space: ({hp_comb_idx}/{len(base_combinations)}) {base_hp}")
        print(f"==================================================")

        hp = add_default_params(base_hp)
        
        for sub_idx in range(NUM_SUBJECTS):
            if DATASET == 'bcicha':
                sub = BCICHA_SUB_ID[sub_idx]
            else:
                sub = sub_idx + 1
            
            # 1. Fetch or execute greedy search to lock the pool
            # Uses the first available seed as the search pipeline reference
            locked_pool = get_or_create_subject_pool(hp, sub, base_run_id=identifiers[0])
            
            # Define both single-subject (baseline) and pooled configurations
            pools_to_evaluate = {
                'single': [sub],
                'pooled': locked_pool
            }
            
            # 2. Execute training for both Single and Pooled setups across all run seeds
            for pool_type, current_pool in pools_to_evaluate.items():
                print(f"\n --- Subject {sub} | Mode: {pool_type.upper()} | Pool: {current_pool} ---")
                
                for idx, run_id in enumerate(identifiers):
                    active_hp = hp.copy()
                    active_hp['run_id'] = run_id
                    
                    print(f" -> Running Subject {sub} ({pool_type}) | ID: {run_id} (Run {idx+1}/{len(identifiers)})")
                    
                    # Run if result is missing
                    metrics = extract_metrics_from_csv(active_hp, current_pool, run_id)
                    if metrics is None:
                        run_training_command(active_hp, current_pool, early_stopping=True, run_id=run_id)
                        metrics = extract_metrics_from_csv(active_hp, current_pool, run_id)
                    else:
                        print("    [Skipping] Archive results already exist for this run configuration.")
                    
                    if metrics:
                        record = {
                            'base_hp_config': get_base_hp_identifier(base_hp),
                            'subject': sub,
                            'pool_type': pool_type,
                            'run_id': run_id,
                            'pool_size': len(current_pool),
                            'pool': '_'.join(str(x) for x in current_pool)
                        }
                        record.update(metrics)
                        raw_results.append(record)
                        
    # ==========================================
    # 3. Aggregate & Save Results
    # ==========================================
    if not raw_results:
        print("No completed metrics were collected.")
        return
        
    df_results = pd.DataFrame(raw_results)
    
    # Save Raw Data
    os.makedirs(f"{BASE_FOLDER}/raw", exist_ok=True)
    os.makedirs(f"{BASE_FOLDER}/hp_search", exist_ok=True)
    df_results.to_csv(f"{BASE_FOLDER}/raw/{MODEL_NAME}_{DATASET}_hp_search_raw_runs.csv", index=False)
    
    # Group by config, subject, and pool_type to calculate Mean & SD for each mode
    group_cols = ['base_hp_config', 'subject', 'pool_type']
    summary_mean = df_results.groupby(group_cols)[TRACKED_METRICS].mean().add_suffix('_mean')
    summary_std = df_results.groupby(group_cols)[TRACKED_METRICS].std().add_suffix('_std')
    
    full_summary = summary_mean.join(summary_std).reset_index()
    full_summary.to_csv(f"{BASE_FOLDER}/hp_search/{MODEL_NAME}_{DATASET}_summary.csv", index=False)
    
    # Create Side-by-Side Comparison Table (Single vs Pooled)
    pivoted_summary = full_summary.pivot(
        index=['base_hp_config', 'subject'],
        columns='pool_type'
    )
    
    # Flatten column names (e.g. val_losses_mean_single, val_losses_mean_pooled)
    pivoted_summary.columns = [f"{metric}_{p_type}" for metric, p_type in pivoted_summary.columns]
    comparison_df = pivoted_summary.reset_index()
    
    # Optional: Calculate loss reduction / accuracy gain deltas
    if 'val_losses_mean_pooled' in comparison_df.columns and 'val_losses_mean_single' in comparison_df.columns:
        comparison_df['val_loss_delta'] = comparison_df['val_losses_mean_pooled'] - comparison_df['val_losses_mean_single']
        
    comparison_df.to_csv(f"{BASE_FOLDER}/hp_search/{MODEL_NAME}_{DATASET}_comparison_side_by_side.csv", index=False)
    
    print("\n--- Hyperparameter Search Strategy Finished ---")
    print(f"1. Raw tracking info: {BASE_FOLDER}/raw/{MODEL_NAME}_{DATASET}_hp_search_raw_runs.csv")
    print(f"2. Grouped Summary: {BASE_FOLDER}/hp_search/{MODEL_NAME}_{DATASET}_summary.csv")
    print(f"3. Side-by-Side Comparison: {BASE_FOLDER}/hp_search/{MODEL_NAME}_{DATASET}_comparison_side_by_side.csv")

if __name__ == "__main__":
    main()