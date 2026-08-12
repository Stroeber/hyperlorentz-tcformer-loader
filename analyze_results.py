import os
import pandas as pd
import numpy as np

# ==========================================
# 1. Configuration
# ==========================================
BASE_FOLDER = "results"
OUTPUT_FOLDER = f"{BASE_FOLDER}/analysis"
MODELS = ["inception", "BaselineDeviationModelIdEmbedHeadLora"]
DATASETS = ["mamem", "bci", "bcicha"]

# The metric used to select the best HP config per subject
SELECTION_METRIC = "val_acc_mean_pooled" ##or pooled? 
# The metric reported in the final tables
REPORT_METRIC = "test_acc" 

# ==========================================
# 2. Helper Functions
# ==========================================
def format_cell(mean, std, is_percentage=True):
    """Formats values as 'Mean ± Std' for easy LaTeX conversion."""
    if pd.isna(mean) or pd.isna(std):
        return "N/A"
    
    # Optional: multiply by 100 if your accuracies are stored as decimals (0.0 to 1.0)
    # If they are already 0-100, set multiplier to 1.
    multiplier = 100.0 if is_percentage and mean <= 1.0 else 1.0
    
    return f"{(mean * multiplier):.2f} ± {(std * multiplier):.2f}"

def format_delta(delta, is_percentage=True):
    if pd.isna(delta):
        return "N/A"
        
    multiplier = 100.0 if is_percentage and abs(delta) <= 1.0 else 1.0
    val = delta * multiplier
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}"

# ==========================================
# 3. Main Analysis Logic
# ==========================================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    for model in MODELS:
        print(f"\n==================================================")
        print(f"Aggregating Results for Model: {model}")
        print(f"==================================================")
        
        dataset_level_records = []
        
        for dataset in DATASETS:
            file_path = f"{BASE_FOLDER}/hp_search/{model}_{dataset}_comparison_side_by_side.csv"
            
            # ----------------------------------------------------
            # Handle Missing Data
            # ----------------------------------------------------
            if not os.path.exists(file_path):
                print(f"  [Missing] Data for {dataset} not found. Marking as N/A.")
                dataset_level_records.append({
                    "Dataset": dataset,
                    "Single Accuracy": "N/A",
                    "Pooled Accuracy": "N/A",
                    "All Accuracy": "N/A",
                    "Delta (Pool vs Single)": "N/A",
                    "Delta (Pool vs All)": "N/A"
                })
                continue
                
            # ----------------------------------------------------
            # Load and Process Data
            # ----------------------------------------------------
            df = pd.read_csv(file_path)
            
            # Select the best HP configuration for each subject based on validation accuracy
            if SELECTION_METRIC in df.columns:
                best_hp_indices = df.groupby('subject')[SELECTION_METRIC].idxmax()
                best_df = df.loc[best_hp_indices].copy()
            else:
                # Fallback if selection metric is missing (e.g. tracking names mismatch)
                best_df = df.copy()
                print(f'[Error] Selection metric was not in df columns')
            
            # Ensure report columns exist
            single_mean_col = f"{REPORT_METRIC}_mean_single"
            single_std_col  = f"{REPORT_METRIC}_std_single"
            pooled_mean_col = f"{REPORT_METRIC}_mean_pooled"
            pooled_std_col  = f"{REPORT_METRIC}_std_pooled"
            all_mean_col    = f"{REPORT_METRIC}_mean_all"
            all_std_col     = f"{REPORT_METRIC}_std_all"
            
            required_cols = [single_mean_col, single_std_col, pooled_mean_col, pooled_std_col]
            missing_cols = [c for c in required_cols if c not in best_df.columns]
            if missing_cols:
                print(f"  [Warning] Missing columns {missing_cols} in {dataset}. Filling with NaN.")
                for c in missing_cols:
                    best_df[c] = np.nan

            # Calculate Deltas
            best_df['delta_pool_single'] = best_df[pooled_mean_col] - best_df[single_mean_col]
            best_df['delta_pool_all'] = best_df[pooled_mean_col] - best_df[all_mean_col]
            
            # ----------------------------------------------------
            # Generate Subject-Level Table
            # ----------------------------------------------------
            subject_records = []
            for _, row in best_df.iterrows():
                subject_records.append({
                    "Subject": row['subject'],
                    "Single Accuracy": format_cell(row[single_mean_col], row[single_std_col]),
                    "All Accuracy": format_cell(row[all_mean_col], row[all_std_col]),
                    "Pooled Accuracy": format_cell(row[pooled_mean_col], row[pooled_std_col]),
                    "Delta (Pool vs Single)": format_delta(row['delta_pool_single']),
                    "Delta (Pool vs All)": format_delta(row['delta_pool_all'])
                })
                
            subject_df = pd.DataFrame(subject_records)
            subject_out_path = f"{OUTPUT_FOLDER}/{model}_{dataset}_subject_level.csv"
            subject_df.to_csv(subject_out_path, index=False)
            print(f"  -> Saved subject-level table: {subject_out_path}")
            
            # ----------------------------------------------------
            # Calculate Dataset-Level Aggregations
            # ----------------------------------------------------
            # Standard practice in BCI is to report the mean of the subject means, 
            # and the standard deviation across subjects (cross-subject variability).
            
            dataset_level_records.append({
                "Dataset": dataset,
                "Single Accuracy": format_cell(best_df[single_mean_col].mean(), best_df[single_mean_col].std()),
                "All Accuracy": format_cell(best_df[all_mean_col].mean(), best_df[all_mean_col].std()),
                "Pooled Accuracy": format_cell(best_df[pooled_mean_col].mean(), best_df[pooled_mean_col].std()),
                "Delta (Pool vs Single)": format_delta(best_df['delta_pool_single'].mean()),
                "Delta (Pool vs All)": format_delta(best_df['delta_pool_all'].mean())
            })
            
        # ----------------------------------------------------
        # Generate Model-Level (All Datasets) Table
        # ----------------------------------------------------
        model_summary_df = pd.DataFrame(dataset_level_records)
        model_out_path = f"{OUTPUT_FOLDER}/{model}_all_datasets_summary.csv"
        model_summary_df.to_csv(model_out_path, index=False)
        print(f"\n  => Saved model summary table: {model_out_path}")

    print("\n--- Aggregation Complete ---")
    print(f"All files have been saved to the '{OUTPUT_FOLDER}' directory.")
    # print("Format is ready for csv-to-LaTeX conversion (e.g., '82.50 ± 4.20').")

if __name__ == "__main__":
    main()