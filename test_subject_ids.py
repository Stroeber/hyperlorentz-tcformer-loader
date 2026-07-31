"""
Test script to verify the return_subject_ids functionality in bcic4_2a loader.
"""

from datamodules.bcic4_2a import _extract_data_from_dataset, _get_session_keys
from utils.load_bcic4 import load_bcic4
import numpy as np

# Minimal preprocessing config
preprocessing_dict = {
    "sfreq": 128,
    "low_cut": 4,
    "high_cut": 38,
    "start": 0,
    "stop": 1,  # Reduced from 4 to fit within recording boundaries
    "z_scale": False,
}

print("Loading BCIC IV 2a dataset for subjects 1 and 2...")
dataset = load_bcic4(subject_ids=[1, 2], dataset="2a", preprocessing_dict=preprocessing_dict)

# Split by session
splitted_ds = dataset.split("session")
train_key, test_key = _get_session_keys(splitted_ds)
train_dataset = splitted_ds[train_key]

print(f"\nSession keys found: train='{train_key}', test='{test_key}'")

# Test WITHOUT subject IDs (backward compatibility)
print("\n--- Testing without return_subject_ids ---")
X, y = _extract_data_from_dataset(train_dataset)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Test WITH subject IDs
print("\n--- Testing with return_subject_ids=True ---")
X, y, subject_ids = _extract_data_from_dataset(train_dataset, return_subject_ids=True)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(subject_ids)
print(f"subject_ids shape: {subject_ids.shape}")
print(f"Unique subject IDs: {np.unique(subject_ids)}")
print(f"Samples per subject: {dict(zip(*np.unique(subject_ids, return_counts=True)))}")

# Verify alignment
print("\n--- Verification ---")
print(f"X and subject_ids have same length: {len(X) == len(subject_ids)}")
print(f"y and subject_ids have same length: {len(y) == len(subject_ids)}")

print("\n✓ Test completed successfully!")
