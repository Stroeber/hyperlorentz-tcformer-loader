import argparse
import torch
import sys
sys.path.append('../')
from utils.utils import get_dataloaders

from distances.autoencoder import EEGAutoencoder
from distances.covariance import CovarianceDistance
from distances.KL_divergence import KLDivergenceDistance

import torch
import pandas as pd
import numpy as np
from collections import defaultdict

def aggregate_all_subject_data(dataloader, target_labels, device='cuda'):
    """
    Groups EEG recordings by subjectID first, and then by label.
    Output structure: subject_data[subject_id][label] = tensor(num_trials, channels, time)
    """
    # Using a nested dictionary factory
    subject_data = defaultdict(lambda: defaultdict(list))
    
    for eeg_batch, subj_id_batch, label_batch in dataloader:
        for eeg_sample, sub_id, label in zip(eeg_batch, subj_id_batch, label_batch):
            lbl_key = label.item() if isinstance(label, torch.Tensor) else label
            sub_key = sub_id.item() if isinstance(sub_id, torch.Tensor) else sub_id
            
            # Only track labels we care about for this dataset
            eeg_sample = eeg_sample.to(device)
            if lbl_key in target_labels:
                subject_data[sub_key][lbl_key].append(eeg_sample)
                
    # Stack individual trials into single tensors per subject-label pair
    for sub_id in subject_data:
        for label in subject_data[sub_id]:
            subject_data[sub_id][label] = torch.stack(subject_data[sub_id][label])
            
    return subject_data


def compute_label_averaged_matrix(subject_data, distance_metric_fn, target_labels):
    """
    Computes a single symmetric NxN distance matrix where each entry 
    is the average distance across all valid labels for that subject pair.
    """
    subjects = sorted(list(subject_data.keys()))
    n_subjects = len(subjects)
    
    distance_matrix = torch.zeros((n_subjects, n_subjects))
    
    for i in range(n_subjects):
        for j in range(i + 1, n_subjects):
            sub_i = subjects[i]
            sub_j = subjects[j]
            
            total_distance = 0.0
            valid_labels_count = 0
            
            # Compute distance for each label and track the running sum
            for label in target_labels:
                # Ensure BOTH subjects have data for this specific condition/label
                if label in subject_data[sub_i] and label in subject_data[sub_j]:
                    dist = distance_metric_fn(subject_data[sub_i][label], subject_data[sub_j][label])
                    total_distance += dist
                    valid_labels_count += 1
            
            # Calculate the average across available conditions
            if valid_labels_count > 0:
                avg_distance = total_distance / valid_labels_count
            else:
                avg_distance = 0.0  # Fallback if they share no common label data
                
            distance_matrix[i, j] = avg_distance
            distance_matrix[j, i] = avg_distance
            
    return distance_matrix, subjects


DATASET_CONFIGS = {
    'bci': {
        'channels': 22,
        'seq_len': 438,
        'target_labels': [0, 1, 2, 3], # Example labels for this dataset
        'default_path': "../data/BCICIV_2a_mat/"
    },
    'mamem': {
        'channels': 8,
        'seq_len': 125,
        'target_labels': [0, 1, 2, 3, 4],
        'default_path': "../data/MAMEM/"
    },
    'bcicha': {
        'channels': 56,
        'seq_len': 160,
        'target_labels': [0, 1],
        'default_path': "../data/BCIcha/"
    }
}



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='bcicha', choices=list(DATASET_CONFIGS.keys()), help='Dataset name')
    ap.add_argument('--method', type=str, default='KL', choices=['AE', 'COV', 'KL'], help='Distance metric method')
    ap.add_argument('--epochs', type=str, default=5, help='AE training epochs (if method=AE)')
    
    args = vars(ap.parse_args())
    dataset_name = args['dataset']
    method_name = args['method']
    
    print(f"=== Starting Pipeline ===")
    print(f"Dataset Selected: {dataset_name.upper()}")
    print(f"Metric Selected:  {method_name.upper()}")
    print(f"=========================")

    # 1. Fetch metadata configurations
    config = DATASET_CONFIGS[dataset_name]
    channels = config['channels']
    seq_len = config['seq_len']
    target_labels = config['target_labels']
    data_path = config['default_path']
    
    # 2. Load dataloader
    trainloader, validloader, testloader, _, _, _= get_dataloaders(dataset=dataset_name, 
                                                            subject='all',
                                                            batch_size=64,
                                                            finetune=None,
                                                            interaug=True,
                                                            data_path=data_path)
    
    # 3. Dynamic Distance Function Initialization
    if method_name == 'AE':
        print("\n[Setup] Initializing and training 1D Convolutional Autoencoder...")
        ae_model = EEGAutoencoder(in_channels=channels, sequence_length=seq_len, latent_dim=64)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        trained_model = ae_model.train_eeg_autoencoder(ae_model, trainloader, epochs=int(args['epochs']), device=device)
        distance_metric = ae_model.LatentEuclideanDistance
        
    elif method_name == 'COV':
        print("\n[Setup] Initializing Log-Euclidean Covariance Matrix Distance...")
        distance_metric = CovarianceDistance(metric='log_euclidean')
        
    elif method_name == 'KL':
        print("\n[Setup] Initializing Symmetric Multivariate Gaussian KL-Divergence...")
        distance_metric = KLDivergenceDistance(eps=1e-6)

    # 4. Global Data Aggregation
    print(f"\nAggregating all subject trial records across labels: {target_labels}")
    all_subject_data = aggregate_all_subject_data(trainloader, target_labels)
    
    # 5. Compute the Label-Averaged Matrix
    print("Computing cross-label averaged distance matrix...")
    final_distance_matrix, ordered_subjects = compute_label_averaged_matrix(
        subject_data=all_subject_data, 
        distance_metric_fn=distance_metric, 
        target_labels=target_labels
    )
    
    # 6. Save/Display final result
    print(f"\n--> Success! Generated single cross-label averaged matrix.")
    print(f"Matrix shape: {final_distance_matrix.shape} for {len(ordered_subjects)} subjects.")
    print("Subject Order:", ordered_subjects)
    print("Matrix Profile:\n", final_distance_matrix)
    
    # Save output to a single file
    final_distance_matrix = pd.DataFrame(final_distance_matrix.detach().numpy())
    output_filename = f"{dataset_name}_{method_name}_averaged_matrix.csv"
    # torch.save({'matrix': final_distance_matrix, 'subjects': ordered_subjects}, output_filename)
    final_distance_matrix.to_csv(output_filename)
    print(f"Saved matrix metadata to {output_filename}")