"""
Pure PyTorch data loading utilities.
Extracts dataloader creation from Lightning datamodules without depending on pl.LightningDataModule.
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.interaug import interaug


def make_collate_fn(preproc):
    """Return a collate function that optionally applies interaug.

    Handles both 2-element (x, y) and 3-element (x, subject_id, y) tuples.
    """

    def collate(batch):
        if len(batch[0]) == 3:  # (x, subject_id, y)
            xs, subject_ids, ys = zip(*batch)
            x = torch.stack(xs)
            sub = torch.tensor(subject_ids, dtype=torch.long).reshape(-1, 1, 1, 1)
            y = torch.tensor(ys, dtype=torch.long)

            if preproc.get("interaug", False):
                # Pass subject IDs to interaug so they're shuffled consistently
                x, y, sub = interaug([x, y], subject_ids=sub)
            return x, sub, y
        else:  # (x, y)
            xs, ys = zip(*batch)
            x = torch.stack(xs)
            y = torch.tensor(ys, dtype=torch.long)

            if preproc.get("interaug", False):
                x, y = interaug([x, y])
            return x, y

    return collate


def make_collate_fn_inter(do_interaug):
    """Return a collate function that optionally applies interaug.

    Handles both 2-element (x, y) and 3-element (x, subject_id, y) tuples.
    """

    def collate(batch):
        if len(batch[0]) == 3:  # (x, subject_id, y)
            xs, subject_ids, ys = zip(*batch)
            x = torch.stack(xs)

            sub = torch.cat(subject_ids, dim=0).unsqueeze(1)[:, 0, 0, 0].reshape(-1, 1, 1, 1)

            y = torch.tensor(ys, dtype=torch.long)

            if do_interaug:
                # Pass subject IDs to interaug so they're shuffled consistently
                x, y, sub = interaug([x, y], subject_ids=sub)
            return x, sub, y
        else:  # (x, y)
            xs, ys = zip(*batch)
            x = torch.stack(xs)
            y = torch.tensor(ys, dtype=torch.long)

            if do_interaug:
                x, y = interaug([x, y])
            return x, y

    return collate


def create_train_dataloader(dataset, preprocessing_dict):
    """Create training dataloader with shuffling and interaug support."""
    num_workers = preprocessing_dict.get("num_workers", 0)  
    return DataLoader(
        dataset,
        batch_size=preprocessing_dict["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=make_collate_fn(preprocessing_dict)
    )


def create_test_dataloader(dataset, preprocessing_dict):
    """Create test/validation dataloader without shuffling."""
    num_workers = preprocessing_dict.get("num_workers", 0) 
    return DataLoader(
        dataset,
        batch_size=preprocessing_dict["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None
    )


def make_tensor_dataset(X, y, subject_ids=None):
    """Create a TensorDataset from numpy arrays.
    
    Args:
        X: Data array
        y: Labels array
        subject_ids: Optional array of subject IDs per sample
        
    Returns:
        TensorDataset yielding (X, y) or (X, subject_id, y) tuples
    """
    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y).type(torch.LongTensor)
    
    if subject_ids is not None:
        subject_tensor = torch.Tensor(subject_ids).type(torch.LongTensor)
        return TensorDataset(X_tensor, subject_tensor, y_tensor)
    return TensorDataset(X_tensor, y_tensor)


class DataLoaderManager:
    """
    Manages data loading for pure PyTorch training.
    Uses the existing datamodule classes to prepare data, then creates standard DataLoaders.
    
    Args:
        datamodule_cls: The datamodule class to use for data preparation
        preprocessing_dict: Dictionary of preprocessing parameters
        subject_id: Subject ID to load data for, or "all" to load all subjects
        include_subject_ids: If True, datasets return (X, subject_id, y) tuples
    """
    
    def __init__(self, datamodule_cls, preprocessing_dict, subject_id, include_subject_ids=False):
        self.preprocessing_dict = preprocessing_dict
        self.subject_id = subject_id
        self.include_subject_ids = include_subject_ids
        
        # Handle "all" subject_id by finding the All variant of the datamodule
        if subject_id == "all":
            # Try to find the "All" variant (e.g., BCICIV2a -> BCICIV2aAll)
            base_name = datamodule_cls.__name__
            all_cls_name = base_name + "All"
            module = datamodule_cls.__module__
            import importlib
            mod = importlib.import_module(module)
            if hasattr(mod, all_cls_name):
                self.datamodule_cls = getattr(mod, all_cls_name)
                self.include_subject_ids = True  # Force include subject IDs for multi-subject
            else:
                raise ValueError(f"No 'All' variant found for {base_name}. "
                               f"Expected {all_cls_name} in {module}")
        else:
            self.datamodule_cls = datamodule_cls
        
        # These will be set after setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """
        Initialize the datamodule and extract datasets.
        We instantiate the Lightning datamodule but only use its data preparation logic.
        """
        # Create datamodule instance and prepare data
        dm = self.datamodule_cls(self.preprocessing_dict, self.subject_id, 
                                  include_subject_ids=self.include_subject_ids)
        dm.prepare_data()
        dm.setup()
        
        # Extract the datasets
        self.train_dataset = dm.train_dataset
        self.test_dataset = dm.test_dataset
        self.val_dataset = getattr(dm, 'val_dataset', None)
        
        # Store num_subjects if available
        self.num_subjects = getattr(self.datamodule_cls, 'num_subjects', 1)
        
        return self
    
    def get_subject_id(self):
        """Return the subject ID used for this data manager."""
        return self.subject_id
    
    def get_num_subjects(self):
        """Return the number of subjects in the dataset."""
        return self.num_subjects
    
    def get_train_loader(self):
        """Get training DataLoader."""
        return create_train_dataloader(self.train_dataset, self.preprocessing_dict)
    
    def get_val_loader(self):
        """Get validation DataLoader (returns test loader if no val set)."""

        # if self.val_dataset is None:
        #     raise ValueError("No val set found for this data manager.")

        dataset = self.val_dataset if self.val_dataset is not None else self.test_dataset

        return create_test_dataloader(dataset, self.preprocessing_dict)
    
    def get_test_loader(self):
        """Get test DataLoader."""
        return create_test_dataloader(self.test_dataset, self.preprocessing_dict)
