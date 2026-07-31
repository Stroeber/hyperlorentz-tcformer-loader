"""
Pure PyTorch metrics tracking utilities.
No PyTorch Lightning dependencies.
"""

import numpy as np
import torch
from torchmetrics.functional import accuracy
from torchmetrics.classification import MulticlassCohenKappa, MulticlassConfusionMatrix


class MetricsTracker:
    """Tracks training and validation metrics across epochs."""
    
    def __init__(self, n_classes: int, device: str = "cpu"):
        self.n_classes = n_classes
        self.device = device
        
        # Per-epoch metric history
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        
        # Test metrics
        self.test_kappa = MulticlassCohenKappa(num_classes=n_classes).to(device)
        self.test_cm = MulticlassConfusionMatrix(num_classes=n_classes).to(device)
        self.test_confmat = None
    
    def update_train(self, loss: float, acc: float):
        """Record training metrics for current epoch."""
        self.train_loss.append(loss)
        self.train_acc.append(acc)
    
    def update_val(self, loss: float, acc: float):
        """Record validation metrics for current epoch."""
        self.val_loss.append(loss)
        self.val_acc.append(acc)
    
    def update_test(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update test metrics with batch predictions."""
        self.test_kappa.update(preds, targets)
        self.test_cm.update(preds, targets)
    
    def compute_test_kappa(self) -> float:
        """Compute final Cohen's Kappa."""
        kappa = self.test_kappa.compute().item()
        self.test_kappa.reset()
        return kappa
    
    def compute_test_confmat(self) -> np.ndarray:
        """Compute and store row-normalized confusion matrix (%)."""
        cm_counts = self.test_cm.compute()
        self.test_cm.reset()
        
        with torch.no_grad():
            row_sums = cm_counts.sum(dim=1, keepdim=True).clamp_min(1)
            cm_percent = cm_counts.float() / row_sums * 100.0
        
        self.test_confmat = cm_percent.cpu().numpy()
        return self.test_confmat
    
    def reset_epoch_metrics(self):
        """Reset running metrics for a new epoch."""
        pass  # Single values are recorded per epoch, no need to reset


def compute_accuracy(y_hat: torch.Tensor, y: torch.Tensor, n_classes: int) -> float:
    """Compute multiclass accuracy."""
    return accuracy(y_hat, y, task="multiclass", num_classes=n_classes).item()


def write_summary(result_dir, model_name, dataset_name, subject_ids,
                  param_count, test_accs, test_losses, test_kappas,
                  train_times, test_times, response_times):
    """Write summary results to a text file (same format as Lightning version)."""
    avg_test_acc = float(np.mean(test_accs))
    std_test_acc = float(np.std(test_accs))
    avg_test_kappa = float(np.mean(test_kappas))
    std_test_kappa = float(np.std(test_kappas))
    avg_test_loss = float(np.mean(test_losses))
    std_test_loss = float(np.std(test_losses))

    total_train_time = float(np.sum(train_times))
    avg_response_time = float(np.mean(response_times))

    with open(result_dir / "results.txt", "w") as f:
        f.write(f"Results for model: {model_name}\n")
        f.write(f"#Params: {param_count}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Subject IDs: {subject_ids}\n\n")
        f.write("Results for each subject:\n")

        for i, subject_id in enumerate(subject_ids):
            f.write(
                f"Subject {subject_id} => Train Time: {train_times[i]:.2f}m, "
                f"Test Time: {test_times[i]:.2f}s, "
                f"Test Acc: {test_accs[i]:.4f}, "
                f"Test Loss: {test_losses[i]:.4f}, "
                f"Test Kappa: {test_kappas[i]:.4f}\n"
            )

        f.write("\n--- Summary Statistics ---\n")
        f.write(f"Average Test Accuracy: {avg_test_acc * 100:.2f} ± {std_test_acc * 100:.2f}\n")
        f.write(f"Average Test Kappa:    {avg_test_kappa:.3f} ± {std_test_kappa:.3f}\n")
        f.write(f"Average Test Loss:     {avg_test_loss:.3f} ± {std_test_loss:.3f}\n")
        f.write(f"Total Training Time: {total_train_time:.2f} min\n")
        f.write(f"Average Response Time: {avg_response_time:.2f} ms\n")

    print("\n=== Summary ===")
    print(f"Average Test Accuracy: {avg_test_acc * 100:.2f} ± {std_test_acc * 100:.2f}")
    print(f"Average Test Kappa:    {avg_test_kappa:.3f} ± {std_test_kappa:.3f}")
    print(f"Average Test Loss:     {avg_test_loss:.3f} ± {std_test_loss:.3f}")
    print(f"Total Training Time: {total_train_time:.2f} min")
    print(f"Average Response Time: {avg_response_time:.2f} ms")
