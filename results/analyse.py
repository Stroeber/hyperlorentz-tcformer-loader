import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class AnalyseResults:
    def __init__(self, dataset, subjects):
        self.dataset = dataset
        self.subjects = subjects if isinstance(subjects, list) else [subjects]
        self.base_path = dataset
        if not os.path.exists(self.base_path):
            print(f"Path does not exist: {self.base_path}")
        # Store full trajectories per model/subject
        self.summary_results = {
            "train_loss": {},
            "train_acc": {},
            "val_loss": {},
            "val_acc": {},
            "test_loss": {},
            "test_acc": {}
        }

    def get_max_val_acc(self, csv_path):
        """Find the max validation accuracy and the corresponding test accuracy."""
        val_accs = []
        test_accs = []
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                val_accs.append(float(row['val_accs']))
                test_accs.append(float(row['test_accs']))
        val_accs_ma = np.convolve(val_accs, np.ones(5) / 5, mode='valid')
        max_index = np.argmax(val_accs_ma)
        max_val_acc = np.argmax(val_accs)
        return max_val_acc, test_accs[max_index+2]

    def find_best_config(self, model_path):
        """Find best configuration for a model by mean test accuracy."""
        config_test_accs = defaultdict(list)
        config_val_accs = defaultdict(list)
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    val_acc, test_acc = self.get_max_val_acc(csv_path)
                    config_path = os.path.dirname(csv_path)
                    config_test_accs[config_path].append(test_acc)
                    config_val_accs[config_path].append(val_acc)

        # Compute stats
        config_stats = {
            config: {
                "val_acc_mean": np.mean(val_accs),
                "val_acc_std": np.std(val_accs),
                "test_acc_mean": np.mean(config_test_accs[config]),
                "test_acc_std": np.std(config_test_accs[config]),
            }
            for config, val_accs in config_val_accs.items()
        }

        # Pick best config by test accuracy mean
        best_config = max(config_stats, key=lambda k: config_stats[k]["test_acc_mean"])
        return config_stats, best_config

    def read_csv(self, csv_path):
        """Read CSV with 6 columns: train_loss, train_acc, val_loss, val_acc, test_loss, test_acc."""
        train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = [], [], [], [], [], []
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for row in reader:
                train_losses.append(float(row[0]))
                train_accs.append(float(row[1]))
                val_losses.append(float(row[2]))
                val_accs.append(float(row[3]))
                test_losses.append(float(row[4]))
                test_accs.append(float(row[5]))
        return train_losses, train_accs, val_losses, val_accs, test_losses, test_accs

    def plot_metrics(self, subject, train_losses, train_accs, val_losses, val_accs, test_losses, test_accs, dataset, model):
        """Plot per-subject metrics for a given model in 2 rows (losses + accuracies)."""
        iterations = range(1, len(train_losses) + 1)
        plt.figure(figsize=(18, 8))

        # --- Row 1: Losses ---
        plt.subplot(2, 3, 1)
        plt.plot(iterations, train_losses, label='Train Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Subject {subject} - Train Loss')

        plt.subplot(2, 3, 2)
        plt.plot(iterations, val_losses, label='Val Loss', color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Subject {subject} - Validation Loss')

        plt.subplot(2, 3, 3)
        plt.plot(iterations, test_losses, label='Test Loss', color='brown')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Subject {subject} - Test Loss')

        # --- Row 2: Accuracies ---
        plt.subplot(2, 3, 4)
        plt.plot(iterations, train_accs, label='Train Acc', color='purple')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title(f'Subject {subject} - Train Accuracy')

        plt.subplot(2, 3, 5)
        plt.plot(iterations, val_accs, label='Val Acc', color='orange')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title(f'Subject {subject} - Validation Accuracy')

        plt.subplot(2, 3, 6)
        plt.plot(iterations, test_accs, label='Test Acc', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title(f'Subject {subject} - Test Accuracy')

        plt.tight_layout()
        fig_dir = "figs"
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f"{dataset}_{model}_sub{subject}.png")
        plt.savefig(fig_path)
        plt.close()

    def plot_summary_over_time(self, subject, models):
        """
        Plot Train/Val/Test Losses in row 1 and Accuracies in row 2 across models for the same subject.
        """
        plt.figure(figsize=(18, 8))

        # --- Row 1: Losses ---
        plt.subplot(2, 3, 1)
        for model in models:
            if subject not in self.summary_results["train_loss"][model]:
                continue
            train_losses = self.summary_results["train_loss"][model][subject]
            plt.plot(range(1, len(train_losses) + 1), train_losses, label=model, linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"Subject {subject} - Train Loss")
        plt.legend()

        plt.subplot(2, 3, 2)
        for model in models:
            if subject not in self.summary_results["val_loss"][model]:
                continue
            val_losses = self.summary_results["val_loss"][model][subject]
            plt.plot(range(1, len(val_losses) + 1), val_losses, label=model, linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"Subject {subject} - Validation Loss")
        plt.legend()

        plt.subplot(2, 3, 3)
        for model in models:
            if subject not in self.summary_results["test_loss"][model]:
                continue
            test_losses = self.summary_results["test_loss"][model][subject]
            plt.plot(range(1, len(test_losses) + 1), test_losses, label=model, linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"Subject {subject} - Test Loss")
        plt.legend()

        # --- Row 2: Accuracies ---
        plt.subplot(2, 3, 4)
        for model in models:
            if subject not in self.summary_results["train_acc"][model]:
                continue
            train_accs = self.summary_results["train_acc"][model][subject]
            plt.plot(range(1, len(train_accs) + 1), train_accs, label=model, linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title(f"Subject {subject} - Train Accuracy")
        plt.legend()

        plt.subplot(2, 3, 5)
        for model in models:
            if subject not in self.summary_results["val_acc"][model]:
                continue
            val_accs = self.summary_results["val_acc"][model][subject]
            plt.plot(range(1, len(val_accs) + 1), val_accs, label=model, linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title(f"Subject {subject} - Validation Accuracy")
        plt.legend()

        plt.subplot(2, 3, 6)
        for model in models:
            if subject not in self.summary_results["test_acc"][model]:
                continue
            test_accs = self.summary_results["test_acc"][model][subject]
            plt.plot(range(1, len(test_accs) + 1), test_accs, label=model, linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title(f"Subject {subject} - Test Accuracy")
        plt.legend()

        plt.tight_layout()
        fig_dir = "figs"
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f"{self.dataset}_{subject}_summary.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()

    def analyse_models(self, models):
        """Analyse all models across all subjects and plot results."""
        overall_best_models = []
        all_best_accuracies = []

        for model in models:
            self.summary_results["train_loss"][model] = {}
            self.summary_results["train_acc"][model] = {}
            self.summary_results["val_loss"][model] = {}
            self.summary_results["val_acc"][model] = {}
            self.summary_results["test_loss"][model] = {}
            self.summary_results["test_acc"][model] = {}

        for subject in self.subjects:
            print(f"\nAnalyzing subject: {subject}")
            subject_best_models = {}
            for model in models:
                model_path = os.path.join(self.dataset, model, f"subject_{subject}")
                if not os.path.exists(model_path):
                    print(f"Model path does not exist: {model_path}")
                    continue

                # Evaluate configs
                config_stats, best_config = self.find_best_config(model_path)
                best_acc = config_stats[best_config]["test_acc_mean"]
                best_std = config_stats[best_config]["test_acc_std"]
                subject_best_models[model] = {
                    'best_acc': best_acc,
                    'std': best_std,
                    'best_config': best_config
                }
                all_best_accuracies.append(best_acc)

                # Pick one run from the best config for plotting + summary
                for root, dirs, files in os.walk(best_config):
                    for file in files:
                        if file.endswith('.csv'):
                            csv_path = os.path.join(root, file)
                            train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = self.read_csv(csv_path)
                            self.plot_metrics(subject, train_losses, train_accs, val_losses, val_accs, test_losses, test_accs, self.dataset, model)

                            # Store full trajectories
                            self.summary_results["train_loss"][model][subject] = train_losses
                            self.summary_results["train_acc"][model][subject] = train_accs
                            self.summary_results["val_loss"][model][subject] = val_losses
                            self.summary_results["val_acc"][model][subject] = val_accs
                            self.summary_results["test_loss"][model][subject] = test_losses
                            self.summary_results["test_acc"][model][subject] = test_accs
                            break
                    break

                # Print all configs for this model
                print(f"\nConfigs for model {model}, subject {subject}:")
                for cfg, stats in config_stats.items():
                    print(f"  Config: {cfg}, "
                          f"TestAcc={stats['test_acc_mean']:.4f} ± {stats['test_acc_std']:.3f}")

            # Best model for this subject
            best_model = max(subject_best_models, key=lambda m: subject_best_models[m]['best_acc'])
            overall_best_models.append((subject, best_model,
                                        subject_best_models[best_model]['best_acc'],
                                        subject_best_models[best_model]['std']))

            # Print per-subject summary
            print(f"\nSummary for Subject {subject}:")
            for model, stats in subject_best_models.items():
                print(f"Model: {model}, Best Accuracy: {stats['best_acc']:.4f}, "
                      f"Std: {stats['std']:.4f}, Best Config: {stats['best_config']}")

            # Plot comparison across models for this subject
            self.plot_summary_over_time(subject, models)

        # Overall summary across subjects
        mean_best_acc = np.mean([acc for _, _, acc, _ in overall_best_models])
        std_best_acc = np.std([acc for _, _, acc, _ in overall_best_models])

        print("\n===== Overall Best Models Across Subjects =====")
        for subject, model, acc, std in overall_best_models:
            print(f"Subject {subject}: Best Model = {model}, Accuracy = {acc:.4f}, Std = {std:.4f}")

        print(f"\nMean Best Accuracy Across Models: {mean_best_acc:.4f}")
        print(f"Std Dev of Best Accuracy Across Models: {std_best_acc:.4f}")




class FindBestConfig:
    def __init__(self, dataset, subject="all"):
        self.dataset = dataset
        self.subject = subject if isinstance(subject, str) else str(subject)

    def get_max_val_acc(self, csv_path):
        """Extract max val_acc and corresponding test_acc from a CSV file."""
        val_accs, test_accs = [], []
        with open(csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                val_accs.append(float(row["val_accs"]))
                test_accs.append(float(row["test_accs"]))
        if not val_accs:
            return None, None
        val_accs_ma = np.convolve(val_accs, np.ones(5) / 5, mode="valid")
        max_index = np.argmax(val_accs_ma)
        return np.max(val_accs), test_accs[max_index]

    def find_best_config_per_window(self, model_path):
        """Return best config per window (dict)."""
        window_results = defaultdict(lambda: defaultdict(list))

        for root, dirs, files in os.walk(model_path):
            for file in files:
                if not file.endswith(".csv"):
                    continue
                csv_path = os.path.join(root, file)
                val_acc, test_acc = self.get_max_val_acc(csv_path)
                if val_acc is None:
                    continue

                config_path = os.path.dirname(csv_path)
                config_name = os.path.basename(config_path)

                # Extract window (e.g. "win1")
                parts = config_name.split("_")
                window = [p for p in parts if p.startswith("win")]
                if not window:
                    continue
                window = window[0]

                window_results[window][config_name].append(test_acc)

        # Pick best config per window
        best_per_window = {}
        for window, configs in window_results.items():
            config_means = {cfg: np.mean(accs) for cfg, accs in configs.items()}
            best_config = max(config_means, key=config_means.get)
            best_per_window[window] = (best_config, config_means[best_config])

        return best_per_window

    def get_best_config(self, model, window):
        """
        Return the best config string + accuracy for a given model, dataset, and window.
        """
        model_path = os.path.join('./results/',self.dataset, model, f"subject_{self.subject}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        best_per_window = self.find_best_config_per_window(model_path)
        if window not in best_per_window:
            raise ValueError(f"No configs found for {model}, {self.dataset}, {window}")

        return best_per_window[window]  # (best_config, mean_accuracy)


if __name__ == "__main__":
    subjects_cross = {'bci': ['all'],
                      'mamem': ['all'],
                      'bcicha': ['all']}
    
    dataset = ('mamem')
    # 'BaselineDeviationModelIdEmbedHeadLora',
    models = ['BaselineDeviationModelIdEmbedHeadLora']
    
    for subject in subjects_cross[dataset]:
        analyser = AnalyseResults(dataset, subject)
        analyser.analyse_models(models)


    #TODO: G



    # subjects_single = {
    #     'bci': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    #     'mamem': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #     'bcicha': [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
    # }
    # subjects_loso= {
    #     'bci': ['loso_1', 'loso_2', 'loso_3', 'loso_4', 'loso_5', 'loso_6', 'loso_7', 'loso_8', 'loso_9'],
    #     'mamem': ['loso_1', 'loso_2', 'loso_3', 'loso_4', 'loso_5', 'loso_6', 'loso_7', 'loso_8', 'loso_9', 'loso_10',
    #               'loso_11'],
    #     'bcicha': ['loso_2', 'loso_6', 'loso_7', 'loso_11', 'loso_12', 'loso_13', 'loso_14', 'loso_16', 'loso_17',
    #                'loso_18', 'loso_20', 'loso_21', 'loso_22', 'loso_23', 'loso_24', 'loso_26']
    # }

    # dataset = 'mamem'
    # # models = ['TCF']
    # models = ['BaselineDeviatBaselineDeviationModelIdEmbedHeadLora/subject_all/win1_bs32_lr0.001_wd0.01_dp0_llr1e-05_procTrue_encTrue_cutTrue_decTrue_predecFalse_loraTrue_sdTrueionModelIdEmbedHeadLora']
   

    # analyser = AnalyseResults(dataset, subjects_single[dataset])
    # analyser.analyse_models(models)


