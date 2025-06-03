# ICN 1D CNN Training Notebook (Improved Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Subset
import torch
import numpy as np

# DEVICE SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# MODEL DEFINITION (NO SIGMOID!)
class CNN1DClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(105, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization for conv1
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)  # Batch normalization for conv2
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x).squeeze(-1)  # logits output
        return x

# PREPROCESSING FUNCTION
def preprocess_icn(icn_array):
    # Normalize across time dimension
    icn_array = (icn_array - icn_array.mean(axis=1, keepdims=True)) / (icn_array.std(axis=1, keepdims=True) + 1e-5)
    return icn_array

# DATASET DEFINITION
class ICN_Dataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        self.max_len = 0
        raw_data = []

        for label, label_name in enumerate(['BP', 'SZ']):
            label_path = os.path.join(data_dir, label_name)
            for subject in os.listdir(label_path):
                tc_path = os.path.join(label_path, subject, "icn_tc.npy")
                if os.path.exists(tc_path):
                    tc = np.load(tc_path)  # shape: (T, 105)
                    tc = tc.T  # shape: (105, T)
                    self.max_len = max(self.max_len, tc.shape[1])
                    raw_data.append((tc.astype(np.float32), label))

        # Pad and preprocess
        for tc, label in raw_data:
            padded = np.zeros((105, self.max_len), dtype=np.float32)
            padded[:, :tc.shape[1]] = tc
            padded = preprocess_icn(padded)
            self.data.append(padded)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

# SETUP DATASET
full_dataset = ICN_Dataset("data/data/train")  # Adjust path accordingly

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np

import pandas as pd

def save_roc_data_to_csv(roc_data, filename):
    # roc_data: list of dicts with keys 'fold', 'fpr', 'tpr', 'auc'
    rows = []
    for entry in roc_data:
        fold = entry['fold']
        auc_val = entry['auc']
        for fpr_val, tpr_val in zip(entry['fpr'], entry['tpr']):
            rows.append({'fold': fold, 'fpr': fpr_val, 'tpr': tpr_val, 'auc': auc_val})
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"ROC data saved to {filename}")

def cross_validate_1dcnn_save_models(model_class, dataset, num_folds=10, num_epochs=20, device='cpu', no_cv = False):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    roc_data = []
    best_val_accuracy = 0.0
    best_model_state = None
    fold_models = []
    fold_val_indices = []
    fold_val_labels = []
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{num_folds}")
        
        # Split dataset into training and validation sets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

        # Initialize model, optimizer, and loss function
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.float().to(device)

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
        # Save model and validation indices/labels
        fold_models.append(model.state_dict())
        fold_val_indices.append(val_idx)
        val_labels = [dataset.labels[i] for i in val_idx]
        fold_val_labels.append(val_labels)

        # Evaluate the model
        model.eval()
        all_labels = []
        all_probs = []
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.float().to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(y_batch.cpu().numpy().flatten())
                val_losses.append(loss.item() * len(y_batch))
        val_loss = np.sum(val_losses) / len(val_idx)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        val_accuracy = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(float))
        val_precision = precision_score(all_labels, (np.array(all_probs) > 0.5).astype(float), zero_division=0)
        val_recall = recall_score(all_labels, (np.array(all_probs) > 0.5).astype(float), zero_division=0)
        val_f1 = f1_score(all_labels, (np.array(all_probs) > 0.5).astype(float), zero_division=0)
        roc_data.append({'fold': fold + 1, 'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc})


        print(f"Fold {fold + 1} Results: AUC: {roc_auc:.4f}, Accuracy: {val_accuracy:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

        # Store results for this fold
        fold_results.append({
            'auc': roc_auc,
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'val_loss': val_loss
        })

    """"
        # Store ROC data for this fold
        roc_data.append((fpr, tpr, roc_auc))

        if no_cv:
            break

    # Compute average metrics across all folds
    avg_results = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0]}
    print("\nAverage Results Across Folds:")
    for metric, value in avg_results.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    """    
    import pandas as pd
    fold_results = pd.DataFrame(fold_results)
    fold_results.to_csv("fold_results_1DCNN.csv", index=False)
    save_roc_data_to_csv(roc_data, "roc_data_1dcnn.csv")

    if best_model_state:
        torch.save(best_model_state, "best_model_1DCNN.pth")
        print("Best model (by val accuracy) saved to 'best_model_1DCNN.pth'")

    #return avg_results, roc_data, fold_results
    
    return fold_models, fold_val_indices, fold_val_labels, fold_results





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Perform 10-fold cross-validation
#avg_results, roc_data, fold_results = cross_validate_1dcnn(CNN1DClassifier, full_dataset, num_folds=10, num_epochs=20, device=device)


def plot_roc_curves_with_mean(roc_data):
    plt.figure(figsize=(10, 8))

    # Arrays to store all FPR and TPR values for averaging
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(all_fpr)

    # Plot individual ROC curves
    for i, (fpr, tpr, roc_auc) in enumerate(roc_data):
        plt.plot(fpr, tpr, lw=1, alpha=0.4, label=f'Fold {i + 1} (AUC = {roc_auc:.2f})')
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    # Compute the mean and standard deviation of TPR
    mean_tpr /= len(roc_data)
    mean_tpr[-1] = 1.0
    std_tpr = np.std([np.interp(all_fpr, fpr, tpr) for fpr, tpr, _ in roc_data], axis=0)

    # Plot the mean ROC curve
    mean_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='blue', lw=2, label=f'Mean ROC (AUC = {mean_auc:.2f})')

    # Plot the standard deviation as a shaded area
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(all_fpr, tpr_lower, tpr_upper, color='blue', alpha=0.2, label='± 1 Std. Dev.')

    # Plot the diagonal line for random guessing
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guessing')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('1D CNN ROC Curves with Mean and Variance (10-Fold Cross-Validation)')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# Plot the ROC curves
#plot_roc_curves_with_mean(roc_data)

import pandas as pd
import copy

def shuffle_icn_region(dataset, icn_index):
    # Clone and perturb a copy of the dataset
    new_dataset = copy.deepcopy(dataset)
    for i in range(len(new_dataset.data)):
        region_length = new_dataset.data[i].shape[1]
        noise = np.random.randn(region_length).astype(np.float32)
        new_dataset.data[i][icn_index] = noise
    return new_dataset

def ablation_on_val_only(model_class, dataset, fold_models, fold_val_indices, icn_idx, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()
    ablation_metrics = []

    for fold, (model_state, val_idx) in enumerate(zip(fold_models, fold_val_indices)):
        # Prepare validation set with shuffled ICN
        val_subset = Subset(dataset, val_idx)
        # Deepcopy to avoid modifying original data
        import copy
        val_data = copy.deepcopy([dataset.data[i] for i in val_idx])
        for arr in val_data:
            region_length = arr.shape[1]
            arr[icn_idx] = np.random.randn(region_length).astype(np.float32)
        # Prepare DataLoader
        x_val = torch.tensor(np.stack(val_data)).to(device)
        y_val = torch.tensor([dataset.labels[i] for i in val_idx]).float().to(device)

        # Load model
        model = model_class().to(device)
        model.load_state_dict(model_state)
        model.eval()

        # Evaluate
        with torch.no_grad():
            logits = model(x_val)
            loss = criterion(logits, y_val)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels = y_val.cpu().numpy().flatten()
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        val_accuracy = accuracy_score(labels, (probs > 0.5).astype(float))
        val_precision = precision_score(labels, (probs > 0.5).astype(float), zero_division=0)
        val_recall = recall_score(labels, (probs > 0.5).astype(float), zero_division=0)
        val_f1 = f1_score(labels, (probs > 0.5).astype(float), zero_division=0)

        ablation_metrics.append({
            'auc': roc_auc,
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'val_loss': loss.item()
        })

    # Average over folds
    avg_metrics = {k: np.mean([fold[k] for fold in ablation_metrics]) for k in ablation_metrics[0]}
    return avg_metrics

# Run baseline to get reference metrics
#baseline_results, _, _ = cross_validate_1dcnn(CNN1DClassifier, full_dataset, num_folds=10, num_epochs=20, device=device)


import pickle

RUN_BASELINE = True  # Set to True to retrain, False to only run ablation
NUM_REPEATS = 10      # Number of ablation repeats per ICN

if RUN_BASELINE:
    # Train and save models, indices, and results
    fold_models, fold_val_indices, fold_val_labels, baseline_fold_results = cross_validate_1dcnn_save_models(
        CNN1DClassifier, full_dataset, num_folds=10, num_epochs=20, device=device
    )
    # Save models and indices
    with open("fold_models_1dcnn.pkl", "wb") as f:
        pickle.dump(fold_models, f)
    with open("fold_val_indices_1dcnn.pkl", "wb") as f:
        pickle.dump(fold_val_indices, f)
    baseline_fold_results.to_csv("baseline_fold_results_1dcnn.csv", index=False)
else:
    # Load models and indices
    with open("fold_models_1dcnn.pkl", "rb") as f:
        fold_models = pickle.load(f)
    with open("fold_val_indices_1dcnn.pkl", "rb") as f:
        fold_val_indices = pickle.load(f)
    baseline_fold_results = pd.read_csv("baseline_fold_results_1dcnn.csv")

baseline_avg = baseline_fold_results.mean().to_dict()

# Repeat ablation for each ICN
ablation_results = []
for icn_idx in range(105):
    print(f"Evaluating ICN {icn_idx} ablation...")
    repeat_metrics = []
    for repeat in range(NUM_REPEATS):
        ablation_metrics = ablation_on_val_only(
            CNN1DClassifier, full_dataset, fold_models, fold_val_indices, icn_idx, device=device
        )
        repeat_metrics.append(ablation_metrics)
    # Collect all metrics for this ICN
    metrics_keys = repeat_metrics[0].keys()
    stats = {}
    for k in metrics_keys:
        values = [m[k] for m in repeat_metrics]
        stats[f'{k}_mean'] = np.mean(values)
        stats[f'{k}_std'] = np.std(values)
        stats[f'{k}_var'] = np.var(values)
        stats[f'{k}_min'] = np.min(values)
        stats[f'{k}_max'] = np.max(values)
        stats[f'{k}_median'] = np.median(values)
    # Calculate delta values (mean ablation - baseline) / baseline
    result = {
        'ICN': icn_idx,
        'Δ Accuracy': (stats['accuracy_mean'] - baseline_avg['accuracy']) / baseline_avg['accuracy'],
        'Δ AUC': (stats['auc_mean'] - baseline_avg['auc']) / baseline_avg['auc'],
        'Δ Precision': (stats['precision_mean'] - baseline_avg['precision']) / baseline_avg['precision'],
        'Δ Recall': (stats['recall_mean'] - baseline_avg['recall']) / baseline_avg['recall'],
        'Δ F1': (stats['f1_mean'] - baseline_avg['f1']) / baseline_avg['f1'],
        'Δ Val Loss': (stats['val_loss_mean'] - baseline_avg['val_loss']) / baseline_avg['val_loss'],
    }
    # Add all stats to result for later analysis
    for k in stats:
        result[k] = stats[k]
    ablation_results.append(result)

# Save to CSV
df_ablation = pd.DataFrame(ablation_results)
df_ablation.sort_values(by='Δ Val Loss', ascending=False, inplace=True)
df_ablation.to_csv("icn_ablation_results_valonly_normloss_multiplerandom.csv", index=False)
print("Ablation results saved to 'icn_ablation_results_valonly_normloss_multiplerandom.csv'")