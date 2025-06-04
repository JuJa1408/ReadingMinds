import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from scipy.spatial.distance import squareform
import pandas as pd
import torch.multiprocessing as mp

# ---------- Model ----------
class FNC2DCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(FNC2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(86528, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        #.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x#self.sigmoid(x)

# ---------- Dataset Builders ----------
def compute_fnc_from_icn(icn_array):
    C = np.corrcoef(icn_array)
    C = np.nan_to_num(C)
    return squareform(C, checks=False)

def build_fnc_dataset_with_ablated_icn(icn_data_dir, region_idx):
    samples = []
    for label_str, label in zip(["BP", "SZ"], [0, 1]):
        label_path = os.path.join(icn_data_dir, label_str)
        for subject in os.listdir(label_path):
            tc_path = os.path.join(label_path, subject, "icn_tc.npy")
            if not os.path.isfile(tc_path):
                continue
            tc = np.load(tc_path).astype(np.float32).T
            T = tc.shape[1]
            tc[region_idx] = np.random.randn(T).astype(np.float32)
            fnc = compute_fnc_from_icn(tc)
            fnc_matrix = squareform(fnc)[np.newaxis, :, :]
            samples.append((fnc_matrix.astype(np.float32), label))
    return samples

class AblatedFNC_Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        fnc_matrix, label = self.samples[idx]
        fnc_matrix = torch.tensor(fnc_matrix, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return fnc_matrix, label

class FNC_Dataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for group, label in zip(["BP", "SZ"], [0, 1]):
            group_dir = os.path.join(root_dir, group)
            for subject in os.listdir(group_dir):
                fnc_path = os.path.join(group_dir, subject, "fnc.npy")
                if os.path.isfile(fnc_path):
                    fnc_vector = np.load(fnc_path).astype(np.float32)
                    fnc_vector = fnc_vector.flatten()
                    #fnc_vector = np.maximum(fnc_vector, 0)
                    fnc_matrix = squareform(fnc_vector)
                    fnc_matrix = fnc_matrix[np.newaxis, :, :]
                    self.samples.append((fnc_matrix, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        fnc_matrix, label = self.samples[idx]
        fnc_matrix = torch.tensor(fnc_matrix, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return fnc_matrix, label
import pandas as pd

def save_roc_data_to_csv(roc_data, filename):
    # Flatten the ROC data for CSV (one row per FPR/TPR point per fold)
    rows = []
    for entry in roc_data:
        fold = entry['fold']
        auc_val = entry['auc']
        for fpr_val, tpr_val in zip(entry['fpr'], entry['tpr']):
            rows.append({'fold': fold, 'fpr': fpr_val, 'tpr': tpr_val, 'auc': auc_val})
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"ROC data saved to {filename}")

# ---------- Cross-validation ----------
def cross_validate_2dcnn_save_models(model_class, dataset, num_folds=3, num_epochs=5, device='cpu'):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_models = []
    fold_val_indices = []
    fold_results = []
    roc_data = []
    best_val_accuracy = 0.0
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"  Starting fold {fold+1}/{num_folds}...")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(num_epochs):
            print(f"    Epoch {epoch+1}/{num_epochs}...")
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Save model and validation indices
        fold_models.append(model.state_dict())
        fold_val_indices.append(val_idx)

        # Evaluate on validation set
        model.eval()
        all_labels = []
        all_probs = []
        val_losses = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                all_probs.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                val_losses.append(loss.item() * len(labels))
        val_loss = np.sum(val_losses) / len(val_idx)
        pred_labels = (np.array(all_probs) > 0.5).astype(float)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        results = {
            'auc': auc(*roc_curve(all_labels, all_probs)[:2]),
            'accuracy': accuracy_score(all_labels, pred_labels),
            'precision': precision_score(all_labels, pred_labels, zero_division=0),
            'recall': recall_score(all_labels, pred_labels, zero_division=0),
            'f1': f1_score(all_labels, pred_labels, zero_division=0),
            'val_loss': val_loss
        }
        print(f"  Fold {fold+1}/{num_folds} results: {results}")
        roc_data.append({'fold': fold+1, 'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': results['auc']})

        if accuracy_score(all_labels, pred_labels) > best_val_accuracy:
            best_val_accuracy = accuracy_score(all_labels, pred_labels)
            best_model_state = model.state_dict()

        fold_results.append(results)

    if best_model_state:
        torch.save(best_model_state, "best_model_2DCNN.pth")
        print("Best model (by val accuracy) saved to 'best_model_2DCNN.pth'")

    save_roc_data_to_csv(roc_data, "roc_data_2dcnn.csv")

    return fold_models, fold_val_indices, fold_results

def ablation_on_val_only_2dcnn(model_class, dataset, fold_models, fold_val_indices, region_idx, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()
    ablation_metrics = []

    for fold, (model_state, val_idx) in enumerate(zip(fold_models, fold_val_indices)):
        # Prepare validation set with shuffled region
        import copy
        val_data = copy.deepcopy([dataset.samples[i][0] for i in val_idx])
        val_labels = [dataset.samples[i][1] for i in val_idx]
        # Shuffle the region_idx for each sample
        for arr in val_data:
            arr[0, region_idx, :] = np.random.randn(arr.shape[2]).astype(np.float32)
            arr[0, :, region_idx] = np.random.randn(arr.shape[1]).astype(np.float32)
        x_val = torch.tensor(np.stack(val_data), dtype=torch.float32).to(device)
        y_val = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1).to(device)

        # Load model
        model = model_class().to(device)
        model.load_state_dict(model_state)
        model.eval()

        with torch.no_grad():
            outputs = model(x_val)
            loss = criterion(outputs, y_val)
            probs = outputs.cpu().numpy().flatten()
            labels = y_val.cpu().numpy().flatten()
        pred_labels = (probs > 0.5).astype(float)
        ablation_metrics.append({
            'auc': auc(*roc_curve(labels, probs)[:2]),
            'accuracy': accuracy_score(labels, pred_labels),
            'precision': precision_score(labels, pred_labels, zero_division=0),
            'recall': recall_score(labels, pred_labels, zero_division=0),
            'f1': f1_score(labels, pred_labels, zero_division=0),
            'val_loss': loss.item()
        })

    avg_metrics = {k: np.mean([fold[k] for fold in ablation_metrics]) for k in ablation_metrics[0]}
    return avg_metrics

# ---------- Evaluation Wrapper ----------
def evaluate_fnc_ablation(region_idx, icn_data_dir, baseline, folds=5, epochs=5, device='cpu'):
    print(f"Evaluating region {region_idx}...")
    samples = build_fnc_dataset_with_ablated_icn(icn_data_dir, region_idx)
    dataset = AblatedFNC_Dataset(samples)
    results, _, _ = cross_validate_2dcnn(FNC2DCNN, dataset, num_folds=folds, num_epochs=epochs, device=device)
    print(f"Evaluation for region {region_idx} done.")
    return {
        'Region': region_idx,
        'Δ Accuracy': results['accuracy'] - baseline['accuracy'],
        'Δ AUC': results['auc'] - baseline['auc'],
        'Δ Precision': results['precision'] - baseline['precision'],
        'Δ Recall': results['recall'] - baseline['recall'],
        'Δ F1': results['f1'] - baseline['f1']
    }

# ---------- Main Script ----------
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    icn_data_dir = "data/data/train"
    RUN_BASELINE = True

    if RUN_BASELINE:
        print("Running baseline evaluation...")
        baseline_dataset = FNC_Dataset(icn_data_dir)
        fold_models, fold_val_indices, baseline_fold_results = cross_validate_2dcnn_save_models(
            FNC2DCNN, baseline_dataset, num_folds=10, num_epochs=20, device='cpu'
        )
        # Save for future use
        import pickle
        with open("fold_models_2dcnn.pkl", "wb") as f:
            pickle.dump(fold_models, f)
        with open("fold_val_indices_2dcnn.pkl", "wb") as f:
            pickle.dump(fold_val_indices, f)
        pd.DataFrame(baseline_fold_results).to_csv("baseline_fold_results_2dcnn.csv", index=False)
    else:
        print("Loading baseline models and indices...")
        baseline_dataset = FNC_Dataset(icn_data_dir)
        import pickle
        with open("fold_models_2dcnn.pkl", "rb") as f:
            fold_models = pickle.load(f)
        with open("fold_val_indices_2dcnn.pkl", "rb") as f:
            fold_val_indices = pickle.load(f)
        baseline_fold_results = pd.read_csv("baseline_fold_results_2dcnn.csv").to_dict(orient='records')

    baseline_avg = {k: np.mean([fold[k] for fold in baseline_fold_results]) for k in baseline_fold_results[0]}

    print("Starting ICN ablation evaluation...")
    ablation_results = []
    num_repeats = 10  # Number of times to repeat each ablation

for region_idx in range(105):
    print(f"Evaluating region {region_idx} ablation...")
    repeat_metrics = []
    for repeat in range(num_repeats):
        ablation_metrics = ablation_on_val_only_2dcnn(
            FNC2DCNN, baseline_dataset, fold_models, fold_val_indices, region_idx, device='cpu'
        )
        repeat_metrics.append(ablation_metrics)
    # Average the metrics over repeats
    avg_metrics = {k: np.mean([m[k] for m in repeat_metrics]) for k in repeat_metrics[0]}
    result = {
        'Region': region_idx,
        'Δ Accuracy': (avg_metrics['accuracy'] - baseline_avg['accuracy']) / baseline_avg['accuracy'],
        'Δ AUC': (avg_metrics['auc'] - baseline_avg['auc']) / baseline_avg['auc'],
        'Δ Precision': (avg_metrics['precision'] - baseline_avg['precision']) / baseline_avg['precision'],
        'Δ Recall': (avg_metrics['recall'] - baseline_avg['recall']) / baseline_avg['recall'],
        'Δ F1': (avg_metrics['f1'] - baseline_avg['f1']) / baseline_avg['f1'],
        'Δ Val Loss': (avg_metrics['val_loss'] - baseline_avg['val_loss']) / baseline_avg['val_loss']
    }
    ablation_results.append(result)

df = pd.DataFrame(ablation_results)
df.sort_values(by='Δ Val Loss', inplace=True)
df.to_csv("fnc_icn_ablation_valonly_normloss_multiplerandom.csv", index=False)
print("Ablation results saved to 'fnc_icn_ablation_valonly_normloss_multiplerandom.csv'")
