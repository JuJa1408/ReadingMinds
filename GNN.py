from torch_geometric.data import Data
from torch_geometric.data import Dataset
from scipy.spatial.distance import squareform
import numpy as np
import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import roc_curve, auc



class FNC_ICN_GraphDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.graphs = []
        label_map = {"BP": 0, "SZ": 1}
        max_features = 0
        raw_data = []

        # First pass: Determine the maximum feature length
        for group in ["BP", "SZ"]:
            group_dir = os.path.join(root_dir, group)
            label = label_map[group]
            for subject in os.listdir(group_dir):
                subject_dir = os.path.join(group_dir, subject)
                fnc_path = os.path.join(subject_dir, "fnc.npy")
                icn_path = os.path.join(subject_dir, "icn_tc.npy")

                if os.path.isfile(fnc_path) and os.path.isfile(icn_path):
                    # Load node features (icn)
                    node_features = np.load(icn_path).T  # Transpose to shape (105, T)
                    max_features = max(max_features, node_features.shape[1])  # Update max feature length
                    raw_data.append((fnc_path, node_features, label))

        # Second pass: Pad and create graphs
        for fnc_path, node_features, label in raw_data:
            
            # Load node features (icn)
            # Pad node features to max_features
            padded_features = np.zeros((node_features.shape[0], max_features), dtype=np.float32)
            padded_features[:, :node_features.shape[1]] = node_features

            # Normalize node features (optional)
            padded_features = (padded_features - padded_features.mean(axis=1, keepdims=True)) / (padded_features.std(axis=1, keepdims=True) + 1e-5)
            

            
            # Load edge weights (fnc)
            fnc_vector = np.load(fnc_path).astype(np.float32)
            fnc_vector = fnc_vector.flatten() 
            #fnc_vector = np.maximum(fnc_vector, 0)  # Set all negative values to 0
            fnc_matrix = squareform(fnc_vector)

            # Create edge_index (fully connected graph)
            num_nodes = fnc_matrix.shape[0]
            node_idx_features = np.arange(num_nodes).reshape(-1, 1).astype(np.float32)  # shape: (num_nodes, 1)
            edge_index = torch.tensor(
                np.array(np.meshgrid(range(num_nodes), range(num_nodes))).reshape(2, -1),
                dtype=torch.long,
            )

            # Create edge weights
            edge_weights = torch.tensor(fnc_matrix.flatten(), dtype=torch.float32)

            # Convert node features to tensor
            x = torch.tensor(padded_features, dtype=torch.float32)
            #x = torch.tensor(node_idx_features, dtype=torch.float32)

            # Create label tensor
            y = torch.tensor([label], dtype=torch.float32)

            # Create a graph object
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=y)
            self.graphs.append(graph)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]
    
    from torch_geometric.loader import DataLoader

dataset = FNC_ICN_GraphDataset("data/data/train")

# Check the first graph
print(dataset[0])  

# Check dataset length
print(f"Number of graphs in the dataset: {len(dataset)}")

from torch.utils.data import random_split

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

from torch_geometric.loader import DataLoader

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Check a batch
for batch in train_loader:
    print(batch)  # This should print a batch of graphs
    break

from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import torch.nn.functional as F
import torch

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super(GNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=False)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim * 4)
        self.fc1 = torch.nn.Linear(hidden_dim, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
def plot_prediction_heatmap(true_labels, predicted_probs):
    # Create a 2D histogram (heatmap)
    heatmap, xedges, yedges = np.histogram2d(predicted_probs, true_labels, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Plot the diagonal line for perfect predictions
    plt.plot([0, 1], [0, 1], "r--", label="Perfect Prediction")

    # Plot the heatmap
    plt.imshow(heatmap.T, extent=extent, origin="lower", aspect="auto", cmap="Blues")
    plt.colorbar(label="Density")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Label")
    plt.title("Prediction Heatmap")
    plt.legend()
    plt.grid()
    plt.show()

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

def cross_validate_gnn(model_class, dataset, num_folds=10, num_epochs=30, device='cpu'):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    roc_data = []
    best_val_accuracy = 0.0
    best_model_state = None
    fold_models = []
    fold_val_indices = []
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{num_folds}")
        
        # Split dataset into training and validation sets
        train_subset = [dataset[i] for i in train_idx]
        val_subset = [dataset[i] for i in val_idx]
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

        # Initialize model, optimizer, and loss function
        model = model_class(input_dim=dataset[0].x.shape[1], hidden_dim=32, output_dim=1).to(device)
        """"
        print(model)

        # Visualize one input graph
        graph = dataset[0]  # Get the first graph in the dataset

        print("Graph Structure:")
        print(f"Number of nodes: {graph.x.shape[0]}")
        print(f"Node feature dimension: {graph.x.shape[1]}")
        print(f"Edge index shape: {graph.edge_index.shape}")
        print(f"Edge attributes shape: {graph.edge_attr.shape}")
        print(f"Graph label: {graph.y}")
        
        import torch
        import networkx as nx
        import matplotlib.pyplot as plt_
        from torch_geometric.utils import to_networkx

        # Convert PyG graph to NetworkX, including edge attributes
        nx_graph = to_networkx(graph, to_undirected=True)
        for i, (u, v) in enumerate(nx_graph.edges()):
            nx_graph[u][v]['edge_attr'] = float(graph.edge_attr[i])  # store scalar edge attribute

        # Select a small subset of nodes (e.g., first 10)
        subset_nodes = list(range(10))
        subgraph = nx_graph.subgraph(subset_nodes)

        # Layout for the subgraph
        pos = nx.spring_layout(subgraph, seed=42)

        # Create a figure and axes explicitly
        fig, ax = plt_.subplots(figsize=(8, 6))
        # Extract feature values for label text
        # Access the full node feature matrix
        node_features = graph.x  # shape: [105, 234]

        # Print the first few features for each node in the subset
        for node_id in subset_nodes:
            features = node_features[node_id][:10]  # adjust number of features shown
            print(f"Node {node_id} features (first 10 dims): {features.tolist()}")

        # Draw the subgraph with node labels    

        # Draw nodes and edges
        nx.draw(subgraph, pos, ax=ax, with_labels=True,
                node_color="skyblue", edge_color="gray",
                node_size=600, font_size=10)

        # Extract edge labels from attributes
        edge_labels = nx.get_edge_attributes(subgraph, 'edge_attr')
        edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}

        # Draw edge labels
        nx.draw_networkx_edge_labels(subgraph, pos, ax=ax,
                                    edge_labels=edge_labels,
                                    font_size=8, label_pos=0.5)

        # Title and show
        ax.set_title("Subgraph with Edge Labels")
        plt_.tight_layout()
        plt_.show()


        #break
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out.squeeze(), batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader):.4f}")

        fold_models.append(model.state_dict())
        fold_val_indices.append(val_idx)

        # Evaluate the model
        model.eval()
        all_labels = []
        all_probs = []
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                probs = torch.sigmoid(out.squeeze())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                val_losses.append(loss.item() * batch.num_graphs)

        # Compute metrics
        val_loss = np.sum(val_losses) / len(val_idx)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        val_accuracy = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(float))
        val_precision = precision_score(all_labels, (np.array(all_probs) > 0.5).astype(float), zero_division=0)
        val_recall = recall_score(all_labels, (np.array(all_probs) > 0.5).astype(float), zero_division=0)
        val_f1 = f1_score(all_labels, (np.array(all_probs) > 0.5).astype(float), zero_division=0)

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

        torch.save(model.state_dict(), f"GNN_model_fold_{fold}.pth")
        np.save(f"GNN_val_indices_fold_{fold}.npy", np.array(val_idx))
        pd.DataFrame(fold_results).to_csv("GNN_fold_results.csv", index=False)
        # Store ROC data for this fold
        roc_data.append({'fold': fold + 1, 'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc})
    """
    # Compute average metrics across all folds
    avg_results = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0]}
    print("\nAverage Results Across Folds:")
    for metric, value in avg_results.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    true_labels = np.array(all_labels)  # Replace with your true labels
    predicted_probs = np.array(all_probs)  # Replace with your predicted probabilities
    plot_prediction_heatmap(true_labels, predicted_probs)
    import pandas as pd
    fold_results = pd.DataFrame(fold_results)
    fold_results.to_csv("fold_results_GNN_edgeonly.csv", index=False)
    """
    if best_model_state:
        torch.save(best_model_state, "best_model_GNN_2005.pth")
        print("Best model (by val accuracy) saved to 'best_model_GNN_2005.pth'")

    #return avg_results, roc_data, fold_results
    save_roc_data_to_csv(roc_data, "roc_data_gnn.csv")

    return fold_models, fold_val_indices, fold_results

def ablation_on_val_only_gnn(model_class, dataset, fold_models, fold_val_indices, node_idx, device='cpu'):
    criterion = torch.nn.BCEWithLogitsLoss()
    ablation_metrics = []

    for fold, (model_state, val_idx) in enumerate(zip(fold_models, fold_val_indices)):
        # Deepcopy to avoid modifying original data
        import copy
        val_subset = [copy.deepcopy(dataset[i]) for i in val_idx]
        # Shuffle the node_idx for each graph in the validation set
        for graph in val_subset:
            # Replace node feature at node_idx with random noise
            graph.x[node_idx] = torch.randn_like(graph.x[node_idx])
            node_features = graph.x.cpu().numpy()
            fnc_matrix = np.corrcoef(node_features)
            graph.edge_attr = torch.tensor(fnc_matrix.flatten(), dtype=torch.float32)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

        # Load model
        model = model_class(input_dim=dataset[0].x.shape[1], hidden_dim=32, output_dim=1).to(device)
        model.load_state_dict(model_state)
        model.eval()

        all_labels = []
        all_probs = []
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                probs = torch.sigmoid(out.squeeze())
                loss = criterion(out.squeeze(), batch.y)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                val_losses.append(loss.item() * batch.num_graphs)
        val_loss = np.sum(val_losses) / len(val_idx)
        pred_labels = (np.array(all_probs) > 0.5).astype(float)
        ablation_metrics.append({
            'auc': auc(*roc_curve(all_labels, all_probs)[:2]),
            'accuracy': accuracy_score(all_labels, pred_labels),
            'precision': precision_score(all_labels, pred_labels, zero_division=0),
            'recall': recall_score(all_labels, pred_labels, zero_division=0),
            'f1': f1_score(all_labels, pred_labels, zero_division=0),
            'val_loss': val_loss
        })

    avg_metrics = {k: np.mean([fold[k] for fold in ablation_metrics]) for k in ablation_metrics[0]}
    return avg_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Perform 10-fold cross-validation and get ROC data
#avg_results, roc_data, fold_results = cross_validate_gnn(GNN, dataset, num_folds=10, num_epochs=20, device=device)

fold_models, fold_val_indices, baseline_fold_results = cross_validate_gnn(
    GNN, dataset, num_folds=10, num_epochs=20, device=device
)
baseline_avg = {k: np.mean([fold[k] for fold in baseline_fold_results]) for k in baseline_fold_results[0]}
print("Baseline average results across folds:")
for metric, value in baseline_avg.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# Ablation: For each node, shuffle only validation set and use saved models
ablation_results = []
for node_idx in range(dataset[0].x.shape[0]):
    print(f"Evaluating node {node_idx} ablation...")
    ablation_metrics = ablation_on_val_only_gnn(
        GNN, dataset, fold_models, fold_val_indices, node_idx, device=device
    )
    result = {
        'Node': node_idx,
        'Δ Accuracy': (ablation_metrics['accuracy'] - baseline_avg['accuracy'])/baseline_avg['accuracy'],
        'Δ AUC': (ablation_metrics['auc'] - baseline_avg['auc'])/baseline_avg['auc'],
        'Δ Precision': (ablation_metrics['precision'] - baseline_avg['precision'])/baseline_avg['precision'],
        'Δ Recall': (ablation_metrics['recall'] - baseline_avg['recall'])/baseline_avg['recall'],
        'Δ F1': (ablation_metrics['f1'] - baseline_avg['f1'])/baseline_avg['f1'],
        'Δ Val Loss': (ablation_metrics['val_loss'] - baseline_avg['val_loss'])/baseline_avg['val_loss']
    }
    ablation_results.append(result)

import pandas as pd
df_ablation = pd.DataFrame(ablation_results)
df_ablation.sort_values(by='Δ Val Loss', ascending=False, inplace=True)
df_ablation.to_csv("gnn_node_ablation_valonly_normloss_newarchitecture.csv", index=False)
print("Ablation results saved to 'gnn_node_ablation_valonly_normloss_newarchitecture.csv'")

"""""
import seaborn as sns
# Plot the ROC curves for all folds
def plot_roc_curves_with_mean(roc_data):
    plt.figure(figsize=(10, 8))

    # Arrays to store all FPR and TPR values for averaging
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(all_fpr)

    # Plot individual ROC curves
    for i, (fpr, tpr, roc_auc) in enumerate(roc_data):
        plt.plot(fpr, tpr, lw=1, alpha=0.4, label=f'Fold {i + 1} (AUC = {roc_auc:.2f})')
        # Interpolate TPR values for consistent averaging
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr[0] = 0.0  # Ensure the curve starts at (0, 0)

    # Compute the mean and standard deviation of TPR
    mean_tpr /= len(roc_data)
    mean_tpr[-1] = 1.0  # Ensure the curve ends at (1, 1)
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

    # Add labels, title, and legend
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves with Mean and Variance (10-Fold Cross-Validation)')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# Call the function to plot the improved ROC curves
#plot_roc_curves_with_mean(roc_data)
"""
"""
# Plot training and validation losses
plt.plot(train_losses[1:], label='Train Loss')
plt.plot(val_losses[1:], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss of the GNN Model")
plt.grid(True)
plt.show()


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve_with_auc(model, val_loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)  # Logits output
            probs = torch.sigmoid(out.squeeze())  # Convert logits to probabilities
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# Plot the ROC curve for the validation set
plot_roc_curve_with_auc(model, val_loader, device)
"""