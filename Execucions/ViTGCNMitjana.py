import os
import numpy as np
import glob
import cv2 as cv
from ismember import ismember
from random import shuffle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score, roc_curve, auc
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp


data = np.load('/fhome/pmarti/TFGPau/LargetissueDades_48_Norm.npz', allow_pickle=True)
data1 = np.load('/fhome/pmarti/TFGPau/DBLarge_FeatMatNew_Reduit_norm.npz/DBLarge_FeatMatNew_Reduit_norm.npz', allow_pickle=True)
# data2 = np.load('/fhome/pmarti/TFGPau/featuresHosp.npz', allow_pickle=True)

features_list = data1['features_list'] 
attn_matrices = data1['attn_matrices']

# features_list_h = data2['features_list_ho'] 
# adj_list_h = data2['adj_mean_list_ho']

y_hosp = data['y_hosp']
y_no_hosp = data['y_no_hosp']

PatID_no_hosp = data['PatID_no_hosp']

def collate_fn(batch):
    return Batch.from_data_list(batch)


# ================= STEP 2: Define GAT Model =================
class GCN(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, use_edge_weight=True):
        super().__init__()
        self.use_edge_weight = use_edge_weight
        self.gcn1 = GCNConv(in_ch, hidden_ch)
        self.gcn2 = GCNConv(hidden_ch, hidden_ch)
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(hidden_ch, hidden_ch),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_ch),
            torch.nn.Linear(hidden_ch, out_ch)
        )

    def forward(self, x, edge_index, edge_weight, batch_idx):
        if self.use_edge_weight and edge_weight is not None:
            x = self.gcn1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.gcn2(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.gcn1(x, edge_index)  # sense edge_weight
            x = F.relu(x)
            x = self.gcn2(x, edge_index)

        g = global_mean_pool(x, batch_idx)
        return self.lin(g), None


def attn_to_graph(attn_matrices, threshold=0.0):
    adj = attn_matrices.mean(axis=0)  # (196, 196)
    mask = adj > threshold
    edge_index = np.array(mask.nonzero())  # (2, num_edges)
    edge_weight = adj[mask]  # vector 1D (num_edges,)
    return edge_index, edge_weight

# Prepare adjacency lists with edge_index and edge_weight
threshold = 0.0
adj_list = []
edge_weight_list = []
for i in range(len(attn_matrices)):
    edge_index_np, edge_weight_np = attn_to_graph(attn_matrices[i], threshold=threshold)

    # Convert edge_index to [2, num_edges] correct shape
    edge_index = torch.tensor(edge_index_np, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_np, dtype=torch.float).view(-1)

    adj_list.append(edge_index)
    edge_weight_list.append(edge_weight)

# Stratified K-Fold Validation
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2)
metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": []}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_loss = []

best_val_auc = 0
best_model_state = None
all_loss = []
roc_data_per_fold = []

for fold, (train_idx, val_idx) in enumerate(skf.split(features_list, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n--- Fold {fold+1} ---")
    
    gcn = GCN(768, 256, 2, use_edge_weight=True).to(device)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_data = [
    Data(x=features_list[i], edge_index=adj_list[i], edge_weight=edge_weight_list[i], y=torch.tensor([y_no_hosp[i]])) 
    for i in train_idx
    ]
    val_data = [
        Data(x=features_list[i], edge_index=adj_list[i], edge_weight=edge_weight_list[i], y=torch.tensor([y_no_hosp[i]])) 
        for i in val_idx
    ]
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    t_loss = []

    for epoch in range(7):
        gcn.train()
        e_loss = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output, _ = gcn(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            loss = loss_fn(output.mean(dim=0, keepdim=True), batch.y.long().to(device))
            loss.backward()
            optimizer.step()
            e_loss.append(loss.item())
        t_loss.append(np.mean(e_loss))
        print(f"Epoch {epoch+1}, Loss: {t_loss[-1]:.4f}")

    all_loss.append(t_loss)

    # Validate
    gcn.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output, _ = gcn(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            probs = F.softmax(output.mean(dim=0), dim=0)
            pred = torch.argmax(probs).item()
            
            y_true.append(batch.y.item())
            y_pred.append(pred)
            y_scores.append(probs[1].item())

    val_auc = roc_auc_score(y_true, y_scores)
    print(f"Fold {fold+1} AUC: {val_auc:.4f}")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    roc_data_per_fold.append((fpr, tpr, roc_auc))

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = gcn.state_dict()

    # Guardem les mètriques per cada fold
    metrics["recall_0"].append(recall_score(y_true, y_pred, pos_label=0))
    metrics["recall_1"].append(recall_score(y_true, y_pred, pos_label=1))
    metrics["precision_0"].append(precision_score(y_true, y_pred, pos_label=0))
    metrics["precision_1"].append(precision_score(y_true, y_pred, pos_label=1))
    metrics["f1_0"].append(f1_score(y_true, y_pred, pos_label=0))
    metrics["f1_1"].append(f1_score(y_true, y_pred, pos_label=1))
    metrics["auc"].append(val_auc)

# Print averaged metrics
print("Averaged Metrics:")
for key, values in metrics.items():
    print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

t_stat, p_value = ttest_1samp(metrics["auc"], 0.5)
print(f"\nT-statistic (AUC vs 0.5): {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output, _ = gcn(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(output.mean(dim=0), dim=0)
            pred = torch.argmax(probs).item()
            
            y_true.append(batch.y.item())  # Guarda tots els valors del batch
            y_pred.append(pred)  # pred ha de ser una llista o un tensor
            y_scores.append(probs[1].item())  # Guarda tots els scores del batch
    
    return np.array(y_true), np.array(y_pred), np.array(y_scores)

plt.figure(figsize=(10, 6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/ViT+GCN.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for i, (fpr, tpr, roc_auc) in enumerate(roc_data_per_fold):
    plt.plot(fpr, tpr, label=f"Fold {i+1} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.title("ROC Curves per Fold - ResNet+MLP")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/fhome/pmarti/TFGPau/RocCurves/ViTGCN_Mitjana_ROC_PerFold.png", dpi=300)
plt.close()

test_data = [Data(x=features_list_h[i], edge_index=adj_list_h[i], y=torch.tensor([y_hosp[i]])) for i in range(len(y_hosp))]
test_loader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=collate_fn)

gcn = GCN(768, 256, 2).to(device)
gcn.load_state_dict(best_model_state)
gcn.eval()

y_true, y_pred, y_scores = evaluate_model(gcn, test_loader, device)

fpr_test, tpr_test, _ = roc_curve(y_true, y_scores)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, label=f"Holdout ROC (AUC = {roc_auc_test:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.title("ROC Curve - Holdout Set")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/fhome/pmarti/TFGPau/RocCurves/ViTGCN_Mitjana_ROC_Holdout.png", dpi=300)
plt.close()

# Calcular mètriques
auc_scores = roc_auc_score(y_true, y_scores)
recall_benigne = recall_score(y_true, y_pred, pos_label=0)
recall_maligne = recall_score(y_true, y_pred, pos_label=1)
precision_benigne = precision_score(y_true, y_pred, pos_label=0)
precision_maligne = precision_score(y_true, y_pred, pos_label=1)
f1_benigne = f1_score(y_true, y_pred, pos_label=0)
f1_maligne = f1_score(y_true, y_pred, pos_label=1)

# Mostrar resultats
print(f'Holdout results: ')
print(f"AUC: {auc_scores:.4f}")
print(f"Recall Benigne: {recall_benigne:.4f}")
print(f"Recall Maligne: {recall_maligne:.4f}")
print(f"Precision Benigne: {precision_benigne:.4f}")
print(f"Precision Maligne: {precision_maligne:.4f}")
print(f"F1-score Benigne: {f1_benigne:.4f}")
print(f"F1-score Maligne: {f1_maligne:.4f}")
