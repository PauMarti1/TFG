import os
import numpy as np
import glob
import cv2 as cv
from ismember import ismember
from random import shuffle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
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


data = np.load('/fhome/pmarti/TFGPau/processedIms.npz', allow_pickle=True)
data1 = np.load('/fhome/pmarti/TFGPau/tissueDades.npz', allow_pickle=True)

features_list = data['features_list_single_no_h'] 
adj_list = data['adj_list_single_no_h']

features_list_h = data['features_list_single_h'] 
adj_list_h = data['adj_list_single_h']

y_hosp = data1['y_hosp']
y_no_hosp = data1['y_no_hosp']

def collate_fn(batch):
    return Batch.from_data_list(batch)

# ================= STEP 2: Define GAT Model =================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Stratified K-Fold Validation
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": []}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_loss = []

classCount = torch.bincount(torch.tensor(y_no_hosp))
classWeights = 1.0 / classCount.float()
classWeights = classWeights / classWeights.sum()
classWeights = classWeights.to(device)

best_val_auc = 0
best_model_state = None
all_loss = []
roc_data_per_fold = []

for fold, (train_idx, val_idx) in enumerate(skf.split(features_list, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n--- Fold {fold+1} ---")
    
    gcn = GCN(in_channels=768, hidden_channels=256, out_channels=2).to(device)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss(weight=classWeights)

    train_data = [Data(x=features_list[i], edge_index=adj_list[i], y=torch.tensor([y_no_hosp[i]])) for i in train_idx]
    val_data = [Data(x=features_list[i], edge_index=adj_list[i], y=torch.tensor([y_no_hosp[i]])) for i in val_idx]
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    t_loss = []

    for epoch in range(25):
        gcn.train()
        e_loss = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = gcn(batch.x, batch.edge_index)
            loss = loss_fn(output.mean(dim=0, keepdim=True), batch.y.to(device))
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
            output = gcn(batch.x, batch.edge_index)
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

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = gcn(batch.x, batch.edge_index)
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

gcn = GCN(in_channels=768, hidden_channels=256, out_channels=2).to(device)
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
