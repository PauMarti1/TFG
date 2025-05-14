import os
import numpy as np
import glob
import cv2 as cv
from ismember import ismember
from random import shuffle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, auc, roc_curve
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
import matplotlib.pyplot as plt

data = np.load('/fhome/pmarti/TFGPau/processedIms.npz', allow_pickle=True)
data1 = np.load('/fhome/pmarti/TFGPau/tissueDades.npz', allow_pickle=True)

features_list = data['features_list_single_no_h'] 
attn_list = data['adj_list_single_no_h']

features_list_h = data['features_list_single_h'] 
attn_list_h = data['adj_list_single_h']

y_hosp = data1['y_hosp']
y_no_hosp = data1['y_no_hosp']

# Collate that preserves both x and attn
from torch_geometric.data import Batch
def collate_fn(batch):
    return Batch.from_data_list(batch)

# ================= STEP 2: Define GCN with Learned Aggregator =================
class GCNWithAgg(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # 1x1 conv to learn combination of the 12 attention maps
        self.agg = torch.nn.Conv2d(12, 1, kernel_size=1)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, attn_tensor):
        # Learn aggregated adjacency
        agg_mat = self.agg(attn_tensor)      # (batch_size,1,N,N)
        agg_mat = agg_mat.squeeze(1)         # (batch_size, N, N)
        # For now threshold at the 90th percentile per graph
        thresh = torch.quantile(agg_mat.flatten(1), 0.9, dim=1)
        # Build edge index (assume batch_size=1)
        mask = agg_mat[0] >= thresh[0]
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()

        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Stratified K-Fold Validation
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {m: [] for m in ["recall_0","recall_1","precision_0","precision_1","f1_0","f1_1","auc"]}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classCount = torch.bincount(torch.tensor(y_no_hosp))
classWeights = 1.0 / classCount.float()
classWeights = (classWeights / classWeights.sum()).to(device)

best_val_auc = 0
best_model_state = None
all_loss = []
roc_data_per_fold = []

for fold, (train_idx, val_idx) in enumerate(skf.split(features_list, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n--- Fold {fold+1} ---")
    gcn = GCNWithAgg(in_channels=768, hidden_channels=256, out_channels=2).to(device)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss(weight=classWeights)

    train_data = [Data(x=features_list[i], attn=attn_list[i], y=torch.tensor([y_no_hosp[i]])) for i in train_idx]
    val_data   = [Data(x=features_list[i], attn=attn_list[i], y=torch.tensor([y_no_hosp[i]])) for i in val_idx]

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_data,   batch_size=1, shuffle=False, collate_fn=collate_fn)

    fold_losses = []
    for epoch in range(25):
        gcn.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = gcn(batch.x, batch.attn.to(device))
            # graph-level prediction via mean
            logits = out.mean(dim=0, keepdim=True)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_loss = np.mean(losses)
        fold_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    all_loss.append(fold_losses)
    
    # Validation
    gcn.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = gcn(batch.x, batch.attn.to(device))
            probs = F.softmax(out.mean(dim=0), dim=0)
            pred = probs.argmax().item()
            y_true.append(batch.y.item())
            y_pred.append(pred)
            y_scores.append(probs[1].item())

    val_auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    roc_data_per_fold.append((fpr, tpr, roc_auc))
    print(f"Fold {fold+1} AUC: {val_auc:.4f}")
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = gcn.state_dict()

    metrics['recall_0'].append(recall_score(y_true, y_pred, pos_label=0))
    metrics['recall_1'].append(recall_score(y_true, y_pred, pos_label=1))
    metrics['precision_0'].append(precision_score(y_true, y_pred, pos_label=0))
    metrics['precision_1'].append(precision_score(y_true, y_pred, pos_label=1))
    metrics['f1_0'].append(f1_score(y_true, y_pred, pos_label=0))
    metrics['f1_1'].append(f1_score(y_true, y_pred, pos_label=1))
    metrics['auc'].append(val_auc)

# Print averaged metrics
print("Averaged Metrics:")
for key, vals in metrics.items():
    print(f"{key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# Plot training losses
plt.figure(figsize=(10,6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/GCNAGG.png', dpi=300)
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
plt.savefig("/fhome/pmarti/TFGPau/RocCurves/ViTGCN_Agregació_ROC_PerFold.png", dpi=300)
plt.close()

# ================= STEP 3: Evaluate on Holdout =================
features_list_h, attn_list_h = process_images(X_hosp)

test_data = [Data(x=features_list_h[i], attn=attn_list_h[i], y=torch.tensor([y_hosp[i]]))
             for i in range(len(y_hosp))]

test_loader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
gcn = GCNWithAgg(in_channels=768, hidden_channels=256, out_channels=2).to(device)
gcn.load_state_dict(best_model_state)
gcn.eval()

y_true, y_pred, y_scores = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = gcn(batch.x, batch.attn.to(device))
        probs = F.softmax(out.mean(dim=0), dim=0)
        pred = probs.argmax().item()
        y_true.append(batch.y.item())
        y_pred.append(pred)
        y_scores.append(probs[1].item())

# Calcular mètriques
auc_scores     = roc_auc_score(y_true, y_scores)
recall_benigne = recall_score(y_true, y_pred, pos_label=0)
recall_maligne = recall_score(y_true, y_pred, pos_label=1)
precision_ben  = precision_score(y_true, y_pred, pos_label=0)
precision_mal  = precision_score(y_true, y_pred, pos_label=1)
f1_ben         = f1_score(y_true, y_pred, pos_label=0)
f1_mal         = f1_score(y_true, y_pred, pos_label=1)

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
plt.savefig("/fhome/pmarti/TFGPau/RocCurves/ViTGCN_Agregació_ROC_Holdout.png", dpi=300)
plt.close()

print('Holdout results:')
print(f"AUC: {auc_scores:.4f}")
print(f"Recall Benigne: {recall_benigne:.4f}")
print(f"Recall Maligne: {recall_maligne:.4f}")
print(f"Precision Benigne: {precision_ben:.4f}")
print(f"Precision Maligne: {precision_mal:.4f}")
print(f"F1-score Benigne: {f1_ben:.4f}")
print(f"F1-score Maligne: {f1_mal:.4f}")
