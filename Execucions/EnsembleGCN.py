import os
import numpy as np
import glob
import cv2 as cv
from ismember import ismember
from random import shuffle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
import matplotlib.pyplot as plt
data   = np.load('/fhome/pmarti/TFGPau/LargetissueDades_48_Norm.npz', allow_pickle=True)
data1  = np.load('/fhome/pmarti/TFGPau/DBLarge_FeatMatNew_Reduit_norm.npz/DBLarge_FeatMatNew_Reduit_norm.npz', allow_pickle=True)

X_hosp         = data['X_hosp']
features_list  = data1['features_list']
attn_list      = data1['attn_matrices']   # shape: [num_images, 12, N, N]
y_hosp         = data['y_hosp']
y_no_hosp      = data['y_no_hosp']
PatID_no_hosp  = data['PatID_no_hosp']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── 2) Convertir attn_list a adj_list ─────────────────────────────────────────────
threshold = 0.0
adj_list = []           # Llista de llargada igual al número d'imatges
edge_weight_list = []   # Cada element contindrà 12 edge_index i 12 edge_weight

for i in range(len(attn_matrices)):
    # Una imatge conté 12 matrius d’atenció (una per cap)
    adj_per_image = []
    edge_weight_per_image = []
    
    for layer_idx in range(12):
        edge_index_np, edge_weight_np = attn_to_graph(attn_matrices[i][layer_idx], threshold=threshold)
        
        edge_index = torch.tensor(edge_index_np, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_np, dtype=torch.float).view(-1)
        
        adj_per_image.append(edge_index)
        edge_weight_per_image.append(edge_weight)

    adj_list.append(adj_per_image)
    edge_weight_list.append(edge_weight_per_image)

# ─── 3) Model GCN ──────────────────────────────────────────────────────────────────
class GCNGraphClassifier(torch.nn.Module):
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

# ─── 4) Configuració d’entrenament ─────────────────────────────────────────────────
loss_fn = torch.nn.CrossEntropyLoss()
skf     = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2)

metrics = {m: [] for m in ["recall_0","recall_1","precision_0","precision_1","f1_0","f1_1","auc"]}
best_val_auc     = 0
best_model_states = [None] * 12
all_true  = [[[] for _ in range(12)] for _ in range(5)]
all_scores= [[[] for _ in range(12)] for _ in range(5)]
all_loss  = []

def collate_fn(batch):
    return Batch.from_data_list(batch)

# ─── 5) Boucle de K-Fold ───────────────────────────────────────────────────────────
for fold, (train_idx, val_idx) in enumerate(skf.split(features_list, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n--- Fold {fold+1} ---")
    fold_losses = [[] for _ in range(12)]

    # Initialitzar 12 models GCN i optimitzadors
    models = [GCNGraphClassifier(768, 256, 2).to(device) for _ in range(12)]
    opts   = [torch.optim.Adam(m.parameters(), lr=0.001) for m in models]

    # Entrenament de cada GCN (una per capa d’atenció)
    for layer_idx in range(12):
        # Dataset de entrenament per la capa layer_idx
        train_data = [
            Data(x=features_list[i],
                 edge_index=adj_list[i][layer_idx],
                 edge_attr=edge_weight_list[i][layer_idx],
                 y=torch.tensor([y_no_hosp[i]]))
            for i in train_idx
        ]
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)

        # Epochs d’entrenament
        for epoch in range(7):
            models[layer_idx].train()
            losses = []
            for batch in train_loader:
                batch = batch.to(device)
                opts[layer_idx].zero_grad()
                out, _ = models[layer_idx](batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = loss_fn(out.mean(dim=0, keepdim=True), batch.y)
                loss.backward()
                opts[layer_idx].step()
                losses.append(loss.item())
            fold_losses[layer_idx].append(np.mean(losses))

    all_loss.append([np.mean(l) for l in fold_losses])

    # Validació i càlcul de ROC per cada GCN
    y_true, y_pred, y_scores = [], [], []
    for i in val_idx:
        probs_list = []
        for layer_idx in range(12):
            data = Data(x=features_list[i],
                        edge_index=adj_list[i][layer_idx],
                        edge_attr=edge_weight_list[i][layer_idx]).to(device)
            models[layer_idx].eval()
            with torch.no_grad():
                out, _ = models[layer_idx](data.x, data.edge_index, batch.edge_attr, data.batch)
                prob = F.softmax(out.mean(dim=0), dim=0)
                probs_list.append(prob.cpu())

                all_true[fold][layer_idx].append(int(y_no_hosp[i]))
                all_scores[fold][layer_idx].append(prob[1].item())

        # Ensemble per vot majoritari (mitjana de probabilitats)
        probs_avg = torch.stack(probs_list).mean(dim=0)
        pred = probs_avg.argmax().item()

        y_true.append(int(y_no_hosp[i]))
        y_pred.append(pred)
        y_scores.append(probs_avg[1].item())

    # Mètriques del fold
    val_auc = roc_auc_score(y_true, y_scores)
    print(f"Fold {fold+1} AUC: {val_auc:.4f}")
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_states = [m.state_dict() for m in models]

    metrics['auc'].append(val_auc)
    metrics['recall_0'].append(recall_score(y_true, y_pred, pos_label=0))
    metrics['recall_1'].append(recall_score(y_true, y_pred, pos_label=1))
    metrics['precision_0'].append(precision_score(y_true, y_pred, pos_label=0))
    metrics['precision_1'].append(precision_score(y_true, y_pred, pos_label=1))
    metrics['f1_0'].append(f1_score(y_true, y_pred, pos_label=0))
    metrics['f1_1'].append(f1_score(y_true, y_pred, pos_label=1))

# Plot training losses
plt.figure(figsize=(10,6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold (avg across GCNs)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/GCNENSEMBLE.png', dpi=300)
plt.close()

for layer_idx in range(12):
    plt.figure(figsize=(8,6))
    for fold in range(5):
        fpr, tpr, _ = roc_curve(all_true[fold][layer_idx], all_scores[fold][layer_idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Fold {fold+1} (AUC={roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.title(f'ROC Curves - GCN Model {layer_idx+1}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/fhome/pmarti/TFGPau/RocCurves/EnsembleGCN/roc_gcn_{layer_idx+1}.png', dpi=300)
    plt.close()

print("Averaged Metrics:")
for key, vals in metrics.items():
    print(f"{key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# Holdout evaluation
models = [GCNGraphClassifier(768, 256, 2).to(device) for _ in range(12)]
for i, model in enumerate(models):
    model.load_state_dict(best_model_states[i])
    model.eval()

y_true, y_pred, y_scores = [], [], []
for i in range(len(X_hosp)):
    probs_list = []
    for layer_idx in range(12):
        adj = attn_list_h[i][layer_idx]
        edge_index = adj
        data = Data(x=features_list_h[i], edge_index=edge_index).to(device)
        with torch.no_grad():
            out, _ = models[layer_idx](batch.x, batch.edge_index, batch.batch)
            prob = F.softmax(out.mean(dim=0), dim=0)
            probs_list.append(prob.cpu())
    probs_tensor = torch.stack(probs_list)
    probs_avg = probs_tensor.mean(dim=0)
    pred = probs_avg.argmax().item()

    y_true.append(y_hosp[i])
    y_pred.append(pred)
    y_scores.append(probs_avg[1].item())

fpr_h, tpr_h, _ = roc_curve(y_true, y_scores)
roc_auc_h = auc(fpr_h, tpr_h)
plt.figure(figsize=(8,6))
plt.plot(fpr_h, tpr_h, label=f'Holdout (AUC={roc_auc_h:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curve - Holdout')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/RocCurves/EnsembleGCN/roc_holdout.png', dpi=300)
plt.close()

print('Holdout results:')
print(f"AUC: {roc_auc_score(y_true, y_scores):.4f}")
print(f"Recall Benigne: {recall_score(y_true, y_pred, pos_label=0):.4f}")
print(f"Recall Maligne: {recall_score(y_true, y_pred, pos_label=1):.4f}")
print(f"Precision Benigne: {precision_score(y_true, y_pred, pos_label=0):.4f}")
print(f"Precision Maligne: {precision_score(y_true, y_pred, pos_label=1):.4f}")
print(f"F1-score Benigne: {f1_score(y_true, y_pred, pos_label=0):.4f}")
print(f"F1-score Maligne: {f1_score(y_true, y_pred, pos_label=1):.4f}")
