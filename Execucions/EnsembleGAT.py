import os
import numpy as np
import glob
import cv2 as cv
from ismember import ismember
from random import shuffle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, auc, roc_curve
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
import matplotlib.pyplot as plt

# -----------------------
# CONFIGURACIÓ I CÀRREGA DE DADES
data = np.load('/fhome/pmarti/TFGPau/LargetissueDades_48_Norm.npz', allow_pickle=True)
data1 = np.load('/fhome/pmarti/TFGPau/DBLarge_FeatMatNew_Reduit_norm.npz/DBLarge_FeatMatNew_Reduit_norm.npz', allow_pickle=True)

X_hosp = data['X_hosp']
features_list = data1['features_list']
attn_list = data1['attn_matrices']
y_hosp = data['y_hosp']
y_no_hosp = data['y_no_hosp']
PatID_no_hosp = data['PatID_no_hosp']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Conversió: attn_matrices ➝ adj_list + edge_attr_list
def attn_to_edge_index_and_attr(attn):
    """Converteix una matriu d'atenció a (edge_index, edge_attr)"""
    n_heads, n_nodes, _ = attn.shape
    edge_indices_per_head, edge_attrs_per_head = [], []
    for h in range(n_heads):
        A = attn[h]
        src, dst = A.nonzero(as_tuple=True)
        edge_attr = A[src, dst]
        edge_index = torch.stack([src, dst], dim=0)
        edge_indices_per_head.append(edge_index)
        edge_attrs_per_head.append(edge_attr.unsqueeze(1))  # shape: [num_edges, 1]
    return edge_indices_per_head, edge_attrs_per_head

adj_list = []
edge_weights_list = []

for attn_per_graph in attn_list:
    edges_12_layers = []
    attrs_12_layers = []
    for layer in attn_per_graph:  # shape [12, num_heads, N, N]
        edge_idx, edge_attr = attn_to_edge_index_and_attr(layer)
        edges_12_layers.append(edge_idx)
        attrs_12_layers.append(edge_attr)
    adj_list.append(edges_12_layers)
    edge_weights_list.append(attrs_12_layers)

# 🔸 Model GAT
class GATGraphClassifier(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, heads=1, use_edge_attr=True, threshold=0.0):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        edge_dim = 1 if use_edge_attr else None
        self.gat1 = GATConv(in_ch, hidden_ch, heads=heads, concat=True, edge_dim=edge_dim)
        self.gat2 = GATConv(hidden_ch * heads, hidden_ch, heads=1, concat=True, edge_dim=edge_dim)
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(hidden_ch, hidden_ch),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_ch),
            torch.nn.Linear(hidden_ch, out_ch)
        )
        self.threshold = threshold

    def forward(self, x, edge_index, edge_attr, batch_idx):
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_attr)
        g = global_mean_pool(x, batch_idx)
        return self.lin(g), edge_attr

loss_fn = torch.nn.CrossEntropyLoss()
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2)

metrics = {m: [] for m in ["recall_0","recall_1","precision_0","precision_1","f1_0","f1_1","auc"]}
best_model_states = [None] * 12
all_train_losses, val_losses_all_folds = [], []
all_true = [[[] for _ in range(12)] for _ in range(5)]
all_scores = [[[] for _ in range(12)] for _ in range(5)]

def collate_fn(batch):
    return Batch.from_data_list(batch)

for fold, (train_idx, val_idx) in enumerate(skf.split(features_list, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n--- Fold {fold+1} ---")
    fold_losses = [[] for _ in range(12)]
    val_fold_losses = []

    models = [GATGraphClassifier(768, 256, 2, use_edge_attr=False).to(device) for _ in range(12)]
    opts = [torch.optim.Adam(m.parameters(), lr=0.001) for m in models]

    for layer_idx in range(12):
        train_data = [
            Data(x=features_list[i],
                 edge_index=adj_list[i][layer_idx],
                 edge_attr=edge_weights_list[i][layer_idx],
                 y=torch.tensor([y_no_hosp[i]]))
            for i in train_idx
        ]
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)

        for epoch in range(25):
            models[layer_idx].train()
            losses = []
            for batch in train_loader:
                batch = batch.to(device)
                opts[layer_idx].zero_grad()
                out, _ = models[layer_idx](batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = loss_fn(out, batch.y.long())
                loss.backward()
                opts[layer_idx].step()
                losses.append(loss.item())
            fold_losses[layer_idx].append(np.mean(losses))

        # Validació
        val_data = [
            Data(x=features_list[i],
                 edge_index=adj_list[i][layer_idx],
                 edge_attr=edge_weights_list[i][layer_idx],
                 y=torch.tensor([y_no_hosp[i]]))
            for i in val_idx
        ]
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

        models[layer_idx].eval()
        losses_val, y_true, y_scores = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out, _ = models[layer_idx](batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                losses_val.append(loss_fn(out, batch.y.long()).item())
                prob = F.softmax(out, dim=1)
                y_true.append(int(batch.y.item()))
                y_scores.append(prob[0,1].item())

        val_fold_losses.append(np.mean(losses_val))
        all_true[fold][layer_idx] = y_true
        all_scores[fold][layer_idx] = y_scores

    val_losses_all_folds.append(val_fold_losses)
    all_train_losses.append(fold_losses)

    # 🔸 Ensemble (vot majoritari)
    y_true_e, y_pred_e, y_scores_e = [], [], []
    for i in val_idx:
        probs = []
        for layer_idx in range(12):
            data = Data(
                x=features_list[i],
                edge_index=adj_list[i][layer_idx],
                edge_attr=edge_weights_list[i][layer_idx]
            ).to(device)
            models[layer_idx].eval()
            with torch.no_grad():
                out, _ = models[layer_idx](data.x, data.edge_index, data.edge_attr, torch.zeros(data.x.size(0), dtype=torch.long, device=device))
                probs.append(F.softmax(out, dim=1).cpu())

        avg = torch.stack(probs).mean(dim=0)
        y_true_e.append(int(y_no_hosp[i]))
        y_pred_e.append(avg[0].argmax().item())
        y_scores_e.append(avg[0, 1].item())

    auc_fold = roc_auc_score(y_true_e, y_scores_e)
    print(f"Fold {fold+1} AUC (ensemble): {auc_fold:.4f}")

    if auc_fold > max(metrics["auc"], default=0):
        best_model_states = [m.state_dict() for m in models]
        torch.save(best_model_states, f"best_models_fold{fold+1}.pth")

    metrics['auc'].append(auc_fold)
    metrics['recall_0'].append(recall_score(y_true_e, y_pred_e, pos_label=0))
    metrics['recall_1'].append(recall_score(y_true_e, y_pred_e, pos_label=1))
    metrics['precision_0'].append(precision_score(y_true_e, y_pred_e, pos_label=0))
    metrics['precision_1'].append(precision_score(y_true_e, y_pred_e, pos_label=1))
    metrics['f1_0'].append(f1_score(y_true_e, y_pred_e, pos_label=0))
    metrics['f1_1'].append(f1_score(y_true_e, y_pred_e, pos_label=1))


for layer_idx in range(12):
    plt.figure(figsize=(8,6))
    for fold in range(5):
        fpr, tpr, _ = roc_curve(all_true[fold][layer_idx], all_scores[fold][layer_idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Fold {fold+1} (AUC={roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.title(f'ROC Curves - GAT Model {layer_idx+1}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/fhome/pmarti/TFGPau/RocCurves/Ensemble/roc_gat_{layer_idx+1}.png')
    plt.close()

for fold_idx, fold_losses in enumerate(all_loss):  # all_loss = list of 12 losses per fold
    plt.figure(figsize=(10, 6))
    for layer_idx in range(12):
        plt.plot(fold_losses[layer_idx], label=f'GAT {layer_idx+1}')
    plt.title(f'Training Loss per GAT (Fold {fold_idx+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/fhome/pmarti/TFGPau/Ensemble/loss_fold_{fold_idx+1}.png')  # 🔽 guarda la figura
    plt.close()

print("Averaged Metrics:")
for key, vals in metrics.items():
    print(f"{key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# Holdout
# models = [GATGraphClassifier(768, 256, 2).to(device) for _ in range(12)]
# for i, model in enumerate(models):
#     model.load_state_dict(best_model_states[i])
#     model.eval()

# y_true, y_pred, y_scores = [], [], []

# y_true, y_pred, y_scores = [], [], []

# for i in range(len(X_hosp)):
#     probs_list = []
#     for layer_idx in range(12):
#         batch = Data(
#             x=features_list_h[i],
#             edge_index=attn_list_h[i][layer_idx],
#             edge_attr=edge_weights_list_h[i][layer_idx]
#         ).to(device)

#         with torch.no_grad():
#             out, _ = models[layer_idx](batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # [1, 2]
#             prob = F.softmax(out, dim=1)  # [1, 2]
#             probs_list.append(prob.cpu())

#     stacked = torch.stack(probs_list)  # [12, 1, 2]
#     probs_avg = stacked.mean(dim=0)  # [1, 2]

#     pred_score = probs_avg[0].argmax().item()
#     score_cls1 = probs_avg[0, 1].item()

#     y_true.append(int(y_hosp[i]))
#     y_pred.append(pred_score)
#     y_scores.append(score_cls1)


# fpr_h, tpr_h, _ = roc_curve(y_true, y_scores)
# roc_auc_h = auc(fpr_h, tpr_h)
# plt.figure(figsize=(8,6))
# plt.plot(fpr_h, tpr_h, label=f'Holdout (AUC={roc_auc_h:.2f})')
# plt.plot([0,1],[0,1],'k--')
# plt.title('ROC Curve - Holdout')
# plt.xlabel('FPR'); plt.ylabel('TPR')
# plt.legend(loc='lower right'); plt.grid(True); plt.tight_layout()
# plt.savefig('/fhome/pmarti/TFGPau/Ensemble/roc_holdout.png')
# plt.close()

# print('Holdout AUC:', roc_auc_h)
# print('Holdout results:')
# print(f"AUC: {roc_auc_score(y_true, y_scores):.4f}")
# print(f"Recall Benigne: {recall_score(y_true, y_pred, pos_label=0):.4f}")
# print(f"Recall Maligne: {recall_score(y_true, y_pred, pos_label=1):.4f}")
# print(f"Precision Benigne: {precision_score(y_true, y_pred, pos_label=0):.4f}")
# print(f"Precision Maligne: {precision_score(y_true, y_pred, pos_label=1):.4f}")
# print(f"F1-score Benigne: {f1_score(y_true, y_pred, pos_label=0):.4f}")
# print(f"F1-score Maligne: {f1_score(y_true, y_pred, pos_label=1):.4f}")
