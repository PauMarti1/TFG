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
from torch_geometric.loader import DataLoader
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
import matplotlib.pyplot as plt

# -----------------------
# CONFIGURACIONS I CARREGADA DE DADES ORIGINALS
data = np.load('/fhome/pmarti/TFGPau/tissueDades.npz', allow_pickle=True)

X_no_hosp = data['X_no_hosp']
y_no_hosp = data['y_no_hosp']
PatID_no_hosp = data['PatID_no_hosp']
coords_no_hosp = data['coords_no_hosp']
infil_no_hosp = data['infil_no_hosp']

X_hosp = data['X_hosp']
y_hosp = data['y_hosp']
PatID_hosp = data['PatID_hosp']
coords_hosp = data['coords_hosp']
infil_hosp = data['infil_hosp']

deit = ViTModel.from_pretrained("google/vit-base-patch16-224").eval()
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

def process_images(image_list):
    features_list, attn_list = [], []
    for image in tqdm(image_list, desc="Processing images"):
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image = Image.open(image).convert("RGB")

        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = deit(**inputs, output_attentions=True)
            features = outputs.last_hidden_state.squeeze(0)
            per_layer = [att.mean(dim=1).squeeze(0) for att in outputs.attentions]
            attn_tensor = torch.stack(per_layer)

        features_list.append(features)
        attn_list.append(attn_tensor)

    return features_list, attn_list

features_list, attn_list = process_images(X_no_hosp)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GCN model
class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

classCount = torch.bincount(torch.tensor(y_no_hosp))
classWeights = 1.0 / classCount.float()
classWeights = (classWeights / classWeights.sum()).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=classWeights)

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {m: [] for m in ["recall_0","recall_1","precision_0","precision_1","f1_0","f1_1","auc"]}
all_loss = []
best_val_auc = 0
best_model_states = [None] * 12

all_true = [[[] for _ in range(12)] for _ in range(5)]
all_scores = [[[] for _ in range(12)] for _ in range(5)]

for fold, (train_idx, val_idx) in enumerate(
        skf.split(features_list, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n--- Fold {fold+1} ---")
    fold_losses = [[] for _ in range(12)]

    # Inicialitzar 12 models GCN
    models = [GCNGraphClassifier(768, 256, 2).to(device) for _ in range(12)]
    opts = [torch.optim.Adam(m.parameters(), lr=0.001) for m in models]

    # Entrenament de cada GCN amb threshold del adjacency
    for layer_idx in range(12):
        train_data = []
        for i in train_idx:
            adj = attn_list[i][layer_idx]
            edge_index = (adj >= torch.quantile(adj, 0.9))\
                .nonzero(as_tuple=False).t().contiguous()
            train_data.append(
                Data(x=features_list[i], edge_index=edge_index,
                     y=torch.tensor([y_no_hosp[i]]))
            )
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

        for epoch in range(25):
            models[layer_idx].train()
            epoch_losses = []
            for batch in train_loader:
                batch = batch.to(device)
                opts[layer_idx].zero_grad()
                out = models[layer_idx](batch.x, batch.edge_index)
                loss = loss_fn(out.mean(dim=0, keepdim=True), batch.y)
                loss.backward()
                opts[layer_idx].step()
                epoch_losses.append(loss.item())
            fold_losses[layer_idx].append(np.mean(epoch_losses))

    # Emmagatzemar loss mitjana de cada GCN
    all_loss.append([np.mean(l) for l in fold_losses])

    # Validació per folds i càlcul de ROC per cada GCN
    y_true, y_pred, y_scores = [], [], []
    for i in val_idx:
        probs_list = []
        for layer_idx in range(12):
            adj = attn_list[i][layer_idx]
            edge_index = (adj >= torch.quantile(adj, 0.9))\
                .nonzero(as_tuple=False).t().contiguous()
            data = Data(x=features_list[i], edge_index=edge_index).to(device)
            models[layer_idx].eval()
            with torch.no_grad():
                out = models[layer_idx](data.x, data.edge_index)
                prob = F.softmax(out.mean(dim=0), dim=0)
                probs_list.append(prob.cpu())

                all_true[fold][layer_idx].append(int(y_no_hosp[i]))
                all_scores[fold][layer_idx].append(prob[1].item())

        probs_tensor = torch.stack(probs_list)
        probs_avg = probs_tensor.mean(dim=0)
        pred = probs_avg.argmax().item()

        y_true.append(int(y_no_hosp[i]))
        y_pred.append(pred)
        y_scores.append(probs_avg[1].item())

    # Càlcul de mètriques i AUC
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

features_list_h, attn_list_h = process_images(X_hosp)

y_true, y_pred, y_scores = [], [], []
for i in range(len(X_hosp)):
    probs_list = []
    for layer_idx in range(12):
        adj = attn_list_h[i][layer_idx]
        edge_index = (adj >= torch.quantile(adj, 0.9)).nonzero(as_tuple=False).t().contiguous()
        data = Data(x=features_list_h[i], edge_index=edge_index).to(device)
        with torch.no_grad():
            prob = F.softmax(models[layer_idx](data.x, data.edge_index).mean(dim=0), dim=0)
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
