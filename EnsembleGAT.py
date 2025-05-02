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
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
import matplotlib.pyplot as plt

# -----------------------
# CONFIGURACIÓ I CÀRREGA DE DADES
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

def process_images(image_list, threshold=0.0):
    features_list, adj_list, edge_attr_list = [], [], []

    for image in tqdm(image_list, desc="Processing images"):
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image = Image.open(image).convert("RGB")
        
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = deit(**inputs, output_attentions=True)
            features = outputs.last_hidden_state.squeeze(0)[1:]  # (196, 768)
            attentions = torch.stack(outputs.attentions)  # (12, 1, heads, 197, 197)
            attentions = attentions.squeeze(1).mean(dim=1)  # (12, 197, 197)
            attn_matrix = attentions[:, 1:, 1:]  # (12, 196, 196)

        edge_indices_heads = []
        edge_weights_heads = []

        for layer_attn in attn_matrix:  # (196, 196) per cap
            layer_attn_np = layer_attn.cpu().numpy()
            edge_list = []
            edge_weights = []
            for i in range(layer_attn_np.shape[0]):
                for j in range(layer_attn_np.shape[1]):
                    weight = float(layer_attn_np[i, j])
                    if weight > threshold:
                        edge_list.append([i, j])
                        edge_weights.append(weight)
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
            edge_indices_heads.append(edge_index)
            edge_weights_heads.append(edge_attr)

        features_list.append(features)
        adj_list.append(edge_indices_heads)
        edge_attr_list.append(edge_weights_heads)

    return features_list, adj_list, edge_attr_list

features_list, attn_list, edge_weights_list = process_images(X_no_hosp)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🔸 Nou model GAT
class GATGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(GATGraphClassifier, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_dim, heads=1, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=True, edge_dim=1)
        self.fc = torch.nn.Linear(hidden_dim, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.fc(x), dim=1)

# Pèrdues pesades per classes
classCount = torch.bincount(torch.tensor(y_no_hosp))
classWeights = 1.0 / classCount.float()
classWeights = (classWeights / classWeights.sum()).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=classWeights)

# Entrenament per ensemble
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {m: [] for m in ["recall_0","recall_1","precision_0","precision_1","f1_0","f1_1","auc"]}
all_loss = []
best_val_auc = 0
best_model_states = [None] * 12
all_train_losses = []        # [num_folds][12][num_epochs]
val_losses_all_folds = []    # [num_folds][12]
best_val_auc = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(features_list, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n--- Fold {fold+1} ---")
    fold_losses = [[] for _ in range(12)]
    val_fold_losses = []  # Per emmagatzemar la validació de cada fold

    models = [GATGraphClassifier(768, 256, 2).to(device) for _ in range(12)]
    opts = [torch.optim.Adam(model.parameters(), lr=0.001) for model in models]

    for layer_idx in range(12):
        # Prepare training data
        train_data = [
            Data(x=features_list[i], edge_index=attn_list[i][layer_idx],
                 edge_attr=edge_weights_list[i][layer_idx],
                 y=torch.tensor([y_no_hosp[i]]))
            for i in train_idx
        ]
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

        for epoch in range(25):
            # Training
            models[layer_idx].train()
            epoch_losses = []
            for batch in train_loader:
                batch = batch.to(device)
                opts[layer_idx].zero_grad()
                out = models[layer_idx](batch)
                loss = loss_fn(out, batch.y)
                loss.backward()
                opts[layer_idx].step()
                epoch_losses.append(loss.item())
            fold_losses[layer_idx].append(np.mean(epoch_losses))

        # --- Validació per fold ---
        val_data = [
            Data(x=features_list[i], edge_index=attn_list[i][layer_idx],
                 edge_attr=edge_weights_list[i][layer_idx],
                 y=torch.tensor([y_no_hosp[i]]))
            for i in val_idx
        ]
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

        # Validació
        models[layer_idx].eval()
        val_epoch_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = models[layer_idx](batch)
                val_loss = loss_fn(out, batch.y)
                val_epoch_losses.append(val_loss.item())
        val_fold_losses.append(np.mean(val_epoch_losses))

    # Guardem losses de cada fold
    val_losses_all_folds.append(val_fold_losses)
    all_train_losses.append(fold_losses)

    # --- Ensemble Validation per fold ---
    y_true, y_pred, y_scores = [], [], []
    for i in val_idx:
        probs_list = []
        for layer_idx in range(12):
            data = Data(
                x=features_list[i],
                edge_index=attn_list[i][layer_idx],
                edge_attr=edge_weights_list[i][layer_idx]
            ).to(device)

            models[layer_idx].eval()
            with torch.no_grad():
                out = models[layer_idx](data)
                prob = F.softmax(out, dim=1)
                probs_list.append(prob.cpu())

        stacked = torch.stack(probs_list)  # [12, 1, 2]
        probs_avg = stacked.mean(dim=0)    # [1, 2]
        pred_score = probs_avg[0].argmax().item()
        score_cls1 = probs_avg[0, 1].item()
        
        y_true.append(int(y_no_hosp[i]))
        y_pred.append(pred_score)
        y_scores.append(score_cls1)

    val_auc = roc_auc_score(y_true, y_scores)
    print(f"Fold {fold+1} AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_states = [model.state_dict() for model in models]
        torch.save(best_model_states, f"best_models_fold{fold+1}.pth")

    metrics['recall_0'].append(recall_score(y_true, y_pred, pos_label=0))
    metrics['recall_1'].append(recall_score(y_true, y_pred, pos_label=1))
    metrics['precision_0'].append(precision_score(y_true, y_pred, pos_label=0))
    metrics['precision_1'].append(precision_score(y_true, y_pred, pos_label=1))
    metrics['f1_0'].append(f1_score(y_true, y_pred, pos_label=0))
    metrics['f1_1'].append(f1_score(y_true, y_pred, pos_label=1))
    metrics['auc'].append(val_auc)

    # Validation Loss Plot
    plt.figure(figsize=(10, 6))
    for layer_idx in range(12):
        plt.plot(val_fold_losses[layer_idx], label=f'Val GAT {layer_idx+1}')
    plt.title(f'Validation Loss per GAT (Fold {fold+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'fhome/pmarti/TFGPau/Ensemble/val_loss_fold_{fold+1}.png')
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
models = [GATGraphClassifier(768, 256, 2).to(device) for _ in range(12)]
for i, model in enumerate(models):
    model.load_state_dict(best_model_states[i])
    model.eval()

features_list_h, attn_list_h, edge_weights_list_h = process_images(X_hosp)

y_true, y_pred, y_scores = [], [], []
for i in range(len(X_hosp)):
    probs_list = []
    for layer_idx in range(12):
        data = Data(
            x=features_list_h[i],
            edge_index=attn_list_h[i][layer_idx],
            edge_attr=edge_weights_list_h[i][layer_idx]
        ).to(device)

        with torch.no_grad():
            out  = models[layer_idx](data)            # [1,2]
            prob = F.softmax(out, dim=1)              # [1,2]
            probs_list.append(prob.cpu())

    stacked   = torch.stack(probs_list)             # [12,1,2]
    probs_avg = stacked.mean(dim=0)                 # [1,2]

    pred_score = probs_avg[0].argmax().item()
    score_cls1 = probs_avg[0, 1].item()

    y_true.append(int(y_hosp[i]))
    y_pred.append(pred_score)
    y_scores.append(score_cls1)

print('Holdout results:')
print(f"AUC: {roc_auc_score(y_true, y_scores):.4f}")
print(f"Recall Benigne: {recall_score(y_true, y_pred, pos_label=0):.4f}")
print(f"Recall Maligne: {recall_score(y_true, y_pred, pos_label=1):.4f}")
print(f"Precision Benigne: {precision_score(y_true, y_pred, pos_label=0):.4f}")
print(f"Precision Maligne: {precision_score(y_true, y_pred, pos_label=1):.4f}")
print(f"F1-score Benigne: {f1_score(y_true, y_pred, pos_label=0):.4f}")
print(f"F1-score Maligne: {f1_score(y_true, y_pred, pos_label=1):.4f}")