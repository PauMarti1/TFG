import os
import numpy as np
import glob
import cv2 as cv
from ismember import ismember
from random import shuffle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
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


# -----------------------
# CONFIGURACIONS I CARREGADA DE DADES ORIGINALS
# Carregar el fitxer .npz
data = np.load('/fhome/pmarti/TFGPau/novesDades.npz', allow_pickle=True)
data1 = np.load('/fhome/pmarti/TFGPau/featuresNoHosp.npz', allow_pickle=True)
data2 = np.load('/fhome/pmarti/TFGPau/featuresHosp.npz', allow_pickle=True)

features_list = data1['features_list'] 
adj_list = data1['adj_mean_list']
edge_attr_list = data1['edge_attr_mean_list']

f_h = data2['features_list_ho'] 
a_h = data2['adj_mean_list_ho']
edge_attr_list_h = data2['edge_attr_mean_list_ho']

y_hosp = data['y_hosp']
y_no_hosp = data['y_no_hosp']

PatID_no_hosp = data['PatID_no_hosp']
PatID_hosp = data['PatID_hosp']

def collate_fn(batch):
    return Batch.from_data_list(batch)

# ================= STEP 2: Define GAT Model =================
class GAT(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, heads=1, use_edge_attr=True, threshold=0.0):
        super().__init__()
        self.use_edge_attr = use_edge_attr

        edge_dim = 1 if use_edge_attr else None
        self.gat1 = GATConv(in_ch, hidden_ch, heads=heads, concat=True, edge_dim=edge_dim)
        self.gat2 = GATConv(hidden_ch*heads, hidden_ch, heads=1, concat=True, edge_dim=edge_dim)
        self.lin = torch.nn.Sequential(
            # torch.nn.Linear(hidden_ch*2, hidden_ch), # Global + Max
            torch.nn.Linear(hidden_ch, hidden_ch),  #--> en cas de fer nomes el global
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_ch),
            torch.nn.Linear(hidden_ch, out_ch)
        )
        #self.lin  = torch.nn.Linear(hidden_ch, out_ch)
        self.threshold = threshold

    def forward(self, x, edge_index, edge_attr, batch_idx):
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_attr)
        g = global_mean_pool(x, batch_idx)
        return self.lin(g), edge_attr

# Stratified K-Fold Validation
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": []}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_loss = []

# classCount = torch.bincount(torch.tensor(y_no_hosp))
# classWeights = 1.0 / classCount.float()
# classWeights = classWeights / classWeights.sum()
# classWeights = classWeights.to(device)

best_val_auc = 0
best_model_state = None
all_loss = []
all_train_losses   = []   # per guardar la corba d'entrenament (10 punts) de cada fold
val_loss_per_fold  = []   # per guardar un sol valor de validació per fold
best_val_auc       = 0
best_model_state   = None
metrics            = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [],
                      "f1_0": [], "f1_1": [], "auc": []}
roc_data_per_fold = []
for fold, (train_idx, val_idx) in enumerate(skf.split(features_list, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n--- Fold {fold+1} ---")
    
    # 1) Inicialitza model, optimitzador i loss
    gat       = GAT(768, 256, 2, use_edge_attr=True).to(device)
    optimizer = torch.optim.Adam(gat.parameters(), lr=0.001)
    loss_fn   = torch.nn.CrossEntropyLoss()
    
    # 2) Carrega dades
    train_data = [
        Data(x=features_list[i], edge_index=adj_list[i], edge_attr=edge_attr_list[i],
             y=torch.tensor([y_no_hosp[i]]))
        for i in train_idx
    ]
    val_data = [
        Data(x=features_list[i], edge_index=adj_list[i], edge_attr=edge_attr_list[i],
             y=torch.tensor([y_no_hosp[i]]))
        for i in val_idx
    ]
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_data,   batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 3) Entrenament (10 epochs)
    t_loss = []
    for epoch in range(25):
        gat.train()
        epoch_losses = []
        for batch in train_loader:
            batch.batch = torch.arange(1).repeat_interleave(196)
            batch = batch.to(device)
            optimizer.zero_grad()
            out, _ = gat(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(out, batch.y.long())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        t_loss.append(np.mean(epoch_losses))
        print(f"  Epoch {epoch+1} — Train Loss: {t_loss[-1]:.4f}")

    all_train_losses.append(t_loss)

    # 4) Validació FINAL amb el model entrenat
    gat.eval()
    val_losses = []
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out, _ = gat(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(out, batch.y.long())
            val_losses.append(loss.item())
            
            probs = F.softmax(out, dim=1)
            y_scores.append(probs[:,1].cpu().item())
            y_pred.append(probs.argmax(dim=1).cpu().item())
            y_true.append(batch.y.cpu().item())

    avg_val_loss = np.mean(val_losses)
    val_loss_per_fold.append(avg_val_loss)
    val_auc = roc_auc_score(np.array(y_true), np.array(y_scores))
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    roc_data_per_fold.append((fpr, tpr, roc_auc))
    print(f"  Fold {fold+1} — Val Loss: {avg_val_loss:.4f}, AUC: {val_auc:.4f}")

    # 5) Guarda millor model
    if val_auc > best_val_auc:
        best_val_auc     = val_auc
        best_model_state = gat.state_dict()

    # 6) Mètriques
    metrics["recall_0"].append(recall_score(y_true, y_pred, pos_label=0))
    metrics["recall_1"].append(recall_score(y_true, y_pred, pos_label=1))
    metrics["precision_0"].append(precision_score(y_true, y_pred, pos_label=0))
    metrics["precision_1"].append(precision_score(y_true, y_pred, pos_label=1))
    metrics["f1_0"].append(f1_score(y_true, y_pred, pos_label=0))
    metrics["f1_1"].append(f1_score(y_true, y_pred, pos_label=1))
    metrics["auc"].append(val_auc)

# 7) Plots finals

# Training loss per fold (corba de 10 punts)
plt.figure(figsize=(8, 5))
for i, losses in enumerate(all_train_losses):
    plt.plot(losses, label=f'Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Train Loss per Fold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/train_loss_per_fold_ViTGAT.png', dpi=300)
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
plt.savefig("/fhome/pmarti/TFGPau/RocCurves/ViTGAT_Mitjana_ROC_PerFold.png", dpi=300)
plt.close()

# Validation loss únic per fold (punt)
plt.figure(figsize=(6, 4))

# 1) amb línia
plt.plot(
    range(1, len(val_loss_per_fold) + 1),   # X = folds 1,2,3…
    val_loss_per_fold,                      # Y = les losses
    'o-',                                   # punts amb línia
    label='Val Loss per Fold'
)

# 2) (opcional) o bé si vols un punt per fold sense línia:
# plt.scatter(range(1, len(val_loss_per_fold) + 1), val_loss_per_fold)

plt.xlabel('Fold')
plt.ylabel('Val Loss')
plt.title('Validation Loss per Fold')
plt.xticks(range(1, len(val_loss_per_fold) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/val_loss_per_fold_ViTGAT.png', dpi=300)
plt.close()

# Print averaged metrics
print("Averaged Metrics:")
for key, values in metrics.items():
    print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

td = [Data(x=f_h[i], edge_index=a_h[i], edge_attr=edge_attr_list_h[i], y=torch.tensor([y_hosp[i]])) for i in range(len(y_hosp))]
tl = DataLoader(td, batch_size=1, shuffle=False, collate_fn=collate_fn)
model = GAT(768,256,2, threshold=0.0).to(device)
model.load_state_dict(best_model_state); model.eval()
yt, yp, ys = [], [], []
with torch.no_grad():
    for b in tl:
        b = b.to(device)
        logits, _ = model(b.x, b.edge_index, b.edge_attr, b.batch)
        probs = F.softmax(logits, dim=1)
        yt.append(b.y.item())
        yp.append(probs.argmax(dim=1).item())
        ys.append(probs[0,1].item())
# final holdout metrics
print("\nHoldout results:")
print(f"AUC: {roc_auc_score(yt, ys):.4f}")
print(f"Recall 0: {recall_score(yt, yp, pos_label=0):.4f}")
print(f"Recall 1: {recall_score(yt, yp, pos_label=1):.4f}")
print(f"Precision 0: {precision_score(yt, yp, pos_label=0):.4f}")
print(f"Precision 1: {precision_score(yt, yp, pos_label=1):.4f}")
print(f"F1 0: {f1_score(yt, yp, pos_label=0):.4f}")
print(f"F1 1: {f1_score(yt, yp, pos_label=1):.4f}")

# ================= STEP 6: Plot Loss Curve =================
plt.figure(figsize=(10,6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(r'/fhome/pmarti/TFGPau/ViTGAT_loss(train).png', dpi=300)
plt.close()

# Validation loss únic per fold (punt)
plt.figure(figsize=(6, 4))

# 1) amb línia
plt.plot(
    range(1, len(val_loss_per_fold) + 1),   # X = folds 1,2,3…
    val_loss_per_fold,                      # Y = les losses
    'o-',                                   # punts amb línia
    label='Val Loss per Fold'
)

# 2) (opcional) o bé si vols un punt per fold sense línia:
# plt.scatter(range(1, len(val_loss_per_fold) + 1), val_loss_per_fold)

plt.xlabel('Fold')
plt.ylabel('Val Loss')
plt.title('Validation Loss per Fold')
plt.xticks(range(1, len(val_loss_per_fold) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/ViTGAT_loss(validacio).png', dpi=300)
plt.close()
