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
data = np.load('/fhome/pmarti/TFGPau/processedIms.npz', allow_pickle=True)
data1 = np.load('/fhome/pmarti/TFGPau/tissueDades.npz', allow_pickle=True)

features_list = data['features_list_single_no_h'] 
adj_list = data['adj_list_single_no_h']
edge_attr_list = data['edge_attr_list__single_no_h']

features_list_h = data['features_list_single_h'] 
adj_list_h = data['adj_list_single_h']
edge_attr_list_h = data['edge_attr_list_single_h']

y_hosp = data1['y_hosp']
y_no_hosp = data1['y_no_hosp']

def collate_fn(batch):
    return Batch.from_data_list(batch)

# ================= STEP 2: Define GAT Model =================
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_dim, heads=1, concat=True, edge_dim=1)  #potser pots posar els 4 heads que tenies tu
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=True, edge_dim=1)
        self.fc = torch.nn.Linear(hidden_dim, out_channels)  # Capa final

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_attr)

        # 🔹 Global Pooling para obtener representación del grafo "AIXÒ ES EL QUE ET DEIA EN COMPTES DE FER EL MEAN FORA DE TRAINING
        x = global_mean_pool(x, batch)

        return F.log_softmax(self.fc(x), dim=1)  # Salida con 2 clases

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
    gat       = GAT(768, 256, 2).to(device)
    optimizer = torch.optim.Adam(gat.parameters(), lr=0.001)
    loss_fn   = torch.nn.CrossEntropyLoss(weight=classWeights)
    
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
            out = gat(batch)
            loss = loss_fn(out, batch.y)
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
            out  = gat(batch)
            loss = loss_fn(out, batch.y)
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
        torch.save(best_model_state, f"best_model_fold{fold+1}.pth")

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

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # Cridem el model amb l'objecte Batch complet
            output = model(batch)                # ara output té shape [batch_size, 2]
            probs  = F.softmax(output, dim=1)    # probabilitats per cada classe
            preds  = probs.argmax(dim=1)         # classes prediccions
            
            # Convertim tensors a valors Python correctament
            for true_label, pred_label, score in zip(batch.y.cpu(), preds.cpu(), probs[:,1].cpu()):
                y_true.append(true_label.item())
                y_pred.append(pred_label.item())
                y_scores.append(score.item())
    
    return np.array(y_true), np.array(y_pred), np.array(y_scores)


features_list, adj_list, edge_attr_list = process_images(X_hosp)

test_data = [Data(x=features_list_h[i], edge_index=adj_list_h[i], edge_attr=edge_attr_list_h[i], y=torch.tensor([y_hosp[i]])) for i in range(len(y_hosp))]
test_loader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=collate_fn)


gat = GAT(in_channels=768, hidden_dim=256, out_channels=2).to(device)
gat.load_state_dict(best_model_state)
torch.save(best_model_state, f"/fhome/pmarti/TFGPau/BestViTGAT.pth")
gat.eval()

y_true, y_pred, y_scores = evaluate_model(gat, test_loader, device)

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
plt.savefig("/fhome/pmarti/TFGPau/RocCurves/ViTGAT_Mitjana_ROC_Holdout.png", dpi=300)
plt.close()
# Calcular mètriques
auc_scores = roc_auc_score(np.array(y_true), np.array(y_scores))
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
