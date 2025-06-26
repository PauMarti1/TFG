import sys
import os
import numpy as np
import pandas as pd
import glob
import cv2 as cv
from matplotlib import pyplot as plt
from ismember import ismember
from random import shuffle, choice
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from timm import create_model
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
from collections import Counter
from timm.models.swin_transformer import SwinTransformer

# Carrega les dades
data = np.load('/fhome/pmarti/TFGPau/LargetissueDades_48_Norm.npz', allow_pickle=True)

# Assignar les variables
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

# Transformacions per als imatges
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Definició del dispositiu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Funcions per a la creació del model base i el carregament de pesos

def build_ctranspath():
    # Crea una instància de SwinTransformer sense la capa de classificació final.
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=0,  # No hi ha cap de classificació final
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2
    )
    return model

def load_ctranspath(model, weights_path):
    state_dict = torch.load(weights_path, map_location='cpu')
    if 'model' in state_dict:  # Si la clau 'model' existeix
        state_dict = state_dict['model']
    # Eliminar prefix "module." si s'hi troba
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(filtered_state_dict, strict=False)
    return model

# Modul per canviar l'ordre de dimensions
class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

# Funció per crear un model complet amb la part base i la head nova.
def build_model():
    # Crea la part base i carrega pesos preentrenats
    base_model = build_ctranspath()
    base_model = load_ctranspath(base_model, '/fhome/pmarti/TFGPau/Models/ctranspath.pth')
    # Afegir la head per a classificació.
    base_model.head = nn.Sequential(
        nn.LayerNorm(base_model.num_features),  # Normalització
        Permute((0, 3, 1, 2)),                  # De [B,H,W,C] a [B,C,H,W]
        nn.AdaptiveAvgPool2d(1),                # Pooling a [B,C,1,1]
        nn.Flatten(),                           # [B,C]
        nn.Linear(base_model.num_features, 2)   # Capa final de classificació binària
    )
    # Congelem tots els paràmetres excepte la head
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.head.requires_grad_(True)
    return base_model

# Creació del dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        image = Image.fromarray(image.astype('uint8'))  # Convertim a PIL
        if self.transform:
            image = self.transform(image)
        return image, label

# Datasets de train/validation i holdout
train_val_dataset = CustomDataset(X_no_hosp, y_no_hosp, transform=transform)
test_dataset = CustomDataset(X_hosp, y_hosp, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()

# Inicialitzar variables per a la validació creuada
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2)
all_loss = []           # Per guardar les pèrdues per fold
val_auc_scores = []     # Per a cada fold, la millor AUC de validació
best_global_auc = 0.0
best_model_state = None
metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": [], "y_true" : [], "y_pred" : [], 'y_scores' : []}
# Validació creuada: per a cada fold, crear un model nou i entrenar-lo
for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(X_no_hosp)), y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n=== Fold {fold + 1}/5 ===")
    
    # Reinicialitzar el model
    model = build_model()
    model.to(device)
    
    # Crear subsets i DataLoaders
    train_dataset = Subset(train_val_dataset, train_idx)
    val_dataset = Subset(train_val_dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    optimizer = optim.Adam(model.head.parameters(), lr=1e-3)
    fold_losses = []
    best_fold_auc = 0.0
    best_fold_state = None

    for epoch in range(7):
        model.train()
        epoch_losses = []
        for images, labels in train_loader:
            # Per aquest exemple, fem servir "cpu" o "device" segons el que necessitis; aquí s'usa device
            images, labels = images.to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        fold_losses.append(avg_loss)
        print(f"Fold {fold + 1} Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        
        # Validació
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.long().to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilitat de classe 1 (maligna)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        current_auc = roc_auc_score(all_labels, all_probs)
        print(f"Fold {fold + 1} Epoch {epoch + 1}, Val AUC: {current_auc:.4f}")
        
        # Si obtenim una AUC millor en aquest fold, guardem l'estat
        if current_auc > best_fold_auc:
            best_fold_auc = current_auc
            best_fold_state = model.state_dict()
        metrics["recall_0"].append(recall_score(all_labels, all_preds, pos_label=0))
        metrics["recall_1"].append(recall_score(all_labels, all_preds, pos_label=1))
        metrics["precision_0"].append(precision_score(all_labels, all_preds, pos_label=0))
        metrics["precision_1"].append(precision_score(all_labels, all_preds, pos_label=1))
        metrics["f1_0"].append(f1_score(all_labels, all_preds, pos_label=0))
        metrics["f1_1"].append(f1_score(all_labels, all_preds, pos_label=1))
        metrics["auc"].append(roc_auc_score(all_labels, all_preds))
        metrics["y_true"].append(all_labels)
        metrics["y_pred"].append(all_preds)
        metrics['y_scores'].append(all_probs)
    
    all_loss.append(fold_losses)
    val_auc_scores.append(best_fold_auc)
    print(f"Fold {fold + 1} millor AUC: {best_fold_auc:.4f}")
    
    # Actualitzem el millor model global si aquesta AUC és superior a les anteriors
    if best_fold_auc > best_global_auc:
        best_global_auc = best_fold_auc
        best_model_state = best_fold_state

# print(all_loss)
# print("\n=== Resultats Globals de Validació ===")
# print(f"Mitjana Val AUC: {np.mean(val_auc_scores):.4f} ± {np.std(val_auc_scores):.4f}")

# print("Averaged Metrics:")
# for key, values in metrics.items():
#     print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# # Carregar el millor model global per al holdout
# best_model = build_model()
# best_model.to(device)
# best_model.load_state_dict(best_model_state)

# # Funció per avaluar el model
# def evaluate_model(model, dataloader, device):
#     model.eval()
#     y_true, y_pred, y_scores = [], [], []
#     with torch.no_grad():
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             probs = torch.softmax(outputs, dim=1)[:, 1]
#             preds = torch.argmax(outputs, dim=1)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(preds.cpu().numpy())
#             y_scores.extend(probs.cpu().numpy())
#     return np.array(y_true), np.array(y_pred), np.array(y_scores)

# y_true_holdout, y_pred_holdout, y_scores_holdout = evaluate_model(best_model, test_loader, device)

# # Calcular mètriques per al holdout
# auc_holdout = roc_auc_score(y_true_holdout, y_scores_holdout)
# recall_benigne_holdout = recall_score(y_true_holdout, y_pred_holdout, pos_label=0)
# recall_maligne_holdout = recall_score(y_true_holdout, y_pred_holdout, pos_label=1)
# precision_benigne_holdout = precision_score(y_true_holdout, y_pred_holdout, pos_label=0)
# precision_maligne_holdout = precision_score(y_true_holdout, y_pred_holdout, pos_label=1)
# f1_benigne_holdout = f1_score(y_true_holdout, y_pred_holdout, pos_label=0)
# f1_maligne_holdout = f1_score(y_true_holdout, y_pred_holdout, pos_label=1)

# print("\n=== Holdout Results ===")
# print(f"AUC: {auc_holdout:.4f}")
# print(f"Recall Benigne: {recall_benigne_holdout:.4f}")
# print(f"Recall Maligne: {recall_maligne_holdout:.4f}")
# print(f"Precision Benigne: {precision_benigne_holdout:.4f}")
# print(f"Precision Maligne: {precision_maligne_holdout:.4f}")
# print(f"F1-score Benigne: {f1_benigne_holdout:.4f}")
# print(f"F1-score Maligne: {f1_maligne_holdout:.4f}")

# Opcional: Plotejar la pèrdua d'entrenament per cada fold
plt.figure(figsize=(10, 6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/ViTPreentrenat.png', dpi=300)
plt.close()

n_splits=5

roc_folds = []
for i in range(n_splits):
    y_true = metrics["y_true"][i]
    y_scores = metrics["y_scores"][i]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_folds.append((fpr, tpr))

np.savez("/fhome/pmarti/TFGPau/NPZs/ViTPreentrenat.npz", 
    metrics=np.array(metrics, dtype=object),
    val_loss_per_fold=np.array(all_loss, dtype=object),
    roc_data_per_fold=np.array(roc_folds, dtype=object), allow_pickle=True)