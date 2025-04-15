import os
import numpy as np
import glob
import cv2 as cv
from ismember import ismember
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from timm import create_model
from torch.utils.data import DataLoader, Subset, TensorDataset
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt

# Carregar dades
data = np.load('/fhome/pmarti/TFGPau/tissueDades.npz', allow_pickle=True)

# Variables
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

# L'hospital amb menys mostres és el de Terrassa, el qual farem servir per holdout.

# Transformacions per al model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pes per a cada classe
classCount = torch.bincount(torch.tensor(y_no_hosp))
classWeights = 1.0 / classCount.float()
classWeights = classWeights / classWeights.sum()
classWeights = classWeights.to(device)

# Creem la classe dataset
class dat():
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            # Convertim a PIL per aplicar les transformacions
            image = self.transform(Image.fromarray(image.astype(np.uint8)))
        return image, label

dataset = dat(X_no_hosp, y_no_hosp, transform=transform)
test_dataset = dat(X_hosp, y_hosp, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss(weight=classWeights)

# Paràmetres d'entrenament
n_epochs = 10
n_splits = 5
skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Variables per guardar mètriques globals de validació (pot ser per imprimir-les al final)
val_auc_scores = []

# Aquí anirem guardant l'estat del millor model (segons AUC de validació)
best_val_auc = 0.0
best_model_state = None

all_loss = []  # Per emmagatzemar les pèrdues de cada fold

metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": []}
# Entrenament per cada fold
for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(X_no_hosp)), y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n=== Fold {fold + 1}/{n_splits} ===")

    # Re-inicialitzar el model per a cada fold
    model = create_model("deit_base_patch16_224", pretrained=True, num_classes=2)
    # Congelem tots els paràmetres menys la capçalera (head)
    for param in model.parameters():
        param.requires_grad = False
    model.head.requires_grad_(True)
    model.to(device)
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    optimizer = optim.Adam(model.head.parameters(), lr=1e-3)
    fold_loss = []
    best_fold_auc = 0.0  # Millor AUC d'aquest fold

    for epoch in range(25):
        model.train()
        epoch_losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        fold_loss.append(avg_loss)
        print(f"Fold {fold + 1} Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        
        # Validació
        model.eval()
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilitat de classe 1
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calcular AUC de validació
        current_auc = roc_auc_score(all_labels, all_probs)
        print(f"Fold {fold + 1} Epoch {epoch + 1}, Val AUC: {current_auc:.4f}")
        if current_auc > best_fold_auc:
            best_fold_auc = current_auc
            # Guardo l'estat del model d'aquest fold amb millor AUC
            best_fold_state = model.state_dict()
            metrics["recall_0"].append(recall_score(all_labels, all_preds, pos_label=0))
        metrics["recall_1"].append(recall_score(all_labels, all_preds, pos_label=1))
        metrics["precision_0"].append(precision_score(all_labels, all_preds, pos_label=0))
        metrics["precision_1"].append(precision_score(all_labels, all_preds, pos_label=1))
        metrics["f1_0"].append(f1_score(all_labels, all_preds, pos_label=0))
        metrics["f1_1"].append(f1_score(all_labels, all_preds, pos_label=1))
        metrics["auc"].append(current_auc)
    
    all_loss.append(fold_loss)
    val_auc_scores.append(best_fold_auc)
    print(f"Fold {fold + 1} millor AUC: {best_fold_auc:.4f}")
    
    
    
    # Actualitzem el millor model global (segons AUC)
    if best_fold_auc > best_val_auc:
        best_val_auc = best_fold_auc
        best_model_state = best_fold_state

print("\n=== Resultats Globals de Validació ===")
print(f"Mitjana AUC de validació: {np.mean(val_auc_scores):.4f} ± {np.std(val_auc_scores):.4f}")

# Creem un nou model i carreguem l'estat del millor model obtingut
best_model = create_model("deit_base_patch16_224", pretrained=True, num_classes=2)
for param in best_model.parameters():
    param.requires_grad = False
best_model.head.requires_grad_(True)
best_model.to(device)
best_model.load_state_dict(best_model_state)

print("Averaged Metrics:")
for key, values in metrics.items():
    print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# Avaluació sobre el holdout
def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_scores)

y_true_holdout, y_pred_holdout, y_scores_holdout = evaluate_model(best_model, test_loader, device)

# Calcular mètriques de holdout
auc_holdout = roc_auc_score(y_true_holdout, y_scores_holdout)
recall_benigne_holdout = recall_score(y_true_holdout, y_pred_holdout, pos_label=0)
recall_maligne_holdout = recall_score(y_true_holdout, y_pred_holdout, pos_label=1)
precision_benigne_holdout = precision_score(y_true_holdout, y_pred_holdout, pos_label=0)
precision_maligne_holdout = precision_score(y_true_holdout, y_pred_holdout, pos_label=1)
f1_benigne_holdout = f1_score(y_true_holdout, y_pred_holdout, pos_label=0)
f1_maligne_holdout = f1_score(y_true_holdout, y_pred_holdout, pos_label=1)

print("\n=== Holdout Results ===")
print(f"AUC: {auc_holdout:.4f}")
print(f"Recall Benigne: {recall_benigne_holdout:.4f}")
print(f"Recall Maligne: {recall_maligne_holdout:.4f}")
print(f"Precision Benigne: {precision_benigne_holdout:.4f}")
print(f"Precision Maligne: {precision_maligne_holdout:.4f}")
print(f"F1-score Benigne: {f1_benigne_holdout:.4f}")
print(f"F1-score Maligne: {f1_maligne_holdout:.4f}")

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
plt.savefig('/fhome/pmarti/TFGPau/Vit.png', dpi=300)
plt.close()