import os
import numpy as np
import glob
import cv2 as cv
from ismember import ismember
from random import shuffle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
from torch import nn, optim
from transformers import ViTModel, ViTFeatureExtractor
from torch.utils.data import Dataset, DataLoader


data = np.load('tissueDades.npz', allow_pickle=True)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
vit_model = ViTModel.from_pretrained(model_name).to(device)

for param in vit_model.parameters():
    param.requires_grad = False

class ViTEmbeddingDataset(Dataset):
    def __init__(self, image_paths, labels, feature_extractor):
        self.image_paths = image_paths
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.fromarray(self.image_paths[idx].astype(np.uint8)).convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = vit_model(**inputs).last_hidden_state[:, 0, :].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embeddings, label

in_dimension = 768
out_dimension = len(np.unique(y_no_hosp))

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(in_features=768, out_features=2, bias=True)
    
    def forward(self, x):
        return self.net(x)

classCount = torch.bincount(torch.tensor(y_no_hosp))
classWeights = 1.0 / classCount.float()
classWeights = classWeights / classWeights.sum()
classWeights = classWeights.to(device)

def train_model(train_loader, val_loader, metrics,model ,epochs=25):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=classWeights)
    t_loss = []

    for epoch in range(epochs):
        e_loss = []
        model.train()
        train_loss = 0.0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            e_loss.append(loss.item())
            train_loss += loss.item()
        t_loss.append(np.mean(e_loss))
        
        # Validación
        model.eval()
        val_preds, val_true, val_scores = [], [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)

                probs = F.softmax(outputs, dim=1)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                val_scores.extend(probs[:, 1].cpu().numpy())
        
        metrics["recall_0"].append(recall_score(val_true, val_preds, pos_label=0))
        metrics["recall_1"].append(recall_score(val_true, val_preds, pos_label=1))
        metrics["precision_0"].append(precision_score(val_true, val_preds, pos_label=0))
        metrics["precision_1"].append(precision_score(val_true, val_preds, pos_label=1))
        metrics["f1_0"].append(f1_score(val_true, val_preds, pos_label=0))
        metrics["f1_1"].append(f1_score(val_true, val_preds, pos_label=1))
        metrics["auc"].append(roc_auc_score(val_true, val_scores))
    
    return model, t_loss

# 6. Stratified K-Fold Cross-Validation
n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)


metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": []}
all_loss = []
model = Classifier().to(device)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_no_hosp, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n=== Fold {fold + 1} ===")
    X_train_fold = [X_no_hosp[i] for i in train_idx]
    y_train_fold = y_no_hosp[train_idx]
    X_val_fold = [X_no_hosp[i] for i in val_idx]
    y_val_fold = y_no_hosp[val_idx]

    # Crear DataLoaders
    train_dataset = ViTEmbeddingDataset(X_train_fold, y_train_fold, feature_extractor)
    val_dataset = ViTEmbeddingDataset(X_val_fold, y_val_fold, feature_extractor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Entrenar y validar
    model, t_loss = train_model(train_loader, val_loader, metrics, model, epochs=10)
    all_loss.append(t_loss)

print("Averaged Metrics:")
for key, values in metrics.items():
    print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

plt.figure(figsize=(10, 6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/Vit+CNN.png', dpi=300)
plt.close()

test_dataset = ViTEmbeddingDataset(X_hosp, y_hosp, feature_extractor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
test_preds, test_true, test_scores = [], [], []
with torch.no_grad():
    for embeddings, labels in test_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        outputs = model(embeddings)
        probs = F.softmax(outputs, dim=1)
        test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        test_true.extend(labels.cpu().numpy())
        test_scores.extend(probs[:, 1].cpu().numpy())

        # Calcular mètriques
        auc_scores = roc_auc_score(test_true, test_scores)
        recall_benigne = recall_score(test_true, test_preds, pos_label=0)
        recall_maligne = recall_score(test_true, test_preds, pos_label=1)
        precision_benigne = precision_score(test_true, test_preds, pos_label=0)
        precision_maligne = precision_score(test_true, test_preds, pos_label=1)
        f1_benigne = f1_score(test_true, test_preds, pos_label=0)
        f1_maligne = f1_score(test_true, test_preds, pos_label=1)

        # Mostrar resultats
        print(f'Holdout results: ')
        print(f"AUC: {auc_scores:.4f}")
        print(f"Recall Benigne: {recall_benigne:.4f}")
        print(f"Recall Maligne: {recall_maligne:.4f}")
        print(f"Precision Benigne: {precision_benigne:.4f}")
        print(f"Precision Maligne: {precision_maligne:.4f}")
        print(f"F1-score Benigne: {f1_benigne:.4f}")
        print(f"F1-score Maligne: {f1_maligne:.4f}")
