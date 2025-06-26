import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Carrega les dades
data = np.load('/fhome/pmarti/TFGPau/LargetissueDades_48_Norm.npz', allow_pickle=True)
X_no_hosp = data['X_no_hosp']
y_no_hosp = data['y_no_hosp']
PatID_no_hosp = data['PatID_no_hosp']
X_hosp = data['X_hosp']
y_hosp = data['y_hosp']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. Transforms i model ResNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resnet = models.densenet121()
resnet = nn.Sequential(*list(resnet.children())[:-1])  # treiem la fc (última capa)
resnet.to(device)
resnet.eval()  # congelar peses
for param in resnet.parameters():
    param.requires_grad = False

embedding_dim = 64  # resnet50 treu embeddings de 2048 dimensions
out_dimension = len(np.unique(y_no_hosp))

# === 2. Dataset
class ResNetEmbeddingDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = transform(self.images[idx])
        image = image.to(device).unsqueeze(0)
        with torch.no_grad():
            embedding = resnet(image).squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding, label

# === 3. Classificador (MLP)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(50176, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim // 2, out_dimension)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

# === 4. Peses de classe

# === 5. Funció d'entrenament
def train_model(train_loader, val_loader, epochs=7):
    model = Classifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_auc = -np.inf
    best_model = None
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * embeddings.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)

        # Validació
        model.eval()
        val_preds, val_true, val_scores = [], [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                probs = F.softmax(outputs, dim=1)[:, 1]
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                val_scores.extend(probs.cpu().numpy())

        val_auc = roc_auc_score(val_true, val_scores)
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()
    
    # Calcular mètriques de validació finals pel millor model
    model.load_state_dict(best_model)
    model.eval()
    val_preds, val_true, val_scores = [], [], []
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            probs = F.softmax(outputs, dim=1)[:, 1]
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_true.extend(labels.cpu().numpy())
            val_scores.extend(probs.cpu().numpy())

    # Guardem mètriques
    metrics["recall_0"].append(recall_score(val_true, val_preds, pos_label=0))
    metrics["recall_1"].append(recall_score(val_true, val_preds, pos_label=1))
    metrics["precision_0"].append(precision_score(val_true, val_preds, pos_label=0))
    metrics["precision_1"].append(precision_score(val_true, val_preds, pos_label=1))
    metrics["f1_0"].append(f1_score(val_true, val_preds, pos_label=0))
    metrics["f1_1"].append(f1_score(val_true, val_preds, pos_label=1))
    metrics["auc"].append(roc_auc_score(val_true, val_scores))
    metrics["y_true"].append(val_true)
    metrics["y_pred"].append(val_preds)
    metrics['y_scores'].append(val_scores)

    return model, best_auc, epoch_losses, val_true, val_scores


# === 6. KFold + Entrenament
n_splits = 5
skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=2)

best_models = []
fold_auc_scores = []
all_loss = []
metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": [], "y_true" : [], "y_pred" : [], 'y_scores' : []}
roc_folds = []  # per guardar (fpr, tpr, auc)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_no_hosp, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n=== Fold {fold + 1} ===")
    train_dataset = ResNetEmbeddingDataset([X_no_hosp[i] for i in train_idx], y_no_hosp[train_idx])
    val_dataset = ResNetEmbeddingDataset([X_no_hosp[i] for i in val_idx], y_no_hosp[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model, best_auc, losses, val_true, val_scores = train_model(train_loader, val_loader)

    fpr, tpr, _ = roc_curve(val_true, val_scores)
    roc_auc = auc(fpr, tpr)
    roc_folds.append((fpr, tpr, roc_auc))


    best_models.append(model)
    fold_auc_scores.append(best_auc)
    all_loss.append(losses)
    print(f"Fold {fold+1} AUC: {best_auc:.4f}")

plt.figure(figsize=(10, 6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/DenseNet_MLP_Loss.png', dpi=300)
plt.close() 

# plt.figure(figsize=(10, 6))
# for i, (fpr, tpr, roc_auc) in enumerate(roc_folds):
#     plt.plot(fpr, tpr, label=f"Fold {i+1} (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], 'k--', label="Random")
# plt.title("ROC Curves per Fold")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('/fhome/pmarti/TFGPau/RocCurves/DenseNet_MLP_ROC_PerFold.png', dpi=300)
# plt.close()


# # Mostrar mètriques mitjanes i desviacions
# print("\n--- Validation Metrics per Fold ---")
# for key in metrics:
#     mean = np.mean(metrics[key])
#     std = np.std(metrics[key])
#     print(f"{key}: {mean:.4f} ± {std:.4f}")

np.savez("/fhome/pmarti/TFGPau/NPZs/DenseNet.npz", 
    metrics=np.array(metrics, dtype=object),
    val_loss_per_fold=np.array(all_loss, dtype=object),
    roc_data_per_fold=np.array(roc_folds, dtype=object), allow_pickle=True)

# # === 7. Test amb holdout
# best_fold_idx = np.argmax(fold_auc_scores)
# best_model = best_models[best_fold_idx]
# print(f"\nMillor model: Fold {best_fold_idx+1} amb AUC = {fold_auc_scores[best_fold_idx]:.4f}")

# test_dataset = ResNetEmbeddingDataset(X_hosp, y_hosp)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# best_model.eval()
# test_preds, test_true, test_scores = [], [], []
# with torch.no_grad():
#     for embeddings, labels in test_loader:
#         embeddings, labels = embeddings.to(device), labels.to(device)
#         outputs = best_model(embeddings)
#         probs = F.softmax(outputs, dim=1)[:, 1]
#         test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
#         test_true.extend(labels.cpu().numpy())
#         test_scores.extend(probs.cpu().numpy())

# # Resultats
# print(f'\nHoldout Results:')
# print(f"AUC: {roc_auc_score(test_true, test_scores):.4f}")
# print(f"Recall (benigne): {recall_score(test_true, test_preds, pos_label=0):.4f}")
# print(f"Recall (maligne): {recall_score(test_true, test_preds, pos_label=1):.4f}")
# print(f"Precision (benigne): {precision_score(test_true, test_preds, pos_label=0):.4f}")
# print(f"Precision (maligne): {precision_score(test_true, test_preds, pos_label=1):.4f}")
# print(f"F1-score (benigne): {f1_score(test_true, test_preds, pos_label=0):.4f}")
# print(f"F1-score (maligne): {f1_score(test_true, test_preds, pos_label=1):.4f}")

# fpr_test, tpr_test, _ = roc_curve(test_true, test_scores)
# roc_auc_test = auc(fpr_test, tpr_test)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr_test, tpr_test, label=f"Holdout ROC (AUC = {roc_auc_test:.2f})", color="darkorange")
# plt.plot([0, 1], [0, 1], 'k--', label='Random')
# plt.title("ROC Curve - Holdout Set")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("/fhome/pmarti/TFGPau/RocCurves/DenseNet_ROC_Holdout.png", dpi=300)
# plt.close()