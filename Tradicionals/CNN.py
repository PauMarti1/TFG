import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, roc_curve

# Load data
data = np.load('/fhome/pmarti/TFGPau/LargetissueDades_48_Norm.npz', allow_pickle=True)
X_no_hosp = data['X_no_hosp']
y_no_hosp = data['y_no_hosp']
PatID_no_hosp = data['PatID_no_hosp']
X_hosp = data['X_hosp']
y_hosp = data['y_hosp']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms per imatges
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Dataset per CNN
class CNNImageDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Model CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 8 * 8, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32x32x32
        x = self.pool(F.relu(self.conv2(x)))   # 64x16x16
        x = self.pool(F.relu(self.conv3(x)))   # 128x8x8
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc(x)
        return x

# Entrenament
def train_model(train_loader, val_loader, model, epochs=25):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    t_loss = []
    metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": [], "y_true" : [], "y_pred" : [], 'y_scores' : []}

    for epoch in range(7):
        model.train()
        epoch_loss = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        t_loss.append(np.mean(epoch_loss))

        # Validació
        model.eval()
        val_preds, val_true, val_scores = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                val_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                val_scores.extend(probs[:, 1].cpu().numpy())

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
    
    return model, t_loss, metrics, val_true, val_scores

# StratifiedGroupKFold
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2)
all_metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": []}
all_loss = []
all_val_true = []
all_val_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_no_hosp, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n=== Fold {fold + 1} ===")
    model = SimpleCNN().to(device)
    train_dataset = CNNImageDataset(X_no_hosp[train_idx], y_no_hosp[train_idx], transform)
    val_dataset = CNNImageDataset(X_no_hosp[val_idx], y_no_hosp[val_idx], transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    model, losses, fold_metrics, val_true, val_scores = train_model(train_loader, val_loader, model)
    all_loss.append(losses)
    all_val_true.append(val_true)
    all_val_scores.append(val_scores)
    for key in all_metrics:
        all_metrics[key].append(np.mean(fold_metrics[key]))

# Resultats
print("\n== Mitjanes cross-validation ==")
for key, values in all_metrics.items():
    print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# Plots
plt.figure(figsize=(10, 6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/fhome/pmarti/TFGPau/images/CNN_Loss_per_fold.png", dpi=300)
plt.close()

roc_folds = []

for i in range(len(all_val_true)):
    fpr, tpr, _ = roc_curve(all_val_true[i], all_val_scores[i])
    roc_folds.append((fpr, tpr))


np.savez(
    '/fhome/pmarti/TFGPau/NPZs/CNN.npz',
    metrics=np.array(fold_metrics, dtype=object),
    val_loss_per_fold=np.array(all_loss, dtype=object),
    roc_data_per_fold=np.array(roc_folds, dtype=object), allow_pickle=True)
# plt.figure(figsize=(10, 6))
# 
#     plt.plot(fpr, tpr, label=f"Fold {i+1} (AUC = {auc_score:.2f})")

# plt.plot([0, 1], [0, 1], 'k--', label="Random")
# plt.title("ROC Curve per Fold")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("/fhome/pmarti/TFGPau/RocCurves/CNN_ROC_per_fold.png", dpi=300)
# plt.close()

# # Avaluació en holdout
# test_dataset = CNNImageDataset(X_hosp, y_hosp, transform)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# model.eval()
# test_preds, test_true, test_scores = [], [], []
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         probs = F.softmax(outputs, dim=1)
#         test_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
#         test_true.extend(labels.cpu().numpy())
#         test_scores.extend(probs[:, 1].cpu().numpy())

# print(f"\n== Holdout Results ==")
# print(f"AUC: {roc_auc_score(test_true, test_scores):.4f}")
# print(f"Recall Benigne: {recall_score(test_true, test_preds, pos_label=0):.4f}")
# print(f"Recall Maligne: {recall_score(test_true, test_preds, pos_label=1):.4f}")
# print(f"Precision Benigne: {precision_score(test_true, test_preds, pos_label=0):.4f}")
# print(f"Precision Maligne: {precision_score(test_true, test_preds, pos_label=1):.4f}")
# print(f"F1-score Benigne: {f1_score(test_true, test_preds, pos_label=0):.4f}")
# print(f"F1-score Maligne: {f1_score(test_true, test_preds, pos_label=1):.4f}")

# fpr, tpr, _ = roc_curve(test_true, test_scores)
# auc_score = roc_auc_score(test_true, test_scores)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
# plt.plot([0, 1], [0, 1], 'k--', label="Random")
# plt.title("ROC Curve - Holdout Set")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("/fhome/pmarti/TFGPau/RocCurves/CNN_ROC_holdout.png", dpi=300)
# plt.close()