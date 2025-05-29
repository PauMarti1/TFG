import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
import matplotlib.pyplot as plt

# Assignar les variables
data = np.load('/fhome/pmarti/TFGPau/novesDades.npz', allow_pickle=True)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
vit_model = ViTModel.from_pretrained(model_name).to(device)

for param in vit_model.parameters():
    param.requires_grad = False

# Dataset per als embeddings de ViT
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

# Model de classificació
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dimension, round(in_dimension / 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(round(in_dimension / 2), out_dimension)
        )
    
    def forward(self, x):
        return self.net(x)
    
# Pes de les classes
classCount = torch.bincount(torch.tensor(y_no_hosp))
classWeights = 1.0 / classCount.float()
classWeights = classWeights / classWeights.sum()
classWeights = classWeights.to(device)

metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": []}

# Funció d'entrenament
def train_model(train_loader, val_loader, epochs=25):
    model = Classifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    t_loss = []
    
    best_auc = -np.inf
    best_model = None
    
    for epoch in range(epochs):
        e_loss = []
        model.train()
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            e_loss.append(loss.item())
        t_loss.append(np.mean(e_loss))
        
        # Validació
        model.eval()
        val_preds, val_true, val_scores = [], [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                # Obtenir la probabilitat de la classe 1
                probs = F.softmax(outputs, dim=1)[:, 1]
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                val_scores.extend(probs.cpu().numpy().tolist())
        auc = roc_auc_score(val_true, val_scores)
        metrics["recall_0"].append(recall_score(val_true, val_preds, pos_label=0))
        metrics["recall_1"].append(recall_score(val_true, val_preds, pos_label=1))
        metrics["precision_0"].append(precision_score(val_true, val_preds, pos_label=0))
        metrics["precision_1"].append(precision_score(val_true, val_preds, pos_label=1))
        metrics["f1_0"].append(f1_score(val_true, val_preds, pos_label=0))
        metrics["f1_1"].append(f1_score(val_true, val_preds, pos_label=1))
        metrics["auc"].append(roc_auc_score(val_true, val_preds))
        if auc > best_auc:
            best_auc = auc
            best_model = model.state_dict()
            
    # Carregar el millor model per a la validació final
    model.load_state_dict(best_model)
    return model, t_loss, best_auc, val_true, val_scores

# Stratified Group K-Fold Cross-Validation
n_splits = 5
skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=2)

all_true, all_scores, all_loss = [], [], []
best_models = []
fold_auc_scores = []  # Afegim aquesta llista per guardar la millor AUC de cada fold

for fold, (train_idx, val_idx) in enumerate(skf.split(X_no_hosp, y_no_hosp, groups=PatID_no_hosp)):
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

    # Entrenar i validar
    model, t_loss, best_fold_auc, v_true, v_scores = train_model(train_loader, val_loader, epochs=25)
    print(f"He fet un train. Fold {fold + 1} millor AUC: {best_fold_auc:.4f}")
    all_loss.append(t_loss)
    best_models.append(model)
    fold_auc_scores.append(best_fold_auc)
    all_true.append(v_true); all_scores.append(v_scores)

# Mostrar pèrdues d'entrenament per fold
plt.figure(figsize=(10, 6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/Vit+CNN_losses.png', dpi=300)
plt.close()

plt.figure(figsize=(8,6))
for i in range(n_splits):
    fpr, tpr, _ = roc_curve(all_true[i], all_scores[i])
    plt.plot(fpr, tpr, label=f'Fold {i+1} (AUC={fold_auc_scores[i]:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curves per Fold')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.legend(loc='lower right'); plt.grid(True)
plt.tight_layout(); plt.savefig('/fhome/pmarti/TFGPau/RocCurves/Vit_CNN_Arc_cv_roc.png', dpi=300); plt.close()

print("Averaged Metrics:")
for key, values in metrics.items():
    print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# Seleccionar el millor model global (per la millor AUC de validació)
best_fold_idx = np.argmax(fold_auc_scores)
best_model = best_models[best_fold_idx]
print(f"\nEl millor model és del fold {best_fold_idx+1} amb una AUC de validació de {fold_auc_scores[best_fold_idx]:.4f}")

# Test amb el millor model (holdout)
test_dataset = ViTEmbeddingDataset(X_hosp, y_hosp, feature_extractor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

best_model.eval()
test_preds, test_true, test_scores = [], [], []
with torch.no_grad():
    for embeddings, labels in test_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        outputs = best_model(embeddings)
        probs = F.softmax(outputs, dim=1)[:, 1]  # Seleccionar la probabilitat per a classe 1
        test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        test_true.extend(labels.cpu().numpy())
        # Afegir totes les probabilitats (vector unidimensional)
        test_scores.extend(probs.cpu().numpy().tolist())

fpr_h, tpr_h, _ = roc_curve(test_true, test_scores)
auc_h = auc(fpr_h, tpr_h)
plt.figure(figsize=(8,6))
plt.plot(fpr_h, tpr_h, label=f'Holdout (AUC={auc_h:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curve - Holdout')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.legend(loc='lower right'); plt.grid(True)
plt.tight_layout(); plt.savefig('/fhome/pmarti/TFGPau/Vit_CNN_Arc_holdout_roc.png', dpi=300); plt.close()

# Calcular mètriques de test
auc_scores_test = roc_auc_score(test_true, test_scores)
recall_benigne = recall_score(test_true, test_preds, pos_label=0)
recall_maligne = recall_score(test_true, test_preds, pos_label=1)
precision_benigne = precision_score(test_true, test_preds, pos_label=0)
precision_maligne = precision_score(test_true, test_preds, pos_label=1)
f1_benigne = f1_score(test_true, test_preds, pos_label=0)
f1_maligne = f1_score(test_true, test_preds, pos_label=1)

# Mostrar resultats del holdout
print(f'\nHoldout results:')
print(f"AUC: {auc_scores_test:.4f}")
print(f"Recall Benigne: {recall_benigne:.4f}")
print(f"Recall Maligne: {recall_maligne:.4f}")
print(f"Precision Benigne: {precision_benigne:.4f}")
print(f"Precision Maligne: {precision_maligne:.4f}")
print(f"F1-score Benigne: {f1_benigne:.4f}")
print(f"F1-score Maligne: {f1_maligne:.4f}")