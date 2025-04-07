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
from sklearn.model_selection import StratifiedKFold
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


# -----------------------
# CONFIGURACIONS I CARREGADA DE DADES ORIGINALS
DataDir = r'/fhome/pmarti/TFGPau/tissue_images'
os.chdir(DataDir)

# Carregar imatges i mÃ scares
files = glob.glob('*_mask.png')
files = np.array(files)
files = files[np.array([file.find('Manresa') for file in files]) == -1]
Centers = ['Terrassa', 'Alicante', 'Manresa', 'Basurto', 'Bellvitge']
filesCenter = {}
for Center in Centers:
    filesCenter[Center] = np.array([file.find(Center) for file in files]) == -1

PatID = []
ims = []
masks = []
fileID = []
CenterID = []
for k in np.arange(len(files)):
    file = files[k]
    im = cv.imread(file.split('_mask')[0] + '.png')[:, :, 0:3]
    if min(im.shape[0:2]) >= 128:
        ims.append(im)
        im_mask = np.mean(cv.imread(file)[:, :, 0:3], axis=2) / 255
        masks.append(im_mask)
        PatID.append(file.split('_SampID')[0])
        fileID.append(file.split('_mask')[0])
        CenterID.append(file.split('_')[0])

# Preprocessament de les mÃ scares: 2 -> infiltraciÃ³ (maligne), 1 -> teixit (benigne), 0 -> fons
for k in np.arange(len(masks)):
    bck = masks[k] > 0
    masks[k][np.nonzero(masks[k] == 1)] = 2
    masks[k][np.nonzero(masks[k] != 2)] = 1
    masks[k] = masks[k] * bck.astype(int)

def ImPatches(im, sze, stride):
    szex, szey = sze
    im_patch = [im[x:x+szex, y:y+szey] 
                for x in range(0, im.shape[0], stride) 
                for y in range(0, im.shape[1], stride)]
    valid_patch = []
    for patch in im_patch:
        if min(patch.shape[0:2]) >= min(sze):
            valid_patch.append(patch)
    if len(valid_patch) > 0:       
        im_patch = np.stack(valid_patch)
    return im_patch

sze = [224, 224]
stride = 224
im_patches = []
PatID_patches = []
fileID_patches = []
tissue = []
patho = []
CenterID_patches = []

for k in np.arange(len(masks)):
    im = masks[k]
    if min(im.shape) > sze[0]:
        patches = ImPatches(im, sze, stride)
        pathotissue_patch = np.sum(patches.reshape(patches.shape[0], np.prod(sze)) == 2, axis=1)
        tissue_patch = np.sum(patches.reshape(patches.shape[0], np.prod(sze)) > 0, axis=1)
        tissue.append(tissue_patch)
        patho.append(pathotissue_patch)
        im_img = ims[k]
        patches = ImPatches(im_img, sze, stride)
        im_patches.append(patches)
        PatID_patches.append(np.repeat(PatID[k], patches.shape[0]))
        fileID_patches.append(np.repeat(fileID[k], patches.shape[0]))
        CenterID_patches.append(np.repeat(CenterID[k], patches.shape[0]))

im_patches = np.concatenate(im_patches, axis=0)
tissue = np.concatenate(tissue, axis=0)
patho = np.concatenate(patho, axis=0)
PatID_patches = np.concatenate(PatID_patches, axis=0)
CenterID_patches = np.concatenate(CenterID_patches, axis=0)
fileID_patches = np.concatenate(fileID_patches, axis=0)

th_percen_tissue = 0.75
id_tissue = np.nonzero(tissue / np.prod(sze) > th_percen_tissue)[0]

y_true = patho[id_tissue] / tissue[id_tissue]
idxpatho = np.nonzero(y_true == 1)[0]
idxother = np.nonzero(y_true < 1)[0]
idxsel = np.concatenate((idxpatho[0::5], idxother))
shuffle(idxsel)
y_true = y_true[idxsel]
y_true = (y_true > 0.5).astype(int)

X = im_patches[id_tissue, :]
CenterID_patches = CenterID_patches[id_tissue]
X = X[idxsel, :]
CenterID_patches = CenterID_patches[idxsel]

X = X.astype(float)
mu = [np.mean(X[:, :, :, ch].flatten()) for ch in range(3)]
std = [np.std(X[:, :, :, ch].flatten()) for ch in range(3)]
for kch in range(3):
    X[:, :, :, kch] = (X[:, :, :, kch] - mu[kch]) / std[kch]

def split_test_hospital(X, y_true, CenterID_patches):
    hospital_counts = Counter(CenterID_patches)
    least_common_hospital = hospital_counts.most_common()[-1][0]
    test_idx = np.where(CenterID_patches == least_common_hospital)[0]
    train_idx = np.where(CenterID_patches != least_common_hospital)[0]
    X_train, y_train = X[train_idx], y_true[train_idx]
    X_test, y_test = X[test_idx], y_true[test_idx]
    return X_train, y_train, X_test, y_test, least_common_hospital

X_no_hosp, y_no_hosp, X_hosp, y_hosp, least = split_test_hospital(X, y_true, CenterID_patches)

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
        self.net = nn.Sequential(
            nn.Linear(in_dimension, round(in_dimension / 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(round(in_dimension / 2), out_dimension)
        )
    
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
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": []}
all_loss = []
model = Classifier().to(device)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_no_hosp, y_no_hosp)):
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