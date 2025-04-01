import os
import numpy as np
import glob
import cv2 as cv
from ismember import ismember
from random import shuffle
import torch
import timm
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from collections import Counter
import torch.nn as nn
from torchvision.models import vit_b_16
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torchvision.transforms as transforms

# -----------------------
# CONFIGURACIONS I CARREGADA DE DADES ORIGINALS
DataDir = r'/Users/paumarti/Desktop/TFG/tissue_images'
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

# -----------------------
# Definicions de models i transformacions
MODEL_NAME = "facebook/deit-base-patch16-224"
vit_model = AutoModelForImageClassification.from_pretrained(MODEL_NAME, output_attentions=True)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
vit_model.eval()  # Per extracció de features posem el model en eval

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset custom
class CustomDataset():
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
            image = self.transform(Image.fromarray(image.astype(np.uint8)))
        return image, label
    
class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=3):
        super(GNNClassifier, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

# -----------------------
# EXTRACCIÓ DE FEATURES PER AL CONJUNT X_no_hosp (KFold)
dataset_kfold = CustomDataset(X_no_hosp, y_no_hosp, transform=transform)
loader_kfold = DataLoader(dataset_kfold, batch_size=32, shuffle=False)
all_features_kfold = []
with torch.no_grad():
    for images, _ in loader_kfold:
        images = images.to("cpu")
        outputs = vit_model(images)
        # Extreure la matriu d'atenciÃ³ del primer bloc (shape: [B, H, N, N])
        attn = outputs.attentions[0]
        features = attn.mean(dim=1)  # Mitjana sobre els caps -> [B, N, N]
        cls_features = features[:, 0, :]  # Agafar el token [CLS]
        all_features_kfold.append(cls_features.cpu())
all_features_kfold = torch.cat(all_features_kfold, dim=0)
print("Shape de features (KFold):", all_features_kfold.shape)

# Construir graf amb kNN per a X_no_hosp
features_np = all_features_kfold.numpy()
edge_index_np = kneighbors_graph(features_np, n_neighbors=5, mode='connectivity').nonzero()
edge_index_kfold = torch.tensor(edge_index_np, dtype=torch.long)
data_kfold = Data(x=all_features_kfold, edge_index=edge_index_kfold, y=torch.tensor(y_no_hosp, dtype=torch.long))

# -----------------------
# StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
indices = np.arange(len(y_no_hosp))

# Llistes per acumular les mètriques de cada fold
recall_benign_list = []
recall_malignant_list = []
precision_benign_list = []
precision_malignant_list = []
f1_benign_list = []
f1_malignant_list = []
auc_list = []

for train_idx, val_idx in skf.split(indices, y_no_hosp):
    # Crear màscares per als nodes de training i validació
    train_mask = torch.zeros(data_kfold.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data_kfold.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    # Inicialitzar un nou model GNN
    gnn_model = GNNClassifier(in_channels=all_features_kfold.shape[1], hidden_channels=64, out_channels=2)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Entrenament (per exemple, 200 epochs)
    for epoch in range(200):
        gnn_model.train()
        optimizer.zero_grad()
        out = gnn_model(data_kfold.x, data_kfold.edge_index)
        loss = loss_fn(out[train_mask], data_kfold.y[train_mask])
        loss.backward()
        optimizer.step()
    
    # Avaluació sobre els nodes de validació
    gnn_model.eval()
    with torch.no_grad():
        logits = gnn_model(data_kfold.x, data_kfold.edge_index)
        val_logits = logits[val_mask]
        val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()  # Probabilitat classe maligna
        val_preds = val_logits.argmax(dim=1).cpu().numpy()
        y_val = data_kfold.y[val_mask].cpu().numpy()
    
    # Càlcul de mètriques per fold
    recall_benign = recall_score(y_val, val_preds, pos_label=0)
    recall_malignant = recall_score(y_val, val_preds, pos_label=1)
    precision_benign = precision_score(y_val, val_preds, pos_label=0)
    precision_malignant = precision_score(y_val, val_preds, pos_label=1)
    f1_benign = f1_score(y_val, val_preds, pos_label=0)
    f1_malignant = f1_score(y_val, val_preds, pos_label=1)
    auc_fold = roc_auc_score(y_val, val_probs)
    
    recall_benign_list.append(recall_benign)
    recall_malignant_list.append(recall_malignant)
    precision_benign_list.append(precision_benign)
    precision_malignant_list.append(precision_malignant)
    f1_benign_list.append(f1_benign)
    f1_malignant_list.append(f1_malignant)
    auc_list.append(auc_fold)

# Mitjanar les mètriques a través dels folds
avg_recall_benign = np.mean(recall_benign_list)
avg_recall_malignant = np.mean(recall_malignant_list)
avg_precision_benign = np.mean(precision_benign_list)
avg_precision_malignant = np.mean(precision_malignant_list)
avg_f1_benign = np.mean(f1_benign_list)
avg_f1_malignant = np.mean(f1_malignant_list)
avg_auc = np.mean(auc_list)

print("MÈTRIQUES MITJANES KFold:")
print(f"Recall Benigne: {avg_recall_benign:.4f}")
print(f"Recall Maligne: {avg_recall_malignant:.4f}")
print(f"Precision Benigne: {avg_precision_benign:.4f}")
print(f"Precision Maligne: {avg_precision_malignant:.4f}")
print(f"F1-score Benigne: {avg_f1_benign:.4f}")
print(f"F1-score Maligne: {avg_f1_malignant:.4f}")
print(f"AUC: {avg_auc:.4f}")

# -----------------------
# HOLDOUT: Avaluació sobre X_hosp i y_hosp
# Extracció de features per al conjunt holdout
dataset_holdout = CustomDataset(X_hosp, y_hosp, transform=transform)
loader_holdout = DataLoader(dataset_holdout, batch_size=32, shuffle=False)
all_features_holdout = []
with torch.no_grad():
    for images, _ in loader_holdout:
        images = images.to("cpu")
        outputs = vit_model(images)
        attn = outputs.attentions[0]
        features = attn.mean(dim=1)
        cls_features = features[:, 0, :]
        all_features_holdout.append(cls_features.cpu())
all_features_holdout = torch.cat(all_features_holdout, dim=0)
print("Shape de features (Holdout):", all_features_holdout.shape)

# Construir graf per al conjunt holdout
features_holdout_np = all_features_holdout.numpy()
edge_index_holdout_np = kneighbors_graph(features_holdout_np, n_neighbors=5, mode='connectivity').nonzero()
edge_index_holdout = torch.tensor(edge_index_holdout_np, dtype=torch.long)
data_holdout = Data(x=all_features_holdout, edge_index=edge_index_holdout, y=torch.tensor(y_hosp, dtype=torch.long))

# Entrenar un model final sobre tot X_no_hosp (data_kfold)
final_model = GNNClassifier(in_channels=all_features_kfold.shape[1], hidden_channels=64, out_channels=2)
optimizer_final = torch.optim.Adam(final_model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(200):
    final_model.train()
    optimizer_final.zero_grad()
    out = final_model(data_kfold.x, data_kfold.edge_index)
    loss = loss_fn(out, data_kfold.y)
    loss.backward()
    optimizer_final.step()

# Avaluar el model final sobre el holdout
final_model.eval()
with torch.no_grad():
    logits_holdout = final_model(data_holdout.x, data_holdout.edge_index)
    holdout_probs = torch.softmax(logits_holdout, dim=1)[:, 1].cpu().numpy()
    holdout_preds = logits_holdout.argmax(dim=1).cpu().numpy()
    y_holdout = data_holdout.y.cpu().numpy()

recall_benign_h = recall_score(y_holdout, holdout_preds, pos_label=0)
recall_malignant_h = recall_score(y_holdout, holdout_preds, pos_label=1)
precision_benign_h = precision_score(y_holdout, holdout_preds, pos_label=0)
precision_malignant_h = precision_score(y_holdout, holdout_preds, pos_label=1)
f1_benign_h = f1_score(y_holdout, holdout_preds, pos_label=0)
f1_malignant_h = f1_score(y_holdout, holdout_preds, pos_label=1)
auc_holdout = roc_auc_score(y_holdout, holdout_probs)

print("\nMÈTRIQUES HOLDOUT:")
print(f"Recall Benigne: {recall_benign_h:.4f}")
print(f"Recall Maligne: {recall_malignant_h:.4f}")
print(f"Precision Benigne: {precision_benign_h:.4f}")
print(f"Precision Maligne: {precision_malignant_h:.4f}")
print(f"F1-score Benigne: {f1_benign_h:.4f}")
print(f"F1-score Maligne: {f1_malignant_h:.4f}")
print(f"AUC: {auc_holdout:.4f}")
