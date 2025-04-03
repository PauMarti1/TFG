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


# ================= STEP 1: Load ViT (DeiT) model =================
deit = ViTModel.from_pretrained("google/vit-base-patch16-224").eval()
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Function to process dataset
def process_images(image_list):
    features_list, adj_list = [], []
    for image in tqdm(image_list, desc="Processing images"):
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image = Image.open(image).convert("RGB")
        
        inputs = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = deit(**inputs, output_attentions=True)
            features = outputs.last_hidden_state.squeeze(0)  # Shape: (197, 768)
            attentions = torch.stack(outputs.attentions).mean(dim=0).squeeze(0)  # Shape: (12, 197, 197)
        
        attn_matrix = attentions.mean(dim=0).cpu().numpy()
        adj_matrix = kneighbors_graph(attn_matrix, 8, mode='connectivity', include_self=True).toarray()
        edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
        
        features_list.append(features)
        adj_list.append(edge_index)
    
    return features_list, adj_list

def collate_fn(batch):
    return Batch.from_data_list(batch)

features_list, adj_list = process_images(X_no_hosp)

# ================= STEP 2: Define GAT Model =================
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

# Stratified K-Fold Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": []}



gat = GAT(in_channels=768, hidden_channels=256, out_channels=2)  # Binary classification

for train_idx, val_idx in skf.split(features_list, y_no_hosp):
    train_data = [Data(x=features_list[i], edge_index=adj_list[i], y=torch.tensor([y_no_hosp[i]])) for i in train_idx]
    val_data = [Data(x=features_list[i], edge_index=adj_list[i], y=torch.tensor([y_no_hosp[i]])) for i in val_idx]
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    
    optimizer = torch.optim.Adam(gat.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Training
    for epoch in range(10):
        for batch in train_loader:
            optimizer.zero_grad()
            output = gat(batch.x, batch.edge_index)
            loss = loss_fn(output.mean(dim=0, keepdim=True), batch.y)
            loss.backward()
            optimizer.step()
    
    # Validation
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            output = gat(batch.x, batch.edge_index)
            probs = F.softmax(output.mean(dim=0), dim=0)
            pred = torch.argmax(probs).item()
            
            y_true.append(batch.y.item())
            y_pred.append(pred)
            y_scores.append(probs[1].item())
    
    # Compute Metrics
    metrics["recall_0"].append(recall_score(y_true, y_pred, pos_label=0))
    metrics["recall_1"].append(recall_score(y_true, y_pred, pos_label=1))
    metrics["precision_0"].append(precision_score(y_true, y_pred, pos_label=0))
    metrics["precision_1"].append(precision_score(y_true, y_pred, pos_label=1))
    metrics["f1_0"].append(f1_score(y_true, y_pred, pos_label=0))
    metrics["f1_1"].append(f1_score(y_true, y_pred, pos_label=1))
    metrics["auc"].append(roc_auc_score(y_true, y_scores))

# Print averaged metrics
print("Averaged Metrics:")
for key, values in metrics.items():
    print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            output = gat(batch.x, batch.edge_index)
            probs = F.softmax(output.mean(dim=0), dim=0)
            pred = torch.argmax(probs).item()
            
            y_true.append(batch.y.item())  # Guarda tots els valors del batch
            y_pred.append(pred)  # pred ha de ser una llista o un tensor
            y_scores.append(probs[1].item())  # Guarda tots els scores del batch
    
    return np.array(y_true), np.array(y_pred), np.array(y_scores)

features_list, adj_list = process_images(X_hosp)

test_data = [Data(x=features_list[i], edge_index=adj_list[i], y=torch.tensor([y_hosp[i]])) for i in range(len(y_hosp))]
test_loader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=collate_fn)


y_true, y_pred, y_scores = evaluate_model(gat, test_loader, "cpu")

# Calcular mètriques
auc_scores = roc_auc_score(y_true, y_scores)
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