import sys
import os
import numpy as np
import pandas as pd
import glob
import cv2 as cv
from matplotlib import pyplot as plt
from ismember import ismember
from random import shuffle
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from timm import create_model
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
from collections import Counter
from random import choice
from sklearn.model_selection import StratifiedKFold
from timm.models.swin_transformer import SwinTransformer

DataDir=r'/Users/paumarti/Desktop/TFG/tissue_images'
os.chdir(DataDir)

# 1. LOAD DATA
files=glob.glob('*_mask.png')
files=np.array(files)
#Remove cases from Manresa due to poor quali
files=files[np.array([file.find('Manresa') for file in files])==-1]
Centers=['Terrassa','Alicante','Manresa','Basurto','Bellvitge']
filesCenter={}

for Center in Centers:
    filesCenter[Center]=np.array([file.find(Center) for file in files])==-1


PatID=[]
ims=[]
masks=[]
fileID=[]
CenterID=[]
for k in np.arange(len(files)):
    
    file=files[k]
    im = cv.imread(file.split('_mask')[0]+'.png')[:,:,0:3]
    
    if min(im.shape[0:2])>=128:
            
        ims.append(im)
        
        im = np.mean(cv.imread(file)[:,:,0:3],axis=2)/255
        masks.append(im)
        
        PatID.append(file.split('_SampID')[0])
        fileID.append(file.split('_mask')[0])
        CenterID.append(file.split('_')[0])
    
    
### 2. PREPROCESSING

## 2.0 Mask labels: 2 is infiltration, 1 healthy, 0 background
for k in np.arange(len(masks)):
    bck=masks[k]>0
    masks[k][np.nonzero(masks[k]==1)]=2
    masks[k][np.nonzero(masks[k]!=2)]=1
    masks[k]=masks[k]*bck.astype(int)

def ImPatches(im,sze,stride):
    """
    Extreu patches de la imatge
        @im: imatge
        @sze: size del patch
        @stride: stride del patch
    """
    szex=sze[0]
    szey=sze[1]
    im_patch=[im[x:x+szex,y:y+szey] 
              for x in range(0,im.shape[0],stride) for y in 
              range(0,im.shape[1],stride)]
    valid_patch=[]
    for k in range(len(im_patch)):
        patch=im_patch[k]
        if min(patch.shape[0:2])>=min(sze):
            valid_patch.append(patch)
    if len(valid_patch)>0:       
        im_patch=np.stack(valid_patch)
    
    return im_patch

sze=[224, 224]
stride=224
im_patches=[]
PatID_patches=[]
fileID_patches=[]
tissue=[]
patho=[]
CenterID_patches=[]

for k in np.arange(len(masks)):
    # Area of Tissue Type
    im=masks[k]
    if min(im.shape)>sze[0]:
        patches=ImPatches(im,sze,stride)
        pathotissue_patch=np.sum(patches.reshape(patches.shape[0],np.prod(sze))==2,axis=1) # Num de pixels en la imatge que son patològics
        tissue_patch=np.sum(patches.reshape(patches.shape[0],np.prod(sze))>0,axis=1) # Num de pixels en la imatge que son teixit (no background)
        tissue.append(tissue_patch)
        patho.append(pathotissue_patch)
        # Im Patches
        im=ims[k]
        patches=ImPatches(im,sze,stride)
        im_patches.append(patches)
        # PatID Patches
        PatID_patches.append(np.repeat(PatID[k],patches.shape[0]))
        fileID_patches.append(np.repeat(fileID[k],patches.shape[0]))
        CenterID_patches.append(np.repeat(CenterID[k],patches.shape[0]))

# Pas a numpy de totes les llistes
im_patches=np.concatenate(im_patches,axis=0)
tissue=np.concatenate(tissue,axis=0)
patho=np.concatenate(patho,axis=0)
PatID_patches=np.concatenate(PatID_patches,axis=0)
CenterID_patches= np.concatenate(CenterID_patches,axis=0)
fileID_patches= np.concatenate(fileID_patches,axis=0)

th_percen_tissue=0.75 # Threshold de teixit per considerar la imatge
id_tissue=np.nonzero(tissue/np.prod(sze)>th_percen_tissue)[0] # Imatges amb teixit suficient

y_true=patho[id_tissue]/tissue[id_tissue] # Proporció de pixels patològics
# Manual Data Balancing
idxpatho=np.nonzero(y_true==1)[0] # Index de les imatges amb tot patològic
idxother=np.nonzero(y_true<1)[0] # Index de les imatges amb teixit mixt
idxsel=np.concatenate((idxpatho[0::5],idxother)) # Selecció random de les imatges per evitar bias
shuffle(idxsel)

y_true=y_true[idxsel]

y_true=(y_true>0.5).astype(int) # Converteix a binari
# y_true=np.concatenate((np.expand_dims(tissue,axis=1),np.expand_dims(patho,axis=1)),axis=1)
# y_true=y_true[id_tissue,:]
X=im_patches[id_tissue,:]
PatID_patches=PatID_patches[id_tissue]
# fileID_patches=fileID_patches[id_tissue]
CenterID_patches=CenterID_patches[id_tissue]

X=X[idxsel,:]
PatID_patches=PatID_patches[idxsel]
# fileID_patches=fileID_patches[idxsel]
CenterID_patches=CenterID_patches[idxsel]

X=X.astype(float)
# Patch Intensity Normalization. OVER ALL SET!!!!
mu=[np.mean(X[:,:,:,0].flatten()),np.mean(X[:,:,:,1].flatten()),np.mean(X[:,:,:,2].flatten())]
std=[np.std(X[:,:,:,0].flatten()),np.std(X[:,:,:,1].flatten()),np.std(X[:,:,:,2].flatten())]
for kch in np.arange(3):
    X[:,:,:,kch]=(X[:,:,:,kch]-mu[kch])/std[kch]

def split_test_hospital(X, y_true, CenterID_patches):
    # Comptar quants elements hi ha de cada hospital
    hospital_counts = Counter(CenterID_patches)
    
    # Trobar l'hospital que apareix menys vegades
    least_common_hospital = hospital_counts.most_common()[-1][0]

    # Filtrar els fitxers per deixar fora els de l'hospital seleccionat
    test_idx = np.where(CenterID_patches == least_common_hospital)[0]
    train_idx = np.where(CenterID_patches != least_common_hospital)[0]
    
    # Crear els conjunts d'entrenament i test
    X_train, y_train = X[train_idx], y_true[train_idx]
    X_test, y_test = X[test_idx], y_true[test_idx]
    
    return X_train, y_train, X_test, y_test, least_common_hospital

X_no_hosp, y_no_hosp, X_hosp, y_hosp, least = split_test_hospital(X, y_true, CenterID_patches)

# L'hospital amb menys mostres és el de Terrassa. Farem servir aquest per al holdout.

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

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
        image = Image.fromarray(image.astype('uint8'))  # Convert numpy array to PIL image
        if self.transform:
            image = self.transform(image)
        return image, label

classCount = torch.bincount(torch.tensor(y_no_hosp))
classWeights = 1.0 / classCount.float()
classWeights = classWeights / classWeights.sum()
classWeights = classWeights.to("cpu")

def build_ctranspath():
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=0,  # No final classification layer
        embed_dim=96,   # Key difference from your attempted version
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2
    )
    return model

def load_ctranspath(model, weights_path):
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # Handle different weight formats
    if 'model' in state_dict:  # Common in trained checkpoints
        state_dict = state_dict['model']
    
    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Filter only matching keys
    model_state_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }
    
    # Load compatible weights
    model.load_state_dict(filtered_state_dict, strict=False)
    
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_ctranspath()
model = load_ctranspath(model, '/Users/paumarti/Desktop/TFG/ctranspath.pth').to(device)

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x):
        return x.permute(*self.dims)

model.head = nn.Sequential(
        nn.LayerNorm(model.num_features),  # Normalize features
        Permute((0, 3, 1, 2)),  # Change from [B,H,W,C] to [B,C,H,W]
        nn.AdaptiveAvgPool2d(1),  # Pool to [B,C,1,1]
        nn.Flatten(),  # Convert to [B,C]
        nn.Linear(model.num_features, 2)  # Classificacio
)


print(model)
for param in model.parameters():
    param.requires_grad = False
model.head.requires_grad_(True)

criterion = nn.CrossEntropyLoss(weight=classWeights)

auc_scores = []
recall_benigne_scores = []
recall_maligne_scores = []
precision_benigne_scores = []
precision_maligne_scores = []
f1_benigne_scores = []
f1_maligne_scores = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_dataset = CustomDataset(X_hosp, y_hosp, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(X_no_hosp)), y_no_hosp)):
    print(f"Fold {fold + 1}/{5}")

    train_dataset = CustomDataset(X_no_hosp[train_idx], y_no_hosp[train_idx], transform=transform)
    val_dataset = CustomDataset(X_no_hosp[val_idx], y_no_hosp[val_idx], transform=transform)

    # train_subset = Subset(X_no_hosp, train_idx)
    # val_subset = Subset(X_no_hosp, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Optimizer
    optimizer = optim.Adam(model.head.parameters(), lr=1e-3)
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to("cpu"), labels.to("cpu")

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to("cpu"), labels.to("cpu")
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilitat de classe maligna
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
            # Mètriques
            auc = roc_auc_score(all_labels, all_probs)
            recall_benigne = recall_score(all_labels, all_preds, pos_label=0)
            recall_maligne = recall_score(all_labels, all_preds, pos_label=1)
            precision_benigne = precision_score(all_labels, all_preds, pos_label=0)
            precision_maligne = precision_score(all_labels, all_preds, pos_label=1)
            f1_benigne = f1_score(all_labels, all_preds, pos_label=0)
            f1_maligne = f1_score(all_labels, all_preds, pos_label=1)
            
            # Guardar mètriques
            auc_scores.append(auc)
            recall_benigne_scores.append(recall_benigne)
            recall_maligne_scores.append(recall_maligne)
            precision_benigne_scores.append(precision_benigne)
            precision_maligne_scores.append(precision_maligne)
            f1_benigne_scores.append(f1_benigne)
            f1_maligne_scores.append(f1_maligne)

            print(f"Fold {fold + 1} - AUC: {auc:.4f}")
            print(f"Recall Benigne: {recall_benigne:.4f}, Recall Maligne: {recall_maligne:.4f}")
            print(f"Precision Benigne: {precision_benigne:.4f}, Precision Maligne: {precision_maligne:.4f}")
            print(f"F1-score Benigne: {f1_benigne:.4f}, F1-score Maligne: {f1_maligne:.4f}")

print("\nFinal Results:")
print(f"AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"Recall Benigne: {np.mean(recall_benigne_scores):.4f} ± {np.std(recall_benigne_scores):.4f}")
print(f"Recall Maligne: {np.mean(recall_maligne_scores):.4f} ± {np.std(recall_maligne_scores):.4f}")
print(f"Precision Benigne: {np.mean(precision_benigne_scores):.4f} ± {np.std(precision_benigne_scores):.4f}")
print(f"Precision Maligne: {np.mean(precision_maligne_scores):.4f} ± {np.std(precision_maligne_scores):.4f}")
print(f"F1-score Benigne: {np.mean(f1_benigne_scores):.4f} ± {np.std(f1_benigne_scores):.4f}")
print(f"F1-score Maligne: {np.mean(f1_maligne_scores):.4f} ± {np.std(f1_maligne_scores):.4f}")

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilitat de classe 1
            preds = torch.argmax(outputs, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred), np.array(y_scores)

y_true, y_pred, y_scores = evaluate_model(model, test_loader, "cpu")

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