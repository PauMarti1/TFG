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

# -----------------------
# CONFIGURACIONS I CARREGADA DE DADES ORIGINALS
DataDir = r'/Users/paumarti/Desktop/TFG/tissue_images'
os.chdir(DataDir)

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

for k in np.arange(len(masks)):
    bck = masks[k] > 0
    masks[k][np.nonzero(masks[k] == 1)] = 2
    masks[k][np.nonzero(masks[k] != 2)] = 1
    masks[k] = masks[k] * bck.astype(int)

def ImPatches(im, sze, stride):
    szex, szey = sze
    im_patch = []
    coords = []
    for x in range(0, im.shape[0], stride):
        for y in range(0, im.shape[1], stride):
            patch = im[x:x+szex, y:y+szey]
            if min(patch.shape[0:2]) >= min(sze):
                im_patch.append(patch)
                coords.append((x, y))
    if len(im_patch) > 0:
        im_patch = np.stack(im_patch)
    return im_patch, coords

sze = [224, 224]
stride = 224
im_patches = []
PatID_patches = []
fileID_patches = []
tissue = []
patho = []
CenterID_patches = []
coords_patches = []
percent_infiltration = []
im_masks = []

for k in np.arange(len(masks)):
    im_mask = masks[k]
    im_img = ims[k]
    if min(im_mask.shape) > sze[0]:
        mask_patches, coords = ImPatches(im_mask, sze, stride)
        tissue_patch = np.sum(mask_patches.reshape(mask_patches.shape[0], -1) > 0, axis=1)
        pathotissue_patch = np.sum(mask_patches.reshape(mask_patches.shape[0], -1) == 2, axis=1)
        percent_inf = pathotissue_patch / np.maximum(tissue_patch, 1)

        tissue.append(tissue_patch)
        patho.append(pathotissue_patch)
        percent_infiltration.append(percent_inf)
        coords_patches.append(coords)

        img_patches, _ = ImPatches(im_img, sze, stride)
        img_masks, _ = ImPatches(im_mask, sze, stride)
        im_patches.append(img_patches)
        im_masks.append(img_masks)
        PatID_patches.append(np.repeat(PatID[k], img_patches.shape[0]))
        fileID_patches.append(np.repeat(fileID[k], img_patches.shape[0]))
        CenterID_patches.append(np.repeat(CenterID[k], img_patches.shape[0]))

im_patches = np.concatenate(im_patches, axis=0)
im_masks = np.concatenate(im_masks, axis=0)
tissue = np.concatenate(tissue, axis=0)
patho = np.concatenate(patho, axis=0)
percent_infiltration = np.concatenate(percent_infiltration, axis=0)
coords_patches = np.concatenate(coords_patches, axis=0)
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
im_masks = im_masks[id_tissue]
percent_infiltration = percent_infiltration[id_tissue]
coords_patches = coords_patches[id_tissue]
CenterID_patches = CenterID_patches[id_tissue]
PatID_patches = PatID_patches[id_tissue]

X = X[idxsel, :]
im_masks = im_masks[idxsel]
percent_infiltration = percent_infiltration[idxsel]
coords_patches = coords_patches[idxsel]
CenterID_patches = CenterID_patches[idxsel]
PatID_patches = PatID_patches[idxsel]

# X = X.astype(float)
# mu = [np.mean(X[:, :, :, ch].flatten()) for ch in range(3)]
# std = [np.std(X[:, :, :, ch].flatten()) for ch in range(3)]
# for kch in range(3):
#     X[:, :, :, kch] = (X[:, :, :, kch] - mu[kch]) / std[kch]

def split_test_hospital(X, y_true, CenterID_patches, PatID_patches, coords_patches, percent_infiltration, im_masks):
    hospital_counts = Counter(CenterID_patches)
    least_common_hospital = hospital_counts.most_common()[-1][0]
    
    test_idx = np.where(CenterID_patches == least_common_hospital)[0]
    train_idx = np.where(CenterID_patches != least_common_hospital)[0]

    X_train, y_train = X[train_idx], y_true[train_idx]
    X_test, y_test = X[test_idx], y_true[test_idx]

    PatID_train, PatID_test = PatID_patches[train_idx], PatID_patches[test_idx]
    coords_train, coords_test = coords_patches[train_idx], coords_patches[test_idx]
    infil_train, infil_test = percent_infiltration[train_idx], percent_infiltration[test_idx]
    mask_train, mask_test = im_masks[train_idx], im_masks[test_idx]

    return X_train, y_train, PatID_train, coords_train, infil_train, \
           X_test, y_test, PatID_test, coords_test, infil_test, \
           mask_train, mask_test, least_common_hospital

X_no_hosp, y_no_hosp, PatID_no_hosp, coords_no_hosp, infil_no_hosp, \
X_hosp, y_hosp, PatID_hosp, coords_hosp, infil_hosp, \
x_no_hosp_mask, x_hosp_mask, least = split_test_hospital(
    X, y_true, CenterID_patches, PatID_patches, coords_patches, percent_infiltration, im_masks
)

np.savez('tissueDades.npz',
         X_no_hosp=X_no_hosp,
         y_no_hosp=y_no_hosp,
         PatID_no_hosp=PatID_no_hosp,
         coords_no_hosp=coords_no_hosp,
         infil_no_hosp=infil_no_hosp,
         X_hosp=X_hosp,
         y_hosp=y_hosp,
         PatID_hosp=PatID_hosp,
         coords_hosp=coords_hosp,
         infil_hosp=infil_hosp,
         x_no_hosp_mask=x_no_hosp_mask,
         x_hosp_mask=x_hosp_mask,
         least=least)