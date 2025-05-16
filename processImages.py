import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
import glob
import cv2 as cv
from ismember import ismember
from random import shuffle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
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

data = np.load('/Users/paumarti/Desktop/TFG/tissueDades.npz', allow_pickle=True)

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
deit = ViTModel.from_pretrained("google/vit-base-patch16-224").eval()
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

def process_images(image_list, mode="single"):
    """
    Processa una llista d'imatges i retorna:
        - features_list: llista de tensors (196, 768)
        - adj_list: llista d'edge_index (un per imatge si 'single', o 12 per imatge si 'ensemble')
        - edge_attr_list: llista de tensors amb els pesos de les arestes
        - attn_matrices: llista de matrius d'atenció (single: (196,196), ensemble: (12,196,196))

    mode: 'single' o 'ensemble'
    """
    assert mode in ["single", "ensemble"], "mode ha de ser 'single' o 'ensemble'"

    features_list, adj_list, edge_attr_list = [], [], []
    attn_matrices = []

    for image in tqdm(image_list, desc="Processing images"):
        # Conversió a PIL Image si cal
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image = Image.open(image).convert("RGB")

        # Preprocessament
        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = deit(**inputs, output_attentions=True)
            features = outputs.last_hidden_state.squeeze(0)[1:]  # (196, 768)

            # Atencions: (12, 1, heads, 197, 197) → mitjana sobre heads → (12, 197, 197)
            attentions = torch.stack(outputs.attentions)
            attentions = attentions.squeeze(1).mean(dim=1)

        if mode == "single":
            # Mitjana sobre capes, eliminant CLS → (196, 196)
            attn_matrix = attentions.mean(dim=0)[1:, 1:]
            attn_matrices.append(attn_matrix.cpu().numpy())

            attn_matrix = attn_matrix.cpu().numpy()

            edge_list, edge_weights = [], []

            for i in range(attn_matrix.shape[0]):
                for j in range(attn_matrix.shape[1]):
                    weight = float(attn_matrix[i, j])
                    edge_list.append([i, j])
                    edge_weights.append(weight)

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32)

            adj_list.append(edge_index)
            edge_attr_list.append(edge_attr)

        elif mode == "ensemble":
            # Eliminem CLS per cada capa → (12, 196, 196)
            attn_matrix = attentions[:, 1:, 1:]
            attn_matrices.append(attn_matrix.cpu().numpy())

            edge_indices_heads = []
            edge_weights_heads = []

            for layer_attn in attn_matrix:  # (196, 196)
                layer_attn_np = layer_attn.cpu().numpy()
                edge_list, edge_weights = [], []

                for i in range(layer_attn_np.shape[0]):
                    for j in range(layer_attn_np.shape[1]):
                        weight = float(layer_attn_np[i, j])
                        edge_list.append([i, j])
                        edge_weights.append(weight)

                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_weights, dtype=torch.float32)

                edge_indices_heads.append(edge_index)
                edge_weights_heads.append(edge_attr)

            adj_list.append(edge_indices_heads)
            edge_attr_list.append(edge_weights_heads)

        features_list.append(features)

    return features_list, adj_list, edge_attr_list, attn_matrices


features_list_single_no_h, adj_list_single_no_h, edge_attr_list__single_no_h, attn_matrices_no_hosp = process_images(X_no_hosp)

features_list_single_h, adj_list_single_h, edge_attr_list_single_h, attn_matrices_hosp = process_images(X_hosp)

features_list_ensemble_no_h, adj_list_ensemble_no_h, edge_attr_list_ensemble_no_h, _ = process_images(X_no_hosp, mode="ensemble")

features_list_ensemble_h, adj_list_ensemble_h, edge_attr_list_ensemble_h, _ = process_images(X_hosp, mode="ensemble")

np.savez(
    'processedIms.npz',
    features_list_single_no_h=features_list_single_no_h,
    adj_list_single_no_h=adj_list_single_no_h,
    edge_attr_list__single_no_h=edge_attr_list__single_no_h,
    features_list_single_h=features_list_single_h, 
    adj_list_single_h=adj_list_single_h, 
    edge_attr_list_single_h=edge_attr_list_single_h,
    features_list_ensemble_no_h=features_list_ensemble_no_h, 
    adj_list_ensemble_no_h=adj_list_ensemble_no_h, 
    edge_attr_list_ensemble_no_h=edge_attr_list_ensemble_no_h,
    features_list_ensemble_h=features_list_ensemble_h, 
    adj_list_ensemble_h=adj_list_ensemble_h, 
    edge_attr_list_ensemble_h=edge_attr_list_ensemble_h,
    attn_matrices_no_hosp=attn_matrices_no_hosp,
    attn_matrices_hosp=attn_matrices_hosp
)
