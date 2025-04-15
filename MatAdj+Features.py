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


# -----------------------
# CONFIGURACIONS I CARREGADA DE DADES ORIGINALS
# Carregar el fitxer .npz
data = np.load('/fhome/pmarti/TFGPau/tissueDades.npz', allow_pickle=True)

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
        
        edge_index = torch.tensor(np.array(np.nonzero(attn_matrix)), dtype=torch.long)
        
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
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {"recall_0": [], "recall_1": [], "precision_0": [], "precision_1": [], "f1_0": [], "f1_1": [], "auc": []}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_loss = []

classCount = torch.bincount(torch.tensor(y_no_hosp))
classWeights = 1.0 / classCount.float()
classWeights = classWeights / classWeights.sum()
classWeights = classWeights.to(device)

best_val_auc = 0
best_model_state = None
all_loss = []

for fold, (train_idx, val_idx) in enumerate(skf.split(features_list, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n--- Fold {fold+1} ---")
    
    gat = GAT(in_channels=768, hidden_channels=256, out_channels=2).to(device)
    optimizer = torch.optim.Adam(gat.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss(weight=classWeights)

    train_data = [Data(x=features_list[i], edge_index=adj_list[i], y=torch.tensor([y_no_hosp[i]])) for i in train_idx]
    val_data = [Data(x=features_list[i], edge_index=adj_list[i], y=torch.tensor([y_no_hosp[i]])) for i in val_idx]
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    t_loss = []

    for epoch in range(25):
        gat.train()
        e_loss = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = gat(batch.x, batch.edge_index)
            loss = loss_fn(output.mean(dim=0, keepdim=True), batch.y.to(device))
            loss.backward()
            optimizer.step()
            e_loss.append(loss.item())
        t_loss.append(np.mean(e_loss))
        print(f"Epoch {epoch+1}, Loss: {t_loss[-1]:.4f}")

    all_loss.append(t_loss)

    # Validate
    gat.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = gat(batch.x, batch.edge_index)
            probs = F.softmax(output.mean(dim=0), dim=0)
            pred = torch.argmax(probs).item()
            
            y_true.append(batch.y.item())
            y_pred.append(pred)
            y_scores.append(probs[1].item())

    val_auc = roc_auc_score(y_true, y_scores)
    print(f"Fold {fold+1} AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = gat.state_dict()

    # Guardem les mètriques per cada fold
    metrics["recall_0"].append(recall_score(y_true, y_pred, pos_label=0))
    metrics["recall_1"].append(recall_score(y_true, y_pred, pos_label=1))
    metrics["precision_0"].append(precision_score(y_true, y_pred, pos_label=0))
    metrics["precision_1"].append(precision_score(y_true, y_pred, pos_label=1))
    metrics["f1_0"].append(f1_score(y_true, y_pred, pos_label=0))
    metrics["f1_1"].append(f1_score(y_true, y_pred, pos_label=1))
    metrics["auc"].append(val_auc)


# Print averaged metrics
print("Averaged Metrics:")
for key, values in metrics.items():
    print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = gat(batch.x, batch.edge_index)
            probs = F.softmax(output.mean(dim=0), dim=0)
            pred = torch.argmax(probs).item()
            
            y_true.append(batch.y.item())  # Guarda tots els valors del batch
            y_pred.append(pred)  # pred ha de ser una llista o un tensor
            y_scores.append(probs[1].item())  # Guarda tots els scores del batch
    
    return np.array(y_true), np.array(y_pred), np.array(y_scores)

plt.figure(figsize=(10, 6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/Vit+GAT.png', dpi=300)
plt.close()

features_list, adj_list = process_images(X_hosp)

test_data = [Data(x=features_list[i], edge_index=adj_list[i], y=torch.tensor([y_hosp[i]])) for i in range(len(y_hosp))]
test_loader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=collate_fn)


gat = GAT(in_channels=768, hidden_channels=256, out_channels=2).to(device)
gat.load_state_dict(best_model_state)
gat.eval()

y_true, y_pred, y_scores = evaluate_model(gat, test_loader, device)

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