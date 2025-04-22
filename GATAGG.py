import os
import numpy as np
import glob
import cv2 as cv
from ismember import ismember
from random import shuffle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
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

def process_images(image_list):
    features_list, attn_list = [], []
    for image in tqdm(image_list, desc="Processing images"):
        # Load image
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image = Image.open(image).convert("RGB")

        # ViT preprocessing and forward
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = deit(**inputs, output_attentions=True)
            # Extract patch features: (1, 197, 768) -> (197, 768)
            features = outputs.last_hidden_state.squeeze(0)
            # Build per-layer attention: list of 12 tensors (197,197)
            per_layer = [att.mean(dim=1).squeeze(0) for att in outputs.attentions]
            attn_tensor = torch.stack(per_layer)  # (12, 197, 197)

        features_list.append(features)
        attn_list.append(attn_tensor)

    return features_list, attn_list

features_list, attn_list = process_images(X_no_hosp)

# Collate that preserves both x and attn
from torch_geometric.data import Batch
def collate_fn(batch):
    return Batch.from_data_list(batch)

# ================= STEP 2: Define GCN with Learned Aggregator =================
class GCNWithAgg(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.agg = torch.nn.Conv2d(12, 1, kernel_size=1)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, attn_tensor):
        agg_mat = self.agg(attn_tensor).squeeze(1)
        thresh = torch.quantile(agg_mat.flatten(1), 0.9, dim=1)
        mask = agg_mat[0] >= thresh[0]
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()

        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

# Stratified K-Fold Validation
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {m: [] for m in ["recall_0","recall_1","precision_0","precision_1","f1_0","f1_1","auc"]}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classCount = torch.bincount(torch.tensor(y_no_hosp))
classWeights = 1.0 / classCount.float()
classWeights = (classWeights / classWeights.sum()).to(device)

best_val_auc = 0
best_model_state = None
all_loss = []

for fold, (train_idx, val_idx) in enumerate(skf.split(features_list, y_no_hosp, groups=PatID_no_hosp)):
    print(f"\n--- Fold {fold+1} ---")
    gcn = GCNWithAgg(in_channels=768, hidden_channels=256, out_channels=2).to(device)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss(weight=classWeights)

    train_data = [Data(x=features_list[i], attn=attn_list[i], y=torch.tensor([y_no_hosp[i]])) for i in train_idx]
    val_data   = [Data(x=features_list[i], attn=attn_list[i], y=torch.tensor([y_no_hosp[i]])) for i in val_idx]

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_data,   batch_size=1, shuffle=False, collate_fn=collate_fn)

    fold_losses = []
    for epoch in range(25):
        gcn.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = gcn(batch.x, batch.attn.to(device))
            # graph-level prediction via mean
            logits = out.mean(dim=0, keepdim=True)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_loss = np.mean(losses)
        fold_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    all_loss.append(fold_losses)
    
    # Validation
    gcn.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = gcn(batch.x, batch.attn.to(device))
            probs = F.softmax(out.mean(dim=0), dim=0)
            pred = probs.argmax().item()
            y_true.append(batch.y.item())
            y_pred.append(pred)
            y_scores.append(probs[1].item())

    val_auc = roc_auc_score(y_true, y_scores)
    print(f"Fold {fold+1} AUC: {val_auc:.4f}")
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = gcn.state_dict()

    metrics['recall_0'].append(recall_score(y_true, y_pred, pos_label=0))
    metrics['recall_1'].append(recall_score(y_true, y_pred, pos_label=1))
    metrics['precision_0'].append(precision_score(y_true, y_pred, pos_label=0))
    metrics['precision_1'].append(precision_score(y_true, y_pred, pos_label=1))
    metrics['f1_0'].append(f1_score(y_true, y_pred, pos_label=0))
    metrics['f1_1'].append(f1_score(y_true, y_pred, pos_label=1))
    metrics['auc'].append(val_auc)

# Print averaged metrics
print("Averaged Metrics:")
for key, vals in metrics.items():
    print(f"{key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# Plot training losses
plt.figure(figsize=(10,6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/GATAGG.png', dpi=300)
plt.close()

# ================= STEP 3: Evaluate on Holdout =================
features_list_h, attn_list_h = process_images(X_hosp)

test_data = [Data(x=features_list_h[i], attn=attn_list_h[i], y=torch.tensor([y_hosp[i]]))
             for i in range(len(y_hosp))]

test_loader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=collate_fn)
gcn = GCNWithAgg(in_channels=768, hidden_channels=256, out_channels=2).to(device)
gcn.load_state_dict(best_model_state)
gcn.eval()

y_true, y_pred, y_scores = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = gcn(batch.x, batch.attn.to(device))
        probs = F.softmax(out.mean(dim=0), dim=0)
        pred = probs.argmax().item()
        y_true.append(batch.y.item())
        y_pred.append(pred)
        y_scores.append(probs[1].item())

# Calcular mètriques
auc_scores     = roc_auc_score(y_true, y_scores)
recall_benigne = recall_score(y_true, y_pred, pos_label=0)
recall_maligne = recall_score(y_true, y_pred, pos_label=1)
precision_ben  = precision_score(y_true, y_pred, pos_label=0)
precision_mal  = precision_score(y_true, y_pred, pos_label=1)
f1_ben         = f1_score(y_true, y_pred, pos_label=0)
f1_mal         = f1_score(y_true, y_pred, pos_label=1)

print('Holdout results:')
print(f"AUC: {auc_scores:.4f}")
print(f"Recall Benigne: {recall_benigne:.4f}")
print(f"Recall Maligne: {recall_maligne:.4f}")
print(f"Precision Benigne: {precision_ben:.4f}")
print(f"Precision Maligne: {precision_mal:.4f}")
print(f"F1-score Benigne: {f1_ben:.4f}")
print(f"F1-score Maligne: {f1_mal:.4f}")
