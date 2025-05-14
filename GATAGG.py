import os
import numpy as np
from random import shuffle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
import matplotlib.pyplot as plt

data = np.load('/fhome/pmarti/TFGPau/processedIms.npz', allow_pickle=True)
data1 = np.load('/fhome/pmarti/TFGPau/tissueDades.npz', allow_pickle=True)

f_no = data['features_list_single_no_h'] 
a_no = data['adj_list_single_no_h']

f_h = data['features_list_single_h'] 
a_h = data['adj_list_single_h']

y_hosp = data1['y_hosp']
y_no_hosp = data1['y_no_hosp']

def collate_fn(batch): return Batch.from_data_list(batch)

# ================= STEP 3: Model definition with aggregation + pooling + edge weights =================
class GATWithPool(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, heads=1, threshold=0.0):
        super(GATWithPool, self).__init__()
        
        # Agregació aprenent de les 12 matrius d’atenció → 1 matriu
        self.attn_agg = torch.nn.Conv2d(12, 1, kernel_size=1)

        # Mateixes capes GAT que al teu model original
        self.gat1 = GATConv(in_channels, hidden_dim, heads=heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=True, edge_dim=1)
        self.fc   = torch.nn.Linear(hidden_dim, out_channels)

        self.threshold = threshold

    def forward(self, data, attn_tensor):
        x, batch_idx = data.x, data.batch

        agg_mat = self.attn_agg(attn_tensor.unsqueeze(0)).squeeze(0).squeeze(0)

        mask = agg_mat > self.threshold
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        edge_attr  = agg_mat[mask]

        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch_idx)

        return F.log_softmax(self.fc(x), dim=1)  # sortida amb 2 classes

# ================= STEP 4: Training & Validation =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = torch.tensor(1.0 / torch.bincount(torch.tensor(y_no_hosp)), device=device)
weights = weights / weights.sum()
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
skf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
# prepare metrics and loss storage
metrics = {k:[] for k in ['recall_0','recall_1','precision_0','precision_1','f1_0','f1_1','auc']}
all_loss = []
best_auc, best_state = 0, None
val_loss_per_fold = []
roc_data_per_fold = []

for fold, (tr, va) in enumerate(skf.split(f_no, y_no_hosp, PatID_no_hosp),1):
    print(f"\n--- Fold {fold} ---")
    model = GATWithPool(768,256,2, threshold=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    td = [Data(x=f_no[i], attn=a_no[i], y=torch.tensor([y_no_hosp[i]])) for i in tr]
    vd = [Data(x=f_no[i], attn=a_no[i], y=torch.tensor([y_no_hosp[i]])) for i in va]
    tl = DataLoader(td, batch_size=1, shuffle=True, collate_fn=collate_fn)
    vl = DataLoader(vd, batch_size=1, shuffle=False, collate_fn=collate_fn)

    fold_losses = []
    v_loss = []
    # training
    for epoch in range(20):
        model.train()
        losses = []
        for b in tl:
            b = b.to(device)
            opt.zero_grad()
            logits = model(b, b.attn.to(device))
            loss = loss_fn(logits, b.y)
            loss.backward(); opt.step()
            losses.append(loss.item())
        epoch_loss = np.mean(losses)
        fold_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    all_loss.append(fold_losses)

    # validation
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for b in vl:
            b = b.to(device)
            logits, _ = model(b.x, b.attn.to(device), b.batch)
            loss = loss_fn(logits, b.y)
            v_loss.append(loss.item()) 
            probs = F.softmax(logits, dim=1)
            y_true.append(b.y.item())
            y_pred.append(probs.argmax(dim=1).item())
            y_scores.append(probs[0,1].item())

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    roc_data_per_fold.append((fpr, tpr, roc_auc))

    val_auc = roc_auc_score(y_true, y_scores)
    val_loss_per_fold.append(np.mean(v_loss))
    print(f"Fold {fold} AUC: {val_auc:.4f}")

    # record metrics
    metrics['recall_0'].append(recall_score(y_true, y_pred, pos_label=0))
    metrics['recall_1'].append(recall_score(y_true, y_pred, pos_label=1))
    metrics['precision_0'].append(precision_score(y_true, y_pred, pos_label=0))
    metrics['precision_1'].append(precision_score(y_true, y_pred, pos_label=1))
    metrics['f1_0'].append(f1_score(y_true, y_pred, pos_label=0))
    metrics['f1_1'].append(f1_score(y_true, y_pred, pos_label=1))
    metrics['auc'].append(val_auc)

    # update best
    if val_auc > best_auc:
        best_auc, best_state = val_auc, model.state_dict()

# print averaged metrics
print("\nAveraged Metrics over folds:")
for k, vals in metrics.items():
    mean, std = np.mean(vals), np.std(vals)
    print(f"{k}: {mean:.4f} ± {std:.4f}")

# ================= STEP 5: Test on Holdout =================
td = [Data(x=f_h[i], attn=a_h[i], y=torch.tensor([y_hosp[i]])) for i in range(len(y_hosp))]
tl = DataLoader(td, batch_size=1, shuffle=False, collate_fn=collate_fn)
model = GATWithPool(768,256,2, threshold=0.0).to(device)
model.load_state_dict(best_state); model.eval()
yt, yp, ys = [], [], []
with torch.no_grad():
    for b in tl:
        b = b.to(device)
        logits, _ = model(b.x, b.attn.to(device), b.batch)
        probs = F.softmax(logits, dim=1)
        yt.append(b.y.item())
        yp.append(probs.argmax(dim=1).item())
        ys.append(probs[0,1].item())

fpr_test, tpr_test, _ = roc_curve(yt, ys)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, label=f"Holdout ROC (AUC = {roc_auc_test:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.title("ROC Curve - Holdout Set")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/fhome/pmarti/TFGPau/RocCurves/ViTGAT_Agregació_ROC_Holdout.png", dpi=300)
plt.close()
# final holdout metrics
print("\nHoldout results:")
print(f"AUC: {roc_auc_score(yt, ys):.4f}")
print(f"Recall 0: {recall_score(yt, yp, pos_label=0):.4f}")
print(f"Recall 1: {recall_score(yt, yp, pos_label=1):.4f}")
print(f"Precision 0: {precision_score(yt, yp, pos_label=0):.4f}")
print(f"Precision 1: {precision_score(yt, yp, pos_label=1):.4f}")
print(f"F1 0: {f1_score(yt, yp, pos_label=0):.4f}")
print(f"F1 1: {f1_score(yt, yp, pos_label=1):.4f}")

# ================= STEP 6: Plot Loss Curve =================
plt.figure(figsize=(10,6))
for i, losses in enumerate(all_loss):
    plt.plot(losses, label=f"Fold {i+1}")
plt.title("Training Loss per Fold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/GATAGG_loss(Entreno).png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for i, (fpr, tpr, roc_auc) in enumerate(roc_data_per_fold):
    plt.plot(fpr, tpr, label=f"Fold {i+1} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.title("ROC Curves per Fold - ResNet+MLP")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/fhome/pmarti/TFGPau/RocCurves/ViTGAT_Agregació_ROC_PerFold.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 4))

plt.plot(
    range(1, len(val_loss_per_fold) + 1),   # X = folds 1,2,3…
    val_loss_per_fold,                      # Y = les losses
    'o-',                                   # punts amb línia
    label='Val Loss per Fold'
)

plt.xlabel('Fold')
plt.ylabel('Val Loss')
plt.title('Validation Loss per Fold')
plt.xticks(range(1, len(val_loss_per_fold) + 1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/fhome/pmarti/TFGPau/GATAGG_loss(validacio).png', dpi=300)
plt.close()
