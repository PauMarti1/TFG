import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, ViTModel
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from scipy.ndimage import zoom

# ==================== CONFIG ====================
DATA_PATH   = '/Users/paumarti/Desktop/TFG/tissueDades.npz'
OUTPUT_DIR  = './attention_triplets'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== LOAD DATA ====================
data   = np.load(DATA_PATH, allow_pickle=True)
X      = data['X_hosp']        # pot ser floats normalitzats en [-0.02,0.01]
PatID  = data['PatID_hosp']
y      = data['y_hosp']

# ==================== ViT MODEL ====================
processor = AutoImageProcessor.from_pretrained("facebook/deit-base-patch16-224")
vit       = ViTModel.from_pretrained("facebook/deit-base-patch16-224").eval().to(DEVICE)

# ==================== GAT MODEL ====================
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_dim, heads=1, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=True, edge_dim=1)
        self.fc   = torch.nn.Linear(hidden_dim, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = self.gat2(x, edge_index, edge_attr)
        return self.fc(x)

gat = GAT(in_channels=768, hidden_dim=256, out_channels=2).to(DEVICE)
gat.load_state_dict(torch.load('/Users/paumarti/Desktop/TFG/BestViTGAT.pth', map_location=DEVICE))
gat.eval()

# ========== UTILS ==========

def extract_vit_attention(img_uint8):
    inputs = processor(images=img_uint8, return_tensors="pt").to(DEVICE)
    outputs = vit(**inputs, output_attentions=True)
    attn    = outputs.attentions[-1]              # (1, heads, tokens, tokens)
    cls_attn= attn[0, :, 0, 1:].mean(0)           # (tokens-1,)
    att_map = cls_attn.reshape(14, 14).detach().numpy()
    return zoom(att_map, 224/14)

def extract_gat_attention(patch_tensor):
    num_nodes  = patch_tensor.size(0)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).T.to(DEVICE)
    edge_attr  = torch.ones(edge_index.size(1), 1, device=DEVICE)
    data       = Data(x=patch_tensor.to(DEVICE),
                      edge_index=edge_index,
                      edge_attr=edge_attr)
    out        = gat(data)
    node_att   = out.softmax(dim=1)[:, 1]         # atenció classe 1
    att_map    = node_att.detach().numpy().reshape(14, 14)
    return zoom(att_map, 224/14)

def prepare_display(orig_arr):
    """
    Si orig_arr és float, li fem un dynamic range stretch a [0,255].
    Retorna uint8.
    """
    if orig_arr.dtype == np.uint8:
        return orig_arr
    # Range stretch
    arr = orig_arr.astype(np.float32)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255.0).astype(np.uint8)
    return arr

def plot_triplet(orig_uint8, vit_attn, gat_attn, filename, patid, label):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    title = f"PatID: {patid} | Label: {label}"
    fig.suptitle(title, fontsize=14)

    # Tractem grayscale vs color
    cmap = 'gray' if orig_uint8.ndim == 2 else None

    axs[0].imshow(orig_uint8, cmap=cmap)
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(orig_uint8, cmap=cmap)
    axs[1].imshow(vit_attn, cmap='jet', alpha=0.5)
    axs[1].set_title("ViT Attention")
    axs[1].axis('off')

    axs[2].imshow(orig_uint8, cmap=cmap)
    axs[2].imshow(gat_attn, cmap='hot', alpha=0.5)
    axs[2].set_title("GAT Attention")
    axs[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)

# ========== PROCESS ALL ==========
for i, img_array in enumerate(X):
    # 1) Preparem la imatge per visualitzar — ara sí que surt amb contrast!
    orig_disp = prepare_display(img_array)

    # 2) Fem extract_vit_attention passant orig_disp (uint8)
    vit_att = extract_vit_attention(orig_disp)

    # 3) Fem embed per GAT amb el mateix orig_disp
    inputs      = processor(images=orig_disp, return_tensors="pt").to(DEVICE)
    vit_output  = vit(**inputs)
    patch_emb   = vit_output.last_hidden_state[0, 1:]  # excloem CLS
    gat_att     = extract_gat_attention(patch_emb)

    # 4) Guardem la imatge
    plot_triplet(orig_disp, vit_att, gat_att, f"patch_{i}.png", PatID[i], y[i])