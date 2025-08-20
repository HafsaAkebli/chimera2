import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from pathlib import Path

# === DEFINE CONCATENATION-BASED MLP MODEL ===
class FusionMLP(nn.Module):
    def __init__(
        self,
        hist_dim=1024,
        clin_dim=34,
        proj_dim=128,
        fusion_hidden=[256,128, 64],
        num_classes=2,
        dropout=0.1,
        norm_type="batchnorm",  # "batchnorm" | "layernorm" | "none"
    ):
        super().__init__()

        def make_norm(d):
            if norm_type == "batchnorm":
                return nn.BatchNorm1d(d)
            elif norm_type == "layernorm":
                return nn.LayerNorm(d)
            else:
                return nn.Identity()

        # per-modality projectors
        self.hist_proj = nn.Sequential(
            nn.Linear(hist_dim, proj_dim),
            make_norm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.clin_proj = nn.Sequential(
            nn.Linear(clin_dim, proj_dim),
            make_norm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # fusion MLP with residuals
        dims = [proj_dim * 2] + fusion_hidden
        layers, norms = [], []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            norms.append(make_norm(dims[i + 1]))
        self.fuse_layers = nn.ModuleList(layers)
        self.fuse_norms  = nn.ModuleList(norms)

        self.classifier = nn.Linear(dims[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, hist_feat, clin_feat):
        h = self.hist_proj(hist_feat)
        c = self.clin_proj(clin_feat)
        x = torch.cat([h, c], dim=1)

        for layer, norm in zip(self.fuse_layers, self.fuse_norms):
            residual = x
            x = layer(x)
            x = norm(x)
            x = self.act(x)
            x = self.dropout(x)
            if x.shape == residual.shape:
                x = x + residual

        return self.classifier(x)  # [B,2]


def predict_probability(
    histology_embedding: np.ndarray,
    clinical_embedding: np.ndarray,
    model_path: Path
) -> float:
    """
    Predicts BRS3 probability using the trained FusionMLP model.

    Args:
        histology_embedding (np.ndarray): shape (1, 1024) or (1024,)
        clinical_embedding (np.ndarray): shape (1, 34) or (34,)
        model_path (Path): path to trained MLP (.pth)

    Returns:
        float: predicted probability of BRS3
    """

    # === Preprocess input embeddings ===
    hist_input = histology_embedding if histology_embedding.shape == (1, 1024) else histology_embedding.reshape(1, -1)
    clin_input = clinical_embedding if clinical_embedding.shape == (1, 34) else clinical_embedding.reshape(1, -1)

    # === Do NOT concatenate here! Pass embeddings directly to the model ===
    input_tensor_hist = torch.tensor(hist_input, dtype=torch.float32)
    input_tensor_clin = torch.tensor(clin_input, dtype=torch.float32)

    print(f"🔎 Final input shape → histology: {input_tensor_hist.shape}, clinical: {input_tensor_clin.shape}")

    # === Load the model ===
    model = FusionMLP(hist_dim=1024, clin_dim=34)  # Same dimensions as the original model
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # === Predict ===
    with torch.no_grad():
        logits = model(input_tensor_hist, input_tensor_clin)  # pass separately, do NOT concatenate
        print(f"🔢 Logits: {logits}")
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    del model
    torch.cuda.empty_cache()

    return float(probs[0])