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
        proj_dim_hist=128,
        proj_dim_clin=128,
        fusion_hidden=[256, 128, 64],
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

        # Per-modality projectors
        self.hist_proj = nn.Sequential(
            nn.Linear(hist_dim, proj_dim_hist),
            make_norm(proj_dim_hist),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.clin_proj = nn.Sequential(
            nn.Linear(clin_dim, proj_dim_clin),
            make_norm(proj_dim_clin),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion MLP
        fusion_input_dim = proj_dim_hist + proj_dim_clin
        fusion_layers = []
        dims = [fusion_input_dim] + fusion_hidden
        for i in range(len(dims) - 1):
            fusion_layers.append(nn.Linear(dims[i], dims[i + 1]))
            fusion_layers.append(make_norm(dims[i + 1]))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(dropout))

        self.fusion_mlp = nn.Sequential(*fusion_layers)
        self.classifier = nn.Linear(fusion_hidden[-1], num_classes)

    def forward(self, hist_feat, clin_feat):
        h = self.hist_proj(hist_feat)
        c = self.clin_proj(clin_feat)
        x = torch.cat([h, c], dim=1)
        x = self.fusion_mlp(x)
        return self.classifier(x)


def predict_probability(
    histology_embedding: np.ndarray,
    clinical_embedding: np.ndarray,
    gat_scaler_path: Path,
    model_path: Path
) -> float:
    """
    Predicts BRS3 probability using the trained FusionMLP model.

    Args:
        histology_embedding (np.ndarray): shape (1, 1024) or (1024,)
        clinical_embedding (np.ndarray): shape (1, 34) or (34,)
        gat_scaler_path (Path): path to histology scaler (.pkl)
        model_path (Path): path to trained MLP (.pth)

    Returns:
        float: predicted probability of BRS3
    """
    # === Preprocess input embeddings ===
    # Ensure the embeddings are reshaped correctly if necessary
    hist_input = histology_embedding if histology_embedding.shape == (1, 1024) else histology_embedding.reshape(1, -1)
    clin_input = clinical_embedding if clinical_embedding.shape == (1, 34) else clinical_embedding.reshape(1, -1)

    # === Load the histology scaler ===
    scaler_hist = joblib.load(gat_scaler_path)

    # === Scale the histology embedding ===
    hist_input_scaled = scaler_hist.transform(hist_input)

    # === Convert to torch tensors ===
    input_tensor_hist = torch.tensor(hist_input_scaled, dtype=torch.float32)
    input_tensor_clin = torch.tensor(clin_input, dtype=torch.float32)

    print(f"🔎 Final input shape → histology: {input_tensor_hist.shape}, clinical: {input_tensor_clin.shape}")

    # === Load the model ===
    model = FusionMLP(hist_dim=1024, clin_dim=34)  # Now using hist_dim=1024 for histology embeddings
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # === Predict ===
    with torch.no_grad():
        logits = model(input_tensor_hist, input_tensor_clin)  # pass histology and clinical features separately
        print(f"🔢 Logits: {logits}")
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Get probability for class 1 (BRS3)

    return float(probs[0])  # Return the predicted probability of BRS3