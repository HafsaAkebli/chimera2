import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# === DEFINE CONCATENATION-BASED MLP MODEL ===
class FusionMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[512, 128], dropout=0.1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))  # ← ADD THIS
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 2))  # Binary classification
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# === PREDICT FUNCTION ===
def predict_probability(
    graph_histology_embedding: np.ndarray,
    clinical_embedding: np.ndarray,
    model_path: Path,
    scaler_hist_path: Path,
    scaler_clin_path: Path
) -> float:
    """
    Predicts BRS3 probability using concatenation FusionMLP with saved scalers.

    Args:
        graph_histology_embedding (np.ndarray): shape (1, 1024) or (1024,)
        clinical_embedding (np.ndarray): shape (1, 1024) or (1024,)
        model_path (Path): path to trained MLP (.pth)
        scaler_hist_path (Path): path to histology StandardScaler (.pkl)
        scaler_clin_path (Path): path to clinical StandardScaler (.pkl)

    Returns:
        float: predicted probability of BRS3
    """

    # === Load trained scalers ===
    scaler_hist = joblib.load(scaler_hist_path)
    scaler_clin = joblib.load(scaler_clin_path)

    # === Preprocess input embeddings ===
    hist_input = graph_histology_embedding if graph_histology_embedding.shape == (1, 1024) else graph_histology_embedding.reshape(1, -1)
    clin_input = clinical_embedding if clinical_embedding.shape == (1, 1024) else clinical_embedding.reshape(1, -1)

    hist_scaled = scaler_hist.transform(hist_input)
    clin_scaled = scaler_clin.transform(clin_input)

    # === Concatenate features ===
    fused_input = np.concatenate([clin_scaled, hist_scaled], axis=1)  # shape: (1, 2048)
    input_tensor = torch.tensor(fused_input, dtype=torch.float32)

    print(f"🔎 Final input shape → input_tensor: {input_tensor.shape}")

    # === Load model ===
    model = FusionMLP(input_dim=2048)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # === Predict ===
    with torch.no_grad():
        logits = model(input_tensor)
        print(f"🔢 Logits: {logits}")
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    del model
    torch.cuda.empty_cache()

    return float(probs[0])