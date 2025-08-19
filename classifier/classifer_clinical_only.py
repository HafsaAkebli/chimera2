# classifier/classifier_clinical_only.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from pathlib import Path

# === DEFINE CLINICAL MLP MODEL ===
class ClinicalMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 128], dropout=0.1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            #layers.append(nn.BatchNorm1d(dims[i + 1]))
            #layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
# === INFERENCE FUNCTION ===
def predict_probability_clinical_only(
    clinical_embedding: np.ndarray,
    model_path: Path,
    scaler_path: Path
) -> float:
    """
    Given clinical embedding and paths, returns the probability of BRS3.

    Args:
        clinical_embedding (np.ndarray): shape (1, C)
        model_path (Path): path to ClinicalMLP .pth file
        scaler_path (Path): path to StandardScaler .pkl

    Returns:
        float: predicted probability of BRS3
    """
    # Load scaler and normalize embedding
    scaler = joblib.load(scaler_path)
    X = scaler.transform(clinical_embedding.reshape(1, -1))
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Load model
    model = ClinicalMLP(input_dim=X.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Predict
    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    del model
    torch.cuda.empty_cache()
    
    return float(probs[0])
