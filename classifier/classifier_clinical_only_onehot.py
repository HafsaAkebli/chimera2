# classifier/classifier_clinical_only.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# === DEFINE CLINICAL MLP MODEL ===
class ClinMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),   # optional layer normalization
                nn.ReLU(),
            ]
        layers.append(nn.Linear(dims[-1], 2))  # output layer (2 classes: BRS3 or non-BRS3)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
# === INFERENCE FUNCTION ===
def predict_probability_clinical_only(
    clinical_embedding: np.ndarray,
    model_path: Path
) -> float:
    """
    Given clinical embedding and path to model, returns the probability of BRS3.

    Args:
        clinical_embedding (np.ndarray): shape (1, C) - clinical embedding vector
        model_path (Path): path to the trained ClinicalMLP .pth file

    Returns:
        float: predicted probability of BRS3
    """
    # Convert clinical embedding to tensor (no scaling as per training setup)
    X_tensor = torch.tensor(clinical_embedding, dtype=torch.float32)

    # Load the trained model
    model = ClinMLP(input_dim=X_tensor.shape[1])  # Ensure the input dimension is correct
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Perform inference
    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Probability of BRS3
    
    return float(probs[0])