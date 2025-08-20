# histology/mean_mil.py

import numpy as np
import torch
import torch.nn as nn


class SimplePatchMLPMean(nn.Module):
    """
    Mean-MIL aggregation: LayerNorm -> FC -> ReLU -> FC + fixed skip (identity).
    Mean over patches → [1, 1024] embedding.
    """
    def __init__(self, in_dim=1536, hidden=512, out_dim=1024):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1  = nn.Linear(in_dim, hidden)
        self.fc2  = nn.Linear(hidden, out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        # DO NOT initialize: weights loaded from checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.norm(x)
        h   = torch.relu(self.fc1(x_n))
        h   = self.fc2(h)
        z   = h + self.skip(x_n)
        return z.mean(dim=0)  # → [out_dim]


def _load_mean_mil_model(mean_mil_path: str, in_dim=1536, hidden=512, out_dim=1024):
    """
    Loads the fixed saved weights for inference — no randomness, fully deterministic.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePatchMLPMean(in_dim=in_dim, hidden=hidden, out_dim=out_dim).to(device)
    state = torch.load(mean_mil_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


@torch.no_grad()
def mean_mil_embed(features: np.ndarray, mean_mil_path: str, in_dim=1536, hidden=512, out_dim=1024) -> np.ndarray:
    """
    Aggregates UNI2 patch features into a single fixed [1, 1024] embedding.

    Args:
        features: np.ndarray of shape [N_patches, 1536]
        mean_mil_path: path to saved weights, e.g. meanMIL_1024_fixed.pt

    Returns:
        np.ndarray of shape [1, 1024]
    """
    if features is None or features.size == 0:
        raise ValueError("❌ Empty features.")
    if features.ndim != 2 or features.shape[1] != in_dim:
        raise ValueError(f"❌ Expected features shape [N, {in_dim}], got {features.shape}")

    model, device = _load_mean_mil_model(mean_mil_path, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    feats_t = torch.as_tensor(features, dtype=torch.float32, device=device)
    pooled  = model(feats_t)
    emb     = pooled.detach().cpu().numpy().astype(np.float32)[None, :]
    print(f"✅ Mean-MIL embedding shape: {emb.shape}")
    return emb
