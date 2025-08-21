# histology/mean_mil.py

import numpy as np
import torch
import torch.nn as nn


class SimplePatchMLPMean(nn.Module):
    """
    Mean-MIL aggregation used in training:
      BatchNorm1d -> Linear -> ReLU -> Linear -> mean over patches
    Output is a single patient embedding of shape [out_dim].
    """
    def __init__(self, in_dim=1536, hidden=512, out_dim=1536):
        super().__init__()
        self.bn  = nn.BatchNorm1d(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

        # Deterministic (fixed) weights for inference-time pooling — match saved checkpoint
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N_patches, in_dim]
        returns: [out_dim]
        """
        x = self.bn(x)      # identity in eval mode with default BN stats
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.mean(dim=0)



def _load_mean_mil_model(mean_mil_path: str, in_dim=1536, hidden=512, out_dim=1536):
    """
    Load the fixed saved weights for deterministic inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePatchMLPMean(in_dim=in_dim, hidden=hidden, out_dim=out_dim).to(device)
    state = torch.load(mean_mil_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()  # BN runs in eval (uses running stats -> identity with defaults)
    return model, device


@torch.no_grad()
def mean_mil_embed(
    features: np.ndarray,
    mean_mil_path: str,
    in_dim: int = 1536,
    hidden: int = 512,
    out_dim: int = 1536,
) -> np.ndarray:
    """
    Aggregate UNI2 patch features into a single fixed embedding.

    Args:
        features: np.ndarray [N_patches, in_dim]
        mean_mil_path: path to saved weights, e.g. ".../meanMIL_1536_fixed.pt"

    Returns:
        np.ndarray of shape [1, out_dim]
    """
    if features is None or features.size == 0:
        raise ValueError("❌ Empty features.")
    if features.ndim != 2 or features.shape[1] != in_dim:
        raise ValueError(f"❌ Expected features shape [N, {in_dim}], got {features.shape}")

    model, device = _load_mean_mil_model(mean_mil_path, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    feats_t = torch.as_tensor(features, dtype=torch.float32, device=device)
    pooled = model(feats_t)                         # [out_dim]
    emb = pooled.detach().cpu().numpy().astype(np.float32)[None, :]  # [1, out_dim]
    # print(f"✅ Mean-MIL embedding shape: {emb.shape}")  # optional log
    return emb
