# histology/mean_mil.py
# Mean-MIL aggregation: LayerNorm -> FC -> ReLU -> FC + fixed skip (QR); mean over patches.
# Loads fixed weights (meanMIL_1024_fixed.pt) and returns a (1, 1024) embedding.

import numpy as np
import torch
import torch.nn as nn


class SimplePatchMLPMean(nn.Module):
    """
    For each patch vector x in R^{1536}:
      x_n = LayerNorm(x)
      h   = ReLU(FC1(x_n)) -> FC2(h)
      z   = h + FixedLinear(x_n)   # deterministic skip via QR init
    Mean over patches -> R^{1024}.
    """
    def __init__(self, in_dim=1536, hidden=512, out_dim=1024):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1  = nn.Linear(in_dim, hidden)
        self.fc2  = nn.Linear(hidden, out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)

        # zero-init nonlinear path so initial output ~ skip(norm(x))
        nn.init.zeros_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

        # deterministic fixed skip (only matters if you DON'T load weights)
        torch.manual_seed(42)
        with torch.no_grad():
            w = torch.randn(out_dim, in_dim)       # [out_dim, in_dim]
            q, _ = torch.linalg.qr(w.t(), mode="reduced")  # q: [in_dim, out_dim]
            self.skip.weight.copy_(q.t())          # [out_dim, in_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N_patches, 1536]
        x_n = self.norm(x)
        h   = torch.relu(self.fc1(x_n))
        h   = self.fc2(h)
        z   = h + self.skip(x_n)
        return z.mean(dim=0)  # [1024]


def _load_mean_mil_model(mean_mil_path: str, in_dim=1536, hidden=512, out_dim=1024):
    """
    Build the model, move to device (GPU if available), and load saved weights.
    Returns (model, device).
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
    Aggregate UNI2 patch features (top-6K) into a single embedding.

    Args:
        features: np.ndarray with shape [N_patches, 1536]
        mean_mil_path: path to "meanmil/meanMIL_1024_fixed.pt"

    Returns:
        np.ndarray with shape (1, 1024)  # ready for the classifier
    """
    if features is None or features.size == 0:
        raise ValueError("❌ features is empty or None.")
    if features.ndim != 2 or features.shape[1] != in_dim:
        raise ValueError(f"❌ Expected features shape [N, {in_dim}], got {features.shape}")

    model, device = _load_mean_mil_model(mean_mil_path, in_dim=in_dim, hidden=hidden, out_dim=out_dim)

    feats_t = torch.as_tensor(features, dtype=torch.float32, device=device)  # [N, 1536]
    pooled  = model(feats_t)                                                # [1024]
    emb     = pooled.detach().cpu().numpy().astype(np.float32)[None, :]     # (1, 1024)

    print(f"✅ Mean-MIL embedding shape: {emb.shape}")
    return emb
