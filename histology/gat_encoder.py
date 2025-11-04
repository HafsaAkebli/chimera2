import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

# === Architecture matching the fixed GAT encoder used during training ===
class WSI_GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, heads=16, n_layers=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True))
        self.norms.append(nn.BatchNorm1d(hidden_dim * heads))

        for _ in range(n_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim * heads, heads=1, concat=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim * heads))

        self.pool = global_mean_pool

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            if x.shape == x_res.shape:
                x = x + x_res
        graph_feat = self.pool(x, batch)
        return None, graph_feat  # maintain return structure
    

# === Graph construction from patch-level features ===
def build_graph_from_features(features, k=5):
    # Standardize patch features (same as training)
    features = StandardScaler().fit_transform(features)
    x = torch.tensor(features, dtype=torch.float)

    # KNN with Euclidean distance
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(features)
    _, indices = nbrs.kneighbors(features)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self-loop
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def extract_patient_embedding_from_features(features, k=5, gat_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = build_graph_from_features(features, k=k)
    graph = graph.to(device)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

    in_dim = graph.x.shape[1]  # should be 1536
    model = WSI_GAT(in_dim=in_dim).to(device)

    # 🔒 Load pretrained GAT weights
    if gat_path is not None:
        state_dict = torch.load(gat_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded pretrained GAT from: {gat_path}")
    else:
        print("⚠️ No GAT weights provided. Using random init (not recommended).")

    model.eval()
    with torch.no_grad():
        _, graph_feat = model(graph.x, graph.edge_index, graph.batch)

    return graph_feat.cpu().numpy()  # shape: (1, 1024)