"""
Unified Multi-Task Polymer GNN supporting GINE, GATv2, PNA, and GraphGPS (with RWSE).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATv2Conv, PNAConv, GPSConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import degree


def compute_degree_histogram(dataset, max_deg=10):
    deg_hist = torch.zeros(max_deg + 1, dtype=torch.long)
    for data in dataset:
        if data.edge_index.numel() > 0:
            d = torch.clamp(degree(data.edge_index[1], num_nodes=data.x.size(0), dtype=torch.long), max=max_deg)
            deg_hist += torch.bincount(d, minlength=max_deg + 1)
    deg_hist[0] = max(int(deg_hist[0]), 1)
    return deg_hist


class PolymerMultiTaskGNN(nn.Module):
    def __init__(self, arch="gps", node_dim=50, pe_dim_in=8, edge_dim=14, global_dim=160,
                 hidden_dim=96, num_layers=3, heads=4, dropout=0.15, use_global_features=True,
                 deg_histogram=None, num_tasks=5):
        super().__init__()
        self.arch = arch.lower()
        self.dropout = dropout
        self.use_global_features = use_global_features

        pe_hidden = 16
        self.pe_norm = nn.BatchNorm1d(pe_dim_in)
        self.pe_lin = nn.Linear(pe_dim_in, pe_hidden)
        self.node_emb = nn.Sequential(nn.Linear(node_dim + pe_hidden, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU())

        self.edge_dim_internal = max(hidden_dim // 4, 16) if self.arch == "pna" else hidden_dim
        self.edge_emb = nn.Sequential(nn.Linear(edge_dim, self.edge_dim_internal), nn.SiLU())

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if self.arch == "gine":
                mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
                self.convs.append(GINEConv(nn=mlp, edge_dim=self.edge_dim_internal, train_eps=True))
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.arch == "gatv2":
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=self.edge_dim_internal, dropout=dropout, concat=True))
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.arch == "pna":
                self.convs.append(PNAConv(hidden_dim, hidden_dim, aggregators=["mean", "min", "max", "std"], scalers=["identity", "amplification", "attenuation"], deg=deg_histogram, edge_dim=self.edge_dim_internal, towers=4, divide_input=True))
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.arch == "gps":
                local_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                local_conv = GINEConv(nn=local_mlp, edge_dim=self.edge_dim_internal, train_eps=True)
                self.convs.append(GPSConv(channels=hidden_dim, conv=local_conv, heads=heads, dropout=dropout, attn_type="multihead", norm="batch_norm"))
                self.norms.append(nn.Identity())

        readout_dim = hidden_dim * 3
        if self.use_global_features:
            self.global_encoder = nn.Sequential(
                nn.BatchNorm1d(global_dim), nn.Linear(global_dim, hidden_dim), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU()
            )
            fused_dim = readout_dim + hidden_dim
            self.physics_skip = nn.Linear(16, num_tasks)
            nn.init.zeros_(self.physics_skip.weight)
            nn.init.zeros_(self.physics_skip.bias)
        else:
            self.global_encoder = None
            self.physics_skip = None
            fused_dim = readout_dim

        self.shared_fusion = nn.Sequential(nn.Linear(fused_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), nn.SiLU(), nn.Dropout(dropout))
        self.task_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Dropout(dropout * 0.5), nn.Linear(hidden_dim, 1))
            for _ in range(num_tasks)
        ])

    def forward(self, batch):
        x, pe, edge_index, edge_attr, b_idx = batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch
        is_bb = x[:, -2:-1]
        h = self.node_emb(torch.cat([x, self.pe_lin(self.pe_norm(pe))], dim=-1))
        e_h = self.edge_emb(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            if self.arch == "gps":
                h = F.silu(conv(h, edge_index, b_idx, edge_attr=e_h))
            else:
                h_new = F.dropout(F.silu(norm(conv(h, edge_index, edge_attr=e_h))), p=self.dropout, training=self.training)
                h = h + h_new

        bb_w = is_bb + 0.05
        pool_bb = global_add_pool(h * bb_w, b_idx) / global_add_pool(bb_w, b_idx).clamp(min=1e-3)
        g_repr = torch.cat([global_mean_pool(h, b_idx), global_max_pool(h, b_idx), pool_bb], dim=-1)

        if self.use_global_features and self.global_encoder is not None:
            u_in = torch.nan_to_num(batch.u, nan=0.0, posinf=0.0, neginf=0.0)
            g_repr = torch.cat([g_repr, self.global_encoder(u_in)], dim=-1)

        shared_h = self.shared_fusion(g_repr)
        preds = torch.cat([head(shared_h) for head in self.task_heads], dim=-1)
        if self.use_global_features and self.physics_skip is not None:
            preds = preds + self.physics_skip(torch.nan_to_num(batch.u[:, :16], nan=0.0, posinf=0.0, neginf=0.0))
        return preds
