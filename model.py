import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialEncoder(nn.Module):

    def __init__(self, spatial_dim: int, num_fixed_bands: int = 6, num_learned_features: int = 16,
                 max_freq: float = 32.0, include_input: bool = True, include_poly: bool = True,
                 learned_scale: float = 1.0):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.num_fixed_bands = num_fixed_bands
        self.num_learned_features = num_learned_features
        self.include_input = include_input
        self.include_poly = include_poly
        if num_fixed_bands > 1:
            exponents = torch.arange(num_fixed_bands, dtype=torch.float32) / (num_fixed_bands - 1)
            bands = max_freq ** exponents
        else:
            bands = torch.ones(num_fixed_bands, dtype=torch.float32)
        self.register_buffer("bands", bands, persistent=False)
        self.learned_freq_matrix = nn.Parameter(learned_scale * torch.randn(spatial_dim, num_learned_features))
        out_dim = 0
        if include_input:
            out_dim += spatial_dim
        if include_poly:
            out_dim += 4 if spatial_dim == 2 else 1
        out_dim += spatial_dim * num_fixed_bands * 2
        out_dim += num_learned_features * 2
        self.out_dim = out_dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = []
        if self.include_input:
            feats.append(coords)
        if self.include_poly:
            if self.spatial_dim == 2:
                x = coords[..., 0:1]
                y = coords[..., 1:2]
                r = torch.sqrt((x ** 2 + y ** 2).clamp_min(1e-12))
                feats.extend([x * x, y * y, x * y, r])
            else:
                feats.append(torch.norm(coords, dim=-1, keepdim=True))
        fixed_proj = math.pi * coords.unsqueeze(-1) * self.bands
        feats.extend([torch.sin(fixed_proj).flatten(start_dim=-2),
                      torch.cos(fixed_proj).flatten(start_dim=-2)])
        learned_proj = math.pi * (coords @ self.learned_freq_matrix)
        feats.extend([torch.sin(learned_proj), torch.cos(learned_proj)])
        return torch.cat(feats, dim=-1)


class TemporalEncoder(nn.Module):

    def __init__(self, time_features: int = 8, max_cycles: float = 32.0):
        super().__init__()
        self.time_features = time_features
        if time_features > 1:
            exponents = torch.arange(time_features, dtype=torch.float32) / (time_features - 1)
            cycles = max_cycles ** exponents
        else:
            cycles = torch.ones(time_features, dtype=torch.float32)
        self.register_buffer("omega", 2.0 * math.pi * cycles, persistent=False)

    def forward(self, t_norm: torch.Tensor, num_nodes: int) -> torch.Tensor:
        angles = t_norm.float().unsqueeze(1) * self.omega.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb.unsqueeze(1).expand(-1, num_nodes, -1)


class SliceAttention(nn.Module):

    def __init__(self, width: int, num_slices: int, num_heads: int, dropout: float = 0.0, ffn_ratio: float = 4.0):
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}")
        self.norm1 = nn.LayerNorm(width)
        self.slice_proj = nn.Linear(width, num_slices)
        self.attn = nn.MultiheadAttention(embed_dim=width, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(width)
        hidden_dim = int(width * ffn_ratio)
        self.ffn = nn.Sequential(nn.Linear(width, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, width))
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        weights = F.softmax(self.slice_proj(x_norm), dim=-1)
        weight_sum = weights.sum(dim=1, keepdim=True).transpose(1, 2).clamp_min(1e-8)
        slices = torch.bmm(weights.transpose(1, 2), x_norm) / weight_sum
        slices_out, _ = self.attn(slices, slices, slices, need_weights=False)
        upd = torch.bmm(weights, slices_out)
        x = x + self.attn_dropout(upd)
        x = x + self.ffn_dropout(self.ffn(self.norm2(x)))
        return x


class TransolverGraphNet(nn.Module):

    def __init__(self, in_ch=5, cond_dim=3, node_static_dim=7, hidden=192, depth=6,
                 num_slices=32, num_heads=8, ffn_mult=4, dropout=0.05, mlp_layers=3,
                 num_fixed_bands=6, num_learned_features=16, spatial_max_freq=32.0,
                 time_features=8, time_max_cycles=32.0, head_reinject=False):
        super().__init__()
        self.in_ch = in_ch
        self.head_reinject = head_reinject
        self.spatial = SpatialEncoder(spatial_dim=2, num_fixed_bands=num_fixed_bands,
                                      num_learned_features=num_learned_features, max_freq=spatial_max_freq)
        self.temporal = TemporalEncoder(time_features, time_max_cycles)
        time_dim = 2 * time_features
        self.cond_mlp = MLP(cond_dim, hidden, hidden, mlp_layers, dropout=dropout)

        embed_in = in_ch + node_static_dim + self.spatial.out_dim + time_dim + hidden
        self.embed = nn.Linear(embed_in, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList([
            SliceAttention(hidden, num_slices, num_heads, dropout=dropout, ffn_ratio=float(ffn_mult))
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(hidden)

        head_in = hidden + (in_ch if self.head_reinject else 0)
        self.head = nn.Sequential(nn.Linear(head_in, hidden), nn.SiLU(), nn.Linear(hidden, in_ch))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, coords, node_static, cond_node, tau):
        cond = self.cond_mlp(cond_node)
        comps = [x, node_static, self.spatial(coords), self.temporal(tau, x.shape[1]).to(x.dtype), cond]
        h = self.input_norm(self.embed(torch.cat(comps, dim=-1)))
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        if self.head_reinject:
            h = torch.cat([h, x], dim=-1)
        return x + self.head(h)


class DeepONet(nn.Module):

    def __init__(self, in_ch=5, cond_dim=3, hidden=512, p=400, branch_layers=4, trunk_layers=6,
                 num_fixed_bands=6, num_learned_features=16, spatial_max_freq=32.0,
                 time_features=8, time_max_cycles=32.0, dropout=0.0, **_):
        super().__init__()
        self.in_ch = in_ch
        self.p = p
        self.spatial = SpatialEncoder(spatial_dim=2, num_fixed_bands=num_fixed_bands,
                                      num_learned_features=num_learned_features, max_freq=spatial_max_freq)
        self.temporal = TemporalEncoder(time_features, time_max_cycles)
        self.branch = MLP(cond_dim, in_ch * p, hidden, branch_layers, dropout=dropout)
        self.trunk = MLP(self.spatial.out_dim + 2 * time_features, p, hidden, trunk_layers, dropout=dropout)
        self.b0 = nn.Parameter(torch.zeros(in_ch))
        self.scale = 1.0 / math.sqrt(p)
        last = self.branch.net[-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, x, coords, node_static, cond_node, tau):
        B, N, _ = x.shape
        cond = cond_node.mean(1)
        Bk = self.branch(cond).view(B, self.in_ch, self.p)
        tin = torch.cat([self.spatial(coords), self.temporal(tau, N).to(x.dtype)], dim=-1)
        Tk = self.trunk(tin)
        return x + torch.einsum("bnp,bcp->bnc", Tk, Bk) * self.scale + self.b0


def build_model(cfg, hp):
    name = cfg.get("model", "transolver")
    if name == "deeponet":
        return DeepONet(in_ch=5, cond_dim=3, hidden=512, p=400, branch_layers=4, trunk_layers=6,
                        dropout=hp["dropout"])
    return TransolverGraphNet(
        in_ch=5, cond_dim=3, node_static_dim=7, hidden=hp["hidden"], depth=hp["depth"],
        num_slices=hp["num_slices"], num_heads=hp["num_heads"], dropout=hp["dropout"],
        head_reinject=bool(cfg.get("head_reinject")))

