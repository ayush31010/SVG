from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, RGATConv, global_mean_pool

from .features import NUM_RELATIONS

class FiLM(nn.Module):
    """Instruction-conditioned feature-wise modulation, applied after every layer."""
    def __init__(self, instr_dim: int, hidden_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(instr_dim, hidden_dim * 2)

    def forward(self, h: torch.Tensor, instr_per_node: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(instr_per_node).chunk(2, dim=-1)
        return h * (1 + gamma) + beta


class SceneGNN(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        instr_dim: int, 
        vision_dim: int, 
        hidden_dim: int = 64, 
        num_relations: int = NUM_RELATIONS,
        num_layers: int = 3, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        self.convs = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            for _ in range(num_layers)
        ])
        self.films = nn.ModuleList([FiLM(instr_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        self.dropout = dropout
        
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # node repr + scene-level context
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: torch.Tensor, 
        instr: torch.Tensor, 
        vision_x: torch.Tensor, 
        has_vision: torch.Tensor, 
        batch: torch.Tensor
    ) -> torch.Tensor:
        h = self.input_proj(x)
        vision_contrib = self.vision_proj(vision_x) * has_vision.unsqueeze(-1)
        h = h + vision_contrib
        
        instr_per_node = instr[batch]
        for conv, film, norm in zip(self.convs, self.films, self.norms):
            h_new = conv(h, edge_index, edge_type)
            h_new = film(h_new, instr_per_node)
            h_new = F.relu(norm(h_new))
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual
            
        scene_ctx = global_mean_pool(h, batch)
        scene_ctx_per_node = scene_ctx[batch]
        
        combined = torch.cat([h, scene_ctx_per_node], dim=-1)
        return self.score(combined).squeeze(-1)


class AttentionSceneGNN(nn.Module):
    """
    Upgraded version using Relational Graph Attention Networks (RGAT).
    This allows the model to dynamically weigh edges based on the nodes and the relation type.
    """
    def __init__(
        self, 
        in_dim: int, 
        instr_dim: int, 
        vision_dim: int, 
        hidden_dim: int = 64, 
        num_relations: int = NUM_RELATIONS,
        num_layers: int = 3, 
        heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        self.convs = nn.ModuleList([
            RGATConv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim // heads, 
                num_relations=num_relations,
                heads=heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.films = nn.ModuleList([FiLM(instr_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        self.dropout = dropout
        
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: torch.Tensor, 
        instr: torch.Tensor, 
        vision_x: torch.Tensor, 
        has_vision: torch.Tensor, 
        batch: torch.Tensor
    ) -> torch.Tensor:
        h = self.input_proj(x)
        vision_contrib = self.vision_proj(vision_x) * has_vision.unsqueeze(-1)
        h = h + vision_contrib
        
        instr_per_node = instr[batch]
        for conv, film, norm in zip(self.convs, self.films, self.norms):
            h_new = conv(h, edge_index, edge_type)
            h_new = film(h_new, instr_per_node)
            h_new = F.relu(norm(h_new))
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual
            
        scene_ctx = global_mean_pool(h, batch)
        scene_ctx_per_node = scene_ctx[batch]
        
        combined = torch.cat([h, scene_ctx_per_node], dim=-1)
        return self.score(combined).squeeze(-1)
