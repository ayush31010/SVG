"""Sparse instruction-conditioned mixture of graph experts for SVG grounding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .graph_features import (
    EDGE_NAMES,
    EXPERT_NAMES,
    GEOMETRY_SLICE,
    NODE_FEATURE_DIM,
    STRUCTURE_SLICE,
    STYLE_SLICE,
    TAG_SLICE,
    SVGGraph,
    expert_index,
)


CHECKPOINT_FORMAT = "svgpatchlab.graph_moe.v1"


class GraphMoEUnavailable(RuntimeError):
    pass


def _torch_dependencies():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - core install omits torch
        raise GraphMoEUnavailable(
            "Graph-MoE requires PyTorch: pip install torch"
        ) from exc
    return torch, nn


@dataclass(frozen=True)
class GraphMoEPrediction:
    selected_expert: str
    router_weights: dict[str, float]
    node_scores: dict[str, float]
    expert_node_scores: dict[str, dict[str, float]]


def _feature_mask(node_feature_dim: int, visual_dim: int, expert: str) -> list[float]:
    mask = [0.0] * (node_feature_dim + visual_dim)

    def enable(selection: slice) -> None:
        for index in range(selection.start or 0, min(selection.stop or 0, node_feature_dim)):
            mask[index] = 1.0

    enable(TAG_SLICE)
    enable(STRUCTURE_SLICE)
    if expert == "attribute":
        enable(STYLE_SLICE)
    elif expert == "spatial":
        enable(GEOMETRY_SLICE)
    elif expert == "semantic":
        enable(STYLE_SLICE)
        enable(GEOMETRY_SLICE)
        for index in range(node_feature_dim, node_feature_dim + visual_dim):
            mask[index] = 1.0
    else:  # pragma: no cover - internal construction uses constants
        raise ValueError(expert)
    return mask


def _make_network(config: Mapping[str, Any]):
    torch, nn = _torch_dependencies()
    node_feature_dim = int(config["node_feature_dim"])
    visual_dim = int(config["visual_dim"])
    text_dim = int(config["text_dim"])
    hidden_dim = int(config["hidden_dim"])
    num_layers = int(config["num_layers"])
    dropout = float(config["dropout"])
    top_k = int(config["top_k"])
    structured_text_dim = int(config.get("structured_text_dim", 32))
    inductive_biases = bool(config.get("inductive_biases", False))
    relation_count = len(EDGE_NAMES)

    class RelationalLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_projection = nn.Linear(hidden_dim, hidden_dim)
            self.relation_projection = nn.Parameter(
                torch.empty(relation_count, hidden_dim, hidden_dim)
            )
            nn.init.xavier_uniform_(self.relation_projection)
            self.normalization = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, hidden, adjacency):
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            aggregate = torch.matmul(adjacency, hidden) / degree
            messages = torch.einsum(
                "rnh,rhk->rnk", aggregate, self.relation_projection
            ).mean(dim=0)
            updated = torch.nn.functional.gelu(
                self.self_projection(hidden) + messages
            )
            return self.normalization(hidden + self.dropout(updated))

    class Expert(nn.Module):
        def __init__(self, name: str, relational: bool):
            super().__init__()
            input_dim = node_feature_dim + visual_dim
            self.register_buffer(
                "feature_mask",
                torch.tensor(_feature_mask(node_feature_dim, visual_dim, name)),
            )
            self.node_projection = nn.Linear(input_dim, hidden_dim)
            self.query_projection = nn.Linear(text_dim, hidden_dim)
            self.layers = nn.ModuleList(
                RelationalLayer() for _ in range(num_layers if relational else 0)
            )
            self.scorer = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.name = name
            if inductive_biases and name == "attribute":
                self.color_match_scale = nn.Parameter(torch.tensor(4.0))
            if inductive_biases and name == "spatial":
                self.geometry_prior_scale = nn.Parameter(torch.tensor(4.0))

        def _attribute_prior(self, node_features, text_features):
            tail = text_dim - structured_text_dim
            query_rgb = text_features[tail : tail + 3]
            query_present = text_features[tail + 3]
            fill_rgb = node_features[:, 10:13]
            fill_present = node_features[:, 16]
            distance = torch.abs(fill_rgb - query_rgb.unsqueeze(0)).mean(dim=-1)
            return query_present * fill_present * (1.0 - 4.0 * distance)

        def _spatial_prior(self, node_features, text_features):
            tail = text_dim - structured_text_dim
            cue = text_features[tail + 4 : tail + 24]
            center_x = node_features[:, 23].clamp(0.0, 1.0)
            center_y = node_features[:, 24].clamp(0.0, 1.0)
            area = node_features[:, 25].clamp_min(0.0)

            # Cue order is defined by HashingInstructionEncoder. Suppress
            # absolute top/bottom priors for relational phrases such as
            # "below the top-middle circle"; those are left to message passing.
            relational_vertical = torch.maximum(cue[9], cue[10])
            left = torch.maximum(cue[0], cue[2])
            right = torch.maximum(cue[1], cue[3])
            top = torch.maximum(cue[4], cue[6]) * (1.0 - relational_vertical)
            bottom = torch.maximum(cue[5], cue[7]) * (1.0 - relational_vertical)
            center = cue[8]
            second_smallest = text_features[tail + 29]
            second_largest = text_features[tail + 30]
            smallest = cue[11] * (1.0 - second_smallest)
            largest = torch.maximum(cue[12], cue[13]) * (1.0 - second_largest)

            contributions = []
            active = []
            if float(left.detach()) > 0.0:
                contributions.append(left * (1.0 - center_x))
                active.append(left)
            if float(right.detach()) > 0.0:
                contributions.append(right * center_x)
                active.append(right)
            if float(top.detach()) > 0.0:
                contributions.append(top * (1.0 - center_y))
                active.append(top)
            if float(bottom.detach()) > 0.0:
                contributions.append(bottom * center_y)
                active.append(bottom)
            if float(center.detach()) > 0.0:
                center_score = 1.0 - (
                    torch.abs(center_x - 0.5) + torch.abs(center_y - 0.5)
                ).clamp(max=1.0)
                contributions.append(center * center_score)
                active.append(center)
            if float(smallest.detach()) > 0.0:
                smaller_rank = (area.unsqueeze(1) < area.unsqueeze(0)).float().mean(dim=1)
                contributions.append(smallest * smaller_rank)
                active.append(smallest)
            if float(largest.detach()) > 0.0:
                larger_rank = (area.unsqueeze(1) > area.unsqueeze(0)).float().mean(dim=1)
                contributions.append(largest * larger_rank)
                active.append(largest)
            if float(second_smallest.detach()) > 0.0:
                smaller_count = (area.unsqueeze(1) > area.unsqueeze(0)).float().sum(dim=1)
                ordinal_score = torch.exp(-torch.abs(smaller_count - 1.0))
                contributions.append(second_smallest * ordinal_score)
                active.append(second_smallest)
            if float(second_largest.detach()) > 0.0:
                smaller_count = (area.unsqueeze(1) > area.unsqueeze(0)).float().sum(dim=1)
                target_rank = max(node_features.shape[0] - 2, 0)
                ordinal_score = torch.exp(
                    -torch.abs(smaller_count - float(target_rank))
                )
                contributions.append(second_largest * ordinal_score)
                active.append(second_largest)
            if not contributions:
                return torch.zeros_like(center_x)
            score = sum(contributions) / torch.stack(active).sum().clamp_min(1.0)
            return 2.0 * score - 1.0

        def forward(self, node_features, adjacency, text_features):
            nodes = self.node_projection(node_features * self.feature_mask)
            query = self.query_projection(text_features)
            hidden = torch.nn.functional.gelu(nodes + query.unsqueeze(0))
            for layer in self.layers:
                hidden = layer(hidden, adjacency)
            expanded = query.unsqueeze(0).expand(hidden.shape[0], -1)
            cross = hidden * expanded
            logits = self.scorer(
                torch.cat((hidden, expanded, cross), dim=-1)
            ).squeeze(-1)
            if inductive_biases and self.name == "attribute":
                logits = logits + self.color_match_scale * self._attribute_prior(
                    node_features, text_features
                )
            if inductive_biases and self.name == "spatial":
                logits = logits + self.geometry_prior_scale * self._spatial_prior(
                    node_features, text_features
                )
            return logits

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(EXPERT_NAMES)),
            )
            if inductive_biases:
                self.router_cue_projection = nn.Linear(
                    len(EXPERT_NAMES), len(EXPERT_NAMES), bias=False
                )
                with torch.no_grad():
                    self.router_cue_projection.weight.copy_(
                        4.0 * torch.eye(len(EXPERT_NAMES))
                    )
            self.experts = nn.ModuleList(
                (
                    Expert("attribute", relational=False),
                    Expert("spatial", relational=True),
                    Expert("semantic", relational=True),
                )
            )

        def route(self, text_features, sparse: bool, forced_expert: int | None = None):
            logits = self.router(text_features)
            if inductive_biases:
                tail = text_dim - structured_text_dim
                route_cues = text_features[tail + 25 : tail + 28]
                logits = logits + self.router_cue_projection(route_cues)
            dense = torch.softmax(logits, dim=-1)
            if forced_expert is not None:
                weights = torch.zeros_like(dense)
                weights[forced_expert] = 1.0
            elif sparse:
                retained, indexes = torch.topk(dense, k=top_k)
                weights = torch.zeros_like(dense).scatter(0, indexes, retained)
                weights = weights / weights.sum().clamp_min(1e-12)
            else:
                weights = dense
            return logits, weights

        def forward(
            self,
            node_features,
            adjacency,
            text_features,
            *,
            sparse: bool,
            forced_expert: int | None = None,
        ):
            router_logits, router_weights = self.route(
                text_features, sparse=sparse, forced_expert=forced_expert
            )
            if sparse:
                # Top-k inference is computationally sparse: inactive experts
                # are not executed. A large negative sentinel keeps diagnostic
                # sigmoid scores near zero without affecting the weighted sum.
                expert_logits = node_features.new_full(
                    (len(EXPERT_NAMES), node_features.shape[0]), -30.0
                )
                active = torch.nonzero(router_weights > 0.0).flatten().tolist()
                for index in active:
                    expert_logits[index] = self.experts[index](
                        node_features, adjacency, text_features
                    )
            else:
                expert_logits = torch.stack(
                    [
                        expert(node_features, adjacency, text_features)
                        for expert in self.experts
                    ],
                    dim=0,
                )
            combined = (expert_logits * router_weights.unsqueeze(1)).sum(dim=0)
            return {
                "combined_logits": combined,
                "expert_logits": expert_logits,
                "router_logits": router_logits,
                "router_weights": router_weights,
            }

    return Network()


class GraphMoEGrounder:
    """Trainable wrapper that keeps torch optional for the rest of the package."""

    def __init__(
        self,
        *,
        node_feature_dim: int = NODE_FEATURE_DIM,
        visual_dim: int = 0,
        text_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        top_k: int = 1,
        structured_text_dim: int = 32,
        inductive_biases: bool = False,
        device: str = "cpu",
        metadata: Mapping[str, Any] | None = None,
    ):
        if node_feature_dim != NODE_FEATURE_DIM:
            raise ValueError(
                f"node_feature_dim must match feature contract {NODE_FEATURE_DIM}"
            )
        if visual_dim < 0 or text_dim < 1 or hidden_dim < 1 or num_layers < 0:
            raise ValueError("Graph-MoE dimensions must be non-negative and non-zero")
        if not 1 <= top_k <= len(EXPERT_NAMES):
            raise ValueError("top_k must fit the number of graph experts")
        if not 28 <= structured_text_dim < text_dim:
            raise ValueError("structured_text_dim must fit the instruction embedding")
        self.config = {
            "node_feature_dim": node_feature_dim,
            "visual_dim": visual_dim,
            "text_dim": text_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "top_k": top_k,
            "structured_text_dim": structured_text_dim,
            "inductive_biases": inductive_biases,
        }
        self.device = device
        self.metadata = dict(metadata or {})
        self.network = _make_network(self.config).to(device)

    def graph_tensors(
        self,
        graph: SVGGraph,
        visual_embeddings: Mapping[str, Sequence[float]] | None = None,
    ):
        torch, _ = _torch_dependencies()
        if not graph.node_ids:
            raise ValueError("cannot score an empty SVG graph")
        nodes = torch.tensor(graph.node_features, dtype=torch.float32, device=self.device)
        adjacency = torch.tensor(graph.adjacency, dtype=torch.float32, device=self.device)
        visual_dim = int(self.config["visual_dim"])
        if visual_dim:
            visual = torch.zeros(
                len(graph.node_ids), visual_dim, dtype=torch.float32, device=self.device
            )
            for index, node_id in enumerate(graph.node_ids):
                raw = list((visual_embeddings or {}).get(node_id, ()))[:visual_dim]
                if raw:
                    visual[index, : len(raw)] = torch.tensor(
                        raw, dtype=torch.float32, device=self.device
                    )
            nodes = torch.cat((nodes, visual), dim=-1)
        return nodes, adjacency

    def forward(
        self,
        graph: SVGGraph,
        text_embedding: Sequence[float],
        *,
        visual_embeddings: Mapping[str, Sequence[float]] | None = None,
        sparse: bool = False,
        forced_expert: str | None = None,
    ):
        torch, _ = _torch_dependencies()
        nodes, adjacency = self.graph_tensors(graph, visual_embeddings)
        text = torch.zeros(
            int(self.config["text_dim"]), dtype=torch.float32, device=self.device
        )
        raw = list(text_embedding)[: int(self.config["text_dim"])]
        if raw:
            text[: len(raw)] = torch.tensor(raw, dtype=torch.float32, device=self.device)
        return self.network(
            nodes,
            adjacency,
            text,
            sparse=sparse,
            forced_expert=(
                expert_index(forced_expert) if forced_expert is not None else None
            ),
        )

    def route(self, text_embedding: Sequence[float]) -> dict[str, float]:
        torch, _ = _torch_dependencies()
        text = torch.zeros(
            int(self.config["text_dim"]), dtype=torch.float32, device=self.device
        )
        raw = list(text_embedding)[: int(self.config["text_dim"])]
        if raw:
            text[: len(raw)] = torch.tensor(raw, dtype=torch.float32, device=self.device)
        self.network.eval()
        with torch.no_grad():
            _, weights = self.network.route(text, sparse=True)
        return {
            name: float(weights[index].detach().cpu())
            for index, name in enumerate(EXPERT_NAMES)
        }

    def predict(
        self,
        graph: SVGGraph,
        text_embedding: Sequence[float],
        *,
        visual_embeddings: Mapping[str, Sequence[float]] | None = None,
        forced_expert: str | None = None,
    ) -> GraphMoEPrediction:
        torch, _ = _torch_dependencies()
        self.network.eval()
        with torch.no_grad():
            output = self.forward(
                graph,
                text_embedding,
                visual_embeddings=visual_embeddings,
                sparse=True,
                forced_expert=forced_expert,
            )
            scores = torch.sigmoid(output["combined_logits"]).detach().cpu().tolist()
            expert_scores = torch.sigmoid(output["expert_logits"]).detach().cpu().tolist()
            weights = output["router_weights"].detach().cpu().tolist()
        selected = max(range(len(weights)), key=weights.__getitem__)
        return GraphMoEPrediction(
            selected_expert=EXPERT_NAMES[selected],
            router_weights={name: float(weights[i]) for i, name in enumerate(EXPERT_NAMES)},
            node_scores={node_id: float(scores[i]) for i, node_id in enumerate(graph.node_ids)},
            expert_node_scores={
                name: {
                    node_id: float(expert_scores[expert][index])
                    for index, node_id in enumerate(graph.node_ids)
                }
                for expert, name in enumerate(EXPERT_NAMES)
            },
        )

    @staticmethod
    def select_targets(
        prediction: GraphMoEPrediction,
        *,
        threshold: float = 0.5,
        max_targets: int = 4,
    ) -> tuple[str, ...]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("target threshold must be between zero and one")
        if max_targets < 1:
            raise ValueError("max_targets must be positive")
        ranked = sorted(
            prediction.node_scores,
            key=lambda node_id: (-prediction.node_scores[node_id], node_id),
        )
        selected = [
            node_id
            for node_id in ranked
            if prediction.node_scores[node_id] >= threshold
        ][:max_targets]
        return tuple(selected or ranked[:1])

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        torch, _ = _torch_dependencies()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "config": dict(self.config),
                "metadata": {**self.metadata, **dict(metadata or {})},
                "state_dict": self.network.state_dict(),
            },
            target,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, device: str = "cpu"):
        torch, _ = _torch_dependencies()
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:  # pragma: no cover - older supported torch
            payload = torch.load(path, map_location=device)
        if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported Graph-MoE checkpoint: {path}")
        config = dict(payload.get("config", {}))
        grounder = cls(device=device, metadata=payload.get("metadata", {}), **config)
        grounder.network.load_state_dict(payload["state_dict"])
        grounder.network.eval()
        return grounder
