from __future__ import annotations

from typing import Any

from .base import Architecture
from .diagnostic import OraclePatchArchitecture, OracleTargetArchitecture, TwoStagePatchArchitecture
from .patching import (
    AnalyticStatsPatchArchitecture,
    ContextRoutedPatchArchitecture,
    FullContextPatchArchitecture,
    RoutedStrictContextRoutedPatchArchitecture,
    RoutedStrictSkeletonPatchArchitecture,
    RoutedStrictVisualStatsPatchArchitecture,
    SkeletonPatchArchitecture,
    StrictAnalyticStatsPatchArchitecture,
    StrictContextRoutedPatchArchitecture,
    StrictSkeletonPatchArchitecture,
    StrictVisualStatsPatchArchitecture,
    VisualGNNPatchArchitecture,
    VisualSkeletonPatchArchitecture,
    VisualStatsPatchArchitecture,
)
from .rewrite import FullRewriteArchitecture
from .rules import RuleBasedPatchArchitecture
from .semantic import SemanticIdPatchArchitecture


ARCHITECTURES: dict[str, type[Architecture]] = {
    "oracle_patch": OraclePatchArchitecture,
    "rule_based_patch": RuleBasedPatchArchitecture,
    "full_rewrite": FullRewriteArchitecture,
    "full_context_patch": FullContextPatchArchitecture,
    "skeleton_patch": SkeletonPatchArchitecture,
    "visual_skeleton_patch": VisualSkeletonPatchArchitecture,
    "visual_stats_patch": VisualStatsPatchArchitecture,
    "analytic_stats_patch": AnalyticStatsPatchArchitecture,
    "strict_analytic_stats_patch": StrictAnalyticStatsPatchArchitecture,
    "strict_skeleton_patch": StrictSkeletonPatchArchitecture,
    "strict_visual_stats_patch": StrictVisualStatsPatchArchitecture,
    "context_routed_patch": ContextRoutedPatchArchitecture,
    "strict_context_routed_patch": StrictContextRoutedPatchArchitecture,
    "routed_strict_context_routed_patch": (
        RoutedStrictContextRoutedPatchArchitecture
    ),
    "routed_strict_skeleton_patch": RoutedStrictSkeletonPatchArchitecture,
    "routed_strict_visual_stats_patch": RoutedStrictVisualStatsPatchArchitecture,
    "visual_gnn_patch": VisualGNNPatchArchitecture,
    "semantic_id_patch": SemanticIdPatchArchitecture,
    "oracle_target_patch": OracleTargetArchitecture,
    "two_stage_patch": TwoStagePatchArchitecture,
}


def create_architecture(name: str, **options: Any) -> Architecture:
    try:
        architecture_type = ARCHITECTURES[name]
    except KeyError as exc:
        raise ValueError(f"unknown architecture: {name}") from exc
    return architecture_type(**options)
