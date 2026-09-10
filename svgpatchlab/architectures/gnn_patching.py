from __future__ import annotations

import json
from pathlib import Path

import torch

from svgpatchlab.core import PatchPolicy, apply_patch, build_scene, validate_patch
from svgpatchlab.core.envelope_repair import parse_patch_with_envelope_repair
from svgpatchlab.models import ModelAdapter
from svgpatchlab.types import ArchitectureResult, BenchmarkCase, ModelRequest
from svgpatchlab.vision import VisionContextAnnotator
from svgpatchlab.gnn import get_text_encoder, scene_to_graph_inputs, VISION_SCALAR_DIM, SceneGNN, AttentionSceneGNN

from .base import Architecture
from .prompts import gnn_patch_prompt
from .patching import _correction_prompt

class GnnPatchArchitecture(Architecture):
    name = "gnn_patch"

    def __init__(
        self,
        policy: PatchPolicy | None = None,
        vision_context_config: dict | None = None,
        max_repair_retries: int = 1,
        gnn_model_path: str = "gnn_node_targeting.pt",
        use_attention: bool = True,
    ):
        self.policy = policy or PatchPolicy()
        self.vision_context = VisionContextAnnotator(vision_context_config)
        self.max_repair_retries = max_repair_retries
        self.use_attention = use_attention
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn = self._load_gnn_model(gnn_model_path)

    def _load_gnn_model(self, path: str) -> torch.nn.Module | None:
        if not Path(path).exists():
            return None
            
        # Dummy dimensions since we know them from the feature extractor
        in_dim = 17 
        instr_dim = 384
        vision_dim = VISION_SCALAR_DIM + 384
        
        if self.use_attention:
            model = AttentionSceneGNN(in_dim, instr_dim, vision_dim)
        else:
            model = SceneGNN(in_dim, instr_dim, vision_dim)
            
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def run(
        self,
        case: BenchmarkCase,
        model: ModelAdapter,
        vision_model: ModelAdapter | None = None,
    ) -> ArchitectureResult:
        result = ArchitectureResult(model_calls=1)
        try:
            scene = build_scene(case.source_svg)
            scene = self.vision_context.annotate(
                case.source_svg,
                scene,
                vision_model,
                request_id=case.case_id,
            )
            context = json.dumps(scene, indent=2, sort_keys=True)
            root_id = scene["root_id"]
            
            predicted_target_id = root_id
            
            # Run GNN
            if self.gnn is not None:
                with torch.no_grad():
                    x, edge_index, edge_type, id_to_idx, vision_scalar_x, vision_summaries = scene_to_graph_inputs(scene)
                    
                    idx_to_id = {v: k for k, v in id_to_idx.items()}
                    
                    text_encoder = get_text_encoder()
                    instr_embed = text_encoder.encode([case.instruction], convert_to_tensor=True).to(self.device)
                    
                    if any(vision_summaries):
                        vision_summary_embeds = text_encoder.encode(vision_summaries, convert_to_tensor=True).to(self.device)
                    else:
                        vision_summary_embeds = torch.zeros((len(scene["nodes"]), 384), device=self.device)
                        
                    has_vision = vision_scalar_x[:, 0].to(self.device)
                    vision_x = torch.cat([vision_scalar_x.to(self.device), vision_summary_embeds], dim=1)
                    
                    batch = torch.zeros(len(scene["nodes"]), dtype=torch.long, device=self.device)
                    
                    scores = self.gnn(
                        x.to(self.device), 
                        edge_index.to(self.device), 
                        edge_type.to(self.device), 
                        instr_embed, 
                        vision_x, 
                        has_vision, 
                        batch
                    )
                    
                    pred_idx = scores.argmax().item()
                    predicted_target_id = idx_to_id[pred_idx]
            else:
                # Fallback if model not loaded
                predicted_target_id = root_id

            last_error: str | None = None
            response_text: str | None = None
            succeeded = False
            attempts = 0

            for attempt_num in range(1, self.max_repair_retries + 2):
                attempts = attempt_num
                if attempt_num == 1:
                    prompt = gnn_patch_prompt(case.instruction, "SVG DOM skeleton", context, predicted_target_id)
                else:
                    prompt = _correction_prompt(
                        case.instruction, "SVG DOM skeleton", context, response_text, last_error
                    )

                response = model.generate(
                    ModelRequest(
                        prompt,
                        metadata={"request_id": case.case_id, "attempt": attempt_num, "gnn_target": predicted_target_id},
                    )
                )
                response_text = response.text
                result.raw_responses.append(response_text)

                try:
                    patch = parse_patch_with_envelope_repair(response_text, task=case.task, root_id=root_id)
                    validate_patch(patch, scene, self.policy, task=case.task)
                except Exception as inner_exc:
                    last_error = f"{type(inner_exc).__name__}: {inner_exc}"
                    continue

                result.patch = patch
                result.output_svg = apply_patch(case.source_svg, result.patch)
                last_error = None
                succeeded = True
                break

            result.model_calls = attempts
            if not succeeded:
                result.error = last_error
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        return result
