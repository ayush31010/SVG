import pytest
import torch
from svgpatchlab.gnn.features import scene_to_graph_inputs
from svgpatchlab.gnn.models import SceneGNN, AttentionSceneGNN

def test_scene_to_graph_inputs_and_gnn_forward():
    # Mock scene
    scene = {
        "root_id": "n0",
        "nodes": [
            {
                "id": "n0",
                "tag": "svg",
                "depth": 0,
                "parent": None,
                "attributes": {},
                "resolved_style": {"fill": "#ffffff"}
            },
            {
                "id": "n1",
                "tag": "rect",
                "depth": 1,
                "parent": "n0",
                "attributes": {},
                "resolved_style": {"fill": "#ff0000"},
                "visual_context": {
                    "summary": "red rectangle",
                    "labels": ["box"],
                    "confidence": 0.9,
                    "visible_pixels": 100
                }
            },
            {
                "id": "n2",
                "tag": "circle",
                "depth": 1,
                "parent": "n0",
                "attributes": {},
                "resolved_style": {"fill": "#ff0000"}
            }
        ]
    }

    # 1. Test Feature Extraction
    x, edge_index, edge_type, id_to_idx, vision_scalar_x, vision_summaries = scene_to_graph_inputs(scene)
    
    assert x.shape == (3, 17)
    assert vision_scalar_x.shape == (3, 4)
    assert len(vision_summaries) == 3
    assert vision_summaries[1] == "red rectangle"
    
    # 2 edges for parent-child (n0-n1, n1-n0, n0-n2, n2-n0) 
    # 2 edges for sibling (n1-n2, n2-n1)
    # 2 edges for same_fill (n1-n2, n2-n1)
    assert edge_index.shape[1] == 8
    assert edge_type.shape[0] == 8
    
    # 2. Test Model Forward Pass
    in_dim = 17
    instr_dim = 384
    vision_dim = 4 + 384
    
    # Dummy instruction embed (1, 384)
    instr = torch.randn(1, 384)
    batch = torch.zeros(3, dtype=torch.long)
    
    # Dummy vision embed (3, 384)
    vision_summary_embeds = torch.randn(3, 384)
    vision_x = torch.cat([vision_scalar_x, vision_summary_embeds], dim=1)
    has_vision = vision_scalar_x[:, 0]
    
    # Standard GNN
    model1 = SceneGNN(in_dim, instr_dim, vision_dim)
    out1 = model1(x, edge_index, edge_type, instr, vision_x, has_vision, batch)
    assert out1.shape == (3,)
    
    # Attention GNN
    model2 = AttentionSceneGNN(in_dim, instr_dim, vision_dim)
    out2 = model2(x, edge_index, edge_type, instr, vision_x, has_vision, batch)
    assert out2.shape == (3,)

if __name__ == "__main__":
    test_scene_to_graph_inputs_and_gnn_forward()
    print("GNN Integration Tests Passed!")
