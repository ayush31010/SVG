import io
import json
import math
from pathlib import Path

import pytest

from scripts.run_semantic_prior_grounding import ALPHA_GRID, choose_alpha, fused_scores, rank

ROOT = Path(__file__).resolve().parents[1]


def test_fused_scores_interpolates_between_mlp_logit_and_scaled_cosine():
    mlp = {'n1': 0.5, 'n2': 0.9, 'g0': 0.99}
    siglip = {'n1': 0.10, 'n2': 0.05}
    fused = fused_scores(mlp, siglip, 0.25, 100.0)
    assert set(fused) == {'n1', 'n2'}, 'only drawable candidates are scored'
    assert fused['n1'] == pytest.approx(0.75 * 0.0 + 0.25 * 10.0)
    assert fused['n2'] == pytest.approx(0.75 * math.log(0.9 / 0.1) + 0.25 * 5.0)
    assert fused_scores(mlp, siglip, 0.0, 100.0)['n2'] == pytest.approx(math.log(9))
    assert fused_scores(mlp, siglip, 1.0, 100.0)['n1'] == pytest.approx(10.0)


def test_fused_scores_rejects_missing_candidates_and_bad_weights():
    with pytest.raises(ValueError):
        fused_scores({'n1': 0.5}, {'n1': 0.1, 'n2': 0.1}, 0.5, 100.0)
    with pytest.raises(ValueError):
        fused_scores({'n1': 0.5}, {'n1': 0.1}, 1.5, 100.0)
    with pytest.raises(ValueError):
        fused_scores({'n1': 0.5}, {'n1': 0.1}, 0.5, 0.0)
    saturated = fused_scores({'n1': 1.0, 'n2': 0.0}, {'n1': 0.0, 'n2': 0.0}, 0.0, 1.0)
    assert all(math.isfinite(v) for v in saturated.values())


def test_rank_breaks_ties_by_node_id():
    assert rank({'n2': 1.0, 'n1': 1.0, 'n3': 2.0}, ['n1', 'n2', 'n3']) == ['n3', 'n1', 'n2']


def test_choose_alpha_prefers_deployed_then_ranking_then_smallest_weight():
    sweep = [
        {'alpha': 0.0, 'deployed_exact': 40, 'oracle_cardinality_exact': 30},
        {'alpha': 0.3, 'deployed_exact': 44, 'oracle_cardinality_exact': 50},
        {'alpha': 0.6, 'deployed_exact': 44, 'oracle_cardinality_exact': 55},
        {'alpha': 0.8, 'deployed_exact': 44, 'oracle_cardinality_exact': 55},
        {'alpha': 1.0, 'deployed_exact': 20, 'oracle_cardinality_exact': 60},
    ]
    assert choose_alpha(sweep) == 0.6
    assert ALPHA_GRID[0] == 0.0 and ALPHA_GRID[-1] == 1.0 and len(ALPHA_GRID) == 11


def test_frozen_config_matches_dev_sweep_choice():
    config = json.loads((ROOT / 'configs/eval/semantic_prior_v1.json').read_text())
    sweep_path = ROOT / 'runs/semantic-prior-v1/dev_sweep.json'
    if config.get('alpha') is None:
        pytest.skip('fusion weight not frozen yet')
    sweep = json.loads(sweep_path.read_text())
    assert config['frozen'] is True
    assert config['alpha'] == sweep['chosen_alpha'] == choose_alpha(sweep['sweep'])
    assert sweep['sweep'][0]['alpha'] == 0.0 and sweep['sweep'][0]['deployed_exact'] == 41


def test_resvg_fallback_centres_non_square_documents():
    pytest.importorskip('resvg_py')
    from PIL import Image
    from svgpatchlab.eval.render import _ResvgSVGRenderer
    import resvg_py
    renderer = _ResvgSVGRenderer(resvg_py)
    png = renderer.svg2png(
        bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20">'
                   b'<rect width="40" height="20" fill="blue"/></svg>',
        output_width=64, output_height=64, background_color=None)
    image = Image.open(io.BytesIO(png)).convert('RGBA')
    assert image.size == (64, 64)
    assert image.getpixel((32, 32))[3] == 255 and image.getpixel((32, 2))[3] == 0
    solid = renderer.svg2png(bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"/>',
                             output_width=8, output_height=8, background_color='white')
    assert Image.open(io.BytesIO(solid)).convert('RGB').getpixel((0, 0)) == (255, 255, 255)


def test_hidden_elements_drop_existing_important_display_declarations():
    import xml.etree.ElementTree as ET
    from svgpatchlab.eval.render import _hide_element as hide_for_stats
    from svgpatchlab.vision.candidate_views import _hide_element as hide_for_views
    for hide in (hide_for_stats, hide_for_views):
        element = ET.Element('rect', {'style': 'fill:red; display:block!important ;opacity:0.5'})
        hide(element)
        declarations = [d.strip() for d in element.attrib['style'].split(';')]
        assert declarations == ['fill:red', 'opacity:0.5', 'display:none!important']
        bare = ET.Element('rect')
        hide(bare)
        assert bare.attrib['style'] == 'display:none!important'
