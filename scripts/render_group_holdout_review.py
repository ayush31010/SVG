"""Render explicit target labels for annotation review; no model predictions."""
import io
import json
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import resvg_py
from PIL import Image, ImageDraw, ImageFont
from svgpatchlab.core.xml import index_tree, parse_svg


def run(root=Path('data/group-holdout-v1')):
    manifest = json.loads((root / 'manifest.json').read_text())
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 15)
    except OSError:
        font = ImageFont.load_default(size=15)
    for batch in range((len(manifest) + 9) // 10):
        sheet = Image.new('RGB', (1100, 1000), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, item in enumerate(manifest[batch*10:batch*10+10]):
            x, y = (j % 2)*550, (j // 2)*200
            tree = parse_svg((root / item['source_path']).read_text())
            tree.set('width', '140')
            tree.set('height', '140')
            for node in index_tree(tree):
                if node.node_id in item['candidate_ids']:
                    node.element.set('stroke', '#d22' if node.node_id in item['gold_target_ids'] else '#b8b8b8')
            png = resvg_py.svg_to_bytes(svg_string=ET.tostring(tree, encoding='unicode'))
            image = Image.open(io.BytesIO(png)).convert('RGBA')
            sheet.paste(image, (x+5, y+40), image)
            label = (f"{batch*10+j+1}. {item['case_id']}\n" +
                     '\n'.join(textwrap.wrap(item['instruction'], 43)) +
                     f"\nGold: {','.join(item['gold_target_ids'])}")
            draw.multiline_text((x+155, y+25), label, font=font, fill='black', spacing=5)
        sheet.save(root / f'review-{batch+1:02d}.png')


if __name__ == '__main__':
    run()
