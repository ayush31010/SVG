"""Matched candidate-set heads on a frozen MLP; natural labels never train heads."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_vector_edits_groups import structural_candidate_groups
from scripts.run_cloud_set_grounding import digest, paired_sets, write_json
from scripts.run_vector_edits_set_grounding import _metrics, _selection_record
from scripts.run_vector_edits_grounding import _candidate_ids
from svgpatchlab.core import build_scene
from svgpatchlab.core.geometry import node_analytic_stats
from svgpatchlab.core.xml import index_tree, parse_svg
from svgpatchlab.vision import (GraphMoEGrounder, StructuralGroupGrounder,
    build_svg_graph, create_instruction_encoder, extract_target_reference)
from train.graph_moe_grounding import generate_cases, build_examples


def drawable_ids(source):
    """Use the same editable units for synthetic training and natural scoring."""
    return _candidate_ids(index_tree(parse_svg(source)))


def candidate_sets(source, ids, scores, mode):
    """Source-only candidate sets. Control has singletons and score prefixes."""
    if mode not in ('prefix', 'groups'):
        raise ValueError(mode)
    order = {n: i for i, n in enumerate(ids)}
    ranking = sorted(ids, key=lambda n: (-scores[n], order[n]))
    candidates = {(n,) for n in ids}
    candidates.update(tuple(sorted(ranking[:k], key=order.__getitem__))
                      for k in range(1, min(6, len(ids)) + 1))
    if mode == 'groups':
        candidates.update(structural_candidate_groups(source, ids))
    return sorted(candidates, key=lambda ns: (len(ns), tuple(order[n] for n in ns)))


def features(graph, prediction, text, candidates):
    import numpy as np
    nodes = np.asarray(graph.node_features, dtype='float32')
    order = {n: i for i, n in enumerate(graph.node_ids)}
    scores = np.asarray([prediction.node_scores[n] for n in graph.node_ids])
    count_probs = prediction.cardinality_probabilities
    rows = []
    for candidate in candidates:
        ix = [order[n] for n in candidate]
        subset, selected = nodes[ix], scores[ix]
        remaining = [i for i in range(len(nodes)) if i not in ix]
        outside = nodes[remaining].mean(0) if remaining else np.zeros(nodes.shape[1])
        k = len(ix)
        extra = [k / len(nodes), min(k, 12) / 12, selected.mean(), selected.min(),
                 selected.max(), selected.std(), selected.sum() / max(scores.sum(), 1e-8),
                 count_probs[k-1] if k <= len(count_probs) else 0,
                 float(k == prediction.predicted_cardinality)]
        rows.append(np.concatenate([text, subset.mean(0), subset.max(0), outside, extra]))
    return np.asarray(rows, dtype='float32')


def training_targets(candidates, gold):
    """Uniform supervision on best-F1 candidates, including impossible cases."""
    import numpy as np
    truth = set(gold)
    f1 = np.array([2 * len(set(c) & truth) / (len(c) + len(truth)) for c in candidates])
    best = np.isclose(f1, f1.max(), atol=1e-8, rtol=0).astype('float32')
    return best / best.sum()


def make_head(dim, hidden):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(),
                               torch.nn.Linear(hidden, 1))


def categorize(selected, gold):
    selected, gold = set(selected), set(gold)
    if selected == gold:
        return 'exact'
    if selected < gold:
        return 'missing_parts'
    if gold < selected:
        return 'extra_parts'
    if not selected & gold:
        return 'wrong_object'
    return 'mixed_missing_and_extra'


def source_bootstrap(left, right, seed=20260910, draws=5000):
    """Paired resampling of source blocks, preserving both instructions."""
    from collections import defaultdict
    import numpy as np
    right_by_id = {r['case_id']: r for r in right}
    if set(right_by_id) != {r['case_id'] for r in left}:
        raise ValueError('Bootstrap requires matching cases')
    blocks = defaultdict(list)
    for row in left:
        other = right_by_id[row['case_id']]
        if row['source_id'] != other['source_id']:
            raise ValueError('Bootstrap source mismatch')
        blocks[row['source_id']].append(int(row['set_exact']) - int(other['set_exact']))
    values = list(blocks.values())
    rng = random.Random(seed)
    distribution = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        distribution.append(sum(sum(v) for v in sample) / sum(len(v) for v in sample))
    return {'sources': len(values), 'draws': draws,
            'mean_difference': sum(sum(v) for v in values) / len(left),
            'percentile_95_interval': np.quantile(distribution, [.025, .975]).tolist(),
            'resampling_unit': 'source SVG; all instructions retained together'}


def run(config_path, device='cuda', output_root='runs/group-candidate-v1'):
    import torch
    from collections import Counter
    config = json.loads(Path(config_path).read_text())
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    torch.set_num_threads(2)
    seed = config['seed']
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    baseline = ROOT / config['baseline_checkpoint']
    if digest(baseline) != config['baseline_sha256']:
        raise ValueError('Frozen baseline checkpoint changed')
    model = GraphMoEGrounder.load_checkpoint(baseline, device=device)
    encoder = create_instruction_encoder(model.metadata['instruction_encoder'])
    training_config = json.loads((ROOT / config['synthetic_config']).read_text())
    cases = generate_cases(training_config)
    examples = build_examples(cases, seed=training_config['seed'],
        validation_fraction=training_config['validation_fraction'], use_target_reference=True)
    source_by_id = {c.case_id: c.source_svg for c in cases}
    holdout = ROOT / config['holdout_manifest']
    if digest(holdout) != config['holdout_sha256']:
        raise ValueError('Frozen holdout manifest changed')
    # The manifest itself is read only after training and checkpoint selection.
    report = {'status': 'training', 'config': config, 'device': device,
              'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
              'torch': torch.__version__, 'training': {}, 'metrics': {}}
    write_json(output / 'summary.json', report)
    prepared = {mode: {'train': [], 'validation': []} for mode in ('prefix', 'groups')}
    print('STAGE preparing identical synthetic examples', flush=True)
    for split, items in examples.items():
        for example in items:
            text = encoder.encode(example.instruction)
            pred = model.predict(example.graph, text)
            ids = drawable_ids(source_by_id[example.case_id])
            for mode in prepared:
                candidates = candidate_sets(source_by_id[example.case_id], ids, pred.node_scores, mode)
                x = torch.tensor(features(example.graph, pred, text, candidates), device=device)
                y = torch.tensor(training_targets(candidates, example.targets), device=device)
                exact = torch.tensor([set(c) == set(example.targets) for c in candidates], device=device)
                prepared[mode][split].append((x, y, exact))
        print(f'Prepared {split}: {len(items)}', flush=True)
    trained = {}
    for mode in prepared:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        data = prepared[mode]
        head = make_head(data['train'][0][0].shape[1], config['hidden_dim']).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'])
        history, best = [], -1
        for epoch in range(config['epochs']):
            order = list(range(len(data['train'])))
            random.Random(f'{seed}:{epoch}').shuffle(order)
            head.train()
            loss_sum = 0.0
            for i in order:
                x, y, _ = data['train'][i]
                optimizer.zero_grad(set_to_none=True)
                loss = -(torch.log_softmax(head(x).flatten(), dim=0) * y).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item()
            head.eval()
            with torch.no_grad():
                exact = sum(bool(g[head(x).flatten().argmax()]) for x, _, g in data['validation'])
            history.append({'epoch': epoch+1, 'loss': loss_sum / len(order),
                            'validation_exact': exact / len(data['validation'])})
            if exact > best:
                best = exact
                torch.save({'state_dict': head.state_dict(), 'input_dim': x.shape[1],
                            'hidden_dim': config['hidden_dim'], 'epoch': epoch+1}, output / (mode+'.pt'))
            print(mode, history[-1], flush=True)
        saved = torch.load(output / (mode+'.pt'), map_location=device, weights_only=True)
        head.load_state_dict(saved['state_dict'])
        trained[mode] = head
        report['training'][mode] = {'history': history, 'selected_epoch': saved['epoch'],
            'examples': {s: len(d) for s, d in data.items()},
            'parameter_count': sum(p.numel() for p in head.parameters())}
        write_json(output / 'summary.json', report)
    print('STAGE frozen holdout evaluation', flush=True)
    manifest = json.loads(holdout.read_text())
    train_hashes = {hashlib.sha256(c.source_svg.encode()).hexdigest() for c in cases}
    records = {arm: [] for arm in ('baseline', 'prefix', 'groups')}
    rules = StructuralGroupGrounder()
    for item in manifest:
        source = (holdout.parent / item['source_path']).read_text()
        if digest(holdout.parent / item['source_path']) != item['source_sha256']:
            raise ValueError('Holdout source changed: '+item['case_id'])
        if hashlib.sha256(source.encode()).hexdigest() in train_hashes:
            raise ValueError('Holdout source overlaps training')
        if tuple(item['candidate_ids']) != tuple(drawable_ids(source)):
            raise ValueError('Manifest uses different editable units')
        case = SimpleNamespace(**item, source_svg=source)
        started = time.perf_counter()
        graph = build_svg_graph(build_scene(source, visual_stats=node_analytic_stats(source)))
        text = encoder.encode(extract_target_reference(case.instruction))
        pred = model.predict(graph, text)
        ranking = sorted(case.candidate_ids, key=lambda n: (-pred.node_scores[n], n))
        base = ranking[:pred.predicted_cardinality]
        rule = rules.predict(source, extract_target_reference(case.instruction), case.candidate_ids)
        shared_time = time.perf_counter() - started
        for arm in records:
            started = time.perf_counter()
            candidates = []
            if arm == 'baseline':
                selected = base
            else:
                candidates = candidate_sets(source, case.candidate_ids, pred.node_scores, arm)
                x = torch.tensor(features(graph, pred, text, candidates), device=device)
                with torch.no_grad():
                    selected = candidates[int(trained[arm](x).flatten().argmax())]
            if not rule.abstained:
                selected = rule.selected_ids
            record = _selection_record(case, selected, rule.rule or arm, {})
            record.update(source_id=item['source_id'],
                failure_category=categorize(selected, case.gold_target_ids),
                wall_seconds=shared_time + time.perf_counter()-started,
                candidate_coverage=any(set(c)==set(case.gold_target_ids) for c in candidates) if candidates else None)
            records[arm].append(record)
    for arm, rows in records.items():
        report['metrics'][arm] = {**_metrics(rows),
            'failure_categories': dict(Counter(r['failure_category'] for r in rows)),
            'candidate_coverage_rate': (sum(r['candidate_coverage'] for r in rows)/len(rows)
                                        if arm != 'baseline' else None),
            'mean_wall_seconds': sum(r['wall_seconds'] for r in rows)/len(rows)}
    report['paired_groups_vs_prefix'] = paired_sets(records['groups'], records['prefix'])
    report['paired_groups_vs_baseline'] = paired_sets(records['groups'], records['baseline'])
    report['source_bootstrap_groups_vs_prefix'] = source_bootstrap(records['groups'], records['prefix'])
    report['source_bootstrap_groups_vs_baseline'] = source_bootstrap(records['groups'], records['baseline'])
    report['status'] = 'complete'
    write_json(output / 'records.json', records)
    write_json(output / 'summary.json', report)
    with zipfile.ZipFile(output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in output.rglob('*'):
            if path.is_file():
                bundle.write(path, path.relative_to(output))
    print('COMPLETE', json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output-root', default='runs/group-candidate-v1')
    args = parser.parse_args()
    run(args.config, args.device, args.output_root)
