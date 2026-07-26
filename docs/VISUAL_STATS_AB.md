# Visual Stats Patch: Clean A/B Pilot

## Question

Does attaching cached, render-derived `visual` fields to the ordinary SVG DOM
skeleton help a frozen text-only Qwen model select the correct SVG node?

Node hiding is only the counterfactual measurement used to produce the fields.
The hidden SVG is discarded. The real edit remains:

```text
instruction + skeleton -> model patch -> validator -> executor
```

## A/B boundary

The control is `skeleton_patch`. The treatment is `visual_stats_patch`.
`visual_stats_patch` overrides only `scene_for()` to attach `visual`; it inherits
the same prompt construction, model call, parser, validator, and executor.

A regression test verifies that:

- removing `visual` from the treatment scene makes it equal to the control;
- both arms send no images, even to an image-capable adapter;
- request metadata is identical;
- an identical model response produces an identical patch and SVG output; and
- the standalone presets use the same dataset, model, and evaluator settings.

## Measurement corrections made before evaluation

- Replaced channel-wise median color with a quantized RGB mode so mixed red and
  blue pixels cannot invent purple.
- Corrected bbox and position projection for SVG `preserveAspectRatio`
  letterboxing.
- Made counterfactual hiding override an existing `display: ... !important`.
- Bumped the visual-stat cache format to `svgpatchlab.visual_stats.v2`.
- Defined `visible: false` precisely: hiding caused no unique pixel delta above
  the render threshold. This is weak negative evidence, not proof of occlusion.

All 36 relevant tests passed on the Linux server with the real Cairo renderer.

## Official SVGEditBench smoke

Command:

```bash
python3 -m svgpatchlab.cli matrix \
  --config configs/experiments/skeleton_patch.json \
  --model-config configs/models/qwen2.5-7b-ollama.json \
  --architectures skeleton_patch visual_stats_patch \
  --limit-per-task 2 \
  --render \
  --output-root runs/qwen2.5-7b-visual-stats-ab-smoke-v1
```

This uses ten matched task cases: two each for change-color, contour,
upside-down, transparency, and crop. The cases reuse two unique source SVGs
across the five tasks, so they are a compatibility smoke rather than ten
independent visual samples.

| Metric | Skeleton | + visual tags |
|---|---:|---:|
| Exact target set | 70% | 90% |
| Exact patch | 20% | 50% |
| Valid output | 60% | 90% |
| Mean failure-aware MSE | 0.4058 | 0.1002 |
| Prompt tokens | 17,694 | 21,279 |
| Mean model latency | 9.42 s | 9.99 s |

Paired outcomes:

- exact targets: 2 visual-only wins, 0 skeleton-only wins;
- exact patches: 3 visual-only wins, 0 skeleton-only wins; and
- MSE: 4 visual wins, 0 skeleton wins, 6 ties.

The four grounding-relevant official cases already had 100% target accuracy in
both arms because ordinary rect/circle coordinates and colors were exposed in
the baseline skeleton. The official smoke is therefore useful as a
non-regression check, but it is not a decisive spatial-grounding test.

## Path-only spatial grounding probe

The decisive pilot uses eight frozen instructions over two layout SVGs built
from four same-color `path` nodes:

- every node has the same tag and fill;
- every hidden `d` string has the same character count;
- path coordinates are replaced by opaque hashes in the model context;
- DOM order is changed across two layouts; and
- instructions identify exactly one node by top-left, top-right, bottom-left,
  or bottom-right.

| Metric | Skeleton | + visual tags |
|---|---:|---:|
| Exact target set | 0% | 75% |
| Exact patch | 0% | 75% |
| Mean failure-aware MSE | 0.01008 | 0.00175 |
| Prompt tokens | 14,272 | 17,224 |
| Mean model latency | 10.27 s | 9.60 s |

The paired target result was 6 visual-only wins, 0 skeleton-only wins, and 2
cases where neither arm was correct. This is direct evidence that the visual
tags supplied node-position information unavailable in the protected skeleton.

## Prompt sensitivity check

An experimental shared prompt explicitly told the model to apply visual
descriptors conjunctively. It strengthened the blind spatial baseline from
0% to 37.5%, while the visual arm remained 75%. On the official smoke it made
both arms worse and reduced the visual arm's exact-patch rate from 50% to 30%.
That prompt experiment was rejected; prompt v4 remains the default.

## Expanded 39-source held-out benchmark

The frozen follow-up uses 39 matched cases and 39 distinct source SVGs:

- 27 position cases cover all nine 3x3 spatial labels three times each;
- 12 size cases cover smallest, second-smallest, second-largest, and largest;
- target node indices and case-specific DOM orders are balanced;
- coordinates are jittered and aspect ratios vary while label validity is
  checked at runtime;
- all candidates remain same-fill `path` nodes with SHA-only protected
  geometry and equal `d` character counts; and
- seed, source/answer hashes, expected targets, prompt v4, and the prompt hash
  are recorded in `manifest.json`.

| Metric | Skeleton | + visual tags |
|---|---:|---:|
| Exact target set | 7.7% | 33.3% |
| Exact patch | 7.7% | 33.3% |
| Mean target precision | 14.7% | 40.3% |
| Mean target recall | 48.7% | 69.2% |
| Valid output | 100% | 100% |
| Mean failure-aware MSE | 0.01400 | 0.01031 |
| Prompt tokens | 92,404 | 116,833 |
| Mean model latency | 11.75 s | 14.44 s |

The paired target result was 10 visual-only wins, 0 skeleton-only wins, 3
correct in both arms, and 26 correct in neither. Position improved from 2/27
(7.4%) to 11/27 (40.7%); relative size improved only from 1/12 (8.3%) to 2/12
(16.7%). An exact two-sided McNemar/sign test over the 10 discordant target
pairs gives `p = 0.001953`, which establishes direction on this controlled
synthetic suite but not an effect size for natural SVGs.

All 78 outputs used the correct operation and attribute values. In the visual
arm, 15 of 26 failures over-selected multiple red paths and 11 selected one
wrong path. The remaining bottleneck is therefore model use of the grounding
metadata, especially ordinal size reasoning, rather than patch execution.

## Interpretation and next experiment

The experiments support the mechanism, but the controlled holdout is still too
narrow for a publication-scale generalization claim. The next evaluation
should be:

1. complete the launched full 500-case-per-arm official paired matrix;
2. separate target ranking from patch planning to prevent same-color
   over-selection;
3. compare ordinal size/centroid fields and the two-render ID-buffer fast path
   as separately named treatments;
4. extend the holdout to real authored paths, groups, overlap, and visibility;
5. report cold-cache and warm-cache preprocessing separately; and
6. build a dedicated complete-versus-incomplete-underlayer deletion benchmark.

The current full official matrix would require 1,000 Qwen calls and is expected
to take several hours on the shared server, so it should use a fresh output
directory and be scheduled deliberately.
