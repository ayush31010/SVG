# Identifiability of SVG nodes from the rendering

## The question

Render-grounded node selection assumes a referring expression can be resolved by
looking at pixels. That assumption has a precondition: the node must *have*
distinctive pixels. The renderer `R` is not injective, so this is not guaranteed.

Using the visible-contribution definition already in this project,

```text
V(n) = R(T) - R(T \ n)
```

every node falls into exactly one of three classes:

| Class | Condition | Recoverable from pixels? |
|---|---|---|
| `identifiable` | `V(n)` non-empty and shared with no other node | yes |
| `render_equivalent` | `V(n)` non-empty but byte-identical to another node's | no |
| `invisible` | `V(n)` empty | no |

The last two are the fibres of `R`. On them, *no* visual evidence distinguishes
the node, regardless of model capacity, raster resolution, or panel count. A
grounding system that always commits to an answer is reporting confidence it
cannot possess.

`svgpatchlab/vision/identifiability.py` computes this classification.
`scripts/identifiability_census.py` aggregates it over a corpus.

## Measurement

Two corpora, chosen to contrast curation against real drawing practice:

- **SVGEditBench** — the curated emoji benchmark this project evaluates on.
- **OpenClipart** — 178,604 CC0 artist-authored SVGs
  (`nyuuzyou/openclipart` on Hugging Face), sampled by reservoir from one of
  18 shards so the sample is not biased toward the front of the file.

300 documents each, rendered at 128px, threshold 0.02, per-document budget of
400 rasterizations. Run on an H100 box because the census needs cairo.

### Node-level shares

| | Emoji | Clipart |
|---|---:|---:|
| Documents analysed | 300 | 216 |
| Nodes | 2,371 | 12,517 |
| `identifiable` | **98.9%** | **58.7%** |
| `render_equivalent` | 0.0% | 10.8% |
| `invisible` | 1.1% | 30.5% |
| **Non-identifiable** | **1.1%** | **41.3%** |
| Documents with >= 1 non-identifiable node | 6.3% | 74.1% |

Emoji are almost perfectly identifiable. Real clipart is not: two nodes in five
cannot be singled out by any visual method, and three documents in four contain
at least one such node.

### The relationship to document size is U-shaped

Mean per-document non-identifiable share, by node-count quartile:

| Bucket | Emoji nodes/doc | Emoji | Clipart nodes/doc | Clipart | Clipart docs affected |
|---|---:|---:|---:|---:|---:|
| Q1 | 2 | 0.0% | 2 | 50.5% | 61.1% |
| Q2 | 4 | 0.8% | 13 | 34.9% | 70.4% |
| Q3 | 9 | 1.0% | 40 | 32.3% | 75.9% |
| Q4 | 16 | 1.4% | 132 | 42.9% | 88.9% |

Two different mechanisms produce the two ends. Tiny clipart files are usually a
wrapper group around a single path, and a group with one child is
render-equivalent to that child by construction. Large files accumulate genuine
occlusion and duplicated geometry. The middle is cleanest.

The **share of documents affected rises monotonically** to 88.9%, which is the
more stable signal: whatever the per-node ratio does, bigger drawings are more
likely to contain at least one unreachable node.

### Do not use the per-document mean as the headline

Documents are wildly uneven in size, so a mean over documents is dominated by
tiny wrapper-only files. The pooled node-level share is the honest statistic.
Raising the budget from 60 to 400 nodes per document moved the pooled clipart
figure from 32.5% to **41.3%** while the per-document mean moved the other way —
the two statistics genuinely disagree, and only one of them answers "what
fraction of nodes are unreachable".

## What is still unmeasured

Of 300 sampled clipart documents, 84 were skipped and named rather than dropped
silently:

| Reason | Count | Notes |
|---|---:|---|
| `too_many_nodes` | 58 | 445 to 18,924 nodes, median 1,504 |
| `SVGParseError` | 19 | malformed documents in the corpus |
| `RecursionError` | 7 | deeply nested trees |

The 58 budget overruns are the largest documents in the sample and remain
unmeasured. Q4 non-identifiability rises to 42.9% and documents-affected to
88.9%, so the unmeasured tail is unlikely to be cleaner than the measured
range — but that is an expectation, not a measurement, and 41.3% should be
reported as the figure for documents up to 400 nodes, not for the corpus.

## Why this matters for the architecture

The project's own ablations already show that visual grounding, supplied as
prompt context, does not beat a structure-only skeleton
(`docs/VISUAL_STATS_AB.md`, and the 500-case split in
`runs/qwen2.5-7b-analytic-ablation-v1-server/occlusion-split-analysis.json`:
skeleton 64.0% against rendered 60.4%, sign test p ~ 0.90). Meanwhile the
context-v3 selector, which poses grounding as closed-choice discrimination,
reaches 96.1% exact node-set on the full test manifest — but loses only 14.4
points when every pixel is blanked, so it is leaning heavily on structural
metadata rather than on appearance.

This census supplies the missing explanation. On emoji, 98.9% of nodes are
identifiable, so pixels are nearly redundant with structure and the two signals
cannot be told apart. Any benchmark built on emoji will therefore understate
both the difficulty of grounding and the value of getting it right.

The open problem is not a better visual encoder. It is what a system should do
on the 41.3%: detect that a referent is not identifiable and abstain or ask,
rather than select a decoy. In the occluded holdout the analytic arm chose the
decoy on 33.3% of occluded cases while scoring 0.0% correct.

## Reproducing

```bash
# Needs cairo; the development laptop does not have it.
python -m scripts.run_remote_census --sample 300 --size 128 --max-nodes 400 \
  --results-dir runs/identifiability-census-400
```

The driver runs `tests/test_identifiability.py` on the host before the census
and stops on failure, so a broken classifier cannot silently emit a
plausible-looking number.
