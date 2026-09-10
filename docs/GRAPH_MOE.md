# Sparse Graph-MoE grounding

The [new-collection candidate-group experiment](GROUP_CANDIDATE_RESULTS.md)
is complete after a drawable-only correction: the group-candidate head scores
23/100 against 27/100 for the matched control and 22/100 for the frozen
baseline, so structural group candidates gave no demonstrated gain.

The [completed Colab experiment and 15-seed follow-up](CLOUD_SET_GROUNDING.md)
now compare the set-aware GNN and no-edge MLP. On the T4 run, the MLP plus
grouping rules reaches 42/80 exact natural target sets; the graph model's
synthetic advantage does not transfer to this exploratory subset.

`graph_moe_patch` is the no-generative-VLM experiment motivated by the
visual-context ablations. The corrected full run showed that adding visual
statistics to every request increased prompt tokens and reduced change-color
target accuracy, while the deliberately spatial holdout improved sharply.
The architecture therefore routes before constructing visual evidence.

```text
instruction -> hashing text encoder -> learned top-1 router
                                      |-- attribute node expert
SVG DOM + geometry -------------------|-- spatial relation GNN
optional node render embedding -------|-- semantic/visual GNN
                                      -> selected node IDs
                                      -> deterministic validated patch
```

The experts have separate parameters and deliberately different feature
views. The attribute expert sees tag, style, and DOM structure but no rendered
geometry. The spatial expert sees tag, normalized geometry, twelve typed graph
relations, and DOM structure. The semantic expert sees all scalar features and
can additionally consume frozen per-node visual embeddings. The learned router
uses sparse top-1 inference; training uses a dense mixture plus auxiliary
reference-type and specialist losses.

## Train

The first preset uses only procedural SVGs. Source identities are assigned
wholly to train or validation, and the official SVGEditBench identities are
not used for optimization.

```bash
python -m train.graph_moe_grounding \
  --config configs/train/graph_moe_v3.json
```

The checkpoint and complete epoch history are written to
`checkpoints/graph-moe-v3/`. The default language representation is a small,
checkpoint-stable feature-hashing encoder with explicit continuous paint and
referring-expression cues. This isolates the value of graph structure before
adding a frozen small Transformer. The semantic expert accepts a configurable
visual vector, but the v1 preset keeps `visual_dim` at zero so the DOM/geometry
baseline is measurable first.

## Frozen mixed-holdout ablation

```bash
python scripts/run_graph_moe_ablation.py \
  --checkpoint checkpoints/graph-moe-v3/graph_moe.pt \
  --output-root runs/graph-moe-ablation-v3 \
  --progress
```

The script evaluates the learned router, oracle router, and every forced
expert. The oracle route is diagnostic rather than a deployable result: its
gap from the learned route measures routing error, while its remaining errors
measure node scoring. Results include target exactness, exact patch rate,
validity, per-reference-family scores, routing choices, feature source, and a
paired exact McNemar comparison. All arms make zero generative-model calls.

## Full benchmark

```bash
python -m svgpatchlab.cli evaluate \
  --config configs/experiments/graph_moe_patch.json
```

Whole-canvas transparency, crop, and vertical-flip tasks bypass the graph and
use the existing deterministic compiler. Local color and contour tasks use
the selected node IDs and a deterministic intent compiler, followed by the
same patch validation and executor used by the VLM architectures.

## September 2026 server results

The first model fit the procedural validation split perfectly but transferred
poorly to the frozen mixed holdout: `16/59` exact target sets. Oracle routing
was only `17/59`, identifying expert generalization rather than routing as the
failure. Version 2 added continuous source-paint comparison, normalized
position/area ranks, path-heavy training, and contour-worded instructions. A
final ordinal-size revision produced the source-disjoint v3 checkpoint.

On the untouched 59-case mixed holdout, v3 obtains `58/59` exact target sets
and exact patches: color `20/20`, position `27/27`, and relative size `11/12`.
The learned and oracle routers are identical (`20` attribute, `39` spatial),
so the remaining miss is a node-cardinality error rather than a routing error.

A matched no-edge control was trained with the same seed, examples, objective,
hidden width, and epochs; only the two graph-convolution layers were removed.
It obtains `51/59` exact target sets: color `20/20`, position `19/27`, and
relative size `12/12`. In the paired comparison, the GNN alone is correct on
eight cases and the MLP alone on one (two-sided exact McNemar `p = 0.0391`).
This isolates the useful contribution of graph message passing to relational
position grounding on this suite. It does not yet establish a benefit for
semantic or appearance-based references.

The hybrid architecture also obtains `500/500` exact patches and zero rendered
MSE on the five localized SVGEditBench tasks, with zero prompt tokens and zero
generative-model calls. This result must be interpreted structurally:
SVGEditBench color/contour instructions expose the source fill, so those cases
use exact-paint symbolic grounding, while the three whole-canvas tasks use the
existing deterministic compilers. It demonstrates safe system composition,
not semantic visual understanding by the GNN.

## Natural-transfer audit

[VectorEdits](https://arxiv.org/abs/2506.15903) contains 271,306 SVG edit
pairs and a 2,000-example collection-held-out test split. Its public dataset
card currently provides generated instructions only for the test split, so
this experiment is an evaluation audit rather than training on VectorEdits.
We retain only pairs whose source and target DOM topology is aligned, whose
changes touch drawable nodes, and which have at least three but not all
candidate nodes changed. The resulting 80 cases contain 64 Hicon and 16
Streetmix examples. Gold targets are derived from aligned DOM attribute
differences; they are proxy labels, not human node annotations.

Natural editing instructions mix the source reference with the requested
destination. For example, `left` in “rotate the arrow to point left” is an
edit parameter, not the arrow's current location. A deterministic decomposer
now extracts `the arrow` before grounding. On the natural subset this improves
the matched no-edge MLP from `12/80` to `30/80` oracle-cardinality exact cases
(`p = 0.00012`, paired exact McNemar) and from `40/80` to `57/80` top-1 hits.

| Grounder input | Top-1 in target | Exact top-k using gold k | Deployed threshold exact |
| --- | ---: | ---: | ---: |
| Graph-MoE, full instruction | 32/80 | 14/80 | 4/80 |
| Graph-MoE, target phrase | 35/80 | 19/80 | 2/80 |
| No-edge MLP, full instruction | 40/80 | 12/80 | 4/80 |
| No-edge MLP, target phrase | 57/80 | 30/80 | 1/80 |
| Frozen SigLIP 2, target phrase | 41/80 | 28/80 | n/a |

“Exact top-k using gold k” measures ranking quality while leaking the true
number of targets. The `1/80` deployable threshold result for the best scalar
arm exposes the next bottleneck: a semantic target frequently spans several
primitive paths, but the model has no group or cardinality objective.

The natural subset also reverses the synthetic GNN result. With target phrases,
the MLP is exact on 30 cases and the GNN on 19; the MLP-only/GNN-only split is
17/6 (`p = 0.0347`). Generic frozen
[SigLIP 2](https://arxiv.org/abs/2502.14786) candidate pooling is not a
universal replacement: it is worse on Hicon (`14/64` exact) but much stronger
on Streetmix (`14/16` versus `0/16`). The MLP and SigLIP successes are highly
complementary: their oracle union covers `55/80` cases.

An exploratory gate that uses SigLIP for SVGs with at least ten drawable
candidates and the MLP otherwise obtains `66/80` top-1 hits and `44/80`
oracle-cardinality exact cases. This is post-hoc and the cutoff also separates
the two collections perfectly, so it may be learning collection identity
rather than transferable complexity. The committed analysis marks the result
as exploratory; the threshold must be frozen before a collection-held-out
confirmatory run. This aligns with [SVGenius](https://arxiv.org/abs/2506.03139),
which stratifies 2,377 real-world SVG-editing queries by structural complexity
and reports degradation as SVG complexity increases.

## Interpretation guardrails

- The constructed position/size/color suite is a mechanism test, not evidence
  of natural-SVG generalization.
- Report learned-router and oracle-router results together.
- A pure scalar-feature graph cannot establish semantic visual understanding.
  Semantic claims require a frozen visual encoder, source-disjoint natural SVG
  data, and unseen-noun/paraphrase evaluation.
- The matched no-edge comparison supports message passing for this suite, but
  should be repeated on a larger frozen relational set with nested groups,
  overlaps, and references such as "left of" and "inside" before making a
  broader graph-learning claim.

## Recommended next experiment

The strongest next direction is a complexity- and intent-conditioned set
grounder, not a larger runtime VLM:

1. Split every instruction into `target_reference` and
   `edit_operation/parameters` before routing or node scoring.
2. Keep exact symbolic matching for explicit source attributes; compare a
   scalar MLP and relation GNN for simple references; invoke frozen visual
   features only for semantic, structurally complex cases.
3. Predict SVG groups and target cardinality jointly. Use DOM groups plus
   learned primitive affinity, then score both nodes and candidate groups so
   one conceptual object can map to multiple paths.
4. Distill offline per-node descriptions or grounding labels into the small
   selector. Do not call a generative VLM at inference time.
5. Freeze the routing rule and all thresholds, then evaluate on additional
   held-out collections and by complexity bucket. Include graph/no-edge,
   scalar/visual, and node/group ablations.

This preserves the benefit seen in the original visual-context experiment:
extra appearance evidence is available where it helps, without forcing it
into basic color and transparency cases where it created information overload.

## Set-grounding follow-up

The first follow-up tests whether multi-node edits can be recovered from
source-only structure. DOM subtrees, sibling groups, shared paint/tag groups,
and geometric containment cover `34/52` multi-target natural cases: `18/36`
Hicon and `16/16` Streetmix. This is an oracle coverage audit; it shows which
sets are representable, not which one a deployed model can choose.

An abstaining structural-group expert now handles two unambiguous source-side
cues before the learned fallback: all nodes nearest to an explicitly named
source paint, and all drawable nodes inside a named container. It also parses
CSS `rgb(...)` paint and coordinated source clauses. On the same exploratory
80-case subset, the group-plus-complexity system obtains `35/80` deployable
exact sets versus `19/80` for the previous threshold/top-1 fallback. The group
expert fires on 16 cases, is exact on all 16, and breaks no previously correct
case. Because the rules were designed after inspecting this subset, these
numbers require independent collection-held-out confirmation.

Version 4 adds a learned 1--6 target-cardinality head. Its training data adds
balanced, source-disjoint synthetic clusters containing one to six primitives;
the head consumes the instruction, pooled SVG features, and node-score
statistics, then selects the predicted top-k nodes. The best GNN checkpoint
reaches `379/400` (`94.75%`) exact sets and `400/400` correct cardinalities on
the synthetic identity-held-out validation split. This remains a mechanism
check until the frozen checkpoint is evaluated on new natural collections.
