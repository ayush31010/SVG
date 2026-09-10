# Frozen SigLIP semantic prior on the new-collection holdout

**A training-free semantic prior is the first arm to beat the frozen baseline
on unseen collections.** Fusing the frozen v4 MLP's node logits with frozen
SigLIP 2 similarities reaches **39/100** exact target sets on the Feather/Tabler
holdout, against 22/100 for the frozen MLP + grouping rules and 27/100 for the
best previous system, the trained singleton/top-k control head. The paired
source-bootstrap 95% intervals are **+10 to +24 points** over the baseline and
**+5 to +19** over the control. Wrong-object failures fall from 48 to 28. The
gain is entirely in single-node targets (18 → 35/55); multi-node targets stay
at 4/45 because the frozen count head predicts one node in 71 of 100 cases.
With the gold set size, the fused ranking would be exact on 60/100.

This meets the milestone set after the group-candidate experiment: an
improvement over the frozen baseline on collections nobody tuned on. It also
identifies the next bottleneck precisely: cardinality, not ranking.

## What was frozen, and when

Nothing trains. The arm scores each drawable candidate with

    (1 − α) · logit(p_MLP) + α · τ · cos(SigLIP image, SigLIP text)

where `p_MLP` is the frozen MLP's node probability, `cos` is the frozen
[SigLIP 2](https://arxiv.org/abs/2502.14786) `base-patch16-224` similarity
between the instruction's target phrase and the candidate's rendered views,
and `τ = 112.67` is SigLIP's own learned logit scale, read from the model at
run time rather than tuned. Views and their weights (isolated 0.5, local crop
0.3, highlighted context 0.2) are the defaults from the earlier 80-case work.
The selector is unchanged: rank, take the frozen count head's `k`, and keep
the grouping-rule override (which abstained on all 100 holdout cases).

The single free parameter, `α`, was chosen on the 80-case Hicon/Streetmix
development set using its cached SigLIP scores, with a rule written before the
sweep: most deployed exact sets, ties to oracle-cardinality exact, then the
smallest weight. `α = 0` reproduces the protocol's 41/80 for the frozen
baseline, confirming the pipeline. The sweep peaks at 44/80 for `α = 0.7`,
which was frozen in [`configs/eval/semantic_prior_v1.json`](../configs/eval/semantic_prior_v1.json)
and committed (`a032c36`) before the holdout stage, which refuses to run
without a frozen weight. The config carries SHA-256 hashes of the checkpoint,
manifest, development parquet and cached scores.

| `α` (dev, 80 cases) | 0.0 | 0.2 | 0.5 | **0.7** | 0.8 | 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Deployed exact sets | 41 | 42 | 43 | **44** | 43 | 23 |
| Exact with gold `k` | 38 | 48 | 59 | 57 | 52 | 28 |
| Top-1 in gold | 68 | 77 | 76 | 70 | 64 | 41 |

The dev plateau is flat from 0.5 to 0.8; 0.7 sits inside it. The full sweep is
in [`dev_sweep.json`](../runs/semantic-prior-v1/dev_sweep.json).

Disclosure. This is the third scoring of the holdout, after the two
candidate-set runs. Before designing the fusion, the author inspected fourteen
holdout records while diagnosing the earlier failures; the fusion form, the
selection rule and the SigLIP defaults were not altered after that or after
seeing any holdout score. A cardinality follow-up was explored on the
development set only and **not** applied to the holdout (below).

## Results on 100 new instructions

| Arm | Exact sets | Feather /50 | Tabler /50 | Single /55 | Multi /45 | Precision | Recall | Exact with gold `k` | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen MLP + grouping rules | 22 | 11 | 11 | 18 | 4 | 43.7% | 39.8% | 35 | 5 ms |
| Trained top-k control head (prior run) | 27 | 15 | 12 | 23 | 4 | 44.0% | 40.1% | – | 25 ms |
| SigLIP only (`α = 1`) | 37 | 21 | 16 | 36 | 1 | 60.2% | 61.4% | 57 | 299 ms |
| **Fused (`α = 0.7`)** | **39** | 22 | 17 | 35 | 4 | 61.1% | 59.2% | 60 | 299 ms |

Latency is a single-pass mean on an Apple M-series GPU including rendering
three 224-px views per candidate and SigLIP inference; the MLP alone is
5 ms. The control head's numbers are copied from its own run for pairing.

| Paired comparison | Left-only correct | Right-only correct | Source-bootstrap mean | 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Fused vs frozen baseline | 19 | 2 | +17 pts | +10 to +24 |
| Fused vs trained control | 18 | 6 | +12 pts | +5 to +19 |
| SigLIP only vs frozen baseline | 25 | 10 | +15 pts | +5 to +25 |
| Fused vs SigLIP only | 10 | 8 | +2 pts | −7 to +11 |

The bootstrap resamples whole sources (5,000 draws) so both instructions per
SVG stay together. Fused and SigLIP-only are not distinguishable; the fusion
mainly preserves the MLP's four multi-node hits (SigLIP alone keeps one) and
trades single-node cases both ways.

All nineteen cases the fusion fixes are single-node semantic references the
hash text encoder could not resolve: `the lightning bolt`, `the bell body`,
`the camera body outline`, `all the radiating curved waves`, `the wavy water
line`. The two
it loses are `feather/watch/1` (`the watch hands`, a small L-shape inside a
circle) and `feather/wifi/2` (`the small dot below the arcs`), both spatial
descriptions of tiny marks that SigLIP's crops do not separate.

## What the failures say now

| Failure category | Frozen baseline | Trained control | SigLIP only | Fused |
| --- | ---: | ---: | ---: | ---: |
| Wrong object, no target overlap | 48 | 50 | 26 | 28 |
| Missing constituent nodes only | 16 | 13 | 14 | 16 |
| Extra nodes only | 9 | 5 | 14 | 11 |
| Both missing and extra nodes | 5 | 5 | 9 | 6 |
| Exact | 22 | 27 | 37 | 39 |

Twenty-one fused cases have the correct ranking at the gold size but the
wrong `k`: the count head returns 1 for six two-node targets and for targets
of three, four and six nodes, and returns 5 or 6 for four single-node
targets. Of the 45 multi-node targets, the fused ranking is exact for 21 with
the gold size and for 4 with the count head. The remaining wrong-object
failures are mostly outlines and enclosing shapes (`the outer clock face
circle`, `the outer floppy disk boundary`, `the mouse body outline`), where
the highlighted context view of the outline and of its interior look alike.

## Cardinality: explored on the development set, not applied

Because the count head is now the limiting factor, several selection rules
were scored on the development set with `α` fixed: score-gap selection on
the fused score or on the SigLIP cosine, the larger of the count head and the
gap set, and the count head's rounded expected size. The best gains three dev
cases (47 versus 44 of 80; oracle 57). The count head predicts one node in
65 of the 80 dev cases while 52 have multi-node gold sets, the same
miscalibration as on the holdout. Text plurality cannot be validated on this
development set: its DOM-difference labels make `the clock hands` a single
path and `the arrow` three paths. No rule was applied to the holdout; the
variants and their dev scores are in
[`dev_cardinality_exploration.json`](../runs/semantic-prior-v1/dev_cardinality_exploration.json).

A cardinality fix needs a signal validated on natural, node-level labels that
are neither this holdout nor DOM-diff proxies: a separately authored
development set from further collections with explicit counts, or synthetic
instructions that carry count cues. That is the next experiment; it should
keep this fused arm, the frozen baseline and these 100 labels untouched.

## Scope and evidence

The holdout, its authorship and its limits are as described for the
[candidate-group experiment](GROUP_CANDIDATE_RESULTS.md): 50 previously
unscored line-icon sources, two agent-authored instructions each, reviewed but
not independently human-labeled, scoring DOM target selection only. SigLIP
adds a 60× latency cost and a frozen model whose download is about 1.4 GB. The result is conditional
on line icons whose parts render legibly in isolation; the development set
already showed SigLIP alone losing to the MLP on Hicon's dense icons, which is
why the fusion, not SigLIP alone, is the reported arm.

- [Frozen config](../configs/eval/semantic_prior_v1.json), [runner](../scripts/run_semantic_prior_grounding.py) and [tests](../tests/test_semantic_prior_grounding.py).
- [Summary](../runs/semantic-prior-v1/summary.json) with all paired comparisons, [all 400 per-arm records](../runs/semantic-prior-v1/records.json) with every MLP, per-view SigLIP and fused score, and [provenance](../runs/semantic-prior-v1/provenance.json).
- Rendering on this machine used the new `resvg-py` fallback in `svgpatchlab.eval.render`; CairoSVG remains first when native Cairo exists.

Validation: the baseline arm's 100 selections are identical to the corrected
candidate-set run's baseline; the frozen checkpoint, manifest and cached dev
scores hash-match their earlier runs; SigLIP's loaded logit scale matched the
frozen `τ`; the grouping rules abstained on all 100 cases in every arm.
