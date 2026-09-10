# New-collection group selection: completed T4 experiment

**Initial run; superseded by a required implementation correction.** A final
audit found synthetic training candidates included `<g>` containers while the
natural manifest used drawables only. The corrected comparison uses identical
drawable units in both stages and is reported separately. The original data,
weights and results below are preserved; do not treat them as the final test
of group candidates.

**Adding structural group candidates did not improve exact-set accuracy over
the matched control.** Both trained heads scored 25/100, compared with 22/100
for the frozen MLP baseline. The three-case difference from baseline is
uncertain: the paired source-bootstrap 95% interval is **−3 to +9 percentage
points**. This experiment does not meet the proposed improvement milestone.

The experiment completed on a Tesla T4 with PyTorch 2.11.0+cu128. Both heads
have 23,425 parameters and trained for eight epochs on the same 1,980 synthetic
examples, with 400 source-disjoint validation examples. Both selected
checkpoints reached 96% synthetic validation exact-set accuracy. The new
natural holdout was read for scoring only after training and checkpoint
selection, and the source, protocol, checkpoint and labels were committed as
`92c8a98` before execution.

## Results on 100 new instructions

| Arm | Exact sets | Feather /50 | Tabler /50 | Precision | Recall | Mean selection latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen MLP + grouping rules | 22/100 | 11 | 11 | 43.70% | 39.75% | 22.9 ms |
| Matched singleton/top-k head | 25/100 | 13 | 12 | 43.88% | 39.75% | 24.2 ms |
| Head with structural group candidates | 25/100 | 13 | 12 | 42.55% | 38.58% | 34.3 ms |

Latency is a single-pass mean including scene/graph features and selection,
with shared base-model work included in each arm. It is not a dedicated
throughput benchmark. The grouping rules abstained on all 100 new cases, so
their large benefit on the old subset does not carry over to these references.

Group candidates cover the correct target set in 80/100 cases versus 72/100
for the control. These are **oracle candidate-coverage diagnostics**, not
deployable accuracy. The added coverage does not translate into successful
selection. Both trained heads get 21/55 single-node cases and only 4/45
multi-node cases exactly right. The baseline gets 18/55 and 4/45 respectively.
The small improvement over baseline is entirely in single-node cases.

The group head alone fixes `feather/monitor/2`; the control alone fixes
`feather/wifi/2`. Both are single-node targets. Group versus control has a
source-bootstrap interval of −3 to +3 percentage points. The three-point
baseline difference and zero-point control difference do not support adopting
the slower group head.

## What the failures say

| Failure category | Frozen baseline | Matched control | Group head |
| --- | ---: | ---: | ---: |
| Wrong object, no target overlap | 48 | 49 | 51 |
| Missing constituent nodes only | 16 | 14 | 13 |
| Extra nodes only | 9 | 6 | 5 |
| Both missing and extra nodes | 5 | 6 | 6 |
| Exact | 22 | 25 | 25 |

The old CPU run had 25 missing-part failures and seven wrong-object failures,
which motivated this experiment. On the new semantic part references,
wrong-object selection dominates instead. The next model experiment should
address semantic grounding with richer training references and visual/text
features, retaining this frozen baseline and avoiding tuning on these 100
labels. Merely making more groups available did not resolve the bottleneck.

## Scope and evidence

The holdout has 50 previously unscored natural SVG sources from Feather and
Tabler, with two authored instructions each. All 100 target sets were reviewed
in the source DOM and in rendered highlights **by the agent**, before scoring;
they are not independent human annotations. The data is curated, uses line
icons, and measures DOM target selection rather than full rendered edits.
Its rates are not directly comparable to the earlier 80-case Hicon/Streetmix
subset. The bootstrap resamples whole sources and is conditional on these two
collections. One target has eight nodes, beyond the old six-node count limit;
it remains in the denominator.

- [Frozen protocol](GROUP_CANDIDATE_PROTOCOL.md) and [holdout documentation](../data/group-holdout-v1/README.md).
- [Colab execution](https://colab.research.google.com/drive/1eNQXwRKMErvSvjt2nNFMa4thLb5Cm81o) and [pinned runnable notebook](../notebooks/group_candidate_cloud.ipynb).
- [Summary](../runs/group-candidate-v1/summary.json), [all 300 per-arm records](../runs/group-candidate-v1/records.json), [raw GPU log](../runs/group-candidate-v1/console_output.txt), and [provenance](../runs/group-candidate-v1/provenance.json).
- Both trained checkpoints are preserved as [prefix.pt](../runs/group-candidate-v1/prefix.pt) and [groups.pt](../runs/group-candidate-v1/groups.pt). Load on CPU with `torch.load(path, map_location='cpu', weights_only=True)`.

Validation: the full suite passed 274 tests with 18 skips and 642 subtests.
Two additional focused checks then passed for source-clustered uncertainty and
frozen artifact hashes. All 300 downloaded predictions match the frozen gold
manifest, metrics and bootstrap intervals recompute exactly, both downloaded
checkpoints load successfully, and the saved GPU console matches the result
archive. No holdout-based model revisions were made after scoring.
