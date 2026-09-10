# New-collection group selection: corrected T4 experiment

**Adding structural group candidates did not improve exact-set accuracy.** In
the corrected drawable-only comparison, the group head scores 23/100 against
27/100 for the matched singleton/top-k control and 22/100 for the frozen MLP
baseline. The group head's paired source-bootstrap 95% interval against the
control is **−9 to +1 percentage points**, and against the baseline −3 to +5.
The control's +5 over the baseline has an interval of −1 to +11 and comes
entirely from single-node targets. All three arms get exactly 4/45 multi-node
targets right. This experiment does not meet the proposed improvement milestone.

This is the second look at the holdout. The [first run](#first-run-superseded)
trained on candidate lists that included `<g>` containers while the natural
manifest used drawable nodes only; the [protocol](GROUP_CANDIDATE_PROTOCOL.md)
records the correction. The rerun kept the same manifest, frozen baseline,
features, seed, hyperparameters and budget, and changed only the training-time
candidate units. Because the repair followed the first scores, treat the
corrected numbers as the reported result but not as a single-look confirmatory
test.

The corrected run executed on a Tesla T4 with PyTorch 2.11.0+cu128 from source
commit `81bf9df`, pinned by the runner at `228e263`. Training and evaluation
took 283 seconds. Both heads have 23,425 parameters and trained for eight
epochs on the same 1,980 synthetic examples with 400 source-disjoint validation
examples. Validation exact-set accuracy selected epoch 8 for the control
(96.25%) and epoch 6 for the group head (96.0%). The natural holdout was read
for scoring only after training and checkpoint selection.

## Results on 100 new instructions

| Arm | Exact sets | Feather /50 | Tabler /50 | Precision | Recall | Mean selection latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen MLP + grouping rules | 22/100 | 11 | 11 | 43.70% | 39.75% | 23.3 ms |
| Matched singleton/top-k head | 27/100 | 15 | 12 | 44.05% | 40.08% | 24.7 ms |
| Head with structural group candidates | 23/100 | 12 | 11 | 41.38% | 36.75% | 34.9 ms |

Latency is a single-pass mean including scene/graph features and selection,
with shared base-model work included in each arm. It is not a dedicated
throughput benchmark. The grouping rules abstained on all 100 new cases, so
their large benefit on the old subset does not carry over to these references.

| Paired comparison | Left-only correct | Right-only correct | Source-bootstrap mean | 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Group head vs control | 3 | 7 | −4 pts | −9 to +1 |
| Group head vs baseline | 4 | 3 | +1 pt | −3 to +5 |
| Control vs baseline | 11 | 6 | +5 pts | −1 to +11 |

The bootstrap resamples whole sources (5,000 draws) so both instructions per
SVG stay together. The control-versus-baseline row is recomputed from the saved
records with the same procedure; the other two are in the summary. Per-case
McNemar values (group vs control p = 0.34, group vs baseline p = 1.0) are
descriptive only because cases share sources.

Group candidates cover the correct target set in 80/100 cases versus 72/100
for the control. These are **oracle candidate-coverage diagnostics**, not
deployable accuracy, and they are unchanged from the first run because the
evaluation candidates were already drawable-only. The added coverage does not
translate into successful selection. Single-node exact sets are 18/55 for the
baseline, 23/55 for the control and 19/55 for the group head; every arm gets
4/45 multi-node targets.

The group head alone fixes `feather/camera/1`, `feather/cloud-lightning/1` and
`feather/wifi/2`; the control alone fixes `feather/bell/1`, `feather/camera/2`,
`feather/clock/2`, `feather/cloud-lightning/2`, `feather/monitor/2`,
`feather/save/2` and `tabler/car/2`. All ten are single-node targets. Offering
structural groups made the selector slower and less precise without recovering
any additional complete object.

## What the failures say

| Failure category | Frozen baseline | Matched control | Group head |
| --- | ---: | ---: | ---: |
| Wrong object, no target overlap | 48 | 50 | 52 |
| Missing constituent nodes only | 16 | 13 | 14 |
| Extra nodes only | 9 | 5 | 5 |
| Both missing and extra nodes | 5 | 5 | 6 |
| Exact | 22 | 27 | 23 |

The old CPU run had 25 missing-part failures and seven wrong-object failures,
which motivated this experiment. On the new semantic part references,
wrong-object selection dominates instead. The next model experiment should
address semantic grounding with richer training references and visual/text
features, retaining this frozen baseline and avoiding tuning on these 100
labels. Merely making more groups available did not resolve the bottleneck.

## First run (superseded)

The initial run from source commit `92c8a98` scored 25/100 for both trained
heads and 22/100 for the baseline (group vs control interval −3 to +3; group
vs baseline −3 to +9). A final audit found that the synthetic training
candidates included `<g>` containers, so the heads had trained on candidate
units that the natural evaluation never offered. Its data, weights and results
are preserved under [`runs/group-candidate-v1`](../runs/group-candidate-v1/)
with their own [provenance](../runs/group-candidate-v1/provenance.json); do not
treat them as the test of group candidates.

## Scope and evidence

The holdout has 50 previously unscored natural SVG sources from Feather and
Tabler, with two authored instructions each. All 100 target sets were reviewed
in the source DOM and in rendered highlights **by the agent**, before scoring;
they are not independent human annotations. The data is curated, uses line
icons, and measures DOM target selection rather than full rendered edits.
Its rates are not directly comparable to the earlier 80-case Hicon/Streetmix
subset. The bootstrap is conditional on these two collections. One target has
eight nodes, beyond the old six-node count limit; it remains in the denominator.

- [Frozen protocol](GROUP_CANDIDATE_PROTOCOL.md) and [holdout documentation](../data/group-holdout-v1/README.md).
- [Corrected Colab execution](https://colab.research.google.com/drive/1JxCXj97Cm2J6g9OTbhqqrlj-BwWJi5sc) and [pinned runnable notebook](../notebooks/group_candidate_cloud.ipynb).
- [Summary](../runs/group-candidate-v1-fixed/summary.json), [all 300 per-arm records](../runs/group-candidate-v1-fixed/records.json), [raw GPU log](../runs/group-candidate-v1-fixed/console_output.txt), [executed notebook](../runs/group-candidate-v1-fixed/executed_notebook.ipynb) and [provenance](../runs/group-candidate-v1-fixed/provenance.json).
- Both trained checkpoints are preserved as [prefix.pt](../runs/group-candidate-v1-fixed/prefix.pt) and [groups.pt](../runs/group-candidate-v1-fixed/groups.pt). Load on CPU with `torch.load(path, map_location='cpu', weights_only=True)`.

Validation of the corrected artifacts: all 300 downloaded predictions match the
frozen gold manifest and its candidate lists, the manifest and baseline
checkpoint still match the hashes in the run configuration, metrics and both
saved bootstrap intervals recompute exactly from the records, both downloaded
checkpoints load with finite weights, and the executed notebook's console names
source commit `81bf9df` and reproduces the downloaded summary byte for byte.
No holdout-based model revisions were made after scoring.
