# Frozen candidate-set experiment v1

Freeze point: before any new holdout model scores are inspected.

The earlier CPU MLP plus grouping system has 41/80 exact sets, 25 missing-part
failures, seven wrong-object failures, four mixed failures and three extra-part
failures. This motivates testing complete group candidates.

## Arms and matched budget

1. Frozen CPU v4 MLP plus the unchanged abstaining grouping rules.
2. The same MLP plus a learned candidate-set head receiving every singleton
   and node-score prefixes of length one through six.
3. An identically sized candidate-set head additionally receiving the existing
   source-derived DOM, paint, tag and containment groups.

Both heads have the same initialization seed, 64 hidden units, learning rate,
eight epochs, example order, features and loss. They use the exact v4 synthetic
source split and the same frozen MLP. Only the available candidate sets differ.
The frozen baseline receives no new training; the prefix head is the matched
training-budget control. All arms retain the same grouping-rule override.

Candidate features combine instruction encoding, pooled node features, node
scores, set size and the frozen count probabilities. Training distributes
cross-entropy supervision uniformly over the available candidates with maximum
gold-set F1. No gold group is inserted into the candidate list. Validation
exact-set accuracy selects the checkpoint, with the earliest epoch winning
ties. The model must tolerate targets absent from its candidate family.

## Evaluation

Implementation correction after the first run: candidate IDs must be drawable
nodes in both synthetic training and natural evaluation. The first run used
all graph nodes during training, inadvertently including `<g>` containers.
The corrected rerun retains the same manifest, baseline, features, seed,
hyperparameters and budget, with a regression test requiring synthetic DOM
groups to match drawable-only gold sets. Both runs are retained. Because this
repair follows the first scores, report that history rather than describing
the corrected run as an untouched single-look confirmatory test.

The primary endpoint is exact target-set accuracy on the frozen 100-case,
50-source Feather/Tabler holdout. Also report precision, recall, per-collection
results, candidate coverage, missing/extra/wrong-object categories and selection
latency including feature extraction. Shared sources require source-clustered
uncertainty; any per-case McNemar output is descriptive only. Report all three
arms regardless of direction. Do not tune on this holdout after scoring.

The source checkpoint, manifest and upstream sources carry hashes. No new
natural labels enter training or checkpoint selection. Instructions and gold
sets are agent-authored and visually reviewed, not independently human labeled.
Any resulting claim is limited to this curated, previously unscored collection
holdout. An independent human review remains useful before a publication claim.
