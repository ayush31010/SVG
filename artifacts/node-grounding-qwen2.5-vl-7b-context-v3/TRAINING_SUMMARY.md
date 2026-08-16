# Context-v3 SVG node grounding result

Date: 2026-08-16

## Outcome

The Qwen2.5-VL node selector now distinguishes same-style SVG candidates by
semantic role, spatial relation, scale, and surrounding context. The fix is
not a larger checkpoint: it is a closed-choice grounding task with
candidate-specific visual evidence, stable-label leakage prevention,
permutation training, and LoRA coverage in the language, vision, and vision
projector paths.

On the matched 90-row held-out development slice, exact node-set selection
rose from **37/90 (41.1%)** for the untouched base model to **87/90 (96.7%)**
for the trained adapter. Cases that remained exactly correct under all three
shuffled A-F label assignments rose from **8/30 (26.7%)** to
**27/30 (90.0%)**. At the primary paired base-case level there were 19
improvements, no regressions, and a two-sided exact sign-test p-value of
3.81e-6.

These are conditional reranking results. The candidate builder injects gold
targets (`oracle_injected`), so candidate-retrieval rate is a construction
check and is not an end-to-end retrieval result.

## What changed

Each candidate is represented by a 224-pixel evidence card containing:

1. the complete SVG with that candidate highlighted;
2. a freshly rasterized contextual vector crop; and
3. a binary ownership mask.

The selector receives only shuffled labels plus label-keyed renderer metadata.
DOM IDs stay private and are mapped back after decoding. Runtime requires this
visual closed-choice route and fails closed if it is unavailable. Candidates
that are pixel-equivalent are collapsed before selection. A singleton pool is
resolved deterministically because there is nothing for a vision model to
compare.

The LoRA adapter is active only for the candidate-rerank schema. It is
temporarily disabled for downstream general SVG patch generation, preserving
the base model's broader instruction behavior.

## Training

- Base checkpoint: `Qwen/Qwen2.5-VL-7B-Instruct`
- Hardware: one available 141-GB NVIDIA GPU on `tatsumaki`
- Precision: bfloat16; no quantization
- Training slice: 400 rows / 200 base groups
- Batch size: 2 with gradient accumulation 2
- Duration: 2 epochs, 200 optimizer steps, 2,005 seconds
- Trainable parameters: 14,327,808 / 8,306,494,464 (0.1725%)
- LoRA rank/alpha/dropout: 16 / 32 / 0.05
- Matched target modules: 112 language-attention, 64 vision-attention,
  2 vision-projector modules
- Best checkpoint: epoch 1, validation loss 0.004494695
- Epoch 2 validation loss: 0.007819439
- Best adapter SHA-256:
  `dc497b620d9cc22d36b541c481e0097941caa12f6cb3aeb5fcbe8c131dfddc49`

The saved root adapter is byte-identical to the epoch-1 checkpoint selected by
validation loss. The epoch-2 checkpoint is retained separately for audit.

Validation used 89 rows / 31 complete permutation groups. It achieved 88/89
exact rows (98.9%), 99.1% micro-F1, and 30/31 groups correct under every
available label permutation (96.8%).

## Data integrity

The generated context-v3 corpus contains 782 train rows, 122 validation rows,
and 153 test rows. All 1,057 image paths are unique and present. Split
assignment uses connected components over both source identity and source SVG
SHA-256, preventing byte-identical art from crossing splits. Generation
asserts zero train/validation/test source-hash overlap.

Training uses two shuffled label assignments per base case; validation and
test use up to three complete assignments. Cyclic seeded permutations ensure
every candidate label moves, and evaluation maps choices back to node IDs
before scoring. All held-out candidate pools contain at least one non-gold
distractor.

The earlier source-leaked run is quarantined under a directory explicitly
named `base-test-leaked-source-split-20260816` and is excluded from every
number in this report.

## Matched held-out development evaluation

The configured evaluator caps the manifest's 51 written test base cases at 90
rows, yielding 30 complete base cases x 3 label permutations. The slice has 30
SVGEditBench-derived rows and 60 synthetic-context rows. It contains 18 unique
source SHA-256 clusters, so the 90 rows must not be treated as independent.

| Metric | Base | Trained | Change |
|---|---:|---:|---:|
| Exact node set | 37/90 (41.1%) | 87/90 (96.7%) | +55.6 pp |
| Micro-F1 | 49.2% | 97.5% | +48.3 pp |
| Top-1 node | 47/90 (52.2%) | 88/90 (97.8%) | +45.6 pp |
| Cardinality | 77/90 (85.6%) | 89/90 (98.9%) | +13.3 pp |
| All permutations exact | 8/30 (26.7%) | 27/30 (90.0%) | +63.3 pp |
| Node permutation consistency | 8/30 (26.7%) | 27/30 (90.0%) | +63.3 pp |
| Synthetic-context exact | 14/60 (23.3%) | 58/60 (96.7%) | +73.3 pp |
| SVGEditBench-derived exact | 23/30 (76.7%) | 29/30 (96.7%) | +20.0 pp |
| Two-target exact | 0/12 (0.0%) | 11/12 (91.7%) | +91.7 pp |

The evaluation-input identity SHA-256 is
`0c7738b8ca249a94693ad635251e126911f70ee456c0fea846e0f297b790e530`.
An independent audit reproduced every stored metric and confirmed identical
row order, instructions, mappings, schemas, targets, label permutations, and
source hashes between base and trained conditions.

## Complete test manifest (2026-08-17)

The capped slice above has since been superseded by an uncapped run over all
51 written test base cases (153 rows x 3 label permutations), executed on an
NVIDIA H100 NVL with the same pinned stack. **The capped slice was not
flattering: the trained result holds, and the measured gain is larger.**

| Metric | Base | Trained | Change |
|---|---:|---:|---:|
| Exact node set | 59/153 (38.6%) | 147/153 (96.1%) | +57.5 pp |
| Micro-F1 | 47.2% | 96.8% | +49.6 pp |
| Top-1 node | 76/153 (49.7%) | 148/153 (96.7%) | +47.1 pp |
| Cardinality | 127/153 (83.0%) | 152/153 (99.3%) | +16.3 pp |
| Strict JSON | 148/153 (96.7%) | 153/153 (100%) | +3.3 pp |
| All permutations exact | 11/51 (21.6%) | 46/51 (90.2%) | +68.6 pp |
| SVGEditBench-derived exact | 34/54 (63.0%) | 53/54 (98.1%) | +35.1 pp |
| Synthetic-context exact | 25/99 (25.3%) | 94/99 (94.9%) | +69.6 pp |
| Two-target exact | 0/21 (0.0%) | 20/21 (95.2%) | +95.2 pp |

Against the earlier 90-row slice, trained exact-set moves 96.7% -> 96.1% and
all-permutation consistency 90.0% -> 90.2%; both are unchanged within the
resolution of the sample. The base model scores *lower* on the full manifest
(41.1% -> 38.6%), so the paired base-case gain rises from 63.3 to 68.6 points.

At the paired base-case level: 35 improvements, **0 regressions**, two-sided
exact sign-test p = 5.82e-11. At the row level: 88 trained-only correct, 0
base-only, p = 6.46e-27. Across the 20 source-SHA-256 clusters, base is fully
correct on 3 and trained on 15, with 12 clusters improved and none regressed.

Ablations on the full manifest agree with the slice: blanking the raster drops
exact selection to 125/153 (81.7%, -14.4 pp) and blanking the instruction to
5/153 (3.3%, -92.8 pp).

Evidence is under `evaluation/full-manifest/`, including both conditions'
per-row predictions, metrics, run logs, and `comparison-full.json`. Every
ablation covered all 153 rows in both conditions (918 generations); the
adapter's SHA-256 was verified on the evaluation host before the run.

This remains a **conditional reranking** result: candidates are still
oracle-injected, so end-to-end candidate retrieval is untouched by it.

## Ablations

- Blank instruction: exact selection falls to 3/90 (3.3%), a 93.3-point drop.
- Blank raster image: exact selection falls to 72/90 (80.0%), a 16.7-point
  drop; all-permutation exact falls from 27/30 to 19/30.

The blank-image condition removes raster pixels but deliberately retains
label-keyed tag, geometry, style, area, depth, and structure metadata. It
therefore measures incremental pixel value, not the removal of every visual or
structured context cue. The pixel drop occurs in synthetic cases; the small
SVGEditBench-derived slice does not establish universal pixel dependence.

## Live deployment smoke

The generic five-case CLI smoke completed with valid model responses and
preserved protected geometry, but its editable cases collapsed to singleton
pools. It verifies general wiring, not candidate comparison.

A separate direct architecture smoke uses deterministic held-out ambiguous
SVGs spanning face, robot, flower, grid, and two-target selection. It records
the exact visual evidence PNG, prompt, response schema, adapter state, chosen
node IDs, expected node IDs, output SVG, and patch-generation response for
each case. Its machine-readable result is under
`evaluation/runtime-targeted/summary.json`.

All five cases entered `visual_closed_choice`, all five returned strict valid
JSON, all five selected the exact expected node set, and all five completed the
architecture without error. The selection request used the trained adapter in
every case; the downstream general patch request had the adapter disabled in
every case. The smoke includes a two-target instruction and selected both
expected nodes.

## Deployment

The default Hugging Face model configuration points to:

```text
artifacts/node-grounding-qwen2.5-vl-7b-context-v3/adapter
```

The bundle includes the best adapter and processor, the epoch-1 safety copy,
data manifests and compressed evidence images, train/base/trained logs,
matched predictions and metrics, live runtime evidence, a source snapshot,
and SHA-256 checksums.

## Claim boundaries

- The complete 51-case test manifest has now been evaluated (see above), so
  the development-slice caveat no longer applies to the headline number. This
  is still not an official zero-shot SVGEditBench result.
- Synthetic probes account for 99/153 rows on the full manifest, and the
  153 rows resolve to only 20 source-SHA-256 clusters, so they are not
  independent samples.
- Candidate targets are oracle-injected before reranking; production candidate
  recall remains a separate problem.
- The base Hugging Face model name is recorded, but an immutable Hub revision
  was not pinned in the original run.
- Larger and natural held-out corpora are still needed for a publication-grade
  generalization claim.
