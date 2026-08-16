# Context-v3 closed-choice SVG node grounding

`train.node_grounding_sft` trains a vision-language model to map an edit
instruction to a closed set of visible candidate labels. The model returns
canonical JSON such as:

```json
{"choices":["B","E"]}
```

The application keeps the corresponding DOM node IDs private and maps the
selected letters back to nodes after decoding. Context-v3 is a generative SFT
pipeline: prompt and image tokens are masked, and language-model loss applies
only to the assistant JSON.

This pipeline addresses the failure mode in which a model recognizes a shape
but cannot distinguish which same-looking SVG node plays the requested role.
It does not claim that model size alone solves grounding, and this document
contains no final accuracy claim. Record metrics only after running the honest
base and trained evaluations described below.

## Model-visible evidence

Every candidate card in the training and runtime evidence sheet has the same
three panels:

1. **Full context** - the complete source rendering with the candidate
   highlighted, preserving its location and relationship to surrounding parts.
2. **Direct vector crop** - a fresh SVG rasterization after changing the root
   `viewBox` to the padded candidate region. It is not an enlarged crop from a
   low-resolution whole-scene bitmap.
3. **Ownership mask** - an opaque, high-contrast mask of candidate coverage,
   which remains legible for pale, transparent, small, or occluded artwork.

The renderer also retains an isolated vector crop as a separately consumable
artifact, although the composed SFT sheet displays the three panels above.
Each card includes a compact caption. The prompt can additionally receive
renderer-derived, label-keyed fields such as normalized center and bounding
box, painted-area fraction, tag, depth, child/descendant counts, fill, stroke,
and effective opacity.

Stable DOM IDs never appear in the sheet, caption, prompt, or response schema.
They remain only in `choice_to_node` metadata. Generated rows store
`renderer: "candidate_evidence_sheet"` and use format
`svgpatchlab.node_grounding.v3`; older manifests must be regenerated.

Within `render_candidate_evidence`, the full scene is shared and each candidate
costs three additional rasterizations: a full-canvas isolation for location,
a direct contextual vector crop, and a direct isolated vector crop. This is a
deliberate fidelity tradeoff, especially for small components.

## Data construction

### SVGEditBench-derived cases

Gold node IDs come from the repository's deterministic patch differ applied to
each source/answer pair. In the default `hard_negative` policy, generation:

- includes every gold target;
- ranks same-tag, same-paint, structurally nearby nodes as distractors;
- produces between two and `candidate_count` choices; and
- renders the final candidate order into the evidence sheet.

Because this policy injects gold targets, its manifest labels retrieval as
`oracle_injected`. Its candidate-retrieval rate is a data-construction check,
not an end-to-end retrieval result.

### Synthetic semantic and spatial scenes

The context-v3 H200 preset adds `synthetic_context_scenes: 100`. Each generated
scene contributes three instructions sampled deterministically from one of four
scene families:

- face parts such as left/right eye, mouth below the nose, and hat above face;
- robot roles such as eye, wheel, head, and paired bottom wheels;
- flower parts distinguished by top/lower-left/right position or center role;
- same-style circle grids distinguished by row, column, relative position, or
  size.

Candidates deliberately repeat visual styles, so source-color lookup alone is
insufficient. Targets require semantic role, absolute position, scale, or a
relation to another object. Rows record `source_family` as either
`synthetic_context` or `svg_edit_bench`, and evaluation reports both families
separately.

Synthetic scenes are controlled probes, not substitutes for a large natural
SVG corpus. A publication-grade claim still requires held-out natural assets
whose referring expressions demand the same distinctions.

## Split and label integrity

The split unit is the source identity:

- all SVGEditBench tasks for one `emoji_id` stay in one split;
- all instructions for one synthetic context scene stay in one split; and
- validation/test source identities do not appear in training.

`label_permutations` renders several deterministic, unique A-F assignments for
the same base case. Context-v3 uses two permutations for training and three for
validation and test. The labels are repainted into each image; JSON metadata is
never permuted independently of the pixels.

Rows share a `base_id` across permutations. Evaluation maps predicted letters
back to nodes and reports:

- node-level cross-permutation consistency; and
- whether every permutation of a base case is exactly correct.

This prevents a fixed preference for A, B, or early DOM order from looking like
context understanding.

## Honest candidate requirements and coverage

With `honest_eval_require_distractor: true`, every validation and test row must
contain at least one non-gold candidate. Target-only pools are excluded because
they measure output cardinality rather than discrimination. Training rows may
still use the oracle-injected hard-negative policy, but held-out scoring fails
closed if a target-only row reaches evaluation.

`manifest.json` records, per split:

- source and targeted base cases;
- candidate pools actually audited;
- cases retrieving every gold target;
- cases with at least one non-gold distractor;
- retrieval rate among audited pools; and
- retrieval coverage over all targeted cases.

The optional `production` candidate policy reuses the inference shortlisting
helpers without injecting, truncating, or adding gold nodes. It reports empty,
singleton, oversized, missing-gold, cardinality-ineligible, and
`production_no_non_gold_distractor` exclusions separately. Use this policy to
measure production candidate recall, not the oracle-injected training set.

SVGEditBench color and contour prompts commonly name the source paint value.
A shortlist containing only matching gold-colored nodes is therefore not
evidence of context awareness and is rejected by honest evaluation.

## LoRA coverage and verification

Qwen2.5-VL uses different projection names in its language and vision paths.
Context-v3 resolves full module names against the loaded model and requires
all of these LoRA families:

- `language_attention`: decoder `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
- `vision_attention`: vision-block `qkv` and `proj`; and
- `vision_projector`: both linear layers of `visual.merger.mlp`.

Generation stops before training if a required regex family matches no module.
After PEFT attachment, it independently verifies that every family owns
trainable `lora_*` parameters. Counts are printed and saved to
`lora_module_report.json`. A checkpoint or Transformers naming change cannot
silently leave the vision tower or projector frozen.

The H200 context-v3 preset uses bfloat16 LoRA (`qlora: false`). The general
`node_grounding_sft.json` preset remains available for lower-memory QLoRA.

## Dependencies

Use Python 3.10+ and install:

```bash
python -m pip install -r requirements-node-grounding.txt
```

This includes the raster stack plus Transformers, Torch, Accelerate, PEFT,
BitsAndBytes, and Safetensors. CairoSVG and Pillow are required for evidence
generation. BitsAndBytes is required only when `qlora` is enabled.

## Context-v3 run sequence

### 1. Generate the v3 dataset

```bash
python -m train.node_grounding_sft \
  --config configs/train/node_grounding_context_h200.json \
  --generate-data-only
```

Generation refuses to replace an existing manifest. Use `--overwrite` only
when deliberately regenerating older data. Outputs are written under:

```text
data/node_grounding-context-v3/
  manifest.json
  train.jsonl
  val.jsonl
  test.jsonl
  images/{train,val,test}/*.png
```

Inspect the manifest coverage, several sheets from every source family, label
mappings across permutations, and skipped-case counters before training.

### 2. Run a training smoke test

The smoke config reads the generated context-v3 data and writes to an isolated
checkpoint directory:

```bash
python -m train.node_grounding_sft \
  --config configs/train/node_grounding_context_h200_smoke.json
```

This verifies model loading, all three LoRA families, the multimodal collator,
one short training pass, and greedy validation.

### 3. Train the full context-v3 adapter

```bash
python -m train.node_grounding_sft \
  --config configs/train/node_grounding_context_h200.json
```

The preset uses Qwen2.5-VL-7B, 224-pixel candidate panels, a larger visual
pixel budget, deterministic permutations, gradient checkpointing, and
bfloat16 LoRA on language, vision, and projector modules. The best
validation-loss adapter and processor are saved under:

```text
checkpoints/node-grounding-qwen2.5-vl-7b-context-v3/
```

Training also writes `validation_metrics.json`, validation predictions, and
the LoRA module verification report. These are development outputs, not final
test results.

### 4. Evaluate the untouched base model

```bash
python -m train.node_grounding_sft \
  --config configs/train/node_grounding_context_h200_base_eval.json \
  --eval-only
```

`evaluation_checkpoint: "base"` loads the base Qwen checkpoint without the
trained adapter. This preset scores the honest test split in the same three
clean/ablation conditions as the trained run and writes predictions plus
`test_metrics.json` under `runs/node-grounding-context-v3-base-test/`.

### 5. Evaluate the trained adapter and ablations

```bash
python -m train.node_grounding_sft \
  --config configs/train/node_grounding_context_h200_eval.json \
  --eval-only
```

This loads the trained adapter and runs:

- `none`: the complete instruction plus evidence sheet;
- `blank_image`: instruction and label-keyed geometry/style metadata retained,
  while image pixels are replaced with white; and
- `blank_instruction`: evidence retained, instruction text withheld.

Predictions are written as `test_predictions.jsonl`,
`test_predictions.blank_image.jsonl`, and
`test_predictions.blank_instruction.jsonl` under the configured evaluation
directory. It also saves the complete aggregate as `test_metrics.json` and
prints the same metrics JSON. Keep the base and trained output directories
separate when comparing systems.

### 6. Evaluate the complete uncapped test manifest

Steps 4 and 5 carry `max_eval_samples: 90`, which scores 30 of the manifest's
51 written test base cases. The `*_full_eval` presets set the cap to `null` so
`_subset` returns every row, covering all 51 base cases and 153 rows:

```bash
python -m train.node_grounding_sft \
  --config configs/train/node_grounding_context_h200_base_full_eval.json \
  --eval-only

python -m train.node_grounding_sft \
  --config configs/train/node_grounding_context_h200_full_eval.json \
  --eval-only
```

These write to `runs/node-grounding-context-v3-{base,trained}-test-full/`, so
they never overwrite the capped run directories. Both presets keep
`honest_eval_require_distractor: true`; the manifest records a non-gold
distractor for all 51 test base cases, so the full split satisfies it.

Budget roughly 3x the capped run: 153 rows x 3 ablations x 2 conditions is 918
greedy generations, plus two 7B model loads.

Then build the matched comparison:

```bash
python -m scripts.compare_node_grounding_runs \
  --base runs/node-grounding-context-v3-base-test-full \
  --trained runs/node-grounding-context-v3-trained-test-full \
  --data-dir data/node_grounding-context-v3 \
  --output runs/node-grounding-context-v3-comparison-full.json
```

The script refuses to compare runs whose evaluation inputs differ row for row,
reuses `score_predictions` for every metric, and reports the paired base-case
analysis as primary because label-permutation rows are correlated. Its
`identity_sha256` uses its own documented canonicalisation and is not
comparable to the hand-built digest in the context-v3 `comparison.json`.

Replacing the capped numbers means re-reporting `TRAINING_SUMMARY.md`,
`comparison.json`, and the `known_limitations` entry in `PROVENANCE.json` that
names the 30-case slice.

## What to report

Do not report a single top-1 number as context awareness. Compare base and
trained runs using the same generated test manifest and include:

- candidate retrieval coverage and non-gold-distractor coverage;
- strict JSON rate;
- exact node-set match and base-case all-permutations exactness;
- cross-permutation consistency after mapping labels back to nodes;
- cardinality accuracy and metrics by gold cardinality;
- micro precision, recall, and F1;
- separate `synthetic_context` and `svg_edit_bench` results; and
- exact-set degradation for blank-image and blank-instruction ablations.

Treat `blank_image` as a pixel ablation, not a complete context ablation: its
structured candidate metadata can still answer some spatial instructions.

No final metric values are committed here. Populate a dated run report only
after both clean evaluations finish on the same dataset manifest and the
ablation predictions have been inspected.

## Deployment compatibility

The training prompt and evidence-sheet renderer are the same closed-choice
contract used by `semantic_id_patch`. Keep runtime `candidate_choice_limit`,
`max_candidates`, candidate evidence size, and crop padding aligned with the
training preset. Configure the Hugging Face model as
`task: "image-text-to-text"` and activate the adapter only for the candidate
rerank response schema so unrelated SVG patch generation continues to use the
base model.

If the runtime candidate pool is oversized, omits the target, or cannot provide
image evidence, the architecture uses its explicit fallback path. Report those
cases separately; reranker accuracy alone cannot repair candidate-retrieval
failures.
