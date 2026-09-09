# SVG Patch Lab

SVG Patch Lab tests whether a small language model can edit SVGs more reliably
by emitting constrained DOM patches instead of regenerating complete files. The
official `SVGEditBench/` clone is treated as read-only evaluation data.

## What is implemented

- Loader for all 600 SVGEditBench prompt/answer pairs.
- Deterministic preorder node IDs (`n0`, `n1`, ...).
- Compact DOM skeletons that replace `d` and `points` values with hashes and
  character counts.
- Versioned JSON patch schema, task-specific allowlists, and protected geometry.
- Deterministic patch executor.
- Gold-patch derivation from SVGEditBench references.
- Full-rewrite, full-context patch, skeleton patch, visual skeleton, two-stage,
  semantic ID patch, sparse Graph-MoE patch, oracle-target, rule-based, and deterministic oracle
  architectures.
- Interchangeable local Transformers, OpenAI-compatible server, and replay
  model adapters.
- Structural, locality, patch precision/recall, and SVGEditBench-compatible MSE
  evaluation.

Compression is kept in the full-rewrite and oracle evaluations. It is excluded
from localized patch presets because compression is an inherently global edit.

## Layout

```text
svgpatchlab/
  architectures/  Experiment strategies and prompts
  core/           SVG parsing, skeletons, patches, policies, executor
  data/           SVGEditBench adapter
  eval/           Rendering, metrics, experiment runner
  models/         Swappable model adapters
configs/
  experiments/    Architecture presets
  models/         Model/runtime presets
docs/              Rendered architecture document
tests/             Standard-library regression tests
SVGEditBench/      Unmodified official benchmark clone
```

## Setup

If you cloned this repository normally, fetch the benchmark submodule first:

```bash
git submodule update --init --recursive
```

Core and structural evaluation have no third-party dependencies:

```bash
python3 -m svgpatchlab.cli inspect SVGEditBench
python3 -m unittest discover -s tests -v
```

Install raster evaluation support for CairoSVG MSE:

```bash
python3 -m pip install -r requirements-eval.txt
```

For in-process Hugging Face inference:

```bash
python3 -m pip install -r requirements-hf.txt
```

## Run Plan A/B/C on Kaggle

The Kaggle preset uses local Hugging Face inference with 4-bit loading for
`Qwen/Qwen3.5-4B`:

```bash
git pull
git submodule update --init
python3 -m pip install -r requirements-hf.txt
python3 scripts/run_kaggle_plans.py --no-render
```

Outputs are written under `runs/kaggle-plans/`:

```text
runs/kaggle-plans/plans-summary.json
runs/kaggle-plans/plan_a_visual_node_understanding/
runs/kaggle-plans/plan_b_basic_tasks/
runs/kaggle-plans/plan_c_chain/
```

Use `--limit-per-task 5` for the same smoke size as the Kaggle run log. Set
`HF_TOKEN` in Kaggle secrets for better Hub download limits.

## Validate the entire benchmark pipeline

The oracle derives the minimal attribute patch from each reference and should
score perfectly:

```bash
python3 -m svgpatchlab.cli evaluate --config configs/experiments/oracle.json
```

Use `--no-render` for structural-only development checks when CairoSVG is not
installed.

Results are written as per-case JSONL plus a summary under `runs/oracle/`.
Invalid model outputs are retained as failures and receive failure-aware MSE 1,
rather than being omitted from averages.

## Run Qwen3.5-4B through an OpenAI-compatible server

Start Qwen using vLLM, SGLang, llama.cpp, or another compatible server at
`http://localhost:8000/v1`, then run:

```bash
python3 -m svgpatchlab.cli evaluate \
  --config configs/experiments/skeleton_patch.json \
  --limit 10
```

Run all principal comparisons by changing only the experiment config:

```bash
configs/experiments/full_rewrite.json
configs/experiments/rule_based.json
configs/experiments/full_context_patch.json
configs/experiments/skeleton_patch.json
configs/experiments/visual_stats_patch.json
configs/experiments/two_stage_patch.json
configs/experiments/visual_skeleton_patch.json
configs/experiments/oracle_target_patch.json
```

## Switch models

Model configuration is isolated from architecture configuration. Override it at
the command line:

```bash
python3 -m svgpatchlab.cli evaluate \
  --config configs/experiments/skeleton_patch.json \
  --model-config configs/models/qwen3.5-4b-huggingface.json
```

To add another model, either create a JSON preset using an existing adapter or
implement one class conforming to `svgpatchlab.models.base.ModelAdapter`, then
register it in `svgpatchlab/models/factory.py`. No architecture, SVG, or metric
code needs to change.

Useful CLI overrides:

```bash
python3 -m svgpatchlab.cli evaluate \
  --config configs/experiments/skeleton_patch.json \
  --architecture full_context_patch \
  --model-config configs/models/my-model.json \
  --limit 25 \
  --output-dir runs/my-model-full-context
```

Run the complete architecture matrix against one model endpoint:

```bash
python3 -m svgpatchlab.cli matrix \
  --model-config configs/models/qwen3.5-4b-openai.json \
  --limit-per-task 2 \
  --output-root runs/qwen3.5-4b-smoke
```

Remove `--limit-per-task` for the complete five-task localized-edit evaluation.
The matrix deliberately uses the same cases for every architecture so results
are paired.

### Run the clean visual-tag A/B

Compare the plain skeleton against the same skeleton with cached render-derived
`visual` fields:

```bash
python3 -m svgpatchlab.cli matrix \
  --config configs/experiments/skeleton_patch.json \
  --model-config configs/models/qwen2.5-7b-ollama.json \
  --architectures skeleton_patch visual_stats_patch \
  --limit-per-task 2 \
  --render \
  --output-root runs/qwen2.5-7b-visual-stats-ab-smoke
```

Both arms use the same cases, instruction, patch prompt, model, validator, and
executor. The only model-visible difference is the per-node `visual` object
attached by `visual_stats_patch`. Node hiding is cached preprocessing used to
measure those fields; it is not an edit. Remove `--limit-per-task 2` for the
full paired run.

Analyze target-level and paired outcomes:

```bash
python3 scripts/analyze_visual_stats_ab.py \
  --root runs/qwen2.5-7b-visual-stats-ab-smoke
```

Run the path-only spatial grounding probe, where path coordinates are hidden
and position cannot be recovered from the plain skeleton:

```bash
python3 scripts/run_spatial_grounding_probe.py \
  --model-config configs/models/qwen2.5-7b-ollama.json \
  --output-root runs/qwen2.5-7b-spatial-grounding-probe
```

Run the frozen expanded holdout (39 distinct path-only SVGs, all nine spatial
regions, and four relative-size targets):

```bash
python3 scripts/run_spatial_grounding_holdout.py \
  --model-config configs/models/qwen2.5-7b-ollama.json \
  --output-root runs/qwen2.5-7b-spatial-grounding-holdout-v1 \
  --seed 20260725
```

The holdout fails closed if path-coordinate cues leak into the skeleton, if
position or size labels no longer match the render geometry, or if the active
patch prompt differs from the frozen version recorded in its manifest.

To isolate the value of rasterizing from the value of geometry itself, run the
schema-constrained three-arm ablation. `strict_analytic_stats_patch` exposes
the same compact field names as the render-derived arm, but computes nominal
geometry directly from the SVG source:

```bash
python3 -m svgpatchlab.cli matrix \
  --config configs/experiments/skeleton_patch.json \
  --model-config configs/models/qwen2.5-7b-ollama-constrained.json \
  --architectures \
    strict_skeleton_patch \
    strict_analytic_stats_patch \
    strict_visual_stats_patch \
  --render \
  --output-root runs/qwen2.5-7b-analytic-ablation-v1
```

The occlusion-isolating v2 holdout then tests the regime in which nominal
geometry is deliberately misleading. Verify all static and rendered
invariants before making model calls:

```bash
python3 scripts/run_occluded_spatial_holdout.py \
  --model-config configs/models/qwen2.5-7b-ollama-constrained.json \
  --output-root runs/occluded-spatial-holdout-v2 \
  --verify-only

python3 scripts/run_occluded_spatial_holdout.py \
  --model-config configs/models/qwen2.5-7b-ollama-constrained.json \
  --output-root runs/occluded-spatial-holdout-v2
```

Version 2 uses 48 unique sources, balances target and decoy IDs within every
spatial label, fixes all protected path lengths, keeps clean candidates
pairwise disjoint, and reserves an immutable output root. Its manifest freezes
the case order, source and answer hashes, model configuration, prompt hash,
render sizes, arm order, and verification report.

In the retained Qwen 2.5 7B run, all 144 outputs were valid. On clean cases,
analytic and rendered target accuracy was 9/24 versus 11/24 (`p=0.625`). On
occluded cases, analytic accuracy was 0/24 and rendered accuracy was 9/24
(`p=0.003906`); exact gold patches were 0/24 versus 6/24 (`p=0.03125`).
These are paired results for a controlled synthetic existence test, not an
effect-size estimate for natural SVGs.

### Route deterministic whole-canvas edits

The completed baseline architectures remain unchanged for reproducibility.
`routed_strict_skeleton_patch` and `routed_strict_visual_stats_patch` add a
zero-model-call compiler for unambiguous `upside_down`, `transparency`, and
`crop_to_half` instructions. The semantic architecture enables the same router
by default. Ambiguous instructions or unsupported canvases fall back to the
normal model path.

Run the frozen 300-case root-task check without a model or renderer:

```bash
python3 -m svgpatchlab.cli evaluate \
  --config configs/experiments/routed_root_tasks.json
```

The pilot design, results, caveats, and next experiment are documented in
[`docs/VISUAL_STATS_AB.md`](docs/VISUAL_STATS_AB.md).

## Ground rendered parts to source nodes

`semantic_id_patch` implements a two-stage rendered-to-DOM grounding path for
instructions such as "remove the small lens":

1. Render the SVG, derive per-node visual statistics, and enumerate lossless
   visual candidates. A source-paint filter is used only when it preserves the
   complete matching set.
2. For a bounded candidate pool, render a labelled evidence card for each node:
   highlighted full context, a fresh direct-from-SVG vector crop, and a binary
   ownership mask. The prompt also receives normalized geometry and compact
   structural/style fields keyed by the visible label.
3. Ask the model for closed choices such as `{"choices":["B"]}`. DOM IDs stay
   private and are restored only after constrained decoding. A singleton pool
   is resolved deterministically without a model call.
4. Hide and rerender only the selected nodes. Compare the full and hidden
   renders pixel-by-pixel to capture changed area and location, transparent
   holes, newly visible or revealed underlayers, RGB/alpha changes, connected
   components, dominant color transitions, and overlap with the selected
   node's ID-buffer ownership.
5. Give a vision model the original, hidden preview, and derived difference
   heatmap. Give a text-only model the same evidence as structured JSON before
   it emits the constrained patch.

The evidence renderer shares one full-context render, then performs three
candidate-specific rasterizations to locate the node and render its contextual
and isolated vector crops. This avoids magnifying a low-resolution bitmap for
small elements. The difference heatmap and statistics reuse the later hidden
previews. Each evaluation record stores selection evidence and
`architecture_details.counterfactual_previews`. Oversized pools, missing image
support, and complex compositing use explicit fallback paths.

For deletion, the preset also enables a guarded lightweight-Qwen completion
fallback. It is bypassed when hiding the selected node reveals an existing
underlayer. When the region instead becomes transparent with almost no revealed
geometry, the same configured Qwen model receives the original SVG, localized
deletion, surviving-node geometry, and difference evidence. It generates one
replacement SVG element for a shortlisted surviving node; the system assembles
and validates that replacement before accepting it. Semicircular path-to-circle
proposals are fitted exactly from their arc endpoints rather than trusting
model arithmetic. Generated results must preserve the canvas, pass
element/reference safety checks, keep the deleted element absent, stay below
the configured MSE outside the deletion mask, and materially differ from both
the original foreground and the plain localized deletion inside that mask. A
rejected candidate is retried with validation feedback; if no attempt passes,
the valid localized deletion is retained. The decision and acceptance evidence
are stored under `architecture_details.qwen_completion`.

Run the frozen complete-versus-missing-underlayer benchmark:

```bash
python3 scripts/run_occlusion_deletion_benchmark.py \
  --model-config configs/models/qwen2.5-7b-ollama.json \
  --output-root runs/qwen2.5-7b-occlusion-deletion-v1
```

The suite pairs four geometries with both conditions. Complete underlayers must
bypass generation; missing semicircular underlayers must trigger a localized
completion. The summary reports target selection, trigger/accept behavior, and
answer-render MSE separately.

Run the provided preset:

```bash
python3 -m svgpatchlab.cli evaluate \
  --config configs/experiments/semantic_id_patch.json \
  --limit 10
```

OpenAI-compatible chat models receive the images by default. For a text-only
server, set `"supports_images": false` in its model configuration; the same
architecture then uses only the compact visual statistics and counterfactual
summaries. Local Hugging Face models receive images only when configured with
`"task": "image-text-to-text"`.

### Train and audit the context-v3 selector

The context-v3 SFT path combines SVGEditBench-derived hard negatives with
synthetic face, robot, flower, and spatial-grid scenes whose same-style nodes
can be distinguished only by role, position, size, or relation. Candidate
letters are deterministically permuted and repainted into separate evidence
sheets. Validation and test require at least one non-gold distractor, and LoRA
startup verifies trainable language-attention, vision-attention, and
visual-projector families.

```bash
# Generate the grouped, permutation-aware v3 manifests and evidence sheets.
python -m train.node_grounding_sft \
  --config configs/train/node_grounding_context_h200.json \
  --generate-data-only

# Smoke test, then train.
python -m train.node_grounding_sft \
  --config configs/train/node_grounding_context_h200_smoke.json
python -m train.node_grounding_sft \
  --config configs/train/node_grounding_context_h200.json

# Compare the untouched base model with the trained adapter.
python -m train.node_grounding_sft \
  --config configs/train/node_grounding_context_h200_base_eval.json \
  --eval-only
python -m train.node_grounding_sft \
  --config configs/train/node_grounding_context_h200_eval.json \
  --eval-only
```

Both evaluations run matched blank-image and blank-instruction ablations.
No context-v3 result is claimed in this README; compare exact node sets,
cardinality, source-family breakdowns, permutation consistency, retrieval
coverage, and ablation drops from a completed paired run. See
[`docs/NODE_GROUNDING_SFT.md`](docs/NODE_GROUNDING_SFT.md) for the data contract,
artifact paths, and reporting protocol.

The executor supports validated `remove_element` patches, but the bundled
SVGEditBench clone does not yet contain a delete-task directory. Deletion
grounding is therefore covered by the frozen occlusion/deletion suite and
synthetic regression tests until such an official split is added.

## Ground without a generative VLM

`graph_moe_patch` learns a sparse instruction router over attribute, spatial,
and semantic/visual graph experts. It scores every editable SVG node and sends
the selected IDs to a deterministic patch compiler, so inference emits no free-
form SVG or JSON and makes zero generative-model calls. Training, the frozen
mixed-holdout ablation, and interpretation constraints are documented in
[`docs/GRAPH_MOE.md`](docs/GRAPH_MOE.md).

## Evaluation protocol

Use the committed 100 SVGEditBench emoji IDs only for final testing. If SFT or
LoRA is added, generate training and validation examples from other Twemoji
files and split by emoji identity. Do not place different tasks for the same SVG
across train and test.

The context-v3 node-grounding preset uses an identity-disjoint development
split of SVGEditBench plus synthetic context scenes. Its test split is useful
for paired base-versus-adapter debugging, but it is not an official zero-shot
SVGEditBench result. Keep the stricter external-corpus rule above for any final
benchmark claim.

Primary reported metrics should include:

- valid and executable output rate;
- gold-patch exactness and patch precision/recall;
- protected-geometry preservation;
- number of changed nodes;
- per-task MSE and failure-aware MSE;
- latency, model calls, and token usage where available.
