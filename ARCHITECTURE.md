# SVG Patch Lab — Architecture & Roadmap

## Overview

SVG Patch Lab tests whether a small language model can edit Scalable Vector Graphics reliably by emitting structured DOM patches rather than regenerating the full file. The central hypothesis is that constrained, local edits — expressed as JSON operations over a compact skeleton of the SVG tree — are easier for a small model to learn and verify than full-file rewrites, which must reproduce every character correctly.

The system is built around three concerns kept deliberately separate: the **data and task definition** (what constitutes a correct edit), the **architecture** (how the model is prompted and its response is applied), and the **evaluation protocol** (how quality is measured). This separation allows any of the three to evolve independently.

---

## 1. Core Data Model

### 1.1 SVG Representation

Every SVG file is normalized through a deterministic preprocessing pipeline before any model interaction:

- **Parsing** (`core/xml.py`): ElementTree-based parsing with a 2 MB cap and rejection of DTD entities. The resulting tree is stable across whitespace-only differences.
- **Node indexing** (`core/xml.py → index_tree()`): A depth-first preorder walk assigns IDs `n0`, `n1`, … to every element. These IDs are the stable handles used in all patches and are deterministic regardless of attribute order.
- **Skeleton construction** (`core/scene.py → build_scene()`): Heavy geometry attributes (`d` for paths, `points` for polygons) are replaced with `(sha256, char_count)` pairs. Fill, stroke, and opacity properties are resolved through CSS inheritance and recorded as `resolved_style`. The resulting JSON object is a compact but complete description of the tree's visual intent without embedding the raw geometry.
- **Geometry protection** (`core/xml.py → protected_geometry()`): The SHA256 hash set produced before any edit is carried forward. If the model's output changes these hashes, the patch fails the protected-geometry check.

### 1.2 Patch Schema

A `Patch` (versioned `v1`, `core/patch.py`) is a JSON array of operations. Three operation types exist:

| Operation | Effect |
|---|---|
| `set_attributes` | Write or overwrite key-value attribute pairs on a list of target nodes |
| `remove_attributes` | Delete named attributes from target nodes |
| `insert_primitive` | Append a new SVG element under a specified parent, after a specified sibling |

Each operation carries `targets` (a list of `nX` IDs), `attributes` (a dict), and for inserts, an `element` field containing the new node as a JSON object.

### 1.3 Patch Validation (`core/validate.py`)

A `PatchPolicy` is constructed per task before any model output is applied. It enforces:

- **Attribute allowlists** — only attributes relevant to the task may be written (e.g., `change_color` → `{fill}`, `set_contour` → `{stroke, stroke-width}`, `upside_down` → `{transform}`).
- **Target scope** — `upside_down`, `transparency`, and `crop_to_half` are root-only: only `n0` may be addressed.
- **Safety checks** — values are scanned for `javascript:`, `url(`, `data:`, `<`, `>` to prevent injection.
- **Cardinality limits** — at most 16 operations and 128 targets per operation.
- **Protected geometry** — the `d` and `points` attributes may never appear in a set or remove operation.

### 1.4 Patch Execution (`core/executor.py`)

`apply_patch()` applies operations sequentially against the parsed SVG ElementTree in memory. The executor is intentionally simple: no transactional rollback, no implicit cascades. If a target node does not exist, the operation is a no-op for that node but execution continues.

---

## 2. Benchmark Dataset

`SVGEditBench/` contains 600 cases across six tasks, each exercising a different class of edit:

| # | Task | Allowed Attributes | Scope |
|---|---|---|---|
| 1 | `change_color` | `fill` | Selected nodes |
| 2 | `set_contour` | `stroke`, `stroke-width` | Selected nodes |
| 3 | `compression` | _(empty — excluded from localized eval)_ | Global |
| 4 | `upside_down` | `transform` | Root only |
| 5 | `transparency` | `opacity` | Root only |
| 6 | `crop_to_half` | `viewBox` | Root only |

Cases are 100 Twemoji emoji per task. The emoji IDs are frozen; the evaluation protocol bars training data from these identities (see §6).

Gold patches are derived deterministically by `derive_patch()`, which diffs the original and reference SVG trees attribute-by-attribute and emits the minimal set of `set_attributes` and `remove_attributes` operations.

---

## 3. Architecture Strategies

Eight prompt/execution strategies are implemented, all conforming to a single `Architecture` interface (`run(case, model) → ArchitectureResult`).

### 3.1 Non-model Baselines

- **`oracle_patch`** — derives the gold patch from the reference answer and applies it. Scores 100% on structural metrics; used to calibrate the evaluation pipeline.
- **`rule_based_patch`** — regex-based inference for simple instructions (color codes, opacity percentages, flip keywords). No model call; useful as a lower-bound baseline.

### 3.2 Full-Rewrite

- **`full_rewrite`** — prompts the model to emit the entire modified SVG inside a markdown fence. The response is extracted by fence stripping; no patch schema is involved. Prone to geometry corruption and verbose output; included as an upper-bound reference for model capability.

### 3.3 Patch Architectures (model-guided)

These all produce a JSON patch, validate it, and apply it via the executor.

- **`full_context_patch`** — includes the complete original SVG as context. High token cost but maximum information.
- **`skeleton_patch`** — replaces the full SVG with the compact JSON skeleton. Dramatically reduces token count while preserving structural and style context. **Primary architecture.**
- **`visual_skeleton_patch`** — skeleton + a base64-encoded rendered PNG of the original SVG as an image input. Enables vision-capable models to use pixel-level cues.
- **`visual_stats_patch`** — skeleton where each node carries a compact human-readable `visual` field: aspect-ratio-correct bounding box in viewBox units, area %, a 3×3-grid position word, and quantized dominant rendered color. `visible: false` means hiding the node caused no measurable unique pixel change at the configured resolution and threshold; it is weak negative evidence, not proof that the node is structurally irrelevant. Stats come from diffing the full render against a render with that node hidden (occlusion-aware, inheritance-faithful; N+1 rasterizations at 64 px), and are disk-cached by SVG content (`.cache/visual_stats/`), so each benchmark input is measured once and reused across runs. This is the cheap, training-free path of Plan A: it targets node selection for frozen text models without embedding vectors.
- **`analytic_stats_patch`** — diagnostic ablation with the same model-visible `visual` field names, computed from nominal SVG geometry without rasterization. Compound paths are flattened per subpath with ordinary `nonzero`/`evenodd` hole handling, and position uses the nominal filled-area centroid. The strict variant, **`strict_analytic_stats_patch`**, uses the same schema-constrained decoding as the strict skeleton and render-derived arms. Analytic and rendered values remain different measurement systems: stroke, clipping, filters, anti-aliasing, and compositing can separate them even without occlusion.
- **`semantic_id_patch`** — two-stage rendered-to-DOM grounding. A normal render and a unique-color element-ID render produce a compact per-node ownership legend in two base rasterizations. The first model call shortlists at most a configured number of node IDs; only those candidates are then hidden and rerendered. The already-rendered full/hidden pair yields a difference heatmap plus structured area, transparency, underlayer-reveal, component, color-transition, spatial, and ID-ownership evidence without another rasterization. The second call receives this evidence and, for a vision model, the original/hidden/difference image triplet before emitting the constrained patch. Text-only models consume the same evidence as JSON. For deletion cases that expose transparency without an encoded underlayer, an optional third call reuses the configured lightweight Qwen model as a conservative replacement-element generator for a shortlisted surviving node. The document is assembled deterministically, with exact semicircular-arc fitting where applicable. Canvas, safety, deleted-element fingerprint, outside-mask locality, and inside-mask reconstruction gates reject harmful or ineffective generations and retain the localized deletion. Evaluation records retain all evidence under `architecture_details`.
- **`routed_strict_skeleton_patch` / `routed_strict_visual_stats_patch`** — preserve the corresponding strict baselines for localized tasks, but compile unambiguous root-only tasks directly. `upside_down` becomes a viewBox-relative root transform, “transparent by half” becomes root opacity `0.5`, and a directed half-crop becomes the corresponding half viewBox. A route is accepted only when its instruction and canvas are sufficient; otherwise the architecture uses the ordinary model path. `semantic_id_patch` enables this bypass by default so whole-canvas tasks do not pay for ID rendering or target selection.

### 3.4 Diagnostic Architectures

- **`oracle_target_patch`** — gold target node IDs are injected into the prompt, ablating the node-selection problem. Measures how much error comes from target selection vs. attribute generation.
- **`two_stage_patch`** — a first model call selects target nodes; a second call generates the attribute patch. Separates selection from writing for analysis.

### 3.5 Prompting (`architectures/prompts.py`)

Prompts are versioned (`v1`, `v2`, `v3`) and loaded from `svgpatchlab/prompt_templates/`. v3 is zero-shot with generic patch rules; v2 adds four few-shot examples. Templates are rendered with Jinja2 and receive the skeleton JSON, task instruction, and (for visual) the image data URL.

---

## 4. Model Adapters

All model interaction is routed through a `ModelAdapter` interface with a single method `generate(ModelRequest) → ModelResponse`. Three adapters ship:

- **`OpenAICompatibleAdapter`** (`models/openai_compatible.py`): HTTP client for vLLM, SGLang, llama.cpp, or any OpenAI-compatible server. Supports chat completions and raw completions endpoints, JSON mode, and image inputs.
- **`HuggingFaceAdapter`** (`models/huggingface.py`): In-process Transformers pipeline; lazy-loaded on first call. Supports dtype, device, and task configuration.
- **`ReplayAdapter`** (`models/replay.py`): Plays back a pre-recorded JSONL file deterministically. Used for offline evaluation and CI.

A `RecordingModelAdapter` wrapper around any adapter captures latency and token counts for every call, producing the `model_calls`, `mean_model_latency_seconds`, and `prompt_tokens`/`completion_tokens` entries in the summary.

---

## 5. Evaluation Infrastructure

### 5.1 Metrics (`eval/metrics.py`)

`evaluate_output()` computes metrics on every case regardless of failure. No cases are dropped from averages.

**Patch metrics** (structural, no rendering required):
- `gold_patch_exact` — whether the predicted patch signature matches the gold signature exactly.
- `patch_precision` / `patch_recall` — overlap between predicted and gold `(op, target, attr, value)` tuples.

**Structural metrics**:
- `changed_nodes` — number of distinct nodes touched.
- `protected_geometry_preserved` — whether all SHA256 hashes of `d`/`points` are unchanged.
- `reference_structure_match` — whether the output tree structure matches the reference (tag names and IDs at every position).

**Raster metrics** (requires CairoSVG):
- `image_mse` — mean squared error over RGB pixel values between rendered output and rendered reference.
- `failure_aware_mse` — `image_mse` when successful, `1.0` on any failure (invalid output, render error, structural mismatch).

### 5.2 Runner (`eval/runner.py`)

`run_evaluation()` loads the dataset, architecture, and model adapter from config; iterates cases; writes per-case JSONL to `results.jsonl`; and writes aggregated `summary.json` with per-task and overall stats. The output directory defaults to `runs/{experiment_name}/`.

### 5.3 Rendering (`eval/render.py`)

CairoSVG renders SVGs to a configurable raster size (default 72×72 px), with a restricted headless Chromium fallback when native Cairo is unavailable. The `render_svg_data_url()` function produces a base64 data URL used by visual architectures. `render_svg_visual_context()` adds the normal/ID render pair, decoded pixel ownership, and automatic counterfactual fallback for complex compositing features.

---

## 6. Evaluation Protocol

- The 100 frozen emoji IDs in `SVGEditBench/` are held out as **test-only data**. If SFT or LoRA fine-tuning is added, training and validation examples must come from other Twemoji files, split by emoji identity. No emoji may appear in both training and test.
- All reported numbers include failures. Invalid or unrenderable outputs receive `failure_aware_mse = 1.0`.
- Primary reported metrics: valid-output rate, gold-patch exact rate, patch precision/recall, protected-geometry rate, changed-node count, per-task MSE, failure-aware MSE, latency, and token counts.

---

## 7. Configuration

Two config layers decouple architecture choice from model choice:

- `configs/experiments/*.json` — architecture name, dataset root, model config path, evaluation settings (`render`, `render_size`, `save_outputs`, `output_dir`).
- `configs/models/*.json` — adapter type, server URL, model name, temperature, max tokens, timeout, JSON mode, endpoint variant.

The `matrix` CLI command runs all principal architectures against one model with identical paired cases, enabling direct head-to-head comparison.

---

---

# Roadmap: Planned Extensions

---

## Plan A — Visual Node Understanding (ViT / VLM / LSTM-GNN)

### Motivation

The current `visual_skeleton_patch` architecture passes a single whole-SVG raster image to a vision-capable model. This is coarse: the model must implicitly localize which node corresponds to which pixel region. For complex SVGs with dozens of overlapping elements, this alignment is noisy and the rendered image provides weak signal for node-level selection.

The goal of this plan is to give the model — or a lightweight upstream module — **a structured visual representation tied to individual SVG nodes**, so that prompts can include per-node visual context rather than a single global image.

### Approach

**Stage 1 — Per-node rasterization:** For each node `nX` in the skeleton, render the SVG twice: once normally and once with only node `nX` visible (all other fill/stroke set to transparent). The difference image isolates the visual footprint of that node. This requires only the existing CairoSVG renderer.

**Stage 2 — Node embedding:** Encode each node's difference image with a frozen ViT (e.g., `ViT-B/16`) or a compact CNN to produce a fixed-size visual embedding vector. This embedding captures spatial position, shape, and color of the node independent of its XML representation.

**Stage 3 — Graph-level reasoning (LSTM-GNN):** The SVG DOM is a tree, which is a special case of a DAG. Build a GNN over the preorder-indexed nodes where edges represent parent-child and sibling adjacency. Node features are the concatenation of the ViT embedding and a tokenized encoding of the node's non-geometry attributes. An LSTM aggregates path-level context (ancestor chain). The output is a per-node vector used for target selection.

**Stage 4 — Integration:** The node vectors are either:
- (a) Used standalone as a learned selector: a small MLP over node vectors produces a relevance score per node, and only high-scoring nodes are passed to the language model as candidates.
- (b) Injected into the skeleton JSON as an additional `visual_embedding` field that a fine-tuned language model can attend to via a cross-attention projection layer.

### Impact on Current Code

- `core/scene.py` — extend `build_scene()` to optionally include per-node visual fields.
- `eval/render.py` — add `render_node_mask(svg, node_id)` function.
- New module: `svgpatchlab/vision/` — ViT inference, GNN definition, node embedding cache.
- `architectures/patching.py` — new `VisualGNNPatchArchitecture` consuming node embeddings.

---

## Plan B — Basic Tasks: Rotation, Flip, and Delete

### Motivation

The six existing SVGEditBench tasks test attribute manipulation (`fill`, `stroke`, `opacity`, `viewBox`, `transform`) but do not cover **structural edits** — operations that change the shape or existence of nodes rather than their styling. Adding rotation, flip (horizontal/vertical), and element deletion rounds out the primitive operation space and makes the benchmark meaningful for a broader class of real-world SVG edits.

These nine operations (existing six + three new) are designated **Basic Tasks**: atomic, single-intent, verifiable edits that serve as the ground vocabulary for Plan C.

### New Tasks

**`rotate`** — Apply a rotation transform to one or more target nodes.
- Allowed attributes: `{transform}` with value `rotate(θ, cx, cy)` or a full matrix.
- Scope: selected nodes or root.
- Gold derivation: diff-based, same as existing tasks.
- Key challenge: rotation centers must be computed from the bounding box of the node's rendered footprint, requiring render-time geometry.

**`flip`** — Apply a horizontal or vertical reflection transform to target nodes.
- Allowed attributes: `{transform}` with value `scale(-1,1)` + translate for horizontal, `scale(1,-1)` + translate for vertical.
- Scope: selected nodes or root.
- Distinct from `upside_down` (which is always vertical and always root): `flip` is directed and can be scoped.

**`delete`** — Remove one or more `<element>` nodes entirely from the SVG tree.
- New operation type `remove_element` in the patch schema (currently only `set_attributes`, `remove_attributes`, `insert_primitive` exist).
- Validation policy: element must exist, must not be the root (`n0`), must not be an ancestor of a protected node.
- Gold derivation: record which node IDs are absent in the answer SVG vs. the original.

### Implementation Changes

- `core/patch.py` — add `remove_element` operation type to `PatchOperation`; update `Patch` schema to v2.
- `core/executor.py` — implement `remove_element` in `apply_patch()`.
- `core/validate.py` — add `ROTATE_ALLOWED_ATTRS`, `FLIP_ALLOWED_ATTRS`; add `delete` policy with element-existence checks; update schema version handling.
- `core/xml.py` — `derive_patch()` must detect absent nodes and emit `remove_element` operations.
- `SVGEditBench/` — new task directories `7_Rotate/`, `8_Flip/`, `9_DeleteItem/` with 100 cases each (generated from Twemoji, held-out emoji IDs).
- `architectures/prompts.py` — extend prompt templates with examples for the three new operations.

---

## Plan C — Complex-Edit Decomposition via SFT + GRPO

### Motivation

Plans A and B produce a system that can execute any of nine well-defined atomic edits reliably. Real SVG editing requests, however, are often compound: "make the background transparent, flip the icon, and change the red elements to blue." Current architecture attempts the entire edit in a single model call, which fails when individual components are individually tractable but jointly overwhelming for a small model.

This plan introduces a **decomposition layer**: a small model trained to convert any complex natural-language edit instruction into an ordered sequence of Basic Task calls, which are then executed one by one using the existing patch infrastructure.

### Architecture

```
User instruction
       │
       ▼
 Decomposer Model
 (seq2seq over instruction → [Basic_1, Basic_2, ..., Basic_k])
       │
       ▼
 For each Basic_i:
   ├─ Current SVG state (skeleton at step i)
   ├─ Basic task instruction
   └─ Patch Model (existing skeleton_patch architecture)
            │
            ▼
       Apply patch → new SVG state
       │
       └─ Loop until i = k
       │
       ▼
 Final SVG output
```

### Training the Decomposer

**Data generation (small-quantity SFT):**

Complex training examples are constructed programmatically by chaining two to four Basic Task samples from the existing corpus:

1. Sample k Basic Task cases whose source SVGs are compatible (same base emoji).
2. Chain them: apply Basic_1 gold patch to SVG_0 to get SVG_1, apply Basic_2 gold patch to SVG_1 to get SVG_2, etc.
3. Write a natural-language composition instruction by concatenating or paraphrasing the individual task instructions.
4. The supervision target is the ordered list of `(task_type, task_instruction)` pairs.

This yields thousands of synthetic complex examples from a small base of 100 × 9 = 900 Basic Task cases, without manual annotation.

**SFT (Supervised Fine-Tuning):**

- Fine-tune a small language model (same size class as the patch model, e.g., Qwen 0.8B–4B) on the decomposition task: input is the complex instruction + current skeleton, output is the JSON list of basic steps.
- Training uses the emoji-disjoint split: decomposer training set uses non-benchmark Twemoji IDs.
- Loss is cross-entropy over the step-list JSON.

**RL via GRPO (Group Relative Policy Optimization):**

After SFT, the decomposer is further trained with GRPO using the final SVG quality as the reward signal. GRPO is well-suited here because:

- The reward is not differentiable (it requires executing the decomposed plan and measuring MSE or patch exactness on the final output).
- GRPO avoids a separate value-model by normalizing rewards within a group of sampled decompositions for the same input.

Reward function for a decomposed plan `[Basic_1, …, Basic_k]`:

```
R = α · (1 − failure_aware_mse)
  + β · gold_step_recall
  + γ · (1 / k) · Σ_i protected_geometry_preserved_i
  − δ · max(0, k − k_gold)
```

where `k_gold` is the number of steps in the ground-truth decomposition, penalizing over-decomposition.

**Iterative Execution Loop:**

The executor wrapper maintains a **session state** — the current SVG string after each applied patch — and feeds it back into the skeleton builder for the next step. A step is skipped (with warning) if its patch fails validation; execution continues with the remaining steps. This makes the loop robust to partial failures.

### Evaluation

- **Decomposition quality:** gold-step precision/recall (are the right task types chosen in the right order?).
- **Final SVG quality:** `failure_aware_mse` and `reference_structure_match` on the final output, compared against baselines that attempt the full edit in a single call.
- **Error propagation:** track how single-step failures compound across k steps.

### Impact on Current Code

- New module: `svgpatchlab/decompose/` — decomposer model wrapper, chain executor, session state manager.
- `eval/runner.py` — add `run_chain_evaluation()` for multi-step plans.
- `eval/metrics.py` — add `step_precision`, `step_recall`, `chain_valid_rate`.
- New config layer: `configs/chains/` — decomposer model path, max steps, step retry policy.
- Training scripts (outside the eval harness): `train/sft_decomposer.py`, `train/grpo_decomposer.py`.
