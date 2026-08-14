# SVG Patch Lab Architecture

The repository now contains one end-to-end pipeline: DOM skeleton creation,
LLM-authored JSON patches, validation, and deterministic SVG patch execution.

## Core DOM Pipeline

1. `core/xml.py` parses SVG safely, rejects DTD/entities, and assigns stable
   preorder node IDs.
2. `core/scene.py` builds the compact DOM skeleton passed to the model. Heavy
   geometry attributes such as `d` and `points` are represented by hashes and
   character counts instead of raw coordinates.
3. `core/patch.py` parses a versioned JSON patch and can derive gold patches
   from SVGEditBench references.
4. `core/validate.py` enforces task allowlists, target scope, value safety, and
   protected geometry rules.
5. `core/executor.py` applies validated patch operations to the original SVG DOM.

## LLM Skeleton Patch Path

`architectures/patching.py` implements `skeleton_patch`.

For each benchmark case it:

1. Builds the SVG DOM skeleton.
2. Optionally runs the **vision context** sub-pipeline (see below) which
   annotates each visible node with semantic labels before the skeleton is
   serialised.
3. Renders `prompt_templates/patch_v3.txt` with the user instruction and
   skeleton JSON (now containing `visual_context` fields where available).
4. Calls the configured model adapter.
5. Parses the model response as a JSON patch.
6. Validates the patch against the skeleton and task policy.
7. Applies the patch to the original SVG.

The only registered architecture is `skeleton_patch`.

## Vision Context Sub-Pipeline

`vision/context.py` implements `VisionContextAnnotator`.

When `vision_context.enabled = true` in the experiment config, it runs **before**
the patch LLM call and annotates each eligible DOM node with a
`visual_context` object:

```json
{
  "summary": "rectangular window on the right side of the house",
  "labels": ["window", "rectangle"],
  "role": "part",
  "confidence": 0.91,
  "bbox": [210, 140, 58, 42],
  "visible_pixels": 1832
}
```

**How it works, per node:**

1. `vision/render.py` renders a three-panel PNG:
   - **left**: full SVG
   - **middle**: only the selected node (all others hidden via `display:none`)
   - **right**: full SVG with the node bbox highlighted in magenta
2. The panel is sent with a structured prompt to a vision LLM.
3. The model returns a JSON annotation with `summary`, `labels`, `role`, and
   `confidence`.
4. Annotations are cached on disk keyed by `sha256(svg)[:16]/nodeId-sS-vV.json`
   so each SVG is annotated at most once per image size.

**Patch prompt integration:**

`patch_v3.txt` instructs the patch LLM to check `visual_context.labels` and
`visual_context.summary` first when the user instruction names a part by its
semantic name (e.g. "windows", "roof", "eyes"), before falling back to colour
matching. This enables natural-language grounding: *"change the color of the
windows from red to white"* resolves to the exact node IDs whose
`visual_context.labels` contains `"window"`.

## Model Adapters

- `OpenAICompatibleAdapter` calls vLLM, SGLang, llama.cpp, or other
  OpenAI-compatible servers. Used for both the patch LLM and the vision LLM
  (typically on separate ports).
- `ReplayAdapter` replays saved JSONL responses for deterministic tests.

## Evaluation

`eval/runner.py` loads SVGEditBench cases, runs `skeleton_patch`, writes
`results.jsonl`, and summarizes structural metrics. Raster MSE remains optional
through `.[eval]`, but rendering is off in the default config.

Vision model call timings are logged separately under `role: "vision_context"`
in `model_call_details` within each result record.
