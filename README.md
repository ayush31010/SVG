# SVG Patch Lab

SVG Patch Lab is a compact SVG editing pipeline built around DOM skeletons and
constrained JSON patches. It keeps path geometry out of the model prompt, asks an
LLM to emit a small patch over stable node IDs, validates that patch, then applies
it back to the original SVG.

## What Is Kept

- Deterministic SVG parsing and preorder node IDs (`n0`, `n1`, ...).
- Compact DOM skeletons that hash heavy geometry attributes like `d` and
  `points`.
- Versioned JSON patch parsing.
- Task-aware patch validation and protected geometry checks.
- Deterministic patch execution against the original SVG DOM.
- One LLM framework: `skeleton_patch`.
- OpenAI-compatible and replay model adapters.
- SVGEditBench loading plus a focused evaluation runner for the skeleton path.

## Layout

```text
svgpatchlab/
  core/             SVG parsing, skeletons, patches, validation, execution
  architectures/    The skeleton_patch LLM architecture
  models/           OpenAI-compatible and replay adapters
  data/             SVGEditBench adapter
  eval/             Focused skeleton-patch evaluation
  prompt_templates/ Active skeleton patch prompt
configs/
  experiments/      skeleton_patch preset
  models/           OpenAI-compatible model preset
tests/              Core and skeleton-runner regression tests
SVGEditBench/       Official benchmark submodule
```

## Setup

Fetch the benchmark submodule if needed:

```bash
git submodule update --init --recursive
```

Core checks use only the Python standard library:

```bash
python3 -m svgpatchlab.cli inspect SVGEditBench
python3 -m unittest discover -s tests -v
```

Optional raster metrics require CairoSVG:

```bash
python3 -m pip install -e '.[eval]'
```

## Run The Skeleton Patch Pipeline

Start any OpenAI-compatible model server at `http://localhost:8000/v1`, then run:

```bash
python3 -m svgpatchlab.cli evaluate \
  --config configs/experiments/skeleton_patch.json \
  --limit 10
```

The default config runs structural evaluation without raster rendering. Add
`--render` after installing `.[eval]` to include image MSE.

Results are written under `runs/skeleton_patch/` with per-case JSONL and a
summary JSON.

## Vision Context Pipeline

The vision context sub-pipeline enriches each node in the DOM skeleton with a
semantic annotation before the patch LLM runs. This lets natural-language
instructions like *"change the color of the windows"* resolve to the correct
node IDs even when no colour hint is available.

### Install vision dependencies

```bash
python3 -m pip install -e '.[vision]'
```

### Start the vision model server (Apple M4 / MLX)

Run Qwen2.5-VL-3B on a second port using `mlx_lm` or `llama.cpp`:

```bash
# Option A – MLX (recommended on Apple Silicon)
pip install mlx-lm
python -m mlx_lm.server \
  --model mlx-community/Qwen2.5-VL-3B-Instruct-8bit \
  --port 8001

# Option B – llama.cpp with Metal
llama-server \
  --hf-repo bartowski/Qwen2.5-VL-3B-Instruct-GGUF \
  --hf-file Qwen2.5-VL-3B-Instruct-Q6_K.gguf \
  --port 8001 --n-gpu-layers 99
```

The text patch LLM (e.g. Qwen3.5-4B) continues to run on port 8000 as before.

### Run the vision experiment

```bash
python3 -m svgpatchlab.cli evaluate \
  --config configs/experiments/skeleton_patch_vision.json \
  --limit 100
```

Results land in `runs/skeleton_patch_vision/`. Node annotations are cached
under `runs/vision_cache/` so subsequent runs skip re-annotating unchanged SVGs.

Use `configs/models/qwen2.5-vl-7b.json` instead of the 3B config if you have
32 GB unified memory and want better label quality.
