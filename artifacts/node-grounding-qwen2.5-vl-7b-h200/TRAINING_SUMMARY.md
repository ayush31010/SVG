# SVG node-grounding Qwen2.5-VL-7B LoRA

Trained on 14 August 2026 on one NVIDIA H200 using the
`svgpatchlab.node_grounding.v2` closed-choice contract.

## Run

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Trainable LoRA parameters: 10,092,544 (0.1216%)
- Hard-negative split: 147 train / 19 validation / 18 test examples
- Schedule: 5 epochs, effective batch 4, 185 optimizer updates
- Runtime: 192 seconds
- Best checkpoint: epoch 3 (`eval_loss=0.1330`)
- Adapter SHA-256: `d3832d159364d2b5ee363bd17f11dde8eff0282c0309beeb2a0edbb99c69e6ad`

## Held-out hard-negative validation (19 examples)

| Metric | Base | Trained |
|---|---:|---:|
| Strict JSON | 26.3% | 100.0% |
| Top-1 accuracy | 5.3% | 73.7% |
| Exact node-set match | 5.3% | 68.4% |
| Micro-F1 | 6.5% | 79.2% |

## Production-selector evaluation

Only nontrivial pools with 2–6 candidates are scored. Singleton pools are
resolved deterministically at runtime.

| Split | Model | Strict JSON | Top-1 | Exact set | Micro-F1 |
|---|---|---:|---:|---:|---:|
| Validation (5) | Base | 4/5 | 4/5 | 0/5 | 50.0% |
| Validation (5) | Trained | 5/5 | 5/5 | 2/5 | 85.7% |
| Test (5) | Base | 3/5 | 3/5 | 0/5 | 46.2% |
| Test (5) | Trained | 5/5 | 5/5 | 2/5 | 82.4% |

The production-selector splits are small, so raw counts are reported and these
figures should be treated as development evidence, not a publication-grade
benchmark claim.

## Files

- `adapter/`: LoRA adapter and Qwen processor/tokenizer files
- `local_model_config.json`: workspace-relative inference configuration
- `evaluation/`: baseline, trained, validation, and test predictions/metrics
- `run/`: training configs, manifests, log, remote config, and source archive

Persistent Connect copy:
`/home/aneeshl/svg-node-grounding-qwen2.5-vl-7b-h200`

