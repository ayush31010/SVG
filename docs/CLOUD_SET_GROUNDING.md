# Matched set grounding: recovered Colab results

The matched experiment completed on a **Tesla T4 in Colab**, using PyTorch
2.11.0+cu128 and source commit `075687c`. The no-edge MLP transfers better than
the graph model on this exploratory natural-SVG subset. The best tested GPU
combination is the MLP with the existing source-paint/container grouping rules:
**42/80 exact target sets (52.5%)**. This is target selection, not end-to-end
rendered edit accuracy.

## GPU results

Both models use the same synthetic data seed, source-disjoint split, training
objective, hidden width and eight-epoch budget. Validation chooses each model's
checkpoint. The natural evaluation contains 64 Hicon and 16 Streetmix cases;
both arms evaluated all 80 with zero inference errors.

| Arm | Synthetic validation exact set | Natural exact set | Natural + grouping rules |
| --- | ---: | ---: | ---: |
| Two-layer GNN | 383/400 (95.75%) | 17/80 (21.25%) | 32/80 (40.0%) |
| No-edge MLP | 379/400 (94.75%) | 27/80 (33.75%) | 42/80 (52.5%) |

The frozen complexity gate and cached SigLIP top-1 fallback did not change the
grouped exact-set totals. The pure learned comparison has one GNN-only success
and eleven MLP-only successes (nominal exact McNemar p=0.00635). The previously
reported grouping system scored 35/80 on the same subset; 42/80 is a descriptive
seven-case improvement, not an independent confirmatory result.

## Local replication and additional ablation

The existing CPU replication covers 15 distinct seeds, varying the synthetic
generation, split and initialization while keeping the natural cases fixed.
Recomputed aggregates match the saved report:

| Metric, mean over 15 CPU seeds | GNN | MLP |
| --- | ---: | ---: |
| Synthetic validation exact set | 96.73% | 96.02% |
| Natural exact set | 23.25% | 31.17% |
| Natural + grouping rules | 43.08% | 50.83% |

The MLP wins on 14 seeds and the GNN on one. These are training-variability
replicates on a fixed evaluation subset, not 15 independent datasets. The
reported sign-test p=0.00098 and sign-flip p=0.00110 are nominal exploratory
values: seed batches were extended after inspecting earlier outcomes. They
should not be interpreted as a preregistered stopping-rule-adjusted test.

The learned-count/visual fusion analysis on the separate CPU base run changes
no exact-match outcomes for either model. Only three cases reach the visual
fallback after the grouping rules fire. This is a narrow negative ablation,
not evidence that visual features cannot help. The GPU fusion analysis was
not recovered or run; its per-case predictions did not survive the runtime.

For the next experiment, retain the MLP plus abstaining grouping rules as the
baseline and evaluate a frozen configuration on new collections. The graph
model's synthetic advantage does not establish natural transfer. These results
also do not isolate the cause of its transfer deficit.

## Evidence and reproduction

- [Saved Colab notebook](https://colab.research.google.com/drive/1JPkh4QMIrcBE4S6PBqXuDW5ubqNViT69).
- [Recovered executed notebook](../runs/colab-set-v4-recovered/executed_notebook.ipynb),
  [console summary](../runs/colab-set-v4-recovered/console_summary.json),
  [raw log](../runs/colab-set-v4-recovered/console_output.txt), and
  [recovery provenance and hashes](../runs/colab-set-v4-recovered/provenance.json).
- [CPU base run](../runs/local-set-v4/summary.json),
  [CPU fusion analysis](../runs/local-set-v4/cardinality_fusion.json), and
  [15-seed replication](../runs/set-v4-seeds/summary.json).
- [Portable Colab/Kaggle notebook](../notebooks/graph_moe_cloud.ipynb): select a
  GPU, enable Internet on Kaggle, and run all. It downloads immutable GitHub
  source, trains both arms, evaluates fusion, and bundles both checkpoints and
  all records into a downloadable ZIP.

The original GPU ZIP and checkpoints were absent after reconnecting on
September 10. Its complete console outputs were recovered from the saved
notebook. The local CPU checkpoints remain available, but are distinct models
and produce different base-run totals (GNN 14/80; MLP 26/80). Do not label those
weights as the recovered GPU models. Download future result ZIPs before Colab's
temporary runtime expires.

The 80 examples use aligned DOM-difference proxy labels. Grouping rules and
the complexity cutoff were developed using this already-inspected subset;
the cutoff also separates its two collections. No natural labels train these
checkpoints, but the reported system still requires collection-held-out
confirmation.
