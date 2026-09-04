# Experiment results

The result tree preserves the raw JSON/CSV metrics used for the paper while
removing machine-specific absolute paths. It includes:

- full pretraining metrics and resolved configuration;
- frozen classification and segmentation results at 64 and 224 pixels;
- BigEarthNet full-encoder adaptation;
- BENv2-14k retrieval;
- held-out WorldCover metadata ablation;
- embedding, normalization, and dense-feature selection summaries.

`summary.csv` provides one row per primary downstream task and protocol. Full
result JSON files retain per-run metrics and validation-selection information.
Cached embeddings, dense features, predictions, and datasets are deliberately
excluded because the supplied scripts regenerate them.

