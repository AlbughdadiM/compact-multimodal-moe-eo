# Analysis notebooks

Notebook outputs and execution counts are removed from the publication snapshot.
Set the following environment variables as required:

```bash
export MEOX_CHECKPOINT="$PWD/weights/pretrained/meox_s_mmearth64_best.pth"
export MEOX_CONFIG="$PWD/configs/pretrain_mmearth_moe_mae_full.yaml"
export MEOX_RUN_DIR="$PWD"
export MMEARTH64_ROOT=/path/to/MMEarth64
export GEO_BENCH_DIR=/path/to/geobench
export MEOX_EXPERIMENT_ROOT=/path/to/regenerated/experiment
```

The routing and MMEarth embedding notebooks require the official validation
data or a separately prepared fixture. The downstream visualization notebook
also requires regenerated test embedding/dense-feature caches; those large
intermediates are not distributed.

The notebooks cover:

- MMEarth embedding exploration and pretraining curves;
- deterministic expert routing, modality association, position NMI, functional
  similarity, and causal expert suppression;
- EuroSAT embedding selection, anisotropy, and normalization ablations;
- segmentation feature selection;
- downstream qualitative visualization and efficiency comparisons.

