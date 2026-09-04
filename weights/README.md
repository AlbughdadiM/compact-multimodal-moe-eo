# Weights

This directory contains only validation-selected checkpoints:

- `pretrained/`: the MEOX-S MMEarth64 checkpoint used by every reported task.
- `downstream/classification/`: selected frozen linear heads.
- `downstream/segmentation/`: selected frozen UPerNet heads.
- `downstream/finetuning/`: adapted BigEarthNet encoder and classifier states.
- `downstream/metadata/`: selected WorldCover metadata-ablation heads.

Per-seed duplicates, optimizer states, milestone checkpoints, and embedding or
dense-feature caches are excluded. `manifest.json` records every file's byte
size and SHA-256 hash. Validate a checkout with:

```bash
sha256sum -c weights/SHA256SUMS
```

All `.pth` files are configured for Git LFS in the repository root.

