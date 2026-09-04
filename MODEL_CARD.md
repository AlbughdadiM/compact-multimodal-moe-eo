# MEOX-S model card

## Model summary

MEOX-S is a compact multimodal masked autoencoder for Earth-observation imagery.
The encoder contains 2,938,897 parameters and returns 144-dimensional patch
features. The complete pretraining model, including its dense decoder, contains
3,114,993 parameters.

The configured raster inputs are:

- Sentinel-2: 13 bands (`B1` through `B12`, including `B8A`).
- Sentinel-1 ascending: `VV`, `VH`.
- Sentinel-1 descending: `VV`, `VH`.

The configured metadata groups are cyclic latitude, cyclic longitude, cyclic
month, and 12 ERA5 climate variables. Every raster, band, metadata group, and
individual value has an explicit missing-data representation.

## Architecture

Each available sensor is patchified by a sensor-specific adapter. A shared
transformer-MoE block processes the sensor streams independently. Learned
validity-aware fusion then combines the sensor representations at each patch.
Four metadata tokens and one CLS token are prepended before 14 additional
transformer-MoE blocks process the fused sequence.

Every MoE layer uses deterministic top-2 routing during evaluation. Experts
share value and output projections within a layer and have private gating
projections and rank-8 residual adapters. No expert capacity limit drops tokens.

## Training

- Data: 1,228,121 MMEarth64 training samples.
- Input size: `64 x 64` pixels.
- Patch size: `4 x 4` pixels.
- Spatial masking: 75%.
- Objective: modality-balanced valid-element masked MSE plus depth-averaged
  Switch-style routing balance loss.
- Sensor dropout: 10% of batches use a non-empty proper subset of sensors while
  reconstruction targets retain all available sensors.
- Optimizer: AdamW, peak learning rate `3e-4`, weight decay `0.05`.
- Schedule: 5% linear warmup followed by cosine decay over 50 epochs.
- Selection: minimum deterministic validation total loss.

## Recommended representation

Use raw fine tokens before the final encoder LayerNorm and average them spatially:

```python
embedding = model.extract_embedding(
    ...,
    token_source="pre_norm",
    pooling="mean_fine",
)
```

This produces a 144-dimensional image representation. For segmentation, use the
final pre-normalization fine-token map rather than a pooled image vector.

## Intended uses

- Frozen image-level classification probes.
- Frozen dense segmentation heads.
- Retrieval and embedding exploration.
- Task-specific finetuning of some or all encoder layers.
- Multisensor inference when one or more configured sensors are missing.

## Out-of-scope uses

- Treating outputs as calibrated physical measurements.
- Using arbitrary unseen sensors or bands without training an adapter.
- Assuming invariance to ground-sampling distance or spatial resampling.
- Assuming S1 and S2 embeddings occupy a fully aligned cross-sensor metric space.
- Operational or safety-critical decisions without task-specific validation.

## Data and preprocessing requirements

Inputs must use correct physical band names and dataset-compatible normalization.
Nodata and unavailable values must be supplied through validity masks instead of
being represented only as zeros. Sentinel-1 units, orbit semantics, and channel
order must match the selected adapter. Missing metadata should be omitted or
marked invalid rather than filled with plausible values.

## Evaluation summary

The frozen encoder was evaluated on four GEO-Bench classification tasks and two
segmentation tasks at 64 and 224 pixels. BigEarthNet full-encoder adaptation,
BENv2-14k retrieval, routing diagnostics, representation ablations, and a
held-out WorldCover metadata ablation were also performed. Exact result objects
are under `results/`; the main table is in `results/summary.csv`.

## Known limitations

The checkpoint was pretrained on one MMEarth release and one 64-pixel spatial
extent. Global attention makes large grids expensive. Frozen performance is
task-dependent, with a larger gap on BigEarthNet and SA crop type than on
EuroSAT and cashew. The learned embedding space is anisotropic, although
removing dominant principal components reduced semantic probe performance.
Cross-sensor retrieval is weaker than same-sensor retrieval.

## License

Code and distributed weights are released under the MIT license. MMEarth,
GEO-Bench, BENv2-14k, and their source imagery remain subject to their own terms.

