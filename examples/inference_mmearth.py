#!/usr/bin/env python3
"""Extract one MEOX embedding from an official MMEarth64 sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets.mmearth import MMEarthDataset
from utils.extract_embeddings import build_model_from_config, load_config, resolve_device


def batched_tensors(values: dict[str, torch.Tensor], device: torch.device):
    return {name: tensor.unsqueeze(0).to(device) for name, tensor in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--config", default="configs/pretrain_mmearth_moe_mae_full.yaml"
    )
    parser.add_argument(
        "--checkpoint", default="weights/pretrained/meox_s_mmearth64_best.pth"
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["sentinel2", "sentinel1_asc", "sentinel1_desc"],
        choices=["sentinel2", "sentinel1_asc", "sentinel1_desc"],
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = load_config(args.config)
    training = config["training"]
    configured_bands = training.get("modality_bands", {})
    dataset = MMEarthDataset(
        root_dir=args.data_root,
        subset=training.get("dataset_subset", "MMEarth64"),
        split="val",
        raster_modalities=args.modalities,
        metadata_modalities=training.get("metadata_modalities", []),
        modality_bands={
            name: configured_bands[name]
            for name in args.modalities
            if name in configured_bands
        },
        normalization_mode=training.get("normalization_mode", "z-score"),
        fill_value=float(training.get("fill_value", 0.0)),
        fallback_split_from_train=bool(training.get("fallback_split_from_train", True)),
        val_fraction=float(training.get("val_fraction", 0.01)),
        split_seed=int(training.get("split_seed", 42)),
    )
    sample = dataset[args.index]
    model = build_model_from_config(config, args.checkpoint, device)

    raster_dict = batched_tensors(sample["raster_dict"], device)
    raster_valid_masks = batched_tensors(sample["raster_valid_masks"], device)
    meta_dict = batched_tensors(sample["meta_dict"], device)
    meta_valid_masks = batched_tensors(sample["meta_valid_masks"], device)

    with torch.inference_mode():
        embedding = model.extract_embedding(
            raster_dict=raster_dict,
            raster_valid_masks=raster_valid_masks,
            raster_band_names=sample["raster_band_names"],
            meta_dict=meta_dict,
            meta_valid_masks=meta_valid_masks,
            token_source="pre_norm",
            pooling="mean_fine",
        )

    print(f"tile_id={sample['tile_id']} date={sample['date']}")
    print(f"modalities={list(raster_dict)}")
    print(f"embedding_shape={tuple(embedding.shape)}")
    output = Path("outputs/example_embedding.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding.cpu(), output)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
