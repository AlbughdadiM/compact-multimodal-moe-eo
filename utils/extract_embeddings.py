"""
Embedding extraction helpers and CLI for pretrained MoE-MAE checkpoints.

This module is used both by the command-line script and by the analysis
notebooks so the inference path stays consistent across workflows.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml
from tqdm import tqdm

from datasets.mmearth import MMEarthDataset
from models.moe_mae import MOEMAE, build_model


def resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    return obj


def maybe_limit_dataset(dataset, limit: Optional[int], seed: int):
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError("max_samples must be a positive integer")
    if len(dataset) <= limit:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def unwrap_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def infer_square_img_size_from_dataset(dataset) -> int:
    sample = dataset[0]
    raster_dict = sample["raster_dict"]
    shapes = {
        name: tuple(tensor.shape[-2:])
        for name, tensor in raster_dict.items()
        if torch.is_tensor(tensor) and tensor.ndim == 3
    }
    unique_shapes = set(shapes.values())
    if len(unique_shapes) != 1:
        raise ValueError(f"All raster modalities must share one spatial size, got {shapes}")
    height, width = next(iter(unique_shapes))
    if height != width:
        raise ValueError(f"Expected square raster inputs, got {(height, width)}")
    return int(height)


def load_config(config_yaml: str) -> dict:
    with open(config_yaml, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clone_config_with_modalities(
    config: dict,
    raster_modalities: Optional[Sequence[str]] = None,
    metadata_modalities: Optional[Sequence[str]] = None,
    modality_bands: Optional[Dict[str, Sequence[str]]] = None,
) -> dict:
    cloned = deepcopy(config)
    if raster_modalities is not None:
        cloned["training"]["raster_modalities"] = list(raster_modalities)
    if metadata_modalities is not None:
        cloned["training"]["metadata_modalities"] = list(metadata_modalities)
    if modality_bands is not None:
        cloned["training"]["modality_bands"] = {
            key: list(value) for key, value in modality_bands.items()
        }
    return cloned


def sample_indices(indices: Sequence[int], limit: Optional[int], seed: int) -> list[int]:
    selected = list(indices)
    if limit is None or len(selected) <= limit:
        return selected
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    generator = torch.Generator().manual_seed(seed)
    keep = torch.randperm(len(selected), generator=generator)[:limit].tolist()
    return [selected[idx] for idx in keep]


def _read_total_num_samples(dataset_path: str) -> int:
    try:
        import h5py  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MMEarth analysis helpers require 'h5py'. Add it to the environment before use."
        ) from exc
    with h5py.File(dataset_path, "r") as handle:
        return int(handle["metadata"].shape[0])


def resolve_mmearth_analysis_indices(config: dict, strategy: str = "non_train") -> list[int]:
    """
    Resolve dataset-global indices for held-out analysis.

    `non_train` uses every index not present in the official training split.
    This is stricter than reusing the pretraining validation subset because it
    excludes all training indices and includes official val/test leftovers.
    """
    probe_dataset = build_mmearth_dataset_from_config(
        config=config,
        split="train",
        max_samples=None,
        sample_seed=0,
        metadata_modalities=[],
    )
    dataset_info = unwrap_dataset(probe_dataset)

    with open(dataset_info.splits_path, "r", encoding="utf-8") as handle:
        split_indices = json.load(handle)

    if strategy == "non_train":
        total_num_samples = _read_total_num_samples(dataset_info.dataset_path)
        train_indices = {int(idx) for idx in split_indices.get("train", [])}
        analysis_indices = [idx for idx in range(total_num_samples) if idx not in train_indices]
        if not analysis_indices:
            raise ValueError(
                "No non-train analysis indices were found. The split file appears to assign "
                "every sample to train."
            )
        return analysis_indices

    if strategy == "val":
        val_indices = [int(idx) for idx in split_indices.get("val", [])]
        if val_indices:
            return val_indices
        raise ValueError("No official validation indices are available in the split file.")

    if strategy == "test":
        test_indices = [int(idx) for idx in split_indices.get("test", [])]
        if test_indices:
            return test_indices
        raise ValueError("No official test indices are available in the split file.")

    raise ValueError("strategy must be one of {'non_train', 'val', 'test'}")


def build_mmearth_dataset_from_config(
    config: dict,
    split: str,
    max_samples: Optional[int] = None,
    sample_seed: int = 42,
    raster_modalities: Optional[Sequence[str]] = None,
    metadata_modalities: Optional[Sequence[str]] = None,
    modality_bands: Optional[Dict[str, Sequence[str]]] = None,
    explicit_indices: Optional[Sequence[int]] = None,
):
    training_cfg = config["training"]
    configured_rasters = training_cfg.get(
        "raster_modalities", list(MMEarthDataset.raster_modalities)
    )
    configured_metadata = training_cfg.get(
        "metadata_modalities", list(MMEarthDataset.metadata_modalities)
    )
    dataset = MMEarthDataset(
        root_dir=training_cfg["dataset_path"],
        subset=training_cfg.get("dataset_subset", training_cfg.get("subset", "MMEarth64")),
        split=split,
        raster_modalities=(
            configured_rasters if raster_modalities is None else raster_modalities
        ),
        metadata_modalities=(
            configured_metadata if metadata_modalities is None else metadata_modalities
        ),
        modality_bands=(
            training_cfg.get("modality_bands")
            if modality_bands is None
            else modality_bands
        ),
        normalization_mode=training_cfg.get("normalization_mode", "z-score"),
        fill_value=float(training_cfg.get("fill_value", 0.0)),
        fallback_split_from_train=bool(training_cfg.get("fallback_split_from_train", True)),
        val_fraction=float(training_cfg.get("val_fraction", 0.1)),
        test_fraction=float(training_cfg.get("test_fraction", 0.0)),
        split_seed=int(training_cfg.get("split_seed", 42)),
        transform=None,
    )
    if explicit_indices is not None:
        dataset.indices = [int(idx) for idx in explicit_indices]
    return maybe_limit_dataset(dataset, max_samples, sample_seed)


def build_raster_band_names(dataset) -> Dict[str, list[str]]:
    dataset_info = unwrap_dataset(dataset)
    return {
        name: list(dataset_info.modality_bands[name]) for name in dataset_info.raster_modalities
    }


def model_input_schema(
    config: dict,
) -> tuple[Dict[str, int], Dict[str, list[str]], Dict[str, int]]:
    """Return the raster and metadata schema used to pretrain a model."""
    training_cfg = config["training"]
    raster_modalities = training_cfg.get("raster_modalities", ["sentinel2"])
    metadata_modalities = training_cfg.get("metadata_modalities", [])
    configured_bands = training_cfg.get("modality_bands", {})

    bands = {
        name: list(configured_bands.get(name, MMEarthDataset.all_modality_bands[name]))
        for name in [*raster_modalities, *metadata_modalities]
    }
    raster_band_names = {name: bands[name] for name in raster_modalities}
    input_adapters = {name: len(names) for name, names in raster_band_names.items()}
    metadata_dims = {name: len(bands[name]) for name in metadata_modalities}
    return input_adapters, raster_band_names, metadata_dims


def build_model_from_config(
    config: dict,
    checkpoint_path: str,
    device: torch.device,
) -> MOEMAE:
    """Build a pretrained model without opening the pretraining dataset."""
    input_adapters, input_band_names, metadata_dims = model_input_schema(config)
    training_cfg = config["training"]
    primary_input_name = training_cfg.get("primary_input_name", next(iter(input_adapters)))
    if primary_input_name not in input_adapters:
        raise ValueError(
            f"training.primary_input_name '{primary_input_name}' is not in the configured "
            f"raster modalities {sorted(input_adapters)}"
        )

    encoder = build_model(
        size=config["model"]["size"],
        img_size=int(config["model"]["img_size"]),
        patch_size=int(config["model"]["patch_size"]),
        moe_balance_weight=float(config["model"].get("moe_balance_weight", 1e-2)),
        input_adapters=input_adapters,
        input_band_names=input_band_names,
        primary_input_name=primary_input_name,
        metadata_dims=metadata_dims,
    )
    model = MOEMAE(encoder)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["model_state"]
        if isinstance(checkpoint, dict) and "model_state" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_model_from_dataset(
    config: dict,
    dataset,
    checkpoint_path: str,
    device: torch.device,
) -> MOEMAE:
    dataset_info = unwrap_dataset(dataset)
    input_adapters = dict(dataset_info.input_adapters)
    input_band_names = build_raster_band_names(dataset_info)
    metadata_dims = dict(dataset_info.metadata_dims)
    primary_input_name = config["training"].get(
        "primary_input_name", dataset_info.primary_input_name
    )
    if primary_input_name not in input_adapters:
        raise ValueError(
            f"training.primary_input_name '{primary_input_name}' is not in the dataset modalities "
            f"{sorted(input_adapters)}"
        )
    encoder = build_model(
        size=config["model"]["size"],
        img_size=infer_square_img_size_from_dataset(dataset),
        patch_size=config["model"]["patch_size"],
        moe_balance_weight=float(config["model"].get("moe_balance_weight", 1e-2)),
        input_adapters=input_adapters,
        input_band_names=input_band_names,
        primary_input_name=primary_input_name,
        metadata_dims=metadata_dims,
    )
    model = MOEMAE(encoder)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_state"]
        if isinstance(checkpoint, dict) and "model_state" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def make_inference_dataloader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": False,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(**loader_kwargs)


@torch.inference_mode()
def extract_embeddings(
    model: MOEMAE,
    dataloader: DataLoader,
    raster_band_names: Dict[str, list[str]],
    device: torch.device,
    token_source: str = "pre_norm",
    pooling: str = "mean_fine",
):
    model.eval()
    embeddings = []
    tile_ids = []
    dates = []
    sample_ids = []
    labels = []

    use_amp = device.type == "cuda"
    autocast_device = device.type if device.type in {"cuda", "cpu", "mps"} else "cpu"

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        raster_dict = move_to_device(batch["raster_dict"], device)
        raster_valid_masks = move_to_device(batch.get("raster_valid_masks"), device)
        meta_dict = move_to_device(batch.get("meta_dict"), device)
        meta_valid_masks = move_to_device(batch.get("meta_valid_masks"), device)

        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            batch_embeddings = model.extract_embedding(
                raster_dict=raster_dict,
                raster_valid_masks=raster_valid_masks,
                raster_band_names=raster_band_names,
                meta_dict=meta_dict,
                meta_valid_masks=meta_valid_masks,
                token_source=token_source,
                pooling=pooling,
            )

        embeddings.append(batch_embeddings.float().cpu())
        if "tile_id" in batch:
            tile_ids.extend([str(tile_id) for tile_id in batch["tile_id"]])
        if "date" in batch:
            dates.extend(["" if date is None else str(date) for date in batch["date"]])
        if "sample_id" in batch:
            sample_ids.extend([str(sample_id) for sample_id in batch["sample_id"]])
        if "label" in batch:
            label = batch["label"]
            label = label.detach().cpu() if torch.is_tensor(label) else torch.as_tensor(label)
            labels.append(label)

    if not embeddings:
        raise ValueError("Cannot extract embeddings from an empty dataloader")

    outputs = {
        "embeddings": torch.cat(embeddings, dim=0).numpy(),
    }
    if tile_ids:
        outputs["tile_ids"] = np.asarray(tile_ids, dtype=str)
    if dates:
        outputs["dates"] = np.asarray(dates, dtype=str)
    if sample_ids:
        outputs["sample_ids"] = np.asarray(sample_ids, dtype=str)
    if labels:
        outputs["labels"] = torch.cat(labels, dim=0).numpy()
    return outputs


def run_cli(args: argparse.Namespace) -> None:
    config = load_config(args.config_yaml)

    dataset_name = config["training"].get("dataset_name", "mmearth").lower()
    if dataset_name != "mmearth":
        raise ValueError(
            "The active extraction script currently supports only training.dataset_name='mmearth'"
        )

    device = resolve_device(args.device)
    dataset = build_mmearth_dataset_from_config(
        config=config,
        split=args.split,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
    )
    dataset_info = unwrap_dataset(dataset)
    batch_size = args.batch_size or config["training"].get(
        "val_batch_size", config["training"]["batch_size"]
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(config["training"].get("num_workers", 0))
    )

    model = build_model_from_dataset(
        config=config,
        dataset=dataset,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )
    dataloader = make_inference_dataloader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )
    raster_band_names = {
        name: list(dataset_info.modality_bands[name]) for name in dataset_info.raster_modalities
    }

    outputs = extract_embeddings(
        model=model,
        dataloader=dataloader,
        raster_band_names=raster_band_names,
        device=device,
        token_source=args.token_source,
        pooling=args.pooling,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    np.savez_compressed(args.output_path, **outputs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract encoder embeddings from a MoE-MAE checkpoint"
    )
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--token_source",
        type=str,
        default="pre_norm",
        choices=["pre_norm", "post_norm"],
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean_fine",
        choices=["cls", "mean_all", "mean_fine"],
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
