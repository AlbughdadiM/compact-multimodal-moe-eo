"""Frozen land-cover probes for metadata presence/absence on pretraining validation data."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import copy, deepcopy
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from utils.extract_embeddings import (
    build_mmearth_dataset_from_config,
    build_model_from_config,
    load_config,
    model_input_schema,
    move_to_device,
    resolve_device,
    sample_indices,
)
from utils.segmentation_probe import (
    DenseFeatureCacheDataset,
    DenseLinearProbe,
    SegmentationMetrics,
    load_segmentation_probe_checkpoint,
)


LABEL_MODALITY = "esa_worldcover"
IGNORE_INDEX = -100
CLASS_NAMES = (
    "Tree cover", "Shrubland", "Grassland", "Cropland", "Built-up",
    "Bare / sparse vegetation", "Snow and ice", "Permanent water",
    "Herbaceous wetland", "Mangroves", "Moss and lichen",
)
CONDITIONS = {
    "all": ("lat", "lon", "month", "era5"),
    "none": (),
    "no_location": ("month", "era5"),
    "no_month": ("lat", "lon", "era5"),
    "no_era5": ("lat", "lon", "month"),
}


def write_json(path, payload):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_labels(values, validity):
    """Preserve valid remapped class zero; ignore only explicitly invalid pixels."""
    values, validity = values.squeeze(0), validity.squeeze(0).bool()
    valid_values = values[validity]
    if not torch.isfinite(valid_values).all() or (
        (valid_values < 0) | (valid_values >= len(CLASS_NAMES))
        | (valid_values != valid_values.round())
    ).any():
        raise ValueError("WorldCover contains valid labels outside the expected classes 0..10")
    return torch.where(validity, values, IGNORE_INDEX).long()


class WorldCoverSamples(Dataset):
    """Remove the supervision raster before constructing any model input."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        rasters = dict(sample["raster_dict"])
        validity = dict(sample["raster_valid_masks"])
        labels = prepare_labels(rasters.pop(LABEL_MODALITY), validity.pop(LABEL_MODALITY))
        if any(tuple(image.shape[-2:]) != tuple(labels.shape) for image in rasters.values()):
            raise ValueError("This experiment requires aligned native-resolution images and labels")
        return {
            "raster_dict": rasters,
            "raster_valid_masks": validity,
            "meta_dict": sample["meta_dict"],
            "meta_valid_masks": sample["meta_valid_masks"],
            "label": labels,
            "tile_id": sample["tile_id"],
            "archive_index": int(self.dataset.indices[index]),
        }


def build_validation_dataset(config, max_samples, split_seed):
    training = config["training"]
    rasters = list(training["raster_modalities"])
    if LABEL_MODALITY in rasters:
        raise ValueError("WorldCover must not have been an encoder input for this experiment")
    if set(training["metadata_modalities"]) != set(CONDITIONS["all"]):
        raise ValueError("The five-condition protocol requires lat, lon, month, and era5")
    dataset = build_mmearth_dataset_from_config(
        config, split=training.get("val_split_name", "val"),
        raster_modalities=[*rasters, LABEL_MODALITY],
    )
    official = json.loads(Path(dataset.splits_path).read_text())
    train_view = copy(dataset)
    train_view.split = training.get("train_split_name", "train")
    train_indices = set(train_view._build_split_indices(official))
    if train_indices.intersection(dataset.indices):
        raise ValueError("Pretraining train/validation indices overlap; refusing this experiment")
    if len(set(dataset.indices)) != len(dataset.indices):
        raise ValueError("Pretraining validation contains duplicate archive indices")
    # Match the optional validation subset limit in pretrain_mae.py exactly.
    subset_seed = int(training.get("subset_sample_seed", training.get("split_seed", 42)))
    dataset.indices = sample_indices(
        dataset.indices, training.get("max_val_samples"), subset_seed + 1
    )
    original_count = len(dataset)
    dataset.indices = sample_indices(dataset.indices, max_samples, split_seed)
    if len(dataset) < 3:
        raise ValueError("At least three held-out samples are needed for probe train/val/test")
    return dataset, original_count


def location_records(dataset):
    """Use source coordinates, not normalized or artificially removed model metadata."""
    records = []
    with h5py.File(dataset.dataset_path, "r") as source:
        for index in dataset.indices:
            tile_id = dataset._decode_tile_id(source["metadata"][index])
            info = dataset.tile_info[tile_id]
            try:
                lat, lon = float(info["lat"]), float(info["lon"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Missing source coordinates for tile {tile_id}") from error
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"Invalid source coordinates for tile {tile_id}: {lat}, {lon}")
            records.append({
                "archive_index": int(index), "tile_id": tile_id,
                "location_group": f"{lat:.5f},{lon:.5f}",
            })
    return records


def split_locations(records, seed):
    groups = np.asarray([row["location_group"] for row in records])
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three distinct locations are needed for disjoint partitions")
    positions = np.arange(len(records))
    train, held_out = next(GroupShuffleSplit(
        n_splits=1, train_size=0.6, random_state=seed
    ).split(positions, groups=groups))
    val, test = next(GroupShuffleSplit(
        n_splits=1, test_size=0.5, random_state=seed + 1
    ).split(held_out, groups=groups[held_out]))
    return {"train": train.tolist(), "val": held_out[val].tolist(), "test": held_out[test].tolist()}


def metadata_for_condition(batch, condition):
    keep = CONDITIONS[condition]
    return (
        {name: value for name, value in batch["meta_dict"].items() if name in keep},
        {name: value for name, value in batch["meta_valid_masks"].items() if name in keep},
    )


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def feature_signature(config, checkpoint, dataset, device):
    payload = {
        "version": 1, "config": config, "device_type": device.type,
        "checkpoint": file_identity(checkpoint), "indices": dataset.indices,
        "sources": [file_identity(path) for path in (
            dataset.dataset_path, dataset.band_stats_path, dataset.tile_info_path, dataset.splits_path,
        )],
        "code": [file_identity(path) for path in (
            Path(__file__), Path(__file__).parent / "models/moe_mae.py",
            Path(__file__).parent / "datasets/mmearth.py",
            Path(__file__).parent / "utils/extract_embeddings.py",
        )],
        "conditions": CONDITIONS, "feature": "final_pre_norm", "storage_dtype": "float16",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@torch.inference_mode()
def cache_features(model, loader, band_names, paths, signature, device, reuse=False):
    pending = []
    for condition, path in paths.items():
        if path.exists():
            with h5py.File(path, "r") as cached:
                matches = cached.attrs.get("signature") == signature
                matches &= cached.attrs.get("condition") == condition
                matches &= cached["features"].shape[0] == len(loader.dataset)
            if not reuse or not matches:
                raise ValueError(f"Cache exists or is stale: {path}. Use matching --reuse_features or a new output_dir.")
        else:
            pending.append(condition)
    if not pending:
        print("Reusing all five matched feature caches.", flush=True)
        return
    model.eval().requires_grad_(False)
    valid_counts = {name: 0 for name in CONDITIONS["all"]}
    total_counts = dict(valid_counts)
    with ExitStack() as stack:
        stores = {}
        for condition in pending:
            temporary = paths[condition].with_suffix(".h5.tmp")
            stores[condition] = stack.enter_context(h5py.File(temporary, "w"))
        offset = 0
        for batch in tqdm(loader, desc="Frozen encoder: metadata conditions"):
            labels = batch["label"].numpy().astype(np.int16)
            count = len(labels)
            for name, mask in batch["meta_valid_masks"].items():
                valid_counts[name] += int(mask.sum())
                total_counts[name] += mask.numel()
            model_batch = move_to_device(batch, device)
            for condition, store in stores.items():
                metadata, validity = metadata_for_condition(model_batch, condition)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    output = model.forward_features(
                        raster_dict=model_batch["raster_dict"],
                        raster_valid_masks=model_batch["raster_valid_masks"],
                        raster_band_names=band_names,
                        meta_dict=metadata, meta_valid_masks=validity,
                        stochastic_routing=False,
                    )
                tokens = output["pre_norm_fine_tokens"]
                height, width = (int(output["token_layout"][key]) for key in ("fine_height", "fine_width"))
                features = tokens.transpose(1, 2).reshape(count, 1, tokens.shape[-1], height, width)
                features = features.to(dtype=torch.float16).cpu().numpy()
                if not np.isfinite(features).all():
                    raise ValueError(f"Non-finite cached features for metadata condition {condition}")
                if offset == 0:
                    for name, values in (("features", features), ("labels", labels)):
                        store.create_dataset(
                            name, shape=(len(loader.dataset), *values.shape[1:]), dtype=values.dtype,
                            chunks=(1, *values.shape[1:]), compression="lzf",
                        )
                    store.create_dataset("sample_ids", (len(loader.dataset),), dtype=h5py.string_dtype())
                    store.create_dataset("archive_indices", (len(loader.dataset),), dtype=np.int64)
                region = slice(offset, offset + count)
                store["features"][region] = features
                store["labels"][region] = labels
                store["sample_ids"][region] = batch["tile_id"]
                store["archive_indices"][region] = batch["archive_index"].numpy()
            offset += count
        if offset != len(loader.dataset):
            raise ValueError("Feature cache length differs from the selected validation dataset")
        for condition, store in stores.items():
            store.attrs["signature"] = signature
            store.attrs["condition"] = condition
            store.attrs["metadata_valid_fraction"] = json.dumps({
                name: valid_counts[name] / max(total_counts[name], 1) for name in valid_counts
            })
    for condition in pending:
        paths[condition].with_suffix(".h5.tmp").replace(paths[condition])


def make_loader(dataset, batch_size, workers, shuffle=False, seed=0):
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=workers > 0, pin_memory=torch.cuda.is_available(),
    )


def masked_cross_entropy(logits, labels):
    valid_count = int((labels != IGNORE_INDEX).sum())
    if not valid_count:
        return None, 0
    loss = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="sum") / valid_count
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite linear-probe loss")
    return loss, valid_count


@torch.inference_mode()
def evaluate_head(head, loader, device, prediction_path=None):
    head.eval()
    metrics = SegmentationMetrics(len(CLASS_NAMES))
    total_loss, pixel_count = 0.0, 0
    predictions = []
    for features, labels in loader:
        labels = labels.to(device)
        logits = head(features.to(device=device, dtype=torch.float32), labels.shape[-2:])
        loss, count = masked_cross_entropy(logits, labels)
        if count:
            total_loss += float(loss) * count
            pixel_count += count
        metrics.update(logits, labels)
        if prediction_path is not None:
            predictions.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
    if not pixel_count:
        raise ValueError("Partition has no valid WorldCover pixels")
    result = metrics.compute()
    support = metrics.confusion.sum(1).numpy()
    # Fix the mIoU class set from GT, rather than letting predictions change its denominator.
    result["mean_iou"] = float(np.mean([
        result["per_class_iou"][index] for index in np.flatnonzero(support)
    ]))
    result.update(cross_entropy=total_loss / pixel_count, valid_pixels=pixel_count,
                  class_support=support.tolist(), confusion_matrix=metrics.confusion.tolist())
    if prediction_path is not None:
        np.save(prediction_path, np.concatenate(predictions))
    return result


def train_head(cache, partitions, condition, args, inference_info):
    torch.manual_seed(args.seed)
    head = DenseLinearProbe(cache.feature_dim, len(CLASS_NAMES)).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = make_loader(Subset(cache, partitions["train"]), args.batch_size, args.num_workers, True, args.seed)
    val_loader = make_loader(Subset(cache, partitions["val"]), args.batch_size, args.num_workers)
    output = args.output_dir / condition
    output.mkdir(exist_ok=True)
    best_score, best_epoch, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss, pixels = 0.0, 0
        for features, labels in train_loader:
            labels = labels.to(args.device)
            logits = head(features.to(device=args.device, dtype=torch.float32), labels.shape[-2:])
            loss, count = masked_cross_entropy(logits, labels)
            if not count:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * count
            pixels += count
        if not pixels:
            raise ValueError("Probe training partition has no valid labels")
        validation = evaluate_head(head, val_loader, args.device)
        history.append({
            "epoch": epoch, "train_cross_entropy": total_loss / pixels,
            "validation_cross_entropy": validation["cross_entropy"],
            "validation_mean_iou": validation["mean_iou"],
        })
        write_csv(output / "history.csv", history)
        print(f"{condition} epoch {epoch}/{args.epochs} train={total_loss / pixels:.4f} "
              f"val={validation['cross_entropy']:.4f} mIoU={validation['mean_iou']:.4f}", flush=True)
        if validation["mean_iou"] > best_score:
            best_score, best_epoch = validation["mean_iou"], epoch
            payload = {
                "format_version": 2, "task": "semantic_segmentation", "head_type": "linear",
                "head_state_dict": {name: value.detach().cpu() for name, value in head.state_dict().items()},
                "feature_dim": cache.feature_dim, "num_classes": len(CLASS_NAMES),
                "candidate": "final_pre_norm", "condition": condition,
                "available_metadata": list(CONDITIONS[condition]), "seed": args.seed,
                "best_epoch": best_epoch, "best_validation_score": best_score,
                "selection_metric": "mean_iou_gt_present", "inference": inference_info,
            }
            temporary = output / "best_head.pth.tmp"
            torch.save(payload, temporary)
            temporary.replace(output / "best_head.pth")
        if args.patience and epoch - best_epoch >= args.patience:
            break
    head, _ = load_segmentation_probe_checkpoint(output / "best_head.pth", args.device)
    return head, {"best_epoch": best_epoch, "validation": evaluate_head(head, val_loader, args.device)}


def save_visualizations(dataset, partitions, paths, output_dir, count):
    if not count:
        return
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    count = min(count, len(partitions["test"]))
    palette = plt.get_cmap("tab20")(np.arange(len(CLASS_NAMES)))
    cmap = ListedColormap(palette)
    cmap.set_bad("#777777")
    norm = BoundaryNorm(np.arange(len(CLASS_NAMES) + 1) - 0.5, len(CLASS_NAMES))
    all_predictions = np.load(output_dir / "all/test_predictions.npy", mmap_mode="r")
    no_predictions = np.load(output_dir / "none/test_predictions.npy", mmap_mode="r")
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    with h5py.File(paths["all"], "r") as cache, h5py.File(dataset.dataset_path, "r") as source:
        for test_position, cache_position in enumerate(partitions["test"][:count]):
            index = dataset.indices[cache_position]
            rgb_indices = [dataset.all_modality_bands["sentinel2"].index(band) for band in ("B4", "B3", "B2")]
            rgb = np.asarray(source["sentinel2"][index])[rgb_indices].transpose(1, 2, 0).astype(float)
            for channel in range(3):
                values = rgb[..., channel]
                valid = np.isfinite(values) & (values != 0)
                lo, hi = np.percentile(values[valid], (2, 98)) if valid.any() else (0, 1)
                rgb[..., channel] = np.where(valid, np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1), 0)
            labels = cache["labels"][cache_position]
            invalid = labels == IGNORE_INDEX
            full, absent = all_predictions[test_position], no_predictions[test_position]
            figure, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes[0, 0].imshow(rgb)
            for axis, values in ((axes[0, 1], labels), (axes[0, 2], full), (axes[1, 0], absent)):
                axis.imshow(np.ma.masked_where(invalid, values), cmap=cmap, norm=norm, interpolation="nearest")
            for axis, values in ((axes[1, 1], full), (axes[1, 2], absent)):
                axis.imshow(np.ma.masked_where(invalid, values != labels), cmap="RdYlGn_r", vmin=0, vmax=1)
            for axis, title in zip(axes.flat, ("S2 RGB", "WorldCover GT", "All metadata", "No metadata", "All: errors (red)", "None: errors (red)")):
                axis.set_title(title)
                axis.axis("off")
            figure.suptitle(f"Held-out probe test sample | archive index {index}")
            figure.legend(handles=[Patch(color=palette[i], label=name) for i, name in enumerate(CLASS_NAMES)],
                          loc="lower center", ncol=4, fontsize=8)
            figure.tight_layout(rect=(0, 0.1, 1, 0.95))
            figure.savefig(figure_dir / f"sample_{index}.png", dpi=140)
            plt.close(figure)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", type=Path, required=True, help="Resolved pretraining run config")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, help="Override only the MMEarth archive location")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_samples", type=int, help="Optional smoke subset of pretraining validation")
    parser.add_argument("--extraction_batch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10, help="Validation mIoU patience; 0 disables stopping")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--reuse_features", action="store_true")
    parser.add_argument("--extract_only", action="store_true")
    parser.add_argument("--visualization_samples", type=int, default=4)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    for name in ("extraction_batch_size", "batch_size", "epochs", "learning_rate"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("num_workers", "weight_decay", "patience", "visualization_samples"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.max_samples is not None and args.max_samples < 3:
        raise ValueError("max_samples must be at least 3")
    args.device = resolve_device(args.device)
    config = deepcopy(load_config(str(args.config_yaml)))
    if args.data_root:
        config["training"]["dataset_path"] = str(args.data_root)
    dataset, validation_count = build_validation_dataset(config, args.max_samples, args.split_seed)
    records = location_records(dataset)
    partitions = split_locations(records, args.split_seed)
    for name, positions in partitions.items():
        for position in positions:
            records[position]["probe_split"] = name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, band_names, _ = model_input_schema(config)
    signature = feature_signature(config, args.checkpoint_path, dataset, args.device)
    paths = {name: args.output_dir / f"features_{name}.h5" for name in CONDITIONS}
    model = build_model_from_config(config, str(args.checkpoint_path), args.device)
    model.eval().requires_grad_(False)
    model_info = {
        "encoder_checkpoint": str(args.checkpoint_path.resolve()), "model_config": config,
        "feature_layers": [len(model.encoder.layers) - 1], "candidate": "final_pre_norm",
        "model_band_names": band_names, "class_names": list(CLASS_NAMES),
        "ignore_index": IGNORE_INDEX, "input_normalization": "unchanged_mmearth_pretraining_stats",
    }
    print(f"Device: {args.device}; pretraining validation={validation_count}; selected={len(dataset)}; "
          f"probe partitions={ {name: len(ids) for name, ids in partitions.items()} }", flush=True)
    print("Encoder frozen; no MAE decoder, pretraining, augmentation, or raster dropout.", flush=True)
    loader = make_loader(WorldCoverSamples(dataset), args.extraction_batch_size, args.num_workers)
    cache_features(model, loader, band_names, paths, signature, args.device, args.reuse_features)
    write_json(args.output_dir / "partitions.json", {"records": records, "positions": partitions})
    del loader, model
    if args.device.type == "cuda":
        torch.cuda.empty_cache()
    with h5py.File(paths["all"], "r") as handle:
        availability = json.loads(handle.attrs["metadata_valid_fraction"])
        native_size = list(handle["labels"].shape[-2:])
        feature_grid = list(handle["features"].shape[-2:])
    model_info.update(input_image_size=native_size, label_image_size=native_size, feature_grid=feature_grid)
    manifest = {
        "task": "mmearth_worldcover_metadata_ablation", "encoder_frozen": True,
        "pretraining_validation_count": validation_count, "selected_count": len(dataset),
        "conditions": CONDITIONS, "feature_signature": signature,
        "metadata_valid_fraction": availability, "inference": model_info,
        "partitions": {name: len(ids) for name, ids in partitions.items()},
        "split_rule": "60/20/20 by location groups; source lat/lon rounded to 5 decimals",
        "miou_classes": "classes with GT support in each evaluation partition",
        "arguments": {key: str(value) if isinstance(value, (Path, torch.device)) else value
                      for key, value in vars(args).items()},
    }
    write_json(args.output_dir / "experiment.json", manifest)
    if args.extract_only:
        return
    matched, removal = {}, {}
    for condition, path in paths.items():
        cache = DenseFeatureCacheDataset(path)
        head, result = train_head(cache, partitions, condition, args, model_info)
        test_loader = make_loader(Subset(cache, partitions["test"]), args.batch_size, args.num_workers)
        result["test"] = evaluate_head(head, test_loader, args.device, args.output_dir / condition / "test_predictions.npy")
        matched[condition] = result
        write_json(args.output_dir / condition / "results.json", result)
        del head, test_loader, cache
    all_head, _ = load_segmentation_probe_checkpoint(args.output_dir / "all/best_head.pth", args.device)
    for condition, path in paths.items():
        cache = DenseFeatureCacheDataset(path)
        loader = make_loader(Subset(cache, partitions["test"]), args.batch_size, args.num_workers)
        removal[condition] = evaluate_head(all_head, loader, args.device)
        del loader, cache
    write_json(args.output_dir / "results.json", {"matched": matched, "test_time_removal": removal})
    rows, class_rows = [], []
    for experiment, metrics_by_condition in (
        ("matched", {name: result["test"] for name, result in matched.items()}),
        ("test_time_removal", removal),
    ):
        reference = metrics_by_condition["all"]["mean_iou"]
        for condition, metrics in metrics_by_condition.items():
            rows.append({
                "experiment": experiment, "metadata": condition,
                "mean_iou_pct": 100 * metrics["mean_iou"],
                "delta_miou_pp": 100 * (metrics["mean_iou"] - reference),
                "pixel_accuracy_pct": 100 * metrics["pixel_accuracy"],
                "cross_entropy": metrics["cross_entropy"],
            })
            for index, name in enumerate(CLASS_NAMES):
                iou = metrics["per_class_iou"][index]
                class_rows.append({"experiment": experiment, "metadata": condition, "class": name,
                                   "iou_pct": None if iou is None else 100 * iou,
                                   "gt_pixels": metrics["class_support"][index]})
    write_csv(args.output_dir / "summary.csv", rows)
    write_csv(args.output_dir / "per_class_iou.csv", class_rows)
    save_visualizations(dataset, partitions, paths, args.output_dir, args.visualization_samples)
    for row in rows:
        print(f"{row['experiment']:18} {row['metadata']:12} mIoU={row['mean_iou_pct']:.2f}% "
              f"delta={row['delta_miou_pp']:+.2f} pp", flush=True)


if __name__ == "__main__":
    main()
