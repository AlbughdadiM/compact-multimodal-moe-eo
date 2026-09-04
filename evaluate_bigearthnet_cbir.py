"""Evaluate frozen multimodal embeddings on the CSMoE BENv2-14k CBIR protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.bigearthnet_cbir import (
    BENv2CBIRDataset,
    BENv2CBIRIndex,
    MMEarthBandNormalizer,
    PREPROCESSING_VERSION,
    S1_MODEL_BANDS,
    S2_SOURCE_BANDS_10,
    S2_SOURCE_BANDS_12,
    resolve_mmearth_stats_path,
)
from utils.extract_embeddings import (
    build_model_from_config,
    load_config,
    model_input_schema,
    resolve_device,
)
from utils.retrieval import (
    evaluate_csmoe_retrieval,
    shared_sensor_mean,
    transform_retrieval_embeddings,
)


SPATIAL_PROTOCOLS = ("model_input", "csmoe_224")
RETRIEVAL_TASKS = {
    "S1_to_S1": ("s1_embeddings", "s1_embeddings"),
    "S2_to_S2": ("s2_embeddings", "s2_embeddings"),
    "S1_to_S2": ("s1_embeddings", "s2_embeddings"),
    "S2_to_S1": ("s2_embeddings", "s1_embeddings"),
}


def _load_archive(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _cache_matches(archive: dict, expected_signature: str) -> bool:
    cached = archive.get("extraction_signature")
    return cached is not None and str(np.asarray(cached).item()) == expected_signature


def _extraction_signature(args, split: str, split_size: int) -> str:
    checkpoint_stat = args.checkpoint_path.stat()
    stats_stat = args.mmearth_stats.stat()
    payload = {
        "version": 1,
        "preprocessing": PREPROCESSING_VERSION,
        "data_root": str(args.data_root.resolve()),
        "metadata_path": None if args.metadata_path is None else str(args.metadata_path.resolve()),
        "split": split,
        "split_size": split_size,
        "checkpoint": str(args.checkpoint_path.resolve()),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "mmearth_stats": str(args.mmearth_stats.resolve()),
        "mmearth_stats_size": stats_stat.st_size,
        "mmearth_stats_mtime_ns": stats_stat.st_mtime_ns,
        "token_source": args.token_source,
        "pooling": args.pooling,
        "input_image_size": args.input_image_size,
        "max_samples": args.max_samples,
        "sample_seed": args.sample_seed,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _select_records(records, max_samples: int | None, seed: int):
    if max_samples is None or len(records) <= max_samples:
        return list(records)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=max_samples, replace=False))
    return [records[index] for index in indices]


def _make_dataloader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


@torch.inference_mode()
def extract_paired_embeddings(
    model,
    dataloader: DataLoader,
    device: torch.device,
    token_source: str,
    pooling: str,
) -> dict:
    """Extract aligned S1 and S2 embeddings while respecting each S1 pass adapter."""
    model.eval()
    s1_outputs = []
    s2_outputs = []
    labels = []
    sample_ids = []
    s1_adapters = []
    use_amp = device.type == "cuda"
    autocast_device = device.type if device.type in {"cuda", "cpu", "mps"} else "cpu"

    for batch in tqdm(dataloader, desc="Extracting paired BENv2 embeddings"):
        s2 = batch["s2"].to(device, non_blocking=True)
        s2_validity = batch["s2_validity"].to(device, non_blocking=True)
        s2_bands = (
            S2_SOURCE_BANDS_10 if s2.shape[1] == len(S2_SOURCE_BANDS_10) else S2_SOURCE_BANDS_12
        )
        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            s2_embedding = model.extract_embedding(
                raster_dict={"sentinel2": s2},
                raster_valid_masks={"sentinel2": s2_validity},
                raster_band_names={"sentinel2": list(s2_bands)},
                token_source=token_source,
                pooling=pooling,
            )

        adapter_names = list(batch["s1_adapter"])
        batch_s1 = None
        for adapter in sorted(set(adapter_names)):
            positions = [index for index, name in enumerate(adapter_names) if name == adapter]
            cpu_indices = torch.as_tensor(positions, dtype=torch.long)
            s1 = batch["s1"].index_select(0, cpu_indices).to(device, non_blocking=True)
            s1_validity = batch["s1_validity"].index_select(0, cpu_indices).to(
                device, non_blocking=True
            )
            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                embedding = model.extract_embedding(
                    raster_dict={adapter: s1},
                    raster_valid_masks={adapter: s1_validity},
                    raster_band_names={adapter: list(S1_MODEL_BANDS)},
                    token_source=token_source,
                    pooling=pooling,
                )
            if batch_s1 is None:
                batch_s1 = torch.empty(
                    len(adapter_names), embedding.shape[1], dtype=torch.float32
                )
            batch_s1.index_copy_(0, cpu_indices, embedding.float().cpu())

        s1_outputs.append(batch_s1)
        s2_outputs.append(s2_embedding.float().cpu())
        labels.append(batch["label"].to(dtype=torch.uint8))
        sample_ids.extend(str(sample_id) for sample_id in batch["sample_id"])
        s1_adapters.extend(adapter_names)

    if not s1_outputs:
        raise ValueError("Cannot extract embeddings from an empty BENv2 split")
    return {
        "s1_embeddings": torch.cat(s1_outputs).numpy(),
        "s2_embeddings": torch.cat(s2_outputs).numpy(),
        "labels": torch.cat(labels).numpy(),
        "sample_ids": np.asarray(sample_ids, dtype=str),
        "s1_adapters": np.asarray(s1_adapters, dtype=str),
    }


def _extract_or_load_split(args, model, index, normalizer, split: str) -> dict:
    path = args.output_dir / f"{split}_embeddings.npz"
    records = _select_records(index.records[split], args.max_samples, args.sample_seed)
    signature = _extraction_signature(args, split, len(records))
    if args.reuse_embeddings and path.exists():
        archive = _load_archive(path)
        if _cache_matches(archive, signature):
            print(f"Reusing {path}")
            return archive
        print(f"Ignoring stale embedding cache: {path}")

    dataset = BENv2CBIRDataset(
        records=records,
        normalizer=normalizer,
        input_image_size=args.input_image_size,
    )
    outputs = extract_paired_embeddings(
        model=model,
        dataloader=_make_dataloader(dataset, args.batch_size, args.num_workers),
        device=args.device,
        token_source=args.token_source,
        pooling=args.pooling,
    )
    outputs["extraction_signature"] = np.asarray(signature)
    np.savez_compressed(path, **outputs)
    return outputs


def _evaluate_representation(args, query: dict, archive: dict, mean=None) -> dict:
    transformed_query = {
        key: transform_retrieval_embeddings(query[key], mean=mean)
        for key in ("s1_embeddings", "s2_embeddings")
    }
    transformed_archive = {
        key: transform_retrieval_embeddings(archive[key], mean=mean)
        for key in ("s1_embeddings", "s2_embeddings")
    }
    results = {}
    for name, (query_key, archive_key) in RETRIEVAL_TASKS.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_csmoe_retrieval(
            query_embeddings=transformed_query[query_key],
            archive_embeddings=transformed_archive[archive_key],
            query_labels=query["labels"],
            archive_labels=archive["labels"],
            k=args.k,
            device=args.device,
            query_chunk_size=args.query_chunk_size,
            archive_chunk_size=args.archive_chunk_size,
        )
    return results


def _write_metrics_csv(path: Path, results: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "representation",
                "direction",
                "precision",
                "recall",
                "f1",
                "precision_percent",
                "recall_percent",
                "f1_percent",
                "k",
                "queries",
                "archive",
            ),
        )
        writer.writeheader()
        for representation, task_results in results.items():
            for direction, metrics in task_results.items():
                writer.writerow(
                    {
                        "representation": representation,
                        "direction": direction,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "precision_percent": 100 * metrics["precision"],
                        "recall_percent": 100 * metrics["recall"],
                        "f1_percent": 100 * metrics["f1"],
                        "k": metrics["k"],
                        "queries": metrics["queries"],
                        "archive": metrics["archive"],
                    }
                )


def _print_results(results: dict) -> None:
    for representation, task_results in results.items():
        label = "primary" if representation == "raw_l2" else "diagnostic"
        print(f"\n{representation} ({label})")
        print(f"{'direction':<12} {'precision':>10} {'recall':>10} {'F1':>10}")
        for direction, metrics in task_results.items():
            print(
                f"{direction:<12} {100 * metrics['precision']:>9.2f}% "
                f"{100 * metrics['recall']:>9.2f}% {100 * metrics['f1']:>9.2f}%"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen BENv2-14k unimodal and cross-modal CBIR evaluation"
    )
    parser.add_argument("--config_yaml", type=Path, required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, default=None)
    parser.add_argument("--mmearth_stats", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--spatial_protocol",
        choices=SPATIAL_PROTOCOLS,
        default="model_input",
        help="Use the checkpoint's native input size or the CSMoE 224x224 size",
    )
    parser.add_argument("--token_source", choices=("pre_norm", "post_norm"), default="pre_norm")
    parser.add_argument("--pooling", choices=("mean_fine", "cls", "mean_all"), default="mean_fine")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--query_chunk_size", type=int, default=512)
    parser.add_argument("--archive_chunk_size", type=int, default=8192)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--expected_pairs", type=int, default=13683)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--reuse_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--centered_ablation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate one shared mean estimated only from the training split",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config_yaml)
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.metadata_path = (
        None if args.metadata_path is None else args.metadata_path.expanduser().resolve()
    )
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.device = resolve_device(args.device)
    args.mmearth_stats = resolve_mmearth_stats_path(config, args.mmearth_stats)

    if args.k <= 0:
        raise ValueError("k must be positive")
    expected_pairs = None if args.expected_pairs == 0 else args.expected_pairs
    index = BENv2CBIRIndex(
        root_dir=args.data_root,
        metadata_path=args.metadata_path,
        expected_pairs=expected_pairs,
    )

    input_adapters, _, _ = model_input_schema(config)
    required = {"sentinel2", "sentinel1_asc", "sentinel1_desc"}
    missing = required - set(input_adapters)
    if missing:
        raise ValueError(f"Checkpoint input schema lacks required CBIR adapters: {sorted(missing)}")

    model_input_size = int(config["model"]["img_size"])
    args.input_image_size = 224 if args.spatial_protocol == "csmoe_224" else model_input_size
    if args.input_image_size % int(config["model"]["patch_size"]) != 0:
        raise ValueError("Resolved input size must be divisible by the model patch size")
    if args.batch_size is None:
        args.batch_size = 1 if args.spatial_protocol == "csmoe_224" else 128

    print(f"Using device: {args.device}")
    print(f"BENv2 split sizes: {index.split_sizes}")
    print(f"Input size: {args.input_image_size} ({args.spatial_protocol})")
    print(f"MMEarth statistics: {args.mmearth_stats}")

    model = build_model_from_config(config, args.checkpoint_path, args.device)
    normalizer = MMEarthBandNormalizer(args.mmearth_stats)
    splits = ("train", "validation", "test") if args.centered_ablation else ("validation", "test")
    archives = {
        split: _extract_or_load_split(args, model, index, normalizer, split)
        for split in splits
    }

    del model
    if args.device.type == "cuda":
        torch.cuda.empty_cache()

    results = {
        "raw_l2": _evaluate_representation(
            args,
            query=archives["validation"],
            archive=archives["test"],
        )
    }
    shared_mean = None
    if args.centered_ablation:
        shared_mean = shared_sensor_mean(
            archives["train"]["s1_embeddings"],
            archives["train"]["s2_embeddings"],
        )
        np.save(args.output_dir / "train_shared_mean.npy", shared_mean)
        results["train_mean_centered_l2"] = _evaluate_representation(
            args,
            query=archives["validation"],
            archive=archives["test"],
            mean=shared_mean,
        )

    manifest = {
        "dataset": "BENv2-14k",
        "data_root": str(index.root),
        "checkpoint": str(args.checkpoint_path),
        "preprocessing": PREPROCESSING_VERSION,
        "normalization": "MMEarth pretraining z-score statistics",
        "mmearth_stats": str(args.mmearth_stats),
        "metadata_policy": "all unavailable checkpoint metadata is explicitly missing",
        "token_source": args.token_source,
        "pooling": args.pooling,
        "spatial_protocol": args.spatial_protocol,
        "input_image_size": args.input_image_size,
        "query_split": "validation",
        "archive_split": "test",
        "mean_estimation_split": "train" if args.centered_ablation else None,
        "primary_representation": "raw_l2",
        "diagnostic_representation": (
            "train_mean_centered_l2" if args.centered_ablation else None
        ),
        "metric_definition": (
            "CSMoE pair-averaged multilabel precision and recall; F1 is their harmonic mean"
        ),
        "k": args.k,
        "split_sizes": index.split_sizes,
        "evaluated_split_sizes": {
            split: int(len(archive["labels"])) for split, archive in archives.items()
        },
        "class_names": index.class_names,
        "source_bands": {
            "sentinel1": list(S1_MODEL_BANDS),
            "sentinel2": list(S2_SOURCE_BANDS_10),
        },
        "s1_adapter_counts": {
            split: {
                adapter: int(np.sum(archive["s1_adapters"] == adapter))
                for adapter in np.unique(archive["s1_adapters"])
            }
            for split, archive in archives.items()
        },
        "results": results,
    }
    with open(args.output_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    _write_metrics_csv(args.output_dir / "metrics.csv", results)
    _print_results(results)


if __name__ == "__main__":
    main()
