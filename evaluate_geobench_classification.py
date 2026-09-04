"""Extract frozen embeddings and run GEO-Bench classification probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from datasets.geobench import CLASSIFICATION_DATASETS, GeoBenchClassificationDataset
from utils.extract_embeddings import (
    build_model_from_config,
    extract_embeddings,
    load_config,
    make_inference_dataloader,
    maybe_limit_dataset,
    model_input_schema,
    resolve_device,
)
from utils.linear_probe import train_linear_probe


DEFAULT_LEARNING_RATES = {
    "m-bigearthnet": 1e-2,
    "m-brick-kiln": 3e-3,
    "m-so2sat": 1e-2,
    "m-eurosat": 5e-2,
}

PRIMARY_INPUT_NORMALIZATION = "geobench_official_per_band_zscore"
SPATIAL_PROTOCOLS = ("geobench_224", "model_input")


def _load_archive(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _extraction_signature(args, dataset) -> str:
    checkpoint_stat = args.checkpoint_path.stat()
    payload = {
        "version": 2,
        "preprocessing": dataset.preprocessing_signature,
        "input_normalization": PRIMARY_INPUT_NORMALIZATION,
        "checkpoint": str(args.checkpoint_path.resolve()),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "token_source": args.token_source,
        "pooling": args.pooling,
        "partition": args.partition,
        "s1_modality": args.s1_modality,
        "input_image_size": args.input_image_size,
        "raster_band_names": {
            name: list(names)
            for name, names in sorted(dataset.raster_band_names.items())
        },
        "max_samples": args.max_samples,
        "sample_seed": args.sample_seed,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _cache_matches_extraction(archive: dict, expected_signature: str) -> bool:
    cached = archive.get("extraction_signature")
    if cached is None:
        return False
    return str(np.asarray(cached).item()) == expected_signature


def _extract_split(
    args, model, model_band_names, dataset_name: str, split: str, dataset=None
):
    if dataset is None:
        dataset = GeoBenchClassificationDataset(
            root_dir=args.data_root,
            dataset_name=dataset_name,
            split=split,
            model_band_names=model_band_names,
            s1_modality=args.s1_modality,
            partition_name=args.partition,
            input_image_size=args.input_image_size,
        )
    extraction_dataset = maybe_limit_dataset(dataset, args.max_samples, args.sample_seed)
    dataloader = make_inference_dataloader(
        extraction_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    outputs = extract_embeddings(
        model=model,
        dataloader=dataloader,
        raster_band_names=dataset.raster_band_names,
        device=args.device,
        token_source=args.token_source,
        pooling=args.pooling,
    )
    return outputs, dataset


def evaluate_dataset(args, model, model_band_names, dataset_name: str) -> None:
    dataset_dir = args.output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    archives = {}
    dataset_info = None

    for split in ("train", "valid", "test"):
        archive_path = dataset_dir / f"{split}_embeddings.npz"
        split_dataset = GeoBenchClassificationDataset(
            root_dir=args.data_root,
            dataset_name=dataset_name,
            split=split,
            model_band_names=model_band_names,
            s1_modality=args.s1_modality,
            partition_name=args.partition,
            input_image_size=args.input_image_size,
        )
        extraction_signature = _extraction_signature(args, split_dataset)
        if args.reuse_embeddings and archive_path.exists():
            cached = _load_archive(archive_path)
            if _cache_matches_extraction(cached, extraction_signature):
                archives[split] = cached
                dataset_info = split_dataset
                continue
            print(f"Ignoring stale embedding cache: {archive_path}")

        outputs, dataset_info = _extract_split(
            args, model, model_band_names, dataset_name, split, dataset=split_dataset
        )
        outputs["preprocessing_signature"] = np.asarray(
            dataset_info.preprocessing_signature
        )
        outputs["extraction_signature"] = np.asarray(extraction_signature)
        np.savez_compressed(archive_path, **outputs)
        archives[split] = outputs

    source_bands = {
        modality: [source for source, _ in pairs]
        for modality, pairs in dataset_info.band_mapping.items()
    }
    manifest = {
        "dataset": dataset_name,
        "checkpoint": str(args.checkpoint_path.resolve()),
        "token_source": args.token_source,
        "pooling": args.pooling,
        "input_normalization": PRIMARY_INPUT_NORMALIZATION,
        "input_image_size": args.input_image_size or "native",
        "spatial_protocol": args.spatial_protocol,
        "nodata_policy": "raw_zero_and_nonfinite_are_invalid",
        "partition": args.partition,
        "s1_modality": args.s1_modality,
        "preprocessing_signature": dataset_info.preprocessing_signature,
        "source_bands": source_bands,
        "model_band_names": dataset_info.raster_band_names,
        "split_sizes": {
            split: int(data["embeddings"].shape[0]) for split, data in archives.items()
        },
        "probe_loss": (
            "balanced_binary_cross_entropy"
            if dataset_info.multilabel
            else "cross_entropy"
        ),
    }

    if not args.extract_only:
        head_checkpoint = dataset_dir / "best_head.pth"
        manifest["linear_probe"] = train_linear_probe(
            train=archives["train"],
            val=archives["valid"],
            test=archives["test"],
            num_classes=dataset_info.num_classes,
            multilabel=dataset_info.multilabel,
            learning_rate=DEFAULT_LEARNING_RATES[dataset_name],
            device=args.device,
            epochs=args.epochs,
            batch_size=args.probe_batch_size,
            seeds=args.seeds,
            checkpoint_path=head_checkpoint,
        )
        manifest["head_checkpoint"] = head_checkpoint.name

    with open(dataset_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\n{dataset_name}: {dataset_info.raster_band_names}")
    if "linear_probe" in manifest:
        for metric, values in manifest["linear_probe"]["aggregate"].items():
            print(f"  {metric}: {100 * values['mean']:.2f} +/- {100 * values['std']:.2f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen linear probing on the GEO-Bench classification tasks"
    )
    parser.add_argument("--config_yaml", type=Path, required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=CLASSIFICATION_DATASETS,
        default=list(CLASSIFICATION_DATASETS),
    )
    parser.add_argument("--partition", default="default")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--probe_batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--spatial_protocol",
        choices=SPATIAL_PROTOCOLS,
        default="geobench_224",
        help=(
            "Resize inputs to 224x224 for the GEO-Bench comparison, or to the "
            "checkpoint's configured image size"
        ),
    )
    parser.add_argument("--token_source", choices=["pre_norm", "post_norm"], default="pre_norm")
    parser.add_argument(
        "--pooling", choices=["cls", "mean_fine", "mean_all"], default="mean_fine"
    )
    parser.add_argument(
        "--s1_modality",
        choices=["sentinel1_asc", "sentinel1_desc"],
        default="sentinel1_asc",
    )
    parser.add_argument("--reuse_embeddings", action="store_true")
    parser.add_argument("--extract_only", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    args.device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(args.config_yaml))
    model_input_size = int(config["model"]["img_size"])
    args.input_image_size = (
        224 if args.spatial_protocol == "geobench_224" else model_input_size
    )
    if args.batch_size is None:
        args.batch_size = 1 if args.spatial_protocol == "geobench_224" else 64
    _, model_band_names, _ = model_input_schema(config)
    model = build_model_from_config(
        config=config,
        checkpoint_path=str(args.checkpoint_path),
        device=args.device,
    )
    for dataset_name in args.datasets:
        evaluate_dataset(args, model, model_band_names, dataset_name)


if __name__ == "__main__":
    main()
