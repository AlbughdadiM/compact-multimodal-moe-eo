"""Run frozen-backbone UPerNet probes on the CSMoE segmentation tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from datasets.geobench import SEGMENTATION_DATASETS, GeoBenchSegmentationDataset
from utils.extract_embeddings import (
    build_model_from_config,
    load_config,
    make_inference_dataloader,
    maybe_limit_dataset,
    model_input_schema,
    resolve_device,
)
from utils.segmentation_probe import (
    DENSE_FEATURE_CANDIDATES,
    dense_cache_matches,
    dense_extraction_signature,
    extract_dense_feature_cache,
    resolve_feature_layers,
    train_segmentation_probe,
)


DEFAULT_LEARNING_RATES = {
    "m-cashew-plant": 3e-4,
    "m-SA-crop-type": 1e-2,
}
SPATIAL_PROTOCOLS = ("geobench_224", "model_input")


def _build_dataset(args, model_band_names, dataset_name: str, split: str):
    return GeoBenchSegmentationDataset(
        root_dir=args.data_root,
        dataset_name=dataset_name,
        split=split,
        model_band_names=model_band_names,
        s1_modality=args.s1_modality,
        partition_name=args.partition,
        input_image_size=args.input_image_size,
        label_image_size=args.label_image_size,
    )


def _prepare_dense_cache(
    args,
    model,
    model_band_names,
    dataset_name: str,
    split: str,
    feature_layers,
    dataset_dir: Path,
):
    dataset = _build_dataset(args, model_band_names, dataset_name, split)
    extraction_dataset = maybe_limit_dataset(
        dataset, args.max_samples, args.sample_seed
    )
    cache_path = dataset_dir / f"{split}_dense_features.h5"
    signature = dense_extraction_signature(
        checkpoint_path=args.checkpoint_path,
        preprocessing_signature=dataset.preprocessing_signature,
        raster_band_names=dataset.raster_band_names,
        feature_layers=feature_layers,
        split=split,
        partition=args.partition,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
    )
    if args.reuse_dense_features and dense_cache_matches(cache_path, signature):
        return cache_path, dataset
    if cache_path.exists():
        print(f"Replacing stale dense feature cache: {cache_path}")

    dataloader = make_inference_dataloader(
        extraction_dataset,
        batch_size=args.extraction_batch_size,
        num_workers=args.num_workers,
    )
    extract_dense_feature_cache(
        model=model,
        dataloader=dataloader,
        raster_band_names=dataset.raster_band_names,
        device=args.device,
        feature_layers=feature_layers,
        output_path=cache_path,
        extraction_signature=signature,
    )
    return cache_path, dataset


def evaluate_dataset(
    args,
    model,
    model_band_names,
    dataset_name: str,
    feature_layers,
) -> None:
    dataset_dir = args.output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    caches = {}
    dataset_info = None
    for split in ("train", "valid", "test"):
        caches[split], dataset_info = _prepare_dense_cache(
            args,
            model,
            model_band_names,
            dataset_name,
            split,
            feature_layers,
            dataset_dir,
        )

    source_bands = {
        modality: [source for source, _ in pairs]
        for modality, pairs in dataset_info.band_mapping.items()
    }
    class_names = [
        str(name)
        for name in (dataset_info.class_names or range(dataset_info.num_classes))
    ]
    manifest = {
        "dataset": dataset_name,
        "task": "semantic_segmentation",
        "checkpoint": str(args.checkpoint_path.resolve()),
        "feature_candidate": args.feature_candidate,
        "feature_layers": list(feature_layers),
        "decoder": "terratorch_2025_plain_vit_upernet",
        "decoder_channels": args.decoder_channels,
        "input_image_size": args.input_image_size or "native",
        "label_image_size": args.label_image_size or "native",
        "spatial_protocol": args.spatial_protocol,
        "input_normalization": "geobench_official_per_band_zscore",
        "nodata_policy": "raw_zero_and_nonfinite_are_invalid",
        "training_augmentation": "none",
        "partition": args.partition,
        "preprocessing_signature": dataset_info.preprocessing_signature,
        "source_bands": source_bands,
        "model_band_names": dataset_info.raster_band_names,
        "num_classes": dataset_info.num_classes,
        "class_names": class_names,
    }
    if not args.extract_only:
        head_checkpoint = dataset_dir / "best_head.pth"
        manifest["segmentation_probe"] = train_segmentation_probe(
            train_cache=caches["train"],
            val_cache=caches["valid"],
            test_cache=caches["test"],
            num_classes=dataset_info.num_classes,
            candidate=args.feature_candidate,
            norm_weight=model.encoder.norm.weight,
            norm_bias=model.encoder.norm.bias,
            learning_rate=DEFAULT_LEARNING_RATES[dataset_name],
            device=args.device,
            head_type="upernet",
            epochs=args.epochs,
            batch_size=args.probe_batch_size,
            num_workers=args.num_workers,
            seeds=args.seeds,
            decoder_channels=args.decoder_channels,
            checkpoint_path=head_checkpoint,
            checkpoint_metadata={
                "encoder_checkpoint": str(args.checkpoint_path.resolve()),
                "feature_layers": list(feature_layers),
                "source_bands": source_bands,
                "model_band_names": dataset_info.raster_band_names,
                "class_names": class_names,
                "input_image_size": args.input_image_size or "native",
                "label_image_size": args.label_image_size or "native",
                "spatial_protocol": args.spatial_protocol,
                "preprocessing_signature": dataset_info.preprocessing_signature,
                "partition": args.partition,
            },
        )
        manifest["head_checkpoint"] = head_checkpoint.name
        manifest["seed_head_checkpoints"] = [
            str(Path(path).relative_to(dataset_dir))
            for path in manifest["segmentation_probe"]["seed_checkpoints"]
        ]

        history_path = dataset_dir / "optimization_history.csv"
        with open(history_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "seed",
                "epoch",
                "learning_rate",
                "train_cross_entropy",
                "validation_cross_entropy",
                "validation_mean_iou",
                "validation_pixel_accuracy",
                "validation_mean_accuracy",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for run in manifest["segmentation_probe"]["runs"]:
                for row in run["history"]:
                    writer.writerow({"seed": run["seed"], **row})
        manifest["optimization_history"] = history_path.name

    with open(dataset_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\n{dataset_name}: {dataset_info.raster_band_names}")
    if "segmentation_probe" in manifest:
        metrics = manifest["segmentation_probe"]["aggregate_test"]
        print(
            "  mean_iou: "
            f"{100 * metrics['mean_iou']['mean']:.2f} +/- "
            f"{100 * metrics['mean_iou']['std']:.2f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen UPerNet probing on the CSMoE GEO-Bench segmentation tasks"
    )
    parser.add_argument("--config_yaml", type=Path, required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SEGMENTATION_DATASETS,
        default=list(SEGMENTATION_DATASETS),
    )
    parser.add_argument("--partition", default="default")
    parser.add_argument("--extraction_batch_size", type=int, default=None)
    parser.add_argument("--probe_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--feature_layers",
        type=int,
        nargs=4,
        default=None,
        metavar=("L1", "L2", "L3", "L4"),
    )
    parser.add_argument(
        "--feature_candidate",
        choices=DENSE_FEATURE_CANDIDATES,
        default="final_pre_norm",
    )
    parser.add_argument("--decoder_channels", type=int, default=256)
    parser.add_argument(
        "--spatial_protocol",
        choices=SPATIAL_PROTOCOLS,
        default="geobench_224",
        help=(
            "Resize both images and labels to 224x224 for the GEO-Bench comparison, "
            "or resize images to the checkpoint size while retaining 224x224 labels"
        ),
    )
    parser.add_argument(
        "--s1_modality",
        choices=["sentinel1_asc", "sentinel1_desc"],
        default="sentinel1_asc",
    )
    parser.add_argument("--reuse_dense_features", action="store_true")
    parser.add_argument("--extract_only", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.decoder_channels <= 0:
        raise ValueError("decoder_channels must be positive")
    args.device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(args.config_yaml))
    model_input_size = int(config["model"]["img_size"])
    args.input_image_size = (
        224 if args.spatial_protocol == "geobench_224" else model_input_size
    )
    args.label_image_size = 224
    if args.extraction_batch_size is None:
        args.extraction_batch_size = (
            1 if args.spatial_protocol == "geobench_224" else 8
        )
    _, model_band_names, _ = model_input_schema(config)
    model = build_model_from_config(
        config=config,
        checkpoint_path=str(args.checkpoint_path),
        device=args.device,
    )
    feature_layers = (
        tuple(args.feature_layers)
        if args.feature_layers is not None
        else resolve_feature_layers(len(model.encoder.layers))
    )
    for dataset_name in args.datasets:
        evaluate_dataset(
            args,
            model,
            model_band_names,
            dataset_name,
            feature_layers,
        )


if __name__ == "__main__":
    main()
