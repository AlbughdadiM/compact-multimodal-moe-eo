"""Finetune all or selected encoder blocks on GEO-Bench BigEarthNet."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import shutil
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.geobench import GeoBenchClassificationDataset
from utils.classification_finetune import (
    ClassificationHead,
    build_classification_finetuning_optimizer,
    configure_encoder_trainability,
    forward_classification,
    set_finetuning_mode,
)
from utils.extract_embeddings import (
    build_model_from_config,
    load_config,
    maybe_limit_dataset,
    model_input_schema,
    move_to_device,
    resolve_device,
)
from utils.linear_probe import balanced_binary_cross_entropy, classification_metrics


DATASET_NAME = "m-bigearthnet"
SPATIAL_PROTOCOLS = ("geobench_224", "model_input")


class ClassificationGeometricAugmentation(Dataset):
    """Apply one of eight square rotations/reflections to training rasters."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        transform_index = int(torch.randint(0, 8, ()).item())
        quarter_turns = transform_index % 4
        reflect = transform_index >= 4

        def transform(tensor: torch.Tensor) -> torch.Tensor:
            output = torch.rot90(tensor, quarter_turns, dims=(-2, -1))
            return torch.flip(output, dims=(-1,)) if reflect else output

        return {
            **sample,
            "raster_dict": {
                name: transform(value) for name, value in sample["raster_dict"].items()
            },
            "raster_valid_masks": {
                name: transform(value)
                for name, value in sample["raster_valid_masks"].items()
            },
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": torch.Generator().manual_seed(seed) if shuffle else None,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


def _build_datasets(args, model_band_names: dict) -> tuple[dict, object]:
    datasets = {}
    dataset_info = None
    for split in ("train", "valid", "test"):
        dataset = GeoBenchClassificationDataset(
            root_dir=args.data_root,
            dataset_name=DATASET_NAME,
            split=split,
            model_band_names=model_band_names,
            s1_modality=args.s1_modality,
            partition_name=args.partition,
            input_image_size=args.input_image_size,
        )
        if not dataset.multilabel:
            raise ValueError("m-bigearthnet must be a multilabel classification task")
        if dataset_info is None:
            dataset_info = dataset
        elif dataset.raster_band_names != dataset_info.raster_band_names:
            raise ValueError("GEO-Bench splits resolved different raster bands")
        datasets[split] = maybe_limit_dataset(
            dataset, args.max_samples, args.sample_seed
        )
    if args.geometric_augmentation:
        datasets["train"] = ClassificationGeometricAugmentation(datasets["train"])
    return datasets, dataset_info


@torch.inference_mode()
def _evaluate(
    model,
    head,
    dataloader: DataLoader,
    raster_band_names: dict,
    device: torch.device,
) -> dict:
    model.eval()
    head.eval()
    logits_batches = []
    label_batches = []
    loss_sum = 0.0
    moe_sum = 0.0
    sample_count = 0
    use_amp = device.type == "cuda"

    for batch in dataloader:
        raster_dict = move_to_device(batch["raster_dict"], device)
        raster_valid_masks = move_to_device(batch["raster_valid_masks"], device)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, moe_loss = forward_classification(
                model=model,
                head=head,
                raster_dict=raster_dict,
                raster_valid_masks=raster_valid_masks,
                raster_band_names=raster_band_names,
                encoder_grad_enabled=False,
                stochastic_routing=False,
            )
            loss = balanced_binary_cross_entropy(logits, labels)
        batch_size = labels.shape[0]
        loss_sum += float(loss) * batch_size
        moe_sum += float(moe_loss) * batch_size
        sample_count += batch_size
        logits_batches.append(logits.float().cpu())
        label_batches.append(labels.float().cpu())

    if not logits_batches:
        raise ValueError("Evaluation loader is empty")
    logits_array = torch.cat(logits_batches).numpy()
    labels_array = torch.cat(label_batches).numpy()
    metrics = classification_metrics(logits_array, labels_array, multilabel=True)
    metrics["balanced_binary_cross_entropy"] = loss_sum / sample_count
    metrics["moe_loss"] = moe_sum / sample_count
    return metrics


def _cpu_state_dict(module: torch.nn.Module) -> dict:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _checkpoint_payload(
    args,
    model,
    head,
    trainability: dict,
    dataset_info,
    class_names: Sequence[str],
    seed: int,
    epoch: int,
    validation: dict,
) -> dict:
    source_bands = {
        modality: [source for source, _ in pairs]
        for modality, pairs in dataset_info.band_mapping.items()
    }
    return {
        "format_version": 1,
        "task": "multilabel_classification_finetune",
        "dataset": DATASET_NAME,
        "encoder_state_dict": _cpu_state_dict(model.encoder),
        "head_state_dict": _cpu_state_dict(head),
        "feature_dim": int(model.encoder.embed_dim),
        "num_classes": int(dataset_info.num_classes),
        "class_names": list(class_names),
        "head_dropout": float(args.head_dropout),
        "feature_candidate": "final_pre_norm_mean_fine",
        "base_checkpoint": str(args.checkpoint_path),
        "trainability": dict(trainability),
        "seed": int(seed),
        "best_epoch": int(epoch),
        "selection_metric": "validation_micro_map",
        "best_validation": validation,
        "inference": {
            "config_yaml": str(args.config_yaml),
            "raster_band_names": dataset_info.raster_band_names,
            "source_bands": source_bands,
            "spatial_protocol": args.spatial_protocol,
            "input_image_size": int(args.input_image_size),
            "partition": args.partition,
            "preprocessing_signature": dataset_info.preprocessing_signature,
            "metadata_policy": "all checkpoint metadata explicitly missing",
        },
        "optimization": {
            "encoder_learning_rate": float(args.encoder_learning_rate),
            "head_learning_rate": float(args.head_learning_rate),
            "encoder_weight_decay": float(args.encoder_weight_decay),
            "head_weight_decay": float(args.head_weight_decay),
            "layer_decay": float(args.layer_decay),
            "moe_loss_scale": float(args.moe_loss_scale),
            "head_warmup_epochs": int(args.head_warmup_epochs),
            "freeze_routers": bool(args.freeze_routers),
            "geometric_augmentation": bool(args.geometric_augmentation),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "gradient_clip_norm": float(args.gradient_clip_norm),
        },
    }


def _build_scheduler(args, optimizer):
    def warmup_cosine(epoch_index: int) -> float:
        if args.lr_warmup_epochs > 0 and epoch_index < args.lr_warmup_epochs:
            return float(epoch_index + 1) / args.lr_warmup_epochs
        decay_epochs = max(args.epochs - args.lr_warmup_epochs - 1, 1)
        progress = min(
            max(epoch_index - args.lr_warmup_epochs, 0) / decay_epochs,
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (
            args.minimum_learning_rate_ratio
            + (1.0 - args.minimum_learning_rate_ratio) * cosine
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)


def _train_one_seed(
    args,
    config: dict,
    datasets: dict,
    dataset_info,
    class_names: Sequence[str],
    seed: int,
) -> dict:
    _seed_everything(seed)
    model = build_model_from_config(
        config=config,
        checkpoint_path=str(args.checkpoint_path),
        device=args.device,
    )
    trainability = configure_encoder_trainability(
        model,
        args.trainable_encoder_layers,
        freeze_routers=args.freeze_routers,
    )
    head = ClassificationHead(
        feature_dim=model.encoder.embed_dim,
        num_classes=dataset_info.num_classes,
        dropout=args.head_dropout,
    ).to(args.device)
    optimizer = build_classification_finetuning_optimizer(
        model=model,
        head=head,
        encoder_learning_rate=args.encoder_learning_rate,
        head_learning_rate=args.head_learning_rate,
        encoder_weight_decay=args.encoder_weight_decay,
        head_weight_decay=args.head_weight_decay,
        layer_decay=args.layer_decay,
    )
    scheduler = _build_scheduler(args, optimizer)
    scaler = torch.amp.GradScaler(args.device.type, enabled=args.device.type == "cuda")
    has_trainable_encoder = trainability["trainable_parameters"] > 0

    train_loader = _make_loader(
        datasets["train"], args.batch_size, args.num_workers, shuffle=True, seed=seed
    )
    eval_batch_size = args.evaluation_batch_size or args.batch_size
    val_loader = _make_loader(
        datasets["valid"], eval_batch_size, args.num_workers, shuffle=False
    )
    test_loader = _make_loader(
        datasets["test"], eval_batch_size, args.num_workers, shuffle=False
    )
    if len(train_loader) == 0:
        raise ValueError("The training loader is empty")

    print(
        f"Seed {seed} | encoder={trainability['mode']} "
        f"layers={trainability['layer_indices']} "
        f"trainable={trainability['trainable_parameters']:,}/"
        f"{trainability['total_parameters']:,} head="
        f"{sum(parameter.numel() for parameter in head.parameters()):,}"
    )

    seed_checkpoint = args.output_dir / "checkpoints" / f"seed_{seed}_best.pth"
    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    use_amp = args.device.type == "cuda"
    num_batches = len(train_loader)

    epoch_iterator = tqdm(range(1, args.epochs + 1), desc=f"seed {seed}")
    for epoch in epoch_iterator:
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        train_objective_sum = 0.0
        train_moe_sum = 0.0
        sample_count = 0
        encoder_grad_enabled = has_trainable_encoder and epoch > args.head_warmup_epochs
        set_finetuning_mode(
            model,
            head,
            trainability,
            freeze_batch_norm=False,
            encoder_training_enabled=encoder_grad_enabled,
        )

        for batch_index, batch in enumerate(train_loader):
            raster_dict = move_to_device(batch["raster_dict"], args.device)
            raster_valid_masks = move_to_device(
                batch["raster_valid_masks"], args.device
            )
            labels = batch["label"].to(args.device, non_blocking=True)
            window_start = (
                batch_index // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            window_size = min(
                args.gradient_accumulation_steps, num_batches - window_start
            )

            with torch.amp.autocast(device_type=args.device.type, enabled=use_amp):
                logits, moe_loss = forward_classification(
                    model=model,
                    head=head,
                    raster_dict=raster_dict,
                    raster_valid_masks=raster_valid_masks,
                    raster_band_names=dataset_info.raster_band_names,
                    encoder_grad_enabled=encoder_grad_enabled,
                    stochastic_routing=not args.freeze_routers,
                )
                classification_loss = balanced_binary_cross_entropy(logits, labels)
                objective = classification_loss
                if encoder_grad_enabled and args.moe_loss_scale > 0:
                    objective = objective + args.moe_loss_scale * moe_loss

            if not torch.isfinite(objective):
                raise RuntimeError(
                    f"Non-finite objective at seed {seed}, epoch {epoch}, "
                    f"batch {batch_index}"
                )
            scaler.scale(objective / window_size).backward()
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == num_batches
            )
            if should_step:
                if args.gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    trainable_parameters = [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                        if parameter.grad is not None
                    ]
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, args.gradient_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = labels.shape[0]
            train_loss_sum += float(classification_loss.detach()) * batch_size
            train_objective_sum += float(objective.detach()) * batch_size
            train_moe_sum += float(moe_loss.detach()) * batch_size
            sample_count += batch_size

        validation = _evaluate(
            model,
            head,
            val_loader,
            dataset_info.raster_band_names,
            args.device,
        )
        row = {
            "epoch": int(epoch),
            "encoder_learning_rate": float(
                max(
                    (
                        group["lr"]
                        for group in optimizer.param_groups
                        if str(group["group_name"]).startswith("encoder")
                    ),
                    default=0.0,
                )
            ),
            "head_learning_rate": float(
                next(
                    group["lr"]
                    for group in optimizer.param_groups
                    if group["group_name"] == "classification_head"
                )
            ),
            "train_objective": train_objective_sum / sample_count,
            "train_balanced_binary_cross_entropy": train_loss_sum / sample_count,
            "train_moe_loss": train_moe_sum / sample_count,
            "validation_balanced_binary_cross_entropy": validation[
                "balanced_binary_cross_entropy"
            ],
            "validation_moe_loss": validation["moe_loss"],
            "validation_micro_map": validation["micro_map"],
            "validation_macro_map": validation["macro_map"],
            "validation_micro_f1": validation["micro_f1"],
        }
        history.append(row)
        epoch_iterator.set_postfix(
            train=f"{row['train_balanced_binary_cross_entropy']:.4f}",
            val=f"{validation['balanced_binary_cross_entropy']:.4f}",
            micro_map=f"{validation['micro_map']:.4f}",
        )

        previous_best = best_score
        if validation["micro_map"] > best_score:
            best_score = validation["micro_map"]
            best_epoch = epoch
            _save_checkpoint(
                _checkpoint_payload(
                    args=args,
                    model=model,
                    head=head,
                    trainability=trainability,
                    dataset_info=dataset_info,
                    class_names=class_names,
                    seed=seed,
                    epoch=epoch,
                    validation=validation,
                ),
                seed_checkpoint,
            )
        if validation["micro_map"] > previous_best + args.early_stopping_min_delta:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if (
            args.early_stopping_patience > 0
            and epoch >= args.minimum_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping seed {seed} at epoch {epoch}; "
                f"best validation micro-mAP={best_score:.4f} at epoch {best_epoch}"
            )
            break

    checkpoint = torch.load(seed_checkpoint, map_location=args.device)
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    head.load_state_dict(checkpoint["head_state_dict"])
    validation = _evaluate(
        model,
        head,
        val_loader,
        dataset_info.raster_band_names,
        args.device,
    )
    test = _evaluate(
        model,
        head,
        test_loader,
        dataset_info.raster_band_names,
        args.device,
    )
    return {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_score),
        "checkpoint": str(seed_checkpoint.relative_to(args.output_dir)),
        "trainability": trainability,
        "history": history,
        "validation": validation,
        "test": test,
    }


def _aggregate(runs: Sequence[dict], split: str) -> dict:
    aggregate = {}
    for name in runs[0][split]:
        values = np.asarray([run[split][name] for run in runs], dtype=np.float64)
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return aggregate


def _write_history(path: Path, runs: Sequence[dict]) -> None:
    fieldnames = ["seed", *runs[0]["history"][0].keys()]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for row in run["history"]:
                writer.writerow({"seed": run["seed"], **row})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finetune all or selected encoder blocks with a multilabel linear "
            "head on GEO-Bench m-bigearthnet"
        )
    )
    parser.add_argument("--config_yaml", type=Path, required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--partition", default="default")
    parser.add_argument(
        "--spatial_protocol", choices=SPATIAL_PROTOCOLS, default="model_input"
    )
    parser.add_argument(
        "--trainable_encoder_layers",
        nargs="+",
        default=["all"],
        help="Use 'all', 'none', 'last:N', or explicit zero-based block indices",
    )
    parser.add_argument("--encoder_learning_rate", type=float, default=5e-6)
    parser.add_argument("--head_learning_rate", type=float, default=1e-3)
    parser.add_argument("--encoder_weight_decay", type=float, default=0.05)
    parser.add_argument("--head_weight_decay", type=float, default=0.0)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--moe_loss_scale", type=float, default=1.0)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--evaluation_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--lr_warmup_epochs", type=int, default=2)
    parser.add_argument("--minimum_learning_rate_ratio", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--minimum_epochs", type=int, default=5)
    parser.add_argument("--head_warmup_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--freeze_routers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze MoE gates and use deterministic routing during finetuning",
    )
    parser.add_argument(
        "--geometric_augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply rotations/reflections to training rasters only",
    )
    parser.add_argument(
        "--s1_modality",
        choices=["sentinel1_asc", "sentinel1_desc"],
        default="sentinel1_asc",
    )
    return parser


def _validate_args(args) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    if args.evaluation_batch_size is not None:
        positive["evaluation_batch_size"] = args.evaluation_batch_size
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if args.encoder_learning_rate <= 0 or args.head_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if (
        args.encoder_weight_decay < 0
        or args.head_weight_decay < 0
        or args.moe_loss_scale < 0
    ):
        raise ValueError("weight decay and moe_loss_scale must be non-negative")
    if not 0 < args.layer_decay <= 1:
        raise ValueError("layer_decay must be in (0, 1]")
    if not 0 <= args.head_dropout < 1:
        raise ValueError("head_dropout must be in [0, 1)")
    if args.gradient_clip_norm < 0:
        raise ValueError("gradient_clip_norm must be non-negative")
    if not 0 < args.minimum_learning_rate_ratio <= 1:
        raise ValueError("minimum_learning_rate_ratio must be in (0, 1]")
    if (
        args.lr_warmup_epochs < 0
        or args.early_stopping_patience < 0
        or args.early_stopping_min_delta < 0
        or args.minimum_epochs < 0
        or args.head_warmup_epochs < 0
    ):
        raise ValueError("warmup and early-stopping values must be non-negative")


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(str(args.config_yaml))
    model_input_size = int(config["model"]["img_size"])
    args.input_image_size = (
        224 if args.spatial_protocol == "geobench_224" else model_input_size
    )
    if args.batch_size is None:
        args.batch_size = 1 if args.spatial_protocol == "geobench_224" else 64
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = (
            8 if args.spatial_protocol == "geobench_224" else 1
        )
    _validate_args(args)

    args.device = resolve_device(args.device)
    args.config_yaml = args.config_yaml.expanduser().resolve()
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, model_band_names, _ = model_input_schema(config)
    datasets, dataset_info = _build_datasets(args, model_band_names)
    raw_class_names = getattr(dataset_info.task_specs.label_type, "class_names", None)
    class_name_values = (
        raw_class_names
        if raw_class_names is not None
        else range(dataset_info.num_classes)
    )
    class_names = [str(name) for name in class_name_values]

    runs = []
    best_score = float("-inf")
    best_checkpoint = args.output_dir / "best_model.pth"
    for seed in args.seeds:
        run = _train_one_seed(
            args=args,
            config=config,
            datasets=datasets,
            dataset_info=dataset_info,
            class_names=class_names,
            seed=int(seed),
        )
        runs.append(run)
        if run["best_validation_score"] > best_score:
            best_score = run["best_validation_score"]
            source = args.output_dir / run["checkpoint"]
            temporary = best_checkpoint.with_suffix(best_checkpoint.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(best_checkpoint)

    _write_history(args.output_dir / "optimization_history.csv", runs)
    results = {
        "dataset": DATASET_NAME,
        "task": "multilabel_classification_finetune",
        "class_names": class_names,
        "base_checkpoint": str(args.checkpoint_path),
        "spatial_protocol": args.spatial_protocol,
        "input_image_size": args.input_image_size,
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "feature_candidate": "final_pre_norm_mean_fine",
        "head": "linear",
        "head_dropout": args.head_dropout,
        "trainable_encoder_layers": args.trainable_encoder_layers,
        "freeze_routers": args.freeze_routers,
        "head_warmup_epochs": args.head_warmup_epochs,
        "layer_decay": args.layer_decay,
        "input_normalization": "geobench_official_per_band_zscore",
        "metadata_policy": "all checkpoint metadata explicitly missing",
        "training_augmentation": (
            "d4_rotations_reflections" if args.geometric_augmentation else "none"
        ),
        "training_loss": "balanced_binary_cross_entropy",
        "selection_metric": "validation_micro_map",
        "best_checkpoint": best_checkpoint.name,
        "runs": runs,
        "aggregate_validation": _aggregate(runs, "validation"),
        "aggregate_test": _aggregate(runs, "test"),
    }
    with open(args.output_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    metrics = results["aggregate_test"]
    print(
        f"\n{DATASET_NAME} test micro-mAP: "
        f"{100 * metrics['micro_map']['mean']:.2f} +/- "
        f"{100 * metrics['micro_map']['std']:.2f}"
    )
    print(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
