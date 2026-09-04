"""
A script to pretrain MoE-MAE.

Author: Mohanad Albughdadi
Created: 2025-09-12
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import random
import shlex
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml
from tqdm import tqdm

from datasets.mmearth import MMEarthDataset
from models.moe_mae import build_model, MOEMAE
from scheduler.schedulers import WarmupCosineLR

OPTIMIZER_LAYOUT = "adamw_selective_weight_decay_v3_dense_decoder"

_NO_WEIGHT_DECAY_PARAMETER_SUFFIXES = (
    "mask_token",
    "modality_token_embed",
    "meta_token_embed",
    "cls_token",
    "meta_missing_embed",
    "fusion_bias",
)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _current_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _write_run_metadata(
    save_path,
    config,
    model,
    device,
    start_epoch,
    total_steps,
    warmup_steps,
    validation_mask_seed,
    resume_requested,
):
    """Persist the effective configuration and reproducibility information."""
    config_path = os.path.join(save_path, "config_resolved.yaml")
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)

    now = datetime.now(timezone.utc).isoformat()
    manifest_path = os.path.join(save_path, "run_manifest.json")
    manifest = {"schema_version": 1, "created_at": now, "launches": []}
    if resume_requested and start_epoch > 0 and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            existing = json.load(manifest_file)
        if isinstance(existing, dict):
            manifest = existing
            manifest.setdefault("schema_version", 1)
            manifest.setdefault("created_at", now)
            manifest.setdefault("launches", [])

    gpu_name = torch.cuda.get_device_name() if device == "cuda" else None
    manifest.update(
        {
            "updated_at": now,
            "git_commit": _current_git_commit(),
            "config_file": os.path.basename(config_path),
            "optimizer_layout": OPTIMIZER_LAYOUT,
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "device": device,
            "gpu_name": gpu_name,
            "seed": config["training"].get("seed"),
            "validation_mask_seed": validation_mask_seed,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
        }
    )
    manifest["launches"].append(
        {
            "started_at": now,
            "command": " ".join(shlex.quote(argument) for argument in [sys.executable, *sys.argv]),
            "resume_requested": bool(resume_requested),
            "start_epoch": start_epoch,
        }
    )
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")


def _uses_weight_decay(name, parameter):
    """Apply decay only to matrix/kernel weights, not scales, biases, or tokens."""
    if parameter.ndim <= 1:
        return False
    return not any(
        name == suffix or name.endswith(f".{suffix}")
        for suffix in _NO_WEIGHT_DECAY_PARAMETER_SUFFIXES
    )


def _resolve_warmup_steps(training_config, total_steps):
    warmup_ratio = float(training_config.get("warmup_ratio", 0.05))
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("training.warmup_ratio must be in [0, 1)")
    return warmup_ratio, int(total_steps * warmup_ratio)


def _build_adamw_optimizer(model, learning_rate, weight_decay):
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = (
            decay_parameters
            if _uses_weight_decay(name, parameter)
            else no_decay_parameters
        )
        target.append(parameter)

    if not decay_parameters or not no_decay_parameters:
        raise ValueError("Expected both decay and no-decay AdamW parameter groups")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": float(weight_decay),
                "group_name": "decay",
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
                "group_name": "no_decay",
            },
        ],
        lr=float(learning_rate),
    )
    return optimizer


def _validate_resume_optimizer_layout(checkpoint):
    if checkpoint.get("optimizer_layout") != OPTIMIZER_LAYOUT:
        raise ValueError(
            "This checkpoint does not use the current optimizer/model layout. "
            "Start a fresh run without --resume."
        )


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    return obj


def _get_batch_size_from_raster_dict(raster_dict):
    for tensor in raster_dict.values():
        if torch.is_tensor(tensor):
            return int(tensor.shape[0])
    raise ValueError("raster_dict must contain at least one tensor to infer batch size")


def _extract_collated_column_first(value):
    if torch.is_tensor(value):
        first = value[0]
        return first.item() if first.ndim == 0 else _extract_collated_column_first(first)
    if isinstance(value, (list, tuple)):
        first = value[0]
        return _extract_collated_column_first(first)
    return value


def _collapse_batched_band_metadata(value, batch_size):
    if isinstance(value, dict):
        return {
            key: _collapse_batched_band_metadata(subvalue, batch_size)
            for key, subvalue in value.items()
        }

    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return _extract_collated_column_first(value)
        return value

    if isinstance(value, (list, tuple)):
        if not value:
            return []

        if all(torch.is_tensor(elem) and elem.ndim > 0 and elem.shape[0] == batch_size for elem in value):
            return [_extract_collated_column_first(elem) for elem in value]

        if all(isinstance(elem, (list, tuple)) and len(elem) == batch_size for elem in value):
            return [_extract_collated_column_first(elem) for elem in value]

        return list(value)

    return value


def _extract_batch(batch, device):
    """Move one dict-style MMEarth batch to the training device."""
    if not isinstance(batch, dict) or not batch.get("raster_dict"):
        raise TypeError("Expected a dict batch containing 'raster_dict'")

    raster_dict = _move_to_device(batch["raster_dict"], device)
    forward_kwargs = {}
    for name in ("meta_dict", "meta_valid_masks", "raster_valid_masks"):
        if batch.get(name) is not None:
            forward_kwargs[name] = _move_to_device(batch[name], device)

    batch_size = _get_batch_size_from_raster_dict(raster_dict)
    for name in ("raster_band_names", "raster_band_indices"):
        if batch.get(name) is not None:
            forward_kwargs[name] = _collapse_batched_band_metadata(
                batch[name], batch_size
            )
    return raster_dict, forward_kwargs


def _forward_model(model, model_input, forward_kwargs):
    return model(raster_dict=model_input, **forward_kwargs)


def _build_patch_targets(model, model_input, forward_kwargs):
    patch_size = model.encoder.patch_size
    prepared_inputs, _, validity_masks, _, _ = model.encoder._prepare_raster_inputs(
        raster_dict=model_input,
        raster_valid_masks=forward_kwargs.get("raster_valid_masks"),
        raster_band_names=forward_kwargs.get("raster_band_names"),
        raster_band_indices=forward_kwargs.get("raster_band_indices"),
    )
    targets = {
        name: F.unfold(tensor, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
        for name, tensor in prepared_inputs.items()
    }
    target_validity = {
        name: F.unfold(mask, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
        for name, mask in validity_masks.items()
    }
    return targets, target_validity


def _apply_structured_modality_dropout(
    model_input,
    forward_kwargs,
    modality_dropout_prob,
):
    if (
        not isinstance(model_input, dict)
        or modality_dropout_prob <= 0.0
        or len(model_input) <= 1
    ):
        return model_input, forward_kwargs

    if torch.rand(1).item() >= modality_dropout_prob:
        return model_input, forward_kwargs

    names = list(model_input)
    subset_code = int(torch.randint(1, 2 ** len(names) - 1, (1,)).item())
    kept_inputs = {
        name: model_input[name]
        for index, name in enumerate(names)
        if subset_code & (1 << index)
    }

    updated_kwargs = dict(forward_kwargs)
    for key in ("raster_band_names", "raster_band_indices", "raster_valid_masks"):
        value = updated_kwargs.get(key)
        if isinstance(value, dict):
            updated_kwargs[key] = {
                name: value[name] for name in kept_inputs if name in value
            }
    return kept_inputs, updated_kwargs


def _compute_patch_loss(
    pred,
    target_patches,
    target_validity,
    mask,
    modality_loss_weights=None,
):
    """Average masked MSE per valid element, then combine modalities explicitly."""
    if not isinstance(pred, dict):
        weights = mask.unsqueeze(-1) * target_validity
        return (((pred - target_patches) ** 2) * weights).sum() / weights.sum().clamp_min(1.0)

    configured_weights = modality_loss_weights or {}
    weighted_losses = []
    active_weights = []
    for name, target in target_patches.items():
        if name not in pred:
            raise KeyError(f"Decoder did not produce required modality '{name}'")
        validity = target_validity[name]
        element_weights = mask.unsqueeze(-1) * validity
        denominator = element_weights.sum()
        modality_weight = float(configured_weights.get(name, 1.0))
        if modality_weight < 0:
            raise ValueError("Modality loss weights must be non-negative")
        if modality_weight == 0:
            continue
        modality_loss = (
            ((pred[name] - target) ** 2) * element_weights
        ).sum() / denominator.clamp_min(1.0)
        active = (denominator > 0).to(modality_loss.dtype)
        weighted_losses.append(modality_loss * modality_weight * active)
        active_weights.append(active * modality_weight)

    if not weighted_losses:
        raise ValueError("No valid reconstruction targets remain in this batch")
    return torch.stack(weighted_losses).sum() / torch.stack(active_weights).sum().clamp_min(1.0)


def train_epoch(
    model,
    loader,
    opt,
    scaler,
    device,
    epoch,
    scheduler,
    total_epochs,
    modality_dropout_prob=0.0,
    modality_loss_weights=None,
):
    """
    Performs one epoch of training, including gradient clipping and mixed precision.
    """
    model.train()

    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_moe_loss = 0.0

    step_in_epoch = 0
    pbar = tqdm(loader, desc=f"Training Epoch {epoch + 1}/{total_epochs}")
    lr_log = []

    for batch in pbar:
        global_step = epoch * len(loader) + step_in_epoch
        lr = scheduler.step(global_step)
        lr_log.append(lr)

        target_input, target_kwargs = _extract_batch(batch, device)
        model_input, forward_kwargs = _apply_structured_modality_dropout(
            target_input,
            target_kwargs,
            modality_dropout_prob,
        )

        opt.zero_grad()
        use_amp = device == "cuda"
        with torch.amp.autocast(enabled=use_amp, device_type=device):
            pred, mask, _, moe_loss = _forward_model(model, model_input, forward_kwargs)
            target_patches, target_validity = _build_patch_targets(
                model, target_input, target_kwargs
            )
            reconstruction_loss = _compute_patch_loss(
                pred,
                target_patches,
                target_validity,
                mask,
                modality_loss_weights=modality_loss_weights,
            )
            loss = reconstruction_loss + moe_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss detected at step {global_step}. Stopping training.")
            return None, None, None, None

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_moe_loss += moe_loss.item()
        step_in_epoch += 1
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "reconstruction": f"{reconstruction_loss.item():.4f}",
                "moe": f"{moe_loss.item():.4f}",
                "lr": f"{lr:.6f}",
            }
        )

    return (
        total_loss / len(loader),
        total_reconstruction_loss / len(loader),
        total_moe_loss / len(loader),
        sum(lr_log) / len(lr_log),
    )


@torch.no_grad()
def validate_epoch(
    model,
    loader,
    device,
    epoch,
    total_epochs,
    modality_loss_weights=None,
    mask_seed=44,
):
    """Evaluate with a fixed mask sequence so losses are comparable across epochs."""
    model.eval()
    mask_generator = torch.Generator(device="cpu").manual_seed(int(mask_seed))

    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_moe_loss = 0.0

    pbar = tqdm(loader, desc=f"Validation Epoch {epoch + 1}/{total_epochs}")
    for batch in pbar:
        model_input, forward_kwargs = _extract_batch(batch, device)
        forward_kwargs = dict(forward_kwargs)
        forward_kwargs["mask_generator"] = mask_generator

        use_amp = device == "cuda"
        with torch.amp.autocast(enabled=use_amp, device_type=device):
            pred, mask, _, moe_loss = _forward_model(model, model_input, forward_kwargs)
            target_patches, target_validity = _build_patch_targets(
                model, model_input, forward_kwargs
            )
            reconstruction_loss = _compute_patch_loss(
                pred,
                target_patches,
                target_validity,
                mask,
                modality_loss_weights=modality_loss_weights,
            )
            loss = reconstruction_loss + moe_loss

        total_loss += loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_moe_loss += moe_loss.item()
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "reconstruction": f"{reconstruction_loss.item():.4f}",
                "moe": f"{moe_loss.item():.4f}",
            }
        )

    return (
        total_loss / len(loader),
        total_reconstruction_loss / len(loader),
        total_moe_loss / len(loader),
    )


def _make_loader(dataset, batch_size, num_workers, shuffle, generator=None):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if generator is not None:
        loader_kwargs["generator"] = generator
        loader_kwargs["worker_init_fn"] = _seed_worker
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(**loader_kwargs)


def _resolve_milestone_epochs(training_cfg):
    configured = training_cfg.get("milestone_epochs", [])
    if configured is None:
        return set()
    if not isinstance(configured, (list, tuple)):
        raise TypeError("training.milestone_epochs must be a list of epoch numbers")

    milestones = {int(epoch) for epoch in configured}
    total_epochs = int(training_cfg["epochs"])
    invalid = sorted(epoch for epoch in milestones if epoch < 1 or epoch > total_epochs)
    if invalid:
        raise ValueError(
            "training.milestone_epochs entries must be between 1 and "
            f"training.epochs ({total_epochs}), got {invalid}"
        )
    return milestones


def _checkpoint_state(epoch, model, optimizer, scaler, best_val_loss):
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "optimizer_layout": OPTIMIZER_LAYOUT,
        "scaler_state": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }


def _build_mmearth_train_val_datasets(config):
    training_cfg = config["training"]
    dataset_path = training_cfg["dataset_path"]
    subset = training_cfg.get("dataset_subset", training_cfg.get("subset", "MMEarth64"))
    raster_modalities = training_cfg.get(
        "raster_modalities", list(MMEarthDataset.raster_modalities)
    )
    metadata_modalities = training_cfg.get(
        "metadata_modalities", list(MMEarthDataset.metadata_modalities)
    )
    modality_bands = training_cfg.get("modality_bands")
    normalization_mode = training_cfg.get("normalization_mode", "z-score")
    fill_value = float(training_cfg.get("fill_value", 0.0))
    fallback_split_from_train = bool(training_cfg.get("fallback_split_from_train", True))
    val_fraction = float(training_cfg.get("val_fraction", 0.1))
    test_fraction = float(training_cfg.get("test_fraction", 0.0))
    split_seed = int(training_cfg.get("split_seed", 42))
    train_split = training_cfg.get("train_split_name", "train")
    val_split = training_cfg.get("val_split_name", "val")

    common_kwargs = {
        "root_dir": dataset_path,
        "subset": subset,
        "raster_modalities": raster_modalities,
        "metadata_modalities": metadata_modalities,
        "modality_bands": modality_bands,
        "normalization_mode": normalization_mode,
        "fill_value": fill_value,
        "fallback_split_from_train": fallback_split_from_train,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "split_seed": split_seed,
        "transform": None,
    }
    train_dataset = MMEarthDataset(split=train_split, **common_kwargs)
    val_dataset = MMEarthDataset(split=val_split, **common_kwargs)
    return train_dataset, val_dataset


def _build_train_val_datasets(config):
    training_cfg = config["training"]
    dataset_name = training_cfg.get("dataset_name", "mmearth").lower()

    if dataset_name == "mmearth":
        return _build_mmearth_train_val_datasets(config)

    raise ValueError(f"Unsupported training.dataset_name: {dataset_name}")


def _validate_non_empty_datasets(train_dataset, val_dataset, dataset_name):
    if len(train_dataset) == 0:
        raise ValueError(f"{dataset_name} training split is empty")
    if len(val_dataset) == 0:
        raise ValueError(
            f"{dataset_name} validation split is empty. "
            "For MMEarth64, enable training.fallback_split_from_train with a non-zero "
            "training.val_fraction, or provide an explicit validation split."
        )


def _maybe_limit_dataset(dataset, limit, seed):
    if limit is None:
        return dataset
    limit = int(limit)
    if limit <= 0:
        raise ValueError("Subset limit must be a positive integer")
    if len(dataset) <= limit:
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def _maybe_apply_subset_limits(train_dataset, val_dataset, training_cfg):
    subset_seed = int(training_cfg.get("subset_sample_seed", training_cfg.get("split_seed", 42)))
    train_limit = training_cfg.get("max_train_samples")
    val_limit = training_cfg.get("max_val_samples")
    return (
        _maybe_limit_dataset(train_dataset, train_limit, subset_seed),
        _maybe_limit_dataset(val_dataset, val_limit, subset_seed + 1),
    )


def _unwrap_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _infer_square_img_size_from_dataset(dataset):
    sample = dataset[0]
    if not isinstance(sample, dict) or "raster_dict" not in sample:
        raise TypeError("Expected dict-style multimodal sample with 'raster_dict'")

    raster_dict = sample["raster_dict"]
    if not raster_dict:
        raise ValueError("Sample raster_dict is empty")

    spatial_shapes = {
        name: tuple(tensor.shape[-2:])
        for name, tensor in raster_dict.items()
        if torch.is_tensor(tensor) and tensor.ndim == 3
    }
    unique_shapes = set(spatial_shapes.values())
    if len(unique_shapes) != 1:
        raise ValueError(
            f"All raster modalities must share the same spatial size, got {spatial_shapes}"
        )

    height, width = next(iter(unique_shapes))
    if height != width:
        raise ValueError(
            f"Current pretraining path expects square raster inputs, got {(height, width)}"
        )
    return int(height)


def main():
    """
    The main training loop.
    """
    parser = argparse.ArgumentParser(description="Pretrain MoE-MAE on MMEarth with validation tracking")
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint if available"
    )
    args = parser.parse_args()

    with open(args.config_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configured_seed = config["training"].get("seed")
    train_generator = None
    val_generator = None
    if configured_seed is not None:
        configured_seed = int(configured_seed)
        _seed_everything(configured_seed)
        train_generator = torch.Generator().manual_seed(configured_seed)
        val_generator = torch.Generator().manual_seed(configured_seed + 1)
        print(f"Random seed: {configured_seed}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    scaler = torch.amp.GradScaler(device=device)
    dataset_name = config["training"].get("dataset_name", "mmearth").lower()

    train_dataset, val_dataset = _build_train_val_datasets(config)
    _validate_non_empty_datasets(train_dataset, val_dataset, dataset_name)
    train_dataset, val_dataset = _maybe_apply_subset_limits(train_dataset, val_dataset, config["training"])
    _validate_non_empty_datasets(train_dataset, val_dataset, dataset_name)
    configured_model_img_size = config["model"].get("img_size")
    model_img_size = int(configured_model_img_size) if configured_model_img_size is not None else None
    model_kwargs = {}
    modality_dropout_prob = float(
        config["training"].get("structured_modality_dropout_prob", 0.0)
    )
    if not 0.0 <= modality_dropout_prob <= 1.0:
        raise ValueError("training.structured_modality_dropout_prob must be in [0, 1]")
    modality_loss_weights = config["training"].get("modality_loss_weights", {})

    if dataset_name == "mmearth":
        dataset_info = _unwrap_dataset(train_dataset)
        input_adapters = dict(dataset_info.input_adapters)
        input_band_names = {
            name: list(dataset_info.modality_bands[name])
            for name in dataset_info.raster_modalities
        }
        metadata_dims = dict(dataset_info.metadata_dims)
        unknown_loss_weights = set(modality_loss_weights) - set(input_adapters)
        if unknown_loss_weights:
            raise ValueError(
                "training.modality_loss_weights contains unknown modalities: "
                f"{sorted(unknown_loss_weights)}"
            )
        primary_input_name = config["training"].get(
            "primary_input_name", dataset_info.primary_input_name
        )
        if primary_input_name not in input_adapters:
            raise ValueError(
                f"training.primary_input_name '{primary_input_name}' is not in the requested "
                f"MMEarth raster modalities: {sorted(input_adapters)}"
            )

        inferred_img_size = _infer_square_img_size_from_dataset(train_dataset)
        if model_img_size is not None and model_img_size != inferred_img_size:
            print(
                f"MMEarth override: config model.img_size={model_img_size} -> "
                f"dataset size {inferred_img_size}"
            )
        model_img_size = inferred_img_size

        model_kwargs = {
            "input_adapters": input_adapters,
            "input_band_names": input_band_names,
            "primary_input_name": primary_input_name,
            "metadata_dims": metadata_dims,
        }
        print(
            "MMEarth setup | "
            f"train={len(train_dataset)} val={len(val_dataset)} | "
            f"primary={primary_input_name} | "
            f"input_adapters={input_adapters} | metadata_dims={metadata_dims}"
        )
    else:
        raise ValueError(f"Unsupported training.dataset_name: {dataset_name}")

    print(f"Dataset sizes | train={len(train_dataset)} val={len(val_dataset)}")
    print(
        "Objective | "
        f"moe_balance_weight={float(config['model'].get('moe_balance_weight', 1e-2)):.4f} "
        f"structured_modality_dropout_prob={modality_dropout_prob:.3f} "
        f"modality_loss_weights={modality_loss_weights or 'equal'}"
    )

    encoder_kwargs = dict(
        size=config["model"]["size"],
        img_size=model_img_size,
        patch_size=config["model"]["patch_size"],
        moe_balance_weight=float(config["model"].get("moe_balance_weight", 1e-2)),
        **model_kwargs,
    )
    encoder = build_model(**encoder_kwargs)
    model = MOEMAE(encoder).to(device)

    opt = _build_adamw_optimizer(
        model,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    decay_count = sum(
        parameter.numel()
        for group in opt.param_groups
        if group["group_name"] == "decay"
        for parameter in group["params"]
    )
    no_decay_count = sum(
        parameter.numel()
        for group in opt.param_groups
        if group["group_name"] == "no_decay"
        for parameter in group["params"]
    )
    print(
        "AdamW parameters | "
        f"decay={decay_count:,} no_decay={no_decay_count:,} "
        f"weight_decay={float(config['training']['weight_decay']):.4f}"
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=True,
        generator=train_generator,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=config["training"].get("val_batch_size", config["training"]["batch_size"]),
        num_workers=config["training"]["num_workers"],
        shuffle=False,
        generator=val_generator,
    )
    save_path = config["training"]["weight_path"]
    os.makedirs(save_path, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")

    ckpt_path = f"{save_path}/checkpoint_{config['model']['size']}.pth"
    if args.resume and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        _validate_resume_optimizer_layout(checkpoint)
        model.load_state_dict(checkpoint["model_state"])
        opt.load_state_dict(checkpoint["opt_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get(
            "best_val_loss", checkpoint.get("best_train_loss", float("inf"))
        )
        print(
            f"Resumed from epoch {start_epoch}, best validation loss {best_val_loss:.4f}"
        )
    total_steps = int(config["training"]["epochs"] * len(train_loader))
    warmup_ratio, warmup_steps = _resolve_warmup_steps(
        config["training"], total_steps
    )
    validation_mask_seed = int(
        config["training"].get(
            "validation_mask_seed",
            configured_seed + 2 if configured_seed is not None else 44,
        )
    )
    scheduler = WarmupCosineLR(
        opt, total_steps, config["training"]["learning_rate"], warmup_steps
    )
    print(
        "Schedule | "
        f"total_steps={total_steps} warmup_steps={warmup_steps} "
        f"warmup_ratio={warmup_ratio:.4f} validation_mask_seed={validation_mask_seed}"
    )
    _write_run_metadata(
        save_path,
        config,
        model,
        device,
        start_epoch,
        total_steps,
        warmup_steps,
        validation_mask_seed,
        args.resume,
    )
    save_every_n_epochs = config["training"].get("save_every_n_epochs", 5)
    milestone_epochs = _resolve_milestone_epochs(config["training"])
    if milestone_epochs:
        print(f"Milestone checkpoints: {sorted(milestone_epochs)}")
    loss_log_path = f"{save_path}/training_metrics_{config['model']['size']}.csv"
    log_mode = "a" if start_epoch > 0 else "w"
    with open(loss_log_path, log_mode, newline="", encoding="utf-8") as loss_log_file:
        csv_writer = csv.writer(loss_log_file)
        if start_epoch == 0:
            csv_writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_reconstruction_loss",
                    "train_moe_loss",
                    "val_loss",
                    "val_reconstruction_loss",
                    "val_moe_loss",
                    "avg_lr",
                ]
            )

        for ep in range(start_epoch, config["training"]["epochs"]):
            (
                train_loss,
                train_reconstruction_loss,
                train_moe_loss,
                avg_lr,
            ) = train_epoch(
                model,
                train_loader,
                opt,
                scaler,
                device,
                ep,
                scheduler,
                config["training"]["epochs"],
                modality_dropout_prob=modality_dropout_prob,
                modality_loss_weights=modality_loss_weights,
            )

            if train_loss is None:
                break

            (
                val_loss,
                val_reconstruction_loss,
                val_moe_loss,
            ) = validate_epoch(
                model,
                val_loader,
                device,
                ep,
                config["training"]["epochs"],
                modality_loss_weights=modality_loss_weights,
                mask_seed=validation_mask_seed,
            )

            csv_writer.writerow(
                [
                    ep,
                    f"{train_loss:.4f}",
                    f"{train_reconstruction_loss:.4f}",
                    f"{train_moe_loss:.4f}",
                    f"{val_loss:.4f}",
                    f"{val_reconstruction_loss:.4f}",
                    f"{val_moe_loss:.4f}",
                    f"{avg_lr:.8f}",
                ]
            )
            print(
                f"Epoch {ep+1}/{config['training']['epochs']} | "
                f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
                f"Avg LR {avg_lr:.8f}"
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            checkpoint_state = _checkpoint_state(
                ep,
                model,
                opt,
                scaler,
                best_val_loss,
            )

            if is_best:
                torch.save(
                    checkpoint_state,
                    f"{save_path}/pretrained_{config['model']['size']}_best.pth",
                )
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

            if (ep + 1) % save_every_n_epochs == 0:
                torch.save(
                    checkpoint_state,
                    ckpt_path,
                )

            completed_epoch = ep + 1
            if completed_epoch in milestone_epochs:
                milestone_path = (
                    f"{save_path}/checkpoint_{config['model']['size']}_"
                    f"epoch_{completed_epoch:03d}.pth"
                )
                torch.save(checkpoint_state, milestone_path)
                print(f"Milestone checkpoint saved: {milestone_path}")
            loss_log_file.flush()
    torch.save(
        _checkpoint_state(ep, model, opt, scaler, best_val_loss),
        f"{save_path}/pretrained_{config['model']['size']}_last.pth",
    )
    print("Training complete. Last epoch model saved.")


if __name__ == "__main__":
    main()
