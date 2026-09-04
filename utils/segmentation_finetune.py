"""Utilities for end-to-end GEO-Bench semantic segmentation finetuning."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from utils.segmentation_probe import UPerNetProbe, resolve_feature_layers


_NO_WEIGHT_DECAY_SUFFIXES = (
    "cls_token",
    "modality_token_embed",
    "meta_token_embed",
    "meta_missing_embed",
    "fusion_bias",
)


def configure_encoder_trainability(
    model: nn.Module,
    values: Sequence[str],
    freeze_routers: bool = False,
) -> dict:
    """Freeze the MAE decoder and select all, none, or explicit encoder blocks."""
    if not values:
        raise ValueError("trainable_encoder_layers must not be empty")
    values = [str(value).strip().lower() for value in values]
    depth = len(model.encoder.layers)

    if values == ["all"]:
        mode = "all"
        layer_indices = tuple(range(depth))
    elif values == ["none"]:
        mode = "none"
        layer_indices = ()
    elif len(values) == 1 and values[0].startswith("last:"):
        try:
            count = int(values[0].split(":", maxsplit=1)[1])
        except ValueError as exc:
            raise ValueError("last:N requires a positive integer N") from exc
        if count <= 0 or count > depth:
            raise ValueError(f"last:N requires N in [1, {depth}], got {count}")
        mode = "layers"
        layer_indices = tuple(range(depth - count, depth))
    else:
        if "all" in values or "none" in values:
            raise ValueError("'all' and 'none' cannot be combined with layer indices")
        try:
            layer_indices = tuple(sorted(int(value) for value in values))
        except ValueError as exc:
            raise ValueError(
                "trainable_encoder_layers must be 'all', 'none', 'last:N', or "
                "integer block indices"
            ) from exc
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError("trainable encoder layer indices must be unique")
        invalid = [index for index in layer_indices if index < 0 or index >= depth]
        if invalid:
            raise ValueError(
                f"encoder layer indices must be in [0, {depth - 1}], got {invalid}"
            )
        mode = "layers"

    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "all":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = True
    else:
        for index in layer_indices:
            for parameter in model.encoder.layers[index].parameters():
                parameter.requires_grad = True

    frozen_router_parameters = 0
    if freeze_routers:
        for layer in model.encoder.layers:
            gate = getattr(getattr(layer, "moe", None), "gate", None)
            if gate is None:
                continue
            for parameter in gate.parameters():
                if parameter.requires_grad:
                    frozen_router_parameters += parameter.numel()
                parameter.requires_grad = False

    trainable = sum(
        parameter.numel()
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.encoder.parameters())
    return {
        "mode": mode,
        "layer_indices": list(layer_indices),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "routers_frozen": bool(freeze_routers),
        "frozen_router_parameters": int(frozen_router_parameters),
    }


def set_finetuning_mode(
    model: nn.Module,
    head: nn.Module,
    trainability: dict,
    freeze_batch_norm: bool,
    encoder_training_enabled: bool = True,
) -> None:
    """Enable stochastic/dropout behavior only in trainable encoder components."""
    model.eval()
    if encoder_training_enabled and trainability["mode"] == "all":
        model.encoder.train()
    elif encoder_training_enabled and trainability["mode"] == "layers":
        for index in trainability["layer_indices"]:
            model.encoder.layers[index].train()

    head.train()
    if freeze_batch_norm:
        batch_norm_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
        )
        for module in head.modules():
            if isinstance(module, batch_norm_types):
                module.eval()


@contextmanager
def _capture_encoder_layers(
    model: nn.Module,
    layer_indices: Sequence[int],
) -> Iterator[dict[int, torch.Tensor]]:
    """Capture fused encoder outputs without changing the model's public API."""
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in layer_indices:

        def capture_output(_module, _inputs, output, index=layer_index):
            captured[index] = output[0] if isinstance(output, tuple) else output

        handles.append(
            model.encoder.layers[layer_index].register_forward_hook(capture_output)
        )
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def intermediate_pre_norm_pyramid(
    captured_tokens: Dict[int, torch.Tensor],
    layer_indices: Sequence[int],
    token_layout: dict,
    prefix_tokens: int,
) -> torch.Tensor:
    """Convert intermediate fused-layer tokens to UPerNet feature maps."""
    height = int(token_layout["fine_height"])
    width = int(token_layout["fine_width"])
    maps = []
    for layer_index in layer_indices:
        if layer_index not in captured_tokens:
            raise RuntimeError(f"Encoder layer {layer_index} was not captured")
        tokens = captured_tokens[layer_index][:, prefix_tokens:]
        if tokens.shape[1] != height * width:
            raise ValueError(
                f"Layer {layer_index} has {tokens.shape[1]} fine tokens; "
                f"expected {height * width} for grid {height}x{width}"
            )
        maps.append(
            tokens.transpose(1, 2).reshape(
                tokens.shape[0], tokens.shape[2], height, width
            )
        )
    return torch.stack(maps, dim=1)


def forward_segmentation(
    model: nn.Module,
    head: nn.Module,
    raster_dict: Dict[str, torch.Tensor],
    raster_valid_masks: Dict[str, torch.Tensor],
    raster_band_names: Dict[str, Sequence[str]],
    target_size: Sequence[int],
    encoder_grad_enabled: bool,
    feature_layers: Sequence[int] | None = None,
    stochastic_routing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the live encoder and a true multilayer pre-norm UPerNet pyramid."""
    if feature_layers is None:
        feature_layers = resolve_feature_layers(len(model.encoder.layers))
    feature_layers = tuple(int(index) for index in feature_layers)
    if len(feature_layers) != 4:
        raise ValueError("UPerNet finetuning requires exactly four feature layers")
    if min(feature_layers) < 1 or max(feature_layers) >= len(model.encoder.layers):
        raise ValueError("Feature layers must be valid fused encoder layer indices")

    with _capture_encoder_layers(model, feature_layers) as captured:
        if encoder_grad_enabled:
            features = model.forward_features(
                raster_dict=raster_dict,
                raster_valid_masks=raster_valid_masks,
                raster_band_names=raster_band_names,
                stochastic_routing=stochastic_routing,
            )
        else:
            with torch.no_grad():
                features = model.forward_features(
                    raster_dict=raster_dict,
                    raster_valid_masks=raster_valid_masks,
                    raster_band_names=raster_band_names,
                    stochastic_routing=stochastic_routing,
                )
    pyramid = intermediate_pre_norm_pyramid(
        captured_tokens=captured,
        layer_indices=feature_layers,
        token_layout=features["token_layout"],
        prefix_tokens=model.encoder.num_meta_tokens + 1,
    )
    logits = head(pyramid, target_size)
    return logits, features["moe_loss"]


def soft_dice_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean multiclass soft Dice loss over classes present in the minibatch."""
    probabilities = logits.float().softmax(dim=1)
    targets = F.one_hot(labels, num_classes=logits.shape[1]).movedim(-1, 1).float()
    reduce_dims = (0, 2, 3)
    intersection = (probabilities * targets).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
    present = targets.sum(dim=reduce_dims) > 0
    if not torch.any(present):
        return logits.sum() * 0.0
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - dice[present].mean()


def _uses_weight_decay(name: str, parameter: nn.Parameter) -> bool:
    if parameter.ndim <= 1:
        return False
    return not any(
        name == suffix or name.endswith(f".{suffix}")
        for suffix in _NO_WEIGHT_DECAY_SUFFIXES
    )


def build_finetuning_optimizer(
    model: nn.Module,
    head: nn.Module,
    encoder_learning_rate: float,
    head_learning_rate: float,
    encoder_weight_decay: float,
    head_weight_decay: float = 0.0,
    layer_decay: float = 1.0,
) -> torch.optim.Optimizer:
    """Build AdamW groups with a small encoder LR and an independent head LR."""
    if encoder_learning_rate <= 0 or head_learning_rate <= 0:
        raise ValueError("encoder and head learning rates must be positive")
    if encoder_weight_decay < 0 or head_weight_decay < 0:
        raise ValueError("weight decay must be non-negative")
    if not 0 < layer_decay <= 1:
        raise ValueError("layer_decay must be in (0, 1]")

    depth = len(model.encoder.layers)
    encoder_groups: dict[tuple[int, bool], list[nn.Parameter]] = {}
    for name, parameter in model.encoder.named_parameters():
        if not parameter.requires_grad:
            continue
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
            layer_index = int(parts[1])
        else:
            layer_index = -1
        uses_decay = _uses_weight_decay(name, parameter)
        encoder_groups.setdefault((layer_index, uses_decay), []).append(parameter)

    groups = []
    for (layer_index, uses_decay), parameters in sorted(encoder_groups.items()):
        exponent = depth if layer_index < 0 else depth - 1 - layer_index
        learning_rate = float(encoder_learning_rate) * layer_decay**exponent
        layer_name = "stem" if layer_index < 0 else f"layer_{layer_index}"
        decay_name = "decay" if uses_decay else "no_decay"
        groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": float(encoder_weight_decay) if uses_decay else 0.0,
                "group_name": f"encoder_{layer_name}_{decay_name}",
            }
        )
    groups.append(
        {
            "params": list(head.parameters()),
            "lr": float(head_learning_rate),
            "weight_decay": float(head_weight_decay),
            "group_name": "segmentation_head",
        }
    )
    return torch.optim.AdamW(groups)


def load_finetuned_segmentation_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, nn.Module, dict]:
    """Restore an encoder and UPerNet head saved by the finetuning script."""
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("task") != "semantic_segmentation_finetune":
        raise ValueError(f"Not a segmentation-finetuning checkpoint: {path}")
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.to(device).eval()
    head = UPerNetProbe(
        feature_dim=int(checkpoint["feature_dim"]),
        num_classes=int(checkpoint["num_classes"]),
        channels=int(checkpoint["decoder_channels"]),
        dropout=float(checkpoint.get("head_dropout", 0.0)),
        normalization=checkpoint.get("head_normalization", "batch"),
    )
    head.load_state_dict(checkpoint["head_state_dict"])
    head.to(device).eval()
    return model, head, checkpoint
