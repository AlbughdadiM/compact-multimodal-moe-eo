"""Utilities for end-to-end GEO-Bench classification finetuning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import torch
from torch import nn

from utils.segmentation_finetune import (
    build_finetuning_optimizer,
    configure_encoder_trainability,
    set_finetuning_mode,
)


class ClassificationHead(nn.Module):
    """A linear multilabel head with optional training-time dropout."""

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))


def build_classification_finetuning_optimizer(
    model: nn.Module,
    head: nn.Module,
    encoder_learning_rate: float,
    head_learning_rate: float,
    encoder_weight_decay: float,
    head_weight_decay: float = 0.0,
    layer_decay: float = 1.0,
) -> torch.optim.Optimizer:
    """Build the shared layer-decayed AdamW optimizer for classification."""
    optimizer = build_finetuning_optimizer(
        model=model,
        head=head,
        encoder_learning_rate=encoder_learning_rate,
        head_learning_rate=head_learning_rate,
        encoder_weight_decay=encoder_weight_decay,
        head_weight_decay=head_weight_decay,
        layer_decay=layer_decay,
    )
    for group in optimizer.param_groups:
        if group.get("group_name") == "segmentation_head":
            group["group_name"] = "classification_head"
    return optimizer


def forward_classification(
    model: nn.Module,
    head: nn.Module,
    raster_dict: Dict[str, torch.Tensor],
    raster_valid_masks: Dict[str, torch.Tensor],
    raster_band_names: Dict[str, Sequence[str]],
    encoder_grad_enabled: bool,
    stochastic_routing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool final pre-norm fine tokens and apply the multilabel head."""
    forward_kwargs = {
        "raster_dict": raster_dict,
        "raster_valid_masks": raster_valid_masks,
        "raster_band_names": raster_band_names,
        "stochastic_routing": stochastic_routing,
    }
    if encoder_grad_enabled:
        features = model.forward_features(**forward_kwargs)
    else:
        with torch.no_grad():
            features = model.forward_features(**forward_kwargs)
    embedding = features["pre_norm_fine_tokens"].mean(dim=1)
    return head(embedding), features["moe_loss"]


def load_finetuned_classification_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, ClassificationHead, dict]:
    """Restore an encoder and head saved by the classification script."""
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("task") != "multilabel_classification_finetune":
        raise ValueError(f"Not a classification-finetuning checkpoint: {path}")
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.to(device).eval()
    head = ClassificationHead(
        feature_dim=int(checkpoint["feature_dim"]),
        num_classes=int(checkpoint["num_classes"]),
        dropout=float(checkpoint.get("head_dropout", 0.0)),
    )
    head.load_state_dict(checkpoint["head_state_dict"])
    head.to(device).eval()
    return model, head, checkpoint


__all__ = [
    "ClassificationHead",
    "build_classification_finetuning_optimizer",
    "configure_encoder_trainability",
    "forward_classification",
    "load_finetuned_classification_checkpoint",
    "set_finetuning_mode",
]
