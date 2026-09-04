"""Frozen-embedding linear probe training and metrics."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def _predict(head: nn.Module, features: np.ndarray, batch_size: int, device: torch.device):
    batches = DataLoader(torch.from_numpy(features).float(), batch_size=batch_size)
    return torch.cat([head(batch.to(device)).cpu() for batch in batches]).numpy()


def classification_metrics(
    logits: np.ndarray, labels: np.ndarray, multilabel: bool
) -> Dict[str, float]:
    if multilabel:
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        predictions = probabilities >= 0.5
        return {
            "micro_map": float(average_precision_score(labels, probabilities, average="micro")),
            "macro_map": float(average_precision_score(labels, probabilities, average="macro")),
            "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        }

    predictions = logits.argmax(axis=1)
    return {
        "average_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "overall_accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def balanced_binary_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Balance positive and negative entries as defined by GEO-Bench."""
    element_losses = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    return element_losses[labels == 0].mean() + element_losses[labels == 1].mean()


def _save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_linear_probe_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[nn.Linear, dict]:
    """Load a saved linear probe and its training metadata."""
    checkpoint = torch.load(path, map_location=device)
    if (
        checkpoint.get("task") != "classification"
        or checkpoint.get("head_type") != "linear"
    ):
        raise ValueError(f"Not a classification linear-probe checkpoint: {path}")
    head = nn.Linear(checkpoint["input_dim"], checkpoint["num_classes"])
    head.load_state_dict(checkpoint["head_state_dict"])
    head.to(device).eval()
    return head, checkpoint


def train_linear_probe(
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
    num_classes: int,
    multilabel: bool,
    learning_rate: float,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 1024,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    checkpoint_path: Optional[Path] = None,
) -> dict:
    """Train one linear layer per seed, selecting its epoch on validation data."""
    input_dim = int(train["embeddings"].shape[1])
    label_dtype = torch.float32 if multilabel else torch.long
    train_dataset = TensorDataset(
        torch.from_numpy(train["embeddings"]).float(),
        torch.from_numpy(train["labels"]).to(label_dtype),
    )
    selection_metric = "micro_map" if multilabel else "average_accuracy"
    runs = []
    checkpoint_info = None

    for seed in seeds:
        _seed_everything(int(seed))
        head = nn.Linear(input_dim, num_classes).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.0)
        if multilabel:
            loss_fn = balanced_binary_cross_entropy
        else:
            loss_fn = nn.CrossEntropyLoss()

        generator = torch.Generator().manual_seed(int(seed))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        best_score = float("-inf")
        best_epoch = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            head.train()
            for features, labels in train_loader:
                logits = head(features.to(device))
                loss = loss_fn(logits, labels.to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            head.eval()
            val_logits = _predict(head, val["embeddings"], batch_size, device)
            val_metrics = classification_metrics(val_logits, val["labels"], multilabel)
            if val_metrics[selection_metric] > best_score:
                best_score = val_metrics[selection_metric]
                best_epoch = epoch
                best_state = deepcopy(head.state_dict())

        head.load_state_dict(best_state)
        head.eval()
        test_logits = _predict(head, test["embeddings"], batch_size, device)
        runs.append(
            {
                "seed": int(seed),
                "best_epoch": best_epoch,
                "best_val_score": best_score,
                "test": classification_metrics(test_logits, test["labels"], multilabel),
            }
        )
        if checkpoint_path is not None and (
            checkpoint_info is None
            or best_score > checkpoint_info["best_validation_score"]
        ):
            _save_checkpoint(
                {
                    "format_version": 1,
                    "task": "classification",
                    "head_type": "linear",
                    "head_state_dict": {
                        name: value.detach().cpu()
                        for name, value in head.state_dict().items()
                    },
                    "input_dim": input_dim,
                    "num_classes": int(num_classes),
                    "multilabel": bool(multilabel),
                    "selection_metric": selection_metric,
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "best_validation_score": float(best_score),
                },
                Path(checkpoint_path),
            )
            checkpoint_info = {
                "path": str(checkpoint_path),
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "best_validation_score": float(best_score),
            }

    metric_names = runs[0]["test"]
    aggregate = {}
    for name in metric_names:
        values = np.asarray([run["test"][name] for run in runs], dtype=np.float64)
        aggregate[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    result = {
        "selection_metric": selection_metric,
        "runs": runs,
        "aggregate": aggregate,
    }
    if checkpoint_info is not None:
        result["checkpoint"] = checkpoint_info
    return result
