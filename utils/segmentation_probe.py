"""Dense feature extraction and frozen-backbone semantic segmentation probes."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Dict, Optional, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.extract_embeddings import move_to_device


DENSE_FEATURE_CANDIDATES = (
    "multilayer_pre_norm",
    "multilayer_post_norm",
    "final_pre_norm",
    "final_post_norm",
)


def resolve_feature_layers(depth: int, num_stages: int = 4) -> tuple[int, ...]:
    """Select the end of four approximately equal fused-encoder depth ranges."""
    fused_layers = np.arange(1, depth)
    if len(fused_layers) < num_stages:
        raise ValueError(
            f"Encoder depth {depth} cannot provide {num_stages} fused stages"
        )
    return tuple(int(group[-1]) for group in np.array_split(fused_layers, num_stages))


def dense_extraction_signature(
    checkpoint_path: Path,
    preprocessing_signature: str,
    raster_band_names: Dict[str, Sequence[str]],
    feature_layers: Sequence[int],
    split: str,
    partition: str,
    max_samples: Optional[int],
    sample_seed: int,
) -> str:
    checkpoint_stat = checkpoint_path.stat()
    payload = {
        "version": 2,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "preprocessing": preprocessing_signature,
        "raster_band_names": {
            name: list(names) for name, names in sorted(raster_band_names.items())
        },
        "feature_layers": list(feature_layers),
        "split": split,
        "partition": partition,
        "max_samples": max_samples,
        "sample_seed": sample_seed,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def dense_cache_matches(path: Path, expected_signature: str) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return handle.attrs.get("extraction_signature") == expected_signature
    except OSError:
        return False


@torch.inference_mode()
def extract_dense_feature_cache(
    model: nn.Module,
    dataloader: DataLoader,
    raster_band_names: Dict[str, list[str]],
    device: torch.device,
    feature_layers: Sequence[int],
    output_path: Path,
    extraction_signature: str,
) -> None:
    """Extract raw dense token maps once and store them in a streaming HDF5 cache."""
    feature_layers = tuple(int(index) for index in feature_layers)
    if len(set(feature_layers)) != len(feature_layers):
        raise ValueError("feature_layers must not contain duplicates")
    if feature_layers != tuple(sorted(feature_layers)):
        raise ValueError("feature_layers must be in increasing encoder-depth order")
    if not feature_layers or min(feature_layers) < 1:
        raise ValueError(
            "feature_layers must use fused encoder layers with indices >= 1"
        )
    if max(feature_layers) >= len(model.encoder.layers):
        raise ValueError("feature_layers contains an index outside the encoder")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    model.eval()

    captured = {}
    handles = []
    for layer_index in feature_layers:

        def capture_output(_module, _inputs, output, index=layer_index):
            captured[index] = output[0]

        handles.append(
            model.encoder.layers[layer_index].register_forward_hook(capture_output)
        )

    use_amp = device.type == "cuda"
    autocast_device = device.type if device.type in {"cuda", "cpu", "mps"} else "cpu"
    write_index = 0
    total_samples = len(dataloader.dataset)

    try:
        with h5py.File(temporary_path, "w") as cache:
            feature_store = None
            label_store = None
            sample_store = None
            for batch in tqdm(dataloader, desc=f"Caching {output_path.stem}"):
                captured.clear()
                raster_dict = move_to_device(batch["raster_dict"], device)
                raster_valid_masks = move_to_device(
                    batch.get("raster_valid_masks"), device
                )
                with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                    features = model.forward_features(
                        raster_dict=raster_dict,
                        raster_valid_masks=raster_valid_masks,
                        raster_band_names=raster_band_names,
                    )

                missing_layers = set(feature_layers) - set(captured)
                if missing_layers:
                    raise RuntimeError(
                        f"Encoder hooks did not capture layers {sorted(missing_layers)}"
                    )
                fine_height = int(features["token_layout"]["fine_height"])
                fine_width = int(features["token_layout"]["fine_width"])
                prefix_count = model.encoder.num_meta_tokens + 1
                stage_maps = []
                for layer_index in feature_layers:
                    tokens = captured[layer_index][:, prefix_count:]
                    if tokens.shape[1] != fine_height * fine_width:
                        raise ValueError(
                            "Dense token count does not match the runtime grid"
                        )
                    stage_maps.append(
                        tokens.transpose(1, 2).reshape(
                            tokens.shape[0], tokens.shape[2], fine_height, fine_width
                        )
                    )
                dense_features = (
                    torch.stack(stage_maps, dim=1).to(dtype=torch.float16).cpu().numpy()
                )
                labels = batch["label"].cpu().numpy().astype(np.int16, copy=False)
                batch_size = dense_features.shape[0]

                if feature_store is None:
                    _, stages, channels, height, width = dense_features.shape
                    feature_store = cache.create_dataset(
                        "features",
                        shape=(total_samples, stages, channels, height, width),
                        dtype=np.float16,
                        chunks=(1, 1, channels, height, width),
                        compression="lzf",
                    )
                    label_store = cache.create_dataset(
                        "labels",
                        shape=(total_samples, *labels.shape[1:]),
                        dtype=np.int16,
                        chunks=(1, *labels.shape[1:]),
                        compression="lzf",
                    )
                    sample_store = cache.create_dataset(
                        "sample_ids",
                        shape=(total_samples,),
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )

                next_index = write_index + batch_size
                feature_store[write_index:next_index] = dense_features
                label_store[write_index:next_index] = labels
                sample_store[write_index:next_index] = [
                    str(sample_id) for sample_id in batch["sample_id"]
                ]
                write_index = next_index

            if write_index != total_samples:
                raise RuntimeError(
                    f"Cached {write_index} samples, expected {total_samples}"
                )
            cache.attrs["extraction_signature"] = extraction_signature
            cache.attrs["feature_layers"] = json.dumps(list(feature_layers))
            cache.attrs["raster_band_names"] = json.dumps(
                {name: list(names) for name, names in raster_band_names.items()},
                sort_keys=True,
            )
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        for handle in handles:
            handle.remove()


class DenseFeatureCacheDataset(Dataset):
    """Worker-safe random access to one dense HDF5 feature cache."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle = None
        with h5py.File(self.path, "r") as handle:
            required = {"features", "labels", "sample_ids"}
            missing = required - set(handle)
            if missing:
                raise ValueError(
                    f"Dense cache {self.path} is missing {sorted(missing)}"
                )
            if not (
                handle["features"].shape[0]
                == handle["labels"].shape[0]
                == handle["sample_ids"].shape[0]
            ):
                raise ValueError(f"Dense cache {self.path} has inconsistent lengths")
            self.length = int(handle["features"].shape[0])
            self.num_stages = int(handle["features"].shape[1])
            self.feature_dim = int(handle["features"].shape[2])
            self.label_shape = tuple(int(value) for value in handle["labels"].shape[1:])

    def __len__(self) -> int:
        return self.length

    def _get_handle(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index: int):
        handle = self._get_handle()
        return (
            torch.from_numpy(np.asarray(handle["features"][index])),
            torch.from_numpy(np.asarray(handle["labels"][index], dtype=np.int64)),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self):
        if self._handle is not None:
            self._handle.close()


def _apply_layer_norm(
    features: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
) -> torch.Tensor:
    channels_last = features.movedim(2, -1)
    norm_weight = norm_weight.to(device=features.device, dtype=features.dtype)
    norm_bias = norm_bias.to(device=features.device, dtype=features.dtype)
    normalized = F.layer_norm(
        channels_last,
        (channels_last.shape[-1],),
        weight=norm_weight,
        bias=norm_bias,
    )
    return normalized.movedim(-1, 2)


def prepare_dense_candidate(
    features: torch.Tensor,
    candidate: str,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
) -> torch.Tensor:
    """Build equal-width final/multilayer and pre/post-norm dense candidates."""
    if candidate not in DENSE_FEATURE_CANDIDATES:
        raise ValueError(f"Unknown dense feature candidate '{candidate}'")
    if candidate.startswith("final_"):
        features = features[:, -1:].expand(-1, features.shape[1], -1, -1, -1)
    if candidate.endswith("post_norm"):
        features = _apply_layer_norm(features, norm_weight, norm_bias)
    return features


def _save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _normalization_2d(num_channels: int, normalization: str) -> nn.Module:
    if normalization == "batch":
        return nn.BatchNorm2d(num_channels)
    if normalization == "group":
        num_groups = min(32, max(1, num_channels // 2))
        while num_channels % num_groups:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)
    raise ValueError("normalization must be 'batch' or 'group'")


class ConvNormAct(nn.Sequential):
    """Convolution block used by TerraTorch's UPerNet implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        normalization: str = "batch",
    ):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=False
            ),
            _normalization_2d(out_channels, normalization),
            nn.ReLU(),
        )


class PyramidPooling(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, normalization: str = "batch"
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvNormAct(in_channels, out_channels, 1, normalization),
                )
                for scale in (1, 2, 3, 6)
            ]
        )
        self.bottleneck = ConvNormAct(
            in_channels + 4 * out_channels, out_channels, 3, normalization
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        pooled = [feature]
        for branch in self.branches:
            output = branch(feature)
            pooled.append(
                F.interpolate(
                    output,
                    size=feature.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            )
        return self.bottleneck(torch.cat(pooled, dim=1))


class DenseLinearProbe(nn.Module):
    """Equal-capacity shared 1x1 probe used only for representation selection."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Conv2d(feature_dim, num_classes, 1)

    def forward(
        self, features: torch.Tensor, target_size: Sequence[int]
    ) -> torch.Tensor:
        logits = torch.stack(
            [self.classifier(features[:, index]) for index in range(features.shape[1])],
            dim=0,
        ).mean(dim=0)
        return F.interpolate(
            logits, size=tuple(target_size), mode="bilinear", align_corners=False
        )


class UPerNetProbe(nn.Module):
    """TerraTorch-style UPerNet decoder for a plain transformer backbone."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        channels: int = 256,
        dropout: float = 0.0,
        normalization: str = "batch",
    ):
        super().__init__()
        if feature_dim % 4:
            raise ValueError("feature_dim must be divisible by four")
        if normalization not in {"batch", "group"}:
            raise ValueError("normalization must be 'batch' or 'group'")
        self.normalization = normalization

        # TerraTorch's 2025 plain-ViT UPerNet path creates a spatial pyramid
        # from four same-resolution transformer feature maps.
        self.scale_modules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(feature_dim, feature_dim // 2, 2, 2),
                    _normalization_2d(feature_dim // 2, normalization),
                    nn.GELU(),
                    nn.ConvTranspose2d(feature_dim // 2, feature_dim // 4, 2, 2),
                ),
                nn.ConvTranspose2d(feature_dim, feature_dim // 2, 2, 2),
                nn.Identity(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )
        pyramid_channels = (
            feature_dim // 4,
            feature_dim // 2,
            feature_dim,
            feature_dim,
        )
        self.lateral = nn.ModuleList(
            [
                ConvNormAct(stage_channels, channels, 1, normalization)
                for stage_channels in pyramid_channels[:3]
            ]
        )
        self.psp = PyramidPooling(feature_dim, channels, normalization)
        self.fpn = nn.ModuleList(
            [ConvNormAct(channels, channels, 3, normalization) for _ in range(3)]
        )
        self.fusion = ConvNormAct(4 * channels, channels, 3, normalization)
        self.dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def _scale_features(self, features: torch.Tensor) -> list[torch.Tensor]:
        return [
            scale_module(features[:, index])
            for index, scale_module in enumerate(self.scale_modules)
        ]

    def forward(
        self, features: torch.Tensor, target_size: Sequence[int]
    ) -> torch.Tensor:
        if features.shape[1] != 4:
            raise ValueError("UPerNetProbe expects exactly four encoder stages")
        scaled = self._scale_features(features)
        laterals = [self.lateral[index](scaled[index]) for index in range(3)]
        laterals.append(self.psp(scaled[-1]))
        for index in range(2, -1, -1):
            laterals[index] = laterals[index] + F.interpolate(
                laterals[index + 1],
                size=laterals[index].shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        fpn_outputs = [self.fpn[index](laterals[index]) for index in range(3)]
        fpn_outputs.append(laterals[-1])
        full_size = fpn_outputs[0].shape[-2:]
        fused = torch.cat(
            [
                output
                if output.shape[-2:] == full_size
                else F.interpolate(
                    output, size=full_size, mode="bilinear", align_corners=True
                )
                for output in fpn_outputs
            ],
            dim=1,
        )
        logits = self.classifier(self.dropout(self.fusion(fused)))
        return F.interpolate(
            logits, size=tuple(target_size), mode="bilinear", align_corners=False
        )


def load_segmentation_probe_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict]:
    """Load a saved segmentation probe and its training metadata."""
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("task") != "semantic_segmentation":
        raise ValueError(f"Not a segmentation-probe checkpoint: {path}")
    if checkpoint["head_type"] == "linear":
        head = DenseLinearProbe(checkpoint["feature_dim"], checkpoint["num_classes"])
    elif checkpoint["head_type"] == "upernet":
        head = UPerNetProbe(
            checkpoint["feature_dim"],
            checkpoint["num_classes"],
            channels=checkpoint["decoder_channels"],
            normalization=checkpoint.get("head_normalization", "batch"),
        )
    else:
        raise ValueError(f"Unknown segmentation head type: {checkpoint['head_type']}")
    head.load_state_dict(checkpoint["head_state_dict"])
    head.to(device).eval()
    return head, checkpoint


class SegmentationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        predictions = logits.argmax(dim=1).cpu()
        labels = labels.cpu()
        valid = (labels >= 0) & (labels < self.num_classes)
        indices = self.num_classes * labels[valid] + predictions[valid]
        self.confusion += torch.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        confusion = self.confusion.to(torch.float64)
        true_positive = confusion.diag()
        union = confusion.sum(dim=0) + confusion.sum(dim=1) - true_positive
        class_count = confusion.sum(dim=1)
        valid_iou = union > 0
        valid_accuracy = class_count > 0
        per_class_iou = torch.where(
            valid_iou, true_positive / union.clamp_min(1), torch.nan
        )
        return {
            "mean_iou": float(per_class_iou[valid_iou].mean()),
            "pixel_accuracy": float(true_positive.sum() / confusion.sum().clamp_min(1)),
            "mean_accuracy": float(
                (true_positive[valid_accuracy] / class_count[valid_accuracy]).mean()
            ),
            "per_class_iou": [
                None if torch.isnan(value) else float(value) for value in per_class_iou
            ],
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
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


@torch.inference_mode()
def _evaluate(
    head: nn.Module,
    dataloader: DataLoader,
    candidate: str,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> dict:
    head.eval()
    metrics = SegmentationMetrics(num_classes)
    use_amp = device.type == "cuda"
    loss_sum = 0.0
    pixel_count = 0
    for raw_features, labels in dataloader:
        raw_features = raw_features.to(device, non_blocking=True)
        if not use_amp:
            raw_features = raw_features.float()
        labels = labels.to(device, non_blocking=True)
        dense_features = prepare_dense_candidate(
            raw_features, candidate, norm_weight, norm_bias
        )
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = head(dense_features, labels.shape[-2:])
            loss = F.cross_entropy(logits, labels)
        loss_sum += float(loss.detach()) * labels.numel()
        pixel_count += labels.numel()
        metrics.update(logits, labels)
    result = metrics.compute()
    result["cross_entropy"] = loss_sum / max(pixel_count, 1)
    return result


def _aggregate_runs(runs: Sequence[dict], metric_key: str) -> dict:
    metric_names = runs[0][metric_key]
    aggregate = {}
    for name, first_value in metric_names.items():
        if isinstance(first_value, list):
            continue
        values = np.asarray([run[metric_key][name] for run in runs], dtype=np.float64)
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return aggregate


def train_segmentation_probe(
    train_cache: Path,
    val_cache: Path,
    num_classes: int,
    candidate: str,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    learning_rate: float,
    device: torch.device,
    test_cache: Optional[Path] = None,
    head_type: str = "upernet",
    epochs: int = 50,
    batch_size: int = 8,
    num_workers: int = 0,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    decoder_channels: int = 64,
    checkpoint_path: Optional[Path] = None,
    checkpoint_metadata: Optional[dict] = None,
) -> dict:
    """Train a pixel probe per seed, selecting each run on validation mIoU."""
    if head_type not in {"linear", "upernet"}:
        raise ValueError("head_type must be 'linear' or 'upernet'")
    train_dataset = DenseFeatureCacheDataset(train_cache)
    val_dataset = DenseFeatureCacheDataset(val_cache)
    test_dataset = DenseFeatureCacheDataset(test_cache) if test_cache else None
    cache_datasets = {
        "train": train_dataset,
        "validation": val_dataset,
        **({"test": test_dataset} if test_dataset is not None else {}),
    }
    for split_name, dataset in cache_datasets.items():
        if dataset.feature_dim != train_dataset.feature_dim:
            raise ValueError(f"{split_name} feature dimensions do not match training")
        if dataset.num_stages != 4:
            raise ValueError(f"{split_name} cache must contain four encoder stages")
        if dataset.label_shape != train_dataset.label_shape:
            raise ValueError(f"{split_name} label shape does not match training")

    norm_weight = norm_weight.detach().to(device)
    norm_bias = norm_bias.detach().to(device)
    val_loader = _make_loader(val_dataset, batch_size, num_workers, shuffle=False)
    test_loader = (
        _make_loader(test_dataset, batch_size, num_workers, shuffle=False)
        if test_dataset is not None
        else None
    )
    use_amp = device.type == "cuda"
    runs = []
    checkpoint_info = None
    seed_checkpoints = []

    for seed in seeds:
        _seed_everything(int(seed))
        if head_type == "linear":
            head = DenseLinearProbe(train_dataset.feature_dim, num_classes)
        else:
            head = UPerNetProbe(
                train_dataset.feature_dim,
                num_classes,
                channels=decoder_channels,
            )
        head.to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        train_loader = _make_loader(
            train_dataset,
            batch_size,
            num_workers,
            shuffle=True,
            seed=int(seed),
        )
        best_score = float("-inf")
        best_epoch = 0
        best_state = None
        history = []

        epoch_iterator = tqdm(
            range(1, epochs + 1),
            desc=f"{head_type} {candidate} seed {seed}",
            leave=False,
        )
        for epoch in epoch_iterator:
            head.train()
            train_loss_sum = 0.0
            train_pixel_count = 0
            for raw_features, labels in train_loader:
                raw_features = raw_features.to(device, non_blocking=True)
                if not use_amp:
                    raw_features = raw_features.float()
                labels = labels.to(device, non_blocking=True)
                dense_features = prepare_dense_candidate(
                    raw_features, candidate, norm_weight, norm_bias
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = head(dense_features, labels.shape[-2:])
                    loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite segmentation loss for seed {seed}, epoch {epoch}"
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_sum += float(loss.detach()) * labels.numel()
                train_pixel_count += labels.numel()

            val_metrics = _evaluate(
                head,
                val_loader,
                candidate,
                norm_weight,
                norm_bias,
                num_classes,
                device,
            )
            train_loss = train_loss_sum / max(train_pixel_count, 1)
            history.append(
                {
                    "epoch": int(epoch),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train_cross_entropy": train_loss,
                    "validation_cross_entropy": val_metrics["cross_entropy"],
                    "validation_mean_iou": val_metrics["mean_iou"],
                    "validation_pixel_accuracy": val_metrics["pixel_accuracy"],
                    "validation_mean_accuracy": val_metrics["mean_accuracy"],
                }
            )
            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_metrics['cross_entropy']:.4f}",
                val_miou=f"{val_metrics['mean_iou']:.4f}",
            )
            if val_metrics["mean_iou"] > best_score:
                best_score = val_metrics["mean_iou"]
                best_epoch = epoch
                best_state = deepcopy(head.state_dict())

        head.load_state_dict(best_state)
        run = {
            "seed": int(seed),
            "best_epoch": best_epoch,
            "history": history,
            "validation": _evaluate(
                head,
                val_loader,
                candidate,
                norm_weight,
                norm_bias,
                num_classes,
                device,
            ),
        }
        if test_loader is not None:
            run["test"] = _evaluate(
                head,
                test_loader,
                candidate,
                norm_weight,
                norm_bias,
                num_classes,
                device,
            )
        runs.append(run)
        if checkpoint_path is not None:
            payload = {
                "format_version": 2,
                "task": "semantic_segmentation",
                "head_type": head_type,
                "head_state_dict": {
                    name: value.detach().cpu()
                    for name, value in head.state_dict().items()
                },
                "feature_dim": int(train_dataset.feature_dim),
                "num_classes": int(num_classes),
                "candidate": candidate,
                "decoder_channels": int(decoder_channels),
                "head_normalization": "batch",
                "selection_metric": "mean_iou",
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "best_validation_score": float(best_score),
                "inference": dict(checkpoint_metadata or {}),
            }
            seed_path = Path(checkpoint_path).parent / "heads" / f"head_seed_{seed}.pth"
            _save_checkpoint(payload, seed_path)
            seed_checkpoints.append(str(seed_path))
            if (
                checkpoint_info is None
                or best_score > checkpoint_info["best_validation_score"]
            ):
                _save_checkpoint(payload, Path(checkpoint_path))
                checkpoint_info = {
                    "path": str(checkpoint_path),
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "best_validation_score": float(best_score),
                }

    result = {
        "candidate": candidate,
        "head_type": head_type,
        "selection_metric": "mean_iou",
        "runs": runs,
        "aggregate_validation": _aggregate_runs(runs, "validation"),
    }
    if test_loader is not None:
        result["aggregate_test"] = _aggregate_runs(runs, "test")
    if checkpoint_info is not None:
        result["checkpoint"] = checkpoint_info
        result["seed_checkpoints"] = seed_checkpoints
    return result
