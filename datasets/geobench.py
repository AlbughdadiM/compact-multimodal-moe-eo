"""Small GEO-Bench adapter for frozen classification probes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


CLASSIFICATION_DATASETS = (
    "m-bigearthnet",
    "m-brick-kiln",
    "m-so2sat",
    "m-eurosat",
)

SEGMENTATION_DATASETS = (
    "m-cashew-plant",
    "m-SA-crop-type",
)

GEOBENCH_PREPROCESSING_VERSION = "v3_zero_nodata_masked_resize"

# EuroSAT GeoTIFFs store channels as B1, ..., B12, B8A. The GEO-Bench
# conversion assigned those arrays to the standard S2 metadata order, which
# places B8A between B8 and B9. These aliases recover the physical bands from
# the resulting m-eurosat files before mapping them to model input names.
EUROSAT_SOURCE_BAND_BY_TARGET = {
    "B8A": "B12",
    "B9": "B8A",
    "B10": "B9",
    "B11": "B10",
    "B12": "B11",
}


def _normalized_band_number(value: str) -> str:
    value = value.upper().strip().split(" - ", maxsplit=1)[0]
    if value.startswith("B"):
        value = value[1:]
    return value.lstrip("0") or "0"


def _find_s2_band(bands_info, target_name: str):
    target_number = _normalized_band_number(target_name)
    for band_info in bands_info:
        if band_info.__class__.__name__ != "Sentinel2":
            continue
        names = [band_info.name, *getattr(band_info, "alt_names", ())]
        if any(_normalized_band_number(name) == target_number for name in names):
            return band_info.name
    return None


def _find_s1_band(bands_info, polarization: str):
    polarization = polarization.upper()
    candidates = []
    for band_info in bands_info:
        if band_info.__class__.__name__ != "Sentinel1":
            continue
        names = [band_info.name, *getattr(band_info, "alt_names", ())]
        upper_names = [name.upper().strip() for name in names]
        if polarization in upper_names:
            candidates.append((0, band_info.name))
            continue

        full_name = band_info.name.upper()
        if polarization not in full_name:
            continue
        is_filtered_intensity = (
            "LEE FILTERED" in full_name
            and ".REAL" not in full_name
            and ".IMAGINARY" not in full_name
        )
        candidates.append((1 if is_filtered_intensity else 2, band_info.name))

    if not candidates:
        return None
    return min(candidates)[1]


def resolve_compatible_bands(
    bands_info,
    model_band_names: Dict[str, Sequence[str]],
    s1_modality: str,
    dataset_name: str | None = None,
) -> Dict[str, list[tuple[str, str]]]:
    """Map available GEO-Bench bands to names understood by the pretrained model."""
    selected: Dict[str, list[tuple[str, str]]] = {}
    if "sentinel2" in model_band_names:
        pairs = []
        for target_name in model_band_names["sentinel2"]:
            lookup_name = target_name
            if dataset_name == "m-eurosat":
                lookup_name = EUROSAT_SOURCE_BAND_BY_TARGET.get(target_name, target_name)
            source_name = _find_s2_band(bands_info, lookup_name)
            if source_name is not None:
                pairs.append((source_name, target_name))
        if pairs:
            selected["sentinel2"] = pairs

    if s1_modality in model_band_names:
        pairs = []
        for target_name in model_band_names[s1_modality]:
            source_name = _find_s1_band(bands_info, target_name)
            if source_name is not None:
                pairs.append((source_name, target_name))
        if pairs:
            selected[s1_modality] = pairs

    if not selected:
        raise ValueError("The dataset has no bands compatible with the pretrained model")
    return selected


def _prepare_raster_modality(sample, pairs, dataset):
    source_names = [source for source, _ in pairs]
    array, _ = sample.pack_to_3d(
        band_names=source_names,
        resample=True,
        resample_order=0,
    )
    array = array.astype(np.float32, copy=False)
    # GEO-Bench reserves raw zero for nodata in its raster archive.
    validity = np.isfinite(array) & (array != 0)
    array = np.where(validity, array, 0.0)

    means = np.asarray(
        [dataset.band_stats[name].mean for name in source_names], dtype=np.float32
    )
    stds = np.asarray(
        [dataset.band_stats[name].std for name in source_names], dtype=np.float32
    )
    if np.any(stds <= 0):
        raise ValueError(f"Non-positive band standard deviation for {source_names}")
    array = (array - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)
    array = np.where(validity, array, 0.0)
    return (
        torch.from_numpy(np.moveaxis(array, -1, 0).copy()),
        torch.from_numpy(np.moveaxis(validity, -1, 0).copy()),
    )


def _resize_raster(
    tensor: torch.Tensor,
    validity: torch.Tensor,
    target_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize normalized rasters without blending nodata zeros into valid pixels."""
    validity_float = validity.unsqueeze(0).float()
    weights = F.interpolate(
        validity_float,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    values = F.interpolate(
        (tensor * validity).unsqueeze(0),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    resized_validity = weights > 0
    resized = torch.where(
        resized_validity,
        values / weights.clamp_min(1e-6),
        torch.zeros_like(values),
    )
    return resized.squeeze(0), resized_validity.squeeze(0)


class GeoBenchClassificationDataset(Dataset):
    """Convert an official GEO-Bench classification split to the model input format."""

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        split: str,
        model_band_names: Dict[str, Sequence[str]],
        s1_modality: str = "sentinel1_asc",
        partition_name: str = "default",
        input_image_size: int | None = 64,
    ):
        if dataset_name not in CLASSIFICATION_DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset_name}'")
        if s1_modality not in {"sentinel1_asc", "sentinel1_desc"}:
            raise ValueError("s1_modality must be 'sentinel1_asc' or 'sentinel1_desc'")
        if input_image_size is not None and input_image_size <= 0:
            raise ValueError("input_image_size must be positive or None")
        self.input_image_size = input_image_size

        os.environ.setdefault("GEO_BENCH_DIR", str(Path(root_dir).expanduser()))
        try:
            from geobench.dataset import GeobenchDataset  # pylint: disable=import-outside-toplevel
            from geobench.task import load_task_specs  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "GEO-Bench evaluation requires the optional 'geobench' package"
            ) from exc

        dataset_dir = Path(root_dir).expanduser() / "classification_v1.0" / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"GEO-Bench dataset directory not found: {dataset_dir}")

        self.task_specs = load_task_specs(dataset_dir)
        self.band_mapping = resolve_compatible_bands(
            self.task_specs.bands_info,
            model_band_names=model_band_names,
            s1_modality=s1_modality,
            dataset_name=dataset_name,
        )
        self.preprocessing_signature = (
            f"{GEOBENCH_PREPROCESSING_VERSION}_{dataset_name}_"
            f"input_{input_image_size or 'native'}"
        )
        source_names = [source for pairs in self.band_mapping.values() for source, _ in pairs]
        self.dataset = GeobenchDataset(
            dataset_dir=dataset_dir,
            split=split,
            partition_name=partition_name,
            band_names=source_names,
            format="hdf5",
        )
        self.raster_band_names = {
            modality: [target for _, target in pairs]
            for modality, pairs in self.band_mapping.items()
        }
        self.num_classes = int(self.task_specs.label_type.n_classes)
        self.multilabel = (
            self.task_specs.label_type.__class__.__name__ == "MultiLabelClassification"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        raster_dict = {}
        raster_valid_masks = {}
        spatial_shape = None
        for modality, pairs in self.band_mapping.items():
            tensor, validity = _prepare_raster_modality(sample, pairs, self.dataset)
            if self.input_image_size is not None:
                target_size = (self.input_image_size, self.input_image_size)
                tensor, validity = _resize_raster(tensor, validity, target_size)
            if spatial_shape is None:
                spatial_shape = tensor.shape[-2:]
            elif tensor.shape[-2:] != spatial_shape:
                raise ValueError("GEO-Bench modalities do not share one spatial shape")
            raster_dict[modality] = tensor
            raster_valid_masks[modality] = validity

        label_dtype = torch.float32 if self.multilabel else torch.long
        return {
            "raster_dict": raster_dict,
            "raster_valid_masks": raster_valid_masks,
            "label": torch.as_tensor(sample.label, dtype=label_dtype),
            "sample_id": sample.sample_name,
        }


class GeoBenchSegmentationDataset(Dataset):
    """Convert an official GEO-Bench segmentation split to the model input format."""

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        split: str,
        model_band_names: Dict[str, Sequence[str]],
        s1_modality: str = "sentinel1_asc",
        partition_name: str = "default",
        input_image_size: int | None = 64,
        label_image_size: int | None = 224,
    ):
        if dataset_name not in SEGMENTATION_DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset_name}'")
        if s1_modality not in {"sentinel1_asc", "sentinel1_desc"}:
            raise ValueError("s1_modality must be 'sentinel1_asc' or 'sentinel1_desc'")
        if input_image_size is not None and input_image_size <= 0:
            raise ValueError("input_image_size must be positive or None")
        if label_image_size is not None and label_image_size <= 0:
            raise ValueError("label_image_size must be positive or None")
        self.input_image_size = input_image_size
        self.label_image_size = label_image_size

        os.environ.setdefault("GEO_BENCH_DIR", str(Path(root_dir).expanduser()))
        try:
            from geobench.dataset import (
                GeobenchDataset,
            )  # pylint: disable=import-outside-toplevel
            from geobench.task import (
                load_task_specs,
            )  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "GEO-Bench evaluation requires the optional 'geobench' package"
            ) from exc

        dataset_dir = Path(root_dir).expanduser() / "segmentation_v1.0" / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"GEO-Bench dataset directory not found: {dataset_dir}"
            )

        self.task_specs = load_task_specs(dataset_dir)
        self.band_mapping = resolve_compatible_bands(
            self.task_specs.bands_info,
            model_band_names=model_band_names,
            s1_modality=s1_modality,
            dataset_name=dataset_name,
        )
        self.preprocessing_signature = (
            f"{GEOBENCH_PREPROCESSING_VERSION}_{dataset_name}_segmentation_"
            f"input_{input_image_size or 'native'}_label_"
            f"{label_image_size or 'native'}"
        )
        source_names = [
            source for pairs in self.band_mapping.values() for source, _ in pairs
        ]
        self.dataset = GeobenchDataset(
            dataset_dir=dataset_dir,
            split=split,
            partition_name=partition_name,
            band_names=source_names,
            format="hdf5",
        )
        self.raster_band_names = {
            modality: [target for _, target in pairs]
            for modality, pairs in self.band_mapping.items()
        }
        self.num_classes = int(self.task_specs.label_type.n_classes)
        self.class_names = self.task_specs.label_type.class_names

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        raster_dict = {}
        raster_valid_masks = {}
        spatial_shape = None
        for modality, pairs in self.band_mapping.items():
            tensor, validity = _prepare_raster_modality(
                sample, pairs, self.dataset
            )
            if self.input_image_size is not None:
                target_size = (self.input_image_size, self.input_image_size)
                tensor, validity = _resize_raster(tensor, validity, target_size)
            if spatial_shape is None:
                spatial_shape = tensor.shape[-2:]
            elif tensor.shape[-2:] != spatial_shape:
                raise ValueError("GEO-Bench modalities do not share one spatial shape")
            raster_dict[modality] = tensor
            raster_valid_masks[modality] = validity

        label = np.asarray(sample.label.data)
        label = np.squeeze(label)
        if label.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional segmentation label, got {label.shape}"
            )
        if not np.issubdtype(label.dtype, np.integer):
            if not np.all(np.isfinite(label)) or not np.all(label == np.floor(label)):
                raise ValueError(
                    "Segmentation labels must contain finite integer class indices"
                )
        label = label.astype(np.int64, copy=False)
        if label.min() < 0 or label.max() >= self.num_classes:
            raise ValueError(
                f"Segmentation labels must be in [0, {self.num_classes - 1}]"
            )

        label_tensor = torch.from_numpy(label.copy())
        if self.label_image_size is not None:
            label_tensor = F.interpolate(
                label_tensor[None, None].float(),
                size=(self.label_image_size, self.label_image_size),
                mode="nearest",
            )[0, 0].long()

        return {
            "raster_dict": raster_dict,
            "raster_valid_masks": raster_valid_masks,
            "label": label_tensor,
            "sample_id": sample.sample_name,
        }
