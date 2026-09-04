"""
PyTorch Dataset for loading MMEarth data from the official HDF5 release.

This loader is designed around the official MMEarth/MMEarth64/MMEarth100k
directory structure, but it also accepts a root path that already points to
the chosen subset directory:

  <root_dir>/
    data_1M_v001_64/
      data_1M_v001_64.h5
      data_1M_v001_64_band_stats.json
      data_1M_v001_64_splits.json
      data_1M_v001_64_tile_info.json

It returns dict-style samples suitable for the multimodal pretraining path:

  {
      "raster_dict": {"sentinel2": Tensor[C,H,W], ...},
      "raster_valid_masks": {"sentinel2": BoolTensor[C,H,W], ...},
      "meta_dict": {"lat": Tensor[2], "lon": Tensor[2], ...},
      "meta_valid_masks": {"lat": BoolTensor[2], "lon": BoolTensor[2], ...},
      "tile_id": str,
      "date": str,
  }

Author: Mohanad Albughdadi
Created: 2026-08-04
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MMEarthDataset(Dataset):
    """Dataset wrapper for the official MMEarth releases."""

    subsets = {
        "MMEarth": "data_1M_v001",
        "MMEarth64": "data_1M_v001_64",
        "MMEarth100k": "data_100k_v001",
    }

    raster_modalities = (
        "sentinel2",
        "sentinel1_asc",
        "sentinel1_desc",
        "aster",
        "canopy_height_eth",
        "dynamic_world",
        "esa_worldcover",
        "sentinel2_cloudmask",
        "sentinel2_cloudprod",
        "sentinel2_scl",
    )

    metadata_modalities = (
        "lat",
        "lon",
        "month",
        "era5",
    )

    all_modality_bands: Dict[str, list[str]] = {
        "sentinel2": [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8A",
            "B8",
            "B9",
            "B10",
            "B11",
            "B12",
        ],
        "sentinel1_asc": ["VV", "VH", "HH", "HV"],
        "sentinel1_desc": ["VV", "VH", "HH", "HV"],
        "aster": ["b1", "slope"],
        "canopy_height_eth": ["height", "std"],
        "dynamic_world": ["label"],
        "esa_worldcover": ["Map"],
        "sentinel2_cloudmask": ["QA60"],
        "sentinel2_cloudprod": ["MSK_CLDPRB"],
        "sentinel2_scl": ["SCL"],
        "lat": ["sin", "cos"],
        "lon": ["sin", "cos"],
        "month": ["sin_month", "cos_month"],
        "era5": [
            "month1_temperature_2m",
            "month1_temperature_2m_min",
            "month1_temperature_2m_max",
            "month1_total_precipitation_sum",
            "month2_temperature_2m",
            "month2_temperature_2m_min",
            "month2_temperature_2m_max",
            "month2_total_precipitation_sum",
            "0_temperature_2m_mean",
            "1_temperature_2m_min_min",
            "2_temperature_2m_max_max",
            "3_total_precipitation_sum_sum",
        ],
    }

    no_data_vals: Dict[str, int | float] = {
        "sentinel2": 0,
        "sentinel1_asc": float("-inf"),
        "sentinel1_desc": float("-inf"),
        "aster": float("-inf"),
        "canopy_height_eth": 255,
        "dynamic_world": 0,
        "esa_worldcover": 255,
        "sentinel2_cloudmask": 65535,
        "sentinel2_cloudprod": 65535,
        "sentinel2_scl": 255,
        "lat": float("-inf"),
        "lon": float("-inf"),
        "month": float("-inf"),
        "era5": float("inf"),
    }

    def __init__(
        self,
        root_dir: str,
        subset: str = "MMEarth64",
        split: str = "train",
        raster_modalities: Sequence[str] = ("sentinel2",),
        metadata_modalities: Sequence[str] = ("lat", "lon", "month"),
        modality_bands: Optional[Dict[str, Sequence[str]]] = None,
        normalization_mode: str = "z-score",
        fill_value: float = 0.0,
        fallback_split_from_train: bool = False,
        val_fraction: float = 0.1,
        test_fraction: float = 0.0,
        split_seed: int = 42,
        transform=None,
    ):
        super().__init__()
        if subset not in self.subsets:
            raise ValueError(f"Unsupported subset '{subset}'. Expected one of {sorted(self.subsets)}")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train', 'val', 'test'}")
        if normalization_mode not in {"z-score", "min-max"}:
            raise ValueError("normalization_mode must be 'z-score' or 'min-max'")

        self.root_dir = root_dir
        self.subset = subset
        self.split = split
        self.normalization_mode = normalization_mode
        self.fill_value = fill_value
        self.fallback_split_from_train = fallback_split_from_train
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.split_seed = split_seed
        self.transform = transform

        dataset_dir_name = self.subsets[subset]
        self.dataset_dir = self._resolve_dataset_dir(root_dir, dataset_dir_name)
        self.dataset_path = os.path.join(self.dataset_dir, f"{dataset_dir_name}.h5")
        self.band_stats_path = os.path.join(
            self.dataset_dir, f"{dataset_dir_name}_band_stats.json"
        )
        self.splits_path = os.path.join(self.dataset_dir, f"{dataset_dir_name}_splits.json")
        self.tile_info_path = os.path.join(
            self.dataset_dir, f"{dataset_dir_name}_tile_info.json"
        )
        self._verify()

        with open(self.band_stats_path, "r", encoding="utf-8") as f:
            self.band_stats = json.load(f)
        with open(self.splits_path, "r", encoding="utf-8") as f:
            split_indices = json.load(f)
        with open(self.tile_info_path, "r", encoding="utf-8") as f:
            self.tile_info = json.load(f)

        self.sample_tile_info = next(iter(self.tile_info.values()))
        self.all_modality_bands = self._build_available_modality_bands()
        self.raster_modalities = tuple(raster_modalities)
        self.metadata_modalities = tuple(metadata_modalities)
        self._validate_requested_modalities()
        self.modality_bands = self._build_modality_bands(modality_bands)
        self._validate_modality_bands(self.modality_bands)

        self.indices = self._build_split_indices(split_indices)
        self._h5_file = None

    @staticmethod
    def _resolve_dataset_dir(root_dir: str, dataset_dir_name: str) -> str:
        root_dir = os.path.abspath(root_dir)
        direct_path = os.path.join(root_dir, f"{dataset_dir_name}.h5")
        if os.path.exists(direct_path):
            return root_dir

        nested_dir = os.path.join(root_dir, dataset_dir_name)
        nested_path = os.path.join(nested_dir, f"{dataset_dir_name}.h5")
        if os.path.exists(nested_path):
            return nested_dir

        return root_dir

    def _verify(self) -> None:
        required = (
            self.dataset_path,
            self.band_stats_path,
            self.splits_path,
            self.tile_info_path,
        )
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "MMEarth dataset files are missing:\n" + "\n".join(missing)
            )

    def _build_available_modality_bands(self) -> Dict[str, list[str]]:
        sample_bands = self.sample_tile_info.get("BANDS", {})
        bands = {
            "sentinel2": list(self.__class__.all_modality_bands["sentinel2"]),
            "sentinel1_asc": list(self.__class__.all_modality_bands["sentinel1_asc"]),
            "sentinel1_desc": list(self.__class__.all_modality_bands["sentinel1_desc"]),
            "aster": list(sample_bands.get("aster", [])),
            "canopy_height_eth": list(sample_bands.get("canopy_height_eth", [])),
            "dynamic_world": list(sample_bands.get("dynamic_world", [])),
            "esa_worldcover": list(sample_bands.get("esa_worldcover", [])),
            "sentinel2_cloudmask": list(self.__class__.all_modality_bands["sentinel2_cloudmask"]),
            "sentinel2_cloudprod": list(self.__class__.all_modality_bands["sentinel2_cloudprod"]),
            "sentinel2_scl": list(self.__class__.all_modality_bands["sentinel2_scl"]),
            "lat": list(self.__class__.all_modality_bands["lat"]),
            "lon": list(self.__class__.all_modality_bands["lon"]),
            "month": list(self.__class__.all_modality_bands["month"]),
            "era5": list(self.__class__.all_modality_bands["era5"]),
        }
        return bands

    def _validate_requested_modalities(self) -> None:
        if not self.raster_modalities:
            raise ValueError("At least one raster modality must be requested")
        unknown_rasters = set(self.raster_modalities) - set(self.__class__.raster_modalities)
        if unknown_rasters:
            raise ValueError(f"Unsupported raster modalities: {sorted(unknown_rasters)}")
        unknown_meta = set(self.metadata_modalities) - set(self.__class__.metadata_modalities)
        if unknown_meta:
            raise ValueError(f"Unsupported metadata modalities: {sorted(unknown_meta)}")
        unavailable_rasters = [
            name for name in self.raster_modalities if not self.all_modality_bands.get(name)
        ]
        if unavailable_rasters:
            raise ValueError(
                f"Requested raster modalities are not available in this release: {unavailable_rasters}"
            )

    def _build_modality_bands(
        self, modality_bands: Optional[Dict[str, Sequence[str]]]
    ) -> Dict[str, list[str]]:
        requested = self.raster_modalities + self.metadata_modalities
        bands: Dict[str, list[str]] = {}
        for modality in requested:
            if modality_bands is not None and modality in modality_bands:
                bands[modality] = list(modality_bands[modality])
            else:
                bands[modality] = list(self.all_modality_bands[modality])
        return bands

    def _validate_modality_bands(self, modality_bands: Dict[str, list[str]]) -> None:
        for modality, bands in modality_bands.items():
            if modality not in self.all_modality_bands:
                raise ValueError(f"Unknown modality '{modality}'")
            invalid = set(bands) - set(self.all_modality_bands[modality])
            if invalid:
                raise ValueError(
                    f"Invalid bands for modality '{modality}': {sorted(invalid)}"
                )

    def _build_split_indices(self, split_indices: Dict[str, list[int]]) -> list[int]:
        if self.split not in split_indices:
            raise KeyError(f"Split '{self.split}' not found in {self.splits_path}")

        official_train = list(split_indices.get("train", []))
        official_val = list(split_indices.get("val", []))
        official_test = list(split_indices.get("test", []))
        requested = list(split_indices[self.split])

        if not self.fallback_split_from_train:
            return requested
        if official_val or official_test:
            return requested

        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("test_fraction must be in [0, 1)")

        train_indices = np.asarray(official_train, dtype=np.int64)
        rng = np.random.default_rng(self.split_seed)
        permuted = rng.permutation(train_indices)

        num_test = int(round(len(permuted) * self.test_fraction))
        num_val = int(round(len(permuted) * self.val_fraction))
        if num_test + num_val >= len(permuted):
            raise ValueError("Fallback split leaves no training samples")

        test_indices = permuted[:num_test]
        val_indices = permuted[num_test : num_test + num_val]
        train_split = permuted[num_test + num_val :]

        if self.split == "train":
            return train_split.tolist()
        if self.split == "val":
            return val_indices.tolist()
        return test_indices.tolist()

    def _get_h5(self):
        if self._h5_file is None:
            try:
                import h5py  # pylint: disable=import-outside-toplevel
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "MMEarthDataset requires 'h5py'. Add it to the environment before use."
                ) from exc
            self._h5_file = h5py.File(self.dataset_path, "r")
        return self._h5_file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def input_adapters(self) -> Dict[str, int]:
        return {name: len(self.modality_bands[name]) for name in self.raster_modalities}

    @property
    def metadata_dims(self) -> Dict[str, int]:
        return {name: len(self.modality_bands[name]) for name in self.metadata_modalities}

    @property
    def primary_input_name(self) -> str:
        return self.raster_modalities[0]

    def _storage_key(self, modality: str) -> str:
        if modality in {"sentinel1_asc", "sentinel1_desc"}:
            return "sentinel1"
        return modality

    def _band_stats_key(self, modality: str, tile_info: Dict[str, Any]) -> str:
        if modality == "sentinel2":
            s2_type = str(tile_info.get("S2_type", "")).lower()
            candidate = "sentinel2_l2a" if s2_type == "l2a" else "sentinel2_l1c"
            if candidate in self.band_stats:
                return candidate
        if modality == "sentinel2_cloudmask":
            s2_type = str(tile_info.get("S2_type", "")).lower()
            candidate = (
                "sentinel2_cloudmask_l2a" if s2_type == "l2a" else "sentinel2_cloudmask_l1c"
            )
            if candidate in self.band_stats:
                return candidate
        if modality == "sentinel2_cloudprod":
            s2_type = str(tile_info.get("S2_type", "")).lower()
            candidate = (
                "sentinel2_cloudprod_l2a" if s2_type == "l2a" else "sentinel2_cloudprod_l1c"
            )
            if candidate in self.band_stats:
                return candidate
        if modality.startswith("sentinel1") and "sentinel1" in self.band_stats:
            return "sentinel1"
        return modality

    def _select_indices_for_modality(self, modality: str, bands: Sequence[str]) -> list[int]:
        if modality == "sentinel1_desc":
            offset = len(self.all_modality_bands["sentinel1_asc"])
            return [self.all_modality_bands["sentinel1_desc"].index(band) + offset for band in bands]
        return [self.all_modality_bands[modality].index(band) for band in bands]

    def _normalize_modality(
        self,
        data: np.ndarray,
        modality: str,
        bands: Sequence[str],
        tile_info: Dict[str, Any],
    ) -> np.ndarray:
        stats_key = self._band_stats_key(modality, tile_info)
        if stats_key not in self.band_stats:
            return data

        indices = self._select_indices_for_modality(modality, bands)
        stats = self.band_stats[stats_key]
        if self.normalization_mode == "z-score":
            mean = np.asarray(stats["mean"], dtype=np.float32)[indices]
            std = np.asarray(stats["std"], dtype=np.float32)[indices]
            std = np.where(std == 0, 1.0, std)
            if data.ndim == 3:
                return (data - mean[:, None, None]) / std[:, None, None]
            return (data - mean) / std

        min_val = np.asarray(stats["min"], dtype=np.float32)[indices]
        max_val = np.asarray(stats["max"], dtype=np.float32)[indices]
        denom = np.where((max_val - min_val) == 0, 1.0, max_val - min_val)
        if data.ndim == 3:
            return (data - min_val[:, None, None]) / denom[:, None, None]
        return (data - min_val) / denom

    def _build_invalid_mask(self, data: np.ndarray, modality: str) -> np.ndarray:
        invalid = ~np.isfinite(data)
        no_data_value = self.no_data_vals.get(modality)
        if no_data_value is not None and np.isfinite(no_data_value):
            invalid |= data == no_data_value
        return invalid

    def _sanitize_float_data(
        self,
        data: np.ndarray,
        invalid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if invalid_mask is None:
            invalid_mask = ~np.isfinite(data)
        data[invalid_mask] = self.fill_value
        return data

    def _remap_dynamic_world(self, data: np.ndarray) -> np.ndarray:
        invalid_mask = data == self.no_data_vals["dynamic_world"]
        remapped = data.copy()
        old_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        for old, new in zip(old_values, new_values):
            remapped = np.where(remapped == old, new, remapped)
        remapped = remapped.astype(np.float32, copy=False)
        remapped[invalid_mask] = self.fill_value
        return remapped.astype(np.float32)

    def _remap_esa_worldcover(self, data: np.ndarray) -> np.ndarray:
        invalid_mask = data == self.no_data_vals["esa_worldcover"]
        remapped = data.copy()
        old_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
        new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for old, new in zip(old_values, new_values):
            remapped = np.where(remapped == old, new, remapped)
        remapped = remapped.astype(np.float32, copy=False)
        remapped[invalid_mask] = self.fill_value
        return remapped.astype(np.float32)

    def _process_modality(
        self,
        data: np.ndarray,
        modality: str,
        bands: Sequence[str],
        tile_info: Dict[str, Any],
        target_hw: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        indices = self._select_indices_for_modality(modality, bands)
        data = data[indices, ...]
        invalid_mask = self._build_invalid_mask(data, modality)

        if modality == "dynamic_world":
            return torch.from_numpy(self._remap_dynamic_world(data))
        if modality == "esa_worldcover":
            return torch.from_numpy(self._remap_esa_worldcover(data))
        if modality in {"sentinel2_cloudmask", "sentinel2_cloudprod", "sentinel2_scl"}:
            data = data.astype(np.float32, copy=False)
            data = self._sanitize_float_data(data, invalid_mask=invalid_mask)
            tensor = torch.from_numpy(data)
            if target_hw is not None and tensor.shape[-2:] != target_hw:
                resize_mode = "bilinear" if modality == "sentinel2_cloudprod" else "nearest"
                tensor = self._resize_tensor(tensor, target_hw, mode=resize_mode)
            return tensor

        data = data.astype(np.float32, copy=False)
        data = self._normalize_modality(data, modality, bands, tile_info)
        data = self._sanitize_float_data(data, invalid_mask=invalid_mask)
        return torch.from_numpy(data)

    def _process_modality_with_validity(
        self,
        data: np.ndarray,
        modality: str,
        bands: Sequence[str],
        tile_info: Dict[str, Any],
        target_hw: Optional[tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized values and the validity mask from the raw source data."""
        indices = self._select_indices_for_modality(modality, bands)
        selected = data[indices, ...]
        validity = torch.from_numpy(~self._build_invalid_mask(selected, modality))
        tensor = self._process_modality(
            data,
            modality,
            bands,
            tile_info,
            target_hw=target_hw,
        )
        if target_hw is not None and validity.shape[-2:] != target_hw:
            validity = self._resize_tensor(validity.float(), target_hw, mode="nearest").bool()
        return tensor, validity

    @staticmethod
    def _resize_tensor(
        tensor: torch.Tensor,
        target_hw: tuple[int, int],
        mode: str,
    ) -> torch.Tensor:
        tensor = tensor.unsqueeze(0)
        if mode in {"bilinear", "bicubic"}:
            resized = F.interpolate(tensor, size=target_hw, mode=mode, align_corners=False)
        else:
            resized = F.interpolate(tensor, size=target_hw, mode=mode)
        return resized.squeeze(0)

    @staticmethod
    def _decode_tile_id(raw_metadata: Any) -> str:
        if isinstance(raw_metadata, bytes):
            return raw_metadata.decode("utf-8")
        if isinstance(raw_metadata, str):
            return raw_metadata
        if isinstance(raw_metadata, np.void):
            if raw_metadata.dtype.names:
                first_field = raw_metadata[raw_metadata.dtype.names[0]]
                return MMEarthDataset._decode_tile_id(first_field)
            if len(raw_metadata) > 0:
                return MMEarthDataset._decode_tile_id(raw_metadata[0])
        if isinstance(raw_metadata, np.ndarray):
            if raw_metadata.ndim == 0:
                return MMEarthDataset._decode_tile_id(raw_metadata.item())
            if raw_metadata.size >= 1:
                return MMEarthDataset._decode_tile_id(raw_metadata.reshape(-1)[0])
        if isinstance(raw_metadata, (list, tuple)) and len(raw_metadata) >= 1:
            return MMEarthDataset._decode_tile_id(raw_metadata[0])
        return str(raw_metadata)

    @staticmethod
    def _load_storage_array(
        h5_file,
        storage_cache: Dict[str, np.ndarray],
        storage_key: str,
        ds_index: int,
    ) -> np.ndarray:
        if storage_key not in storage_cache:
            storage_cache[storage_key] = h5_file[storage_key][ds_index][:]
        return storage_cache[storage_key]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ds_index = self.indices[index]
        h5_file = self._get_h5()
        tile_id = self._decode_tile_id(h5_file["metadata"][ds_index])
        tile_info = self.tile_info[tile_id]

        raster_dict = {}
        raster_valid_masks = {}
        target_hw = None
        storage_cache: Dict[str, np.ndarray] = {}

        for modality in self.raster_modalities:
            if modality in {"sentinel2_cloudmask", "sentinel2_cloudprod", "sentinel2_scl"}:
                continue
            storage_key = self._storage_key(modality)
            bands = self.modality_bands[modality]
            modality_data = self._load_storage_array(h5_file, storage_cache, storage_key, ds_index)
            tensor, validity = self._process_modality_with_validity(
                modality_data, modality, bands, tile_info
            )
            raster_dict[modality] = tensor
            raster_valid_masks[modality] = validity
            if target_hw is None and tensor.ndim == 3:
                target_hw = (tensor.shape[-2], tensor.shape[-1])

        for modality in self.raster_modalities:
            if modality not in {"sentinel2_cloudmask", "sentinel2_cloudprod", "sentinel2_scl"}:
                continue
            storage_key = self._storage_key(modality)
            bands = self.modality_bands[modality]
            modality_data = self._load_storage_array(h5_file, storage_cache, storage_key, ds_index)
            tensor, validity = self._process_modality_with_validity(
                modality_data, modality, bands, tile_info, target_hw=target_hw
            )
            raster_dict[modality] = tensor
            raster_valid_masks[modality] = validity

        meta_dict = {}
        meta_valid_masks = {}
        for modality in self.metadata_modalities:
            storage_key = self._storage_key(modality)
            bands = self.modality_bands[modality]
            modality_data = self._load_storage_array(h5_file, storage_cache, storage_key, ds_index)
            tensor, validity = self._process_modality_with_validity(
                modality_data, modality, bands, tile_info
            )
            meta_dict[modality] = tensor.view(-1)
            meta_valid_masks[modality] = validity.view(-1)

        sample = {
            "raster_dict": raster_dict,
            "raster_valid_masks": raster_valid_masks,
            "raster_band_names": {
                name: list(self.modality_bands[name]) for name in self.raster_modalities
            },
            "meta_dict": meta_dict,
            "meta_valid_masks": meta_valid_masks,
            "tile_id": tile_id,
            "date": tile_info.get("S2_DATE"),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
