"""BigEarthNet-v2 paired S1/S2 adapter for frozen CBIR evaluation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from datasets.mmearth import MMEarthDataset


S1_SOURCE_BANDS = ("VH", "VV")
S1_MODEL_BANDS = ("VV", "VH")
S2_SOURCE_BANDS_10 = (
    "B2",
    "B3",
    "B4",
    "B8",
    "B5",
    "B6",
    "B7",
    "B8A",
    "B11",
    "B12",
)
S2_SOURCE_BANDS_12 = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
)
BEN19_CLASS_NAMES = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)
VALID_SPLITS = ("train", "validation", "test")
PREPROCESSING_VERSION = "benv2_cbir_v1_mmearth_zscore_masked_resize"


@dataclass(frozen=True)
class BENv2Pair:
    """One paired S1/S2 sample and its multi-label target."""

    sample_id: str
    s1_path: Path
    s2_path: Path
    split: str
    label: np.ndarray
    s1_adapter: str


def _canonical_split(value: Any) -> str | None:
    text = str(value).strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
        "testing": "test",
    }
    return aliases.get(text)


def _split_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        split = _canonical_split(part)
        if split is not None:
            return split
    return None


def infer_s1_adapter(sample_id: str) -> str:
    """Infer the European S1 pass from its UTC acquisition time."""
    match = re.search(r"_(\d{8})T(\d{6})_", sample_id)
    if match is None:
        raise ValueError(f"Cannot infer Sentinel-1 acquisition time from '{sample_id}'")
    utc_hour = int(match.group(2)[:2])
    # Sentinel-1 passes Europe near 06:00 local descending and 18:00 local ascending.
    return "sentinel1_asc" if utc_hour >= 12 else "sentinel1_desc"


def resolve_mmearth_stats_path(config: dict, explicit_path: Path | None = None) -> Path:
    """Resolve the band-statistics JSON used by MMEarth pretraining."""
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MMEarth band statistics not found: {path}")
        return path

    training = config["training"]
    subset = training.get("dataset_subset", training.get("subset", "MMEarth64"))
    if subset not in MMEarthDataset.subsets:
        raise ValueError(f"Unknown MMEarth subset '{subset}'")
    directory_name = MMEarthDataset.subsets[subset]
    filename = f"{directory_name}_band_stats.json"
    dataset_path = Path(training["dataset_path"]).expanduser()
    candidates = (dataset_path / filename, dataset_path / directory_name / filename)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Could not resolve the MMEarth pretraining statistics. Pass "
        f"--mmearth_stats explicitly. Checked: {', '.join(map(str, candidates))}"
    )


class MMEarthBandNormalizer:
    """Apply the exact z-score statistics used for MMEarth pretraining."""

    def __init__(self, stats_path: Path):
        self.stats_path = stats_path
        with open(stats_path, "r", encoding="utf-8") as handle:
            self.stats = json.load(handle)

    def _indices(self, modality: str, bands: Sequence[str]) -> list[int]:
        available = MMEarthDataset.all_modality_bands[modality]
        offset = len(available) if modality == "sentinel1_desc" else 0
        return [available.index(band) + offset for band in bands]

    def normalize(
        self,
        values: np.ndarray,
        modality: str,
        bands: Sequence[str],
    ) -> np.ndarray:
        stats_key = "sentinel1" if modality.startswith("sentinel1") else "sentinel2_l2a"
        if stats_key not in self.stats:
            raise KeyError(f"Band statistics do not contain '{stats_key}'")
        indices = self._indices(modality, bands)
        stats = self.stats[stats_key]
        means = np.asarray(stats["mean"], dtype=np.float32)[indices]
        stds = np.asarray(stats["std"], dtype=np.float32)[indices]
        if np.any(stds <= 0):
            raise ValueError(f"Non-positive MMEarth standard deviation for {modality}")
        return (values - means[:, None, None]) / stds[:, None, None]


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    by_normalized = {_normalize_column_name(column): column for column in frame.columns}
    for alias in aliases:
        column = by_normalized.get(_normalize_column_name(alias))
        if column is not None:
            return column
    return None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ModuleNotFoundError(
                "Reading BENv2 parquet metadata requires pyarrow. Install it with "
                "'pip install pyarrow'."
            ) from exc
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        try:
            return pd.read_json(path, lines=suffix == ".jsonl")
        except ValueError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported metadata format: {path}")


def _candidate_metadata_paths(root: Path, explicit_path: Path | None) -> list[Path]:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"BENv2 metadata not found: {path}")
        return [path]

    suffixes = {".parquet", ".csv", ".json", ".jsonl"}
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and any(word in path.name.lower() for word in ("metadata", "label", "train", "val", "test"))
    ]
    candidates.sort(
        key=lambda path: (
            0 if path.name.lower() == "metadata.parquet" else 1,
            0 if "metadata" in path.name.lower() else 1,
            len(path.parts),
            str(path),
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "No BENv2 label metadata was found. Pass --metadata_path pointing to the "
            "Kaggle metadata parquet/CSV/JSON file."
        )
    return candidates


def _standardize_metadata(path: Path) -> pd.DataFrame | None:
    frame = _read_table(path)
    s1_column = _find_column(
        frame,
        ("s1_name", "s1_id", "sentinel1", "sentinel1_name", "s1_path", "image_s1"),
    )
    s2_column = _find_column(
        frame,
        ("patch_id", "s2_name", "s2_id", "sentinel2", "sentinel2_name", "s2_path", "image_s2"),
    )
    label_column = _find_column(frame, ("labels", "label", "target", "targets"))
    split_column = _find_column(frame, ("split", "partition", "subset"))
    if s1_column is None or s2_column is None:
        return None

    standardized = pd.DataFrame(
        {
            "s1_name": frame[s1_column],
            "s2_name": frame[s2_column],
        }
    )
    if label_column is not None:
        standardized["labels"] = frame[label_column]
        standardized.attrs["class_names"] = None
    else:
        excluded = {s1_column, s2_column, split_column}
        binary_columns = []
        for column in frame.columns:
            if column in excluded:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            finite = values.dropna()
            if len(finite) == len(frame) and set(finite.unique()).issubset({0, 1}):
                binary_columns.append(column)
        if len(binary_columns) < 2:
            return None
        standardized["labels"] = list(frame[binary_columns].to_numpy(dtype=np.uint8))
        standardized.attrs["class_names"] = [str(column) for column in binary_columns]

    if split_column is not None:
        standardized["split"] = frame[split_column].map(_canonical_split)
    else:
        inferred = _split_from_path(path)
        standardized["split"] = inferred
    standardized.attrs["source_path"] = str(path)
    return standardized


def _load_metadata(root: Path, explicit_path: Path | None) -> tuple[pd.DataFrame, list[str] | None]:
    accepted = []
    for path in _candidate_metadata_paths(root, explicit_path):
        standardized = _standardize_metadata(path)
        if standardized is None:
            continue
        accepted.append(standardized)
        present_splits = set(standardized["split"].dropna())
        if set(VALID_SPLITS).issubset(present_splits):
            return standardized, standardized.attrs.get("class_names")

    if not accepted:
        raise ValueError(
            "BENv2 metadata must identify S1, S2, and labels. Supported identifier "
            "columns include s1_name and patch_id."
        )
    combined = pd.concat(accepted, ignore_index=True)
    class_names = accepted[0].attrs.get("class_names")
    return combined, class_names


def _parse_label_value(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if pd.isna(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                separator = ";" if ";" in text else "|" if "|" in text else None
                return [part.strip() for part in text.split(separator)] if separator else [text]
        return list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
    return [value]


def _image_map(root: Path, directory_name: str) -> Dict[str, Path]:
    directories = [path for path in root.rglob(directory_name) if path.is_dir()]
    if len(directories) != 1:
        raise FileNotFoundError(
            f"Expected one '{directory_name}' directory under {root}, found {len(directories)}"
        )
    mapping: Dict[str, Path] = {}
    for path in directories[0].rglob("*.tif"):
        if path.stem in mapping:
            raise ValueError(f"Duplicate BENv2 image identifier '{path.stem}'")
        mapping[path.stem] = path
    if not mapping:
        raise FileNotFoundError(f"No GeoTIFF files found under {directories[0]}")
    return mapping


def _identifier_stem(value: Any) -> str:
    return Path(str(value).strip()).stem


def _encode_labels(
    rows: list[tuple[str, Path, Path, str, list[Any], str]],
    metadata_class_names: list[str] | None,
) -> tuple[list[BENv2Pair], list[str]]:
    parsed = [row[4] for row in rows]
    class_names = list(BEN19_CLASS_NAMES)
    class_to_index = {
        _normalize_column_name(name): index for index, name in enumerate(class_names)
    }
    vector_labels = bool(parsed) and all(
        len(label) >= 2
        and len(label) == len(parsed[0])
        and set(label).issubset({0, 1, False, True})
        for label in parsed
    )
    if vector_labels:
        if metadata_class_names is None:
            if len(parsed[0]) != len(class_names):
                raise ValueError(
                    "Unlabeled BENv2 target vectors must follow the 19-class order; "
                    f"found width {len(parsed[0])}"
                )
            labels = [np.asarray(label, dtype=np.uint8) for label in parsed]
        else:
            if len(metadata_class_names) != len(parsed[0]):
                raise ValueError("BENv2 class-name count does not match target-vector width")
            source_indices = {}
            for source_index, name in enumerate(metadata_class_names):
                normalized = _normalize_column_name(name)
                if normalized not in class_to_index:
                    raise ValueError(f"Unknown BENv2 class column '{name}'")
                if normalized in source_indices:
                    raise ValueError(f"Duplicate BENv2 class column '{name}'")
                source_indices[normalized] = source_index
            labels = []
            for label in parsed:
                encoded = np.zeros(len(class_names), dtype=np.uint8)
                for normalized, source_index in source_indices.items():
                    encoded[class_to_index[normalized]] = label[source_index]
                labels.append(encoded)
    else:
        labels = []
        for label in parsed:
            encoded = np.zeros(len(class_names), dtype=np.uint8)
            for item in label:
                normalized = _normalize_column_name(str(item))
                if normalized not in class_to_index:
                    raise ValueError(f"Unknown BENv2 class label '{item}'")
                encoded[class_to_index[normalized]] = 1
            labels.append(encoded)

    if any(label.sum() == 0 for label in labels):
        raise ValueError("Every BENv2 sample must have at least one positive label")

    pairs = [
        BENv2Pair(
            sample_id=row[0],
            s1_path=row[1],
            s2_path=row[2],
            split=row[3],
            label=label,
            s1_adapter=row[5],
        )
        for row, label in zip(rows, labels)
    ]
    return pairs, class_names


class BENv2CBIRIndex:
    """Resolve and validate paired BENv2-14k records once for all splits."""

    def __init__(
        self,
        root_dir: Path,
        metadata_path: Path | None = None,
        expected_pairs: int | None = 13683,
    ):
        root = root_dir.expanduser().resolve()
        nested = root / "BEN_14k"
        if nested.is_dir():
            root = nested
        if not root.is_dir():
            raise FileNotFoundError(f"BENv2-14k directory not found: {root}")
        self.root = root

        s1_images = _image_map(root, "BigEarthNet-S1")
        s2_images = _image_map(root, "BigEarthNet-S2")
        metadata, metadata_class_names = _load_metadata(root, metadata_path)

        rows = []
        seen = set()
        for record in metadata.itertuples(index=False):
            s1_id = _identifier_stem(record.s1_name)
            s2_id = _identifier_stem(record.s2_name)
            s1_path = s1_images.get(s1_id)
            s2_path = s2_images.get(s2_id)
            if s1_path is None or s2_path is None:
                continue
            split = _canonical_split(record.split)
            s1_split = _split_from_path(s1_path)
            s2_split = _split_from_path(s2_path)
            split = split or s1_split or s2_split
            if split is None or split not in VALID_SPLITS:
                raise ValueError(f"Cannot resolve split for BENv2 pair '{s2_id}'")
            if s1_split is not None and s1_split != split:
                raise ValueError(f"S1 split mismatch for '{s1_id}'")
            if s2_split is not None and s2_split != split:
                raise ValueError(f"S2 split mismatch for '{s2_id}'")
            pair_key = (s1_id, s2_id)
            if pair_key in seen:
                continue
            seen.add(pair_key)
            rows.append(
                (
                    s2_id,
                    s1_path,
                    s2_path,
                    split,
                    _parse_label_value(record.labels),
                    infer_s1_adapter(s1_id),
                )
            )

        pairs, self.class_names = _encode_labels(rows, metadata_class_names)
        if expected_pairs is not None and len(pairs) != expected_pairs:
            raise ValueError(
                f"Expected {expected_pairs} BENv2-14k pairs, resolved {len(pairs)}. "
                "Check the extracted Kaggle archive and metadata path."
            )
        self.records = {
            split: sorted(
                [pair for pair in pairs if pair.split == split],
                key=lambda pair: pair.sample_id,
            )
            for split in VALID_SPLITS
        }
        empty = [split for split, records in self.records.items() if not records]
        if empty:
            raise ValueError(f"BENv2-14k has empty required splits: {empty}")

    @property
    def split_sizes(self) -> Dict[str, int]:
        return {split: len(records) for split, records in self.records.items()}


def _resize_raster(
    values: torch.Tensor,
    validity: torch.Tensor,
    target_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[-2:] == (target_size, target_size):
        return values, validity
    weights = F.interpolate(
        validity[None].float(),
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    weighted_values = F.interpolate(
        (values * validity)[None],
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    resized_validity = weights > 0
    resized = torch.where(
        resized_validity,
        weighted_values / weights.clamp_min(1e-6),
        torch.zeros_like(weighted_values),
    )
    return resized[0], resized_validity[0]


def _read_stacked_tiff(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import rasterio  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("BENv2 CBIR evaluation requires rasterio") from exc
    with rasterio.open(path) as dataset:
        values = dataset.read().astype(np.float32, copy=False)
        validity = dataset.read_masks() > 0
        if dataset.nodata is not None and np.isfinite(dataset.nodata):
            validity &= values != dataset.nodata
    validity &= np.isfinite(values)
    return values, validity


class BENv2CBIRDataset(Dataset):
    """Load paired, normalized BENv2 S1/S2 images for embedding extraction."""

    def __init__(
        self,
        records: Sequence[BENv2Pair],
        normalizer: MMEarthBandNormalizer,
        input_image_size: int,
    ):
        if input_image_size <= 0:
            raise ValueError("input_image_size must be positive")
        self.records = list(records)
        self.normalizer = normalizer
        self.input_image_size = input_image_size

    def __len__(self) -> int:
        return len(self.records)

    def _load_s1(self, record: BENv2Pair) -> tuple[torch.Tensor, torch.Tensor]:
        values, validity = _read_stacked_tiff(record.s1_path)
        if values.shape[0] != len(S1_SOURCE_BANDS):
            raise ValueError(f"Expected two S1 bands in {record.s1_path}, got {values.shape[0]}")
        # The Kaggle stack is VH,VV; the model adapters use VV,VH.
        values = values[[1, 0]]
        validity = validity[[1, 0]]
        values = self.normalizer.normalize(values, record.s1_adapter, S1_MODEL_BANDS)
        values = np.where(validity, values, 0.0)
        return _resize_raster(
            torch.from_numpy(values.copy()),
            torch.from_numpy(validity.copy()),
            self.input_image_size,
        )

    def _load_s2(self, record: BENv2Pair) -> tuple[torch.Tensor, torch.Tensor]:
        values, validity = _read_stacked_tiff(record.s2_path)
        if values.shape[0] == len(S2_SOURCE_BANDS_10):
            bands = S2_SOURCE_BANDS_10
        elif values.shape[0] == len(S2_SOURCE_BANDS_12):
            bands = S2_SOURCE_BANDS_12
        else:
            raise ValueError(
                f"Expected 10 or 12 S2 bands in {record.s2_path}, got {values.shape[0]}"
            )
        validity &= values != 0
        values = self.normalizer.normalize(values, "sentinel2", bands)
        values = np.where(validity, values, 0.0)
        return _resize_raster(
            torch.from_numpy(values.copy()),
            torch.from_numpy(validity.copy()),
            self.input_image_size,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        s1, s1_validity = self._load_s1(record)
        s2, s2_validity = self._load_s2(record)
        return {
            "s1": s1,
            "s1_validity": s1_validity,
            "s1_adapter": record.s1_adapter,
            "s2": s2,
            "s2_validity": s2_validity,
            "label": torch.from_numpy(record.label.copy()),
            "sample_id": record.sample_id,
        }
