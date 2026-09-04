"""Exact nearest-neighbor retrieval metrics for frozen embeddings."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 embeddings."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must have shape [samples, dimensions]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Cannot normalize a zero-length embedding")
    return values / norms


def shared_sensor_mean(
    sentinel1_embeddings: np.ndarray,
    sentinel2_embeddings: np.ndarray,
) -> np.ndarray:
    """Estimate one equally weighted mean from paired S1 and S2 embeddings."""
    s1 = np.asarray(sentinel1_embeddings, dtype=np.float32)
    s2 = np.asarray(sentinel2_embeddings, dtype=np.float32)
    if s1.ndim != 2 or s2.ndim != 2 or s1.shape[1] != s2.shape[1]:
        raise ValueError("S1 and S2 embeddings must be matrices with the same dimension")
    if len(s1) == 0 or len(s2) == 0:
        raise ValueError("Cannot estimate a mean from an empty embedding matrix")
    return 0.5 * (s1.mean(axis=0) + s2.mean(axis=0))


def transform_retrieval_embeddings(
    embeddings: np.ndarray,
    mean: np.ndarray | None = None,
) -> np.ndarray:
    """Optionally center embeddings, then apply the retrieval L2 normalization."""
    values = np.asarray(embeddings, dtype=np.float32)
    if mean is not None:
        mean = np.asarray(mean, dtype=np.float32)
        if mean.shape != (values.shape[1],):
            raise ValueError("mean must have one value per embedding dimension")
        values = values - mean
    return l2_normalize(values)


@torch.inference_mode()
def chunked_cosine_topk(
    query_embeddings: np.ndarray,
    archive_embeddings: np.ndarray,
    k: int,
    device: torch.device,
    query_chunk_size: int = 512,
    archive_chunk_size: int = 8192,
) -> np.ndarray:
    """Find exact cosine top-k neighbors without materializing all pair scores."""
    queries = np.asarray(query_embeddings, dtype=np.float32)
    archive = np.asarray(archive_embeddings, dtype=np.float32)
    if queries.ndim != 2 or archive.ndim != 2 or queries.shape[1] != archive.shape[1]:
        raise ValueError("Query and archive embeddings must share one feature dimension")
    if not 0 < k <= len(archive):
        raise ValueError("k must be positive and no larger than the archive")
    if query_chunk_size <= 0 or archive_chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive")

    output = np.empty((len(queries), k), dtype=np.int64)
    archive_cpu = torch.from_numpy(archive)

    for query_start in range(0, len(queries), query_chunk_size):
        query_stop = min(query_start + query_chunk_size, len(queries))
        query = torch.from_numpy(queries[query_start:query_stop]).to(device)
        best_scores = None
        best_indices = None

        for archive_start in range(0, len(archive), archive_chunk_size):
            archive_stop = min(archive_start + archive_chunk_size, len(archive))
            archive_chunk = archive_cpu[archive_start:archive_stop].to(device)
            similarities = query @ archive_chunk.T
            local_k = min(k, archive_stop - archive_start)
            scores, indices = similarities.topk(local_k, dim=1)
            indices = indices + archive_start

            if best_scores is not None:
                scores = torch.cat([best_scores, scores], dim=1)
                indices = torch.cat([best_indices, indices], dim=1)
            keep_k = min(k, scores.shape[1])
            best_scores, keep = scores.topk(keep_k, dim=1)
            best_indices = indices.gather(1, keep)

        output[query_start:query_stop] = best_indices.cpu().numpy()

    return output


def csmoe_multilabel_metrics(
    query_labels: np.ndarray,
    archive_labels: np.ndarray,
    neighbor_indices: np.ndarray,
) -> Dict[str, float]:
    """Compute the pair-averaged precision, recall, and F1 used by CSMoE."""
    queries = np.asarray(query_labels, dtype=bool)
    archive = np.asarray(archive_labels, dtype=bool)
    neighbors = np.asarray(neighbor_indices, dtype=np.int64)
    if queries.ndim != 2 or archive.ndim != 2 or queries.shape[1] != archive.shape[1]:
        raise ValueError("Query and archive labels must share one class dimension")
    if neighbors.ndim != 2 or neighbors.shape[0] != len(queries):
        raise ValueError("neighbor_indices must have shape [queries, k]")
    if neighbors.size == 0 or neighbors.min() < 0 or neighbors.max() >= len(archive):
        raise ValueError("neighbor_indices contains an invalid archive index")

    retrieved = archive[neighbors]
    overlap = np.logical_and(queries[:, None, :], retrieved).sum(axis=2)
    retrieved_positives = retrieved.sum(axis=2)
    query_positives = queries.sum(axis=1, keepdims=True)
    if np.any(retrieved_positives == 0) or np.any(query_positives == 0):
        raise ValueError("CSMoE retrieval metrics require at least one label per image")

    precision = float((overlap / retrieved_positives).mean())
    recall = float((overlap / query_positives).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_csmoe_retrieval(
    query_embeddings: np.ndarray,
    archive_embeddings: np.ndarray,
    query_labels: np.ndarray,
    archive_labels: np.ndarray,
    k: int,
    device: torch.device,
    query_chunk_size: int = 512,
    archive_chunk_size: int = 8192,
) -> Dict[str, float | int]:
    """Run exact top-k retrieval and return CSMoE-compatible metrics."""
    neighbors = chunked_cosine_topk(
        query_embeddings=query_embeddings,
        archive_embeddings=archive_embeddings,
        k=k,
        device=device,
        query_chunk_size=query_chunk_size,
        archive_chunk_size=archive_chunk_size,
    )
    metrics = csmoe_multilabel_metrics(query_labels, archive_labels, neighbors)
    return {
        **metrics,
        "k": int(k),
        "queries": int(len(query_embeddings)),
        "archive": int(len(archive_embeddings)),
    }
