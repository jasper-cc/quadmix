"""Sampling helpers for large-scale datasets.

Provides:
  - sample_with_optimal_params: apply optimal QuaDMix params to select documents
  - save_sampled_dataset: save selected documents to parquet/jsonl
"""

from typing import Callable, List, Optional, Tuple
import os

import numpy as np
import numpy.typing as npt
import pandas as pd

from quadmix.core.types import ParameterSet
from quadmix.core.sampler import compute_sampling_values


def _select_documents_vectorized(
    sampling_values: npt.NDArray[np.float64],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Select documents based on sampling values (fully vectorized).

    Args:
        sampling_values: Fractional sampling expectations per document.
        rng: Random number generator.

    Returns:
        Tuple of (selected_indices, selection_weights).
        Each index may appear multiple times (if sampling_value > 1).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    int_part = np.floor(sampling_values).astype(np.int64)
    frac_part = sampling_values - int_part
    random_mask = rng.uniform(size=len(sampling_values)) < frac_part

    repeats = int_part + random_mask.astype(np.int64)
    doc_indices = np.arange(len(sampling_values), dtype=np.int64)
    selected = np.repeat(doc_indices, repeats)

    weights = 1.0 / np.maximum(sampling_values[selected], 1e-10)

    return selected, weights


def sample_with_optimal_params(
    quality_ranks: npt.NDArray[np.float64],
    domain_labels: npt.NDArray[np.int64],
    params: ParameterSet,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Apply optimal QuaDMix parameters to produce a sampled dataset.

    Args:
        quality_ranks: Per-document quality ranks [0, 1], 0 = best.
        domain_labels: Per-document domain labels.
        params: Optimal QuaDMix parameter set.
        rng: Random number generator.

    Returns:
        Tuple of (selected_indices, sampling_values, selection_weights).
    """
    sampling_values = compute_sampling_values(quality_ranks, domain_labels, params)
    selected_indices, selection_weights = _select_documents_vectorized(sampling_values, rng)
    return selected_indices, sampling_values, selection_weights


def save_sampled_dataset(
        get_text_fn: Callable[[npt.NDArray[np.int64]], List[str]],
        num_total_docs: int,
    selected_indices: npt.NDArray[np.int64],
    output_path: str,
    domain_labels: Optional[npt.NDArray[np.int64]] = None,
    quality_ranks: Optional[npt.NDArray[np.float64]] = None,
    sampling_values: Optional[npt.NDArray[np.float64]] = None,
    doc_id_fn: Optional[Callable[[int], str]] = None,
    format: str = "parquet",
    text_col: str = "text",
    domain_col: str = "domain",
    batch_size: int = 100000,
):
    """Save the sampled dataset with metadata (OOM-safe).

    Uses a callback (get_text_fn) to retrieve texts on-demand instead of
    requiring the full corpus in memory. Writes in batches to keep peak
    memory proportional to batch_size, not total dataset size.

    Args:
        get_text_fn: Callable accepting a numpy array of indices and returning
            a list of text strings. For sharded datasets, use
            metadata_manager.read_texts directly. For in-memory datasets,
            wrap with lambda: lambda idx: [texts[i] for i in idx].
        num_total_docs: Total number of documents in the original corpus.
        selected_indices: Indices of selected documents (may repeat).
        output_path: Where to save the sampled dataset.
        domain_labels: Original domain labels (for joining).
        quality_ranks: Original quality ranks (for metadata).
        sampling_values: Sampling values at selection time.
        doc_id_fn: Callable returning doc_id for a given index. If None,
            uses the index itself as doc_id.
        format: Output format ("parquet" or "jsonl").
        text_col: Column name for text.
        domain_col: Column name for domain.
        batch_size: Number of rows per write batch (controls peak memory).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n_selected = len(selected_indices)
    batches = []

    for start in range(0, n_selected, batch_size):
        end = min(start + batch_size, n_selected)
        batch_indices = selected_indices[start:end]

        batch_texts = get_text_fn(batch_indices)
        records = {text_col: batch_texts}

        if doc_id_fn is not None:
            records["doc_id"] = [doc_id_fn(i) for i in batch_indices]
        else:
            records["doc_id"] = batch_indices.tolist()

        if domain_labels is not None:
            records[domain_col] = domain_labels[batch_indices].tolist()

        if quality_ranks is not None:
            records["quality_rank"] = quality_ranks[batch_indices].tolist()

        if sampling_values is not None:
            records["sampling_weight"] = 1.0 / np.maximum(sampling_values[batch_indices], 1e-10)
            records["sampling_value"] = sampling_values[batch_indices].tolist()

        batches.append(pd.DataFrame(records))

    df = pd.concat(batches, ignore_index=True)

    if format == "parquet":
        df.to_parquet(output_path, index=False)
    elif format == "jsonl":
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"[Save] Sampled dataset saved to: {output_path}")
    print(f"[Save]   Original docs: {num_total_docs}")
    print(f"[Save]   Selected docs: {n_selected}")
    print(f"[Save]   Sampling ratio: {n_selected / max(1, num_total_docs):.4f}x")
