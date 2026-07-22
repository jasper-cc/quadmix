"""
Quality merging — Equation (1) from the paper.

    ¯q = Σ_{n=1}^{N} σ(q_n) · α_{n,m}

Where:
    σ — normalization function aligning quality criterion scales
    α_m = (α_{1,m}, ..., α_{N,m}) — merging parameters for domain m
    q_n — raw quality score from criterion n

QuaDMix Input Contract:
    quality_matrix: ndarray of shape (num_docs, num_criteria)
    Convention: higher = better (e.g., higher probability, higher confidence).
    Users pass raw scores directly — no negation needed.
    Most real-world quality scorers output "higher = better" naturally.
"""

import numpy as np
import numpy.typing as npt

from quadmix.core.types import MergedQualityConfig
from quadmix.utils.normalization import get_normalizer


def compute_merged_quality_scores(
    quality_matrix: npt.NDArray[np.float64],
    domain_labels: npt.NDArray[np.int64],
    merge_config: MergedQualityConfig,
    normalizer: str = "rank",
) -> npt.NDArray[np.float64]:
    """
    Compute merged quality scores ¯q for all documents (Equation 1).

    Args:
        quality_matrix: Shape (num_docs, N) — raw quality scores.
                        q_{n,doc} = quality_matrix[doc, n]
                        HIGHER values = BETTER quality.
        domain_labels: Shape (num_docs,) — domain label for each doc.
        merge_config: Merging parameters α_m for each domain.
        normalizer: Name of normalization function σ to use.

    Returns:
        Array of merged quality scores ¯q for each document.
        Shape: (num_docs,) — higher = better quality.
    """
    num_docs, num_criteria = quality_matrix.shape
    normalize_fn = get_normalizer(normalizer)

    # Normalize each quality criterion independently
    normalized_quality = np.zeros_like(quality_matrix)
    for n in range(num_criteria):
        normalized_quality[:, n] = normalize_fn(quality_matrix[:, n])

    # Compute merged score per document
    merged_scores = np.zeros(num_docs, dtype=np.float64)

    # Get all unique domains
    unique_domains = np.unique(domain_labels)

    neg_count = int((domain_labels < 0).sum())
    if neg_count > 0:
        import logging
        logging.getLogger(__name__).warning(
            f"Skipped {neg_count} unlabeled documents (domain_label < 0) "
            f"during quality merging. These docs will receive merged_score=0."
        )

    max_idx = len(merge_config.domain_weights) // num_criteria - 1
    non_contiguous = unique_domains[(unique_domains >= 0) & (unique_domains > max_idx)]
    if len(non_contiguous) > 0:
        import warnings
        warnings.warn(
            f"Domain labels contain values beyond 0..{max_idx}: "
            f"{non_contiguous.tolist()}. These domains will get "
            f"merged_scores=0. Ensure domain labels are contiguous "
            f"0..M-1 or provide domain_names in schema."
        )

    for m in unique_domains:
        if m < 0:  # skip unlabeled docs
            continue
        mask = domain_labels == m
        num_masked = mask.sum()
        if num_masked == 0:
            continue

        # Get final weights α_m for this domain
        alpha_m = merge_config.get_final_weights(m)

        # Weighted sum: ¯q = Σ σ(q_n) · α_{n,m}
        merged_scores[mask] = normalized_quality[mask] @ alpha_m

    return merged_scores
