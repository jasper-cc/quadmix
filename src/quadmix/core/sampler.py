"""
Quality sampling function — Equation (3) from the paper.

    S(¯r) = (2 / (1 + e^{-λ(ω-¯r)}))^η + ε,  if ¯r <= ω
    S(¯r) = ε,                                if ¯r > ω

Where:
    ¯r — merged quality rank in [0, 1]; 0 = best quality
    λ  — steepness of sigmoid decay (higher = sharper cutoff)
    ω  — quality threshold; only documents with ¯r <= ω get quality boost
    η  — scaling exponent on sigmoid output
    ε  — base sampling rate for low-quality tail (ensures some coverage)

QuaDMix Input Contract:
    quality_ranks: ndarray of shape (num_docs,) in [0, 1]
    Convention: 0 = best quality, 1 = worst quality.
    This is derived from merged_scores via compute_quality_ranks().
    Users pass raw scores (higher = better) to the merging step;
    the rank computation handles direction internally.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional

from quadmix.core.types import ParameterSet


def compute_sampling_values(
    quality_ranks: npt.NDArray[np.float64],
    domain_labels: npt.NDArray[np.int64],
    params: ParameterSet,
) -> npt.NDArray[np.float64]:
    """
    Compute sampling values S(¯r) for all documents (Equation 3).

    Args:
        quality_ranks: Per-document quality ranks ¯r in [0, 1].
                       Shape: (num_docs,) — 0 = best.
        domain_labels: Per-document domain labels.
                       Shape: (num_docs,).
        params: Full parameter set with per-domain sampling configs.

    Returns:
        Sampling frequency per document.
        Shape: (num_docs,) — fractional sampling expectation.
    """
    num_docs = len(quality_ranks)
    sampling_values = np.zeros(num_docs, dtype=np.float64)

    unique_domains = np.unique(domain_labels)

    max_expected = len(params.sampling_configs) - 1
    non_contiguous = unique_domains[(unique_domains >= 0) & (unique_domains > max_expected)]
    if len(non_contiguous) > 0:
        import warnings
        warnings.warn(
            f"Domain labels contain values beyond 0..{max_expected}: "
            f"{non_contiguous.tolist()}. These domains will be skipped "
            f"in sampling (sampling_value=0). Ensure domain labels are "
            f"contiguous 0..M-1 or provide domain_names in schema."
        )

    for m in unique_domains:
        mask = domain_labels == m
        indices = np.where(mask)[0]

        if len(indices) == 0 or m >= len(params.sampling_configs):
            continue

        sc = params.sampling_configs[m]
        r = quality_ranks[indices]

        # Equation (3): sigmoid-based sampling
        # For ¯r <= ω: S = (2/(1+e^{-λ(ω-¯r)}))^η + ε
        # For ¯r > ω:  S = ε

        # Compute sigmoid portion only where ¯r <= ω
        within_threshold = r <= sc.omega

        # Initialize with epsilon for all
        vals = np.full(len(r), sc.epsilon, dtype=np.float64)

        if within_threshold.any():
            exponent = -sc.lambda_ * (sc.omega - r[within_threshold])
            sigmoid = 2.0 / (1.0 + np.exp(np.clip(exponent, -100, 100)))
            vals[within_threshold] = sigmoid ** sc.eta + sc.epsilon

        sampling_values[indices] = vals

    return sampling_values
