"""
Core type definitions for the QuaDMix framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import numpy.typing as npt

# Type aliases
QualityScore = float
DomainLabel = int  # Domain index 0..M-1


@dataclass
class MergedQualityConfig:
    """
    Configuration for merging multiple quality scores (Equation 1).

    For each domain m, we have merging parameters α_m = (α_{1,m}, ..., α_{N,m}).
    A global weight vector ã = (ã₁, ..., ãₙ) is shared across all domains.
    The final weight for criterion n in domain m is:
        w_{n,m} = ãₙ · b̃_{n,m}
    where b̃_m = (b̃_{1,m}, ..., b̃_{N,m}) are domain-specific normalized weights.
    """
    # Global criteria weights (ã₁, ..., ãₙ) — shared across domains
    global_weights: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    # Domain-specific weights (b̃_{1,m}, ..., b̃_{N,m}) — one per domain
    domain_weights: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))

    @property
    def num_criteria(self) -> int:
        return len(self.global_weights)

    def get_final_weights(self, domain_idx: int) -> npt.NDArray[np.float64]:
        """Get the final weight vector α_m for domain m."""
        start = domain_idx * self.num_criteria
        end = start + self.num_criteria
        return self.domain_weights[start:end]

    def __eq__(self, other):
        if not isinstance(other, MergedQualityConfig):
            return NotImplemented
        return np.array_equal(self.global_weights, other.global_weights) and \
               np.array_equal(self.domain_weights, other.domain_weights)

    def __hash__(self):
        return hash((self.global_weights.tobytes(), self.domain_weights.tobytes()))


@dataclass
class SamplingConfig:
    """
    Sampling parameters β_m for a single domain.

    These are the parameters from Equation (3):
        S(¯r) = (2 / (1 + e^{-λ(ω-¯r)}))^η + ε,  if ¯r <= ω
        S(¯r) = ε,                                if ¯r > ω

    Where:
        λ  — controls how fast sampling frequency decreases with quality rank
        ω  — quality percentile threshold (minimum quality to retain)
        η  — scaling parameter that adjusts sampling values
        ε  — base sampling rate for low-quality data (randomness)
    """
    lambda_: float  # λ — steepness of sigmoid decay (rescaled: λ̃ = 10³·λ)
    omega: float     # ω — quality threshold percentile (rescaled: ω̃ = 0.1·ω)
    eta: float       # η — scaling factor
    epsilon: float   # ε — base sampling rate for tail data (rescaled: ε̃ = ε/1000)

    def sampling_value(self, quality_rank: float) -> float:
        """
        Compute S(¯r) from Equation (3).

        Args:
            quality_rank: ¯r = merged quality percentile within domain [0, 1]
                          Lower values mean higher quality (0 = best).

        Returns:
            Expected sampling frequency (fractional value).
        """
        if quality_rank > self.omega:
            return self.epsilon

        exponent = -self.lambda_ * (self.omega - quality_rank)
        exponent = max(-100.0, min(100.0, exponent))
        sigmoid = 2.0 / (1.0 + np.exp(exponent))
        return float(sigmoid ** self.eta + self.epsilon)


@dataclass
class ParameterSet:
    """
    Full set of QuaDMix parameters θ for all domains.

    Structure: θ = [(α_1, β_1), ..., (α_M, β_M)]
    Total parameters: (N + 4) × M
        N = number of quality criteria
        M = number of domains
    """
    # Merging config (shared across pipeline)
    merge_config: MergedQualityConfig
    # Per-domain sampling configs
    sampling_configs: List[SamplingConfig]

    @property
    def num_domains(self) -> int:
        return len(self.sampling_configs)

    @property
    def num_criteria(self) -> int:
        return self.merge_config.num_criteria

    def flatten(self) -> npt.NDArray[np.float64]:
        """
        Flatten parameters into a 1D array for regression input.
        Order: [domain_weights, sampling_params...]
        Total: (N + 4) x M  (per Algorithm 1, no global_weights)
        """
        parts = [
            self.merge_config.domain_weights,
        ]
        for sc in self.sampling_configs:
            parts.append(np.array([sc.lambda_, sc.omega, sc.eta, sc.epsilon]))
        return np.concatenate(parts)

    @classmethod
    def from_flattened(
        cls,
        array: npt.NDArray[np.float64],
        num_domains: int,
        num_criteria: int,
    ) -> "ParameterSet":
        """Reconstruct ParameterSet from flattened array."""
        expected_len = (num_criteria + 4) * num_domains
        if len(array) != expected_len:
            raise ValueError(
                f"Flattened array length {len(array)} != expected "
                f"(num_criteria+4)*num_domains = {expected_len} "
                f"(N={num_criteria}, M={num_domains})"
            )
        offset = 0

        domain_weights = array[offset:offset + num_criteria * num_domains]
        offset += num_criteria * num_domains

        # global_weights are not in flattened array (intermediate in Algorithm 1)
        global_weights = np.ones(num_criteria, dtype=np.float64) / num_criteria

        merge_config = MergedQualityConfig(
            global_weights=global_weights,
            domain_weights=domain_weights,
        )

        sampling_configs = []
        for _ in range(num_domains):
            sc = SamplingConfig(
                lambda_=array[offset],
                omega=array[offset + 1],
                eta=array[offset + 2],
                epsilon=array[offset + 3],
            )
            sampling_configs.append(sc)
            offset += 4

        return cls(merge_config=merge_config, sampling_configs=sampling_configs)

    @classmethod
    def from_dict(
        cls,
        quality_weights: Dict[str, Dict[str, float]],
        sampling_params: Dict[str, Dict[str, float]],
    ) -> "ParameterSet":
        """Reconstruct ParameterSet from JSON-style dicts.

        Args:
            quality_weights: {domain_name: {criterion_name: weight}}
            sampling_params: {domain_name: {"lambda": ..., "omega": ..., "eta": ..., "epsilon": ...}}
        """
        if set(quality_weights.keys()) != set(sampling_params.keys()):
            raise ValueError(
                f"quality_weights and sampling_params must cover same domains: "
                f"quality has {set(quality_weights.keys())}, "
                f"sampling has {set(sampling_params.keys())}"
            )
        domain_names = list(quality_weights.keys())
        quality_names = list(list(quality_weights.values())[0].keys())
        M = len(domain_names)
        N = len(quality_names)

        domain_weights = np.zeros(M * N, dtype=np.float64)
        for m, dname in enumerate(domain_names):
            for n, qname in enumerate(quality_names):
                domain_weights[m * N + n] = quality_weights[dname][qname]

        global_weights = np.ones(N, dtype=np.float64) / N
        merge_config = MergedQualityConfig(
            global_weights=global_weights,
            domain_weights=domain_weights,
        )

        sampling_configs = []
        for dname in domain_names:
            sp = sampling_params[dname]
            sampling_configs.append(SamplingConfig(
                lambda_=sp["lambda"],
                omega=sp["omega"],
                eta=sp["eta"],
                epsilon=sp["epsilon"],
            ))

        return cls(merge_config=merge_config, sampling_configs=sampling_configs)

    @property
    def flat_dim(self) -> int:
        """Total dimension of flattened parameter vector: (N + 4) x M."""
        return (self.num_criteria + 4) * self.num_domains


@dataclass
class QuaDMixConfig:
    """
    Top-level configuration for a QuaDMix run.
    """
    num_domains: int                # M — number of domains
    num_quality_criteria: Optional[int] = None  # N — auto-derived from schema.quality_cols if None

    # Experiment configuration
    num_proxy_experiments: int = 3000
    num_search_points: int = 100000
    top_k_average: int = 10
    search_weight_mode: str = "r2_weighted"

    # Sampling bounds (paper defaults)
    lambda_min: float = 0.0
    lambda_max: float = 1.0
    lambda_scale: float = 1000.0

    omega_min: float = 0.0
    omega_max: float = 1.0
    omega_scale: float = 0.1

    eta_min: float = 0.0
    eta_max: float = 1.0
    eta_scale: float = 1.0

    epsilon_min: float = 0.0
    epsilon_max: float = 1.0
    epsilon_scale: float = 0.001

    # Proxy model config (to be passed to actual training)
    proxy_model_size: str = "1M"
    proxy_training_tokens: int = 1_000_000_000  # 1B tokens

    # Regression config
    regression_train_ratio: float = 0.8  # 80/20 split for stable val R²
    regression_cv_folds: int = 5         # K-fold CV for R² estimation (0 = single split)

    # Target dataset size (0 = no scaling)
    target_tokens: int = 0  # Desired total tokens in sampled dataset

    # Pipeline seed — controls all randomness (Alg.1 parameter sampling,
    # Eq.1-3 document selection, budget subsampling, LightGBM search).
    #   None  (default): each run is non-deterministic (OS entropy),
    #          so different executions naturally produce diverse experiments.
    #          Merging multiple runs increases LightGBM regression coverage.
    #   int   (e.g. 42): fully reproducible — same seed always gives the
    #          same parameter sets, sampled docs, and training subsets.
    #          Useful for debugging or controlled A/B comparisons.
    seed: Optional[int] = None

    def __post_init__(self):
        if self.num_quality_criteria is not None and self.num_quality_criteria <= 0:
            raise ValueError(f"num_quality_criteria must be positive, got {self.num_quality_criteria}")

    @property
    def num_criteria(self) -> int:
        """N — number of quality criteria. Raises if not set (must derive from schema)."""
        if self.num_quality_criteria is None:
            raise ValueError(
                "num_quality_criteria not set. Must be derived from schema.quality_cols "
                "or passed explicitly."
            )
        return self.num_quality_criteria

    def get_domain_param_dim(self) -> int:
        """Parameter dimension per domain: (N + 4)"""
        return self.num_criteria + 4

    def total_param_dim(self) -> int:
        """Total parameter dimension: (N + 4) × M"""
        return self.get_domain_param_dim() * self.num_domains


@dataclass
class ProxyResult:
    """
    Result of a single proxy experiment.
    """
    parameters: ParameterSet
    validation_loss: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    per_task_losses: Optional[Dict[str, float]] = None