"""Sequential active search for QuaDMix proxy experiments.

This module deliberately sits beside the one-shot pipeline.  It reuses the
existing parameter sampler, proxy runner, and LightGBM optimizer without
changing their behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import os
import tempfile
import time

import numpy as np
import numpy.typing as npt

from quadmix.core.types import ParameterSet, ProxyResult, QuaDMixConfig
from quadmix.pipeline.optimizer import QuaDMixOptimizer
from quadmix.pipeline.param_sampler import ParameterSampler


FIXED_TRAIN_STEPS = 5000


@dataclass(frozen=True)
class ActiveSearchConfig:
    """Configuration for fixed-fidelity sequential active search."""

    initial_experiments: int = 64
    active_rounds: int = 4
    batch_size: int = 32
    candidate_pool_size: int = 100_000
    final_search_points: int = 100_000
    top_k: int = 10
    beta_start: float = 1.5
    beta_decay: float = 0.75
    seed: int = 42
    num_workers: int = 8
    train_steps: int = FIXED_TRAIN_STEPS

    def validate(self) -> None:
        positive = {
            "initial_experiments": self.initial_experiments,
            "batch_size": self.batch_size,
            "candidate_pool_size": self.candidate_pool_size,
            "final_search_points": self.final_search_points,
            "top_k": self.top_k,
            "num_workers": self.num_workers,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.active_rounds < 0:
            raise ValueError("active_rounds must be non-negative")
        if self.train_steps != FIXED_TRAIN_STEPS:
            raise ValueError(
                f"Active search is fixed at {FIXED_TRAIN_STEPS} steps per "
                f"experiment, got {self.train_steps}"
            )
        if self.beta_start < 0 or not 0 < self.beta_decay <= 1:
            raise ValueError("beta_start must be >= 0 and beta_decay in (0, 1]")


@dataclass(frozen=True)
class CandidateChoice:
    parameters: ParameterSet
    source: str
    predicted_mean: Optional[float] = None
    predicted_std: Optional[float] = None
    acquisition_score: Optional[float] = None


def _parameter_key(params: ParameterSet) -> Tuple[float, ...]:
    return tuple(np.round(params.flatten(), decimals=12).tolist())


def _quota(batch_size: int) -> Dict[str, int]:
    exploit = batch_size // 2
    uncertain = batch_size // 4
    diverse = batch_size // 8
    return {
        "exploit": exploit,
        "uncertain": uncertain,
        "diverse": diverse,
        "random": batch_size - exploit - uncertain - diverse,
    }


def select_active_batch(
    candidates: Sequence[ParameterSet],
    predicted_mean: npt.NDArray[np.float64],
    predicted_std: npt.NDArray[np.float64],
    batch_size: int,
    beta: float,
    rng: np.random.Generator,
    excluded_keys: Optional[set[Tuple[float, ...]]] = None,
) -> List[CandidateChoice]:
    """Select an exploit/explore/diversity/random active-learning batch."""

    if len(candidates) != len(predicted_mean) or len(candidates) != len(predicted_std):
        raise ValueError("Candidate and prediction lengths differ")
    excluded_keys = excluded_keys or set()

    unique_indices: List[int] = []
    seen = set(excluded_keys)
    for index, params in enumerate(candidates):
        key = _parameter_key(params)
        if key not in seen:
            seen.add(key)
            unique_indices.append(index)

    if len(unique_indices) < batch_size:
        raise ValueError(
            f"Only {len(unique_indices)} unseen candidates for batch_size={batch_size}"
        )

    valid = np.asarray(unique_indices, dtype=np.int64)
    means = np.asarray(predicted_mean, dtype=np.float64)
    stds = np.asarray(predicted_std, dtype=np.float64)
    lcb = means - beta * stds
    quotas = _quota(batch_size)
    selected: List[int] = []
    sources: Dict[int, str] = {}

    def add(indices: Iterable[int], source: str, limit: int) -> None:
        for index in indices:
            index = int(index)
            if index not in sources:
                selected.append(index)
                sources[index] = source
                if sum(value == source for value in sources.values()) >= limit:
                    return

    exploit_order = valid[np.argsort(means[valid])]
    add(exploit_order, "exploit", quotas["exploit"])

    uncertain_order = valid[np.argsort(lcb[valid])]
    add(uncertain_order, "uncertain", quotas["uncertain"])

    # Diversity is chosen among the best 20% LCB candidates.  All dimensions
    # are normalized first so lambda cannot dominate epsilon by scale alone.
    remaining = np.asarray([i for i in valid if i not in sources], dtype=np.int64)
    shortlist_n = max(quotas["diverse"] * 20, math.ceil(len(valid) * 0.20))
    shortlist = valid[np.argsort(lcb[valid])[:shortlist_n]]
    shortlist = np.asarray([i for i in shortlist if i not in sources], dtype=np.int64)
    if quotas["diverse"] and len(shortlist):
        matrix = np.vstack([params.flatten() for params in candidates])
        low = np.min(matrix, axis=0)
        span = np.maximum(np.max(matrix, axis=0) - low, 1e-12)
        normalized = (matrix - low) / span
        references = normalized[selected] if selected else normalized[shortlist[:1]]
        for _ in range(min(quotas["diverse"], len(shortlist))):
            distances = np.min(
                np.sum(
                    (normalized[shortlist, None, :] - references[None, :, :]) ** 2,
                    axis=2,
                ),
                axis=1,
            )
            chosen = int(shortlist[int(np.argmax(distances))])
            if chosen not in sources:
                selected.append(chosen)
                sources[chosen] = "diverse"
                references = np.vstack([references, normalized[chosen]])
            shortlist = shortlist[shortlist != chosen]
            if not len(shortlist):
                break

    remaining = np.asarray([i for i in valid if i not in sources], dtype=np.int64)
    if quotas["random"]:
        random_order = rng.permutation(remaining)
        add(random_order, "random", quotas["random"])

    # Fill any quota shortfall deterministically by acquisition score.
    if len(selected) < batch_size:
        fill = [i for i in uncertain_order if int(i) not in sources]
        for index in fill[: batch_size - len(selected)]:
            selected.append(int(index))
            sources[int(index)] = "fill"

    return [
        CandidateChoice(
            parameters=candidates[index],
            source=sources[index],
            predicted_mean=float(means[index]),
            predicted_std=float(stds[index]),
            acquisition_score=float(lcb[index]),
        )
        for index in selected[:batch_size]
    ]


class ActiveSearchController:
    """Run fixed-5000-step QuaDMix experiments in sequential rounds."""

    def __init__(
        self,
        quadmix_config: QuaDMixConfig,
        active_config: ActiveSearchConfig,
        proxy_runner: Any,
        output_dir: str,
        domain_names: Optional[List[str]] = None,
        quality_names: Optional[List[str]] = None,
    ):
        active_config.validate()
        runner_steps = int(getattr(proxy_runner, "tiny_steps", -1))
        if runner_steps != FIXED_TRAIN_STEPS:
            raise ValueError(
                f"Proxy runner must use exactly {FIXED_TRAIN_STEPS} steps, "
                f"got {runner_steps}"
            )
        self.quadmix_config = quadmix_config
        self.active_config = active_config
        self.proxy_runner = proxy_runner
        self.output_dir = Path(output_dir)
        self.domain_names = domain_names
        self.quality_names = quality_names
        self.history_path = self.output_dir / "active_history.json"
        self.state_path = self.output_dir / "active_state.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        self._validate_resume_config()
        history = self._load_history()
        completed_rounds = sorted({int(item["round_id"]) for item in history})
        expected_completed = list(range(len(completed_rounds)))
        if completed_rounds != expected_completed:
            raise RuntimeError(f"Non-contiguous completed rounds: {completed_rounds}")

        total_rounds = self.active_config.active_rounds + 1
        print("=" * 72)
        print("  QuaDMix fixed-fidelity active search")
        print(f"  Train steps per experiment: {FIXED_TRAIN_STEPS}")
        print(f"  Initial experiments: {self.active_config.initial_experiments}")
        print(f"  Active rounds: {self.active_config.active_rounds}")
        print(f"  Batch size: {self.active_config.batch_size}")
        print(f"  Resume at round: {len(completed_rounds)}")
        print("=" * 72)

        for round_id in range(len(completed_rounds), total_rounds):
            results = self._history_to_results(history)
            if round_id == 0:
                sampler = ParameterSampler(
                    self.quadmix_config, seed=self.active_config.seed
                )
                choices = [
                    CandidateChoice(parameters=params, source="initial")
                    for params in sampler.sample_batch(
                        self.active_config.initial_experiments
                    )
                ]
                beta = None
            else:
                beta = self.active_config.beta_start * (
                    self.active_config.beta_decay ** (round_id - 1)
                )
                choices = self._propose_active_batch(results, round_id, beta)

            new_records = self._run_round(round_id, choices, len(history), beta)
            history.extend(new_records)
            self._atomic_json(self.history_path, history)
            self._write_state(history, round_id + 1, total_rounds)

        final_results = self._history_to_results(history)
        optimizer = QuaDMixOptimizer(self.quadmix_config)
        optimizer.add_proxy_results(final_results)
        optimizer.train_regressor()
        optimal, _, predicted = optimizer.search_optimal(
            n_search_points=self.active_config.final_search_points,
            top_k=self.active_config.top_k,
        )
        serialized = self._serialize_params(optimal)
        serialized.update(
            {
                "active_search": {
                    "train_steps": FIXED_TRAIN_STEPS,
                    "num_experiments": len(history),
                    "num_rounds": total_rounds,
                    "best_predicted_loss": float(np.min(predicted)),
                }
            }
        )
        self._atomic_json(self.output_dir / "optimal_parameters.json", serialized)

        summary = {
            "status": "complete",
            "active_config": asdict(self.active_config),
            "num_experiments": len(history),
            "num_rounds": total_rounds,
            "best_observed_loss": float(
                min(item["validation_loss"] for item in history)
            ),
            "best_predicted_loss": float(np.min(predicted)),
            "optimal_parameters": str(self.output_dir / "optimal_parameters.json"),
        }
        self._atomic_json(self.output_dir / "active_search_summary.json", summary)
        self._write_state(history, total_rounds, total_rounds, status="complete")
        return summary

    def _propose_active_batch(
        self, results: List[ProxyResult], round_id: int, beta: float
    ) -> List[CandidateChoice]:
        optimizer = QuaDMixOptimizer(self.quadmix_config)
        optimizer.add_proxy_results(results)
        optimizer.train_regressor()

        sampler = ParameterSampler(
            self.quadmix_config, seed=self.active_config.seed + round_id * 10_003
        )
        candidates = sampler.sample_batch(self.active_config.candidate_pool_size)
        bootstrap_models = getattr(optimizer, "_bootstrap_models", [])
        if bootstrap_models:
            all_predictions = np.vstack(
                [model.predict(candidates) for model in bootstrap_models]
            )
            means = np.mean(all_predictions, axis=0)
            stds = np.std(all_predictions, axis=0)
        else:
            if optimizer.regressor is None:
                raise RuntimeError("LightGBM regressor was not trained")
            means = optimizer.regressor.predict(candidates)
            stds = np.zeros_like(means)

        excluded = {_parameter_key(result.parameters) for result in results}
        rng = np.random.default_rng(self.active_config.seed + round_id * 97)
        return select_active_batch(
            candidates=candidates,
            predicted_mean=np.asarray(means, dtype=np.float64),
            predicted_std=np.asarray(stds, dtype=np.float64),
            batch_size=self.active_config.batch_size,
            beta=beta,
            rng=rng,
            excluded_keys=excluded,
        )

    def _run_round(
        self,
        round_id: int,
        choices: List[CandidateChoice],
        global_offset: int,
        beta: Optional[float],
    ) -> List[Dict[str, Any]]:
        round_dir = self.output_dir / "rounds" / f"round_{round_id:02d}"
        if round_dir.exists() and not (round_dir / "round_complete.json").exists():
            raise RuntimeError(
                f"Incomplete round directory exists: {round_dir}. "
                "Move or remove it after inspecting its proxy outputs, then resume."
            )
        round_dir.mkdir(parents=True, exist_ok=True)
        self.proxy_runner.output_dir = str(round_dir / "proxy_experiments")

        selection = [
            {
                "local_experiment_id": index,
                "global_experiment_id": global_offset + index,
                "source": choice.source,
                "predicted_mean": choice.predicted_mean,
                "predicted_std": choice.predicted_std,
                "acquisition_score": choice.acquisition_score,
                "parameters_flattened": choice.parameters.flatten().tolist(),
            }
            for index, choice in enumerate(choices)
        ]
        self._atomic_json(round_dir / "selected_candidates.json", selection)

        params = [choice.parameters for choice in choices]
        started = time.time()
        print(f"\n[ActiveSearch] Round {round_id}: {len(params)} experiments")
        all_selected = self.proxy_runner.precompute_samples(params)
        self.proxy_runner.tokenize_all_needed(all_selected)
        results = self.proxy_runner.run_batch_parallel(
            params,
            all_selected,
            num_workers=self.active_config.num_workers,
            device_type=self.proxy_runner.device_type,
        )
        if len(results) != len(choices):
            raise RuntimeError(
                f"Round {round_id} returned {len(results)} results for "
                f"{len(choices)} choices"
            )

        records: List[Dict[str, Any]] = []
        for index, (choice, result) in enumerate(zip(choices, results)):
            if not np.isfinite(result.validation_loss):
                raise RuntimeError(
                    f"Round {round_id} experiment {index} has non-finite loss"
                )
            records.append(
                {
                    "round_id": round_id,
                    "local_experiment_id": index,
                    "global_experiment_id": global_offset + index,
                    "source": choice.source,
                    "predicted_mean": choice.predicted_mean,
                    "predicted_std": choice.predicted_std,
                    "acquisition_score": choice.acquisition_score,
                    "validation_loss": float(result.validation_loss),
                    "parameters_flattened": choice.parameters.flatten().tolist(),
                    "metadata": self._jsonable(result.metadata),
                    "per_task_losses": self._jsonable(result.per_task_losses),
                }
            )

        round_summary = {
            "round_id": round_id,
            "beta": beta,
            "num_experiments": len(records),
            "elapsed_seconds": time.time() - started,
            "mean_loss": float(np.mean([r["validation_loss"] for r in records])),
            "best_loss": float(np.min([r["validation_loss"] for r in records])),
        }
        self._atomic_json(round_dir / "round_records.json", records)
        self._atomic_json(round_dir / "round_complete.json", round_summary)
        return records

    def _load_history(self) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        if self.history_path.exists():
            with self.history_path.open("r", encoding="utf-8") as handle:
                history = json.load(handle)
        if not isinstance(history, list):
            raise ValueError(f"Invalid history file: {self.history_path}")

        # Recover the narrow crash window after a round was atomically saved
        # but before the top-level history file was replaced.
        known_rounds = {int(item["round_id"]) for item in history}
        round_id = 0
        recovered = False
        while True:
            round_dir = self.output_dir / "rounds" / f"round_{round_id:02d}"
            complete = round_dir / "round_complete.json"
            records = round_dir / "round_records.json"
            if not complete.exists():
                break
            if not records.exists():
                raise RuntimeError(
                    f"Completed round is missing its records file: {round_dir}"
                )
            if round_id not in known_rounds:
                with records.open("r", encoding="utf-8") as handle:
                    recovered_records = json.load(handle)
                history.extend(recovered_records)
                known_rounds.add(round_id)
                recovered = True
            round_id += 1
        if recovered:
            history.sort(key=lambda item: int(item["global_experiment_id"]))
            self._atomic_json(self.history_path, history)
        return history

    def _validate_resume_config(self) -> None:
        if not self.state_path.exists():
            return
        with self.state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        previous = state.get("active_config")
        current = asdict(self.active_config)
        if previous is not None and previous != current:
            raise RuntimeError(
                "Active-search configuration differs from the existing output "
                f"directory. Existing={previous}, requested={current}"
            )

    def _history_to_results(self, history: List[Dict[str, Any]]) -> List[ProxyResult]:
        return [
            ProxyResult(
                parameters=ParameterSet.from_flattened(
                    np.asarray(item["parameters_flattened"], dtype=np.float64),
                    self.quadmix_config.num_domains,
                    self.quadmix_config.num_quality_criteria,
                ),
                validation_loss=float(item["validation_loss"]),
                metadata=item.get("metadata") or {},
                per_task_losses=item.get("per_task_losses"),
            )
            for item in history
        ]

    def _serialize_params(self, params: ParameterSet) -> Dict[str, Any]:
        domains = self.domain_names or [
            f"domain_{index}" for index in range(params.num_domains)
        ]
        qualities = self.quality_names or [
            f"criterion_{index}" for index in range(params.num_criteria)
        ]
        weights = params.merge_config.domain_weights
        quality_weights = {}
        for domain_index, domain in enumerate(domains):
            start = domain_index * params.num_criteria
            quality_weights[domain] = {
                quality: round(float(weights[start + quality_index]), 6)
                for quality_index, quality in enumerate(qualities)
            }
        sampling_params = {
            domains[index]: {
                "lambda": round(float(config.lambda_), 4),
                "omega": round(float(config.omega), 6),
                "eta": round(float(config.eta), 6),
                "epsilon": round(float(config.epsilon), 6),
            }
            for index, config in enumerate(params.sampling_configs)
        }
        return {
            "quality_weights": quality_weights,
            "sampling_params": sampling_params,
        }

    def _write_state(
        self,
        history: List[Dict[str, Any]],
        next_round: int,
        total_rounds: int,
        status: str = "running",
    ) -> None:
        self._atomic_json(
            self.state_path,
            {
                "status": status,
                "next_round": next_round,
                "total_rounds": total_rounds,
                "completed_experiments": len(history),
                "active_config": asdict(self.active_config),
            },
        )

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): ActiveSearchController._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ActiveSearchController._jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
