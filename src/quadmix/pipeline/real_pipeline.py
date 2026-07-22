"""
QuaDMix pipeline — unified entry point.

Two modes:
  Normal mode:    pipeline.run(data_path, ...)
    Loads raw data → extracts features (quality scoring + domain classification) → ...
  Precomputed mode: pipeline.run(data_path, precomputed=True, ...)
    Loads preprocessed parquet with existing domain labels + quality scores → skips feature extraction → ...

Pipeline stages:
  0. Load data (raw or precomputed)
  1. Feature extraction (only in normal mode)
  2. Parameter sampling (Alg.1)
  3. Proxy experiments (real model training with EssentialWebProxyRunner)
  4. LightGBM regression
  5. Optimal parameter search
  6. Final dataset sampling (Eq.1+Eq.2+Eq.3 with optimal params)
  7. Save outputs
  8. Generate comparison report
"""

from typing import Dict, List, Optional, Any
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from quadmix.core.types import (
    QuaDMixConfig,
    ParameterSet,
    MergedQualityConfig,
)
from quadmix.core.quality_merger import compute_merged_quality_scores
from quadmix.core.quality_rank import compute_quality_ranks
from quadmix.pipeline.param_sampler import ParameterSampler
from quadmix.pipeline.optimizer import QuaDMixOptimizer
from quadmix.data.base import BaseDataAdapter, UnifiedData
from quadmix.data.registry import get_adapter
from quadmix.sampling.batch_sampler import save_sampled_dataset, sample_with_optimal_params
from quadmix.pipeline.report import generate_report, save_report
from quadmix.data.dataset_schema import DatasetSchema
from quadmix.utils.json_utils import sanitize_for_json


@dataclass
class PipelineOutput:
    """Structured output of the pipeline."""
    optimal_params: ParameterSet
    serialized_params: Dict[str, Any]
    train_r2: float
    val_r2: float
    val_mae: float
    ensemble_val_r2: Optional[float]
    ensemble_val_mae: Optional[float]
    equal_weight_r2: Optional[float]
    equal_weight_mae: Optional[float]
    spearman_corr: Optional[float]
    top_k_recall: Optional[float]
    top_k_value: Optional[int]
    search_lift: Optional[float]
    best_predicted_loss: float
    selected_indices: npt.NDArray[np.int64]
    sampling_values: npt.NDArray[np.float64]
    domain_distribution_before: List[int]
    domain_distribution_after: List[int]
    num_experiments: int
    num_search_points: int
    num_original_docs: int
    num_selected_docs: int
    elapsed_seconds: float
    config: Dict[str, Any]


class QuaDMixPipeline:
    """
    Unified QuaDMix pipeline.

    Supports two modes:
      Normal:       run(data_path, scorers=..., classifier=...)
      Precomputed:  run(data_path, precomputed=True, quality_cols=..., domain_col=...)
    """

    def __init__(self, config: QuaDMixConfig):
        self.config = config
        self._quality_scorers: List = []
        self._domain_classifier = None
        self._optimizer: Optional[QuaDMixOptimizer] = None
        self._param_sampler = ParameterSampler(config, seed=config.seed)

        # Internal state
        self._precomputed_mode = False
        self._quality_scores: Optional[np.ndarray] = None
        self._domain_labels: Optional[np.ndarray] = None
        self._data: Optional[UnifiedData] = None
        self._texts: Optional[List[str]] = None
        self._dataset_size_prediction: Optional[Dict[str, Any]] = None

    # ── Public config ──────────────────────────────────────────

    def set_quality_scorers(self, scorers: List):
        """Set quality scorers for normal mode."""
        if len(scorers) != self.config.num_quality_criteria:
            raise ValueError(
                f"Expected {self.config.num_quality_criteria} scorers, got {len(scorers)}"
            )
        self._quality_scorers = scorers

    def set_domain_classifier(self, classifier):
        """Set domain classifier for normal mode."""
        if classifier.num_domains != self.config.num_domains:
            raise ValueError(
                f"Domain classifier reports {classifier.num_domains} domains, "
                f"but config expects {self.config.num_domains}"
            )
        self._domain_classifier = classifier

    # ── Data loading ───────────────────────────────────────────

    def load_data(self, data_path: str, **load_kwargs) -> UnifiedData:
        """Stage 0: Load raw data via auto-detected adapter (normal mode)."""
        print(f"\n[Stage 0] Loading data from: {data_path}")
        adapter = get_adapter(data_path)
        self._data = adapter.load(data_path, **load_kwargs)
        print(f"  Loaded {len(self._data)} documents")
        print(f"  Format: {self._data.metadata.get('format', 'unknown')}")
        if self._data.token_counts is None:
            from quadmix.data.parquet_adapter import ParquetDataAdapter
            dummy = ParquetDataAdapter()
            self._data.token_counts = np.array(
                [dummy._estimate_token_count(t) for t in self._data.texts],
                dtype=np.int64,
            )
        return self._data

    def load_precomputed(
        self,
        data_path: str,
        text_col: Optional[str] = None,
        domain_col: Optional[str] = None,
        quality_cols: Optional[List[str]] = None,
        doc_limit: Optional[int] = None,
    ):
        """
        Stage 0: Load preprocessed parquet with existing domain labels + quality scores.
        Single-file mode: loads all text upfront.
        """
        if text_col is None:
            text_col = self._schema.text_col
        if domain_col is None:
            domain_col = self._schema.domain_col
        if quality_cols is None:
            quality_cols = self._schema.quality_cols
        expected_n = self.config.num_quality_criteria
        if len(quality_cols) != expected_n:
            raise ValueError(
                f"Expected {expected_n} quality columns, got {len(quality_cols)}"
            )

        self._precomputed_mode = True
        print(f"\n[Stage 0] Loading precomputed data from: {data_path}")

        df = pd.read_parquet(data_path)
        if doc_limit and doc_limit < len(df):
            df = df.head(doc_limit).reset_index(drop=True)
            print(f"  Limited to {doc_limit} documents")

        self._texts = df[text_col].astype(str).tolist()
        self._domain_labels = df[domain_col].to_numpy(dtype=np.int64)
        self._quality_scores = df[quality_cols].to_numpy(dtype=np.float64)

        # Estimate token counts
        token_counts = np.array(
            [max(1, len(t) // 4) for t in self._texts], dtype=np.int64
        )

        print(f"  Documents: {len(self._texts):,}")
        print(f"  Domains:   {self.config.num_domains} "
              f"({len(np.unique(self._domain_labels))} present)")
        print(f"  Criteria:  {self.config.num_quality_criteria}")
        print(f"  Tokens:    {token_counts.sum():,} (estimated)")

        return self._texts, self._domain_labels, self._quality_scores, token_counts

    def load_precomputed_sharded(
        self,
        metadata_manager: object,
        doc_limit: Optional[int] = None,
    ):
        """
        Stage 0: Load preprocessed data via ShardMetadataManager (multi-shard).

        Reads only domain + quality columns from all shard parquets into memory.
        Text is NOT loaded upfront — use metadata_manager.read_texts() on demand.

        Args:
            metadata_manager: ShardMetadataManager instance.
            doc_limit: Limit documents for testing (truncates metadata).
        """
        self._precomputed_mode = True
        self._metadata_manager = metadata_manager
        print(f"\n[Stage 0] Loading precomputed data via "
              f"ShardMetadataManager ({metadata_manager.num_shards} shards, "
              f"{metadata_manager.num_docs:,} docs)")

        self._domain_labels = metadata_manager.domain_labels.copy()
        self._quality_scores = metadata_manager.quality_scores.copy()
        self._texts = None

        quality_directions = self._quality_directions if hasattr(self, '_quality_directions') else metadata_manager.quality_directions
        if quality_directions:
            for n, hb in enumerate(quality_directions):
                if not hb:
                    self._quality_scores[:, n] = -self._quality_scores[:, n]

        n = metadata_manager.num_docs
        if doc_limit and doc_limit < n:
            self._domain_labels = self._domain_labels[:doc_limit]
            self._quality_scores = self._quality_scores[:doc_limit]
            n = doc_limit
            print(f"  Limited to {doc_limit} documents")

        # Estimate token counts (lightweight, no text loaded)
        token_counts = metadata_manager.estimate_token_counts()
        if doc_limit and doc_limit < len(token_counts):
            token_counts = token_counts[:doc_limit]

        print(f"  Documents: {n:,}")
        print(f"  Domains:   {self.config.num_domains} "
              f"({len(np.unique(self._domain_labels))} present)")
        print(f"  Criteria:  {self.config.num_quality_criteria}")
        print(f"  [Sharded] Text will be loaded on-demand from {metadata_manager.num_shards} shards")

        return None, self._domain_labels, self._quality_scores, token_counts

    # ── Feature extraction ──────────────────────────────────────

    def extract_features(self, documents: List[str]) -> Dict[str, np.ndarray]:
        """
        Stage 1: Extract quality scores and domain labels (normal mode only).

        Uses real scorers and classifier. Not called in precomputed mode.
        """
        if not self._quality_scorers:
            raise RuntimeError("Quality scorers not set. Call set_quality_scorers().")
        if self._domain_classifier is None:
            raise RuntimeError("Domain classifier not set. Call set_domain_classifier().")

        print(f"\n[Stage 1] Classifying {len(documents)} documents "
              f"into {self.config.num_domains} domains...")
        t0 = time.time()
        self._domain_labels = self._domain_classifier.classify(documents)
        domain_names = getattr(
            self._domain_classifier, 'get_all_domain_names',
            lambda: [f"D{i}" for i in range(self.config.num_domains)]
        )()
        unique, counts = np.unique(self._domain_labels, return_counts=True)
        for d, c in zip(unique, counts):
            name = domain_names[d] if d < len(domain_names) else f"D{d}"
            print(f"    Domain {d} ({name}): {c} docs ({c/len(documents)*100:.1f}%)")
        print(f"  Domain classification: {time.time()-t0:.1f}s")

        quality_scores_list = []
        for scorer in self._quality_scorers:
            print(f"\n[Stage 1] Scoring with '{scorer.name}'...")
            t0 = time.time()
            scores = scorer.score(documents)
            print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}], "
                  f"mean={scores.mean():.4f}, std={scores.std():.4f}")
            print(f"  Scoring time: {time.time()-t0:.1f}s")
            quality_scores_list.append(scores)

        self._quality_scores = np.column_stack(quality_scores_list)
        return {"quality_scores": self._quality_scores, "domain_labels": self._domain_labels}

    # ── Quality ranks ──────────────────────────────────────────

    def compute_quality_ranks(
        self,
        quality_scores: np.ndarray,
        domain_labels: np.ndarray,
        merge_config: Optional[ParameterSet] = None,
        token_counts: Optional[np.ndarray] = None,
        stage_idx: int = 7,
    ) -> np.ndarray:
        """Compute merged quality scores and ranks (Eq.1 & Eq.2)."""
        if merge_config is None:
            N = self.config.num_quality_criteria
            M = self.config.num_domains
            global_weights = np.ones(N, dtype=np.float64) / N
            domain_weights = np.tile(global_weights, M)
            merge_config_plain = MergedQualityConfig(
                global_weights=global_weights, domain_weights=domain_weights,
            )
        else:
            merge_config_plain = merge_config.merge_config

        print(f"[Stage {stage_idx}] Merging quality scores (Eq.1)...")
        merged = compute_merged_quality_scores(
            quality_scores, domain_labels, merge_config_plain,
        )
        print(f"  Merged scores: [{merged.min():.4f}, "
              f"{merged.max():.4f}]")

        print(f"[Stage {stage_idx}] Computing quality ranks (Eq.2)...")
        ranks = compute_quality_ranks(
            merged, domain_labels, token_counts,
        )
        print(f"  Quality ranks: [{ranks.min():.4f}, "
              f"{ranks.max():.4f}]")

        return ranks

    # ── Main entry ─────────────────────────────────────────────

    def run(
        self,
        data_path: str,
        output_dir: str = "./quadmix_output",
        num_experiments: Optional[int] = None,
        num_search: Optional[int] = None,
        proxy_runner: Optional[object] = None,
        output_format: str = "parquet",
        precomputed: bool = False,
        text_col: Optional[str] = None,
        domain_col: Optional[str] = None,
        quality_cols: Optional[List[str]] = None,
        doc_limit: Optional[int] = None,
        domain_names: Optional[List[str]] = None,
        quality_names: Optional[List[str]] = None,
        quality_directions: Optional[List[bool]] = None,
        parallel_workers: int = 1,
        val_set: Optional[str] = None,
        schema: Optional[DatasetSchema] = None,
        **load_kwargs,
    ) -> PipelineOutput:
        """Run the complete QuaDMix pipeline."""
        os.makedirs(output_dir, exist_ok=True)
        t_start = time.time()

        print("=" * 70)
        print("  QuaDMix Pipeline")
        print(f"  Paper: arXiv:2504.16511v2 (ByteDance, 2025)")
        print(f"  Output: {output_dir}")
        print("=" * 70)

        stage_times: Dict[str, float] = {}

        if schema is None:
            schema = DatasetSchema()
        self._schema = schema
        if text_col is None:
            text_col = schema.text_col
        if domain_col is None:
            domain_col = schema.domain_col
        if quality_cols is None:
            quality_cols = schema.quality_cols

        self._quality_directions = quality_directions

        texts, domain_labels, quality_scores, token_counts, text_source, mm = \
            self._stage0_load(data_path, precomputed, text_col, domain_col,
                              quality_cols, doc_limit, stage_times, load_kwargs)

        quality_scores, domain_labels = \
            self._stage1_features(texts, precomputed, quality_scores, domain_labels, stage_times)

        n_exp, param_sets = self._stage3_param_sampling(num_experiments, stage_times)

        results, proxy_loss_stats = self._stage4_proxy_experiments(
            proxy_runner, n_exp, param_sets, parallel_workers, stage_times)

        self._stage5_lightgbm(results, stage_times)

        n_search, optimal_params, predicted_losses, top_k_avg_loss = \
            self._stage6_search(num_search, mm, token_counts, stage_times)

        final_ranks, selected_indices, sampling_values, orig_dist, sel_dist = \
            self._stage7_final_sampling(
                quality_scores, domain_labels, optimal_params, token_counts,
                domain_names, stage_times)

        domain_dist_change = {}
        for m in range(self.config.num_domains):
            if orig_dist[m] > 0:
                name = (domain_names[m] if domain_names and m < len(domain_names)
                        else f"D{m}")
                domain_dist_change[name] = {
                    "original": int(orig_dist[m]),
                    "selected": int(sel_dist[m]),
                    "ratio": round(float(sel_dist[m]) / orig_dist[m], 4),
                }

        serialized, summary, n_docs_save = self._stage8_save(
            output_dir, output_format, data_path, optimal_params, domain_names,
            quality_names, selected_indices, sampling_values, final_ranks,
            domain_labels, token_counts, texts, text_source, n_exp, n_search,
            predicted_losses, top_k_avg_loss, proxy_loss_stats, proxy_runner,
            val_set, stage_times, t_start, domain_dist_change)

        self._stage9_report(
            output_dir, data_path, optimal_params, selected_indices,
            domain_labels, token_counts, summary, t_start, text_source, stage_times,
            domain_names=domain_names, quality_names=quality_names)

        elapsed = time.time() - t_start
        self._print_timing_summary(stage_times, elapsed, output_dir)

        print(f"\n{'=' * 70}")
        print(f"  Pipeline Complete! ({elapsed:.1f}s)")
        print(f"  Aggregate Val R² = {self._optimizer.val_r2:.4f} (diagnostic)")
        ens_r2 = self._optimizer.ensemble_val_r2
        ens_mae = self._optimizer.ensemble_val_mae
        eq_r2 = self._optimizer.equal_weight_r2
        eq_mae = self._optimizer.equal_weight_mae
        if ens_r2 is not None:
            print(f"  Overall   Val R² = {ens_r2:.4f}, MAE = {ens_mae:.4f}")
            print(f"    → R²(Σ wᵢ·z_predᵢ, Σ wᵢ·z_actualᵢ): search objective quality")
        if eq_r2 is not None:
            print(f"  Equal-Wt  Val R² = {eq_r2:.4f}, MAE = {eq_mae:.4f}")
            print(f"    → R²((1/K)Σ z_predᵢ, (1/K)Σ z_actualᵢ): downstream goal quality")
        sp = self._optimizer.spearman_corr
        tk = self._optimizer.top_k_recall
        tk_val = self._optimizer.top_k_value
        sl = self._optimizer.search_lift
        if sp is not None:
            print(f"  Spearman Rank Corr = {sp:.4f} (ranking ability)")
        if tk is not None:
            print(f"  Top-{tk_val} Recall = {tk:.4f} ({int(tk*tk_val)}/{tk_val} hits)")
        if sl is not None:
            print(f"  Search Lift = {sl:.4f} σ (search vs random)")
        print(f"  Output: {output_dir}/")
        print(f"    ├── optimal_parameters.json")
        print(f"    ├── pipeline_summary.json")
        print(f"    ├── sampled_dataset.parquet")
        print(f"    ├── quadmix_report.md")
        print(f"    ├── fig1_domain_distribution.png")
        print(f"    └── fig2_quality_weights.png")
        print("=" * 70)

        return PipelineOutput(
            optimal_params=optimal_params,
            serialized_params=serialized,
            train_r2=self._optimizer.train_r2,
            val_r2=self._optimizer.val_r2,
            val_mae=self._optimizer.val_mae,
            ensemble_val_r2=self._optimizer.ensemble_val_r2,
            ensemble_val_mae=self._optimizer.ensemble_val_mae,
            equal_weight_r2=self._optimizer.equal_weight_r2,
            equal_weight_mae=self._optimizer.equal_weight_mae,
            spearman_corr=self._optimizer.spearman_corr,
            top_k_recall=self._optimizer.top_k_recall,
            top_k_value=self._optimizer.top_k_value,
            search_lift=self._optimizer.search_lift,
            best_predicted_loss=float(predicted_losses.min()),
            selected_indices=selected_indices,
            sampling_values=sampling_values,
            domain_distribution_before=orig_dist.tolist(),
            domain_distribution_after=sel_dist.tolist(),
            num_experiments=n_exp,
            num_search_points=n_search,
            num_original_docs=n_docs_save,
            num_selected_docs=len(selected_indices),
            elapsed_seconds=elapsed,
            config={
                "num_domains": self.config.num_domains,
                "num_quality_criteria": self.config.num_quality_criteria,
            },
        )

    # ── Stage methods ──────────────────────────────────────────

    def _stage0_load(self, data_path, precomputed, text_col, domain_col,
                     quality_cols, doc_limit, stage_times, load_kwargs):
        _t = time.time()
        text_source = "memory"
        mm = None
        if precomputed:
            mm = load_kwargs.pop("metadata_manager", None)
            if mm is not None:
                texts, domain_labels, quality_scores, token_counts = \
                    self.load_precomputed_sharded(mm, doc_limit=doc_limit)
                text_source = "sharded"
            else:
                texts, domain_labels, quality_scores, token_counts = \
                    self.load_precomputed(
                        data_path, text_col=text_col, domain_col=domain_col,
                        quality_cols=quality_cols, doc_limit=doc_limit,
                    )
        else:
            data = self.load_data(data_path, **load_kwargs)
            texts = data.texts
            token_counts = data.token_counts
            domain_labels = None
            quality_scores = None
        stage_times["stage0_load_data"] = time.time() - _t
        print(f"[Stage 0] Load data: {stage_times['stage0_load_data']:.1f}s")
        return texts, domain_labels, quality_scores, token_counts, text_source, mm

    def _stage1_features(self, texts, precomputed, quality_scores, domain_labels, stage_times):
        _t = time.time()
        if not precomputed:
            features = self.extract_features(texts)
            quality_scores = features["quality_scores"]
            domain_labels = features["domain_labels"]
        stage_times["stage1_features"] = time.time() - _t
        return quality_scores, domain_labels

    def _stage3_param_sampling(self, num_experiments, stage_times):
        _t = time.time()
        n_exp = num_experiments or self.config.num_proxy_experiments
        print(f"\n[Stage 3] Sampling {n_exp} parameter configurations (Alg.1)...")
        param_sets = self._param_sampler.sample_batch(n_exp)
        stage_times["stage3_param_sampling"] = time.time() - _t
        print(f"[Stage 3] Parameter sampling: {stage_times['stage3_param_sampling']:.1f}s")
        return n_exp, param_sets

    def _stage4_proxy_experiments(self, proxy_runner, n_exp, param_sets,
                                  parallel_workers, stage_times):
        if proxy_runner is None:
            raise ValueError("proxy_runner is required. Pass an EssentialWebProxyRunner instance.")
        print(f"\n[Stage 4] Running {n_exp} proxy experiments...")

        _t_stage4 = time.time()
        if parallel_workers > 1 and hasattr(proxy_runner, 'precompute_samples'):
            print(f"[Stage 4] Using {parallel_workers} parallel workers (dynamic task queue)")
            print(f"[Stage 4] Step 1: Pre-sampling (Eq.1-3, pure numpy, CPU only)")
            _t = time.time()
            all_selected = proxy_runner.precompute_samples(param_sets)
            stage_times["stage4a_precompute"] = time.time() - _t
            print(f"[Stage 4] Pre-sampling: {stage_times['stage4a_precompute']:.1f}s")

            print(f"[Stage 4] Step 1.5: Pre-tokenize all docs (parallel, one-shot)")
            _t = time.time()
            proxy_runner.tokenize_all_needed(all_selected)
            stage_times["stage4b_tokenize"] = time.time() - _t
            print(f"[Stage 4] Tokenize: {stage_times['stage4b_tokenize']:.1f}s")

            print(f"[Stage 4] Step 2: Dynamic parallel training (NPU workers, cache hits only)")
            _t = time.time()
            results = proxy_runner.run_batch_parallel(
                param_sets, all_selected,
                num_workers=parallel_workers,
                device_type=proxy_runner.device_type,
            )
            stage_times["stage4c_training"] = time.time() - _t
            print(f"[Stage 4] Training: {stage_times['stage4c_training']:.1f}s")
        elif hasattr(proxy_runner, 'precompute_samples'):
            print(f"[Stage 4] CPU mode: precompute → tokenize union → sequential run")
            _t = time.time()
            all_selected = proxy_runner.precompute_samples(param_sets)
            stage_times["stage4a_precompute"] = time.time() - _t
            print(f"[Stage 4] Pre-sampling: {stage_times['stage4a_precompute']:.1f}s")

            _t = time.time()
            proxy_runner.tokenize_all_needed(all_selected)
            stage_times["stage4b_tokenize"] = time.time() - _t
            print(f"[Stage 4] Tokenize: {stage_times['stage4b_tokenize']:.1f}s")

            _t = time.time()
            results = []
            all_selected_train = getattr(proxy_runner, '_all_selected_train', all_selected)
            for i, (params, sel, sel_train) in enumerate(
                zip(param_sets, all_selected, all_selected_train)
            ):
                r = proxy_runner.run_experiment(
                    params, experiment_id=i, selected_idx=sel_train,
                    sampled_doc_count=len(sel),
                )
                results.append(r)
            stage_times["stage4c_training"] = time.time() - _t
            print(f"[Stage 4] Training: {stage_times['stage4c_training']:.1f}s")
        else:
            _t = time.time()
            results = proxy_runner.run_batch(param_sets)
            stage_times["stage4c_training"] = time.time() - _t
        stage_times["stage4_total"] = time.time() - _t_stage4

        losses = np.array([r.validation_loss for r in results])
        print(f"  Aggregate loss stats ({len(results)} experiments): "
              f"mean={losses.mean():.4f}, std={losses.std():.4f}, "
              f"min={losses.min():.4f}, max={losses.max():.4f}")
        proxy_loss_stats = self._compute_proxy_loss_stats(results)
        return results, proxy_loss_stats

    def _stage5_lightgbm(self, results, stage_times):
        _t = time.time()
        print(f"\n[Stage 5] Training LightGBM regressor...")
        self._optimizer = QuaDMixOptimizer(self.config)
        self._optimizer.add_proxy_results(results)
        self._optimizer.train_regressor()
        stage_times["stage5_lightgbm"] = time.time() - _t
        print(f"[Stage 5] LightGBM: {stage_times['stage5_lightgbm']:.1f}s")

    def _stage6_search(self, num_search, mm, token_counts, stage_times):
        _t = time.time()
        n_search = num_search or self.config.num_search_points
        print(f"\n[Stage 6] Searching optimal parameters ({n_search} points)...")
        optimal_params, candidates, predicted_losses = self._optimizer.search_optimal(
            n_search_points=n_search, top_k=self.config.top_k_average,
        )
        stage_times["stage6_search"] = time.time() - _t
        print(f"[Stage 6] Search: {stage_times['stage6_search']:.1f}s")
        print(f"  Best predicted loss: {predicted_losses.min():.4f}")

        k = self.config.top_k_average
        top_indices = np.argsort(predicted_losses)[:k]
        top_k_avg_loss = float(predicted_losses[top_indices].mean())

        self._predict_dataset_size(optimal_params, mm, token_counts)
        return n_search, optimal_params, predicted_losses, top_k_avg_loss

    def _stage7_final_sampling(self, quality_scores, domain_labels, optimal_params,
                               token_counts, domain_names, stage_times):
        _t = time.time()
        print(f"\n[Stage 7] Applying optimal parameters for final sampling...")
        final_ranks = self.compute_quality_ranks(
            quality_scores, domain_labels, optimal_params, token_counts,
        )
        selected_indices, sampling_values, _ = sample_with_optimal_params(
            final_ranks, domain_labels, optimal_params,
        )

        selected_indices = self._apply_target_tokens(selected_indices, token_counts)

        n_docs = len(domain_labels)
        print(f"  Original documents: {n_docs:,}")
        print(f"  Selected samples:   {len(selected_indices):,}")
        print(f"  Sampling ratio:     {len(selected_indices)/n_docs:.4f}x")

        orig_dist = np.bincount(domain_labels[domain_labels >= 0],
                                 minlength=self.config.num_domains)
        sel_dist = np.bincount(
            domain_labels[selected_indices][domain_labels[selected_indices] >= 0],
            minlength=self.config.num_domains)
        print("\n  Domain distribution change:")
        for m in range(self.config.num_domains):
            if orig_dist[m] > 0:
                ratio = sel_dist[m] / orig_dist[m]
                name = (domain_names[m] if domain_names and m < len(domain_names)
                        else f"D{m}")
                print(f"    [{m}] {name:>10s}: {orig_dist[m]:>7,} → {sel_dist[m]:>7,}  ({ratio:.2f}x)")
        stage_times["stage7_final_sampling"] = time.time() - _t
        print(f"[Stage 7] Final sampling: {stage_times['stage7_final_sampling']:.1f}s")
        return final_ranks, selected_indices, sampling_values, orig_dist, sel_dist

    def _stage8_save(self, output_dir, output_format, data_path, optimal_params,
                     domain_names, quality_names, selected_indices, sampling_values,
                     final_ranks, domain_labels, token_counts, texts, text_source,
                     n_exp, n_search, predicted_losses, top_k_avg_loss,
                     proxy_loss_stats, proxy_runner, val_set, stage_times, t_start,
                     domain_dist_change=None):
        _t = time.time()
        params_path = os.path.join(output_dir, "optimal_parameters.json")
        serialized = self._serialize_params(optimal_params, domain_names, quality_names)
        with open(params_path, "w") as f:
            json.dump(sanitize_for_json(serialized), f, indent=2)
        print(f"\n[Stage 8] Optimal parameters saved to: {params_path}")

        elapsed = time.time() - t_start
        n_docs_save = len(domain_labels)
        normalizer = getattr(proxy_runner, "_normalizer_name", "unknown")
        summary = {
            "config": {
                "num_domains": self.config.num_domains,
                "num_quality_criteria": self.config.num_quality_criteria,
                "num_proxy_experiments": n_exp,
                "num_search_points": n_search,
                "normalizer": normalizer,
                "val_set": val_set,
            },
            "metrics": {
                "aggregate_train_r2": self._optimizer.train_r2,
                "aggregate_val_r2": self._optimizer.val_r2,
                "aggregate_val_mae": self._optimizer.val_mae,
                "ensemble_val_r2": self._optimizer.ensemble_val_r2,
                "ensemble_val_mae": self._optimizer.ensemble_val_mae,
                "equal_weight_r2": self._optimizer.equal_weight_r2,
                "equal_weight_mae": self._optimizer.equal_weight_mae,
                "spearman_corr": self._optimizer.spearman_corr,
                "top_k_recall": self._optimizer.top_k_recall,
                "top_k_value": self._optimizer.top_k_value,
                "search_lift": self._optimizer.search_lift,
                "best_predicted_loss": float(predicted_losses.min()),
                "top_k_avg_loss": top_k_avg_loss,
            },
            "proxy_loss_stats": proxy_loss_stats,
            "reliability": {
                "bootstrap": self._optimizer.bootstrap_details,
                "sample_sufficient": self._optimizer.sample_sufficient,
                "overfit_gap": self._optimizer.overfit_gap,
                "n_features": self._optimizer.n_features,
                "n_train_samples": getattr(self._optimizer, "_n_train", None),
            },
            "sampling": {
                "num_original_docs": n_docs_save,
                "num_selected_docs": len(selected_indices),
                "sampling_ratio": len(selected_indices) / n_docs_save,
                "domain_distribution_change": domain_dist_change,
            },
            "dataset_size_prediction": getattr(self, "_dataset_size_prediction", None),
            "per_task_analysis": self._optimizer.per_task_analysis,
            "elapsed_seconds": elapsed,
            "stage_times": {k: round(v, 1) for k, v in stage_times.items()},
            "input_file": data_path,
        }
        summary_path = os.path.join(output_dir, "pipeline_summary.json")
        with open(summary_path, "w") as f:
            json.dump(sanitize_for_json(summary), f, indent=2)

        if text_source == "sharded":
            print(f"[Stage 8] Reading {len(selected_indices):,} sampled texts from shards...")
            sampled_texts = self._metadata_manager.read_texts(selected_indices)
            sel_domain = domain_labels[selected_indices]
            sel_rank = final_ranks[selected_indices]
            sel_sv = sampling_values[selected_indices]
            sel_weights = 1.0 / np.maximum(sel_sv, 1e-10)

            sampled_path = os.path.join(output_dir, "sampled_dataset.parquet")
            os.makedirs(os.path.dirname(sampled_path) or ".", exist_ok=True)
            schema = self._schema
            print(f"[Stage 8] Writing sampled dataset to parquet ({len(selected_indices):,} rows)...")
            pd.DataFrame({
                schema.text_col: sampled_texts,
                "doc_id": selected_indices,
                schema.domain_col: sel_domain,
                "quality_rank": sel_rank,
                "sampling_weight": sel_weights,
                "sampling_value": sel_sv,
            }).to_parquet(sampled_path, index=False)
            print(f"[Stage 8] Sampled dataset saved (sharded): {sampled_path}")
        else:
            sampled_path = os.path.join(output_dir, "sampled_dataset.parquet")
            schema = self._schema
            save_sampled_dataset(
                original_texts=texts,
                selected_indices=selected_indices,
                output_path=sampled_path,
                domain_labels=domain_labels,
                quality_ranks=final_ranks,
                sampling_values=sampling_values,
                doc_ids=np.arange(len(texts)),
                format=output_format,
                text_col=schema.text_col,
                domain_col=schema.domain_col,
            )
        stage_times["stage8_save"] = time.time() - _t
        print(f"[Stage 8] Save outputs: {stage_times['stage8_save']:.1f}s")
        return serialized, summary, n_docs_save

    def _stage9_report(self, output_dir, data_path, optimal_params, selected_indices,
                       domain_labels, token_counts, summary, t_start, text_source, stage_times,
                       domain_names=None, quality_names=None):
        _t = time.time()
        elapsed = time.time() - t_start
        print(f"\n[Stage 9] Generating comparison report...")
        report = generate_report(
            output_dir=output_dir,
            data_path=data_path,
            optimal_params=optimal_params,
            optimal_selected_indices=selected_indices,
            domain_labels=domain_labels,
            token_counts=token_counts,
            num_domains=self.config.num_domains,
            num_criteria=self.config.num_quality_criteria,
            config=summary["config"],
            metrics=summary["metrics"],
            elapsed=elapsed,
            use_sharded=(text_source == "sharded"),
            reliability=summary.get("reliability"),
            proxy_loss_stats=summary.get("proxy_loss_stats"),
            per_task_analysis=summary.get("per_task_analysis"),
            dataset_size_prediction=summary.get("dataset_size_prediction"),
            stage_times={k: v for k, v in stage_times.items() if k != "stage9_report"},
            domain_names=domain_names,
            quality_names=quality_names,
            domain_col=self._schema.domain_col,
        )
        save_report(report, output_dir)
        stage_times["stage9_report"] = time.time() - _t
        print(f"[Stage 9] Report: {stage_times['stage9_report']:.1f}s")

    def _compute_proxy_loss_stats(self, results) -> Dict:
        stats: Dict[str, Any] = {}
        train_losses = np.array([r.metadata["train_loss"] for r in results if "train_loss" in r.metadata])
        val_losses = np.array([r.metadata["val_loss"] for r in results if "val_loss" in r.metadata])
        if len(train_losses) > 0:
            stats["train_loss"] = {
                "mean": float(train_losses.mean()), "std": float(train_losses.std()),
                "min": float(train_losses.min()), "max": float(train_losses.max()),
            }
        if len(val_losses) > 0:
            stats["val_loss"] = {
                "mean": float(val_losses.mean()), "std": float(val_losses.std()),
                "min": float(val_losses.min()), "max": float(val_losses.max()),
            }
        has_per_task = all(r.per_task_losses is not None for r in results)
        if has_per_task:
            tasks = sorted(results[0].per_task_losses.keys())
            per_task_stats = {}
            for task in tasks:
                task_losses = np.array([r.per_task_losses[task] for r in results])
                per_task_stats[task] = {
                    "mean": float(np.mean(task_losses)),
                    "std": float(np.std(task_losses)),
                    "min": float(np.min(task_losses)),
                    "max": float(np.max(task_losses)),
                }
            stats["per_task_loss"] = per_task_stats
        return stats

    def _predict_dataset_size(self, optimal_params, mm, token_counts):
        omega_values = [sc.omega for sc in optimal_params.sampling_configs]
        avg_omega = float(np.mean(omega_values))
        max_omega = float(np.max(omega_values))
        min_omega = float(np.min(omega_values))

        total_tokens_est = None
        if mm is not None:
            total_tokens_est = mm.get_total_tokens_estimate()
        elif token_counts is not None:
            total_tokens_est = int(np.sum(token_counts))

        if total_tokens_est is None:
            self._dataset_size_prediction = None
            return

        estimated_tokens = int(total_tokens_est * avg_omega)
        print(f"\n  ── θ* 数据量预测 ────────────────────────")
        print(f"    数据集总大小:     {total_tokens_est/1e9:.1f}B tokens")
        print(f"    ω 范围:          [{min_omega:.3f}, {max_omega:.3f}] (平均 {avg_omega:.3f})")
        print(f"    预计数据量:       ~{estimated_tokens/1e9:.2f}B tokens")

        target_tokens = self.config.target_tokens
        note = None
        if target_tokens > 0:
            target_b = target_tokens / 1e9
            if estimated_tokens < target_tokens * 0.8:
                note = f"预计 {estimated_tokens/1e9:.2f}B < target {target_b:.1f}B"
                print(f"    [提示] {note}")
                print(f"    论文: 'More tokens not always good' (30B > 90B > 180B)")
                print(f"    θ* 产生更少数据但 loss 更优，代码将继续执行")
            elif estimated_tokens > target_tokens * 1.2:
                discard_pct = (estimated_tokens - target_tokens) / estimated_tokens * 100
                note = f"预计 {estimated_tokens/1e9:.2f}B > target {target_b:.1f}B, 将随机丢弃约 {discard_pct:.1f}%"
                print(f"    [提示] {note}")

        self._dataset_size_prediction = {
            "total_tokens_est": total_tokens_est,
            "total_tokens_est_B": round(total_tokens_est / 1e9, 1),
            "omega_min": round(min_omega, 6),
            "omega_max": round(max_omega, 6),
            "omega_avg": round(avg_omega, 6),
            "estimated_tokens": estimated_tokens,
            "estimated_tokens_B": round(estimated_tokens / 1e9, 2),
            "target_tokens": target_tokens if target_tokens > 0 else None,
            "note": note,
        }

    def _apply_target_tokens(self, selected_indices, token_counts):
        if self.config.target_tokens <= 0 or token_counts is None:
            return selected_indices

        actual_tokens = float(np.sum(token_counts[selected_indices]))
        print(f"\n  Target token adjustment:")
        print(f"    θ* produces:       {actual_tokens:,.0f} tokens ({actual_tokens/1e9:.2f}B)")
        print(f"    Target:            {self.config.target_tokens:,.0f} tokens ({self.config.target_tokens/1e9:.1f}B)")

        if actual_tokens > self.config.target_tokens:
            keep_prob = self.config.target_tokens / actual_tokens
            rng = np.random.default_rng(42)
            keep_mask = rng.random(len(selected_indices)) < keep_prob
            selected_indices = selected_indices[keep_mask]
            final_tokens = float(np.sum(token_counts[selected_indices]))
            print(f"    Action:            Uniform discard (keep_prob={keep_prob:.4f})")
            print(f"    Final:             {final_tokens:,.0f} tokens ({final_tokens/1e9:.2f}B)")
        elif actual_tokens < self.config.target_tokens * 0.95:
            print(f"    Action:            [WARN] θ* produces less than target")
            print(f"    [建议] 调整 ω 参数放宽质量阈值，或降低 target_tokens")
        else:
            print(f"    Action:            Accept θ* result (within tolerance)")
        return selected_indices

    def _print_timing_summary(self, stage_times, elapsed, output_dir):
        print(f"\n{'─' * 50}")
        print(f"  STAGE TIMING SUMMARY")
        print(f"{'─' * 50}")
        for name, secs in sorted(stage_times.items(), key=lambda x: -x[1]):
            pct = secs / max(elapsed, 1) * 100
            bar = '█' * int(pct / 2)
            print(f"  {name:30s} {secs:7.1f}s ({pct:4.1f}%) {bar}")
        print(f"{'─' * 50}")

        try:
            from quadmix.utils.perf_timer import PerfTimer
            if PerfTimer._enabled:
                perf_text = PerfTimer.report()
                print(perf_text)
                perf_report_path = os.path.join(output_dir, "perf_report.txt")
                with open(perf_report_path, "w") as f:
                    f.write("STAGE TIMING SUMMARY\n")
                    f.write("=" * 50 + "\n")
                    for name, secs in sorted(stage_times.items(), key=lambda x: -x[1]):
                        pct = secs / max(elapsed, 1) * 100
                        f.write(f"  {name:30s} {secs:7.1f}s ({pct:4.1f}%)\n")
                    f.write("\n")
                    f.write(perf_text)
                print(f"[PerfTimer] Report saved to: {perf_report_path}")
        except ImportError:
            pass

    # ── Internal helpers ───────────────────────────────────────

    def _serialize_params(
        self, params: ParameterSet,
        domain_names: Optional[List[str]] = None,
        quality_names: Optional[List[str]] = None,
    ) -> Dict:
        """Serialize ParameterSet to JSON-friendly dict."""
        M = params.num_domains
        N = params.num_criteria
        d_names = domain_names or [f"domain_{m}" for m in range(M)]
        q_names = quality_names or [f"criterion_{n}" for n in range(N)]
        dw = params.merge_config.domain_weights

        quality_weights = {}
        for m in range(M):
            start = m * N
            quality_weights[d_names[m]] = {
                q_names[n]: round(float(dw[start + n]), 6) for n in range(N)
            }

        sampling_params = {}
        for m, sc in enumerate(params.sampling_configs):
            sampling_params[d_names[m]] = {
                "lambda": round(sc.lambda_, 4),
                "omega": round(sc.omega, 6),
                "eta": round(sc.eta, 6),
                "epsilon": round(sc.epsilon, 6),
            }

        return {
            "quality_weights": quality_weights,
            "sampling_params": sampling_params,
        }