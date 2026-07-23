#!/usr/bin/env python3
"""
Resume QuaDMix pipeline from Stage 5 (LightGBM regression).

Loads proxy experiment results from disk (meta.json) and runs:
  Stage 5: LightGBM regression
  Stage 6: Optimal parameter search
  Stage 7: Final sampling
  Stage 8: Save outputs + report

Usage:
  python scripts/runners/resume_from_stage5.py \
      --proxy-dir /path/to/output/proxy_experiments \
      --preprocessed-dir /path/to/preprocessed \
      --output /path/to/output
"""

import argparse
import json
import os
import sys
import time
try:
    import quadmix
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import numpy as np

from quadmix import QuaDMixConfig
from quadmix.core.types import ParameterSet, ProxyResult
from quadmix.data.metadata_manager import ShardMetadataManager
from quadmix.data.dataset_schema import DatasetSchema
from quadmix.pipeline.real_pipeline import QuaDMixPipeline


def _build_domain_dist_change(domain_labels, selected_indices, num_domains, domain_names):
    orig_dist = np.bincount(domain_labels[domain_labels >= 0], minlength=num_domains)
    sel_dist = np.bincount(
        domain_labels[selected_indices][domain_labels[selected_indices] >= 0],
        minlength=num_domains)
    change = {}
    for m in range(num_domains):
        if orig_dist[m] > 0:
            name = domain_names[m] if m < len(domain_names) else f"D{m}"
            change[name] = {
                "original": int(orig_dist[m]),
                "selected": int(sel_dist[m]),
                "ratio": round(float(sel_dist[m]) / orig_dist[m], 4),
            }
    return change


def _compute_dataset_size_prediction(optimal_params, mm, config):
    omega_values = [sc.omega for sc in optimal_params.sampling_configs]
    avg_omega = float(np.mean(omega_values))
    max_omega = float(np.max(omega_values))
    min_omega = float(np.min(omega_values))
    total_tokens_est = mm.get_total_tokens_estimate() if mm is not None else None
    if total_tokens_est is None:
        return None
    estimated_tokens = int(total_tokens_est * avg_omega)
    target_tokens = config.target_tokens
    note = None
    if target_tokens > 0:
        target_b = target_tokens / 1e9
        if estimated_tokens < target_tokens * 0.8:
            note = f"estimated {estimated_tokens/1e9:.2f}B < target {target_b:.1f}B"
        elif estimated_tokens > target_tokens * 1.2:
            discard_pct = (estimated_tokens - target_tokens) / estimated_tokens * 100
            note = f"estimated {estimated_tokens/1e9:.2f}B > target {target_b:.1f}B, discard ~{discard_pct:.1f}%"
    return {
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


def _compute_proxy_loss_stats(results):
    stats = {}
    train_losses = np.array([r.metadata["train_loss"] for r in results if "train_loss" in r.metadata])
    val_losses = np.array([r.validation_loss for r in results])
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
    return stats


def load_proxy_results(proxy_dir: str):
    results = []
    exp_dirs = sorted(
        d for d in os.listdir(proxy_dir)
        if d.startswith("exp_") and os.path.isdir(os.path.join(proxy_dir, d))
    )
    for exp_name in exp_dirs:
        meta_path = os.path.join(proxy_dir, exp_name, "meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        params = ParameterSet.from_dict(meta["quality_weights"], meta["sampling_params"])
        per_task_losses = meta.get("per_task_losses")
        results.append(ProxyResult(
            parameters=params,
            validation_loss=meta["val_loss"],
            per_task_losses=per_task_losses,
            metadata=meta,
        ))

    return results


def build_parser():
    p = argparse.ArgumentParser(description="Resume QuaDMix from Stage 5")
    p.add_argument("--proxy-dir", required=True,
                   help="Path to proxy_experiments/ directory")
    p.add_argument("--preprocessed-dir", required=True,
                   help="Path to preprocessed shards directory")
    p.add_argument("--output", "-o", default=None,
                   help="Output directory (default: <project>/result/reoptimize_<timestamp>)")
    p.add_argument("--num-search", type=int, default=5000)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--target-tokens", type=float, default=0.0,
                   help="Target tokens in billions (0 = no target)")
    p.add_argument("--search-mode", default="equal_weight",
                   choices=["r2_weighted", "equal_weight", "r2_sigma_weighted"],
                   help="Search weighting mode (default: equal_weight)")
    p.add_argument("--schema", default=None,
                   help="Path to dataset schema YAML (default: Essential-Web schema)")
    return p


def main():
    args = build_parser().parse_args()

    if args.output is None:
        _project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.output = os.path.join(
            _project_dir,
            "result",
            f"reoptimize_{time.strftime('%Y%m%d_%H%M%S')}",
        )

    print(f"[Resume] Loading proxy results from: {args.proxy_dir}")
    results = load_proxy_results(args.proxy_dir)
    print(f"[Resume] Loaded {len(results)} experiment results")

    losses = np.array([r.validation_loss for r in results])
    print(f"  Aggregate loss stats ({len(results)} experiments): "
          f"mean={losses.mean():.4f}, std={losses.std():.4f}, "
          f"min={losses.min():.4f}, max={losses.max():.4f}")

    per_task_loss_stats = None
    has_per_task = all(r.per_task_losses is not None for r in results)
    if has_per_task:
        tasks = sorted(results[0].per_task_losses.keys())
        per_task_loss_stats = {}
        for task in tasks:
            task_losses = np.array([r.per_task_losses[task] for r in results])
            per_task_loss_stats[task] = {
                "mean": float(np.mean(task_losses)),
                "std": float(np.std(task_losses)),
                "min": float(np.min(task_losses)),
                "max": float(np.max(task_losses)),
            }
        print(f"\n  Per-task loss stats ({len(tasks)} tasks):")
        print(f"    {'Task':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"    {'-'*70}")
        for task in sorted(tasks, key=lambda t: -per_task_loss_stats[t]["mean"]):
            s = per_task_loss_stats[task]
            print(f"    {task:<30} {s['mean']:>8.4f} {s['std']:>8.4f} "
                  f"{s['min']:>8.4f} {s['max']:>8.4f}")

    n_exp = len(results)
    n_search = args.num_search
    top_k = args.top_k

    schema = DatasetSchema.from_yaml(args.schema) if args.schema else DatasetSchema()

    mm = ShardMetadataManager(args.preprocessed_dir, schema=schema)
    print(f"[Resume] {mm.num_docs:,} docs across {mm.num_shards} shards")
    domain_names = mm.detected_domain_names
    quality_names = mm.detected_quality_names

    config = QuaDMixConfig(
        num_domains=mm.num_domains, num_quality_criteria=mm.num_quality_criteria,
        num_proxy_experiments=n_exp, num_search_points=n_search,
        top_k_average=top_k,
        target_tokens=int(args.target_tokens * 1e9) if args.target_tokens > 0 else 0,
        search_weight_mode=args.search_mode,
    )

    pipeline = QuaDMixPipeline(config)

    domain_labels = mm.domain_labels
    quality_scores = mm.quality_scores
    token_counts = mm.estimate_token_counts()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()
    stage_times = {}

    # ── Stage 5: LightGBM Regression ────────────────────
    _t = time.time()
    print(f"\n[Stage 5] Training LightGBM regressor...")
    from quadmix.pipeline.optimizer import QuaDMixOptimizer
    pipeline._optimizer = QuaDMixOptimizer(config)
    pipeline._optimizer.add_proxy_results(results)
    pipeline._optimizer.train_regressor()
    stage_times["stage5_lightgbm"] = time.time() - _t
    print(f"[Stage 5] LightGBM: {stage_times['stage5_lightgbm']:.1f}s")

    # ── Stage 6: Optimal Parameter Search ───────────────
    _t = time.time()
    print(f"\n[Stage 6] Searching optimal parameters ({n_search} points)...")
    optimal_params, candidates, predicted_losses = pipeline._optimizer.search_optimal(
        n_search_points=n_search, top_k=top_k,
    )
    stage_times["stage6_search"] = time.time() - _t
    print(f"[Stage 6] Search: {stage_times['stage6_search']:.1f}s")
    print(f"  Best predicted loss: {predicted_losses.min():.4f}")
    k = config.top_k_average
    top_indices = np.argsort(predicted_losses)[:k]
    top_k_avg_loss = float(predicted_losses[top_indices].mean())

    # ── Stage 7: Final Sampling ─────────────────────────
    _t = time.time()
    print(f"\n[Stage 7] Applying optimal parameters for final sampling...")
    from quadmix.core.quality_merger import compute_merged_quality_scores
    from quadmix.core.quality_rank import compute_quality_ranks
    from quadmix.sampling.batch_sampler import sample_with_optimal_params

    print(f"[Stage 7] Merging quality scores (Eq.1)...")
    merged = compute_merged_quality_scores(
        quality_scores, domain_labels, optimal_params.merge_config,
    )
    print(f"  Merged scores: [{merged.min():.4f}, {merged.max():.4f}]")

    print(f"[Stage 7] Computing quality ranks (Eq.2)...")
    final_ranks = compute_quality_ranks(merged, domain_labels, token_counts)
    print(f"  Quality ranks: [{final_ranks.min():.4f}, {final_ranks.max():.4f}]")
    selected_indices, sampling_values, _ = sample_with_optimal_params(
        final_ranks, domain_labels, optimal_params,
    )

    n_docs = len(domain_labels)
    print(f"  Original documents: {n_docs:,}")
    print(f"  Selected samples:   {len(selected_indices):,}")
    print(f"  Sampling ratio:     {len(selected_indices)/n_docs:.4f}x")

    orig_dist = np.bincount(domain_labels[domain_labels >= 0],
                             minlength=config.num_domains)
    sel_dist = np.bincount(
        domain_labels[selected_indices][domain_labels[selected_indices] >= 0],
        minlength=config.num_domains)
    print("\n  Domain distribution change:")
    for m in range(config.num_domains):
        if orig_dist[m] > 0:
            ratio = sel_dist[m] / orig_dist[m]
            name = domain_names[m] if m < len(domain_names) else f"D{m}"
            print(f"    [{m}] {name:>10s}: {orig_dist[m]:>7,} -> {sel_dist[m]:>7,}  ({ratio:.2f}x)")
    stage_times["stage7_final_sampling"] = time.time() - _t
    print(f"[Stage 7] Final sampling: {stage_times['stage7_final_sampling']:.1f}s")

    # ── Stage 8: Save Outputs ───────────────────────────
    _t = time.time()
    params_path = os.path.join(output_dir, "optimal_parameters.json")
    serialized = pipeline._serialize_params(optimal_params, mm.detected_domain_names, mm.detected_quality_names)
    with open(params_path, "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"\n[Stage 8] Optimal parameters saved to: {params_path}")

    elapsed = time.time() - t_start
    summary = {
        "config": {
            "num_domains": config.num_domains,
            "num_quality_criteria": config.num_quality_criteria,
            "num_proxy_experiments": n_exp,
            "num_search_points": n_search,
            "search_weight_mode": config.search_weight_mode,
        },
        "metrics": {
            "aggregate_train_r2": pipeline._optimizer.train_r2,
            "aggregate_val_r2": pipeline._optimizer.val_r2,
            "aggregate_val_mae": pipeline._optimizer.val_mae,
            "ensemble_val_r2": pipeline._optimizer.ensemble_val_r2,
            "ensemble_val_mae": pipeline._optimizer.ensemble_val_mae,
            "equal_weight_r2": pipeline._optimizer.equal_weight_r2,
            "equal_weight_mae": pipeline._optimizer.equal_weight_mae,
            "spearman_corr": pipeline._optimizer.spearman_corr,
            "top_k_recall": pipeline._optimizer.top_k_recall,
            "top_k_value": pipeline._optimizer.top_k_value,
            "search_lift": pipeline._optimizer.search_lift,
            "best_predicted_loss": float(predicted_losses.min()),
            "top_k_avg_loss": top_k_avg_loss,
        },
        "reliability": {
            "bootstrap": pipeline._optimizer.bootstrap_details,
            "sample_sufficient": pipeline._optimizer.sample_sufficient,
            "overfit_gap": pipeline._optimizer.overfit_gap,
            "n_features": pipeline._optimizer.n_features,
            "n_train_samples": getattr(pipeline._optimizer, "_n_train", None),
        },
        "sampling": {
            "num_original_docs": n_docs,
            "num_selected_docs": len(selected_indices),
            "sampling_ratio": len(selected_indices) / n_docs,
            "domain_distribution_change": _build_domain_dist_change(
                domain_labels, selected_indices, config.num_domains, domain_names),
        },
        "proxy_loss_stats": _compute_proxy_loss_stats(results),
        "per_task_loss_stats": per_task_loss_stats,
        "per_task_analysis": pipeline._optimizer.per_task_analysis,
        "dataset_size_prediction": _compute_dataset_size_prediction(optimal_params, mm, config),
        "elapsed_seconds": elapsed,
        "stage_times": {k: round(v, 1) for k, v in stage_times.items()},
        "input_file": args.preprocessed_dir,
        "resumed_from": "stage5",
    }
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)

    sampled_texts = mm.read_texts(selected_indices)
    import pandas as pd
    sel_domain = domain_labels[selected_indices]
    sel_rank = final_ranks[selected_indices]
    sel_sv = sampling_values[selected_indices]
    sel_weights = 1.0 / np.maximum(sel_sv, 1e-10)

    sampled_path = os.path.join(output_dir, "sampled_dataset.parquet")
    pd.DataFrame({
        schema.text_col: sampled_texts,
        "doc_id": selected_indices,
        schema.domain_col: sel_domain,
        "quality_rank": sel_rank,
        "sampling_weight": sel_weights,
        "sampling_value": sel_sv,
    }).to_parquet(sampled_path, index=False)
    print(f"[Stage 8] Sampled dataset saved: {sampled_path}")
    stage_times["stage8_save"] = time.time() - _t
    print(f"[Stage 8] Save outputs: {stage_times['stage8_save']:.1f}s")

    # ── Stage 9: Report ──
    _t = time.time()
    print(f"\n[Stage 9] Generating comparison report...")
    from quadmix.pipeline.report import generate_report, save_report
    report = generate_report(
        output_dir=output_dir,
        data_path=args.preprocessed_dir,
        optimal_params=optimal_params,
        optimal_selected_indices=selected_indices,
        domain_labels=domain_labels,
        token_counts=token_counts,
        num_domains=config.num_domains,
        num_criteria=config.num_criteria,
        config=summary["config"],
        metrics=summary["metrics"],
        elapsed=elapsed,
        use_sharded=True,
        reliability=summary.get("reliability"),
        proxy_loss_stats=summary.get("proxy_loss_stats"),
        per_task_analysis=summary.get("per_task_analysis"),
        dataset_size_prediction=summary.get("dataset_size_prediction"),
        stage_times={k: v for k, v in stage_times.items() if k != "stage9_report"},
        domain_names=domain_names,
        quality_names=quality_names,
        domain_col=schema.domain_col,
    )
    save_report(report, output_dir)
    stage_times["stage9_report"] = time.time() - _t
    print(f"[Stage 9] Report: {stage_times['stage9_report']:.1f}s")

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Resume Complete! ({total_elapsed:.1f}s)")
    print(f"  Search mode: {config.search_weight_mode}")
    print(f"  Aggregate Val R² = {pipeline._optimizer.val_r2:.4f} (diagnostic)")
    ens_r2 = pipeline._optimizer.ensemble_val_r2
    ens_mae = pipeline._optimizer.ensemble_val_mae
    eq_r2 = pipeline._optimizer.equal_weight_r2
    eq_mae = pipeline._optimizer.equal_weight_mae
    if ens_r2 is not None:
        print(f"  Overall   Val R² = {ens_r2:.4f}, MAE = {ens_mae:.4f}")
        print(f"    → R²(Σ wᵢ·z_predᵢ, Σ wᵢ·z_actualᵢ): search objective quality")
    if eq_r2 is not None:
        print(f"  Equal-Wt  Val R² = {eq_r2:.4f}, MAE = {eq_mae:.4f}")
        print(f"    → R²((1/K)Σ z_predᵢ, (1/K)Σ z_actualᵢ): downstream goal quality")
    sp = pipeline._optimizer.spearman_corr
    tk = pipeline._optimizer.top_k_recall
    tk_val = pipeline._optimizer.top_k_value
    sl = pipeline._optimizer.search_lift
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
