#!/usr/bin/env python3
"""
Re-evaluate saved proxy experiments on a new validation set.

Loads trained 1M proxy model weights from a previous pipeline run,
switches to a different validation set, re-evaluates all experiments,
then re-runs LightGBM fitting, optimal search, final sampling, and
report generation.

Usage:
  python scripts/runners/reval_with_new_valset.py \
      --result-dir result/quadmix_20260609_120000 \
      --val-set core \
      --preprocessed-dir /path/to/preprocessed \
      --output result/reval_core_20260610_150000

  python scripts/runners/reval_with_new_valset.py \
      --result-dir result/quadmix_20260609_120000 \
      --val-path /path/to/custom_val.pt \
      --preprocessed-dir /path/to/preprocessed
"""

import argparse
import json
import os
import shutil
import ssl
import sys
import time
import urllib.request
try:
    import quadmix
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np

from quadmix import QuaDMixConfig
from quadmix.core.types import ParameterSet, ProxyResult
from quadmix.data.metadata_manager import ShardMetadataManager
from quadmix.data.dataset_schema import DatasetSchema
from quadmix.pipeline.real_pipeline import QuaDMixPipeline
from quadmix.constants import (
    PROJECT_DIR, DEFAULT_TEMP_DIR,
    DEFAULT_VAL_DIR, HF_ENDPOINT, HF_RESOLVE,
    HF_OPENHERMES_DATASET, HF_OPENHERMES_FILENAME,
    HF_CORE_DATASET, HF_CORE_FILENAME,
    HF_CORE_BMK_V2_DATASET, HF_CORE_BMK_V2_FILENAME,
    HF_CORE_BMK_V3_DATASET, HF_CORE_BMK_V3_FILENAME,
    HF_CORE_BMK_V4_DATASET, HF_CORE_BMK_V4_FILENAME,
    HF_CORE_BMK_V42_DATASET, HF_CORE_BMK_V42_FILENAME,
    HF_CORE_BMK_V43_DATASET, HF_CORE_BMK_V43_FILENAME,
    HF_CORE_BMK_V5_DATASET, HF_CORE_BMK_V5_FILENAME,
    HF_CORE_BMK_V6_DATASET, HF_CORE_BMK_V6_FILENAME,
    HF_CAP_V1_DATASET, HF_CAP_V1_FILENAME,
    HF_STEM_V1_DATASET, HF_STEM_V1_FILENAME,
    DEFAULT_EVAL_BUNDLE,
)

QUADMIX_DIR = PROJECT_DIR
QUADMIX_TEMP_DIR = DEFAULT_TEMP_DIR


def _hf_remote_size(repo_id: str, filename: str) -> int:
    url = f"{HF_ENDPOINT}/datasets/{repo_id}/resolve/main/{filename}"
    try:
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except Exception:
        pass
    return 0


def _download_hf_file(repo_id: str, filename: str, local_path: str) -> bool:
    url = HF_RESOLVE.format(repo=repo_id, file=filename)
    print(f"[Setup] Downloading from:\n  {url}")
    try:
        urllib.request.urlretrieve(url, local_path)
        size_mb = os.path.getsize(local_path) / 1024**2
        print(f"[Setup] Downloaded: {local_path} ({size_mb:.0f} MB)")
        return True
    except Exception as e:
        print(f"[Setup] Download failed: {e}")
        return False


def _check_and_download(local_path: str, repo_id: str, filename: str) -> str:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    remote_size = _hf_remote_size(repo_id, filename)
    if os.path.exists(local_path):
        if remote_size > 0 and os.path.getsize(local_path) == remote_size:
            print(f"[Setup] Validation set OK: {local_path}")
            return local_path
        print(f"[Setup] Local file size mismatch, re-downloading...")
    if _download_hf_file(repo_id, filename, local_path):
        return local_path
    if os.path.exists(local_path):
        print(f"[Setup] Download failed, using existing local file")
        return local_path
    raise FileNotFoundError(f"Cannot obtain validation file: {local_path}")


def resolve_val_path(val_set: str, val_path: str) -> str:
    if val_path:
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation file not found: {val_path}")
        return val_path
    if val_set == "core":
        local = os.path.join(DEFAULT_VAL_DIR, HF_CORE_FILENAME)
        return _check_and_download(local, HF_CORE_DATASET, HF_CORE_FILENAME)
    if val_set == "core_bmk_v2":
        local = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V2_FILENAME)
        return _check_and_download(local, HF_CORE_BMK_V2_DATASET, HF_CORE_BMK_V2_FILENAME)
    if val_set == "core_bmk_v3":
        local = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V3_FILENAME)
        return _check_and_download(local, HF_CORE_BMK_V3_DATASET, HF_CORE_BMK_V3_FILENAME)
    if val_set == "core_bmk_v4":
        local = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V4_FILENAME)
        return _check_and_download(local, HF_CORE_BMK_V4_DATASET, HF_CORE_BMK_V4_FILENAME)
    if val_set == "core_bmk_v4.2":
        local = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V42_FILENAME)
        return _check_and_download(local, HF_CORE_BMK_V42_DATASET, HF_CORE_BMK_V42_FILENAME)
    if val_set == "core_bmk_v4.3":
        local = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V43_FILENAME)
        return _check_and_download(local, HF_CORE_BMK_V43_DATASET, HF_CORE_BMK_V43_FILENAME)
    if val_set == "core_bmk_v5":
        local = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V5_FILENAME)
        return _check_and_download(local, HF_CORE_BMK_V5_DATASET, HF_CORE_BMK_V5_FILENAME)
    if val_set == "core_bmk_v6":
        local = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V6_FILENAME)
        return _check_and_download(local, HF_CORE_BMK_V6_DATASET, HF_CORE_BMK_V6_FILENAME)
    if val_set == "cap_v1":
        local = os.path.join(DEFAULT_VAL_DIR, HF_CAP_V1_FILENAME)
        return _check_and_download(local, HF_CAP_V1_DATASET, HF_CAP_V1_FILENAME)
    if val_set == "stem_v1":
        local = os.path.join(DEFAULT_VAL_DIR, HF_STEM_V1_FILENAME)
        return _check_and_download(local, HF_STEM_V1_DATASET, HF_STEM_V1_FILENAME)
    local = os.path.join(DEFAULT_VAL_DIR, HF_OPENHERMES_FILENAME)
    return _check_and_download(local, HF_OPENHERMES_DATASET, HF_OPENHERMES_FILENAME)


def reconstruct_params_from_meta(meta: dict) -> ParameterSet:
    return ParameterSet.from_dict(meta["quality_weights"], meta["sampling_params"])


def build_parser():
    p = argparse.ArgumentParser(
        description="Re-evaluate saved proxy experiments on a new validation set",
    )
    p.add_argument("--result-dir", required=True,
                   help="Path to original pipeline result directory "
                        "(e.g. result/quadmix_20260609_120000)")
    p.add_argument("--preprocessed-dir", required=True,
                   help="Path to preprocessed shards directory")
    p.add_argument("--val-set", type=str, default="core",
                   choices=["openhermes", "core", "core_bmk_v2", "core_bmk_v3", "core_bmk_v4", "core_bmk_v4.2", "core_bmk_v4.3", "core_bmk_v5", "core_bmk_v6", "cap_v1", "stem_v1"],
                   help="New validation set to evaluate on (default: core)")
    p.add_argument("--val-path", type=str, default=None,
                   help="Path to custom validation .pt file (overrides --val-set)")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Output directory (default: <project>/result/revalidate_<valset>_<timestamp>)")
    p.add_argument("--device-type", type=str, default="cpu",
                   choices=["cpu", "cuda", "npu"],
                   help="Device for re-evaluation (default: cpu)")
    p.add_argument("--num-gpus", type=int, default=None,
                   help="Number of GPUs to use for parallel reval (default: all available)")
    p.add_argument("--num-search", type=int, default=100000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--target-tokens", type=float, default=0.0,
                   help="Target tokens in billions (0 = no target)")
    p.add_argument("--block-size", type=int, default=2048,
                   help="Block size for validation (must match training)")
    p.add_argument("--model-variant", type=str, default="tinyllama_1M",
                   help="Proxy model variant (must match original training)")
    p.add_argument("--search-mode", default="equal_weight",
                   choices=["r2_weighted", "equal_weight", "r2_sigma_weighted"],
                   help="Search weighting mode (default: equal_weight)")
    p.add_argument("--schema", default=None,
                   help="Path to dataset schema YAML (default: Essential-Web schema)")
    return p


def main():
    args = build_parser().parse_args()

    result_dir = args.result_dir
    proxy_dir = os.path.join(result_dir, "proxy_experiments")
    if not os.path.isdir(proxy_dir):
        print(f"[Error] proxy_experiments not found: {proxy_dir}")
        return 1

    val_path = resolve_val_path(args.val_set, args.val_path)
    val_set_name = args.val_set if not args.val_path else os.path.basename(args.val_path)

    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(
            QUADMIX_DIR,
            "result",
            f"revalidate_{val_set_name}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  QuaDMix Re-evaluation Pipeline")
    print(f"  Source:  {result_dir}")
    print(f"  Val set: {val_set_name} ({val_path})")
    print(f"  Output:  {output_dir}")
    print("=" * 70)

    t_start = time.time()
    stage_times = {}

    # ── Stage 0: Load metadata ─────────────────────────────
    _t = time.time()
    schema = DatasetSchema.from_yaml(args.schema) if args.schema else DatasetSchema()
    print(f"\n[Stage 0] Loading metadata from: {args.preprocessed_dir}")
    mm = ShardMetadataManager(args.preprocessed_dir, schema=schema)
    domain_names = mm.detected_domain_names
    quality_names = mm.detected_quality_names
    print(f"[Stage 0] {mm.num_docs:,} docs across {mm.num_shards} shards")
    domain_labels = mm.domain_labels
    quality_scores = mm.quality_scores
    token_counts = mm.estimate_token_counts()
    stage_times["stage0_load"] = time.time() - _t

    # ── Stage 1: Scan experiments ──────────────────────────
    _t = time.time()
    exp_dirs = sorted(
        d for d in os.listdir(proxy_dir)
        if d.startswith("exp_") and os.path.isdir(os.path.join(proxy_dir, d))
    )
    experiments = []
    skipped = 0
    for exp_name in exp_dirs:
        exp_path = os.path.join(proxy_dir, exp_name)
        meta_path = os.path.join(exp_path, "meta.json")
        model_path = os.path.join(exp_path, "model.pt")
        if not os.path.exists(meta_path):
            skipped += 1
            continue
        if not os.path.exists(model_path):
            skipped += 1
            continue
        experiments.append((exp_name, exp_path, meta_path, model_path))
    stage_times["stage1_scan"] = time.time() - _t
    print(f"\n[Stage 1] Found {len(experiments)} experiments with model.pt "
          f"({skipped} skipped)")

    if not experiments:
        print("[Error] No experiments with model.pt found.")
        print("  Ensure the original run saved model weights (model.pt in each exp dir).")
        return 1

    output_proxy_dir = os.path.join(output_dir, "proxy_experiments")
    os.makedirs(output_proxy_dir, exist_ok=True)

    done_experiments = set()
    for d in os.listdir(output_proxy_dir):
        if os.path.exists(os.path.join(output_proxy_dir, d, "meta.json")):
            done_experiments.add(d)

    pending_experiments = [e for e in experiments if e[0] not in done_experiments]
    if done_experiments:
        print(f"  [Resume] {len(done_experiments)} already done, "
              f"{len(pending_experiments)} pending")

    config = QuaDMixConfig(
        num_domains=mm.num_domains, num_quality_criteria=mm.num_quality_criteria,
        num_proxy_experiments=len(experiments),
        num_search_points=args.num_search,
        top_k_average=args.top_k,
        target_tokens=int(args.target_tokens * 1e9) if args.target_tokens > 0 else 0,
        search_weight_mode=args.search_mode,
    )

    if not pending_experiments:
        print(f"  [Resume] All {len(experiments)} experiments already done, skipping Stage 2")
    else:
        # ── Stage 2: Re-evaluate on new validation set ─────────
        _t = time.time()
        print(f"\n[Stage 2] Re-evaluating {len(pending_experiments)} models on {val_set_name}...")

        from quadmix.pipeline.essential_proxy_runner import EssentialWebProxyRunner

        runner = EssentialWebProxyRunner(
            config=config,
            metadata_manager=mm,
            val_data_path=val_path,
            output_dir=output_proxy_dir,
            device_type=args.device_type,
            model_variant=args.model_variant,
            test_block_size=args.block_size,
            token_cache_dir=os.path.join(QUADMIX_TEMP_DIR, "token_cache"),
            checkpoint_interval=0,
        )

        def _save_reval_result(idx: int, val_loss: float, per_task_losses):
            exp_name = pending_experiments[idx][0]
            meta_path = pending_experiments[idx][2]
            with open(meta_path) as f:
                meta = json.load(f)
            old_val_loss = meta["val_loss"]
            new_meta = dict(meta)
            new_meta["val_loss"] = val_loss
            new_meta["val_ppl"] = float(np.exp(val_loss))
            new_meta["original_val_loss"] = old_val_loss
            new_meta["reval_source"] = result_dir
            new_meta["reval_val_set"] = val_set_name
            if per_task_losses is not None:
                new_meta["per_task_losses"] = per_task_losses
            else:
                new_meta.pop("per_task_losses", None)
            exp_out_dir = os.path.join(output_proxy_dir, exp_name)
            os.makedirs(exp_out_dir, exist_ok=True)
            with open(os.path.join(exp_out_dir, "meta.json"), "w") as f:
                json.dump(new_meta, f, indent=2)
            idx_path = os.path.join(proxy_dir, exp_name, "selected_indices.npy")
            if os.path.exists(idx_path):
                shutil.copy2(idx_path, os.path.join(exp_out_dir, "selected_indices.npy"))

        model_paths = [exp[3] for exp in pending_experiments]
        runner.revalidate_batch_parallel(
            model_paths,
            device_type=args.device_type,
            num_gpus=args.num_gpus,
            on_result=_save_reval_result,
        )
        stage_times["stage2_reval"] = time.time() - _t
        print(f"[Stage 2] Re-evaluation: {stage_times['stage2_reval']:.1f}s")

    # ── Load all results (done + new) ──────────────────────
    results = []
    reval_meta = []
    for exp_name, exp_path, meta_path, model_path in experiments:
        out_meta_path = os.path.join(output_proxy_dir, exp_name, "meta.json")
        if not os.path.exists(out_meta_path):
            continue
        with open(out_meta_path) as f:
            new_meta = json.load(f)
        with open(meta_path) as f:
            old_meta = json.load(f)
        new_val_loss = new_meta["val_loss"]
        old_val_loss = old_meta["val_loss"]
        new_per_task_losses = new_meta.get("per_task_losses")
        params = reconstruct_params_from_meta(old_meta)
        results.append(ProxyResult(
            parameters=params,
            validation_loss=new_val_loss,
            metadata=new_meta,
            per_task_losses=new_per_task_losses,
        ))
        reval_meta.append({
            "exp_name": exp_name,
            "old_val_loss": old_val_loss,
            "new_val_loss": new_val_loss,
            "delta": new_val_loss - old_val_loss,
        })

    losses = np.array([r.validation_loss for r in results])
    print(f"  Aggregate loss stats ({len(results)} experiments): "
          f"mean={losses.mean():.4f}, std={losses.std():.4f}, "
          f"min={losses.min():.4f}, max={losses.max():.4f}")

    old_losses = np.array([m["old_val_loss"] for m in reval_meta])
    corr = np.corrcoef(old_losses, losses)[0, 1] if len(losses) > 1 else 0
    print(f"  Correlation (old vs new): {corr:.4f}")

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

    # ── Stage 5: LightGBM Regression ────────────────────
    _t = time.time()
    print(f"\n[Stage 5] Training LightGBM regressor...")
    from quadmix.pipeline.optimizer import QuaDMixOptimizer
    pipeline = QuaDMixPipeline(config)
    pipeline._optimizer = QuaDMixOptimizer(config)
    pipeline._optimizer.add_proxy_results(results)
    pipeline._optimizer.train_regressor()
    stage_times["stage5_lightgbm"] = time.time() - _t
    print(f"[Stage 5] LightGBM: {stage_times['stage5_lightgbm']:.1f}s")

    # ── Stage 6: Optimal Parameter Search ───────────────
    _t = time.time()
    n_search = args.num_search
    top_k = args.top_k
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

    merged = compute_merged_quality_scores(
        quality_scores, domain_labels, optimal_params.merge_config,
    )
    final_ranks = compute_quality_ranks(merged, domain_labels, token_counts)
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
    serialized = pipeline._serialize_params(optimal_params, domain_names, quality_names)
    with open(params_path, "w") as f:
        json.dump(serialized, f, indent=2)

    elapsed = time.time() - t_start
    summary = {
        "config": {
            "num_domains": config.num_domains,
            "num_quality_criteria": config.num_quality_criteria,
            "num_proxy_experiments": len(experiments),
            "num_search_points": n_search,
            "val_set": val_set_name,
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
        "proxy_loss_stats": {
            "new_val_loss": {
                "mean": float(losses.mean()), "std": float(losses.std()),
                "min": float(losses.min()), "max": float(losses.max()),
            },
            "old_val_loss": {
                "mean": float(old_losses.mean()), "std": float(old_losses.std()),
                "min": float(old_losses.min()), "max": float(old_losses.max()),
            },
            "old_new_correlation": float(corr),
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
                orig_dist, sel_dist, config.num_domains, domain_names=domain_names),
        },
        "per_task_loss_stats": per_task_loss_stats,
        "per_task_analysis": pipeline._optimizer.per_task_analysis,
        "reval": {
            "source_result_dir": result_dir,
            "original_val_set": _detect_original_val_set(result_dir),
            "new_val_set": val_set_name,
            "new_val_path": val_path,
            "num_experiments_reevaluated": len(experiments),
            "num_experiments_skipped": skipped,
            "old_loss_mean": float(old_losses.mean()),
            "old_loss_std": float(old_losses.std()),
            "new_loss_mean": float(losses.mean()),
            "new_loss_std": float(losses.std()),
            "loss_correlation": float(corr),
        },
        "dataset_size_prediction": _compute_dataset_size_prediction(optimal_params, mm, config),
        "elapsed_seconds": elapsed,
        "stage_times": {k: round(v, 1) for k, v in stage_times.items()},
        "input_file": args.preprocessed_dir,
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

    reval_header = (
        f"\n## Re-evaluation Info\n\n"
        f"This report is based on re-evaluating saved proxy model weights "
        f"from a previous pipeline run on a **new validation set**.\n\n"
        f"| Field | Value |\n"
        f"|:------|:------|\n"
        f"| Source result | `{result_dir}` |\n"
        f"| Original val set | {_detect_original_val_set(result_dir)} |\n"
        f"| New val set | **{val_set_name}** |\n"
        f"| Experiments re-evaluated | {len(experiments)} |\n"
        f"| Loss correlation (old vs new) | {corr:.4f} |\n"
        f"| Old loss (mean +/- std) | {old_losses.mean():.4f} +/- {old_losses.std():.4f} |\n"
        f"| New loss (mean +/- std) | {losses.mean():.4f} +/- {losses.std():.4f} |\n\n"
    )
    report = report.replace("# QuaDMix", "# QuaDMix (Re-evaluated)\n" + reval_header + "\n# QuaDMix", 1)

    save_report(report, output_dir)
    stage_times["stage9_report"] = time.time() - _t
    print(f"[Stage 9] Report: {stage_times['stage9_report']:.1f}s")

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Re-evaluation Complete! ({total_elapsed:.1f}s)")
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
        print(f"    → R²((1/K)Σ predᵢ, (1/K)Σ actualᵢ): downstream goal quality")
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
    print(f"  Loss correlation (old vs new): {corr:.4f}")
    print(f"  Output: {output_dir}/")
    print(f"    ├── optimal_parameters.json")
    print(f"    ├── pipeline_summary.json")
    print(f"    ├── sampled_dataset.parquet")
    print(f"    ├── quadmix_report.md")
    print(f"    ├── fig1_domain_distribution.png")
    print(f"    ├── fig2_quality_weights.png")
    print(f"    └── proxy_experiments/")
    print("=" * 70)
    return 0


def _build_domain_dist_change(orig_dist, sel_dist, num_domains, domain_names=None):
    change = {}
    for m in range(num_domains):
        if orig_dist[m] > 0:
            name = domain_names[m] if domain_names and m < len(domain_names) else f"D{m}"
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


def _detect_original_val_set(result_dir: str) -> str:
    summary_path = os.path.join(result_dir, "pipeline_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        return s.get("reval", {}).get("new_val_set",
               s.get("config", {}).get("val_set", "unknown"))
    exp_dirs = sorted(
        d for d in os.listdir(os.path.join(result_dir, "proxy_experiments"))
        if d.startswith("exp_")
    ) if os.path.isdir(os.path.join(result_dir, "proxy_experiments")) else []
    if exp_dirs:
        meta_path = os.path.join(result_dir, "proxy_experiments", exp_dirs[0], "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("assistant_loss"):
                return "openhermes (assistant-only loss)"
    return "unknown"


if __name__ == "__main__":
    sys.exit(main())
