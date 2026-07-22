#!/usr/bin/env python3
"""
Resample a (possibly expanded) data pool using previously learned optimal parameters.

Given an optimal_parameters.json from a prior QuaDMix run and a new (larger) data pool
with the same distribution, applies Eq.1 + Eq.2 + Eq.3 to produce a new sampled dataset.

Usage:
  python scripts/runners/resample_with_optimal_params.py \
      --data-dir /path/to/essential-web-v1 \
      --params-file result/quadmix_20260609_120000/optimal_parameters.json \
      --output result/resample_20260611_120000
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
try:
    import quadmix
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import numpy as np
import pandas as pd

from quadmix.core.types import ParameterSet
from quadmix.data.metadata_manager import ShardMetadataManager
from quadmix.data.dataset_schema import DatasetSchema
from quadmix.constants import PROJECT_DIR

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_QUADMIX_DIR = PROJECT_DIR

DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "QuaDMix", "resample",
)


def reconstruct_params_from_json(json_path: str) -> ParameterSet:
    with open(json_path) as f:
        data = json.load(f)
    return ParameterSet.from_dict(data["quality_weights"], data["sampling_params"])


def build_parser():
    p = argparse.ArgumentParser(
        description="Resample data pool with previously learned optimal QuaDMix parameters",
    )
    p.add_argument("--data-dir", required=True,
                   help="Directory containing raw parquet shards (essential-web-v1 format)")
    p.add_argument("--params-file", required=True,
                   help="Path to optimal_parameters.json from a prior QuaDMix run")
    p.add_argument("--preprocessed-dir", default=None,
                   help="Preprocessed shards output dir "
                        "(default: ~/.cache/QuaDMix/resample/preprocessed)")
    p.add_argument("--output", "-o", default=None,
                   help="Output directory (default: result/resample_<timestamp>)")
    p.add_argument("--target-tokens", type=float, default=0.0,
                   help="Target tokens in billions (0 = no limit)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--force", action="store_true",
                   help="Force re-preprocess even if output exists")
    p.add_argument("--workers", type=int, default=64,
                   help="Number of parallel preprocessing workers")
    p.add_argument("--schema", default=None,
                   help="Path to dataset schema YAML (default: Essential-Web schema)")
    return p


def main():
    args = build_parser().parse_args()

    if args.preprocessed_dir:
        preprocessed_dir = args.preprocessed_dir
    else:
        preprocessed_dir = os.path.join(DEFAULT_CACHE_DIR, "preprocessed")

    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(
            _QUADMIX_DIR,
            f"result/resample_{time.strftime('%Y%m%d_%H%M%S')}",
        )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  QuaDMix Resample with Optimal Parameters")
    print(f"  Data:       {args.data_dir}")
    print(f"  Params:     {args.params_file}")
    print(f"  Cache:      {preprocessed_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Seed:       {args.seed}")
    if args.target_tokens > 0:
        print(f"  Target:     {args.target_tokens}B tokens")
    print("=" * 70)

    t_start = time.time()
    stage_times = {}

    # ── Stage 0: Preprocess ──────────────────────────────────
    _t = time.time()
    print(f"\n[Stage 0] Preprocessing raw shards...")
    preprocess_cmd = [
        sys.executable,
        os.path.join(_SCRIPT_DIR, "..", "preprocess", "preprocess_essential_web_v1_sharded.py"),
        "--input-dir", args.data_dir,
        "--output-dir", preprocessed_dir,
        "--workers", str(args.workers),
    ]
    if args.force:
        preprocess_cmd.append("--force")

    print(f"  Command: {' '.join(preprocess_cmd)}")
    result = subprocess.run(preprocess_cmd)
    if result.returncode != 0:
        print(f"[Error] Preprocessing failed with exit code {result.returncode}")
        return 1
    stage_times["stage0_preprocess"] = time.time() - _t
    print(f"[Stage 0] Preprocess: {stage_times['stage0_preprocess']:.1f}s")

    # ── Stage 1: Load metadata ───────────────────────────────
    _t = time.time()
    print(f"\n[Stage 1] Loading metadata from: {preprocessed_dir}")
    schema = DatasetSchema.from_yaml(args.schema) if args.schema else DatasetSchema()
    mm = ShardMetadataManager(preprocessed_dir, schema=schema)
    domain_names = mm.detected_domain_names
    print(f"[Stage 1] {mm.num_docs:,} docs across {mm.num_shards} shards")
    domain_labels = mm.domain_labels
    quality_scores = mm.quality_scores
    token_counts = mm.estimate_token_counts()
    stage_times["stage1_load"] = time.time() - _t
    print(f"[Stage 1] Load: {stage_times['stage1_load']:.1f}s")

    # ── Stage 2: Reconstruct optimal parameters ─────────────
    _t = time.time()
    print(f"\n[Stage 2] Loading optimal parameters from: {args.params_file}")
    optimal_params = reconstruct_params_from_json(args.params_file)
    print(f"  Domains: {optimal_params.num_domains}, "
          f"Criteria: {optimal_params.num_criteria}")
    for m, sc in enumerate(optimal_params.sampling_configs):
        name = domain_names[m] if m < len(domain_names) else f"D{m}"
        print(f"    [{m}] {name}: λ={sc.lambda_:.2f}, ω={sc.omega:.6f}, "
              f"η={sc.eta:.4f}, ε={sc.epsilon:.6f}")
    stage_times["stage2_params"] = time.time() - _t

    # ── Stage 3: Eq.1 — Merge quality scores ─────────────────
    _t = time.time()
    print(f"\n[Stage 3] Merging quality scores (Eq.1)...")
    from quadmix.core.quality_merger import compute_merged_quality_scores
    merged = compute_merged_quality_scores(
        quality_scores, domain_labels, optimal_params.merge_config,
    )
    print(f"  Merged scores: [{merged.min():.4f}, {merged.max():.4f}]")
    stage_times["stage3_eq1"] = time.time() - _t

    # ── Stage 4: Eq.2 — Compute quality ranks ────────────────
    _t = time.time()
    print(f"\n[Stage 4] Computing quality ranks (Eq.2)...")
    from quadmix.core.quality_rank import compute_quality_ranks
    final_ranks = compute_quality_ranks(merged, domain_labels, token_counts)
    print(f"  Quality ranks: [{final_ranks.min():.4f}, {final_ranks.max():.4f}]")
    stage_times["stage4_eq2"] = time.time() - _t

    # ── Stage 5: Eq.3 — Sigmoid sampling ─────────────────────
    _t = time.time()
    print(f"\n[Stage 5] Applying sigmoid sampling (Eq.3)...")
    from quadmix.sampling.batch_sampler import sample_with_optimal_params
    rng = np.random.default_rng(args.seed)
    selected_indices, sampling_values, _ = sample_with_optimal_params(
        final_ranks, domain_labels, optimal_params, rng=rng,
    )

    n_docs = len(domain_labels)
    print(f"  Original documents: {n_docs:,}")
    print(f"  Selected samples:   {len(selected_indices):,}")
    print(f"  Sampling ratio:     {len(selected_indices)/n_docs:.4f}x")
    stage_times["stage5_eq3"] = time.time() - _t

    # ── Stage 6: Target token post-processing ────────────────
    target_tokens = int(args.target_tokens * 1e9) if args.target_tokens > 0 else 0
    if target_tokens > 0:
        _t = time.time()
        actual_tokens = float(np.sum(token_counts[selected_indices]))
        print(f"\n[Stage 6] Target token adjustment:")
        print(f"  θ* produces:  {actual_tokens:,.0f} tokens ({actual_tokens/1e9:.2f}B)")
        print(f"  Target:       {target_tokens:,.0f} tokens ({target_tokens/1e9:.1f}B)")

        if actual_tokens > target_tokens:
            keep_prob = target_tokens / actual_tokens
            rng_discard = np.random.default_rng(args.seed + 1)
            keep_mask = rng_discard.random(len(selected_indices)) < keep_prob
            selected_indices = selected_indices[keep_mask]
            final_tokens = float(np.sum(token_counts[selected_indices]))
            print(f"  Action:       Uniform discard (keep_prob={keep_prob:.4f})")
            print(f"  Final:        {final_tokens:,.0f} tokens ({final_tokens/1e9:.2f}B)")
        elif actual_tokens < target_tokens * 0.95:
            print(f"  Action:       [WARN] θ* produces less than target")
            print(f"  [建议] 调整 ω 参数放宽质量阈值，或降低 target_tokens")
        else:
            print(f"  Action:       Accept θ* result (within tolerance)")
        stage_times["stage6_target"] = time.time() - _t
    else:
        print(f"\n[Stage 6] No target token limit, skipping adjustment")
        stage_times["stage6_target"] = 0.0

    # ── Domain distribution stats ────────────────────────────
    num_domains = optimal_params.num_domains
    orig_dist = np.bincount(domain_labels[domain_labels >= 0], minlength=num_domains)
    sel_dist = np.bincount(
        domain_labels[selected_indices][domain_labels[selected_indices] >= 0],
        minlength=num_domains,
    )
    print(f"\n  Domain distribution:")
    for m in range(num_domains):
        if orig_dist[m] > 0:
            ratio = sel_dist[m] / orig_dist[m]
            name = domain_names[m] if m < len(domain_names) else f"D{m}"
            print(f"    [{m}] {name:>10s}: {orig_dist[m]:>7,} → {sel_dist[m]:>7,}  ({ratio:.2f}x)")

    # ── Stage 7: Save outputs ────────────────────────────────
    _t = time.time()
    print(f"\n[Stage 7] Saving outputs...")

    sampled_texts = mm.read_texts(selected_indices)
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
    print(f"  Sampled dataset: {sampled_path}")

    shutil.copy2(args.params_file, os.path.join(output_dir, "optimal_parameters.json"))

    elapsed = time.time() - t_start
    total_tokens_est = float(np.sum(token_counts[selected_indices]))
    summary = {
        "params_file": args.params_file,
        "data_dir": args.data_dir,
        "preprocessed_dir": preprocessed_dir,
        "seed": args.seed,
        "target_tokens_billions": args.target_tokens,
        "num_original_docs": n_docs,
        "num_shards": mm.num_shards,
        "num_selected_docs": len(selected_indices),
        "sampling_ratio": len(selected_indices) / n_docs,
        "estimated_tokens": total_tokens_est,
        "estimated_tokens_billions": round(total_tokens_est / 1e9, 3),
        "domain_distribution": {
            domain_names[m] if m < len(domain_names) else f"D{m}": {
                "original": int(orig_dist[m]),
                "selected": int(sel_dist[m]),
                "ratio": round(sel_dist[m] / orig_dist[m], 4) if orig_dist[m] > 0 else 0,
            }
            for m in range(num_domains) if orig_dist[m] > 0
        },
        "elapsed_seconds": round(elapsed, 1),
        "stage_times": {k: round(v, 1) for k, v in stage_times.items()},
    }
    summary_path = os.path.join(output_dir, "resample_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating,))
                  else int(x) if isinstance(x, (np.integer,)) else x)
    print(f"  Summary: {summary_path}")

    stage_times["stage7_save"] = time.time() - _t

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Resample Complete! ({total_elapsed:.1f}s)")
    print(f"  Original docs:  {n_docs:,}")
    print(f"  Selected docs:  {len(selected_indices):,}")
    print(f"  Sampling ratio: {len(selected_indices)/n_docs:.4f}x")
    print(f"  Est. tokens:    {total_tokens_est/1e9:.2f}B")
    print(f"  Output: {output_dir}/")
    print(f"    ├── sampled_dataset.parquet")
    print(f"    ├── optimal_parameters.json")
    print(f"    └── resample_summary.json")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
