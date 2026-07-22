#!/usr/bin/env python3
"""Run fixed-5000-step sequential active search on sharded STEM data."""

from __future__ import annotations

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.join(REPO_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import run_essential_web_v1 as base_runner

from quadmix.core.types import QuaDMixConfig
from quadmix.data.dataset_schema import DatasetSchema
from quadmix.data.metadata_manager import ShardMetadataManager
from quadmix.pipeline.active_search import (
    ActiveSearchConfig,
    ActiveSearchController,
    FIXED_TRAIN_STEPS,
)


def build_parser():
    parser = base_runner.build_parser()
    parser.description = (
        "QuaDMix fixed-5000-step sequential active search on sharded data"
    )
    parser.set_defaults(
        tiny_steps=FIXED_TRAIN_STEPS,
        checkpoint_interval=0,
        seed=42,
    )
    parser.add_argument(
        "--initial-experiments",
        type=int,
        default=64,
        help="Random Algorithm-1 experiments in round 0 (default: 64)",
    )
    parser.add_argument(
        "--active-rounds",
        type=int,
        default=4,
        help="Model-guided rounds after round 0 (default: 4)",
    )
    parser.add_argument(
        "--active-batch-size",
        type=int,
        default=32,
        help="Real proxy experiments in each active round (default: 32)",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=100_000,
        help="Cheap LightGBM-scored candidates per active round",
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=1.5,
        help="Initial lower-confidence-bound exploration coefficient",
    )
    parser.add_argument(
        "--beta-decay",
        type=float,
        default=0.75,
        help="Per-round beta multiplier",
    )
    # Some legacy help strings contain literal percentages. argparse treats
    # them as %-format placeholders when rendering --help.
    for action in parser._actions:
        if action.help:
            action.help = action.help.replace("%", "%%")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.quick or args.full or args.num_experiments is not None:
        raise SystemExit(
            "Active search does not use --quick, --full, or --num-experiments; "
            "use --initial-experiments, --active-rounds, and "
            "--active-batch-size."
        )
    if args.tiny_steps != FIXED_TRAIN_STEPS:
        raise SystemExit(
            f"This runner requires --tiny-steps {FIXED_TRAIN_STEPS}; "
            f"got {args.tiny_steps}."
        )

    output_dir = args.output or os.path.join(
        REPO_DIR, f"result/active_stem_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Setup] Loading dataset schema: {args.schema}")
    schema = DatasetSchema.from_yaml(args.schema)
    print(f"[Setup] Loading sharded metadata: {args.preprocessed_dir}")
    metadata_manager = ShardMetadataManager(
        args.preprocessed_dir,
        schema=schema,
        shard_limit=args.shard_limit,
        max_workers=int(os.environ.get("STEM_METADATA_WORKERS", "8")),
    )
    print(
        f"[Setup] {metadata_manager.num_docs:,} docs, "
        f"{metadata_manager.num_shards} shards, "
        f"{metadata_manager.num_domains} domains, "
        f"{metadata_manager.num_quality_criteria} quality criteria"
    )

    total_experiments = (
        args.initial_experiments + args.active_rounds * args.active_batch_size
    )
    final_search_points = args.num_search or 100_000
    config = QuaDMixConfig(
        num_domains=metadata_manager.num_domains,
        num_quality_criteria=metadata_manager.num_quality_criteria,
        num_proxy_experiments=total_experiments,
        num_search_points=final_search_points,
        top_k_average=args.top_k,
        target_tokens=int(args.target_tokens * 1e9) if args.target_tokens > 0 else 0,
        search_weight_mode=args.search_mode,
        seed=args.seed,
    )
    active_config = ActiveSearchConfig(
        initial_experiments=args.initial_experiments,
        active_rounds=args.active_rounds,
        batch_size=args.active_batch_size,
        candidate_pool_size=args.candidate_pool_size,
        final_search_points=final_search_points,
        top_k=args.top_k,
        beta_start=args.beta_start,
        beta_decay=args.beta_decay,
        seed=args.seed,
        num_workers=args.npu_devices,
        train_steps=args.tiny_steps,
    )

    proxy_runner = base_runner.create_proxy_runner(
        config, args, output_dir, metadata_manager
    )
    controller = ActiveSearchController(
        quadmix_config=config,
        active_config=active_config,
        proxy_runner=proxy_runner,
        output_dir=output_dir,
        domain_names=metadata_manager.detected_domain_names,
        quality_names=metadata_manager.detected_quality_names,
    )
    summary = controller.run()

    print("\n" + "=" * 72)
    print("  Active search complete")
    print(f"  Experiments: {summary['num_experiments']}")
    print(f"  Best observed loss: {summary['best_observed_loss']:.6f}")
    print(f"  Parameters: {summary['optimal_parameters']}")
    print("  Final dataset export is intentionally not run in every round.")
    print("  Use resample_with_optimal_params.py once after reviewing the result.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
