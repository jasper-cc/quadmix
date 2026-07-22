# Fixed-5000-step active search

This variant keeps every real proxy experiment at exactly 5000 training
steps. It changes search allocation, not proxy-training fidelity.

The one-shot pipeline and `scripts/run_stem_full.sh` are unchanged. The new
entry point is:

```bash
bash scripts/run_stem_active_5000.sh
```

## Default schedule

- Round 0: 64 Algorithm-1 random configurations.
- Rounds 1-4: 32 configurations per round.
- Each active batch is 50% predicted best, 25% lower-confidence-bound
  exploration, 12.5% parameter diversity, and 12.5% random exploration.
- Each real experiment trains for exactly 5000 steps.
- Each round scores 100,000 cheap candidates with LightGBM.
- The final model scores 100,000 candidates and averages the top 10.

The default schedule therefore runs 192 real experiments. `NUM_SEARCH` and
`CANDIDATE_POOL_SIZE` count LightGBM search points, not NPU experiments.

## Server paths

Defaults match the direct STEM setup:

```text
input:  /data/l00916525/parquet_filter——v3
cache:  /data/l00916525/quadmix_metadata_cache/stem_v3
output: /home/ma-user/work/data-mixing/quadmix_result/stem_active_5000
```

Override them with environment variables. For example:

```bash
OUTPUT_DIR=/home/ma-user/work/data-mixing/quadmix_result/stem_active_trial \
INITIAL_EXPERIMENTS=16 \
ACTIVE_ROUNDS=2 \
ACTIVE_BATCH_SIZE=8 \
bash scripts/run_stem_active_5000.sh
```

`TRAIN_STEPS` is intentionally not configurable in this variant. The Python
runner rejects any `--tiny-steps` value other than 5000.

## Outputs and resume behavior

```text
OUTPUT_DIR/
├── active_state.json
├── active_history.json
├── active_search_summary.json
├── optimal_parameters.json
└── rounds/
    ├── round_00/
    │   ├── selected_candidates.json
    │   ├── round_complete.json
    │   └── proxy_experiments/
    └── ...
```

Rerunning the same command and output directory skips completed rounds using
`active_history.json`. If a process stops inside a round, its directory is
left untouched for inspection; the runner refuses to silently overwrite it.

Final dataset export is deliberately separate so every active round does not
repeat the expensive full-text Stage 8. After reviewing
`optimal_parameters.json`, run the repository's existing resampling/export
workflow once.
