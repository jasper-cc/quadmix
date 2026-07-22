#!/usr/bin/env bash
# Fixed-5000-step sequential active search for direct STEM parquet input.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/ma-user/miniforge3/envs/nano}"
PYTHON_BIN="$CONDA_ENV_PATH/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python not found: $PYTHON_BIN" >&2
    exit 2
fi

export PATH="$CONDA_ENV_PATH/bin:${PATH}"
export CONDA_PREFIX="$CONDA_ENV_PATH"
export CONDA_DEFAULT_ENV="$(basename "$CONDA_ENV_PATH")"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export STEM_METADATA_WORKERS="${STEM_METADATA_WORKERS:-32}"
export PRESAMPLE_MAX_WORKERS="${PRESAMPLE_MAX_WORKERS:-64}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PARQUET_FILTER_DIR="${PARQUET_FILTER_DIR:-/data/l00916525/parquet_filter——v3}"
STEM_METADATA_CACHE_DIR="${STEM_METADATA_CACHE_DIR:-/data/l00916525/quadmix_metadata_cache/stem_v3}"
QUADMIX_TOKENIZER_PATH="${QUADMIX_TOKENIZER_PATH:-/home/ma-user/work/data-mixing/quadmix-jasper/data/tokenizers/gpt-neox-20b}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ma-user/work/data-mixing/quadmix_result/stem_active_5000}"
SCHEMA="${SCHEMA:-$REPO_DIR/configs/schema_stem.yaml}"

export STEM_METADATA_CACHE_DIR
export QUADMIX_TOKENIZER_PATH
export QUADMIX_TEMP_DIR="${QUADMIX_TEMP_DIR:-$REPO_DIR/temp/stem_active_5000}"

INITIAL_EXPERIMENTS="${INITIAL_EXPERIMENTS:-64}"
ACTIVE_ROUNDS="${ACTIVE_ROUNDS:-4}"
ACTIVE_BATCH_SIZE="${ACTIVE_BATCH_SIZE:-32}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-100000}"
NUM_SEARCH="${NUM_SEARCH:-100000}"
TOP_K="${TOP_K:-10}"
NPU_DEVICES="${NPU_DEVICES:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
RANK_REF_SIZE="${RANK_REF_SIZE:-10000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-0}"
VAL_SET="${VAL_SET:-stem_v1}"

if [[ ! -d "$PARQUET_FILTER_DIR" ]]; then
    echo "ERROR: Parquet directory not found: $PARQUET_FILTER_DIR" >&2
    exit 2
fi

echo "========================================================================"
echo "  QuaDMix fixed-5000-step active search"
echo "  Source:             $PARQUET_FILTER_DIR"
echo "  Output:             $OUTPUT_DIR"
echo "  Round 0:            $INITIAL_EXPERIMENTS experiments"
echo "  Active rounds:      $ACTIVE_ROUNDS x $ACTIVE_BATCH_SIZE experiments"
echo "  Total experiments:  $((INITIAL_EXPERIMENTS + ACTIVE_ROUNDS * ACTIVE_BATCH_SIZE))"
echo "  Steps/experiment:   5000 (fixed)"
echo "  NPU devices:        $NPU_DEVICES"
echo "========================================================================"

exec "$PYTHON_BIN" "$REPO_DIR/scripts/runners/run_active_stem.py" \
    --preprocessed-dir "$PARQUET_FILTER_DIR" \
    --schema "$SCHEMA" \
    --output "$OUTPUT_DIR" \
    --initial-experiments "$INITIAL_EXPERIMENTS" \
    --active-rounds "$ACTIVE_ROUNDS" \
    --active-batch-size "$ACTIVE_BATCH_SIZE" \
    --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
    --num-search "$NUM_SEARCH" \
    --top-k "$TOP_K" \
    --tiny-steps 5000 \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --block-size "$BLOCK_SIZE" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --rank-ref-size "$RANK_REF_SIZE" \
    --val-set "$VAL_SET" \
    --search-mode r2_sigma_weighted \
    --device-type npu \
    --npu-devices "$NPU_DEVICES" \
    "$@"
