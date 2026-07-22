#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Demo: QuaDMix STEM — 数学/化学/生物学/物理 四域数据集
# ──────────────────────────────────────────────────────────────
# STEM parquet 已包含 domain/quality/text 列，无需预处理
# schema: configs/schema_stem.yaml
#
# 需要 GPU/NPU（CPU 不可行）
#
# Usage:
#   bash scripts/demo_run_stem.sh
#
# 8 张 NPU 并行：
#   bash scripts/demo_run_stem.sh --device-type npu --npu-devices 8
#
# 自定义规模：
#   NUM_EXPERIMENTS=100 bash scripts/demo_run_stem.sh
#
# 缓存控制：
#   CLEAN_TOKEN_CACHE=0  — 不清理 token cache（默认清理）
#
# ──────────────────────────────────────────────────────────────

set -euo pipefail

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning:torch_npu.utils._path_manager}"

# ── 使用 conda nano 环境（包含 pyarrow 等依赖）─────────────
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate nano
fi

QUADMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${QUADMIX_DIR}/src:${PYTHONPATH:-}"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# Temp/cache dir: override via QUADMIX_TEMP_DIR env var
export QUADMIX_TEMP_DIR="${QUADMIX_TEMP_DIR:-$HOME/.cache/QuaDMix/temp}"

STEM_SCHEMA="$QUADMIX_DIR/configs/schema_stem.yaml"

# ── STEM 数据路径（需要用户指定）───────────────────────────
# 默认路径，可通过环境变量 STEM_DATA_DIR 覆盖
STEM_DATA_DIR="${STEM_DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"

if [ ! -d "$STEM_DATA_DIR" ]; then
    echo "ERROR: STEM 数据目录不存在: $STEM_DATA_DIR"
    echo "请设置 STEM_DATA_DIR 环境变量指向 STEM parquet 文件目录："
    echo "  STEM_DATA_DIR=/path/to/stem/parquets bash scripts/demo_run_stem.sh"
    exit 1
fi

NUM_SHARDS=$(ls "$STEM_DATA_DIR"/*.parquet 2>/dev/null | wc -l || echo 0)
if [ "$NUM_SHARDS" -eq 0 ]; then
    echo "ERROR: STEM 数据目录中没有 .parquet 文件: $STEM_DATA_DIR"
    exit 1
fi
echo "  [配置] 发现 $NUM_SHARDS shards in $STEM_DATA_DIR"

NUM_EXPERIMENTS="${NUM_EXPERIMENTS:-8}"

# ── 扫描 --val-set 参数（默认 cap_v1）──────────────────
VAL_SET="stem_v1"
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--val-set" ]]; then
        VAL_SET="$arg"
        break
    fi
    prev_arg="$arg"
done

# ── 验证集下载（条件化）──────────────────────────────────
source "$QUADMIX_DIR/scripts/ensure_val_data.sh"
ensure_val_set "$VAL_SET" "$QUADMIX_DIR/data" || exit 1

# ── Token cache 清理 ──────────────────────────────────────────
CLEAN_TOKEN_CACHE="${CLEAN_TOKEN_CACHE:-1}"
TOKEN_CACHE_DIR="$QUADMIX_TEMP_DIR/token_cache"

if [ "$CLEAN_TOKEN_CACHE" = "1" ] && [ -d "$TOKEN_CACHE_DIR" ]; then
    echo "  [清理] token cache 目录: $TOKEN_CACHE_DIR"
    rm -rf "$TOKEN_CACHE_DIR"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$QUADMIX_DIR/result/demo_stem_$(date +%Y%m%d_%H%M%S)}"

echo "═══════════════════════════════════════════"
echo "  QuaDMix Demo — STEM (4 域, 5 质量指标)"
echo "  $NUM_SHARDS shards, $NUM_EXPERIMENTS 实验, 5000 步"
echo "  数学/化学/生物学/物理, $VAL_SET 验证集"
echo "═══════════════════════════════════════════"
cat << PARAMS

  Output:  $OUTPUT_DIR

  ┌────────────────────────────┬──────────────┐
  │ 参数                       │  值          │
  ├────────────────────────────┼──────────────┤
  │ Shards                     │        $NUM_SHARDS  │
   │ 实验数                     │       $NUM_EXPERIMENTS  │
  │ 搜索点                     │       5,000  │
  │ Top-K 平均                 │           5  │
  │ seq_len (block_size)       │       2,048  │
  │ 训练步数                   │       5,000  │
  │ 全局 batch size            │          64  │
   │ 微批大小                   │          32  │ (ga=2)
  │ warmup                     │         4%   │
  │ 验证集                     │  $VAL_SET  │
  │ 排名参考集大小             │      10,000  │
  │ 代理模型                   │  tinyllama_1M│
  │ 数据集                     │  STEM        │
  │ 域                         │  4 (数学/化学/生物学/物理) │
   │ 质量指标                   │  5           │
  └────────────────────────────┴──────────────┘

  ⏱ 预计耗时: precompute ~3-5 exp/s (loky 进程池), 训练取决于 NPU 数
PARAMS

if command -v npu-smi &> /dev/null; then
    NPU_DEVICES="${NPU_DEVICES:-8}"
    DEVICE_ARG="--device-type npu --npu-devices $NPU_DEVICES"
    echo "  ✓ NPU 已启用 (Ascend, $NPU_DEVICES 卡)"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    DEVICE_ARG="--device-type cuda"
    echo "  ✓ GPU 已启用 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
else
    DEVICE_ARG=""
    echo "  ⚠ 当前为 CPU 模式，预计耗时 >4 小时！"
    echo "     建议: CUDA_VISIBLE_DEVICES=0 bash scripts/demo_run_stem.sh"
fi

# Performance timer: set to 1 to enable detailed timing report
export QUADMIX_PERF_TIMER="${QUADMIX_PERF_TIMER:-1}"

# ── Pipeline seed ───────────────────────────────────────────────
# QUADMIX_SEED: controls all randomness in the pipeline.
#   unset / empty  → each run is non-deterministic (different experiments each time)
#   int (e.g. 42)  → fully reproducible (same seed → same results)
# To merge multiple runs for better regression, leave unset so each run
# naturally produces diverse parameter configurations.
QUADMIX_SEED="${QUADMIX_SEED:-}"
SEED_ARG=""
if [ -n "$QUADMIX_SEED" ]; then
    SEED_ARG="--seed $QUADMIX_SEED"
fi

# ── STEM 数据不需要预处理，直接用原始 parquet ──────────────
# metadata_manager 会根据 schema 自动读取 domain/quality/text 列

python3 "$QUADMIX_DIR/scripts/runners/run_essential_web_v1.py" \
    --schema "$STEM_SCHEMA" \
    --preprocessed-dir "$STEM_DATA_DIR" \
    --num-experiments "$NUM_EXPERIMENTS" \
    --num-search 5000 \
    --top-k 5 \
    --block-size 2048 \
    --tiny-steps 5000 \
    --micro-batch-size 32 \
    --global-batch-size 64 \
    --rank-ref-size 10000 \
    --checkpoint-interval 0 \
    --val-set "$VAL_SET" \
    --search-mode r2_weighted \
    --output "$OUTPUT_DIR" \
    $DEVICE_ARG \
    $SEED_ARG \
    "$@" || exit $?

echo ""
echo "═══════════════════════════════════════════"
echo "  Demo STEM complete!"
echo "  Output: $OUTPUT_DIR/"
echo "    ├── quadmix_report.md"
echo "    ├── optimal_parameters.json"
echo "    ├── pipeline_summary.json"
echo "    ├── sampled_dataset.parquet"
echo "    └── proxy_experiments/"
echo "═══════════════════════════════════════════"
