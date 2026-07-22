#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Demo: QuaDMix Full — 大规模验证，适合 NPU 集群
# ──────────────────────────────────────────────────────────────
# 目标：500 shards (~39.5B tokens)，200 实验，5000 步
# 基于 FDC L2 标签（22 域），使用 v6 验证集
# 数据不存在则自动下载 + 预处理
#
# 需要 GPU/NPU（CPU 不可行）
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/demo_run_full.sh
#
# 8 张 NPU 并行：
#   bash scripts/demo_run_full.sh --device-type npu --npu-devices 8
#
# 自定义规模：
#   NUM_SHARDS=100 bash scripts/demo_run_full.sh
#   NUM_EXPERIMENTS=100 bash scripts/demo_run_full.sh
#
# 最终输出数据集大小（target_tokens，单位 B）：
#   bash scripts/demo_run_full.sh --target-tokens 10  # 输出 ~10B tokens
#
# 缓存控制：
#   CLEAN_PREPROCESSED=1 — 强制清理预处理目录后重新预处理
#   CLEAN_PREPROCESSED=0 — 强制不清理（默认 auto：旧 shard 多于当前时自动清理）
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

# Temp/cache dir: override via QUADMIX_TEMP_DIR env var, defaults to ~/.cache/QuaDMix/temp/
export QUADMIX_TEMP_DIR="${QUADMIX_TEMP_DIR:-$HOME/.cache/QuaDMix/temp}"

PREPROCESSED_DIR="$QUADMIX_TEMP_DIR/preprocessed"
RAW_DATA_DIR="${RAW_DATA_DIR:-/home/ma-user/work/QuaDMix/data/essential-web}"

# ── 扫描 --val-set 参数（默认 cap_v1）──────────────────
VAL_SET="cap_v1"
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

# ── 下载规模控制 ──────────────────────────────────
# 每 shard ≈ 79M tokens (char//4) / 246 MB 原始 parquet
# 完整 3291 shards ≈ 260B tokens / 791 GB
# 500 shards ≈ 39.5B tokens / 123 GB
# 如果通过环境变量 $NUM_SHARDS 控制下载量：
NUM_SHARDS="${NUM_SHARDS:-500}"
NUM_EXPERIMENTS="${NUM_EXPERIMENTS:-500}"
TOKEN_ESTIMATE=$(( NUM_SHARDS * 79000000 ))
TOKEN_ESTIMATE_B=$(echo "scale=1; $TOKEN_ESTIMATE / 1000000000" | bc 2>/dev/null || echo "~$(( TOKEN_ESTIMATE / 1000000000 ))")
echo "  [配置] 将使用 ~$NUM_SHARDS shards（~$TOKEN_ESTIMATE_B B tokens）"
echo "  [提示] 设 NUM_SHARDS=3291 用满全部数据，NUM_SHARDS=20 快速测试"

# ── 数据就绪检查（逐个 shard 检查，补充下载）──────────────────
NEED_DOWNLOAD=0
MISSING_SHARDS=()

mkdir -p "$RAW_DATA_DIR"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    IDX=$(printf '%05d' $i)
    SHARD_FILE="$RAW_DATA_DIR/train-${IDX}-of-03291.parquet"
    if [ ! -f "$SHARD_FILE" ]; then
        MISSING_SHARDS+=($i)
        NEED_DOWNLOAD=1
    fi
done

if [ $NEED_DOWNLOAD -eq 1 ]; then
    echo "╔══ 数据就绪: 下载缺失 shard ═══╗"
    echo ""
    EXISTING=$(ls "$RAW_DATA_DIR"/*.parquet 2>/dev/null | wc -l || echo 0)
    echo "  已有 shard: $EXISTING, 需要: $NUM_SHARDS"
    echo "  缺失 ${#MISSING_SHARDS[@]} 个 shard，补充下载..."
    python3 "$QUADMIX_DIR/scripts/preprocess/download_essential_web.py" \
        --num-files "$NUM_SHARDS" --output-dir "$RAW_DATA_DIR" \
        --workers "${DOWNLOAD_WORKERS:-16}"
    echo "  ✓ 下载完成"
    echo ""
    echo "╚══════════════════════════════════════╝"
    echo ""
fi

# ── 缓存清理 + 预处理 ──────────────────────────────────────────────
# CLEAN_PREPROCESSED: 强制清理 preprocessed/ 目录（默认自动检测）
#   - 旧 shard 数量 > 当前 NUM_SHARDS 时自动清理（防止旧文件残留）
#   - 设为 1 强制清理，设为 0 强制不清
# CLEAN_TOKEN_CACHE:  清理 token_cache/ 目录（默认清理）
CLEAN_PREPROCESSED="${CLEAN_PREPROCESSED:-auto}"
CLEAN_TOKEN_CACHE="${CLEAN_TOKEN_CACHE:-1}"
RUN_PREPROCESS=0

TOKEN_CACHE_DIR="$QUADMIX_TEMP_DIR/token_cache"

# 自动检测 shard 数量是否变化（减少时需要清理旧文件）
NEED_CLEAN_PREPROCESSED=0
if [ "$CLEAN_PREPROCESSED" = "1" ]; then
    NEED_CLEAN_PREPROCESSED=1
elif [ "$CLEAN_PREPROCESSED" = "0" ]; then
    NEED_CLEAN_PREPROCESSED=0
    elif [ -f "$PREPROCESSED_DIR/shard_index.json" ]; then
    OLD_SHARDS=$(python3 -c "import json; print(json.load(open('$PREPROCESSED_DIR/shard_index.json'))['num_shards'])" 2>/dev/null || echo 0)
    if [ "$OLD_SHARDS" -gt "$NUM_SHARDS" ]; then
        NEED_CLEAN_PREPROCESSED=1
        echo ""
        echo "  [警告] 旧预处理有 $OLD_SHARDS 个 shard，当前只需 $NUM_SHARDS 个"
        echo "         将自动清理旧预处理目录防止残留文件干扰"
    elif [ "$OLD_SHARDS" -eq 0 ] && [ -d "$PREPROCESSED_DIR" ]; then
        # shard_index.json 存在但解析失败或为 0，目录有残留文件
        RESIDUAL=$(ls "$PREPROCESSED_DIR"/preprocessed_*.parquet 2>/dev/null | wc -l || echo 0)
        if [ "$RESIDUAL" -gt 0 ]; then
            NEED_CLEAN_PREPROCESSED=1
            echo ""
            echo "  [警告] shard_index.json 损坏（shards=0），但有 $RESIDUAL 个残留文件"
            echo "         将自动清理旧预处理目录"
        fi
    fi
fi

# 是否需要预处理：清理预处理目录 或 索引不存在 或 shard 数量不一致
if [ $NEED_CLEAN_PREPROCESSED -eq 1 ]; then
    RUN_PREPROCESS=1
elif [ ! -f "$PREPROCESSED_DIR/shard_index.json" ]; then
    RUN_PREPROCESS=1
elif [ -f "$PREPROCESSED_DIR/shard_index.json" ]; then
    OLD_SHARDS=$(python3 -c "import json; print(json.load(open('$PREPROCESSED_DIR/shard_index.json'))['num_shards'])" 2>/dev/null || echo 0)
    RAW_SHARDS=$(ls "$RAW_DATA_DIR"/*.parquet 2>/dev/null | wc -l || echo 0)
    if [ "$OLD_SHARDS" -ne "$RAW_SHARDS" ]; then
        RUN_PREPROCESS=1
        echo ""
        echo "  [信息] 预处理有 $OLD_SHARDS 个 shard，raw data 有 $RAW_SHARDS 个"
        echo "         数量不一致，将重新预处理"
    fi
fi

if [ $RUN_PREPROCESS -eq 1 ]; then
    echo ""
    echo "╔══ 预处理数据 ═══╗"
    echo ""
    if [ $NEED_CLEAN_PREPROCESSED -eq 1 ] && [ -d "$PREPROCESSED_DIR" ]; then
        echo "  [清理] 预处理目录: $PREPROCESSED_DIR"
        rm -rf "$PREPROCESSED_DIR"
    fi
    if [ "$CLEAN_TOKEN_CACHE" = "1" ] && [ -d "$TOKEN_CACHE_DIR" ]; then
        echo "  [清理] token cache 目录: $TOKEN_CACHE_DIR"
        rm -f "$TOKEN_CACHE_DIR"/*.pt 2>/dev/null || true
        rm -rf "$TOKEN_CACHE_DIR" 2>/dev/null || true
    fi
    mkdir -p "$PREPROCESSED_DIR"
    echo "  运行多 shard 预处理脚本..."
    python3 "$QUADMIX_DIR/scripts/preprocess/preprocess_essential_web_v1_sharded.py" \
        --input-dir "$RAW_DATA_DIR" \
        --output-dir "$PREPROCESSED_DIR"
    echo "  ✓ 预处理完成 (multi-shard)"
    echo ""
    echo "╚══════════════════════════════════════╝"
    echo ""
elif [ "$CLEAN_TOKEN_CACHE" = "1" ] && [ -d "$TOKEN_CACHE_DIR" ]; then
    echo "  [清理] token cache 目录: $TOKEN_CACHE_DIR"
    rm -rf "$TOKEN_CACHE_DIR"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$QUADMIX_DIR/result/demo_full_$(date +%Y%m%d_%H%M%S)}"

echo "═══════════════════════════════════════════"
echo "  QuaDMix Demo — Full (大规模, FDC L2)"
echo "  $NUM_SHARDS shards, $NUM_EXPERIMENTS 实验, 5000 步"
echo "  22 域 (FDC L2), $VAL_SET 验证集"
echo "═══════════════════════════════════════════"
cat << PARAMS

  Output:  $OUTPUT_DIR

  ┌────────────────────────────┬──────────────┐
  │ 参数                       │  值          │
  ├────────────────────────────┼──────────────┤
  │ Shards                     │        $NUM_SHARDS  │
  │ 实验数                     │ $NUM_EXPERIMENTS  │
  │ 搜索点                     │       5,000  │
  │ Top-K 平均                 │           5  │
  │ seq_len (block_size)       │       2,048  │
  │ 训练步数                   │       5,000  │
  │ 全局 batch size            │          64  │
  │ 微批大小                   │          64  │ (ga=1)
  │ warmup                     │         4%   │
  │ 验证集                     │  $VAL_SET  │
  │ 排名参考集大小             │      10,000  │
  │ 代理模型                   │  tinyllama_1M│
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
    echo "     建议: CUDA_VISIBLE_DEVICES=0 bash scripts/demo_run_full.sh"
    echo ""
    echo "     快速测试: bash scripts/demo_run_quick.sh  # 小规模验证流程"
    echo ""
fi

# Performance timer: set to 1 to enable detailed timing report
export QUADMIX_PERF_TIMER="${QUADMIX_PERF_TIMER:-1}"

python3 "$QUADMIX_DIR/scripts/runners/run_essential_web_v1.py" \
    --schema "$QUADMIX_DIR/configs/schema_essential_web.yaml" \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --num-experiments "$NUM_EXPERIMENTS" \
    --num-search 5000 \
    --top-k 5 \
    --block-size 2048 \
    --tiny-steps 5000 \
    --micro-batch-size 64 \
    --global-batch-size 64 \
    --rank-ref-size 10000 \
    --checkpoint-interval 0 \
    --val-set "$VAL_SET" \
    --search-mode r2_sigma_weighted \
    --output "$OUTPUT_DIR" \
    $DEVICE_ARG \
    "$@" || exit $?

echo ""
echo "═══════════════════════════════════════════"
echo "  Demo Full complete!"
echo "  Output: $OUTPUT_DIR/"
echo "    ├── quadmix_report.md"
echo "    ├── optimal_parameters.json"
echo "    ├── pipeline_summary.json"
echo "    ├── sampled_dataset.parquet"
echo "    └── proxy_experiments/"
echo "═══════════════════════════════════════════"