# QuaDMix — Quality-Diversity Balanced Data Selection

> **⚠️ Disclaimer**: This repository is a personal clean-room implementation of the QuaDMix paper,
> not an official ByteDance release. For educational and research purposes only.

> **Paper**: Fengze Liu, Weidong Zhou, Binbin Liu, et al. (ByteDance, 2025)
>
> [arXiv:2504.16511v2](https://arxiv.org/abs/2504.16511)

Selects high-quality, domain-balanced subsets from large unlabeled corpora for LLM pretraining.
Works with any dataset — just write a YAML schema config.

## Algorithm Pipeline

```
Raw Shards (parquet) → Preprocess → Metadata (domain labels + quality scores) in memory
  ↓                                    (schema-driven: any domain/quality columns)
Alg.1: Sample θ = (α₁…αₘ, β₁…βₘ)
  ↓
For each θ:
  Eq.1: Merge quality signals with αₘ → ¯q
  Eq.2: Rank ¯q within domain (subsample reference) → ¯r
  Eq.3: S(¯r) = sigmoid(λ(ω-¯r))^η + ε → Bernoulli sample
  Train 1M param proxy model → per-task val_loss (full-sequence)
  (RegMix-style: permutation shuffle, warmup_fraction=4%)
  ↓
Per-task LightGBM: R(θ) → predicted loss per capability cluster
  (Adaptive regime: conservative/moderate/aggressive based on n/p ratio)
  (K-fold CV for R² estimation, R²≤0 tasks filtered)
  ↓
Search: Alg.1 × 5K → R²-weighted z-score → top-K average → θ*
  ↓
Eq.1+Eq.2(θ*.αₘ) → Eq.3(θ*.βₘ) → Sampled Dataset
```

## Project Structure

```
quadmix/
├── configs/                        # YAML dataset schema configs
│   └── schema_essential_web.yaml       # Essential-Web (示例数据集)
├── src/quadmix/                    # Python package (pip install -e .)
│   ├── constants.py                    # Domain names, quality criteria, HF paths (Essential-Web defaults)
│   ├── core/                       # Core algorithm (Eq.1-3, proxy model)
│   │   ├── quality_merger.py           # Eq.1: Merged quality score
│   │   ├── quality_rank.py             # Eq.2: Quality percentile within domain
│   │   ├── sampler.py                  # Eq.3: Sigmoid sampling
│   │   ├── proxy_model.py              # ~1M param proxy (SwiGLU+RMSNorm)
│   │   ├── domain_classifier.py        # FDC L2 URL-based domain classifier
│   │   ├── quality_scorer.py           # FastText quality signal scoring
│   │   └── types.py                    # Core data types
│   ├── pipeline/                   # Pipeline orchestration
│   │   ├── real_pipeline.py            # Main pipeline runner
│   │   ├── param_sampler.py            # Alg.1: Parameter generation
│   │   ├── optimizer.py                # Per-task LightGBM + R²-weighted search
│   │   ├── report.py                   # MD report + figures
│   │   ├── essential_proxy_runner.py   # Shard-aware proxy experiments
│   │   ├── proxy_runner.py             # Proxy training loop
│   │   ├── parallel_dispatch.py        # Multi-NPU dynamic task dispatch
│   │   ├── tokenize_worker.py          # Parallel tokenization workers
│   │   ├── shared_memory.py            # Shared memory utilities
│   │   └── loss_utils.py              # Per-task loss computation
│   ├── sampling/
│   │   └── batch_sampler.py            # Bernoulli batch sampling
│   ├── data/
│   │   ├── dataset_schema.py           # DatasetSchema — YAML-driven column mapping
│   │   ├── metadata_manager.py         # ShardMetadataManager (schema-driven, MMap-aware)
│   │   ├── base.py                     # Data adapter base class
│   │   ├── registry.py                 # Adapter registry
│   │   ├── parquet_adapter.py          # Parquet format adapter
│   │   ├── jsonl_adapter.py            # JSONL format adapter
│   │   ├── csv_adapter.py              # CSV format adapter
│   │   └── txt_adapter.py              # Plain text adapter
│   ├── npu/
│   │   └── device.py                   # DeviceManager (CPU/CUDA/NPU)
│   └── utils/
│       └── normalization.py
├── scripts/
│   ├── runners/
│   │   ├── run_essential_web_v1.py     # Main entry (full pipeline)
│   │   ├── reval_with_new_valset.py    # Re-evaluate with new validation set
│   │   ├── resume_from_stage5.py       # Re-optimize from existing losses
│   │   └── resample_with_optimal_params.py  # Resample with θ* on expanded data
│   ├── preprocess/
│   │   ├── preprocess_essential_web_v1_sharded.py  # Multi-shard preprocessing
│   │   └── download_essential_web.py   # Download tool
│   ├── validation_set/             # Validation set preparation scripts
│   │   ├── prepare_cap_v1.py           # CAP v1 (default, capability-aligned)
│   │   ├── prepare_stem_v1.py          # STEM v1 (4 STEM benchmarks, aligned with downstream)
│   │   ├── prepare_core_bmk_v6.py      # CORE-BMK v6 (21 benchmarks)
│   │   └── ...                         # v2-v5 historical versions
│   ├── analysis/                   # Diagnostic & analysis scripts
│   ├── ensure_val_data.sh          # Validation set auto-download helper
│   ├── demo_run_quick.sh           # NPU 快速验证 (~1.5h, 8x NPU)
│   ├── demo_run_full.sh            # 大规模验证 (~数小时, 8x NPU)
│   ├── demo_revalidate.sh          # 换验证集重新评估 (无需重训 proxy)
│   ├── demo_reoptimize.sh          # 用已有 loss 重新优化
│   └── resample.sh                 # 用 θ* 对扩容数据重新采样
├── nanochat_mid_compare/           # 下游中训练对比框架
│   ├── run_experiment.sh               # QuadMix vs Random vs Quality 对比实验
│   ├── prepare_data.py                 # 数据准备 (token 精确对齐)
│   ├── generate_quadmix_report.py      # 对比报告生成
│   └── diagnose_domain_distribution.py # 域分布诊断
├── docs/
│   ├── CAP_DESIGN.md                   # CAP v1 验证集设计文档
│   ├── FDC_DOMAIN_MAPPING_DESIGN.md    # FDC L2 域映射设计
│   ├── PER_TASK_LOSS_V4.2_DESIGN.md    # Per-task loss 设计
│   ├── ARCHITECTURE.md                 # 架构文档
│   └── NPU_DEPLOYMENT.md              # NPU 部署指南
├── result/                         # Final results (one dir per run)
├── data/                           # Validation sets (auto-downloaded)
│   ├── cap_v1_tokenized.pt             # CAP v1 (default)
│   ├── stem_v1_tokenized.pt            # STEM v1 (4 tasks: GSM8K/MMLU-22STEM/ARC-Easy/ARC-Challenge)
│   ├── core_bmk_21tasks_v6_tokenized.pt # CORE-BMK v6
│   └── ...                             # Historical versions
└── temp/                           # Intermediate data (deletable)
    ├── preprocessed/                   # Preprocessed shards (FDC L2 + quality)
    └── token_cache/                    # mmap-mapped tokenization cache
```

## Quick Start

```bash
# Install
pip install -e .

# Essential-Web demo (~1.5h, 8x NPU) — 使用示例数据集配置
bash scripts/demo_run_quick.sh

# Large-scale Essential-Web demo (~数h, 8x NPU, 500 shards)
bash scripts/demo_run_full.sh

# Custom dataset — 参考 configs/schema_essential_web.yaml 写自己的 YAML
python scripts/runners/run_essential_web_v1.py \
    --schema configs/schema_essential_web.yaml \
    --preprocessed-dir temp/preprocessed \
    --num-experiments 500 \
    --num-search 5000 \
    --block-size 2048 \
    --val-set cap_v1 \
    --search-mode r2_sigma_weighted \
    --output result/my_run

# Re-evaluate with new validation set (no proxy retraining)
bash scripts/demo_revalidate.sh --result-dir result/xxx --val-set cap_v1

# Re-optimize from existing losses
bash scripts/demo_reoptimize.sh --result-dir result/xxx

# Resample with optimal θ* on expanded data pool
DATA_DIR=/path/to/data PARAMS_FILE=result/xxx/optimal_parameters.json \
  bash scripts/resample.sh
```

> **Note**: The default validation set (`cap_v1`) is automatically downloaded from
> [HuggingFace](https://huggingface.co/datasets/liujin99/quadmix-cap-v1) on first run.
> Other validation sets (`core_bmk_v6`, `openhermes`, etc.) are also available via `--val-set`.

## Custom Datasets

QuaDMix supports arbitrary datasets via YAML schema configs. All datasets (including Essential-Web)
must specify `--schema`. See `configs/schema_essential_web.yaml` as a reference.

### Your parquet must have:

| Column type | Required | dtype | Example |
|---|---|---|---|
| domain | ✓ | string or int | `category_name` (string) → auto-mapped to int 0..M-1 |
| quality | ✓ (1+ cols) | float | `category_score`, `stem_relevance` |
| text | ✓ | string | `content` |
| char_count | optional | int | `doc_len` (if null → computed from text length) |
| row_in_shard | optional | int | `row_in_shard` (if null → positional indexing) |

### Example: STEM dataset config

```yaml
# configs/schema_stem.yaml — actual config (copy and modify for your dataset)
domain_col: category_name            # string column → auto int mapping via domain_names
domain_names:                        # controls string→int mapping order (数学=0, 化学=1, ...)
  - 数学
  - 化学
  - 生物学
  - 物理

quality_cols:                        # all default to higher_better=true
  - stem_relevance                   #   (pre-processed: higher score = better quality)
  - knowledge_value
  - notation_fidelity
  - rigor_coherence
  - noise_level                      #   (pre-processed: higher score = less noise = better)

quality_names:                       # display names in reports/figures (optional, defaults to col names)
  - stem_relevance
  - knowledge_value
  - notation_fidelity
  - rigor_coherence
  - noise_level

text_col: text                       # document text column
char_count_col: char_count_col       # pre-computed token proxy (null → compute from text)
row_in_shard_col: source_record_idx  # explicit row indexing (null → positional)
```

For quality columns where lower scores mean better quality, use the dict syntax:

```yaml
quality_cols:
  - stem_relevance                   # higher_better=true (default)
  - name: noise_level                # override: lower score = better quality
    higher_better: false
```

### Then run:

```bash
python scripts/runners/run_essential_web_v1.py \
    --schema configs/schema_stem.yaml \
    --preprocessed-dir /path/to/stem/shards \
    --output result/stem_run
```

See `docs/DATASET_SCHEMA_DESIGN.md` for full details.

## Validation Sets

| 验证集 | 来源 | 特点 | 默认 |
|--------|------|------|------|
| **cap_v1** | 外部训练数据(70%) + benchmark train(30%) | 能力对齐，绕过 C2 假设，5 clusters × 8K = 40K samples | ✓ |
| **stem_v1** | 4 STEM benchmarks (GSM8K/MMLU-22STEM/ARC-Easy/ARC-Challenge) | 与下游直接对应 (arc_easy/arc_challenge/mmlu_stem/gpqa_diamond/gsm8k_cot/math_cot)，full-sequence loss，~10.5K samples | |
| core_bmk_v6 | 21 个 CORE benchmark | Benchmark 对齐，per-task loss，~31K samples | |
| openhermes | OpenHermes-2.5-1M | SFT 对话格式，通用质量信号，10K samples | |

## Validation Status

**已验证环境：**

| 配置项 | 规格 |
|-------|------|
| NPU | 8x Ascend 910B3, 64GB VRAM each |
| 内存 | 1500 GB |
| CPU | ARM 192 vCPUs |
| 验证脚本 | `demo_run_quick.sh` (8 experiments) / `demo_run_full.sh` (500 experiments) |
| 验证结果 | ✓ RegMix 风格训练 + CAP v1 验证集 + per-task LightGBM |

验证日志：
- 并行 tokenize (多 worker + 多线程) 正常执行
- 动态任务队列调度正常（8 NPU 并行）
- Per-task val_loss 计算正常（5 capability clusters）
- Permutation shuffle 训练循环正常（无放回 epoch 遍历）
- Per-task LightGBM 回归 + R²-weighted 搜索正常
- NPU HBM 显式释放 — 实验间 `gc.collect()` + `torch.npu.empty_cache()`
- Optimizer inf/nan 过滤 — 防止非有限 val_loss 污染 LightGBM

## Demo Scripts

| 脚本 | 设备 | 数据量 | 实验 | 搜索点 | 步数 | 耗时 | 用途 |
|------|------|--------|------|--------|------|------|------|
| `demo_run_quick.sh` | 8x NPU | 20 shards | 8 | 1,000 | 5,000 | ~1.5h | **快速测试** |
| `demo_run_full.sh` | 8x NPU | 500 shards | 500 | 5,000 | 5,000 | ~数h | 大规模验证 |
| `demo_revalidate.sh` | NPU | — | — | 100,000 | — | ~30min | 换验证集重评估 |
| `demo_reoptimize.sh` | CPU | — | — | 100,000 | — | ~5min | 用已有 loss 重优化 |
| `resample.sh` | CPU | 全量 | — | — | — | ~1h | θ* 重采样 |

所有训练 demo 使用 CAP v1 验证集（40K samples）、warmup_fraction=4%、`configs/schema_essential_web.yaml` schema（22 域）。

## Architecture Highlights

### Multi-Shard Scalability
|- **Schema-driven**: YAML config maps any parquet columns to domain/quality/text
|- **Metadata in memory**: only domain labels + quality scores (~12 GB for 275M docs, Essential-Web)
|- **Text on demand**: per-shard parquet, loaded only for selected documents
|- **mmap token cache**: `np.load(path, mmap_mode='r')` — pages loaded lazily, not full file
|- **Multi-format adapters**: Parquet, JSONL, CSV, plain text via unified registry

### Multi-NPU Parallelism
|- **Dynamic task queue**: Workers fetch tasks on-demand, no batch boundaries
|- **Auto load-balancing**: Fast workers naturally do more experiments
|- **CPU-NPU overlap**: Tokenize thread runs independently, ahead of training
|- **~8x speedup**: 3000 experiments reduced from ~100h to ~15h with 8 NPUs

### Per-Task LightGBM
|- **5 capability clusters**: world_knowledge, symbol_logic, language_understanding, reading_comprehension, common_sense_reasoning
|- **Adaptive regime**: conservative (n/p<3) / moderate (3-8) / aggressive (≥8) hyperparameters
|- **K-fold CV**: Per-task R² estimation, R²≤0 tasks auto-filtered
|- **R²-weighted search**: High-R² tasks contribute more to search objective

### Incremental Workflow
|- **Re-validate**: Swap validation set without retraining proxy models
|- **Re-optimize**: Re-run LightGBM + search with existing losses
|- **Resample**: Apply θ* to expanded data pool without re-running pipeline

### Directory Layout
| Path | Content | Persistence |
|------|---------|-------------|
| `result/<experiment>/` | optimal_parameters.json, sampled_dataset.parquet, report, figures | Permanent |
| `temp/preprocessed/` | domain labels + quality scores per shard (schema-driven) | Keep until data changes |
| `temp/token_cache/` | mmap-able .npy files (one per shard) | Auto-regenerated |

### Data Flow
```
Raw: 3291 shards × 246 MB = ~808 GB total (Essential-Web, ~260B tokens)
  │                                       (or any dataset with YAML schema)
  ▼ [preprocess: domain + quality signals, schema-driven]
Preprocessed: 1 shard → 1 parquet (~180 MB each)
  │
  ▼ [ShardMetadataManager: in-memory metadata, schema-driven]
In-memory: domain(275M int64) + quality_scores(275M×N float64)
  │
  ▼ [EssentialWebProxyRunner: per-experiment, schema-driven column names]
Token Cache: shard_{idx}_bs{bs}.npz (mmap-mode, int32 + row_index)
Token Cache: shard_{idx}_bs{bs}.npz (mmap-mode, int32 + row_index)
  │
  ▼ [Per-task LightGBM + R²-weighted search]
Final: optimal_parameters.json + sampled_dataset.parquet
```

## Output

```
result/<experiment_name>/
├── optimal_parameters.json        # Optimal θ (α_m + β_m)
├── pipeline_summary.json          # Config + metrics + statistics
├── sampled_dataset.parquet        # Final optimally sampled dataset
├── quadmix_report.md              # Comparison report (with per-task R²)
├── fig1_domain_distribution.png   # Original vs optimal domain distribution
├── fig2_quality_weights.png       # Quality signal weights
└── proxy_experiments/             # Per-experiment results
    ├── exp_0000/
    │   ├── meta.json                # Full parameters + per-task val_loss
    │   ├── model.pt                 # Saved proxy model weights (for re-validation)
    │   ├── selected_indices.npy     # Selected document indices
    │   └── checkpoint_trajectory.json  # val_loss at checkpoint intervals
    └── exp_0001/ ...
```

## Downstream Evaluation

`nanochat_mid_compare/` provides a framework for comparing QuadMix against baselines via mid-training:

```bash
# QuadMix vs Random vs Quality(dclm) vs Quality(fineweb_edu)
QUADMIX_SAMPLED_DATA=result/xxx/sampled_dataset.parquet \
PREPROCESSED_DATA_DIR=temp/preprocessed \
bash nanochat_mid_compare/run_experiment.sh
```

Baselines are token-budget aligned (same total tokens as QuadMix output) and share the same base model checkpoint.

## License

Apache 2.0
