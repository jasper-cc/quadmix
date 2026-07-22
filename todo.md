# QuaDMix 项目审计问题清单

> 2026-07-22 第二轮深度分析更新。已修复项移至历史表，新增问题按严重性分类。

---

## 已修复的 bug（按 commit 时间顺序）

| Commit | Bug | 文件 | 描述 |
|--------|-----|------|------|
| `645fe5e` | SSL 绕过 | `_hf_remote_size()` | HF mirror 环境下载失败 |
| `8dddca9` | 预计算线程爆炸 | precompute | 未限制线程数 |
| `c39c3eb` | ConcurrencyConfig 中央化 | concurrency.py | 环境变量分散管理 |
| `5fe3f75` | GIL 修复 | parallel_dispatch.py | loky ProcessPoolExecutor + 共享内存 |
| `8db1d96` | 每进程内存估算修复 | parallel_dispatch.py | 共享内存被重复计入 |
| `2f00da7` | demo 脚本过时 env var | 3 个 demo_run_*.sh | 移除 TOKENIZE_WORKERS/THREADS |
| `df57be9` | 共享内存计入 + CANN 警告 | parallel_dispatch.py | 共享内存不计入每进程限制 |
| `f877f18` | row_in_shard_col ID 空间 | 3 文件 | STEM source_record_idx 不匹配 |
| `ab466ad` | CANN 警告模块级 | 3 文件 | suppress 在 import 之后才生效 |
| `43ee2be` | 自适应 val_bs + chunk_size | loss_utils.py, parallel_dispatch.py | 32GB NPU OOM |
| `44aeb8e` | micro_batch_size=32 | demo_run_stem.sh | 梯度累积=2，200实验 |
| `a6b0522` | worker crash + chunk_size + Stage8 卡住 | 4 文件 | ① except 块 import torch UnboundLocalError ② cs>seq_len 跳过 ③ read_texts 并行化 |
| `84bccdd` | `_stage9_report` NameError | real_pipeline.py | domain_names/quality_names/mm 未定义 → 加参数 |
| `a7e1744` | C1-C10 全部修复 | 多文件 | 9个 CRITICAL bug：optimizer NameError、M/N未定义、QUALITY_SHORT、bincount含-1、跨shard code不一致、非连续domain索引、$prev_arg、reval domain_names NameError、preprocess路径 |
| `9a84ee2` | H1+H4 修复 | shared_memory.py + 5处np.load | SharedMemory fd 泄漏 + NpzFile 不关闭 |
| `049f816` | H5-H11 修复 | 多文件 | jsonl text_key、from_shared崩溃、scalar exponent clip、perf_timer无锁、硬编码路径、demo过期目录、reval SSL+choices |
| `19c2371` | H2+H3+H8+H9 修复 | 多文件 | exp_shm_info无锁、memory_cache无锁、perf_timer类级状态、硬编码/home/ma-user路径 |
| `12822f0` | CJK 字体自动检测 | report.py | 动态检测matplotlib CJK字体支持 |
| `b6df3bb` | read_texts 性能优化 | metadata_manager.py | ProcessPoolExecutor(spawn) + pyarrow + 自适应策略 |
| `02fefaa` | tokenize 性能优化 | metadata_manager.py | pd.read_parquet → pyarrow + 自适应策略 |
| `94a5297` | shard_tasks unpack | parallel_dispatch.py | 5-tuple (sid, path, miss_rows, is_seq, total_rows) |
| `4629a34` | normalization + domain_indices skip | essential_proxy_runner.py | worker模式跳过归一化省100s |
| `b903f50` | NaN/Inf JSON + CJK字体 | 多文件 | sanitize JSON输出 + matplotlib CJK检测 |
| `568540e` | NPUGraph capture | parallel_dispatch.py | 消除Python dispatch间隙 |
| `c993ff8` | M1-M16 全部修复 + L22 | 多文件 | 16个MEDIUM bug：except:pass、env var不恢复、tied rank、__eq__ ndarray、from_flattened无校验、Python循环remap、read_texts全列、row_col_to_local重复建dict、ensure_*重复、crash shm清理、cache非原子写、负domain无warning、_validate只查首shard、assert运行时、concurrency numpy已导入后无效、csv/parquet cast、from_shared绕过__init__ |
| `0833478` | dead --seed + pyyaml | run_essential_web_v1.py + pyproject.toml + requirements.txt | 删除无效--seed参数 + 补缺失pyyaml依赖 |

---

## HIGH — 严重行为错误或安全风险（未修复）

### H12. `param_sampler.py:52,64,119` — 除零风险 → NaN 全链路传播
- `a / a.sum()` 和 `a_all / a_all.sum(axis=1)` — N=1 且 uniform 采样到 0.0 时 sum=0
- NaN 传播：domain_weights→merged_quality→ranks→sampling→LightGBM 全链路污染
- **修复**：`a_norm = a / max(a.sum(), 1e-12)` 或采样前加 `a += 1e-10`

### H13. `essential_proxy_runner.py:248,884` + `parallel_dispatch.py:462` — torch.load weights_only=False
- pickle 反序列化允许任意代码执行
- 验证数据含非 tensor 对象（task_labels list），无法用 weights_only=True
- **修复**：将 task_labels 改为 tensor 存储，或对 .pt 文件做 SHA256 校验后信任

### H14. README YAML 示例与实际 config 矛盾
- README 示例：`quality_cols: [category_score, stem_relevance, {name: noise_level, higher_better: false}]`
- 实际 `schema_stem.yaml`：5个不同列名，纯字符串无 `higher_better` 标注
- 用户按 README 写的 config 列名/结构完全错误
- **修复**：更新 README 示例与实际 schema_stem.yaml 一致

---

## MEDIUM — 设计/性能/可维护性（未修复）

### M17. `essential_proxy_runner.py` 2266行巨型单文件
- 训练循环、tokenization、mmap缓存、并发管理全在一个类
- Eq.1-3 有重复实现（`_compute_ranks_for_params` vs 核心模块），bug fix需同步两处
- **修复**：拆分为 training_loop / tokenization / cache_manager / sampling_logic 四个模块

### M18. `types.py` vs CLI — search_weight_mode 默认值矛盾
- `QuaDMixConfig.search_weight_mode` 默认 `"equal_weight"`
- CLI `--search-mode` 默认 `"r2_sigma_weighted"`
- **修复**：统一默认值为 `"r2_sigma_weighted"`（论文推荐）

### M19. `constants.py` VAL_SHA256 定义但未使用
- 11个验证集的 SHA256 hash 已硬编码，但 `_ensure_hf_data` 只比较文件大小
- 崩溃/篡改的下载数据无声影响 proxy 结果
- **修复**：下载后 `hashlib.sha256` 校验，不匹配则重下载

### M20. `types.py` + `real_pipeline.py` — num_quality_criteria 无早期校验
- 用户手动设定 `num_quality_criteria`，与实际数据列数不匹配时只有 numpy shape 报错
- **修复**：pipeline load 阶段加 `assert quality_scores.shape[1] == config.num_quality_criteria`

### M21. `quality_directions` 就地取反 — 数据流难追踪
- `self._quality_scores[:, n] = -self._quality_scores[:, n]` 就地修改
- 默认 `rank_normalize` 不受影响（rank 对取反不变），但 `zscore_normalize` 会产出不同值
- **修复**：改为 `negated_scores = self._quality_scores.copy(); negated_scores[:, n] *= -1`

### M22. `run_essential_web_v1.py` — --block-size 默认 64 vs 论文 2048
- help 说 "Full paper: 2048" 但默认 64
- 用户可能误用 demo 配置跑正式实验
- **修复**：默认改为 2048，quick demo 用参数覆盖

### M23. `optimizer.py:606` — RNG API 混用
- `_train_per_task_models` 用 `np.random.RandomState(42)` (旧API)
- 其他地方用 `np.random.default_rng(42)` (新API)
- **修复**：统一为 `default_rng`

### M24. `real_pipeline.py:352` — DatasetSchema() 无参构造必崩溃
- `run()` 默认 `schema=None` → `DatasetSchema()` → `__post_init__` ValueError
- 不可达路径（CLI --schema required），但 API 陷阱
- **修复**：`run()` schema 参数改为必填或去掉默认值

### M25. `batch_sampler.py:100` — selected_texts Python list 构建 OOM 风险
- `[original_texts[i] for i in selected_indices]` 对亿级文档创建新 Python list
- **修复**：流式写入 parquet，不构建中间 list

### M26. `run_essential_web_v1.py` — --checkpoint-steps 已废弃但仍接受
- **修复**：从 argparse 移除

---

## LOW — 小优化或理论边界（未修复）

### L1. `parallel_dispatch.py + tokenize_worker.py` — 重复 tokenizer 缓存定义
### L2. `essential_proxy_runner.py:338-346` — `_memory_cache_add_rows` 不必要的数据拷贝
### L3. `parallel_dispatch.py:451-452` — 验证数据永久占 GPU 内存
### L4. `essential_proxy_runner.py:838-842` — legacy mode 硬编码 quality 列名
### L6. `essential_proxy_runner.py:2031-2048` — 首批等待 900s 无 tokenize 线程存活检查
### L7. `normalization.py:68` — rank_normalize 不产生精确 1.0
### L8. `normalization.py:42,53` — 常量输入归一化静默返回零，无 warning
### L9. `perf_timer.py:34-36` — _timings 无界增长
### L10. `proxy_model.py:49-50,60-61` — rotary embedding 重复切片
### L11. `base.py:58-59` — MD5 截断 12 hex（48 bit）碰撞风险
### L12. `txt_adapter.py:41` — strip() 重复调用
### L13. `proxy_model.py:192-204` — tied lm_head/embed 双重初始化
### L14. `types.py:57-60` — SamplingConfig 参数无范围校验
### L15. `demo_run_quick.sh:206,242` — 显示 10 实验实际 8
### L16. `demo_reoptimize.sh:50,73` — help 文本默认值与实际不符
### L17. `demo_revalidate.sh:58,90` — 默认 npu 但 help 说 cpu
### L18. `demo_run_stem.sh:150` — search-mode 与其他 demo 不一致
### L19. `demo_revalidate.sh:108` — usage 中脚本名错误
### L20. `base.py:45-68` — `from_dataframe` 不保留 quality_scores
### L21. `metadata_manager.py:462` — 不必要的 `pd.read_parquet(columns=[])`
### L23. `types.py:66-69` — SamplingConfig 字段名不含 ~标记，用户误以为是原始值而非重标度值
### L24. `essential_proxy_runner.py:421` — fcntl.flock Unix-only，Windows不兼容
### L25. `pyproject.toml` — 依赖无上限版本(>=X.Y)，numpy 2.0 可能破坏
### L26. `perf_timer.py` — _timings 类级共享状态，多进程只报告主进程
### L27. repo 中 7个PDF + 6个对话日志违反 .gitignore 规则
### L28. `report.py` — 输出中英混合（有意为之，但与其他模块英文风格不一致）

---

## 非问题备注（二次分析确认）

- **noise_level `higher_better=True`**：STEM 数据的 noise_level 已在预处理时翻转（分越高=噪声越低=质量越好），不是 bug
- **--seed 无意义**：多次实验合并拟合需随机参数多样性，固定 seed 反而降低搜索覆盖。已从主入口移除
- **近似 vs 精确 rank**：proxy 实验用10K近似（速度），最终采样用精确rank（质量），是已知设计选择
- **论文 smaller=better vs 代码 higher=better**：全链路自洽，不影响结果

---

## 优先修复计划

### 第一梯队：安全 + 数据正确性
1. **H13** — torch.load weights_only=False（pickle 注入风险）
2. **H14** — README YAML 示例与实际 config 矛盾
3. **H12** — ParameterSampler 除零保护

### 第二梯队：API/配置一致性
4. **M18** — search_weight_mode 默认值矛盾
5. **M24** — DatasetSchema() 无参构造崩溃陷阱
6. **M22** — block-size 默认值与论文矛盾
7. **M26** — 废弃 --checkpoint-steps 移除

### 第三梯队：可维护性
8. **M17** — essential_proxy_runner 拆分（最大工程量）
9. **M19** — VAL_SHA256 校验
10. **M20** — num_quality_criteria 早期校验

### 第四梯队：零测试覆盖（长期）
- 核心算法单元测试（Eq.1-3、Alg.1）
- Pipeline 集成测试
- pytest 框架搭建 + CI 配置
