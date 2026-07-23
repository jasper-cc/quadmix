#!/usr/bin/env python3
"""
Run QuaDMix on essential-web-v1.0 with multi-shard preprocessing.

Loads preprocessed shards via ShardMetadataManager → skips feature extraction →
quality ranks → parameter sampling → real proxy experiments (on-demand text) →
LightGBM regression → optimal search → final sampling → save outputs.

Usage:
  # Quick validation (200 experiments, fast)
  python scripts/runners/run_essential_web_v1.py --quick --preprocessed-dir .../preprocessed

  # Full production (3000 experiments, needs GPU)
  python scripts/runners/run_essential_web_v1.py --full --preprocessed-dir .../preprocessed

  # Or use the auto-detected defaults:
  python scripts/runners/run_essential_web_v1.py --quick

The validation set (openhermes_10k_assistant_tokenized.pt) will be
automatically downloaded from HuggingFace if not found locally.
Alternatively, use --val-set=core to use a CORE benchmark-based
validation set (22 tasks, continuation-only loss), --val-set=core_bmk_v2
to use the BMK v2 set (10 BMK-like tasks, full-sequence loss), or
--val-set=core_bmk_v3 to use the BMK v3 set (10 tasks selected by
1M proxy learnability, full-sequence loss).
"""

import argparse, os, sys, time, urllib.request, ssl
_cpu_count = os.cpu_count() or 4
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(max(1, _cpu_count // 4)))
os.environ.setdefault('OMP_NUM_THREADS', str(max(1, _cpu_count // 4)))
os.environ.setdefault('MKL_NUM_THREADS', str(max(1, _cpu_count // 4)))
os.environ.setdefault('NUMEXPR_NUM_THREADS', str(max(1, _cpu_count // 4)))
os.environ.setdefault('RAYON_NUM_THREADS', '4')
try:
    import quadmix
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from quadmix import QuaDMixConfig
from quadmix.pipeline.real_pipeline import QuaDMixPipeline
from quadmix.data.metadata_manager import ShardMetadataManager
from quadmix.constants import (
    PROJECT_DIR, DEFAULT_TEMP_DIR, DEFAULT_PREPROCESSED_DIR,
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
DEFAULT_VAL_PATH = os.path.join(DEFAULT_VAL_DIR, HF_OPENHERMES_FILENAME)


def _hf_remote_size(repo_id: str, filename: str) -> int:
    """Get remote file size from HuggingFace via HEAD request. Returns 0 if failed."""
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
    """Download a file from HuggingFace. Returns True on success."""
    url = HF_RESOLVE.format(repo=repo_id, file=filename)
    if HF_ENDPOINT != "https://huggingface.co":
        print(f"[Setup] Using HF mirror: {HF_ENDPOINT}")
    print(f"[Setup] Downloading from:\n  {url}")
    try:
        urllib.request.urlretrieve(url, local_path)
        size_mb = os.path.getsize(local_path) / 1024**2
        print(f"[Setup] Downloaded: {local_path} ({size_mb:.0f} MB)")
        return True
    except Exception as e:
        print(f"[Setup] Download failed: {e}")
        return False


def _ensure_hf_data(val_path: str, hf_dataset: str, hf_filename: str,
                    label: str = "", eval_bundle: str = None,
                    prepare_script_name: str = None) -> str:
    """Generic: check local size → check remote → download/generate → return path.

    Args:
        val_path: Local path to store the file.
        hf_dataset: HuggingFace dataset repo ID.
        hf_filename: HuggingFace file name within the dataset.
        label: Human-readable label (e.g. "CORE-BMK v5") for log messages.
        eval_bundle: If provided, fall back to local generation via eval_bundle
                     when download fails or remote is unreachable.
        prepare_script_name: Filename of the prepare script (e.g. "prepare_core_bmk_v5.py")
                             located in scripts/validation_set/. Required if eval_bundle is set.
    """
    os.makedirs(os.path.dirname(val_path), exist_ok=True)

    local_size = os.path.getsize(val_path) if os.path.exists(val_path) else 0
    remote_size = _hf_remote_size(hf_dataset, hf_filename)
    display_label = label or hf_filename

    if remote_size == 0:
        if local_size > 0:
            print(f"\n[Setup] Warning: Cannot connect to HuggingFace to check {hf_filename}")
            print(f"[Setup] Using local file ({local_size / 1024**2:.0f} MB)")
            print(f"[Setup] To force re-download, delete: {val_path}")
            return val_path
        if not eval_bundle:
            raise RuntimeError(
                f"Cannot connect to HuggingFace and no local file.\n"
                f"  Please download manually from:\n"
                f"    https://huggingface.co/datasets/{hf_dataset}\n"
                f"  And place at: {val_path}"
            )
    elif local_size == remote_size:
        print(f"[Setup] {display_label} validation set up-to-date: {hf_filename} ({local_size / 1024**2:.0f} MB)")
        return val_path
    elif local_size > 0:
        print(f"\n[Setup] {hf_filename} version mismatch")
        print(f"[Setup]   Local:  {local_size / 1024**2:.0f} MB")
        print(f"[Setup]   Remote: {remote_size / 1024**2:.0f} MB")
        print(f"[Setup]   Re-downloading...")
        os.remove(val_path)
    else:
        print(f"\n[Setup] {display_label} validation set not found at:\n  {val_path}")

    if remote_size > 0:
        if _download_hf_file(hf_dataset, hf_filename, val_path):
            return val_path
        if eval_bundle:
            print(f"[Setup] Falling back to local generation...")
        else:
            raise RuntimeError(
                f"Failed to download {display_label} validation set.\n"
                f"  You can manually download from:\n"
                f"    https://huggingface.co/datasets/{hf_dataset}\n"
                f"  Or place the file at: {val_path}"
            )

    if eval_bundle and prepare_script_name:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prepare_script = os.path.join(script_dir, "..", "validation_set", prepare_script_name)

        if not os.path.exists(prepare_script):
            raise FileNotFoundError(
                f"{display_label} validation set not found at:\n  {val_path}\n"
                f"Preparation script also missing:\n  {prepare_script}\n"
                f"You can manually download from:\n"
                f"  https://huggingface.co/datasets/{hf_dataset}"
            )

        print(f"[Setup] Auto-generating from eval_bundle: {eval_bundle}")

        import subprocess
        result = subprocess.run(
            [
                sys.executable, prepare_script,
                "--output-dir", os.path.dirname(val_path),
                "--eval-bundle", eval_bundle,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to generate {display_label} validation set.\n"
                f"  stdout: {result.stdout}\n"
                f"  stderr: {result.stderr}\n"
                f"You can manually download from:\n"
                f"  https://huggingface.co/datasets/{hf_dataset}"
            )
        print(result.stdout)
        if not os.path.exists(val_path):
            raise RuntimeError(
                f"{display_label} validation set generation completed but file not found:\n  {val_path}"
            )
        size_mb = os.path.getsize(val_path) / 1024**2
        print(f"[Setup] Generated: {val_path} ({size_mb:.0f} MB)")
        return val_path

    raise FileNotFoundError(
        f"{display_label} validation set not available:\n  {val_path}\n"
        f"Download from: https://huggingface.co/datasets/{hf_dataset}"
    )


def ensure_val_data(val_path: str) -> str:
    return _ensure_hf_data(val_path, HF_OPENHERMES_DATASET, HF_OPENHERMES_FILENAME)

def ensure_core_val_data(val_path: str, eval_bundle: str) -> str:
    return _ensure_hf_data(val_path, HF_CORE_DATASET, HF_CORE_FILENAME,
                           label="CORE", eval_bundle=eval_bundle,
                           prepare_script_name="prepare_core_val_set.py")

def ensure_core_bmk_v2_data(val_path: str, eval_bundle: str) -> str:
    return _ensure_hf_data(val_path, HF_CORE_BMK_V2_DATASET, HF_CORE_BMK_V2_FILENAME,
                           label="CORE-BMK v2", eval_bundle=eval_bundle,
                           prepare_script_name="prepare_core_bmk_v2.py")

def ensure_core_bmk_v3_data(val_path: str, eval_bundle: str) -> str:
    return _ensure_hf_data(val_path, HF_CORE_BMK_V3_DATASET, HF_CORE_BMK_V3_FILENAME,
                           label="CORE-BMK v3", eval_bundle=eval_bundle,
                           prepare_script_name="prepare_core_bmk_v3.py")

def ensure_core_bmk_v4_data(val_path: str, eval_bundle: str) -> str:
    return _ensure_hf_data(val_path, HF_CORE_BMK_V4_DATASET, HF_CORE_BMK_V4_FILENAME,
                           label="CORE-BMK v4", eval_bundle=eval_bundle,
                           prepare_script_name="prepare_core_bmk_v4.py")

def ensure_core_bmk_v42_data(val_path: str, eval_bundle: str) -> str:
    return _ensure_hf_data(val_path, HF_CORE_BMK_V42_DATASET, HF_CORE_BMK_V42_FILENAME,
                           label="CORE-BMK v4.2", eval_bundle=eval_bundle,
                           prepare_script_name="prepare_core_bmk_v4.2.py")

def ensure_core_bmk_v43_data(val_path: str, eval_bundle: str) -> str:
    return _ensure_hf_data(val_path, HF_CORE_BMK_V43_DATASET, HF_CORE_BMK_V43_FILENAME,
                           label="CORE-BMK v4.3", eval_bundle=eval_bundle,
                           prepare_script_name="prepare_core_bmk_v4.3.py")

def ensure_core_bmk_v5_data(val_path: str, eval_bundle: str) -> str:
    return _ensure_hf_data(val_path, HF_CORE_BMK_V5_DATASET, HF_CORE_BMK_V5_FILENAME,
                           label="CORE-BMK v5", eval_bundle=eval_bundle,
                           prepare_script_name="prepare_core_bmk_v5.py")

def ensure_core_bmk_v6_data(val_path: str, eval_bundle: str) -> str:
    return _ensure_hf_data(val_path, HF_CORE_BMK_V6_DATASET, HF_CORE_BMK_V6_FILENAME,
                           label="CORE-BMK v6", eval_bundle=eval_bundle,
                           prepare_script_name="prepare_core_bmk_v6.py")

def ensure_cap_v1_data(val_path: str) -> str:
    return _ensure_hf_data(val_path, HF_CAP_V1_DATASET, HF_CAP_V1_FILENAME,
                           label="CAP v1")

def ensure_stem_v1_data(val_path: str) -> str:
    return _ensure_hf_data(val_path, HF_STEM_V1_DATASET, HF_STEM_V1_FILENAME,
                           label="STEM v1")

from quadmix.data.dataset_schema import DatasetSchema


def build_parser():
    p = argparse.ArgumentParser(description="QuaDMix on essential-web-v1 (sharded mode)")
    p.add_argument("--preprocessed-dir", default=DEFAULT_PREPROCESSED_DIR,
                   help="Directory of preprocessed parquet shards")
    p.add_argument("--shard-limit", type=int, default=None,
                   help="Use only the first N shards (smoke testing)")
    p.add_argument("--quick", action="store_true", help="Quick: 200 exp, 2000 search")
    p.add_argument("--full", action="store_true", help="Full: 3000 exp, 100K search")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--num-experiments", type=int, default=None)
    p.add_argument("--num-search", type=int, default=None)
    # Note: --doc-limit removed. Proxy experiments should use full data pool.
    # Use --target-tokens to control final output size instead.
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--target-tokens", type=float, default=0.0,
                   help="Desired max tokens in output (in B). θ* may produce less; "
                        "if more, uniformly discard to preserve distribution. "
                        "Paper: 'more tokens not always good' (30B > 90B > 180B)")
    p.add_argument("--search-mode", default="r2_weighted",
                   choices=["r2_weighted", "equal_weight", "r2_sigma_weighted"],
                   help="Search weighting mode (default: r2_weighted)")
    p.add_argument("--device-type", default="cpu",
                   choices=["cpu", "cuda", "npu"],
                   help="Device type for proxy model training")
    p.add_argument("--npu-devices", type=int, default=1,
                   help="Number of NPU devices (1=sequential, 8=8-way parallel)")

    # Proxy model config
    p.add_argument("--block-size", type=int, default=2048,
                   help="Sequence length (tokens). Paper default: 2048")
    p.add_argument("--tiny-steps", type=int, default=3,
                   help="Proxy training steps (0 = use max_step=25000)")
    p.add_argument("--micro-batch-size", type=int, default=2,
                   help="Micro batch size per forward pass")
    p.add_argument("--global-batch-size", type=int, default=8,
                   help="Effective global batch size (grad acc = global/micro)")
    p.add_argument("--rank-ref-size", type=int, default=10000,
                   help="Reference subset size for rank estimation")
    p.add_argument("--checkpoint-interval", type=int, default=1000,
                   help="Record val_loss every N steps during proxy training "
                        "(default: 1000, 0 = disable). "
                        "Results saved to each exp dir as checkpoint_trajectory.json.")
    p.add_argument("--val-set", type=str, default="openhermes",
                   choices=["openhermes", "core", "core_bmk_v2", "core_bmk_v3", "core_bmk_v4", "core_bmk_v4.2", "core_bmk_v4.3", "core_bmk_v5", "core_bmk_v6", "cap_v1", "stem_v1"],
                   help="Validation set: 'openhermes' (default, auto-download), "
                        "'core' (CORE benchmark 22-task, continuation-only loss), "
                        "'core_bmk_v2' (10 BMK-like tasks, full-sequence loss), "
                        "'core_bmk_v3' (10 tasks selected by 1M proxy learnability), "
                        "'core_bmk_v4' (continuation=full-seq, QA=answer-only mask), "
                        "'core_bmk_v4.2' (21 tasks, per-task loss strategy), "
                        "'core_bmk_v4.3' (21 tasks, piqa/arc moved to full-seq), "
                        "'core_bmk_v5' (21 tasks, HF-loaded, per-task hybrid, data quality fixes), "
                        "'core_bmk_v6' (21 tasks, HellaSwag tags cleaned, MC format fixed), "
                        "'cap_v1' (5 capability clusters, 70% external + 30% benchmark train), or "
                        "'stem_v1' (4 STEM benchmarks: GSM8K/MMLU-22STEM/ARC-Easy/ARC-Challenge)")
    p.add_argument("--val-path", type=str, default=None,
                   help="Path to validation .pt file (overrides --val-set)")
    p.add_argument("--eval-bundle", type=str, default=DEFAULT_EVAL_BUNDLE,
                   help="Path to CORE eval bundle directory "
                        "(used when --val-set=core, default: $EVAL_BUNDLE_DIR)")
    p.add_argument("--schema", type=str, required=True,
                   help="YAML dataset schema config file (必填)。"
                        "示例数据集: configs/schema_essential_web.yaml")
    p.add_argument("--seed", type=int, default=None,
                   help="Pipeline seed for reproducibility. "
                        "None (default): each run is non-deterministic (OS entropy), "
                        "so different executions produce diverse experiments. "
                        "Set an int (e.g. 42) for fully reproducible results. "
                        "Merging runs with different seeds increases regression coverage.")
    return p


def create_proxy_runner(config, args, output_dir, metadata_manager):
    """Create an EssentialWebProxyRunner with sharded metadata manager."""
    from quadmix.pipeline.essential_proxy_runner import EssentialWebProxyRunner
    proxy_dir = os.path.join(output_dir, "proxy_experiments")

    # Resolve validation path
    if args.val_path:
        val_path = args.val_path
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation file not found: {val_path}")
    elif args.val_set == "core":
        val_path = os.path.join(DEFAULT_VAL_DIR, HF_CORE_FILENAME)
        val_path = ensure_core_val_data(val_path, args.eval_bundle)
    elif args.val_set == "core_bmk_v2":
        val_path = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V2_FILENAME)
        val_path = ensure_core_bmk_v2_data(val_path, args.eval_bundle)
    elif args.val_set == "core_bmk_v3":
        val_path = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V3_FILENAME)
        val_path = ensure_core_bmk_v3_data(val_path, args.eval_bundle)
    elif args.val_set == "core_bmk_v4":
        val_path = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V4_FILENAME)
        val_path = ensure_core_bmk_v4_data(val_path, args.eval_bundle)
    elif args.val_set == "core_bmk_v4.2":
        val_path = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V42_FILENAME)
        val_path = ensure_core_bmk_v42_data(val_path, args.eval_bundle)
    elif args.val_set == "core_bmk_v4.3":
        val_path = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V43_FILENAME)
        val_path = ensure_core_bmk_v43_data(val_path, args.eval_bundle)
    elif args.val_set == "core_bmk_v5":
        val_path = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V5_FILENAME)
        val_path = ensure_core_bmk_v5_data(val_path, args.eval_bundle)
    elif args.val_set == "core_bmk_v6":
        val_path = os.path.join(DEFAULT_VAL_DIR, HF_CORE_BMK_V6_FILENAME)
        val_path = ensure_core_bmk_v6_data(val_path, args.eval_bundle)
    elif args.val_set == "cap_v1":
        val_path = os.path.join(DEFAULT_VAL_DIR, HF_CAP_V1_FILENAME)
        val_path = ensure_cap_v1_data(val_path)
    elif args.val_set == "stem_v1":
        val_path = os.path.join(DEFAULT_VAL_DIR, HF_STEM_V1_FILENAME)
        val_path = ensure_stem_v1_data(val_path)
    else:
        val_path = os.path.join(DEFAULT_VAL_DIR, HF_OPENHERMES_FILENAME)
        val_path = ensure_val_data(val_path)

    # Parse checkpoint interval
    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval is not None else 1000

    runner = EssentialWebProxyRunner(
        config=config,
        metadata_manager=metadata_manager,
        val_data_path=val_path,
        output_dir=proxy_dir,
        device_type=args.device_type,
        npu_device_id=0,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        tiny_steps=args.tiny_steps,
        doc_limit=None,
        test_block_size=args.block_size,
        rank_ref_size=args.rank_ref_size,
        token_cache_dir=os.path.join(QUADMIX_TEMP_DIR, "token_cache"),
        checkpoint_interval=checkpoint_interval,
        domain_names=metadata_manager.detected_domain_names,
        quality_names=metadata_manager.detected_quality_names,
        quality_directions=metadata_manager.quality_directions,
    )
    return runner


def main():
    args = build_parser().parse_args()

    if args.quick:
        n_exp, n_search, top_k = 200, 2000, args.top_k
    elif args.full:
        n_exp, n_search, top_k = 3000, 100000, args.top_k
    else:
        n_exp = args.num_experiments or 500
        n_search = args.num_search or 5000
        top_k = args.top_k

    output_dir = args.output or os.path.join(
        QUADMIX_DIR, f"result/quadmix_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    # ── Load dataset schema ──
    print(f"\n[Setup] Loading dataset schema from: {args.schema}")
    schema = DatasetSchema.from_yaml(args.schema)
    print(f"[Setup] Schema: domain_col='{schema.domain_col}', "
          f"quality_cols={schema.quality_cols}, "
          f"text_col='{schema.text_col}', "
          f"char_count_col={schema.char_count_col}")

    # ── Load metadata manager (reads only domain + quality from all shards) ──
    print(f"\n[Setup] Loading ShardMetadataManager from: {args.preprocessed_dir}")
    metadata_manager = ShardMetadataManager(
        args.preprocessed_dir,
        schema=schema,
        shard_limit=args.shard_limit,
        max_workers=int(os.environ.get("STEM_METADATA_WORKERS", "8")),
    )
    print(f"[Setup] {metadata_manager.num_docs:,} docs across "
          f"{metadata_manager.num_shards} shards, "
          f"{metadata_manager.num_domains} domains, "
          f"{metadata_manager.num_quality_criteria} quality criteria")

    # ── Dataset size estimation ──
    total_tokens_est = metadata_manager.get_total_tokens_estimate()
    total_chars = metadata_manager.get_total_chars()
    print(f"\n════════════════════════════════════════════════════════")
    print(f"  数据集信息:")
    print(f"    总文档数:         {metadata_manager.num_docs:,}")
    print(f"    总字符数:         {total_chars:,}")
    print(f"    估算 token 数:    {total_tokens_est:,} ({total_tokens_est/1e9:.1f}B)")
    print(f"    (按 4 chars/token 估算，GPT-NeoX tokenizer)")
    print(f"════════════════════════════════════════════════════════")

    # ── omega range guidance ──
    # Paper: ω sampled from [0,1] → rescaled to [0, 0.1]
    # Simplified: sampling_ratio ≈ omega
    omega_min, omega_max = 0.0, 0.1  # Paper default range (after rescaling)
    tokens_min = int(total_tokens_est * omega_min)
    tokens_max = int(total_tokens_est * omega_max)

    print(f"\n  omega (ω) 参数与数据量估算:")
    print(f"    论文默认范围:     [{omega_min:.2f}, {omega_max:.2f}]")
    print(f"    预计数据量范围:   {tokens_min/1e9:.1f}B - {tokens_max/1e9:.1f}B tokens")
    print(f"    (简化估算: sampling_ratio ≈ ω，忽略 sigmoid 形态)")

    if args.target_tokens > 0:
        target_b = args.target_tokens
        print(f"\n  你设置的 target:   {target_b:.1f}B tokens")
        # Rough estimate: omega ≈ target / total (for reference only)
        target_omega_est = target_b * 1e9 / total_tokens_est
        print(f"    参考信息: 若 ω ≈ {target_omega_est:.3f}，可能产生约 {target_b:.1f}B 数据")
        print(f"    注意: 系统会全范围搜索最优参数，不限制 omega 范围")
        if target_omega_est > 0.1:
            print(f"    [提示] target {target_b:.1f}B 对应 ω > 0.1，可能超出论文推荐范围")
    print(f"════════════════════════════════════════════════════════")

    config = QuaDMixConfig(
        num_domains=metadata_manager.num_domains,
        num_quality_criteria=metadata_manager.num_quality_criteria,
        num_proxy_experiments=n_exp, num_search_points=n_search,
        top_k_average=top_k,
        target_tokens=int(args.target_tokens * 1e9) if args.target_tokens > 0 else 0,
        search_weight_mode=args.search_mode,
        seed=args.seed,
    )

    pipeline = QuaDMixPipeline(config)
    proxy_runner = create_proxy_runner(config, args, output_dir, metadata_manager)

    print(f"\n[Setup] Proxy runner: {n_exp} experiments, "
         f"{args.tiny_steps} steps each, "
         f"val={args.val_set}"
         f"{', ' + str(args.npu_devices) + ' NPU devices' if args.npu_devices > 1 else ''}")

    domain_names = metadata_manager.detected_domain_names
    quality_names = metadata_manager.detected_quality_names
    quality_directions = metadata_manager.quality_directions

    schema = metadata_manager.schema

    pipeline.run(
        data_path=args.preprocessed_dir,
        output_dir=output_dir,
        precomputed=True,
        metadata_manager=metadata_manager,
        doc_limit=None,
        domain_names=domain_names,
        quality_names=quality_names,
        quality_directions=quality_directions,
        proxy_runner=proxy_runner,
        parallel_workers=args.npu_devices,
        val_set=args.val_set,
        schema=schema,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
