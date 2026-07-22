#!/usr/bin/env python3
"""
Preprocess essential-web-v1.0: extract domain labels and quality signals,
outputting one parquet per shard (multi-shard mode).

Usage:
  python scripts/preprocess/preprocess_essential_web_v1_sharded.py \
      --input-dir data/essential-web-v1 \
      --output-dir temp/preprocessed
"""

import argparse, json, os, sys, time, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
try:
    import quadmix
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from quadmix.constants import FDC_PREFIX_TO_DOMAIN, FASTTEXT_FIELDS, QUALITY_COLUMNS, PROJECT_DIR

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_QUADMIX_DIR = PROJECT_DIR


def extract_domain_level_2(eai_taxonomy):
    """Extract FDC code prefix and map to 22 L2 domains.

    Returns domain ID (0-21) or -1 to discard (unmapped/invalid FDC code).
    Discards: Other_Languages (43x-49x), FDC code=-1, parse failures.
    """
    if isinstance(eai_taxonomy, str):
        try:
            eai_taxonomy = json.loads(eai_taxonomy)
        except (json.JSONDecodeError, ValueError):
            return -1
    if not isinstance(eai_taxonomy, dict):
        return -1
    fdc = eai_taxonomy.get("free_decimal_correspondence", {})
    if not isinstance(fdc, dict):
        return -1
    primary = fdc.get("primary", {})
    if not isinstance(primary, dict):
        return -1
    code = primary.get("code", "")
    if not isinstance(code, str) or len(code) < 2:
        return -1
    prefix = code[:2]
    return FDC_PREFIX_TO_DOMAIN.get(prefix, -1)


def extract_quality_signals(quality_signals):
    """Extract quality signals from raw fasttext data.

    Raw fasttext scores are "higher = better" (probability/confidence).
    QuaDMix accepts "higher = better" as input convention — no negation needed.
    """
    if not isinstance(quality_signals, dict):
        return np.zeros(len(FASTTEXT_FIELDS), dtype=np.float32)
    fasttext = quality_signals.get("fasttext", {})
    if not isinstance(fasttext, dict):
        return np.zeros(len(FASTTEXT_FIELDS), dtype=np.float32)
    return np.array([
        (fasttext.get(field, 0.0) or 0.0)
        for field in FASTTEXT_FIELDS
    ], dtype=np.float32)


def process_shard(shard_path: str, shard_idx: int, output_dir: str) -> dict:
    """Process one raw shard, save preprocessed version. Returns stats dict."""
    t0 = time.time()
    try:
        df = pd.read_parquet(shard_path,
                             columns=["text", "eai_taxonomy", "quality_signals"])
    except Exception as e:
        print(f"  [{shard_idx:05d}] ERROR: Failed to read {shard_path}: {e}")
        return None
    n = len(df)

    # Extract domain labels
    domains = df["eai_taxonomy"].apply(extract_domain_level_2)

    # Extract quality signals
    quality = df["quality_signals"].apply(extract_quality_signals)
    quality_matrix = np.stack(quality.to_numpy())

    valid_mask = domains.values >= 0
    n_discarded = n - valid_mask.sum()
    if n_discarded > 0:
        df = df[valid_mask]
        domains = domains[valid_mask]
        quality_matrix = quality_matrix[valid_mask]
        n = len(df)

    output = pd.DataFrame({
        "text": df["text"],
        "doc_char_count": df["text"].str.len().to_numpy(dtype=np.int64),
        "domain": domains,
        "shard_idx": np.full(n, shard_idx, dtype=np.int64),
        "row_in_shard": np.arange(n, dtype=np.int64),
        QUALITY_COLUMNS[0]: quality_matrix[:, 0],
        QUALITY_COLUMNS[1]: quality_matrix[:, 1],
        QUALITY_COLUMNS[2]: quality_matrix[:, 2],
        QUALITY_COLUMNS[3]: quality_matrix[:, 3],
        QUALITY_COLUMNS[4]: quality_matrix[:, 4],
    })

    # Save
    out_name = f"preprocessed_{shard_idx:05d}.parquet"
    out_path = os.path.join(output_dir, out_name)
    output.to_parquet(out_path, index=False, row_group_size=1000)

    elapsed = time.time() - t0
    print(f"  [{shard_idx:05d}] {out_name}: {n:,} docs"
          f" ({n_discarded} discarded), {elapsed:.1f}s")

    return {
        "shard_idx": int(shard_idx),
        "file": out_name,
        "path": out_path,
        "num_docs": int(n),
        "num_discarded": int(n_discarded),
        "elapsed_seconds": float(elapsed),
    }


def parse_shard_idx_from_path(shard_path: str) -> int:
    """Extract shard index from filename like 'train-00000-of-03291.parquet'."""
    basename = os.path.basename(shard_path)
    name = basename.replace(".parquet", "")
    if name.startswith("train-") and "-of-" in name:
        return int(name.split("-")[1])
    return -1


def main():
    p = argparse.ArgumentParser(
        description="Preprocess essential-web-v1 in multi-shard mode")
    p.add_argument("--input-dir",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'essential-web'),
                   help="Directory containing raw parquet shards")
    p.add_argument("--output-dir",
                   default=os.path.join(os.path.expanduser("~"), ".cache", "QuaDMix", "temp", "preprocessed"),
                   help="Output directory for preprocessed shards")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of shards to process (for testing)")
    p.add_argument("--force", action="store_true",
                   help="Force reprocess even if output already exists")
    p.add_argument("--workers", type=int, default=128,
                   help="Number of parallel workers (default: 128)")
    args = p.parse_args()

    # --force: clean output dir first to avoid stale/corrupted files
    if args.force and os.path.isdir(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover shards
    shard_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not shard_paths:
        print(f"Error: no parquet files found in {args.input_dir}")
        return 1

    if args.limit:
        shard_paths = shard_paths[:args.limit]

    print(f"Found {len(shard_paths)} shards in {args.input_dir}")

    # ── Incremental: check existing preprocessed files ──
    existing_preprocessed = set()
    if not args.force:
        for fname in os.listdir(args.output_dir):
            if fname.startswith("preprocessed_") and fname.endswith(".parquet"):
                # Parse shard_idx from preprocessed_00000.parquet
                idx_str = fname.replace("preprocessed_", "").replace(".parquet", "")
                try:
                    existing_preprocessed.add(int(idx_str))
                except ValueError:
                    pass

    if existing_preprocessed and not args.force:
        print(f"  Already preprocessed: {len(existing_preprocessed)} shards")

    # ── Detect missing shards (deleted mid-shards) ──
    needed_shard_indices = set()
    for sp in shard_paths:
        shard_idx = parse_shard_idx_from_path(sp)
        if shard_idx >= 0:
            needed_shard_indices.add(shard_idx)

    missing_shards = needed_shard_indices - existing_preprocessed
    if missing_shards and not args.force:
        print(f"  Missing {len(missing_shards)} shards (will reprocess): {sorted(missing_shards)[:5]}{'...' if len(missing_shards) > 5 else ''}")

    # Process each shard with ORIGINAL shard_idx (from filename)
    t_start = time.time()
    shard_index = []
    skipped = 0

    # Separate shards to process vs skip
    to_process = []
    for sp in shard_paths:
        shard_idx = parse_shard_idx_from_path(sp)
        if shard_idx < 0:
            shard_idx = len(shard_index)

        # Skip if already processed (incremental mode)
        if shard_idx in existing_preprocessed and shard_idx not in missing_shards and not args.force:
            out_name = f"preprocessed_{shard_idx:05d}.parquet"
            out_path = os.path.join(args.output_dir, out_name)
            try:
                df = pd.read_parquet(out_path, columns=["domain"])
                skipped += 1
                n = len(df)
                shard_index.append({
                    "shard_idx": int(shard_idx),
                    "file": out_name,
                    "path": out_path,
                    "num_docs": int(n),
                    "num_discarded": 0,
                    "elapsed_seconds": 0.0,
                })
                continue
            except Exception:
                # Corrupted preprocessed file — reprocess
                print(f"  [WARN] Corrupted: {out_name}, will reprocess")

        to_process.append((sp, shard_idx, args.output_dir))

    # Parallel processing
    workers = min(args.workers, len(to_process)) if to_process else 1
    failed_shards = []
    if len(to_process) > 1:
        print(f"  Processing {len(to_process)} shards with {workers} workers (process pool)...")
        t_proc_start = time.time()
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_shard, sp, si, odir): (sp, si)
                       for sp, si, odir in to_process}
            for future in as_completed(futures):
                stats = future.result()
                if stats is None:
                    sp, si = futures[future]
                    failed_shards.append((si, sp))
                else:
                    shard_index.append(stats)
                completed += 1
                # Print progress every 10 shards or at completion
                if completed % 10 == 0 or completed == len(to_process):
                    elapsed = time.time() - t_proc_start
                    speed = completed / elapsed if elapsed > 0 else 0
                    eta = (len(to_process) - completed) / speed if speed > 0 else 0
                    print(f"  [Preprocess Progress] {completed}/{len(to_process)} shards "
                          f"({completed*100//len(to_process)}%), "
                          f"{speed:.1f} shards/s, ETA {eta:.0f}s")
    else:
        for sp, shard_idx, odir in to_process:
            stats = process_shard(sp, shard_idx, odir)
            if stats is None:
                failed_shards.append((shard_idx, sp))
            else:
                shard_index.append(stats)

    # Sort by shard_idx for consistent output
    shard_index.sort(key=lambda s: s["shard_idx"])

    if skipped > 0:
        print(f"  Skipped {skipped} already-preprocessed shards")
    
    if failed_shards:
        print(f"  Failed {len(failed_shards)} shards (corrupted or unreadable):")
        for si, sp in failed_shards[:10]:
            print(f"    [{si:05d}] {sp}")
        if len(failed_shards) > 10:
            print(f"    ... and {len(failed_shards) - 10} more")

    # Save shard index
    index_path = os.path.join(args.output_dir, "shard_index.json")
    total_docs = sum(s["num_docs"] for s in shard_index)
    total_discarded = sum(s["num_discarded"] for s in shard_index)
    index_data = {
        "num_shards": len(shard_index),
        "total_docs": total_docs,
        "total_discarded": total_discarded,
        "shards": shard_index,
    }
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Preprocessing complete!")
    print(f"  Shards:     {len(shard_index)}")
    print(f"  Total docs: {total_docs:,}")
    print(f"  Discarded:  {total_discarded:,} ({total_discarded/total_docs*100:.1f}%)")
    print(f"  Index:      {index_path}")
    print(f"  Duration:   {elapsed:.1f}s")
    print(f"  Output:     {args.output_dir}/")
    print(f"  [Next] Use --preprocessed-dir {args.output_dir} for run script")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())