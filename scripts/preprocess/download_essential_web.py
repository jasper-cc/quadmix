#!/usr/bin/env python3
"""
Download Essential-Web v1.0 parquet shards from Hugging Face.

Usage:
    python scripts/preprocess/download_essential_web.py [--num-files N] 
        [--output-dir DIR] [--crawl CRAWL_NAME]

Downloads N shards (default 50) from the specified CommonCrawl dump
into the output directory. Supports parallel downloads and HF mirror.

Examples:
    # Download 50 files (default) to data/essential-web/
    python scripts/preprocess/download_essential_web.py

    # Download just 2 files (for testing)
    python scripts/preprocess/download_essential_web.py --num-files 2

    # Custom output with 8 parallel workers
    python scripts/preprocess/download_essential_web.py \
        --output-dir ./data/essential-web \
        --num-files 50 \
        --workers 8

    # Use HF mirror (for users in China)
    HF_ENDPOINT=https://hf-mirror.com python scripts/preprocess/download_essential_web.py
"""

import argparse
import os
import ssl
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# ---------------------------------------------------------------------------
REPO_ID = "EssentialAI/essential-web-v1.0"
DEFAULT_CRAWL = "CC-MAIN-2024-38"
DEFAULT_NUM_FILES = 50
DEFAULT_WORKERS = 16
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'essential-web')

# Support HF mirror via environment variable
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HF_API_TREE = f"{HF_ENDPOINT}/api/datasets/{{repo}}/tree/main/{{path}}"
HF_RESOLVE = f"{HF_ENDPOINT}/datasets/{{repo}}/resolve/main/{{path}}?download=true"


def list_crawl_files(repo: str, crawl: str) -> list[dict]:
    """List all parquet files in a given crawl directory via the HF tree API."""
    import urllib.request
    import urllib.error

    path = f"data/crawl={crawl}"
    url = HF_API_TREE.format(repo=repo, path=path)
    parts = []

    cursor = None
    while True:
        api_url = url + (f"?cursor={cursor}" if cursor else "")
        req = urllib.request.Request(api_url)
        try:
            with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"[Error] Failed to list files: {e}")
            sys.exit(1)

        for item in data:
            if item["type"] == "file" and item["path"].endswith(".parquet"):
                parts.append(item)

        # Check for pagination
        link = resp.headers.get("Link", "") if cursor is not None else ""
        if link and 'rel="next"' in link:
            cursor = resp.headers.get("X-Cursor") or (data and data[-1].get("path"))
        else:
            break

    return parts


def format_size(mb: float) -> str:
    if mb < 1024:
        return f"{mb:.0f} MB"
    return f"{mb / 1024:.1f} GB"


def get_file_info(url: str) -> tuple[bool, int]:
    """Get file info via HEAD request. Returns (supports_range, real_size)."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
            accept_ranges = resp.headers.get("Accept-Ranges", "")
            content_length = resp.headers.get("Content-Length", "0")
            supports_range = "bytes" in accept_ranges.lower()
            real_size = int(content_length) if content_length.isdigit() else 0
            return supports_range, real_size
    except Exception:
        return False, 0


def download_file(url: str, dest: str, index: int, total: int, size_mb: float,
                  expected_size: int = None, show_progress: bool = True) -> tuple[bool, str]:
    """Download a single file with resume support. Returns (success, filename)."""
    import urllib.request
    import urllib.error
    import shutil

    basename = os.path.basename(dest)

    # Get real file info via HEAD request
    supports_range, real_size = get_file_info(url)

    # Check local file against real size from server
    existing_size = 0
    if os.path.exists(dest):
        existing_size = os.path.getsize(dest)

        if real_size > 0:
            if existing_size == real_size:
                # File is complete (verified against server)
                if show_progress:
                    print(
                        f"  [{index + 1}/{total}] {basename} (complete, verified {existing_size / (1024 * 1024):.0f}MB)")
                return True, basename
            elif existing_size > real_size:
                # Local file larger than server (corrupted?), re-download
                if show_progress:
                    print(
                        f"  [{index + 1}/{total}] {basename} (size mismatch: {existing_size} > {real_size}, re-download) ... ",
                        end="", flush=True)
                existing_size = 0  # Force full download
            elif existing_size > 0 and supports_range:
                # Partial file, server supports resume
                if show_progress:
                    print(
                        f"  [{index + 1}/{total}] {basename} (resume from {existing_size / (1024 * 1024):.0f}MB/{real_size / (1024 * 1024):.0f}MB) ... ",
                        end="", flush=True)
                # Will use Range header
            elif existing_size > 0:
                # Partial file, server does NOT support resume
                if show_progress:
                    print(f"  [{index + 1}/{total}] {basename} ({size_mb:.0f}MB, server no Range, re-download) ... ",
                          end="", flush=True)
                existing_size = 0  # Force full download
            else:
                if show_progress:
                    print(f"  [{index + 1}/{total}] {basename} ({size_mb:.0f}MB) ... ", end="", flush=True)
        else:
            # Could not get real size, proceed with download
            if show_progress:
                print(f"  [{index + 1}/{total}] {basename} ({size_mb:.0f}MB) ... ", end="", flush=True)
    else:
        if show_progress:
            print(f"  [{index + 1}/{total}] {basename} ({size_mb:.0f}MB) ... ", end="", flush=True)

    try:
        # Open file in append mode if resuming
        mode = "ab" if existing_size > 0 else "wb"

        # Build request with Range header for resume
        req = urllib.request.Request(url)
        if existing_size > 0:
            req.add_header("Range", f"bytes={existing_size}-")

        with open(dest, mode) as f:
            with urllib.request.urlopen(req, timeout=300, context=SSL_CTX) as resp:
                # Read in chunks to avoid memory issues
                chunk_size = 8 * 1024 * 1024  # 8MB chunks
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)

        actual = os.path.getsize(dest) / (1024 * 1024)
        if show_progress:
            if existing_size > 0:
                print(f"resumed (+{actual - existing_size / (1024 * 1024):.0f}MB → {actual:.0f}MB)")
            else:
                print(f"done ({actual:.0f}MB)")
        return True, basename

    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range Not Satisfiable
            # Verify with HEAD request
            _, head_size = get_file_info(url)
            actual_size = os.path.getsize(dest) if os.path.exists(dest) else 0

            if head_size > 0 and actual_size == head_size:
                # File is actually complete
                if show_progress:
                    print(f"done (416 but verified complete, {actual_size / (1024 * 1024):.0f}MB)")
                return True, basename
            else:
                # Re-download to temp file
                if show_progress:
                    print(f"416, re-downloading...")
                temp_dest = dest + ".tmp"
                try:
                    req = urllib.request.Request(url)
                    with open(temp_dest, "wb") as f:
                        with urllib.request.urlopen(req, timeout=300, context=SSL_CTX) as resp:
                            chunk_size = 8 * 1024 * 1024
                            while True:
                                chunk = resp.read(chunk_size)
                                if not chunk:
                                    break
                                f.write(chunk)
                    shutil.move(temp_dest, dest)
                    actual = os.path.getsize(dest) / (1024 * 1024)
                    if show_progress:
                        print(f"done ({actual:.0f}MB)")
                    return True, basename
                except Exception as e2:
                    if show_progress:
                        print(f"FAILED: {e2}")
                    if os.path.exists(temp_dest):
                        os.remove(temp_dest)
                    return False, basename
        else:
            if show_progress:
                print(f"FAILED: {e}")
            return False, basename

    except (urllib.error.URLError, OSError) as e:
        if show_progress:
            print(f"FAILED: {e}")
        return False, basename


def download_file_hf_transfer(url: str, dest: str, index: int, total: int, size_mb: float,
                              expected_size: int = None, show_progress: bool = True) -> tuple[bool, str]:
    """Download using hf_transfer. Note: doesn't support resume, falls back to download_file."""
    import urllib.request
    import urllib.error

    basename = os.path.basename(dest)

    # Check for partial file - hf_transfer doesn't support resume, fall back to regular download
    if os.path.exists(dest) and expected_size is not None:
        existing_size = os.path.getsize(dest)
        if existing_size > 0 and existing_size < expected_size:
            # Partial file exists, use regular download for resume support
            return download_file(url, dest, index, total, size_mb,
                                 expected_size=expected_size, show_progress=show_progress)

    try:
        import hf_transfer

        if show_progress:
            print(f"  [{index + 1}/{total}] {basename} ({size_mb:.0f} MB) [hf_transfer] ... ", end="", flush=True)

        # Read the file
        with urllib.request.urlopen(url, context=SSL_CTX) as response:
            data = response.read()

        with open(dest, "wb") as f:
            f.write(data)

        actual = os.path.getsize(dest) / (1024 * 1024)
        if show_progress:
            print(f"done ({actual:.0f} MB)")
        return True, basename
    except ImportError:
        # Fall back to regular download (with resume support)
        return download_file(url, dest, index, total, size_mb,
                             expected_size=expected_size, show_progress=show_progress)
    except Exception as e:
        if show_progress:
            print(f"FAILED: {e}")
        return False, basename


def main():
    parser = argparse.ArgumentParser(
        description="Download Essential-Web v1.0 parquet shards from Hugging Face."
    )
    parser.add_argument("--num-files", type=int, default=DEFAULT_NUM_FILES,
                        help=f"Number of parquet files to download (default: {DEFAULT_NUM_FILES})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--crawl", type=str, default=DEFAULT_CRAWL,
                        help=f"CommonCrawl dump name (default: {DEFAULT_CRAWL})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel download workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--use-hf-transfer", action="store_true",
                        help="Use hf_transfer for faster downloads (requires: pip install hf-transfer)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start downloading from this file index (default: 0)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Check for hf_transfer
    use_hf_transfer = args.use_hf_transfer
    if use_hf_transfer:
        try:
            import hf_transfer
        except ImportError:
            print("[Warning] hf_transfer not installed. Install with: pip install hf-transfer")
            print("[Warning] Falling back to parallel urllib download.")
            use_hf_transfer = False

    # ── Step 1: List files on HF ──
    print(f"[1/3] Listing parquet files in {args.crawl}...")
    if HF_ENDPOINT != "https://huggingface.co":
        print(f"  Using HF mirror: {HF_ENDPOINT}")
    files = list_crawl_files(REPO_ID, args.crawl)
    if not files:
        print(f"[Error] No parquet files found in crawl '{args.crawl}'")
        print(f"  Check: {HF_API_TREE.format(repo=REPO_ID, path=f'data/crawl={args.crawl}')}")
        sys.exit(1)

    # Slice to requested range
    start_idx = args.start_index
    if start_idx > 0:
        files = files[start_idx:]
        print(f"  Skipping first {start_idx} files (start-index={start_idx})")

    total_remote = len(files)
    total_size_mb = sum(f["size"] for f in files) / (1024 * 1024)
    print(f"  Found {total_remote} files ({format_size(total_size_mb)}) in crawl '{args.crawl}'")

    # ── Step 2: Check what's already downloaded (verify file size) ──
    print(f"[2/3] Checking existing files in {output_dir}...")

    # Build local index: remote file → train-XXXXX-of-03291.parquet
    # Use original HF filename as local filename
    remote_to_local = {}
    for f in files:
        local_name = os.path.basename(f["path"])
        remote_to_local[f["path"]] = (local_name, f["size"])

    complete = set()
    corrupted = []
    if os.path.isdir(output_dir):
        for remote_path, (local_name, expected_size) in remote_to_local.items():
            local_path = os.path.join(output_dir, local_name)
            if os.path.exists(local_path):
                local_size = os.path.getsize(local_path)
                if local_size >= expected_size * 0.99:
                    with open(local_path, "rb") as fh:
                        fh.seek(-4, 2)
                        magic = fh.read(4)
                    if magic == b"PAR1":
                        complete.add(local_name)
                    else:
                        corrupted.append(local_name)

    if corrupted:
        print(f"  Corrupted (bad magic bytes, will re-download): {len(corrupted)}")
        for c in corrupted:
            os.remove(os.path.join(output_dir, c))

    # Need to download: train-XXXXX-of-03291.parquet missing or incomplete
    need = []
    for f in files:
        local_name, expected_size = remote_to_local[f["path"]]
        if local_name not in complete:
            need.append(f)
    have = len(complete)

    if have >= args.num_files:
        print(f"  Already have {have} complete files (>= {args.num_files} requested). Nothing to do.")
        sys.exit(0)

    # Filter to requested number
    need = need[:args.num_files - have]
    size_need_mb = sum(f["size"] for f in need) / (1024 * 1024)

    # Check for partial files that will be resumed
    partial_count = 0
    for f in need:
        local_name, _ = remote_to_local[f["path"]]
        local_path = os.path.join(output_dir, local_name)
        if os.path.exists(local_path):
            partial_count += 1

    if partial_count > 0:
        print(f"  Complete: {have} | Partial (will resume): {partial_count}")
    else:
        print(f"  Complete: {have}")
    print(f"  Need to download: {len(need)} ({format_size(size_need_mb)})")
    print(f"  Output: {output_dir}")
    print(f"  Workers: {args.workers}")

    # ── Step 3: Download (parallel) ──
    print(f"[3/3] Downloading {len(need)} files with {args.workers} workers...")
    start = time.time()

    download_func = download_file_hf_transfer if use_hf_transfer else download_file
    ok = 0
    fail = 0
    failed_files = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for i, f in enumerate(need):
            url = HF_RESOLVE.format(repo=REPO_ID, path=f["path"])
            local_name, _ = remote_to_local[f["path"]]
            dest = os.path.join(output_dir, local_name)
            size_mb = f["size"] / (1024 * 1024)
            expected_size = f["size"]
            future = executor.submit(download_func, url, dest, i, len(need), size_mb,
                                     expected_size=expected_size,
                                     show_progress=not args.quiet)
            futures[future] = local_name

        for future in as_completed(futures):
            success, fname = future.result()
            if success:
                ok += 1
            else:
                fail += 1
                failed_files.append(fname)
                if fail >= 3 and not args.quiet:
                    print(f"  [Warning] {fail} failures so far, continuing...")

    elapsed = time.time() - start
    speed = size_need_mb / max(1, elapsed) if ok > 0 else 0
    print(f"\n  Downloaded: {ok}/{len(need)} files ({format_size(size_need_mb)} in {elapsed:.0f}s, {speed:.1f} MB/s)")

    if failed_files:
        print(f"  Failed files: {', '.join(failed_files[:5])}" + ("..." if len(failed_files) > 5 else ""))

    # ── Summary ──
    final_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".parquet"))
    final_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in final_files) / (1024 * 1024)
    print(f"\n  Total files in {output_dir}: {len(final_files)} ({format_size(final_size)})")
    if final_files:
        print(f"  Range: {final_files[0]} ... {final_files[-1]}")


if __name__ == "__main__":
    main()