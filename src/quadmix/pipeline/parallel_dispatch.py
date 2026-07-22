"""Parallel dispatch: worker loops, tokenize workers, and parallel orchestration.

Standalone functions used by EssentialWebProxyRunner for multi-NPU parallel training.
"""

import os
import time
import gc
import warnings
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*owner does not match.*")

import numpy as np
import torch

from quadmix.utils.perf_timer import PerfTimer
from quadmix.utils.concurrency import ConcurrencyConfig
from quadmix.pipeline.shared_memory import shared_to_ndarray


_tokenizer_cache: Dict[str, "Tokenizer"] = {}


def _get_tokenizer(tokenizer_path: str) -> "Tokenizer":
    """Get or create a cached tokenizer instance (per-process cache)."""
    if tokenizer_path not in _tokenizer_cache:
        from tokenizers import Tokenizer
        if os.path.isdir(tokenizer_path):
            tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
            if not os.path.isfile(tokenizer_file):
                raise FileNotFoundError(
                    f"local tokenizer.json not found: {tokenizer_file}"
                )
            tokenizer = Tokenizer.from_file(tokenizer_file)
        elif os.path.isfile(tokenizer_path):
            tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            tokenizer = Tokenizer.from_pretrained(tokenizer_path)
        _tokenizer_cache[tokenizer_path] = tokenizer
    return _tokenizer_cache[tokenizer_path]


def _io_read_shard(
        sid: int,
        shard_path: str,
        miss_rows: List[int],
        text_col: str = "text",
        row_in_shard_col: str = "row_in_shard",
        has_row_in_shard: bool = True,
        is_row_col_sequential: bool = False,
        shard_total_rows: int = 0,
) -> Tuple[int, np.ndarray, List[str], float]:
    """Stage 1 worker: read one shard's parquet, return (sid, rows, texts, io_time).

    Uses pyarrow with adaptive strategy instead of pd.read_parquet.
    """
    io_t0 = time.time()
    from quadmix.data.metadata_manager import _read_one_shard_texts_with_rows

    row_col = row_in_shard_col if has_row_in_shard else None
    row_col_values = np.array(miss_rows, dtype=np.int64) if has_row_in_shard else None

    texts, parsed_rows = _read_one_shard_texts_with_rows(
        shard_path, text_col, row_col, row_col_values,
        has_row_in_shard, is_row_col_sequential, shard_total_rows,
    )
    io_time = time.time() - io_t0
    return (sid, parsed_rows, texts, io_time)


def _process_shard_full(
        sid: int,
        shard_path: str,
        miss_rows: List[int],
        tokenizer_path: str,
        block_size: int,
        threads_per_worker: int = 4,
        text_col: str = "text",
        row_in_shard_col: str = "row_in_shard",
        has_row_in_shard: bool = True,
        is_row_col_sequential: bool = False,
        shard_total_rows: int = 0,
) -> Tuple[int, np.ndarray, np.ndarray, float, float, float]:
    """Process one shard: IO (pyarrow) + tokenize in sequence.

    Uses pyarrow with adaptive strategy instead of pd.read_parquet.

    Returns (sid, parsed_rows, tokens_array, io_time, tok_time, total_time).
    """
    io_t0 = time.time()
    from quadmix.data.metadata_manager import _read_one_shard_texts_with_rows

    row_col = row_in_shard_col if has_row_in_shard else None
    row_col_values = np.array(miss_rows, dtype=np.int64) if has_row_in_shard else None

    texts, parsed_rows = _read_one_shard_texts_with_rows(
        shard_path, text_col, row_col, row_col_values,
        has_row_in_shard, is_row_col_sequential, shard_total_rows,
    )
    io_time = time.time() - io_t0

    tok_t0 = time.time()
    chunk = [(sid, idx, text) for idx, text in enumerate(texts)]
    meta, tokens_array = _tokenize_chunk_to_array(chunk, tokenizer_path, block_size, threads_per_worker)
    tok_time = time.time() - tok_t0

    return (sid, parsed_rows, tokens_array, io_time, tok_time, io_time + tok_time)


def _tokenize_shard_parallel(
        shard_tasks: List[Tuple],
        tokenizer_path: str,
        block_size: int,
        text_col: str = "text",
        row_in_shard_col: Optional[str] = "row_in_shard",
        has_row_in_shard: bool = True,
        is_row_col_sequential: bool = False,
) -> List[Tuple[int, np.ndarray, np.ndarray, float, float, float]]:
    """Parallel tokenize using ProcessPoolExecutor to bypass GIL.

    Each process handles one shard end-to-end (IO + tokenize) with its own
    Python interpreter and GIL, enabling true CPU parallelism.

    Config:
      TOKENIZE_WORKERS env var controls process count (default: min(48, cpu_count))
      RAYON_NUM_THREADS=4 per process → total threads = workers × 4

    Returns list of (sid, rows, tokens_int32, io_time, tok_time, total_time).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import threading

    cfg = ConcurrencyConfig()
    n_shards = len(shard_tasks)

    threads_per_worker = cfg.blas_threads_for(max(1, cfg.max_io_workers // 4))
    _saved_env = {}
    for _k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        _saved_env[_k] = os.environ.get(_k)
    os.environ["RAYON_NUM_THREADS"] = str(4)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)

    try:
        env_workers = int(os.environ.get("TOKENIZE_WORKERS", "0"))
        if env_workers >= 1:
            n_workers = env_workers
        else:
            n_workers = min(cfg.max_io_workers, n_shards)

        print(f"  [ParallelTokenize] {n_shards} shards, {n_workers} processes "
              f"× {threads_per_worker} Rust threads = {n_workers * threads_per_worker} total threads")

        results = []
        t0 = time.time()

        completed = [0]
        lock = threading.Lock()

        def on_done(fut):
            with lock:
                completed[0] += 1
                c = completed[0]
            if c % 10 == 0 or c == n_shards:
                elapsed = time.time() - t0
                speed = c / elapsed if elapsed > 0 else 0
                eta = (n_shards - c) / speed if speed > 0 else 0
                print(f"  [Tokenize Progress] {c}/{n_shards} shards "
                      f"({c*100//n_shards}%), "
                      f"{speed:.1f} shards/s, ETA {eta:.0f}s")

        with PerfTimer.section("parallel_tokenize", "parallel_tokenize"):
            from quadmix.pipeline.tokenize_worker import _process_shard_full as _worker_process_shard
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                futs = []
                for sid, shard_path, miss_rows, is_seq, total_rows in shard_tasks:
                    fut = executor.submit(
                        _worker_process_shard,
                        sid, shard_path, miss_rows,
                        tokenizer_path, block_size, threads_per_worker,
                        text_col, row_in_shard_col if row_in_shard_col is not None else "row_in_shard", has_row_in_shard,
                        is_seq, total_rows,
                    )
                    fut.add_done_callback(on_done)
                    futs.append(fut)

                failed_shards = []
                for fut in futs:
                    try:
                        sid, parsed_rows, tokens_array, io_time, tok_time, total_time = fut.result()
                        results.append((sid, parsed_rows, tokens_array, io_time, tok_time, total_time))
                    except Exception as e:
                        print(f"  [Tokenize Error] {e}")
                        import traceback
                        traceback.print_exc()
                        failed_shards.append(str(e))

        if failed_shards:
            raise RuntimeError(
                f"[ParallelTokenize] {len(failed_shards)} shard(s) failed to tokenize. "
                f"First error: {failed_shards[0]}"
            )

        total_time = time.time() - t0
        total_docs = sum(len(r[1]) for r in results)
        print(f"  [ParallelTokenize] {total_docs:,} docs in {total_time:.1f}s "
              f"({total_docs / total_time:.0f} docs/s)")

        return results
    finally:
        for _k, _v in _saved_env.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v


def _tokenize_chunk_with_meta(
        chunk: List[Tuple[int, int, str]],
        tokenizer_path: str,
        block_size: int,
) -> List[Tuple[int, int, List[int]]]:
    """Tokenize a chunk of (sid, idx, text) items. Returns (sid, idx, token_ids).

    Uses the Rust-based tokenizers library directly for max throughput.
    """
    tok = _get_tokenizer(tokenizer_path)
    texts = [item[2] for item in chunk]
    encodings = tok.encode_batch(texts)

    PAD_TOKEN = 50256

    results = []
    for (sid, idx, _), enc in zip(chunk, encodings):
        ids = list(enc.ids)
        if len(ids) > block_size:
            ids = ids[:block_size]
        if len(ids) < block_size:
            ids = ids + [PAD_TOKEN] * (block_size - len(ids))
        results.append((sid, idx, ids))
    return results


def _tokenize_chunk_to_array(
        chunk: List[Tuple[int, int, str]],
        tokenizer_path: str,
        block_size: int,
        threads_per_worker: int = 4,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Tokenize a chunk, returning compact numpy array instead of per-doc lists.

    Returns ((sid, idx) pairs, np.array[N x block_size, dtype=int32]).
    """
    _saved_env = {}
    for _k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS"):
        _saved_env[_k] = os.environ.get(_k)
    os.environ["RAYON_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    try:
        tok = _get_tokenizer(tokenizer_path)
        texts = [item[2] for item in chunk]
        encodings = tok.encode_batch(texts)

        PAD_TOKEN = 50256
        N = len(chunk)
        tokens_array = np.full((N, block_size), PAD_TOKEN, dtype=np.int32)

        meta = []
        for (i, (sid, idx, _)), enc in zip(enumerate(chunk), encodings):
            ids = list(enc.ids)
            n = min(len(ids), block_size)
            tokens_array[i, :n] = ids[:n]
            meta.append((sid, idx))

        return meta, tokens_array
    finally:
        for _k, _v in _saved_env.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v


def _worker_dynamic_loop(
        worker_id: int,
        device_type: str,
        config_dict: dict,
        task_queue,
        result_queue,
):
    """Worker loop for dynamic mode: pull task → run → push result → repeat.

    Uses shared memory for metadata arrays when available, avoiding
    per-worker 15GB+ parquet reload. Falls back to disk loading gracefully.
    """
    from quadmix.pipeline.essential_proxy_runner import EssentialWebProxyRunner
    from quadmix.core.types import ParameterSet

    with open(f"/tmp/worker_{worker_id}_entry.log", "w") as ef:
        ef.write(f"[Worker {worker_id}] FUNCTION ENTERED at {time.time()}")
    try:
        from quadmix.data.metadata_manager import ShardMetadataManager

        shared_dl = config_dict.get("shared_domain_labels")
        shared_qs = config_dict.get("shared_quality_scores")
        shared_cc = config_dict.get("shared_doc_char_counts")
        shared_nq = config_dict.get("shared_normalized_quality")
        per_shard_info = config_dict.get("per_shard_info")

        use_shared = (shared_dl is not None and shared_qs is not None
                      and shared_cc is not None and per_shard_info is not None)

        if use_shared:
            t0 = time.time()
            domain_labels = shared_to_ndarray(shared_dl)
            quality_scores = shared_to_ndarray(shared_qs)
            doc_char_counts = shared_to_ndarray(shared_cc)

            shard_starts_arr = np.array(
                [s["start_idx"] for s in per_shard_info], dtype=np.int64
            )

            mgr = ShardMetadataManager.from_shared(
                domain_labels=domain_labels,
                quality_scores=quality_scores,
                doc_char_counts=doc_char_counts,
                per_shard_info=per_shard_info,
                shard_starts=shard_starts_arr,
                preprocessed_dir=config_dict.get("preprocessed_dir", ""),
                schema=config_dict.get("schema"),
                num_domains=config_dict.get("num_domains"),
                num_quality_criteria=config_dict.get("num_quality_criteria"),
                detected_domain_names=config_dict.get("detected_domain_names"),
                detected_quality_names=config_dict.get("detected_quality_names"),
                quality_directions=config_dict.get("quality_directions"),
                domain_label_map=config_dict.get("domain_label_map"),
            )
            print(f"[Worker {worker_id}] Mapped {len(domain_labels):,} docs from shared memory "
                  f"({time.time() - t0:.1f}s vs ~60s disk reload)")
        else:
            if config_dict.get("preprocessed_dir"):
                mgr = ShardMetadataManager(config_dict["preprocessed_dir"], schema=config_dict.get("schema"))
            else:
                mgr = None

        runner = EssentialWebProxyRunner(
            config=config_dict["config"],
            metadata_manager=mgr,
            val_data_path=config_dict["val_data_path"],
            output_dir=config_dict["output_dir"],
            device_type=device_type,
            npu_device_id=worker_id,
            model_variant=config_dict["model_variant"],
            global_batch_size=config_dict["global_batch_size"],
            micro_batch_size=config_dict["micro_batch_size"],
            max_step=config_dict["max_step"],
            warmup_fraction=config_dict.get("warmup_fraction", 0.04),
            learning_rate=config_dict["learning_rate"],
            weight_decay=config_dict["weight_decay"],
            grad_clip=config_dict["grad_clip"],
            tiny_steps=config_dict["tiny_steps"],
            doc_limit=config_dict["doc_limit"],
            test_block_size=config_dict["test_block_size"],
            rank_ref_size=config_dict["rank_ref_size"],
            token_cache_dir=config_dict["token_cache_dir"],
            checkpoint_interval=config_dict.get("checkpoint_interval", 1000),
            domain_names=config_dict.get("domain_names"),
            quality_names=config_dict.get("quality_names"),
            quality_directions=config_dict.get("quality_directions"),
            worker_mode=True,
        )

        completed = 0
        current_shm_info = None
        while True:
            task = task_queue.get()

            if task is None:
                print(f"[Worker {worker_id}] Shutdown, completed {completed} experiments")
                break

            exp_id, params, selected_idx, shm_info = task[:4]
            current_shm_info = shm_info
            sampled_doc_count = task[4] if len(task) > 4 else None
            print(f"[Worker {worker_id}] Running exp {exp_id}")

            r = runner.run_experiment(params, experiment_id=exp_id, selected_idx=selected_idx,
                                      shm_info=shm_info,
                                      sampled_doc_count=sampled_doc_count,
                                      checkpoint_interval=config_dict.get("checkpoint_interval", 1000))
            result_queue.put(r)
            completed += 1

            if device_type == "npu":
                torch.npu.synchronize()
            gc.collect()
            if device_type == "npu":
                try:
                    torch.npu.empty_cache()
                except Exception:
                    pass

            if shm_info is not None:
                from multiprocessing.shared_memory import SharedMemory
                try:
                    shm = SharedMemory(name=shm_info[0])
                    shm.close()
                    shm.unlink()
                    print(f"[Worker {worker_id}] Released SharedMemory for exp {exp_id}")
                except Exception as e:
                    print(f"[Worker {worker_id}] SharedMemory cleanup failed for exp {exp_id}: {e}")
            else:
                exp_token_path = runner._get_exp_token_path(exp_id)
                if os.path.exists(exp_token_path):
                    os.remove(exp_token_path)
                    print(f"[Worker {worker_id}] Cleaned temp file for exp {exp_id}")

        result_queue.put(None)

    except Exception as top_err:
        import sys, traceback
        from quadmix.core.types import ProxyResult, ParameterSet
        err_path = f"/tmp/worker_{worker_id}_error.log"
        tb_str = traceback.format_exc()
        try:
            with open(err_path, "w") as ef:
                ef.write(f"[Worker {worker_id}] CRASH: {top_err}\n")
                ef.write(tb_str)
            print(f"[Worker {worker_id}] ERROR -> {err_path}", flush=True)
        except Exception:
            pass
        try:
            result_queue.put(ProxyResult(
                parameters=ParameterSet(),
                validation_loss=float('inf'),
                metadata={
                    "experiment_id": -1,
                    "worker_id": worker_id,
                    "error": str(top_err),
                    "traceback": tb_str,
                    "is_worker_crash": True,
                },
            ))
        except Exception:
            pass
        try:
            if device_type == "npu":
                torch.npu.empty_cache()
        except Exception:
            pass
        sys.exit(1)


def _reval_worker(
        worker_id: int,
        device_str: str,
        val_data_path: str,
        model_config_dict: dict,
        model_paths: list,
        exp_indices: list,
        result_queue,
):
    """Worker for parallel revalidation across multiple GPUs.

    Each worker loads validation data once, then iterates over assigned models.
    device_str: e.g. 'cuda:0', 'cuda:1', 'npu:0', 'npu:1', 'cpu'.
    """
    import sys
    try:
        from quadmix.core.proxy_model import ProxyModel, ProxyConfig
        from quadmix.pipeline.loss_utils import chunked_loss_per_token_from_hidden, compute_val_batch_size

        device = torch.device(device_str)

        val_data = torch.load(val_data_path, map_location="cpu", weights_only=True)
        val_token_ids = val_data["token_ids"]
        val_loss_mask = val_data["loss_mask"]
        val_task_labels = val_data.get("task_labels", None)
        del val_data

        block_size = model_config_dict["block_size"]
        val_n = len(val_token_ids)
        val_tokens = val_token_ids[:val_n, :block_size].to(device)
        val_mask = val_loss_mask[:val_n, :block_size].to(device)

        print(f"[RevalWorker {worker_id}] {device_str}: {len(model_paths)} models, "
              f"{val_n} val docs", flush=True)

        for i, (model_path, exp_idx) in enumerate(zip(model_paths, exp_indices)):
            model = ProxyModel(config=ProxyConfig(**model_config_dict)).to(device)
            if device.type == "npu":
                model = model.to(torch.bfloat16)
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)

            model.eval()
            with torch.no_grad():
                val_bs, val_chunk_size = compute_val_batch_size(
                    device, model.config.vocab_size, block_size - 1, max_val_bs=128,
                )
                val_bs = min(val_bs, val_n, 8)
                per_doc_losses = []
                for start in range(0, len(val_tokens), val_bs):
                    end = min(start + val_bs, len(val_tokens))
                    ids_in = val_tokens[start:end, :-1]
                    ids_tgt = val_tokens[start:end, 1:]
                    mask_tgt = val_mask[start:end, 1:]
                    hidden = model(ids_in, return_hidden=True)
                    loss = chunked_loss_per_token_from_hidden(
                        model, hidden, ids_tgt, chunk_size=val_chunk_size,
                    )
                    assistant_count = mask_tgt.float().sum(dim=1).clamp(min=1)
                    per_doc = (loss * mask_tgt.float()).sum(dim=1) / assistant_count
                    per_doc_losses.append(per_doc)
                    del hidden, loss, per_doc

                all_losses = torch.cat(per_doc_losses)
                val_loss = float(all_losses.mean())

                per_task_losses = None
                if val_task_labels is not None:
                    per_task_losses = {}
                    for task in sorted(set(val_task_labels)):
                        task_indices = [
                            j for j, t in enumerate(val_task_labels) if t == task
                        ]
                        if task_indices:
                            per_task_losses[task] = float(
                                all_losses[task_indices].mean()
                            )

            del model, state_dict, per_doc_losses, all_losses
            if device.type == "npu":
                import gc
                gc.collect()
                try:
                    torch.npu.empty_cache()
                except Exception:
                    pass
            elif device.type == "cuda":
                torch.cuda.empty_cache()

            result_queue.put((exp_idx, val_loss, per_task_losses))

            if (i + 1) % 10 == 0 or i == len(model_paths) - 1:
                print(f"[RevalWorker {worker_id}] {device_str}: "
                      f"{i+1}/{len(model_paths)} done", flush=True)

        result_queue.put(None)

    except Exception as e:
        import traceback
        print(f"[RevalWorker {worker_id}] ERROR: {e}\n{traceback.format_exc()}",
              flush=True)
        try:
            result_queue.put(None)
        except Exception:
            pass
        sys.exit(1)
