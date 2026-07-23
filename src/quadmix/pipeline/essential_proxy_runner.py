"""
EssentialWebProxyRunner — Real proxy training on essential-web-v1 data.

Shard-aware mode:
  Uses ShardMetadataManager → loads only metadata (domain+quality) upfront,
  reads text on-demand per experiment. Per-shard disk cache for tokens.

Multi-NPU Parallelism:
  Dynamic task queue mode (run_batch_parallel):
    - Workers fetch tasks from shared queue, no batch boundaries
    - Fast workers naturally do more experiments
    - Tokenize thread runs ahead, independent of NPU training
    - Each worker binds to NPU device by worker_id

Aligned with RegMix:
  - GPT-NeoX-20B BPE tokenizer (same as GPT-NeoX)
  - On-demand tokenization with per-shard disk cache
  - Validation on openhermes-10k with assistant-only loss
  - RegMix training loop: gradient accumulation, cosine LR, AdamW
"""

import os, math, time, json, glob, threading
import multiprocessing as mp
import multiprocessing.shared_memory
from functools import partial
from typing import List, Optional, Dict, Tuple, Callable
from joblib import Parallel, delayed
from contextlib import contextmanager

import warnings
warnings.filterwarnings("ignore", message=".*owner does not match.*")

import numpy as np
import torch
import torch.nn.functional as F

from quadmix.core.types import ParameterSet, ProxyResult, QuaDMixConfig
from quadmix.core.quality_merger import compute_merged_quality_scores
from quadmix.core.quality_rank import compute_quality_ranks
from quadmix.core.sampler import compute_sampling_values
from quadmix.pipeline.proxy_runner import BaseProxyRunner
from quadmix.utils.perf_timer import PerfTimer
from quadmix.utils.concurrency import (
    ConcurrencyConfig,
    get_available_memory_bytes,
    get_blas_threads,
    set_blas_threads,
)
from quadmix.utils.json_utils import sanitize_for_json
from quadmix.pipeline.loss_utils import (
    chunked_loss_from_hidden, chunked_loss_per_token_from_hidden, compute_val_batch_size,
)
from quadmix.pipeline.shared_memory import SharedArrayInfo, ndarray_to_shared, shared_to_ndarray
from quadmix.pipeline.parallel_dispatch import (
    _worker_dynamic_loop,
    _tokenize_shard_parallel,
    _tokenize_chunk_to_array,
)

from collections import namedtuple

_PreSampleData = namedtuple("_PreSampleData", [
    "sub_quality_infos", "domain_indices_infos",
    "sub_token_counts_infos", "has_token_counts",
    "num_domains", "rank_ref_size", "num_docs",
    "seed_offset",
])


def _presample_one(args):
    i, params, data = args
    from quadmix.pipeline.shared_memory import shared_to_ndarray
    sub_qualities = {m: shared_to_ndarray(info) for m, info in data.sub_quality_infos.items()}
    domain_indices = {m: shared_to_ndarray(info) for m, info in data.domain_indices_infos.items()}
    sub_tc = {}
    if data.has_token_counts:
        sub_tc = {m: shared_to_ndarray(info) for m, info in data.sub_token_counts_infos.items()}

    seed_offset = data.seed_offset
    M = data.num_domains
    rng_eq2 = np.random.default_rng(i + seed_offset + 1729)
    rng_sample = np.random.default_rng(i + seed_offset + 42)
    has_tokens = data.has_token_counts

    domain_selected = []

    for m in range(M):
        if m not in sub_qualities or len(sub_qualities[m]) == 0:
            continue
        if m >= len(params.sampling_configs):
            continue

        sub_q = sub_qualities[m]
        indices = domain_indices.get(m)
        n_m = len(sub_q)
        alpha_m = params.merge_config.get_final_weights(m)
        domain_scores = sub_q @ alpha_m

        k = min(data.rank_ref_size, n_m)
        ref_idx = rng_eq2.choice(n_m, k, replace=False)
        ref_scores_unsorted = domain_scores[ref_idx]
        sort_order = np.argsort(-ref_scores_unsorted)
        ref_scores = ref_scores_unsorted[sort_order]
        positions = np.searchsorted(-ref_scores, -domain_scores, side='right')

        if has_tokens and m in sub_tc:
            ref_tokens = sub_tc[m][ref_idx][sort_order].astype(np.float64)
            cum_tokens = np.concatenate(([0.0], np.cumsum(ref_tokens)))
            total_ref_tokens = cum_tokens[-1]
            if total_ref_tokens > 0:
                domain_ranks = cum_tokens[positions] / total_ref_tokens
            else:
                domain_ranks = positions.astype(np.float64) / k
        else:
            domain_ranks = positions.astype(np.float64) / k

        sc = params.sampling_configs[m]
        within_threshold = domain_ranks <= sc.omega
        sv = np.full(n_m, sc.epsilon, dtype=np.float64)
        if within_threshold.any():
            exponent = -sc.lambda_ * (sc.omega - domain_ranks[within_threshold])
            sigmoid = 2.0 / (1.0 + np.exp(np.clip(exponent, -100, 100)))
            sv[within_threshold] = sigmoid ** sc.eta + sc.epsilon

        int_part = np.floor(sv).astype(np.int64)
        frac_part = sv - int_part
        random_mask = rng_sample.uniform(size=n_m) < frac_part
        repeats = int_part + random_mask.astype(np.int64)
        selected_local = np.repeat(np.arange(n_m), repeats)
        if len(selected_local) > 0:
            domain_selected.append(indices[selected_local])

    if domain_selected:
        return i, np.concatenate(domain_selected)
    else:
        rng2 = np.random.default_rng(i + seed_offset + 42)
        return i, rng2.choice(np.arange(data.num_docs), 100, replace=False)


class EssentialWebProxyRunner(BaseProxyRunner):
    """
    Proxy runner with:
      - GPT-NeoX-20B BPE tokenizer (matching RegMix)
      - Training data: essential-web-v1 via ShardMetadataManager (on-demand)
      - Validation: openhermes-10k (pre-tokenized with assistant loss mask)
      - Per-shard disk cache for tokenized training data
      - Loss: assistant-only for validation, full LM for training
    """

    def __init__(
            self,
            config: QuaDMixConfig,
            val_data_path: str,
            metadata_manager: Optional[object] = None,
            output_dir: str = "./proxy_validation",
            device_type: str = "cpu",
            npu_device_id: int = 0,
            model_variant: str = "tinyllama_1M",
            global_batch_size: int = 64,
            micro_batch_size: int = 8,
            max_step: int = 25000,
            warmup_fraction: float = 0.04,
            learning_rate: float = 4e-4,
            weight_decay: float = 0.1,
            grad_clip: float = 1.0,
            tiny_steps: int = 10,
            doc_limit: Optional[int] = None,
            test_block_size: Optional[int] = None,
            rank_ref_size: int = 10000,
            token_cache_dir: Optional[str] = None,
            memory_cache_max_gb: float = 500.0,
            checkpoint_interval: int = 1000,
            domain_names: Optional[List[str]] = None,
            quality_names: Optional[List[str]] = None,
            quality_directions: Optional[List[bool]] = None,
            worker_mode: bool = False,
    ):
        from quadmix.constants import DEFAULT_TOKEN_CACHE_DIR
        if token_cache_dir is None:
            token_cache_dir = DEFAULT_TOKEN_CACHE_DIR

        self.config = config
        self._seed_offset = config.seed if config.seed is not None else np.random.default_rng().integers(0, 2**31)
        self._concurrency = ConcurrencyConfig()
        self.metadata_manager = metadata_manager
        self.val_data_path = val_data_path
        self.output_dir = output_dir
        self.model_variant = model_variant
        self.device_type = device_type
        self.npu_device_id = npu_device_id
        self.memory_cache_max_gb = memory_cache_max_gb
        self.tiny_steps = tiny_steps
        self.doc_limit = doc_limit
        self.rank_ref_size = rank_ref_size
        self.token_cache_dir = token_cache_dir
        self.checkpoint_interval = checkpoint_interval

        self._domain_names = domain_names
        self._quality_names = quality_names
        self._quality_directions = quality_directions
        self._negated_cols: List[int] = []
        self._worker_mode = worker_mode

        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.max_step = max_step
        self.warmup_fraction = warmup_fraction
        self.warmup_steps = 0
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        from quadmix.core.proxy_model import ProxyConfig
        self.model_config = ProxyConfig.from_name(
            model_variant, block_size=test_block_size
        )
        self.block_size = self.model_config.block_size
        self.batch_size = global_batch_size // 1
        self.gradient_accumulation_steps = max(1, self.batch_size // micro_batch_size)

        from transformers import AutoTokenizer
        tokenizer_source = os.environ.get(
            "QUADMIX_TOKENIZER_PATH",
            "EleutherAI/gpt-neox-20b",
        )
        tokenizer_is_local = os.path.exists(tokenizer_source)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            local_files_only=tokenizer_is_local,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        assert self.tokenizer.vocab_size <= self.model_config.vocab_size, \
            f"Tokenizer vocab ({self.tokenizer.vocab_size}) > model vocab ({self.model_config.vocab_size})"
        print(f"[ProxyRunner] GPT-NeoX tokenizer: {tokenizer_source}, "
              f"vocab={self.tokenizer.vocab_size}")
        print(f"[ProxyRunner] Model config: {model_variant}, "
              f"block={self.block_size}, model_vocab={self.model_config.vocab_size}")
        print(f"[ProxyRunner] Training: batch={self.batch_size}, "
              f"micro_batch={self.micro_batch_size}, "
              f"grad_acc={self.gradient_accumulation_steps}")

        if metadata_manager is not None:
            self._mode = "sharded"
            self._load_metadata_only()
        else:
            raise ValueError("metadata_manager must be provided (legacy single-file path removed)")

        print(f"[ProxyRunner] Loading validation set: {self.val_data_path}")
        val_data = torch.load(self.val_data_path, map_location="cpu", weights_only=True)
        self._val_token_ids = val_data["token_ids"]
        self._val_loss_mask = val_data["loss_mask"]
        self._val_task_labels = val_data.get("task_labels", None)
        if self._val_task_labels is not None:
            unique_tasks = sorted(set(self._val_task_labels))
            print(f"[ProxyRunner] Val tasks: {len(unique_tasks)} tasks: {unique_tasks}")
        else:
            print(f"[ProxyRunner] Val: no task_labels (aggregate-only mode)")
        print(f"[ProxyRunner] Val tokens: {self._val_token_ids.shape}, "
              f"assistant tokens: {self._val_loss_mask.sum().item()}/"
              f"{self._val_loss_mask.numel()}")

    def _load_metadata_only(self):
        """Load domain labels + quality scores from metadata manager (no text)."""
        t0 = time.time()
        mgr = self.metadata_manager
        self._domain_labels = mgr.domain_labels
        self._quality_scores = mgr.quality_scores.copy()
        self._token_counts = mgr.estimate_token_counts()
        self._num_docs = mgr.num_docs
        self._train_idx = np.arange(self._num_docs)

        quality_directions = self._quality_directions or mgr.quality_directions
        self._negated_cols = []
        if quality_directions:
            for n, hb in enumerate(quality_directions):
                if not hb:
                    self._quality_scores[:, n] = -self._quality_scores[:, n]
                    self._negated_cols.append(n)

        print(f"[ProxyRunner] Sharded mode: {self._num_docs:,} docs "
              f"(metadata only, {mgr.num_shards} shards) ({time.time() - t0:.0f}s)")

        if self._worker_mode:
            self._normalized_quality = None
            self._domain_indices: Dict[int, np.ndarray] = {}
            print(f"[ProxyRunner] Worker mode: skipped normalization + domain indices "
                  f"(selected_idx pre-computed by main process)")
        else:
            from quadmix.utils.normalization import get_normalizer
            self._normalizer_name = "rank"
            normalize_fn = get_normalizer(self._normalizer_name)

            t1 = time.time()
            num_criteria = self._quality_scores.shape[1]
            n_jobs = min(num_criteria, os.cpu_count()) if self._num_docs > 50000 else 1
            normalized_cols = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(normalize_fn)(self._quality_scores[:, n]) for n in range(num_criteria)
            )
            self._normalized_quality = np.column_stack(normalized_cols).astype(self._quality_scores.dtype)
            print(f"[ProxyRunner] Pre-normalized {num_criteria} quality criteria "
                  f"({time.time() - t1:.1f}s) — Eq.1 now ~5x faster per experiment")

            t2 = time.time()
            sort_idx = np.argsort(self._domain_labels)
            sorted_labels = self._domain_labels[sort_idx]
            boundaries = np.concatenate([[0], np.where(sorted_labels[:-1] != sorted_labels[1:])[0] + 1, [self._num_docs]])
            self._domain_indices: Dict[int, np.ndarray] = {}
            for i in range(len(boundaries) - 1):
                domain_id = int(sorted_labels[boundaries[i]])
                self._domain_indices[domain_id] = sort_idx[boundaries[i]:boundaries[i + 1]]
            print(f"[ProxyRunner] Pre-computed domain indices for {len(self._domain_indices)} domains "
                  f"({time.time() - t2:.1f}s) — Eq.1 mask elimination")

        os.makedirs(self.token_cache_dir, exist_ok=True)
        self._cache_hits = 0
        self._cache_misses = 0

        self._memory_cache: Dict[int, dict] = {}
        self._memory_cache_bytes: int = 0
        self._memory_cache_lru: List[int] = []
        self._memory_cache_lock = threading.Lock()

    def _tokenize_texts(self, texts: List[str]) -> torch.Tensor:
        """Tokenize a list of texts into [M, block_size] int64 tensor."""
        B = 500
        all_ids = []
        for i in range(0, len(texts), B):
            batch = texts[i:i + B]
            enc = self.tokenizer(
                batch, max_length=self.block_size,
                truncation=True, padding="max_length",
                return_tensors="pt",
            )
            all_ids.append(enc["input_ids"])
        return torch.cat(all_ids, dim=0)

    def _memory_cache_get_rows(self, sid: int) -> set:
        """Return set of row_in_shard already in memory cache for this shard."""
        with self._memory_cache_lock:
            if sid not in self._memory_cache:
                return set()
            if sid in self._memory_cache_lru:
                self._memory_cache_lru.remove(sid)
                self._memory_cache_lru.append(sid)
            return set(int(r) for r in self._memory_cache[sid]["rows"])

    def _memory_cache_add_rows(self, sid: int, new_rows: np.ndarray, new_tokens: np.ndarray,
                                skip_eviction: bool = False):
        """Add new rows to memory cache. LRU eviction when over limit."""
        with self._memory_cache_lock:
            old_bytes = 0
            if sid in self._memory_cache:
                old_data = self._memory_cache[sid]
                old_bytes = old_data["rows"].nbytes + old_data["tokens"].nbytes

            if sid not in self._memory_cache:
                self._memory_cache[sid] = {
                    "rows": np.array([], dtype=np.int64),
                    "tokens": np.zeros((0, new_tokens.shape[1]), dtype=np.int32),
                }

            old = self._memory_cache[sid]
            old_rows = old["rows"]
            old_tokens = old["tokens"]

            combined_rows = np.concatenate([old_rows, new_rows])
            combined_tokens = np.concatenate([old_tokens, new_tokens])

            row_to_idx = {}
            for i, r in enumerate(combined_rows):
                row_to_idx[int(r)] = i

            unique_rows = np.array(sorted(row_to_idx.keys()), dtype=np.int64)
            final_tokens = combined_tokens[[row_to_idx[int(r)] for r in unique_rows]]

            new_bytes = unique_rows.nbytes + final_tokens.nbytes
            self._memory_cache[sid] = {"rows": unique_rows, "tokens": final_tokens}

            self._memory_cache_bytes += new_bytes - old_bytes
            if sid in self._memory_cache_lru:
                self._memory_cache_lru.remove(sid)
            self._memory_cache_lru.append(sid)

            if skip_eviction:
                return

            max_bytes = int(self.memory_cache_max_gb * 1024 ** 3)
            while self._memory_cache_bytes > max_bytes and self._memory_cache_lru:
                victim_sid = self._memory_cache_lru.pop(0)
                if victim_sid in self._memory_cache:
                    victim = self._memory_cache.pop(victim_sid)
                    self._memory_cache_bytes -= (victim["rows"].nbytes + victim["tokens"].nbytes)

    def _memory_cache_query(self, sid: int, requested_rows: List[int]) -> Tuple[np.ndarray, List[int], List[int]]:
        """Query memory cache for requested rows."""
        with self._memory_cache_lock:
            if sid not in self._memory_cache:
                return np.zeros((0, self.block_size), dtype=np.int32), [], requested_rows
            cache_data = self._memory_cache[sid]
            cache_rows_set = set(int(r) for r in cache_data["rows"])
            hit_rows_set = [r for r in requested_rows if int(r) in cache_rows_set]
            miss_rows = [r for r in requested_rows if int(r) not in cache_rows_set]

            if not hit_rows_set:
                return np.zeros((0, self.block_size), dtype=np.int32), [], miss_rows

            cache_rows_arr = cache_data["rows"]
            cache_tokens = cache_data["tokens"]

            sorted_hit_rows = sorted(hit_rows_set)
            positions = np.searchsorted(cache_rows_arr, sorted_hit_rows)

            valid_mask = positions < len(cache_rows_arr)
            assert valid_mask.all(), f"Some hit rows not in cache: {sorted_hit_rows}"

            tokens = cache_tokens[positions].copy()
            return tokens, sorted_hit_rows, miss_rows

    def _get_shard_token_path(self, shard_idx: int) -> str:
        """Path to disk cache for a shard's selected tokens (npz, mmap-compatible)."""
        return os.path.join(
            self.token_cache_dir,
            f"shard_{shard_idx:05d}_bs{self.block_size}.npz",
        )

    def _cached_shard_rows(self, sid: int) -> set:
        """Return set of row_in_shard already cached for this shard."""
        cache_path = self._get_shard_token_path(sid)
        if not os.path.exists(cache_path):
            return set()
        with np.load(cache_path) as data:
            rows = set(data['rows'].tolist())
        return rows

    def _cache_add_rows(self, sid: int, new_rows: np.ndarray, new_tokens: torch.Tensor):
        """Add new rows to shard cache (immediate write with file lock)."""
        import fcntl

        cache_path = self._get_shard_token_path(sid)
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)

        new_np = new_tokens.numpy().astype(np.int32)

        cache_no_ext = cache_path[:-4]
        temp_path = cache_no_ext + f".tmp.{int(time.time() * 1000000)}"
        actual_temp = temp_path + ".npz"

        lock_path = cache_path + ".lock"
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)

        with open(lock_path, 'w') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                if os.path.exists(cache_path):
                    with np.load(cache_path) as old:
                        old_rows = old['rows'].copy()
                        old_tokens = old['tokens'].copy()
                else:
                    old_rows = np.array([], dtype=np.int64)
                    old_tokens = np.zeros((0, new_np.shape[1]), dtype=np.int32)

                combined_rows = np.concatenate([old_rows, new_rows])
                combined_tokens = np.concatenate([old_tokens, new_np])

                row_to_idx = {int(r): i for i, r in enumerate(combined_rows)}
                unique_rows = np.array(sorted(row_to_idx.keys()), dtype=np.int64)
                final_tokens = combined_tokens[[row_to_idx[int(r)] for r in unique_rows]]

                np.savez(temp_path, tokens=final_tokens, rows=unique_rows)
                os.replace(actual_temp, cache_path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                if os.path.exists(actual_temp):
                    try:
                        os.remove(actual_temp)
                    except OSError:
                        pass

    def _get_exp_token_path(self, exp_id: int) -> str:
        """Path to temporary token file for a single experiment."""
        return os.path.join(
            self.token_cache_dir,
            f"exp_{exp_id:04d}_tokens.npy"
        )

    def _tokenize_batch_union(
            self,
            batch_selected: List[np.ndarray],
            batch_exp_ids: List[int],
            async_write_queue: Optional["Queue"] = None,
            shm_store: Optional[Dict[int, tuple]] = None,
            shm_lock: Optional[threading.Lock] = None,
    ) -> Dict[int, str]:
        """NPU Parallel Mode: Batch tokenize union miss rows across all experiments."""
        t0 = time.time()

        if hasattr(self, '_global_index') and self._global_index is not None:
            sorted_global_ids, all_tokens_flat, sort_idx = self._global_index

            exp_token_paths: Dict[int, str] = {}
            pack_t0 = time.time()
            n_batch = len(batch_exp_ids)

            print(f"[BatchTokenize] {n_batch} exps, using _global_index "
                  f"({len(sorted_global_ids):,} docs, skip cache check)")

            for i, (exp_id, selected_idx) in enumerate(zip(batch_exp_ids, batch_selected)):
                positions = np.searchsorted(sorted_global_ids, selected_idx)
                positions = np.clip(positions, 0, len(sorted_global_ids) - 1)
                matched = sorted_global_ids[positions] == selected_idx
                if not matched.all():
                    n_missing = int((~matched).sum())
                    raise RuntimeError(
                        f"[Pack] Experiment {exp_id}: {n_missing}/{len(selected_idx)} "
                        f"documents not found in tokenized cache. "
                        f"Check for shard tokenization failures."
                    )
                flat_positions = sort_idx[positions]
                result = all_tokens_flat[flat_positions]

                if shm_store is not None:
                    from multiprocessing.shared_memory import SharedMemory
                    shm = SharedMemory(create=True, size=result.nbytes)
                    shm_array = np.ndarray(result.shape, dtype=result.dtype, buffer=shm.buf)
                    shm_array[:] = result[:]
                    if shm_lock:
                        with shm_lock:
                            shm_store[exp_id] = (shm.name, result.shape, result.dtype.str)
                    else:
                        shm_store[exp_id] = (shm.name, result.shape, result.dtype.str)
                    shm.close()
                    exp_token_paths[exp_id] = f"shm://{shm.name}"
                else:
                    exp_token_path = self._get_exp_token_path(exp_id)
                    np.save(exp_token_path, result)
                    exp_token_paths[exp_id] = exp_token_path

                elapsed = time.time() - pack_t0
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_batch - i - 1) / speed if speed > 0 else 0
                print(f"  [Pack] {i+1}/{n_batch} exps ({(i+1)*100//n_batch}%), "
                      f"{elapsed:.1f}s elapsed, ETA {eta:.0f}s")

            pack_time = time.time() - pack_t0
            print(f"[BatchTokenize] Pack {n_batch} exps: {pack_time:.1f}s")

            elapsed = time.time() - t0
            print(f"[BatchTokenize] Total: {elapsed:.1f}s for {n_batch} experiments")

            return exp_token_paths

        mgr = self.metadata_manager

        shard_to_exp_rows: Dict[int, Dict[int, List[int]]] = {}

        for exp_id, selected_idx in zip(batch_exp_ids, batch_selected):
            shard_groups = mgr.global_to_shard_rows(selected_idx)
            for sid, (shard_path, local_rows) in shard_groups.items():
                if sid not in shard_to_exp_rows:
                    shard_to_exp_rows[sid] = {}
                shard_to_exp_rows[sid][exp_id] = list(int(r) for r in local_rows)

        n_shards = len(shard_to_exp_rows)
        print(f"[BatchTokenize] {len(batch_exp_ids)} exps, {n_shards} shards")

        total_miss_rows = 0
        shard_miss_info: Dict[int, Tuple[str, np.ndarray]] = {}

        for sid, exp_rows_dict in shard_to_exp_rows.items():
            all_needed_rows = set()
            for exp_id, rows in exp_rows_dict.items():
                all_needed_rows.update(rows)

            all_needed_arr = np.array(sorted(all_needed_rows), dtype=np.int64)
            row_col_vals = mgr.local_to_row_col(sid, all_needed_arr)
            row_col_int = set(int(v) for v in row_col_vals)

            memory_cached = self._memory_cache_get_rows(sid)
            hit_in_memory = [v for v in row_col_int if v in memory_cached]

            remaining = [v for v in row_col_int if v not in memory_cached]
            disk_cached = self._cached_shard_rows(sid)
            hit_in_disk = [v for v in remaining if v in disk_cached]

            miss_row_col = [v for v in remaining if v not in disk_cached]

            if miss_row_col:
                shard_path = mgr._per_shard_info[sid]["path"]
                miss_arr = np.array(sorted(miss_row_col), dtype=np.int64)
                shard_total_rows = mgr._per_shard_info[sid]["num_docs"]
                shard_miss_info[sid] = (shard_path, miss_arr, mgr._is_row_col_sequential, shard_total_rows)
                total_miss_rows += len(miss_row_col)

            if hit_in_disk:
                cache_path = self._get_shard_token_path(sid)
                with np.load(cache_path) as data:
                    disk_rows = data['rows']
                    disk_tokens = data['tokens']

                    row_to_pos = {int(r): i for i, r in enumerate(disk_rows)}
                    positions = np.array([row_to_pos[int(r)] for r in hit_in_disk], dtype=np.int64)
                    hit_tokens = disk_tokens[positions]

                self._memory_cache_add_rows(sid, np.array(hit_in_disk, dtype=np.int64), hit_tokens)

        if total_miss_rows == 0:
            print(f"[BatchTokenize] All {n_shards} shards fully cached, 0 miss rows")
        else:
            print(f"[BatchTokenize] {total_miss_rows:,} miss rows across {len(shard_miss_info)} shards")

        if shard_miss_info:
            schema = self.metadata_manager.schema
            read_t0 = time.time()

            shard_tasks = [
                (sid, shard_path, miss_rows_arr.tolist(), is_seq, total_rows)
                for sid, (shard_path, miss_rows_arr, is_seq, total_rows) in shard_miss_info.items()
            ]

            parallel_results = _tokenize_shard_parallel(
                shard_tasks, self.tokenizer.name_or_path, self.block_size,
                text_col=schema.text_col,
                row_in_shard_col=schema.row_in_shard_col,
                has_row_in_shard=self.metadata_manager._has_row_in_shard,
                is_row_col_sequential=self.metadata_manager._is_row_col_sequential,
            )

            for sid, parsed_rows, miss_tokens, io_time, tokenize_time, total_time in parallel_results:
                self._memory_cache_add_rows(sid, parsed_rows, miss_tokens)

                if async_write_queue is not None:
                    async_write_queue.put((sid, parsed_rows, miss_tokens))
                else:
                    self._cache_add_rows(sid, parsed_rows, torch.from_numpy(miss_tokens))

                print(
                    f"  [Shard {sid}] {len(parsed_rows):,} docs (IO {io_time:.1f}s, tok {tokenize_time:.1f}s, total {total_time:.1f}s)")

            read_time = time.time() - read_t0
            print(f"[BatchTokenize] Parquet read + tokenize (parallel): {read_time:.1f}s")

        exp_token_paths: Dict[int, str] = {}
        pack_t0 = time.time()
        n_batch = len(batch_exp_ids)

        if hasattr(self, '_global_index') and self._global_index is not None:
            all_global_ids, all_tokens_flat = self._global_index
        else:
            shard_starts = mgr._shard_starts
            global_ids_list = []
            tokens_list = []
            with self._memory_cache_lock:
                cache_snapshot = list(self._memory_cache.items())
            for sid, cache_data in cache_snapshot:
                rows = cache_data["rows"]
                tokens = cache_data["tokens"]
                seq_positions = mgr.row_col_to_local(sid, rows)
                global_ids = shard_starts[sid] + seq_positions
                global_ids_list.append(global_ids)
                tokens_list.append(tokens)

            all_global_ids = np.concatenate(global_ids_list).astype(np.int64)
            all_tokens_flat = np.concatenate(tokens_list, axis=0)
            sort_idx = np.argsort(all_global_ids)
            all_global_ids = all_global_ids[sort_idx]
            all_tokens_flat = all_tokens_flat[sort_idx]

        print(f"  [GlobalIndex] {len(all_global_ids):,} docs indexed")

        for i, (exp_id, selected_idx) in enumerate(zip(batch_exp_ids, batch_selected)):
            positions = np.searchsorted(all_global_ids, selected_idx)
            positions = np.clip(positions, 0, len(all_global_ids) - 1)
            matched = all_global_ids[positions] == selected_idx
            if not matched.all():
                n_missing = int((~matched).sum())
                raise RuntimeError(
                    f"[Pack] Experiment {exp_id}: {n_missing}/{len(selected_idx)} "
                    f"documents not found in tokenized cache. "
                    f"Check for shard tokenization failures."
                )
            result = all_tokens_flat[positions]

            if shm_store is not None:
                from multiprocessing.shared_memory import SharedMemory
                shm = SharedMemory(create=True, size=result.nbytes)
                shm_array = np.ndarray(result.shape, dtype=result.dtype, buffer=shm.buf)
                shm_array[:] = result[:]
                if shm_lock:
                    with shm_lock:
                        shm_store[exp_id] = (shm.name, result.shape, result.dtype.str)
                else:
                    shm_store[exp_id] = (shm.name, result.shape, result.dtype.str)
                shm.close()
                exp_token_paths[exp_id] = f"shm://{shm.name}"
            else:
                exp_token_path = self._get_exp_token_path(exp_id)
                np.save(exp_token_path, result)
                exp_token_paths[exp_id] = exp_token_path

            elapsed = time.time() - pack_t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_batch - i - 1) / speed if speed > 0 else 0
            print(f"  [Pack] {i+1}/{n_batch} exps ({(i+1)*100//n_batch}%), "
                  f"{elapsed:.1f}s elapsed, ETA {eta:.0f}s")

        pack_time = time.time() - pack_t0
        print(f"[BatchTokenize] Pack {n_batch} exps: {pack_time:.1f}s")

        elapsed = time.time() - t0
        print(f"[BatchTokenize] Total: {elapsed:.1f}s for {len(batch_exp_ids)} experiments")

        return exp_token_paths

    def _load_tokens_for_experiment(
            self, selected_idx: np.ndarray, exp_id: int = None,
            shm_info: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Load tokens for selected document indices."""
        if shm_info is not None:
            from multiprocessing.shared_memory import SharedMemory
            shm_name, shape, dtype_str = shm_info
            shm = SharedMemory(name=shm_name)
            shm_array = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
            result = torch.from_numpy(shm_array.copy()).long()
            shm.close()
            print(f"  [TokenLoad] exp {exp_id:04d}: {len(result):,} docs from SharedMemory")
            return result

        if exp_id is not None:
            exp_token_path = self._get_exp_token_path(exp_id)
            if os.path.exists(exp_token_path):
                print(f"  [TokenLoad] WARNING: exp {exp_id:04d} fallback to temp file, "
                      f"this should not happen after tokenize_all_needed")
                result = torch.from_numpy(np.load(exp_token_path, mmap_mode='r')).long()
                print(f"  [TokenLoad] exp {exp_id:04d}: {len(result):,} docs from temp file")
                return result

        if exp_id is not None:
            print(f"  [TokenLoad] WARNING: exp {exp_id:04d} fallback to shard cache, "
                  f"this should not happen after tokenize_all_needed")
        t0 = time.time()
        mgr = self.metadata_manager
        shard_groups = mgr.global_to_shard_rows(selected_idx)

        all_tokens = []
        for sid, (shard_path, local_rows) in shard_groups.items():
            cache_path = self._get_shard_token_path(sid)

            row_col_vals = mgr.local_to_row_col(sid, local_rows)
            row_col_int = [int(v) for v in row_col_vals]

            if os.path.exists(cache_path):
                with np.load(cache_path) as data:
                    token_data = data['tokens']
                    row_index = data['rows']
                    row_to_pos = {int(r): i for i, r in enumerate(row_index)}

                    hit_rcv = [v for v in row_col_int if v in row_to_pos]
                    miss_rcv = [v for v in row_col_int if v not in row_to_pos]

                    hit_tokens = None
                    miss_tokens = None

                    if hit_rcv:
                        positions = np.array(
                            [row_to_pos[v] for v in hit_rcv], dtype=np.int64
                        )
                        hit_tokens = torch.from_numpy(
                            token_data[positions].astype(np.int64).copy()
                        )
                        self._cache_hits += len(hit_rcv)

                if miss_rcv:
                    self._cache_misses += len(miss_rcv)
                    miss_rcv_arr = np.array(miss_rcv, dtype=np.int64)

                    from quadmix.data.metadata_manager import _read_one_shard_texts_with_rows
                    schema = self.metadata_manager.schema
                    text_col = schema.text_col
                    row_col = schema.row_in_shard_col if self.metadata_manager._has_row_in_shard else None
                    shard_total_rows = mgr._per_shard_info[sid]["num_docs"]

                    selected_texts, parsed_rows = _read_one_shard_texts_with_rows(
                        shard_path, text_col, row_col, miss_rcv_arr,
                        self.metadata_manager._has_row_in_shard,
                        mgr._is_row_col_sequential, shard_total_rows,
                    )

                    print(f"    [Partial miss] shard {sid}: "
                          f"tokenizing {len(miss_rcv):,} docs "
                          f"(hit {len(hit_rcv):,} from cache)...")

                    miss_tokens_full = self._tokenize_texts(selected_texts)
                    self._cache_add_rows(sid, parsed_rows, miss_tokens_full)

                    row_to_pos_new = {int(r): i for i, r in enumerate(parsed_rows)}
                    positions = np.array(
                        [row_to_pos_new[v] for v in miss_rcv], dtype=np.int64
                    )
                    miss_tokens = torch.from_numpy(
                        miss_tokens_full.numpy().astype(np.int64)[positions]
                    )

                if miss_rcv:
                    row_to_token = {}
                    if hit_tokens is not None:
                        for i, v in enumerate(hit_rcv):
                            row_to_token[v] = hit_tokens[i]
                    if miss_tokens is not None:
                        for i, v in enumerate(miss_rcv):
                            row_to_token[v] = miss_tokens[i]
                    shard_tokens = torch.stack([
                        row_to_token[v] for v in row_col_int
                    ])
                else:
                    shard_tokens = hit_tokens
            else:
                self._cache_misses += len(local_rows)
                from quadmix.data.metadata_manager import _read_one_shard_texts_with_rows
                schema = self.metadata_manager.schema
                text_col = schema.text_col
                row_col = schema.row_in_shard_col if self.metadata_manager._has_row_in_shard else None
                row_col_arr = np.array(row_col_int, dtype=np.int64) if row_col is not None else None
                shard_total_rows = mgr._per_shard_info[sid]["num_docs"]

                selected_texts, parsed_rows = _read_one_shard_texts_with_rows(
                    shard_path, text_col, row_col, row_col_arr,
                    self.metadata_manager._has_row_in_shard,
                    mgr._is_row_col_sequential, shard_total_rows,
                )

                print(f"    [Cache miss] shard {sid}: "
                      f"tokenizing {len(selected_texts):,} docs...")

                tokenized = self._tokenize_texts(selected_texts)
                self._cache_add_rows(sid, parsed_rows, tokenized)

                row_to_pos = {int(r): i for i, r in enumerate(parsed_rows)}
                positions = np.array(
                    [row_to_pos[v] for v in row_col_int], dtype=np.int64
                )
                shard_tokens = torch.from_numpy(
                    tokenized.numpy().astype(np.int64)[positions]
                )

            all_tokens.append(shard_tokens)

        result = torch.cat(all_tokens, dim=0)
        elapsed = time.time() - t0
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(1, total) * 100
        print(f"  [TokenLoad] {len(result)} docs loaded, "
              f"cache: {self._cache_hits}/{total} hits ({hit_rate:.0f}%) "
              f"({elapsed:.1f}s)")
        return result

    def _compute_ranks_for_params(
            self, params: ParameterSet, experiment_id: int,
    ) -> np.ndarray:
        """Per-experiment Eq.1 + Eq.2 (optimized with pre-normalized quality + pre-computed domain indices)."""
        M = self.config.num_domains
        rng = np.random.default_rng(experiment_id + 1729)

        merged_scores = np.zeros(self._num_docs, dtype=np.float64)

        for m in range(M):
            indices = self._domain_indices.get(m)
            if indices is None or len(indices) == 0:
                continue
            alpha_m = params.merge_config.get_final_weights(m)
            merged_scores[indices] = self._normalized_quality[indices] @ alpha_m

        ranks = np.zeros(self._num_docs, dtype=np.float64)
        has_tokens = hasattr(self, '_token_counts') and self._token_counts is not None
        for m in range(M):
            indices = self._domain_indices.get(m)
            if indices is None:
                continue
            n_domain = len(indices)
            if n_domain == 0:
                continue
            domain_scores = merged_scores[indices]

            k = min(self.rank_ref_size, n_domain)
            ref_idx = rng.choice(n_domain, k, replace=False)
            ref_scores_unsorted = domain_scores[ref_idx]
            sort_order = np.argsort(-ref_scores_unsorted)
            ref_scores = ref_scores_unsorted[sort_order]

            positions = np.searchsorted(-ref_scores, -domain_scores, side='right')

            if has_tokens:
                ref_tokens = self._token_counts[indices[ref_idx]][sort_order].astype(np.float64)
                cum_tokens = np.concatenate(([0.0], np.cumsum(ref_tokens)))
                total_ref_tokens = cum_tokens[-1]
                if total_ref_tokens > 0:
                    ranks[indices] = cum_tokens[positions] / total_ref_tokens
                else:
                    ranks[indices] = positions.astype(np.float64) / k
            else:
                ranks[indices] = positions.astype(np.float64) / k

        return ranks

    def _training_token_budget(self) -> int:
        """Total tokens training will consume: num_steps × global_batch_size × block_size."""
        num_steps = self.tiny_steps if self.tiny_steps > 0 else self.max_step
        return num_steps * self.global_batch_size * self.block_size

    def _subsample_for_budget(
        self, selected_idx: np.ndarray, seed: int = 0,
    ) -> np.ndarray:
        """Subsample selected_idx to match training token budget."""
        tokens_needed = self._training_token_budget()

        if self.metadata_manager is not None:
            est_tokens = np.maximum(
                self.metadata_manager.doc_char_counts[selected_idx] // 4, 1
            )
        else:
            est_tokens = np.full(len(selected_idx), self.block_size // 2, dtype=np.int64)

        total_est = int(est_tokens.sum())
        if total_est <= tokens_needed:
            return selected_idx

        avg_tok = total_est / len(selected_idx)
        docs_needed = max(100, int(tokens_needed / avg_tok * 1.2))
        if docs_needed >= len(selected_idx):
            return selected_idx

        rng = np.random.default_rng(seed)
        return rng.choice(selected_idx, size=docs_needed, replace=False)

    def _sample_one_experiment(
            self, params: ParameterSet, experiment_id: int,
    ) -> np.ndarray:
        """Process one experiment: Eq.1-3 + sampling, domain-by-domain."""
        cancel = getattr(self, '_cancel_flag', None)
        M = self.config.num_domains
        so = self._seed_offset
        rng_eq2 = np.random.default_rng(experiment_id + so + 1729)
        rng_sample = np.random.default_rng(experiment_id + so + 42)
        has_tokens = hasattr(self, '_token_counts') and self._token_counts is not None

        domain_selected = []

        for m in range(M):
            if cancel is not None and cancel.is_set():
                return np.array([], dtype=np.int64)
            indices = self._domain_indices.get(m)
            if indices is None or len(indices) == 0:
                continue
            if m >= len(params.sampling_configs):
                continue

            n_m = len(indices)

            alpha_m = params.merge_config.get_final_weights(m)
            domain_scores = self._normalized_quality[indices] @ alpha_m

            k = min(self.rank_ref_size, n_m)
            ref_idx = rng_eq2.choice(n_m, k, replace=False)
            ref_scores_unsorted = domain_scores[ref_idx]
            sort_order = np.argsort(-ref_scores_unsorted)
            ref_scores = ref_scores_unsorted[sort_order]
            positions = np.searchsorted(-ref_scores, -domain_scores, side='right')

            if has_tokens:
                ref_tokens = self._token_counts[indices[ref_idx]][sort_order].astype(np.float64)
                cum_tokens = np.concatenate(([0.0], np.cumsum(ref_tokens)))
                total_ref_tokens = cum_tokens[-1]
                if total_ref_tokens > 0:
                    domain_ranks = cum_tokens[positions] / total_ref_tokens
                else:
                    domain_ranks = positions.astype(np.float64) / k
            else:
                domain_ranks = positions.astype(np.float64) / k

            sc = params.sampling_configs[m]
            within_threshold = domain_ranks <= sc.omega
            sv = np.full(n_m, sc.epsilon, dtype=np.float64)
            if within_threshold.any():
                exponent = -sc.lambda_ * (sc.omega - domain_ranks[within_threshold])
                sigmoid = 2.0 / (1.0 + np.exp(np.clip(exponent, -100, 100)))
                sv[within_threshold] = sigmoid ** sc.eta + sc.epsilon

            int_part = np.floor(sv).astype(np.int64)
            frac_part = sv - int_part
            random_mask = rng_sample.uniform(size=n_m) < frac_part
            repeats = int_part + random_mask.astype(np.int64)
            selected_local = np.repeat(np.arange(n_m), repeats)
            if len(selected_local) > 0:
                domain_selected.append(indices[selected_local])

        if domain_selected:
            return np.concatenate(domain_selected)
        else:
            rng2 = np.random.default_rng(experiment_id + self._seed_offset + 42)
            return rng2.choice(np.arange(self._num_docs), 100, replace=False)

    def run_experiment(
            self,
            params: ParameterSet,
            experiment_id: int = 0,
            selected_idx: Optional[np.ndarray] = None,
            shm_info: Optional[tuple] = None,
            checkpoint_interval: Optional[int] = None,
            sampled_doc_count: Optional[int] = None,
    ) -> ProxyResult:
        """Train one proxy model. Validates on openhermes-10k."""

        from quadmix.core.proxy_model import ProxyModel
        from quadmix.npu.device import DeviceManager, DeviceType

        if checkpoint_interval is None:
            checkpoint_interval = self.checkpoint_interval

        os.makedirs(self.output_dir, exist_ok=True)
        exp_dir = os.path.join(self.output_dir, f"exp_{experiment_id:04d}")
        os.makedirs(exp_dir, exist_ok=True)

        device_mgr = DeviceManager(
            device_type=DeviceType(self.device_type),
            npu_device_id=self.npu_device_id,
        )
        device = device_mgr.get_device()

        selected_idx_out: Optional[np.ndarray] = None
        if selected_idx is None:
            quality_ranks = self._compute_ranks_for_params(params, experiment_id)
            sv = compute_sampling_values(quality_ranks, self._domain_labels, params)
            train_sv = sv[self._train_idx]

            int_part = np.floor(train_sv).astype(np.int64)
            frac_part = train_sv - int_part
            rng = np.random.default_rng(experiment_id + 42)
            random_mask = rng.uniform(size=len(train_sv)) < frac_part
            repeats = int_part + random_mask.astype(np.int64)
            selected_idx_out = self._train_idx[
                np.repeat(np.arange(len(self._train_idx)), repeats)
            ]
            if len(selected_idx_out) < 10:
                rng2 = np.random.default_rng(experiment_id + 42)
                selected_idx_out = rng2.choice(self._train_idx, 100, replace=False)

            sampled_doc_count = len(selected_idx_out)
            selected_idx_out = self._subsample_for_budget(selected_idx_out, seed=experiment_id + self._seed_offset)
        else:
            selected_idx_out = np.asarray(selected_idx, dtype=np.int64)

        if sampled_doc_count is None:
            sampled_doc_count = len(selected_idx_out)

        print(f"  [Exp {experiment_id:04d}] QuaDMix sampled {sampled_doc_count} docs "
              f"(from {len(self._train_idx):,}), training with {len(selected_idx_out)}")

        _timer_prefix = f"exp{experiment_id:04d}"

        with PerfTimer.section("load_tokens", _timer_prefix):
            train_tokens = self._load_tokens_for_experiment(selected_idx_out, exp_id=experiment_id, shm_info=shm_info)

        with PerfTimer.section("create_model", _timer_prefix):
            model = ProxyModel(config=self.model_config).to(device)
            if device.type == "npu":
                model = model.to(torch.bfloat16)
                print(f"  [Exp {experiment_id:04d}] Model converted to bf16 for NPU")

        if self.device_type != "npu" and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='max-autotune')
                print(f"  [Exp {experiment_id:04d}] Model compiled with torch.compile")
            except Exception as e:
                print(f"  [Exp {experiment_id:04d}] torch.compile failed: {e}, using eager mode")

        non_emb = model.count_params(non_embedding_only=True)
        print(f"  [Exp {experiment_id:04d}] Model on {device}: "
              f"{model.count_params():,} total, {non_emb:,} non-emb")

        use_fused = self.device_type == "cuda"
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate,
            betas=(0.9, 0.95), weight_decay=self.weight_decay, fused=use_fused,
        )

        with PerfTimer.section("data_prep", _timer_prefix):
            pad_id = self.tokenizer.pad_token_id
            eos_id = self.tokenizer.eos_token_id
            real_mask = train_tokens != pad_id
            non_empty = real_mask.any(dim=1)
            real_tokens_list = []
            eos_buf = torch.tensor([eos_id], dtype=train_tokens.dtype)
            for doc in train_tokens[non_empty]:
                real_tokens_list.append(doc[doc != pad_id])
                real_tokens_list.append(eos_buf)
            flat_train = torch.cat(real_tokens_list)
            del train_tokens, real_tokens_list, real_mask, non_empty, eos_buf

        flat_train_cpu = flat_train
        del flat_train

        num_steps = self.tiny_steps if self.tiny_steps > 0 else self.max_step
        grad_acc = self.gradient_accumulation_steps
        max_iters = num_steps * grad_acc
        tok_per_step = self.micro_batch_size * self.block_size

        warmup_steps = max(1, int(num_steps * self.warmup_fraction))
        self.warmup_steps = warmup_steps

        total_blocks = max(1, flat_train_cpu.size(0) - self.block_size)

        epoch_rng = np.random.default_rng(experiment_id + self._seed_offset + 42)

        def get_epoch_permutation():
            return epoch_rng.permutation(total_blocks)

        perm = get_epoch_permutation()
        epoch_pos = 0
        epoch = 0

        accum_bs = self.micro_batch_size * grad_acc
        batch_buf = torch.empty(accum_bs, self.block_size + 1, dtype=torch.long, device=device)
        arange_cpu = torch.arange(self.block_size + 1, dtype=torch.long)

        if device.type == "npu":
            torch.npu.empty_cache()

        _train_t0 = time.perf_counter()
        model.train()
        loss_accum = torch.tensor(0.0, device=device)
        self._ckpt_results = {}
        iter_ct = 0
        step_ct = 0
        log_int = max(1, num_steps // 5)
        t_start = time.time()

        print(f"  [Exp {experiment_id:04d}] Training {num_steps} steps "
              f"(grad_acc={grad_acc}, warmup={warmup_steps} steps ({self.warmup_fraction * 100:.0f}%), "
              f"micro_batch={self.micro_batch_size}, "
              f"blocks={total_blocks}, epochs~{math.ceil(max_iters / total_blocks)})"
              f"{', checkpoint every ' + str(checkpoint_interval) + ' steps' if checkpoint_interval > 0 else ''})...")

        while iter_ct < max_iters:
            micro_in_step = iter_ct % grad_acc
            mb_start = micro_in_step * self.micro_batch_size
            mb_end = mb_start + self.micro_batch_size

            if micro_in_step == 0:
                remaining = total_blocks - epoch_pos
                if remaining >= accum_bs:
                    block_starts_cpu = perm[epoch_pos:epoch_pos + accum_bs].copy()
                    epoch_pos += accum_bs
                else:
                    block_starts_cpu = np.empty(accum_bs, dtype=np.int64)
                    block_starts_cpu[:remaining] = perm[epoch_pos:epoch_pos + remaining]
                    filled = remaining
                    while filled < accum_bs:
                        perm = get_epoch_permutation()
                        epoch_pos = 0
                        epoch += 1
                        chunk = min(accum_bs - filled, total_blocks)
                        block_starts_cpu[filled:filled + chunk] = perm[:chunk]
                        filled += chunk
                        epoch_pos = chunk

                block_starts_cpu_tensor = torch.from_numpy(block_starts_cpu)
                idx_cpu = block_starts_cpu_tensor.unsqueeze(1) + arange_cpu.unsqueeze(0)
                batch_cpu = flat_train_cpu[idx_cpu]
                batch_buf.copy_(batch_cpu.to(device))
                del batch_cpu

            batch = batch_buf[mb_start:mb_end]
            inp = batch[:, :self.block_size].contiguous()
            tgt = batch[:, 1:self.block_size + 1].contiguous()

            hidden = model(inp, return_hidden=True)
            loss = chunked_loss_from_hidden(model, hidden, tgt, chunk_size=2048)

            is_acc = (iter_ct + 1) % grad_acc != 0
            (loss / grad_acc).backward()

            if not is_acc:
                lr = self._lr_schedule(iter_ct, max_iters)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_ct += 1

                if checkpoint_interval > 0 and step_ct % checkpoint_interval == 0 and step_ct < num_steps:
                    if device.type == "npu":
                        torch.npu.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
                    ckpt_val, _ = self._run_validation(model, device)
                    self._ckpt_results[step_ct] = ckpt_val
                    elapsed_ckpt = time.time() - t_start
                    print(f"    [Exp {experiment_id:04d}] [CHECKPOINT step={step_ct}] val_loss={ckpt_val:.4f} ({elapsed_ckpt:.0f}s)")

            loss_accum += loss.detach()
            iter_ct += 1

            if not is_acc and (step_ct % log_int == 0 or step_ct == 1):
                avg = (loss_accum / iter_ct).item()
                elapsed = time.time() - t_start
                rem = (num_steps - step_ct) * elapsed / max(1, step_ct)
                print(f"    [Exp {experiment_id:04d}] Step {step_ct}/{num_steps}, loss={avg:.4f}, "
                      f"lr={lr:.2e}, {tok_per_step * step_ct / elapsed:.0f} tok/s, "
                      f"ETA: {rem:.0f}s")

        _train_elapsed = time.perf_counter() - _train_t0
        PerfTimer._timings.setdefault(f"{_timer_prefix}.training_loop", []).append(_train_elapsed)

        with PerfTimer.section("free_resources", _timer_prefix):
            del flat_train_cpu, batch_buf, optimizer, perm
            if device.type == "npu":
                import gc as _gc
                _gc.collect()
                torch.npu.empty_cache()
            elif device.type == "cuda":
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()

        with PerfTimer.section("validation", _timer_prefix):
            val_loss, per_task_losses = self._run_validation(model, device)

        with PerfTimer.section("save_metadata", _timer_prefix):
            avg_train = (loss_accum / iter_ct).item() if iter_ct > 0 else 0

            M = params.num_domains
            N = params.num_criteria
            domain_names = self._domain_names or [f"domain_{m}" for m in range(M)]
            quality_names = self._quality_names or [f"criterion_{n}" for n in range(N)]
            dw = params.merge_config.domain_weights

            quality_weights = {}
            for m in range(M):
                start = m * N
                quality_weights[domain_names[m]] = {
                    quality_names[n]: round(float(dw[start + n]), 6) for n in range(N)
                }

            sampling_params = {}
            for m, sc in enumerate(params.sampling_configs):
                sampling_params[domain_names[m]] = {
                    "lambda": round(sc.lambda_, 4),
                    "omega": round(sc.omega, 6),
                    "eta": round(sc.eta, 6),
                    "epsilon": round(sc.epsilon, 6),
                }

            meta = {
                "experiment_id": experiment_id,
                "variant": self.model_variant,
                "train_loss": avg_train,
                "val_loss": val_loss,
                "val_ppl": float(np.exp(val_loss)),
                "num_steps": step_ct,
                "sampled_docs": sampled_doc_count,
                "training_docs": len(selected_idx_out),
                "val_docs": len(self._val_token_ids),
                "assistant_loss": True,
                "quality_weights": quality_weights,
                "sampling_params": sampling_params,
                "checkpoint_steps": dict(self._ckpt_results) if hasattr(self, '_ckpt_results') else {},
                "training_config": {
                    "global_batch_size": self.global_batch_size,
                    "micro_batch_size": self.micro_batch_size,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "warmup_fraction": self.warmup_fraction,
                    "grad_clip": self.grad_clip,
                    "block_size": self.block_size,
                    "device_type": self.device_type,
                    "precision": "bf16" if self.device_type == "npu" else "fp32",
                    "attention_type": "flash" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "explicit",
                    "normalizer": getattr(self, "_normalizer_name", "unknown"),
                },
                "timing": {
                    "training_elapsed_s": _train_elapsed,
                    "total_elapsed_s": time.perf_counter() - _train_t0,
                },
            }
            if per_task_losses is not None:
                meta["per_task_losses"] = per_task_losses
            np.save(os.path.join(exp_dir, "selected_indices.npy"), selected_idx_out)
            with open(os.path.join(exp_dir, "meta.json"), "w") as f:
                json.dump(sanitize_for_json(meta), f, indent=2)
            torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))

            ckpt_results = dict(self._ckpt_results) if hasattr(self, '_ckpt_results') else {}
            if ckpt_results:
                ckpt_trajectory = {
                    "experiment_id": experiment_id,
                    "checkpoint_interval": checkpoint_interval,
                    "num_steps": step_ct,
                    "final_val_loss": val_loss,
                    "checkpoints": {str(k): v for k, v in sorted(ckpt_results.items())},
                }
                ckpt_path = os.path.join(exp_dir, "checkpoint_trajectory.json")
                with open(ckpt_path, "w") as f:
                    json.dump(sanitize_for_json(ckpt_trajectory), f, indent=2)

        print(f"  [Exp {experiment_id:04d}] Done. train_loss={avg_train:.4f}, "
              f"val_loss={val_loss:.4f} (ppl={np.exp(val_loss):.1f})")
        with PerfTimer.section("free_npu", _timer_prefix):
            del model
            if device.type == "npu":
                import gc
                gc.collect()
                torch.npu.empty_cache()

        return ProxyResult(parameters=params, validation_loss=val_loss, metadata=meta, per_task_losses=per_task_losses)

    def _run_validation(self, model, device) -> tuple[float, dict[str, float] | None]:
        """Run validation on full validation set.
        
        Returns:
            Tuple of (aggregate_loss, per_task_losses or None)
        """
        model.eval()
        bs = self.block_size
        val_n = len(self._val_token_ids)
        val_tokens = self._val_token_ids[:val_n, :bs].to(device)
        val_mask = self._val_loss_mask[:val_n, :bs].to(device)

        val_bs, val_chunk_size = compute_val_batch_size(
            device, model.config.vocab_size, bs - 1, max_val_bs=96,
        )
        val_bs = min(val_bs, val_n, 8)

        if val_bs < 96 or val_chunk_size < 2048:
            if device.type in ("npu", "cuda"):
                api = torch.npu if device.type == "npu" else torch.cuda
                total_mem = api.get_device_properties(device).total_memory
                allocated = api.memory_allocated(device)
                print(f"    [Validation] val_bs={val_bs}, chunk_size={val_chunk_size} "
                      f"(auto: {device.type}, {total_mem / 1e9:.1f}GB total, "
                      f"{allocated / 1e9:.2f}GB used)")

        with torch.no_grad():
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
            if self._val_task_labels is not None:
                per_task_losses = {}
                for task in sorted(set(self._val_task_labels)):
                    task_indices = [i for i, t in enumerate(self._val_task_labels) if t == task]
                    if task_indices:
                        task_loss = float(all_losses[task_indices].mean())
                        per_task_losses[task] = task_loss
        
        del val_tokens, val_mask, per_doc_losses, all_losses
        if device.type == "npu":
            torch.npu.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        model.train()
        return val_loss, per_task_losses

    def revalidate_from_saved(
        self, model_path: str, device_type: str = "cpu",
    ) -> tuple[float, dict[str, float] | None]:
        from quadmix.core.proxy_model import ProxyModel
        from quadmix.npu.device import DeviceManager, DeviceType
        device_mgr = DeviceManager(device_type=DeviceType(device_type))
        device = device_mgr.get_device()
        model = ProxyModel(config=self.model_config).to(device)
        if device.type == "npu":
            model = model.to(torch.bfloat16)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        val_loss, per_task_losses = self._run_validation(model, device)
        del model, state_dict
        if device.type == "npu":
            import gc
            gc.collect()
            torch.npu.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        return val_loss, per_task_losses

    def revalidate_batch_parallel(
        self,
        model_paths: list[str],
        device_type: str = "cuda",
        num_gpus: int | None = None,
        on_result: Optional[Callable[[int, float, Optional[Dict[str, float]]], None]] = None,
    ) -> list[tuple[float, dict[str, float] | None]]:
        """Revalidate multiple models in parallel across GPUs.

        Args:
            model_paths: list of saved model weight paths
            device_type: 'cuda' or 'npu'
            num_gpus: number of GPUs to use (default: all available)
            on_result: optional callback(idx, val_loss, per_task_losses)
                       called as each result arrives (for incremental saving)

        Returns:
            list of (val_loss, per_task_losses) in same order as model_paths
        """
        from quadmix.pipeline.parallel_dispatch import _reval_worker

        n_models = len(model_paths)

        if device_type == "cuda":
            available = torch.cuda.device_count()
        elif device_type == "npu":
            available = torch.npu.device_count()
        else:
            available = 1

        if num_gpus is None:
            num_gpus = max(1, available)
        num_gpus = min(num_gpus, available, n_models)

        if num_gpus <= 1:
            results = []
            for i, mpath in enumerate(model_paths):
                r = self.revalidate_from_saved(mpath, device_type=device_type)
                results.append(r)
                if on_result is not None:
                    on_result(i, r[0], r[1])
                if (i + 1) % 50 == 0 or i == n_models - 1:
                    print(f"  [{i+1}/{n_models}] sequential reval done")
            return results

        print(f"[RevalParallel] {n_models} models on {num_gpus} GPUs ({device_type})")

        model_config_dict = {
            "n_layer": self.model_config.n_layer,
            "n_head": self.model_config.n_head,
            "n_embd": self.model_config.n_embd,
            "vocab_size": self.model_config.vocab_size,
            "padding_multiple": self.model_config.padding_multiple,
            "block_size": self.model_config.block_size,
            "bias": self.model_config.bias,
            "norm_eps": self.model_config.norm_eps,
            "rope_base": self.model_config.rope_base,
            "intermediate_size": self.model_config.intermediate_size,
        }

        worker_models: list[list[str]] = [[] for _ in range(num_gpus)]
        worker_indices: list[list[int]] = [[] for _ in range(num_gpus)]
        for i, mpath in enumerate(model_paths):
            w = i % num_gpus
            worker_models[w].append(mpath)
            worker_indices[w].append(i)

        for w in range(num_gpus):
            print(f"  [RevalParallel] GPU {w}: {len(worker_models[w])} models")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        processes = []
        for w in range(num_gpus):
            device_str = f"{device_type}:{w}"
            p = ctx.Process(
                target=_reval_worker,
                args=(
                    w, device_str, self.val_data_path,
                    model_config_dict, worker_models[w],
                    worker_indices[w], result_queue,
                ),
            )
            p.start()
            processes.append(p)

        all_results: list[tuple[float, dict[str, float] | None] | None] = [None] * n_models
        finished_workers = 0
        collected = 0

        while finished_workers < num_gpus:
            item = result_queue.get()
            if item is None:
                finished_workers += 1
            else:
                exp_idx, val_loss, per_task_losses = item
                all_results[exp_idx] = (val_loss, per_task_losses)
                collected += 1
                if on_result is not None:
                    on_result(exp_idx, val_loss, per_task_losses)
                if collected % 50 == 0 or collected == n_models:
                    print(f"  [RevalParallel] {collected}/{n_models} collected")

        for p in processes:
            p.join()

        failed = [i for i, r in enumerate(all_results) if r is None]
        if failed:
            raise RuntimeError(
                f"[RevalParallel] {len(failed)} models failed: {failed[:10]}..."
            )

        print(f"[RevalParallel] Done: {n_models} models on {num_gpus} GPUs")
        return all_results

    def _lr_schedule(self, it: int, max_iters: int) -> float:
        """Cosine LR with linear warmup."""
        warm = self.warmup_steps * self.gradient_accumulation_steps
        if it < warm:
            return self.learning_rate * it / max(1, warm)
        if it > max_iters:
            return self.learning_rate * 0.025
        dr = (it - warm) / max(1, max_iters - warm)
        coeff = 0.5 * (1.0 + math.cos(math.pi * dr))
        return self.learning_rate * 0.025 + coeff * (
                self.learning_rate - self.learning_rate * 0.025
        )

    def save_summary(self, results: List[ProxyResult], path: str):
        summary = {
            "num_experiments": len(results),
            "experiments": [{
                "id": r.metadata["experiment_id"],
                "val_loss": r.validation_loss,
                "params_flattened": r.parameters.flatten().tolist(),
                **r.metadata,
            } for r in results],
            "mean_val_loss": float(np.mean([r.validation_loss for r in results])),
            "std_val_loss": float(np.std([r.validation_loss for r in results])),
        }
        with open(path, "w") as f:
            json.dump(sanitize_for_json(summary), f, indent=2)
        print(f"[ProxyRunner] Summary: {path}")

    def precompute_samples(
            self, all_params: List[ParameterSet]
    ) -> List[np.ndarray]:
        """Run Eq.1-3 + fractional sampling for all experiments."""
        from quadmix.pipeline.shared_memory import ndarray_to_shared, SharedArrayInfo

        cfg = self._concurrency
        t0 = time.time()
        n = len(all_params)
        num_cpus = cfg.cpu_count
        seed_offset = self._seed_offset

        q_cols = self._normalized_quality.shape[1]
        q_itemsize = self._normalized_quality.dtype.itemsize
        shm_data_size = (
            sum(len(idx) * q_cols * q_itemsize for idx in self._domain_indices.values())
            + sum(idx.nbytes for idx in self._domain_indices.values())
        )
        if self._token_counts is not None:
            shm_data_size += sum(len(idx) * self._token_counts.dtype.itemsize for idx in self._domain_indices.values())
        n_m_max = max((len(idx) for idx in self._domain_indices.values()), default=self._num_docs)
        compute_mem = 6 * n_m_max * 8
        mem_per_process = compute_mem + 50 * 1024**2

        available_ram = get_available_memory_bytes() * 0.7
        shm_reserved = min(shm_data_size, available_ram * 0.1)
        compute_budget = available_ram - shm_reserved
        max_by_mem = max(1, int(compute_budget / mem_per_process))
        worker_cap = max(1, int(os.environ.get("PRESAMPLE_MAX_WORKERS", "64")))
        n_workers = min(n, num_cpus, max_by_mem, worker_cap)

        blas_threads = cfg.blas_threads_for(n_workers)
        print(
            f"[PreSample] Pre-sampling {n} experiments (Eq.1-3) "
            f"with {n_workers} processes × {blas_threads} BLAS threads "
            f"(={n_workers * blas_threads} effective, "
            f"shm={shm_data_size/1024**2:.0f}MB (shared), "
            f"mem/process={mem_per_process/1024**2:.0f}MB "
            f"[compute={compute_mem/1024**2:.0f}MB + overhead=50MB], "
            f"available={available_ram/1024**3:.0f}GB)"
        )

        shm_blocks = []
        sq_infos = {}
        t_pre = time.time()
        for m, idx in self._domain_indices.items():
            sub_q = np.ascontiguousarray(self._normalized_quality[idx])
            info = ndarray_to_shared(sub_q, f"presample_sq_{m}")
            sq_infos[m] = info
            shm_blocks.append(info.name)
        di_infos = {}
        for m, idx in self._domain_indices.items():
            info = ndarray_to_shared(idx, f"presample_di_{m}")
            di_infos[m] = info
            shm_blocks.append(info.name)
        has_tc = self._token_counts is not None
        stc_infos = {}
        if has_tc:
            for m, idx in self._domain_indices.items():
                sub_tc = np.ascontiguousarray(self._token_counts[idx])
                info = ndarray_to_shared(sub_tc, f"presample_stc_{m}")
                stc_infos[m] = info
                shm_blocks.append(info.name)
        pre_elapsed = time.time() - t_pre
        print(f"[PreSample] Pre-extraction (contiguous per-domain): {pre_elapsed:.1f}s")

        data = _PreSampleData(
            sub_quality_infos=sq_infos,
            domain_indices_infos=di_infos,
            sub_token_counts_infos=stc_infos,
            has_token_counts=has_tc,
            num_domains=self.config.num_domains,
            rank_ref_size=self.rank_ref_size,
            num_docs=self._num_docs,
            seed_offset=seed_offset,
        )

        orig_env = {}
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            orig_env[k] = os.environ.get(k)
        os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
        os.environ["OMP_NUM_THREADS"] = str(blas_threads)
        os.environ["MKL_NUM_THREADS"] = str(blas_threads)

        all_selected: List[Optional[np.ndarray]] = [None] * n
        completed = 0
        try:
            with PerfTimer.section("eq123_sampling", "precompute"):
                results = Parallel(
                    n_jobs=n_workers,
                    backend="loky",
                    prefer="processes",
                    verbose=0,
                    return_as="generator",
                    max_nbytes=None,
                )(
                    delayed(_presample_one)((i, params, data))
                    for i, params in enumerate(all_params)
                )
                for idx, arr in results:
                    all_selected[idx] = arr
                    completed += 1
                    if completed % max(1, n // 10) == 0 or completed == n:
                        elapsed = time.time() - t0
                        speed = completed / elapsed if elapsed > 0 else 0
                        eta = (n - completed) / speed if speed > 0 else 0
                        print(f"[PreSample] {completed}/{n} done ({elapsed:.1f}s, "
                              f"{speed:.1f} exp/s, ETA: {eta:.0f}s")
        except KeyboardInterrupt:
            print("\n[PreSample] Interrupted — loky will cancel remaining workers")
            raise
        finally:
            for k, v in orig_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            for shm_name in shm_blocks:
                try:
                    shm = mp.shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass

        elapsed = time.time() - t0
        total_docs = sum(len(s) for s in all_selected)
        print(f"[PreSample] {n} experiments pre-sampled in {elapsed:.1f}s "
              f"({n / elapsed:.1f} exp/s)")
        print(f"[PreSample] Total selected docs: {total_docs:,} "
              f"(avg {total_docs // max(1, n):,}/exp)")

        budget_tokens = self._training_token_budget()
        print(f"[PreSample] Training budget: {budget_tokens:,} tokens "
              f"({self.tiny_steps if self.tiny_steps > 0 else self.max_step} steps × "
              f"{self.global_batch_size} GBS × {self.block_size} BS)")

        all_selected_train: List[np.ndarray] = []
        total_train_docs = 0
        total_sampled_docs = 0
        for i, sel in enumerate(all_selected):
            train_sel = self._subsample_for_budget(sel, seed=i + seed_offset)
            all_selected_train.append(train_sel)
            total_train_docs += len(train_sel)
            total_sampled_docs += len(sel)
        self._all_selected_train = all_selected_train

        if total_train_docs < total_sampled_docs:
            reduction = (1 - total_train_docs / total_sampled_docs) * 100
            print(f"[PreSample] Token budget subsample: {total_sampled_docs:,} → "
                  f"{total_train_docs:,} docs ({reduction:.1f}% reduction, "
                  f"avg {total_train_docs // max(1, n):,}/exp)")
        else:
            print(f"[PreSample] All sampled docs fit within training budget")

        with PerfTimer.section("collect_unique", "precompute"):
            if self.metadata_manager is not None:
                all_unique = np.unique(np.concatenate(all_selected_train))
                shard_groups = self.metadata_manager.global_to_shard_rows(all_unique)
                self._per_shard_needed_rows = {
                    sid: rows for sid, (_, rows) in shard_groups.items()
                }
                unique_ratio = len(all_unique) / max(1, self._num_docs) * 100
                print(f"[PreSample] Unique docs to tokenize: {len(all_unique):,} "
                      f"({unique_ratio:.1f}% of pool) "
                      f"across {len(shard_groups)} shards")
                avg_per_shard = len(all_unique) // max(1, len(shard_groups))
                print(f"[PreSample] Avg {avg_per_shard:,} docs/shard to tokenize "
                      f"(vs {self.metadata_manager._per_shard_info[0]['num_docs']:,} full)")

        return all_selected

    def tokenize_all_needed(self, all_selected: List[np.ndarray]):
        """Tokenize all documents needed by all experiments (union)."""
        if self._mode != "sharded":
            print("[TokenizeAll] Legacy mode: tokens already loaded")
            return

        mgr = self.metadata_manager
        t0 = time.time()

        tokenize_source = getattr(self, '_all_selected_train', all_selected)

        with PerfTimer.section("collect_union", "tokenize_all"):
            all_unique = np.unique(np.concatenate(tokenize_source))
            shard_groups = mgr.global_to_shard_rows(all_unique)

        total_tokenized = 0
        total_cached = 0

        print(f"\n[TokenizeAll] Union: {len(all_unique):,} unique docs across {len(shard_groups)} shards")

        shard_miss_info = []
        shard_miss_meta = {}

        with PerfTimer.section("check_cache", "tokenize_all"):
            for sid, (shard_path, local_rows) in shard_groups.items():
                cached_rows = self._cached_shard_rows(sid)
                memory_cached = self._memory_cache_get_rows(sid)

                row_col_vals = mgr.local_to_row_col(sid, local_rows)
                row_col_int = [int(v) for v in row_col_vals]
                hit_rows = [v for v in row_col_int if v in cached_rows or v in memory_cached]
                miss_row_col = [v for v in row_col_int if v not in cached_rows and v not in memory_cached]

                total_cached += len(hit_rows)

                if miss_row_col:
                    miss_row_col_arr = np.array(sorted(miss_row_col), dtype=np.int64)
                    shard_total_rows = mgr._per_shard_info[sid]["num_docs"]
                    shard_miss_info.append((sid, shard_path, miss_row_col_arr.tolist(),
                                           mgr._is_row_col_sequential, shard_total_rows))
                    shard_miss_meta[sid] = len(miss_row_col)

        if shard_miss_info:
            print(f"[TokenizeAll] {sum(shard_miss_meta.values()):,} miss rows across {len(shard_miss_info)} shards, parallel tokenizing...")

            with PerfTimer.section("parallel_tokenize", "tokenize_all"):
                schema = self.metadata_manager.schema
                parallel_results = _tokenize_shard_parallel(
                    shard_miss_info, self.tokenizer.name_or_path, self.block_size,
                    text_col=schema.text_col,
                    row_in_shard_col=schema.row_in_shard_col,
                    has_row_in_shard=self.metadata_manager._has_row_in_shard,
                    is_row_col_sequential=self.metadata_manager._is_row_col_sequential,
                )

            with PerfTimer.section("cache_results", "tokenize_all"):
                for sid, parsed_rows, miss_tokens, io_time, tokenize_time, total_time in parallel_results:
                    self._memory_cache_add_rows(sid, parsed_rows, miss_tokens, skip_eviction=True)
                    total_tokenized += len(parsed_rows)
        else:
            print(f"[TokenizeAll] All {len(shard_groups)} shards fully cached, 0 miss rows")

        elapsed = time.time() - t0
        print(f"[TokenizeAll] Done: {total_tokenized:,} new docs tokenized, "
              f"{total_cached:,} from cache ({elapsed:.1f}s)")

        pack_t0 = time.time()
        shard_starts = mgr._shard_starts
        with self._memory_cache_lock:
            cache_snapshot2 = list(self._memory_cache.items())
        global_ids_list = []
        tokens_list = []
        for sid, cache_data in cache_snapshot2:
            rows = cache_data["rows"]
            tokens = cache_data["tokens"]
            seq_positions = mgr.row_col_to_local(sid, rows)
            global_ids = shard_starts[sid] + seq_positions
            global_ids_list.append(global_ids)
            tokens_list.append(tokens)

        n_docs_est = sum(len(g) for g in global_ids_list)
        print(f"[TokenizeAll] Concatenating {len(global_ids_list)} shards "
              f"({n_docs_est:,} docs)...", flush=True)
        all_global_ids = np.concatenate(global_ids_list).astype(np.int64)
        all_tokens_flat = np.concatenate(tokens_list, axis=0)
        del global_ids_list, tokens_list

        freed_gb = self._memory_cache_bytes / (1024 ** 3)
        with self._memory_cache_lock:
            self._memory_cache.clear()
            self._memory_cache_bytes = 0
        print(f"[TokenizeAll] Concatenated, freed {freed_gb:.1f} GB from memory cache. "
              f"Sorting global IDs...", flush=True)

        sort_idx = np.argsort(all_global_ids)
        sorted_global_ids = all_global_ids[sort_idx]
        self._global_index = (sorted_global_ids, all_tokens_flat, sort_idx)
        del all_global_ids

        elapsed = time.time() - pack_t0
        index_gb = all_tokens_flat.nbytes / (1024 ** 3)
        print(f"[TokenizeAll] Global index built: {len(sorted_global_ids):,} docs "
              f"({elapsed:.1f}s), index: {index_gb:.1f} GB "
              f"(freed {freed_gb:.1f} GB cache, peak reduced from 3X to 2X)")
        print(f"[TokenizeAll] All {len(all_selected)} experiments ready")

    def _serialize_config(self, shared_metadata: Optional[Dict[str, "SharedArrayInfo"]] = None) -> dict:
        """Pickle-safe config for worker processes."""
        cfg = {
            "config": self.config,
            "val_data_path": self.val_data_path,
            "preprocessed_dir": (
                self.metadata_manager._dir if self.metadata_manager else None
            ),
            "output_dir": self.output_dir,
            "device_type": self.device_type,
            "model_variant": self.model_variant,
            "global_batch_size": self.global_batch_size,
            "micro_batch_size": self.micro_batch_size,
            "max_step": self.max_step,
            "warmup_fraction": self.warmup_fraction,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "tiny_steps": self.tiny_steps,
            "doc_limit": self.doc_limit,
            "test_block_size": self.block_size,
            "rank_ref_size": self.rank_ref_size,
            "checkpoint_interval": self.checkpoint_interval,
            "token_cache_dir": self.token_cache_dir,
            "shared_domain_labels": shared_metadata.get("domain_labels") if shared_metadata else None,
            "shared_quality_scores": shared_metadata.get("quality_scores") if shared_metadata else None,
            "shared_doc_char_counts": shared_metadata.get("doc_char_counts") if shared_metadata else None,
            "shared_normalized_quality": shared_metadata.get("normalized_quality") if shared_metadata else None,
            "per_shard_info": self.metadata_manager.shard_info if self.metadata_manager else None,
            "domain_names": self._domain_names,
            "quality_names": self._quality_names,
            "quality_directions": self._quality_directions,
            "num_domains": self.metadata_manager.num_domains if self.metadata_manager else None,
            "num_quality_criteria": self.metadata_manager.num_quality_criteria if self.metadata_manager else None,
            "detected_domain_names": self.metadata_manager.detected_domain_names if self.metadata_manager else None,
            "detected_quality_names": self.metadata_manager.detected_quality_names if self.metadata_manager else None,
            "domain_label_map": self.metadata_manager.domain_label_map if self.metadata_manager else None,
            "schema": self.metadata_manager.schema if self.metadata_manager else None,
        }
        return cfg

    def run_batch_parallel(
            self,
            params_list: List[ParameterSet],
            all_selected: List[np.ndarray],
            num_workers: int = 1,
            device_type: str = "npu",
            tokenize_lookahead: int = None,
    ) -> List:
        """Run proxy experiments in parallel using dynamic task queue."""
        n_exp = len(params_list)
        assert len(all_selected) == n_exp, (
            f"all_selected length {len(all_selected)} != "
            f"params_list length {n_exp}"
        )

        if tokenize_lookahead is None:
            tokenize_lookahead = num_workers * 2

        if num_workers <= 1:
            all_selected_train = getattr(self, '_all_selected_train', all_selected)
            results = []
            for i, (params, sel, sel_train) in enumerate(
                zip(params_list, all_selected, all_selected_train)
            ):
                r = self.run_experiment(
                    params, experiment_id=i, selected_idx=sel_train,
                    sampled_doc_count=len(sel),
                )
                results.append(r)
            return results

        return self._run_batch_dynamic(
            params_list, all_selected, num_workers, device_type, tokenize_lookahead
        )

    def _run_batch_dynamic(
            self,
            params_list: List[ParameterSet],
            all_selected: List[np.ndarray],
            num_workers: int,
            device_type: str,
            tokenize_lookahead: int = None,
    ) -> List:
        """Dynamic task queue mode: workers pull tasks on-demand."""
        import threading
        import queue as thread_queue

        if tokenize_lookahead is None:
            tokenize_lookahead = num_workers * 2

        n_exp = len(params_list)
        all_results = [None] * n_exp
        shared_meta: Optional[Dict[str, "SharedArrayInfo"]] = None
        if self._mode == "sharded" and self.metadata_manager is not None:
            try:
                shared_meta = {}
                mgr = self.metadata_manager
                shared_meta["domain_labels"] = ndarray_to_shared(
                    mgr.domain_labels, f"dl_{os.getpid()}")
                shared_meta["quality_scores"] = ndarray_to_shared(
                    mgr.quality_scores, f"qs_{os.getpid()}")
                shared_meta["doc_char_counts"] = ndarray_to_shared(
                    mgr.doc_char_counts, f"cc_{os.getpid()}")
                if hasattr(self, '_normalized_quality') and self._normalized_quality is not None:
                    shared_meta["normalized_quality"] = ndarray_to_shared(
                        self._normalized_quality, f"nq_{os.getpid()}")
                total_gb = sum(v.nbytes for v in shared_meta.values()) / (1024 ** 3)
                print(f"[SharedMem] Packed {total_gb:.1f} GB metadata for {num_workers} workers")
            except Exception as e:
                print(f"[SharedMem] WARNING: shared memory setup failed ({e}), "
                      f"workers will reload from disk")
                shared_meta = None

        config_ser = self._serialize_config(shared_meta)
        t_start = time.time()

        print(f"\n[DynamicParallel] {n_exp} experiments, {num_workers} workers")
        print(f"[DynamicParallel] tokenize_lookahead={tokenize_lookahead} (batch union mode)")
        print(f"[DynamicParallel] NPU Parallel Mode: SharedMemory + memory_cache + AsyncWrite enabled")

        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue(maxsize=max(n_exp, num_workers * 2))
        result_queue = ctx.Queue()

        async_write_queue = thread_queue.Queue()

        ready_events: Dict[int, bool] = {}
        ready_cond = threading.Condition()
        completed_count = 0

        exp_shm_info: Dict[int, tuple] = {}
        exp_shm_lock = threading.Lock()

        all_selected_train = getattr(self, '_all_selected_train', all_selected)

        def async_write_thread_func():
            """Write shard cache to disk in background, non-blocking."""
            write_count = 0
            while True:
                try:
                    item = async_write_queue.get(timeout=1.0)
                    if item is None:
                        break
                    sid, rows, tokens = item
                    self._cache_add_rows(sid, rows, torch.from_numpy(tokens))
                    write_count += 1
                except thread_queue.Empty:
                    continue

            print(f"[AsyncWrite] Wrote {write_count} shard caches to disk")

        async_write_thread = threading.Thread(target=async_write_thread_func, daemon=True)
        async_write_thread.start()

        def tokenize_thread_func():
            """Continuously pre-tokenize experiments in BATCH UNION mode."""
            print(f"[TokenizeThread] STARTED at {time.time():.0f}")
            pos = 0
            batch_count = 0
            while pos < n_exp:
                batch_size = num_workers if batch_count == 0 else tokenize_lookahead
                end_pos = min(pos + batch_size, n_exp)
                batch_ids = list(range(pos, end_pos))
                batch_selected = [all_selected_train[i] for i in batch_ids]

                try:
                    exp_token_paths = self._tokenize_batch_union(
                        batch_selected, batch_ids, async_write_queue,
                        shm_store=exp_shm_info, shm_lock=exp_shm_lock,
                    )
                    batch_count += 1

                    with ready_cond:
                        for exp_id in batch_ids:
                            ready_events[exp_id] = True
                        ready_cond.notify_all()

                    print(f"[TokenizeThread] Batch {batch_count}: {len(batch_ids)} exps ready ({pos}-{end_pos - 1})")

                except Exception as e:
                    print(f"[TokenizeThread] ERROR batch {batch_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    with ready_cond:
                        for exp_id in batch_ids:
                            ready_events[exp_id] = False
                        ready_cond.notify_all()

                pos = end_pos
                time.sleep(0.05)

            async_write_queue.put(None)
            print(
                f"[TokenizeThread] All {n_exp} experiments tokenized in {batch_count} batches ({time.time() - t_start:.1f}s)")

        tokenize_thread = threading.Thread(target=tokenize_thread_func, daemon=True)
        tokenize_thread.start()

        first_batch_end = min(num_workers, n_exp)
        print(f"[DynamicParallel] Waiting for first batch (exp 0-{first_batch_end - 1}) to be tokenized...")
        wait_start = time.time()
        last_progress_time = 0
        while True:
            with ready_cond:
                ready_count = sum(1 for i in range(first_batch_end) if ready_events.get(i, False))
            if ready_count == first_batch_end:
                break
            if time.time() - wait_start > 900:
                with ready_cond:
                    for i in range(first_batch_end):
                        status = ready_events.get(i, None)
                        print(f"[DynamicParallel] Exp {i} status: {status}")
                print("[DynamicParallel] TIMEOUT waiting for first batch - check tokenize errors above")
                break
            now = time.time()
            if now - last_progress_time > 5:
                print(f"[DynamicParallel] tokenizing... {ready_count}/{first_batch_end} ready")
                last_progress_time = now
            with ready_cond:
                ready_cond.wait(timeout=5.0)
        print(f"[DynamicParallel] First batch tokenized ({first_batch_end} exps), starting workers")

        def dispatcher_thread_func():
            """Push ready tasks when signaled (no busy-wait polling)."""
            pos = 0
            failed_exps = []
            while pos < n_exp:
                task_item = None
                skip_item = None
                with ready_cond:
                    is_ready = ready_events.get(pos, None)
                    if is_ready is True:
                        with exp_shm_lock:
                            shm_info = exp_shm_info.get(pos)
                        task_item = (
                            pos, params_list[pos],
                            all_selected_train[pos], shm_info,
                            len(all_selected[pos]),
                        )
                    elif is_ready is False:
                        skip_item = (pos, params_list[pos])
                    else:
                        ready_cond.wait(timeout=1.0)
                        continue

                if task_item is not None:
                    task_queue.put(task_item)
                    pos += 1
                elif skip_item is not None:
                    skip_pos, skip_params = skip_item
                    print(f"[Dispatcher] Skipping exp {skip_pos} (tokenize failed)")
                    failed_exps.append(skip_pos)
                    result_queue.put(ProxyResult(
                        parameters=skip_params,
                        validation_loss=float('inf'),
                        metadata={"experiment_id": skip_pos, "error": "tokenize_failed"}
                    ))
                    pos += 1

            for _ in range(num_workers):
                task_queue.put(None)

            if failed_exps:
                print(f"[Dispatcher] {len(failed_exps)} experiments failed tokenization")
            print(f"[Dispatcher] All {n_exp} tasks dispatched")

        dispatcher_thread = threading.Thread(target=dispatcher_thread_func, daemon=True)
        dispatcher_thread.start()

        worker_processes = []
        alive_workers = {"count": num_workers}

        def collector_thread_func():
            """Collect results from result queue, with worker health monitoring."""
            nonlocal completed_count
            last_progress_time = time.time()
            while completed_count < n_exp:
                try:
                    result = result_queue.get(timeout=1.0)
                    last_progress_time = time.time()
                    if result is None:
                        alive_workers["count"] -= 1
                        if alive_workers["count"] <= 0 and completed_count < n_exp:
                            missing = [i for i in range(n_exp) if all_results[i] is None]
                            print(f"[Collector] Workers exited; {completed_count}/{n_exp} results, "
                                  f"{len(missing)} missing (exps: {missing})")
                            for eid in missing:
                                all_results[eid] = ProxyResult(
                                    parameters=params_list[eid],
                                    validation_loss=float('inf'),
                                    metadata={"experiment_id": eid, "error": "worker_crash"}
                                )
                                completed_count += 1
                            break
                        continue
                    eid = result.metadata.get("experiment_id", -1)
                    if result.metadata.get("is_worker_crash"):
                        worker_id = result.metadata.get("worker_id", "?")
                        tb = result.metadata.get("traceback", "")
                        print(f"\n[Collector] Worker {worker_id} CRASHED:\n{tb}", flush=True)
                        alive_workers["count"] -= 1
                        if alive_workers["count"] <= 0 and completed_count < n_exp:
                            missing = [i for i in range(n_exp) if all_results[i] is None]
                            print(f"[Collector] All workers crashed; {completed_count}/{n_exp} results, "
                                  f"{len(missing)} missing")
                            for eid in missing:
                                all_results[eid] = ProxyResult(
                                    parameters=params_list[eid],
                                    validation_loss=float('inf'),
                                    metadata={"experiment_id": eid, "error": "worker_crash"}
                                )
                                completed_count += 1
                            break
                        continue
                    eid = result.metadata["experiment_id"]
                    all_results[eid] = result
                    completed_count += 1

                    elapsed = time.time() - t_start
                    eta = (n_exp - completed_count) * elapsed / max(1, completed_count)
                    if completed_count % 50 == 0 or completed_count == n_exp:
                        print(f"[Collector] {completed_count}/{n_exp} done ({elapsed:.0f}s, ETA: {eta:.0f}s)")
                except thread_queue.Empty:
                    pass
                except Exception as e:
                    print(f"[Collector] Unexpected error: {e}")

        collector_thread = threading.Thread(target=collector_thread_func, daemon=True)
        collector_thread.start()

        for wid in range(num_workers):
            p = ctx.Process(
                target=_worker_dynamic_loop,
                args=(wid, device_type, config_ser, task_queue, result_queue),
            )
            p.start()
            worker_processes.append(p)

        interrupted = False
        try:
            collector_thread.join()
            dispatcher_thread.join()
            tokenize_thread.join()
            async_write_thread.join()
        except KeyboardInterrupt:
            interrupted = True
            print("\n[DynamicParallel] Interrupted — sending shutdown to workers...")
            for _ in range(num_workers):
                try:
                    task_queue.put(None, timeout=2)
                except Exception:
                    pass
            for p in worker_processes:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()

        for p in worker_processes:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
            print(f"[DynamicParallel] Worker {p.pid} exitcode={p.exitcode}")

        if not interrupted:
            elapsed = time.time() - t_start
            print(f"\n[DynamicParallel] All {n_exp} experiments complete ({elapsed:.0f}s ≈ {elapsed / 60:.1f}min)")

        if shared_meta:
            import gc
            for name, info in list(shared_meta.items()):
                try:
                    shm = mp.shared_memory.SharedMemory(name=info.name)
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
            print(f"[SharedMem] Cleaned up {len(shared_meta)} metadata blocks")

        leaked = 0
        with exp_shm_lock:
            items = list(exp_shm_info.items())
        for exp_id, (shm_name, shape, dtype_str) in items:
            try:
                shm = mp.shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
                leaked += 1
            except Exception:
                pass
        if leaked:
            print(f"[SharedMem] Cleaned up {leaked} leaked token blocks")

        if interrupted:
            raise KeyboardInterrupt

        return all_results
