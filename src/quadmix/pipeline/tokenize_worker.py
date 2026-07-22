"""Tokenize worker functions - NO torch import to avoid CANN initialization overhead."""
import time
import numpy as np
from typing import List, Optional, Tuple


def _get_tokenizer(tokenizer_path: str):
    """Lazy load tokenizer."""
    from tokenizers import Tokenizer
    import os
    if os.path.isdir(tokenizer_path):
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        if not os.path.isfile(tokenizer_file):
            raise FileNotFoundError(
                f"local tokenizer.json not found: {tokenizer_file}"
            )
        return Tokenizer.from_file(tokenizer_file)
    if os.path.isfile(tokenizer_path):
        return Tokenizer.from_file(tokenizer_path)
    return Tokenizer.from_pretrained(tokenizer_path)


def _tokenize_chunk_to_array(
        chunk: List[Tuple[int, int, str]],
        tokenizer_path: str,
        block_size: int,
        threads_per_worker: int = 4,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Tokenize a chunk, returning compact numpy array."""
    import os as _os
    _os.environ["RAYON_NUM_THREADS"] = str(threads_per_worker)
    _os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)

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

    Uses pyarrow directly with adaptive read strategy instead of
    pd.read_parquet, avoiding pandas DataFrame construction overhead.

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
