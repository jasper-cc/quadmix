"""Shared memory helpers for multiprocessing (avoids re-reading large metadata per worker)."""

import multiprocessing as mp
import multiprocessing.shared_memory

import numpy as np


class SharedArrayInfo:
    """Descriptor for a numpy array in shared memory — pickle-safe for mp."""

    def __init__(self, name: str, shape: tuple, dtype: str, nbytes: int):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes


def ndarray_to_shared(arr: np.ndarray, prefix: str) -> SharedArrayInfo:
    """Copy numpy array into shared memory, return descriptor."""
    shm = mp.shared_memory.SharedMemory(create=True, size=arr.nbytes, name=f"{prefix}_shm")
    shared = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    np.copyto(shared, arr)
    info = SharedArrayInfo(name=shm.name, shape=arr.shape, dtype=str(arr.dtype), nbytes=arr.nbytes)
    shm.close()
    return info


def shared_to_ndarray(info: SharedArrayInfo) -> np.ndarray:
    """Map shared memory back to numpy array.

    IMPORTANT: Return a COPY to avoid segfault when numpy operations
    (slicing, normalization) access shared memory buffer in spawn children.
    """
    shm = mp.shared_memory.SharedMemory(name=info.name)
    arr = np.ndarray(shape=info.shape, dtype=np.dtype(info.dtype), buffer=shm.buf)
    result = arr.copy()
    shm.close()
    return result
