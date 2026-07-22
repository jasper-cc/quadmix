"""
ConcurrencyConfig — centralized parallelism configuration for QuaDMix.

Auto-calculates optimal outer-workers × inner-BAS-threads ≈ cpu_count,
preventing thread explosion on large servers.

Usage:
  # Before numpy import (runner scripts — inline, no import possible):
  import os
  cpu_count = os.cpu_count() or 4
  os.environ.setdefault('OPENBLAS_NUM_THREADS', str(max(1, cpu_count // 4)))
  os.environ.setdefault('OMP_NUM_THREADS', str(max(1, cpu_count // 4)))

  # After numpy import (pipeline modules — use ConcurrencyConfig):
  from quadmix.utils.concurrency import ConcurrencyConfig
  cfg = ConcurrencyConfig()
  n_jobs = min(n_tasks, cfg.max_compute_workers)
  cfg.set_blas_threads(cfg.blas_threads_for(n_jobs))
"""

import ctypes
import os
from dataclasses import dataclass


_BLAS_LIB_NAMES = [
    'libopenblas.so', 'libopenblas.so.0',
    'libscipy_openblas64.so',
    'libmkl_rt.so', 'libmkl_rt.so.1', 'libmkl_rt.so.2',
]


def get_available_memory_bytes() -> int:
    """Return usable RAM, respecting both host and cgroup limits."""
    candidates = []
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
        if available > 0:
            candidates.append(available)
    except (ImportError, OSError, ValueError):
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError, IndexError):
        pass
    for limit_path, current_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
         "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            with open(limit_path, encoding="utf-8") as f:
                limit_text = f.read().strip()
            if limit_text == "max":
                continue
            with open(current_path, encoding="utf-8") as f:
                current = int(f.read().strip())
            limit = int(limit_text)
            if 0 < limit < (1 << 60):
                candidates.append(max(0, limit - current))
        except (OSError, ValueError):
            continue
    if candidates:
        return min(candidates)
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 1024 ** 3


def _probe_blas_lib():
    for lib_name in _BLAS_LIB_NAMES:
        try:
            lib = ctypes.CDLL(lib_name)
            return lib
        except OSError:
            continue
    return None


def get_blas_threads() -> int:
    lib = _probe_blas_lib()
    if lib is None:
        return 0
    if hasattr(lib, 'openblas_get_num_threads'):
        return lib.openblas_get_num_threads()
    if hasattr(lib, 'MKL_Get_Max_Threads'):
        return lib.MKL_Get_Max_Threads()
    return 0


def set_blas_threads(n: int) -> bool:
    lib = _probe_blas_lib()
    if lib is None:
        return False
    if hasattr(lib, 'openblas_set_num_threads'):
        lib.openblas_set_num_threads(n)
        return True
    if hasattr(lib, 'MKL_Set_Num_Threads'):
        lib.MKL_Set_Num_Threads(n)
        return True
    return False


@dataclass(frozen=True)
class ConcurrencyConfig:
    cpu_count: int = 0

    max_compute_workers: int = 0
    max_io_workers: int = 0
    model_n_jobs_nested: int = 1

    def __post_init__(self):
        if self.cpu_count == 0:
            object.__setattr__(self, 'cpu_count', os.cpu_count() or 4)
        if self.max_compute_workers == 0:
            object.__setattr__(self, 'max_compute_workers',
                               min(self.cpu_count, 16))
        if self.max_io_workers == 0:
            object.__setattr__(self, 'max_io_workers',
                               min(self.cpu_count, 32))

    def blas_threads_for(self, n_workers: int) -> int:
        return max(1, round(self.cpu_count / max(1, n_workers)))

    def apply_env_vars(self):
        import sys
        bt = self.blas_threads_for(self.max_compute_workers)
        if "numpy" in sys.modules:
            import logging
            logging.getLogger(__name__).warning(
                "numpy already imported — OPENBLAS/OMP/MKL thread settings "
                "will have no effect. Call apply_env_vars() before importing numpy."
            )
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(bt))
        os.environ.setdefault("OMP_NUM_THREADS", str(bt))
        os.environ.setdefault("MKL_NUM_THREADS", str(bt))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(bt))
        os.environ.setdefault("RAYON_NUM_THREADS", "4")
