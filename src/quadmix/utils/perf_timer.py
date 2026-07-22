"""Lightweight performance timer with nesting support."""

import os
import time
import threading
from contextlib import contextmanager
from typing import Dict, List, Tuple


class PerfTimer:
    """Lightweight performance timer with nesting support."""
    _timings: Dict[str, List[float]] = {}
    _timings_lock = threading.Lock()
    _local = threading.local()
    _enabled: bool = os.environ.get("QUADMIX_PERF_TIMER", "0") == "1"

    @classmethod
    def _get_stack(cls) -> List[Tuple[str, float]]:
        if not hasattr(cls._local, 'stack'):
            cls._local.stack = []
        return cls._local.stack

    @classmethod
    def enable(cls, enabled: bool = True):
        cls._enabled = enabled

    @classmethod
    @contextmanager
    def section(cls, name: str, prefix: str = ""):
        """Context manager for timing a section."""
        if not cls._enabled:
            yield
            return

        full_name = f"{prefix}.{name}" if prefix else name
        start = time.perf_counter()
        stack = cls._get_stack()
        stack.append((full_name, start))
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with cls._timings_lock:
                if full_name not in cls._timings:
                    cls._timings[full_name] = []
                cls._timings[full_name].append(elapsed)
            stack.pop()

    @classmethod
    def report(cls, top_n: int = 20) -> str:
        """Generate performance report."""
        with cls._timings_lock:
            items = list(cls._timings.items())
        if not items:
            return "[PerfTimer] No timings recorded"

        lines = ["\n" + "=" * 70, "PERFORMANCE REPORT", "=" * 70]

        sorted_items = sorted(
            items,
            key=lambda x: sum(x[1]),
            reverse=True
        )[:top_n]

        for name, times in sorted_items:
            total = sum(times)
            count = len(times)
            avg = total / count
            lines.append(f"{name:50s} | total: {total:7.2f}s | count: {count:4d} | avg: {avg:.3f}s")

        lines.append("=" * 70)
        return "\n".join(lines)

    @classmethod
    def reset(cls):
        """Reset all timings."""
        with cls._timings_lock:
            cls._timings.clear()
        cls._local.stack = []
