"""JSON serialization helpers — sanitize NaN/Inf for valid JSON output."""

import math
import numpy as np


def sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats and numpy types with JSON-safe values.

    NaN → None (null), Inf → None (null), numpy types → Python primitives.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj
