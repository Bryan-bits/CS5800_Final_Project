# metrics.py
from __future__ import annotations
import time
import tracemalloc
import numpy as np


def time_and_peak_mem(func, *args, **kwargs):
    """
    Measure:
    - execution time
    - peak memory (bytes)
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, (t1 - t0), peak


def label_parity(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Check equivalence up to permutation of component IDs.
    """
    if a.shape != b.shape:
        return False

    H, W = a.shape
    mapping = {0: 0}
    used = {0}

    for y in range(H):
        for x in range(W):
            va = int(a[y, x])
            vb = int(b[y, x])

            if va not in mapping:
                if vb in used:
                    return False
                mapping[va] = vb
                used.add(vb)
            else:
                if mapping[va] != vb:
                    return False
    return True