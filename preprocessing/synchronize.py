"""Timestamp alignment helpers (all timestamps are integer microseconds)."""
import numpy as np


def nearest(ts_sorted, t):
    ts = np.asarray(ts_sorted)
    i = int(np.searchsorted(ts, t))
    cands = [j for j in (i - 1, i) if 0 <= j < len(ts)]
    j = min(cands, key=lambda k: abs(int(ts[k]) - int(t)))
    return j, int(ts[j]) - int(t)


def check_sync(frame_ts, stream_ts, tol_us=50_000):
    """Return list of (frame_ts, nearest stream ts, delta_us) that exceed ``tol_us``."""
    bad = []
    for t in frame_ts:
        j, d = nearest(stream_ts, t)
        if abs(d) > tol_us:
            bad.append((int(t), int(stream_ts[j]), d))
    return bad


def frame_interval_stats(ts):
    d = np.diff(np.asarray(ts, dtype=np.int64)) * 1e-6
    return {"mean_s": float(d.mean()), "min_s": float(d.min()), "max_s": float(d.max()), "n": int(len(d))}
