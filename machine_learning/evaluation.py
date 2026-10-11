"""Depth-error metrics on VALID pixels only (pure numpy, tested-by-construction helpers)."""
import numpy as np


def align_scale_shift(pred, gt, valid):
    """Least-squares s,t minimising |s*pred + t - gt| over valid pixels (monocular depth has scale ambiguity)."""
    A = np.stack([pred[valid], np.ones(valid.sum())], 1)
    (s, t), *_ = np.linalg.lstsq(A, gt[valid], rcond=None)
    return float(s), float(t)


def depth_metrics(pred, gt, valid):
    n = int(valid.sum())
    if n == 0:
        return {"valid_pct": 0.0}
    e = pred[valid] - gt[valid]
    g, p = gt[valid], pred[valid]
    rel = np.abs(e) / np.maximum(g, 1e-6)
    ratio = np.maximum(p / np.maximum(g, 1e-6), g / np.maximum(p, 1e-6))
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "absrel": float(rel.mean()),
            "median_absrel": float(np.median(rel)), "delta125": float((ratio < 1.25).mean()),
            "valid_pct": 100.0 * n / valid.size}
