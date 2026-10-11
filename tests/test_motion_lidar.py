import os
import sys
import zlib

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "datasets"), os.path.join(ROOT, "datasets", "waymo_pb"),
                os.path.join(ROOT, "preprocessing")]
pb = pytest.importorskip("waymo_open_dataset.protos.compressed_lidar_pb2")
import motion_lidar as ml  # noqa: E402


def encode(img, prec):
    """Reference encoder for the verified layout: channel-major, RLE mask (valid, invalid, ...), residual = deltas."""
    H, W, C = img.shape
    flat = img.transpose(2, 0, 1).reshape(-1)
    q = np.round(flat / np.repeat(np.asarray(prec, np.float32), H * W)).astype(np.int64)
    valid = flat != 0
    runs, cur, n = [], True, 0
    for v in valid:
        if v == cur:
            n += 1
        else:
            runs.append(n)
            cur, n = v, 1
    runs.append(n)
    if not valid[0]:
        runs = [0] + runs
    d = pb.DeltaEncodedData()
    d.mask.extend(runs)
    qs = q[valid]
    d.residual.extend(np.diff(np.concatenate([[0], qs])).tolist())
    d.metadata.shape.extend([H, W, C])
    d.metadata.quant_precision.extend(prec)
    return zlib.compress(d.SerializeToString())


def test_delta_roundtrip_matches_quantised_values():
    rng = np.random.default_rng(0)
    H, W = 8, 40
    img = np.zeros((H, W, 4), np.float32)
    valid = rng.random((H, W)) > 0.3
    img[..., 0] = np.where(valid, rng.uniform(1, 70, (H, W)), 0)
    img[..., 1] = np.where(valid, rng.uniform(0, 1, (H, W)), 0)
    img[..., 2] = np.where(valid, rng.uniform(0, 1, (H, W)), 0)
    prec = [0.005, 0.01, 0.01, 1.0]
    out = ml.delta_decode(encode(img, prec))
    assert out.shape == img.shape
    assert np.allclose(out[..., 0], img[..., 0], atol=prec[0] / 2 + 1e-4)
    assert np.allclose(out[..., 1], img[..., 1], atol=prec[1] / 2 + 1e-4)
    assert ((out[..., 0] > 0) == valid).all()
