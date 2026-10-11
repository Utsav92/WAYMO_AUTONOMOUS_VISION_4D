import numpy as np

import calibration as cal
from decode_lidar import decode_frames, to_world, voxel_downsample
import waymo_adapter as wa


def test_schema_components_present(ctx):
    have = wa.have_components(ctx["root"], ctx["name"], ctx["split"])
    for c in wa.REQUIRED:
        assert have[c], c
    assert not have["lidar_segmentation"]          # absent stays absent - never invented


def test_range_image_roundtrip():
    H, W = 64, 512
    inc = cal.beam_inclinations(H, values=np.linspace(-0.3, 0.1, H))
    az = np.linspace(-3, 3, 200)
    el = inc[np.random.default_rng(0).integers(0, H, 200)]
    r = np.random.default_rng(1).uniform(3, 60, 200)
    P = np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], 1)
    img, rows, cols = cal.points_to_range_image(P, H, W, inc)
    ri = np.zeros((H, W, 4), np.float32)
    ri[..., 0] = img
    pts, attrs, valid, _ = cal.range_image_to_points(ri, np.eye(4), inc)
    # every decoded point lies at the original range, within one pixel of azimuth
    assert valid.sum() > 150
    assert np.allclose(np.linalg.norm(pts, axis=1), attrs[:, 2], atol=1e-3)
    d = np.min(np.linalg.norm(pts[:, None] - P[None], axis=2), axis=1)
    assert np.percentile(d, 90) < 0.35 * np.pi * 2 / W * 60 + 0.2


def test_invalid_range_dropped():
    ri = np.zeros((8, 16, 4), np.float32)
    ri[2, 3, 0] = 10.0
    ri[4, 5, 0] = -1.0
    pts, _, valid, _ = cal.range_image_to_points(ri, np.eye(4), cal.beam_inclinations(8, -0.2, 0.2))
    assert len(pts) == 1 and valid.sum() == 1


def test_decode_ground_is_flat_in_world(ctx):
    fr = next(decode_frames(ctx["root"], ctx["name"], ctx["split"], 1))
    w = to_world(fr)
    ground = w[np.abs(w[:, 2]) < 0.6]
    assert len(ground) > 2000
    assert np.abs(ground[:, 2]).mean() < 0.15       # extrinsic + decoding place the road at z~0
    assert fr["xyz_vehicle"].shape[1] == 3


def test_voxel_downsample_reduces():
    p = np.random.default_rng(0).uniform(0, 1, (5000, 3))
    (q,) = voxel_downsample(p, 0.25)
    assert len(q) <= 64
