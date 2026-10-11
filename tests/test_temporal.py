import numpy as np

from temporal_history import age_fade, static_mask, RollingMap


def test_fade_uses_real_time_not_frame_count():
    # irregular spacing: 0.1 s and 1.9 s old must have different weights even though both are "previous frame"
    w, age = age_fade(2_000_000, [1_900_000, 100_000], decay_s=2.0)
    assert w[0] > w[1] and np.isclose(age[1], 1.9)
    w2, _ = age_fade(0, [1_000_000])
    assert w2[0] == 0.0                              # future observation never shows


def test_static_mask_removes_tracked_box_points():
    pts = np.array([[0, 0, 1.0], [10, 0, 1.0], [2.05, 0, 1.0], [0, 0, 5.0]])
    keep = static_mask(pts, [np.array([0, 0, 1.0])], [np.array([4.0, 2.0, 2.0])], [0.0], margin=0.1)
    assert keep.tolist() == [False, True, False, True]


def test_rolling_map_budget_and_radius():
    m = RollingMap(voxel=0.5, radius=30, max_points=500)
    rng = np.random.default_rng(0)
    for k in range(5):
        m.add(rng.uniform(-80, 80, (2000, 3)), (0.0, 0.0))
    P = m.points()
    assert len(P) <= 500
    assert np.hypot(P[:, 0], P[:, 1]).max() <= 30.5


def test_trail_alignment_static_world(ctx, tmp_path):
    from export_frames import export
    export(ctx["root"], ctx["name"], str(tmp_path), ctx["split"], max_frames=8)
    c = []
    for i in (0, 7):
        b = np.load(tmp_path / ctx["name"] / "frames" / f"f{i:04d}.npz")["boxes"]
        c.append(b[b[:, 7] == 2][0, :2])             # the pedestrian: moves -1.2 m/s in y, 0 in x
    assert abs(c[1][0] - c[0][0]) < 0.1
    assert np.isclose(c[1][1] - c[0][1], -1.2 * 0.7, atol=0.15)
