import numpy as np

from ego_motion import interpolate_pose, boxes_to_world, yaw_of
from synchronize import check_sync, nearest
from object_labels import BOX_DTYPE
import synth


def test_interpolate_midpoint():
    a = synth.Tz(0.0, 0, 0, 0)
    b = synth.Tz(0.4, 10, 2, 0)
    m = interpolate_pose(0, a, 100, b, 50)
    assert np.allclose(m[:3, 3], [5, 1, 0])
    assert np.isclose(yaw_of(m), 0.2, atol=1e-6)
    try:
        interpolate_pose(0, a, 100, b, 150)
        assert False
    except ValueError:
        pass


def test_boxes_to_world_rotates_heading():
    b = np.zeros(1, BOX_DTYPE)
    b["cx"] = 10
    b["heading"] = 0.1
    c, h = boxes_to_world(b, synth.Tz(np.pi / 2, 5, 0, 0))
    assert np.allclose(c[0], [5, 10, 0], atol=1e-9)
    assert np.isclose(h[0], 0.1 + np.pi / 2)


def test_sync_reports_gaps():
    assert nearest([0, 100, 200], 90) == (1, 10)
    bad = check_sync([0, 100_000, 400_000], [0, 100_000], tol_us=50_000)
    assert len(bad) == 1 and bad[0][0] == 400_000


def test_exported_frames_share_world(ctx, tmp_path):
    from export_frames import export
    m = export(ctx["root"], ctx["name"], str(tmp_path), ctx["split"], max_frames=4)
    P = np.array(m["poses_world_vehicle"])
    assert np.allclose(P[0][:3, 3], 0)               # origin is the first ego position
    assert np.allclose(P[3][0, 3], 1.5, atol=0.05)    # 5 m/s * 0.3 s
    d0 = np.load(tmp_path / ctx["name"] / "frames" / "f0000.npz")["pts"]
    d3 = np.load(tmp_path / ctx["name"] / "frames" / "f0003.npz")["pts"]
    # a static wall (y ~ +10.5) must sit at the same world y in every frame
    w0 = d0[(d0[:, 1] > 9.5) & (d0[:, 2] > 1)][:, 1]
    w3 = d3[(d3[:, 1] > 9.5) & (d3[:, 2] > 1)][:, 1]
    assert abs(np.median(w0) - np.median(w3)) < 0.25
