import numpy as np

import calibration as cal


def test_inclination_row0_is_top():
    asc = np.linspace(-0.4, 0.1, 64)
    assert np.all(np.diff(cal.beam_inclinations(64, values=asc)) < 0)
    uni = cal.beam_inclinations(32, -0.4, 0.1)
    assert uni[0] > uni[-1] and uni[0] < 0.1 and uni[-1] > -0.4


def test_inv_rigid_and_world_chain():
    T = np.eye(4)
    T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    T[:3, 3] = [3, 4, 5]
    assert np.allclose(T @ cal.inv_rigid(T), np.eye(4), atol=1e-12)
    ext = np.eye(4)
    ext[:3, 3] = [1, 0, 2]
    p = np.array([[1.0, 0, 0]])
    got = cal.sensor_to_world(p, ext, T)
    assert np.allclose(got, cal.transform(T @ ext, p))
    assert np.allclose(got, [[3, 4 + 2, 5 + 2]])      # x_s=1 -> +1 ext -> (2,0,2) -> rot90 -> (0,2,2)+(3,4,5)


def test_azimuth_correction_uses_extrinsic_yaw():
    H, W = 4, 8
    ri = np.zeros((H, W, 4), np.float32)
    ri[1, 0, 0] = 10.0
    inc = cal.beam_inclinations(H, -0.1, 0.1)
    p0, *_ = cal.range_image_to_points(ri, np.eye(4), inc)
    yaw = np.eye(4)
    yaw[:2, :2] = [[0, -1], [1, 0]]                    # sensor yawed +90 deg in the vehicle
    p1, *_ = cal.range_image_to_points(ri, yaw, inc)
    # rotation of the extrinsic is applied once AND the az correction cancels the yaw in azimuth
    az0 = np.arctan2(p0[0, 1], p0[0, 0])
    az1 = np.arctan2(p1[0, 1], p1[0, 0])
    assert abs(((az1 - az0) + np.pi) % (2 * np.pi) - np.pi) < 1e-5 or abs(az1 - az0) < 1e-5


def test_box_corners_heading():
    c = cal.box_corners([0, 0, 0], (4, 2, 1), 0.0)
    assert np.isclose(c[:, 0].max(), 2) and np.isclose(c[:, 1].max(), 1)
    c90 = cal.box_corners([0, 0, 0], (4, 2, 1), np.pi / 2)
    assert np.isclose(c90[:, 1].max(), 2) and np.isclose(c90[:, 0].max(), 1)


def test_points_beyond_lens_fold_are_not_projected():
    """Regression: k2 = -0.3 makes the distortion polynomial fold back, mapping points at ~80 deg into the image."""
    intr = dict(f_u=2059.0, f_v=2059.0, c_u=935.0, c_v=635.0, k1=0.0, k2=-0.3, p1=0.0, p2=0.0, k3=0.0)
    ok_pt = np.array([[20.0, 3.0, -1.5]])                 # ~8.5 deg off axis: must project
    far = np.array([[1.0, 5.5, 0.0]])                     # ~80 deg off axis: must be rejected
    _, _, _, a = cal.project_to_camera(ok_pt, np.eye(4), intr, 1920, 1280)
    _, _, _, b = cal.project_to_camera(far, np.eye(4), intr, 1920, 1280)
    assert a[0] and not b[0]
    assert np.isclose(cal.distortion_r2_limit(intr), 0.98 * np.sqrt(1 / 1.5), rtol=1e-6) or cal.distortion_r2_limit(intr) < 1.0
