"""SYNTHETIC Waymo-v2-shaped parquet fixture.  FOR TESTS / RENDERER SELF-TEST ONLY - never real sensor data.
Column names follow the v2 naming assumed by datasets/waymo_adapter.py (`key.*`, `[Component].field`); the real
schema must still be confirmed with datasets/inspect_components.py on a downloaded context.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "preprocessing"))
sys.path.insert(0, os.path.join(ROOT, "datasets"))
from calibration import beam_inclinations, points_to_range_image, inv_rigid, transform, box_corners  # noqa: E402

CTX = "SYNTHETIC_SELFTEST_0000"
H, W = 64, 1024
CAM_W, CAM_H, FU = 640, 480, 500.0


def Tz(yaw, x=0.0, y=0.0, z=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = [x, y, z]
    return T


def _surface(rng):
    pts, col = [], []
    g = np.stack(np.meshgrid(np.arange(-20, 140, 0.35), np.arange(-14, 14, 0.35)), -1).reshape(-1, 2)
    pts.append(np.c_[g, np.zeros(len(g))])
    chk = ((np.floor(g[:, 0] / 2) + np.floor(g[:, 1] / 2)) % 2)[:, None]
    col.append(np.where(chk > 0, [90, 90, 95], [60, 60, 66]))
    for side, base in ((10.5, (180, 120, 90)), (-10.5, (100, 140, 190))):
        gx = np.arange(-20, 140, 0.35)
        gz = np.arange(0, 9, 0.35)
        X, Z = np.meshgrid(gx, gz)
        pts.append(np.c_[X.ravel(), np.full(X.size, side), Z.ravel()])
        win = ((np.floor(X.ravel() / 3) + np.floor(Z.ravel() / 2)) % 2)[:, None]
        col.append(np.where(win > 0, base, np.array(base) * 0.6))
    return np.concatenate(pts), np.concatenate(col).astype(np.uint8)


def _box_points(center, size, heading, rng, n=1500):
    l, w, h = size
    u = rng.uniform(-0.5, 0.5, (n, 3)) * [l, w, h]
    face = rng.integers(0, 3, n)
    side = rng.choice([-0.5, 0.5], n)
    for i in range(n):
        u[i, face[i]] = side[i] * (l, w, h)[face[i]]
    c, s = np.cos(heading), np.sin(heading)
    x = c * u[:, 0] - s * u[:, 1] + center[0]
    y = s * u[:, 0] + c * u[:, 1] + center[1]
    return np.stack([x, y, u[:, 2] + center[2]], 1)


def build(root_dir, frames=24, dt=0.1, seed=3, split="validation"):
    rng = np.random.default_rng(seed)
    base_pts, base_col = _surface(rng)
    ext_lidar = Tz(0.0, 1.0, 0.0, 1.9)
    Tcam = Tz(0.0, 1.5, 0.0, 1.6)                                    # camera->vehicle (x fwd, y left, z up)
    incl = np.linspace(-0.35, 0.12, H)                               # stored ascending on purpose
    inc_rows = beam_inclinations(H, values=incl)
    ts0 = 1_500_000_000_000_000
    objs = [  # (track id str, type, size, start xy, vel xy)
        ("veh-A", 1, (4.6, 1.9, 1.6), (18.0, 3.5), (3.0, 0.0)),
        ("veh-B", 1, (4.2, 1.8, 1.5), (30.0, -3.5), (-2.0, 0.0)),
        ("ped-C", 2, (0.6, 0.6, 1.7), (14.0, 7.5), (0.0, -1.2)),
        ("cyc-D", 4, (1.8, 0.6, 1.5), (24.0, -6.5), (2.5, 0.0))]
    obj_col = {1: (60, 255, 100), 2: (255, 204, 0), 4: (0, 216, 255)}
    out = {k: [] for k in ("lidar", "lidar_calibration", "vehicle_pose", "camera_image", "camera_calibration",
                           "lidar_box")}
    cal = {"key.segment_context_name": CTX, "key.laser_name": 1,
           "[LiDARCalibrationComponent].extrinsic.transform": ext_lidar.ravel().tolist(),
           "[LiDARCalibrationComponent].beam_inclination.min": float(incl.min()),
           "[LiDARCalibrationComponent].beam_inclination.max": float(incl.max()),
           "[LiDARCalibrationComponent].beam_inclination.values": incl.tolist()}
    out["lidar_calibration"].append(cal)
    out["camera_calibration"].append({
        "key.segment_context_name": CTX, "key.camera_name": 1,
        "[CameraCalibrationComponent].intrinsic": [FU, FU, CAM_W / 2, CAM_H / 2, 0, 0, 0, 0, 0],
        "[CameraCalibrationComponent].extrinsic.transform": Tcam.ravel().tolist(),
        "[CameraCalibrationComponent].width": CAM_W, "[CameraCalibrationComponent].height": CAM_H})
    truth = []
    for f in range(frames):
        t = f * dt
        ts = ts0 + int(round(t * 1e6))
        Twv = Tz(0.02 * np.sin(t), 5.0 * t, 0.0, 0.0)
        out["vehicle_pose"].append({"key.segment_context_name": CTX, "key.frame_timestamp_micros": ts,
                                    "[VehiclePoseComponent].world_from_vehicle.transform": Twv.ravel().tolist()})
        pts, col = [base_pts], [base_col]
        for name, typ, size, p0, v in objs:
            c = np.array([p0[0] + v[0] * t, p0[1] + v[1] * t, size[2] / 2])
            hd = float(np.arctan2(v[1], v[0])) if (v[0] or v[1]) else 0.0
            pts.append(_box_points(c, size, hd, rng))
            col.append(np.tile(obj_col[typ], (1500, 1)).astype(np.uint8))
            cv = transform(inv_rigid(Twv), c[None])[0]
            hv = hd - float(np.arctan2(Twv[1, 0], Twv[0, 0]))
            out["lidar_box"].append({
                "key.segment_context_name": CTX, "key.frame_timestamp_micros": ts, "key.laser_object_id": name,
                "[LiDARBoxComponent].type": typ,
                "[LiDARBoxComponent].box.center.x": float(cv[0]), "[LiDARBoxComponent].box.center.y": float(cv[1]),
                "[LiDARBoxComponent].box.center.z": float(cv[2]),
                "[LiDARBoxComponent].box.size.x": size[0], "[LiDARBoxComponent].box.size.y": size[1],
                "[LiDARBoxComponent].box.size.z": size[2], "[LiDARBoxComponent].box.heading": hv})
        P = np.concatenate(pts)
        C = np.concatenate(col)
        Pv = transform(inv_rigid(Twv), P)
        Ps = transform(inv_rigid(ext_lidar), Pv)
        r = np.linalg.norm(Ps, axis=1)
        m = (r > 1.5) & (r < 80)
        img, rows, cols = points_to_range_image(Ps[m], H, W, inc_rows)
        ri = np.zeros((H, W, 4), np.float32)
        ri[..., 0] = img
        ri[rows, cols, 1] = 0.5
        ri[rows, cols, 2] = 0.1
        out["lidar"].append({"key.segment_context_name": CTX, "key.frame_timestamp_micros": ts,
                             "key.laser_name": 1,
                             "[LiDARComponent].range_image_return1.values": ri.ravel().tolist(),
                             "[LiDARComponent].range_image_return1.shape": [H, W, 4]})
        # camera image: painter's algorithm from the same world
        pc = transform(inv_rigid(Tcam), Pv)
        ok = pc[:, 0] > 1.0
        u = FU * (-pc[ok, 1] / pc[ok, 0]) + CAM_W / 2
        v = FU * (-pc[ok, 2] / pc[ok, 0]) + CAM_H / 2
        d = pc[ok, 0]
        Cc = C[ok]
        order = np.argsort(-d)
        im = np.full((CAM_H, CAM_W, 3), (150, 170, 200), np.uint8)
        for i in order:
            x, y = int(u[i]), int(v[i])
            if 1 <= x < CAM_W - 1 and 1 <= y < CAM_H - 1:
                im[y - 1:y + 2, x - 1:x + 2] = Cc[i]
        ok_, jpg = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        out["camera_image"].append({"key.segment_context_name": CTX, "key.frame_timestamp_micros": ts,
                                    "key.camera_name": 1, "[CameraImageComponent].image": jpg.tobytes(),
                                    "[CameraImageComponent].pose.transform": Twv.ravel().tolist()})
        truth.append({"ts": ts, "T": Twv})
    for comp, rows in out.items():
        d = os.path.join(root_dir, split, comp)
        os.makedirs(d, exist_ok=True)
        pd.DataFrame(rows).to_parquet(os.path.join(d, CTX + ".parquet"))
    return CTX, truth, {"objs": objs, "dt": dt, "ext_lidar": ext_lidar, "Tcam": Tcam}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "raw_synthetic")
    print(build(out)[0], "->", out)
