"""Waymo Open End-to-End Driving (WOD-E2E, camera) -> processed layout.

Each E2EDFrame = one timestep with 8 camera images, the ego's recent PAST states and 5 s FUTURE states (ground truth
on training/validation), a route INTENT, and on validation several rater-scored PREFERENCE trajectories.
There is no LiDAR in this dataset, so the LiDAR layers are empty for this source.

    python e2e_adapter.py --shard D:/WAYMO_DATA/e2e/val_...tfrecord-00000-of-00093 --seqs 1 --min-frames 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "waymo_pb"))
from tfrecord import iter_records  # noqa: E402

PROJ = os.path.dirname(HERE)
INTENTS = {0: "UNKNOWN", 1: "GO_STRAIGHT", 2: "GO_LEFT", 3: "GO_RIGHT"}


def parse(raw):
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as pb
    f = pb.E2EDFrame()
    f.ParseFromString(raw)
    return f


def states(s):
    if s is None or not len(s.pos_x):
        return np.zeros((0, 3), np.float32)
    return np.stack([np.array(s.pos_x), np.array(s.pos_y), np.array(s.pos_z) if len(s.pos_z) else np.zeros(len(s.pos_x))],
                    1).astype(np.float32)


def calibrations(frame):
    """camera id -> dict(intrinsic, T_vehicle_camera, width, height) from the frame context."""
    out = {}
    for c in frame.context.camera_calibrations:
        v = list(c.intrinsic) + [0.0] * 9
        out[int(c.name)] = {"intrinsic": dict(zip(("f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3"), v[:9])),
                            "T_vehicle_camera": np.array(c.extrinsic.transform, dtype=np.float64).reshape(4, 4).tolist(),
                            "width": int(c.width), "height": int(c.height)}
    return out


def estimate_points(images, cals, stride=4, max_depth=40.0, max_points=131072, seed=0):
    """ESTIMATED RGB point cloud (vehicle frame) from monocular depth.  E2E has no LiDAR, so the metric scale of
    each camera comes from the road plane: pixels that look down at the ground must lie at z = 0, which fixes the
    affine relation between the network's disparity and true inverse depth.  Not a measurement."""
    import cv2
    sys.path.insert(0, os.path.join(PROJ, "machine_learning"))
    sys.path.insert(0, os.path.join(PROJ, "preprocessing"))
    import camera_depth as cd
    rng = np.random.default_rng(seed)
    P, C, CAM = [], [], []
    for cid, jpg in images.items():
        cal = cals.get(cid)
        if cal is None:
            continue
        im = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]
        sx, sy = W / cal["width"], H / cal["height"]
        K = cal["intrinsic"]
        fx, fy, cx, cy = K["f_u"] * sx, K["f_v"] * sy, K["c_u"] * sx, K["c_v"] * sy
        T = np.array(cal["T_vehicle_camera"])
        R, c0 = T[:3, :3], T[:3, 3]
        disp = cd.estimate_inverse_depth(im)
        disp = cv2.bilateralFilter(disp.astype(np.float32), 7, 0.08 * float(np.ptp(disp) + 1e-6), 5)   # de-noise, keep edges
        gx = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)
        fly = grad > np.percentile(grad, 93)                       # depth discontinuities = "flying pixels"
        vv, uu = np.mgrid[0:H:stride, 0:W:stride]
        ray_c = np.stack([np.ones_like(uu, dtype=np.float64), -(uu - cx) / fx, -(vv - cy) / fy], -1).reshape(-1, 3)
        ray_v = ray_c @ R.T
        down = ray_v[:, 2] < -0.03
        t_ground = np.full(len(ray_c), np.inf)
        t_ground[down] = -c0[2] / ray_v[down, 2]                       # camera-x depth of the road along each ray
        ok = down & (t_ground < 60.0)
        d_pred = disp[vv.ravel(), uu.ravel()]
        fly_s = fly[vv.ravel(), uu.ravel()]
        if ok.sum() < 300:
            continue
        x, y = 1.0 / t_ground[ok], d_pred[ok].astype(np.float64)
        w, a, b = np.ones_like(x), 1.0, 0.0
        for _ in range(4):                                             # robust fit  disparity ~ a / depth + b
            A = np.stack([x * w, w], 1)
            (a, b), *_ = np.linalg.lstsq(A, y * w, rcond=None)
            r = np.abs(a * x + b - y)
            w = 1.0 / np.maximum(r / (np.median(r) + 1e-6), 1.0)
        if a <= 1e-6:
            continue
        inv = (d_pred - b) / a
        depth = np.where(inv > 1e-3, 1.0 / np.maximum(inv, 1e-3), np.inf)
        keep = (depth > 1.0) & (depth < max_depth) & (~fly_s)
        pts_c = ray_c[keep] * depth[keep, None]
        pts_v = pts_c @ R.T + c0
        sel = (pts_v[:, 2] > -0.5) & (pts_v[:, 2] < 12.0)
        P.append(pts_v[sel])
        C.append(im[vv.ravel()[keep][sel], uu.ravel()[keep][sel]])
        CAM.append(np.full(sel.sum(), cid, np.int8))
    if not P:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros(0, np.int8)
    P, C, CAM = np.concatenate(P), np.concatenate(C), np.concatenate(CAM)
    if len(P) > max_points:
        s = rng.choice(len(P), max_points, replace=False)
        P, C, CAM = P[s], C[s], CAM[s]
    return P.astype(np.float32), C.astype(np.uint8), CAM


def export_moment(raw, out_root, hz=4.0, estimate=True):
    """One E2EDFrame -> a 9 s timeline built from the REAL past+future ego states (frames at 4 Hz).  Cameras are the
    images captured at the moment t=0 (the dataset has no other images), shown for every timeline frame.
    World frame = the ego vehicle frame at t=0 (x forward, y left, z up)."""
    f = parse(raw)
    name = f.frame.context.name
    odir = os.path.join(out_root, "E2E_" + name)
    os.makedirs(os.path.join(odir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(odir, "cam"), exist_ok=True)
    past, fut = states(f.past_states), states(f.future_states)
    path = np.vstack([past, np.zeros((1, 3), np.float32), fut])        # t = -4 .. 0 .. +5 s (index of t0 = len(past))
    k0 = len(past)
    # heading from the path tangent (keeps the last value when stationary)
    d = np.gradient(path[:, :2], axis=0)
    hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    hd[np.hypot(d[:, 0], d[:, 1]) < 1e-3] = np.nan
    for i in range(len(hd)):
        if np.isnan(hd[i]):
            hd[i] = hd[i - 1] if i else 0.0
    files, jpgs = {}, {}
    for im in f.frame.images:
        fn = f"f0000_{im.name}.jpg"
        with open(os.path.join(odir, "cam", fn), "wb") as fh:
            fh.write(im.image)
        files[int(im.name)] = fn
        jpgs[int(im.name)] = bytes(im.image)
    cals = calibrations(f.frame)
    est = estimate_points(jpgs, cals) if estimate else (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8),
                                                        np.zeros(0, np.int8))
    prefs = [states(p) for p in f.preference_trajectories]
    L = max([len(p) for p in prefs] + [1])
    pref_arr = np.full((len(prefs), L, 3), np.nan, np.float32)
    for j, p in enumerate(prefs):
        pref_arr[j, :len(p)] = p
    np.savez_compressed(os.path.join(odir, "traj.npz"), past=past, future=fut, prefs=pref_arr, k0=k0,
                        pref_scores=np.array([p.preference_score for p in f.preference_trajectories], np.float32),
                        intent=np.int8(f.intent), hz=hz)
    empty = dict(pts=np.zeros((0, 3), np.float32), inten=np.zeros(0, np.float32), rng=np.zeros(0, np.float32),
                 rgb=np.zeros((0, 3), np.uint8), cam=np.zeros(0, np.int8), sensor=np.zeros(0, np.uint8),
                 boxes=np.zeros((0, 10), np.float32))
    poses, ts = [], []
    for k, (p, h) in enumerate(zip(path, hd)):
        c, s_ = np.cos(h), np.sin(h)
        M = np.eye(4)
        M[:2, :2] = [[c, -s_], [s_, c]]
        M[:3, 3] = p
        poses.append(M.tolist())
        ts.append(int(round((k - k0) / hz * 1e6)) + 10_000_000)
        fr = dict(empty)
        if k == k0 and len(est[0]):                     # the captured moment carries the estimated cloud
            n = len(est[0])
            fr.update(pts=est[0], rgb=est[1], cam=est[2], sensor=np.ones(n, np.uint8),
                      rng=np.linalg.norm(est[0], axis=1).astype(np.float32), inten=np.full(n, 0.25, np.float32))
        np.savez_compressed(os.path.join(odir, "frames", f"f{k:04d}.npz"), **fr)
    meta = {"context": "E2E_" + name, "kind": "e2e", "frames": len(path), "timestamps_micros": ts,
            "origin_world": [0, 0, 0], "poses_world_vehicle": poses, "cam_files": [files] * len(path), "max_points": 0,
            "synthetic": False, "dataset": "Waymo Open End-to-End Driving (camera) v1.0.0", "current_index": k0,
            "intent": INTENTS.get(int(f.intent), "?"), "n_preferences": len(prefs), "calibrations": cals,
            "estimated_points": bool(len(est[0])), "max_points": int(len(est[0])),
            "pref_scores": [float(p.preference_score) for p in f.preference_trajectories]}
    with open(os.path.join(odir, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--scan", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(PROJ, "data", "processed"))
    a = ap.parse_args()
    # one moment per route intent (left / right / straight) so the set is varied
    picked, seen = [], set()
    for i, raw in enumerate(iter_records(a.shard, limit=a.scan)):
        f = parse(raw)
        key = int(f.intent)
        if key not in seen and len(f.future_states.pos_x) and len(f.preference_trajectories):
            seen.add(key)
            picked.append(raw)
        if len(picked) >= a.count:
            break
    for raw in picked:
        m = export_moment(raw, a.out)
        print(m["context"], "frames", m["frames"], "intent", m["intent"], "prefs", m["n_preferences"],
              [round(x, 1) for x in m["pref_scores"]])
