"""Validate my LiDAR->camera projection against Waymo's own `lidar_camera_projection` (pixel coordinates the dataset
authors computed with rolling-shutter handling).  Reports pixel error for each model variant.

    python validate_projection.py --root D:/WAYMO_DATA/raw --context NAME --split validation --frames 5 60 120
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "datasets"))
from waymo_adapter import find_col, as_matrix, component_path  # noqa: E402
from calibration import range_image_to_points, beam_inclinations, project_to_camera, transform, inv_rigid  # noqa: E402
from decode_lidar import load_lidar_calibration  # noqa: E402
from decode_camera import load_camera_calibration  # noqa: E402
from motion_compensation import rpy_to_matrix, project_rolling  # noqa: E402


def rows_at(root, comp, ctx, split, ts, cols=None):
    p = component_path(root, comp, ctx, split)
    return pq.read_table(p, columns=cols, filters=[("key.frame_timestamp_micros", "=", int(ts))]).to_pylist()


def cam_meta(root, ctx, split, ts, cam):
    p = component_path(root, "camera_image", ctx, split)
    names = [n for n in pq.ParquetFile(p).schema_arrow.names if not n.endswith("].image")]
    r = pq.read_table(p, columns=names, filters=[("key.frame_timestamp_micros", "=", int(ts)),
                                                 ("key.camera_name", "=", int(cam))]).to_pylist()[0]
    g = lambda s: next(v for k, v in r.items() if k.endswith(s))
    return {"pose": as_matrix(g("pose.transform")),
            "lin_vel": [g("linear_velocity.x"), g("linear_velocity.y"), g("linear_velocity.z")],
            "ang_vel": [g("angular_velocity.x"), g("angular_velocity.y"), g("angular_velocity.z")],
            "pose_timestamp": g("pose_timestamp"), "shutter": g("rolling_shutter_params.shutter"),
            "camera_trigger_time": g("rolling_shutter_params.camera_trigger_time"),
            "camera_readout_done_time": g("rolling_shutter_params.camera_readout_done_time")}


def validate(root, ctx, split, frame_idx, camera=1, max_pts=40000, seed=0):
    rng = np.random.default_rng(seed)
    cams = load_camera_calibration(root, ctx, split)
    cal = load_lidar_calibration(__import__("waymo_adapter").read_component(root, "lidar_calibration", ctx, split))
    pose_tbl = __import__("waymo_adapter").read_component(root, "vehicle_pose", ctx, split)
    ts_all = sorted(pose_tbl[find_col(pose_tbl.columns, "frame_timestamp_micros")].tolist())
    c_dir = pq.read_table(component_path(root, "camera_calibration", ctx, split)).to_pylist()
    direction = next(r["[CameraCalibrationComponent].rolling_shutter_direction"] for r in c_dir
                     if r["key.camera_name"] == camera)
    out = {}
    for fi in frame_idx:
        ts = ts_all[fi]
        T_frame = as_matrix(pose_tbl[pose_tbl[find_col(pose_tbl.columns, "frame_timestamp_micros")] == ts].iloc[0][
            find_col(pose_tbl.columns, "world_from_vehicle.transform")])
        lid = rows_at(root, "lidar", ctx, split, ts)
        proj = {r["key.laser_name"]: r for r in rows_at(root, "lidar_camera_projection", ctx, split, ts)}
        pose_rows = {r["key.laser_name"]: r for r in rows_at(root, "lidar_pose", ctx, split, ts)} \
            if component_path(root, "lidar_pose", ctx, split) else {}
        meta = cam_meta(root, ctx, split, ts, camera)
        cam = cams[camera]
        P_v, P_vc, P_w_top, ref = [], [], [], []
        for r in lid:
            ln = r["key.laser_name"]
            vcol = next(k for k in r if k.endswith("range_image_return1.values"))
            scol = next(k for k in r if k.endswith("range_image_return1.shape"))
            img = np.asarray(r[vcol], np.float32).reshape(tuple(r[scol]))
            inc = beam_inclinations(img.shape[0], cal[ln]["bi_min"], cal[ln]["bi_max"], cal[ln]["values"])
            xyz, _, valid, rc = range_image_to_points(img, cal[ln]["extrinsic"], inc)
            pj = proj[ln]
            pv = np.asarray(pj[next(k for k in pj if k.endswith("range_image_return1.values"))], np.float32)
            pv = pv.reshape(tuple(pj[next(k for k in pj if k.endswith("range_image_return1.shape"))]))
            ref_px = pv[rc[:, 0], rc[:, 1]]                       # (N,6) cam1,x1,y1,cam2,x2,y2
            world_nc = transform(T_frame, xyz.astype(np.float64))
            world_c = world_nc
            if ln == 1 and 1 in pose_rows:
                pr = pose_rows[ln]
                pimg = np.asarray(pr[next(k for k in pr if k.endswith("range_image_return1.values"))], np.float32)
                pimg = pimg.reshape(tuple(pr[next(k for k in pr if k.endswith("range_image_return1.shape"))]))
                p = pimg[rc[:, 0], rc[:, 1]].astype(np.float64)
                R = rpy_to_matrix(p[:, 0], p[:, 1], p[:, 2])
                world_c = np.einsum("nij,nj->ni", R, xyz.astype(np.float64)) + p[:, 3:6]
            P_v.append(world_nc)
            P_vc.append(world_c)
            ref.append(ref_px)
        W_nc, W_c, REF = np.concatenate(P_v), np.concatenate(P_vc), np.concatenate(ref)
        sel = np.nonzero(REF[:, 0] == camera)[0]
        if len(sel) > max_pts:
            sel = rng.choice(sel, max_pts, replace=False)
        ru, rv = REF[sel, 1], REF[sel, 2]
        Tcam = meta["pose"]

        def err(world, mode):
            if mode == "global":
                pv_ = transform(inv_rigid(Tcam), world[sel])
                u, v, d, ok = project_to_camera(pv_, cam["T_vehicle_camera"], cam["intrinsic"], cam["width"], cam["height"])
            else:
                aw, tm = mode
                u, v, d = project_rolling(world[sel], meta, cam["T_vehicle_camera"], cam["intrinsic"], cam["width"],
                                          cam["height"], direction, ang_in_world=aw, time_mode=tm)
                ok = d > 0.1
            e = np.hypot(u - ru, v - rv)[ok]
            return {"n": int(ok.sum()), "median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90)),
                    "mean_px": float(e.mean()), "bias_u": float(np.median((u - ru)[ok])),
                    "bias_v": float(np.median((v - rv)[ok]))}
        res = {}
        for wname, W in (("frame_pose", W_nc), ("pixel_pose", W_c)):
            res[wname + "/global"] = err(W, "global")
            for aw in (True, False):
                for tm in ("center", "none"):
                    res[f"{wname}/rolling(ang_world={aw},{tm})"] = err(W, (aw, tm))
        out[fi] = res
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--frames", type=int, nargs="+", default=[5, 60, 120])
    ap.add_argument("--camera", type=int, default=1)
    a = ap.parse_args()
    res = validate(a.root, a.context, a.split, a.frames, a.camera)
    keys = list(next(iter(res.values())))
    print("pixel error vs Waymo lidar_camera_projection (camera %d), median / p90 [px], averaged over frames" % a.camera)
    for k in keys:
        med = np.mean([res[f][k]["median_px"] for f in res])
        p90 = np.mean([res[f][k]["p90_px"] for f in res])
        bu = np.mean([res[f][k]["bias_u"] for f in res])
        bv = np.mean([res[f][k]["bias_v"] for f in res])
        print(f"  {k:46s} median {med:7.3f}   p90 {p90:7.3f}   bias(u,v) {bu:+.2f},{bv:+.2f}")
