"""Camera images + calibration from Waymo v2 parquet."""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))
from waymo_adapter import find_col, as_matrix, read_component, CAMERA_NAMES  # noqa: E402


def load_camera_calibration(root, context, split=None):
    df = read_component(root, "camera_calibration", context, split)
    cn = find_col(df.columns, "camera_name")
    out = {}
    for _, r in df.iterrows():
        intr = {}
        try:  # v2 stores intrinsic as a flat [f_u,f_v,c_u,c_v,k1,k2,p1,p2,k3] array
            arr = r[find_col(df.columns, "intrinsic")]
            for name, val in zip(("f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3"), arr):
                intr[name] = float(val)
        except Exception:
            for name in ("f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3"):
                try:
                    intr[name] = float(r[find_col(df.columns, "intrinsic." + name)])
                except KeyError:
                    intr[name] = 0.0
        out[int(r[cn])] = {"intrinsic": intr,
                           "T_vehicle_camera": as_matrix(r[find_col(df.columns, "extrinsic.transform")]),
                           "width": int(r[find_col(df.columns, "width")]),
                           "height": int(r[find_col(df.columns, "height")]),
                           "direction": int(r[find_col(df.columns, "rolling_shutter_direction")])
                           if any(c.endswith("rolling_shutter_direction") for c in df.columns) else 5}
    return out


def decode_jpeg(b) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("JPEG decode failed")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def frame_images(root, context, split=None):
    """dict ts -> dict camera_id -> (jpeg_bytes, T_world_vehicle_at_image_time or None)."""
    df = read_component(root, "camera_image", context, split)
    c_ts, c_cam = find_col(df.columns, "frame_timestamp_micros"), find_col(df.columns, "camera_name")
    c_img = find_col(df.columns, "image", contains="CameraImageComponent")
    try:
        c_pose = find_col(df.columns, "pose.transform")
    except KeyError:
        c_pose = None
    def g(r, suf):
        return next((r[c] for c in df.columns if c.endswith(suf)), None)

    out = {}
    for _, r in df.iterrows():
        rs = None
        if g(r, "rolling_shutter_params.camera_trigger_time") is not None and c_pose:
            rs = {"pose": as_matrix(r[c_pose]), "pose_timestamp": float(g(r, "pose_timestamp")),
                  "lin_vel": [float(g(r, f"linear_velocity.{a}")) for a in "xyz"],
                  "ang_vel": [float(g(r, f"angular_velocity.{a}")) for a in "xyz"],
                  "shutter": float(g(r, "rolling_shutter_params.shutter")),
                  "camera_trigger_time": float(g(r, "rolling_shutter_params.camera_trigger_time")),
                  "camera_readout_done_time": float(g(r, "rolling_shutter_params.camera_readout_done_time"))}
        out.setdefault(int(r[c_ts]), {})[int(r[c_cam])] = (r[c_img], as_matrix(r[c_pose]) if c_pose else None, rs)
    return out
