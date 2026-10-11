"""Ground-truth 3D boxes (Waymo LiDAR labels) -> per-frame arrays + track continuity helpers.
These are dataset annotations, NOT model predictions."""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))
from waymo_adapter import find_col, read_component  # noqa: E402

BOX_DTYPE = np.dtype([("ts", "i8"), ("track", "i8"), ("type", "i1"), ("cx", "f8"), ("cy", "f8"), ("cz", "f8"),
                      ("l", "f4"), ("w", "f4"), ("h", "f4"), ("heading", "f4"), ("spd", "f4")])


def load_boxes(root, context, split=None):
    """Boxes are in the VEHICLE frame of their frame (Waymo v2 convention).  Track ids are hashed strings -> int."""
    df = read_component(root, "lidar_box", context, split)
    g = lambda s: find_col(df.columns, s)
    c_ts, c_id = g("frame_timestamp_micros"), g("laser_object_id")
    c_type = g("type")
    cx, cy, cz = g("box.center.x"), g("box.center.y"), g("box.center.z")
    sx, sy, sz = g("box.size.x"), g("box.size.y"), g("box.size.z")
    hd = g("box.heading")
    ids = {}
    out = np.zeros(len(df), BOX_DTYPE)
    for i, (ts, oid) in enumerate(zip(df[c_ts], df[c_id])):
        out["track"][i] = ids.setdefault(oid, len(ids))
        out["ts"][i] = ts
    out["type"] = df[c_type].to_numpy()
    for k, c in (("cx", cx), ("cy", cy), ("cz", cz), ("l", sx), ("w", sy), ("h", sz), ("heading", hd)):
        out[k] = df[c].to_numpy()
    out["spd"] = np.nan                       # provided metadata speed (m/s) when the component has it
    try:
        vx, vy = df[g("speed.x")].to_numpy(dtype=float), df[g("speed.y")].to_numpy(dtype=float)
        out["spd"] = np.hypot(vx, vy)
    except KeyError:
        pass
    return out, ids


def tracks(boxes):
    d = defaultdict(list)
    for b in boxes:
        d[int(b["track"])].append(b)
    for k in d:
        d[k].sort(key=lambda b: b["ts"])
    return d


def track_gaps(boxes, frame_ts):
    """For every track: list of frame indices where it is absent between first and last appearance."""
    pos = {int(t): i for i, t in enumerate(sorted(frame_ts))}
    gaps = {}
    for tid, bs in tracks(boxes).items():
        idx = [pos[int(b["ts"])] for b in bs]
        missing = sorted(set(range(min(idx), max(idx) + 1)) - set(idx))
        if missing:
            gaps[tid] = missing
    return gaps


def speed_from_track(bs, i, world_pos):
    """m/s from consecutive observations (timestamps in micros).  ``world_pos[i]`` = [x,y,z] for bs[i].
    Returns None unless two valid observations exist - speed is never invented."""
    if i <= 0 or i >= len(bs):
        return None
    dt = (bs[i]["ts"] - bs[i - 1]["ts"]) * 1e-6
    if dt <= 0 or dt > 0.5:
        return None
    return float(np.linalg.norm(np.asarray(world_pos[i]) - np.asarray(world_pos[i - 1])) / dt)
