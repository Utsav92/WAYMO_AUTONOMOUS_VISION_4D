"""Decode Waymo v2 LiDAR range-image rows into world-frame point clouds (one dict per frame)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))
from waymo_adapter import find_col, as_matrix, read_component, component_path, LASER_NAMES  # noqa: E402
from calibration import beam_inclinations, range_image_to_points, transform  # noqa: E402
from motion_compensation import compensate_top  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def load_lidar_calibration(df) -> dict:
    """laser_name -> {extrinsic, inclination-source}"""
    c_laser = find_col(df.columns, "laser_name")
    c_ext = find_col(df.columns, "extrinsic.transform")
    out = {}
    for _, row in df.iterrows():
        info = {"extrinsic": as_matrix(row[c_ext]), "bi_min": None, "bi_max": None, "values": None}
        for k, suf in (("bi_min", "beam_inclination.min"), ("bi_max", "beam_inclination.max")):
            try:
                v = row[find_col(df.columns, suf)]
                info[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
            except KeyError:
                pass
        try:
            vals = row[find_col(df.columns, "beam_inclination.values")]
            if vals is not None and len(vals) > 0:
                info["values"] = np.asarray(vals, dtype=np.float64)
        except KeyError:
            pass
        out[int(row[c_laser])] = info
    return out


def decode_row(values, shape, calib, ret=1):
    img = np.asarray(values, dtype=np.float32).reshape(tuple(int(s) for s in shape))
    H = img.shape[0]
    inc = beam_inclinations(H, calib["bi_min"], calib["bi_max"], calib["values"])
    return range_image_to_points(img, calib["extrinsic"], inc)


_POSE_CACHE = {}


def pose_image_at(root, context, split, ts, laser=1):
    """(H,W,6) roll,pitch,yaw,x,y,z for the TOP range image of frame ts (None if the component is absent).
    The component is a single big row group, so it is read ONCE per process and indexed afterwards."""
    p = component_path(root, "lidar_pose", context, split)
    if p is None:
        return None
    if p not in _POSE_CACHE:
        tb = pq.read_table(p)
        names = tb.column_names
        _POSE_CACHE.clear()
        _POSE_CACHE[p] = {
            "tb": tb,
            "ts": np.asarray(tb[next(n for n in names if n.endswith("frame_timestamp_micros"))].to_numpy()),
            "ln": np.asarray(tb[next(n for n in names if n.endswith("laser_name"))].to_numpy()),
            "v": tb[next(n for n in names if n.endswith("range_image_return1.values"))],
            "s": tb[next(n for n in names if n.endswith("range_image_return1.shape"))],
        }
    c = _POSE_CACHE[p]
    idx = np.nonzero((c["ts"] == int(ts)) & (c["ln"] == int(laser)))[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    vals = c["v"][i].values.to_numpy(zero_copy_only=False).astype(np.float32)
    return vals.reshape(tuple(int(x) for x in c["s"][i].as_py()))


def _row_group_frames(path, c_ts_name, wanted, n_lasers):
    """Yield (ts, DataFrame of that frame's lidar rows) in timestamp order, reading one row group at a time."""
    f = pq.ParquetFile(path)
    buf, want = {}, set(int(t) for t in wanted)
    for rg in range(f.num_row_groups):
        df = f.read_row_group(rg).to_pandas()
        for ts, g in df.groupby(c_ts_name, sort=True):
            if int(ts) in want:
                buf.setdefault(int(ts), []).append(g)
        del df
        for ts in sorted(buf):
            if sum(len(x) for x in buf[ts]) >= n_lasers:
                import pandas as pd
                yield ts, pd.concat(buf.pop(ts))
            else:
                break
    for ts in sorted(buf):
        import pandas as pd
        yield ts, pd.concat(buf[ts])


def _load_seg_labels(root, context, split):
    """(ts, laser) -> semantic image (H,W) for the frames Waymo labelled; {} when the component is absent."""
    p = component_path(root, "lidar_segmentation", context, split)
    if p is None:
        return {}
    out = {}
    for r in pq.read_table(p).to_pylist():
        v = np.asarray(r[next(k for k in r if k.endswith("range_image_return1.values"))], np.int32)
        sh = tuple(r[next(k for k in r if k.endswith("range_image_return1.shape"))])
        out[(int(r["key.frame_timestamp_micros"]), int(r["key.laser_name"]))] = v.reshape(sh)[..., 1].astype(np.uint8)
    return out


def decode_frames(root, context, split=None, max_frames=None, returns=(1,), frame_stride=1, motion_comp=True,
                  sem_model=None):
    """Yields dict(ts, xyz_vehicle[N,3], attrs[N,3], sensor[N] uint8, T_world_vehicle[4,4]).  Streams the lidar
    parquet one row group at a time (the file is too large to hold as one pandas frame)."""
    cal = load_lidar_calibration(read_component(root, "lidar_calibration", context, split))
    pose = read_component(root, "vehicle_pose", context, split)
    c_pose_ts = find_col(pose.columns, "frame_timestamp_micros")
    c_pose = find_col(pose.columns, "world_from_vehicle.transform")
    poses = {int(t): as_matrix(m) for t, m in zip(pose[c_pose_ts], pose[c_pose])}
    stamps = sorted(poses)[::frame_stride]
    if max_frames:
        stamps = stamps[:max_frames]
    seg_gt = _load_seg_labels(root, context, split)
    seg_mod = None
    if sem_model is not None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "machine_learning"))
        import lidar_segmentation as LS
        seg_mod = LS.load(sem_model)
    lpath = component_path(root, "lidar", context, split)
    names = pq.ParquetFile(lpath).schema_arrow.names
    c_ts = next(n for n in names if n.endswith("frame_timestamp_micros"))
    c_laser = next(n for n in names if n.endswith("laser_name"))
    for ts, rows in _row_group_frames(lpath, c_ts, stamps, len(cal)):
        pose_img = pose_image_at(root, context, split, ts) if motion_comp else None
        xs, at, sn, sm = [], [], [], []
        for _, r in rows.iterrows():
            ln = int(r[c_laser])
            for ret in returns:
                vcol = next((n for n in names if n.endswith(f"range_image_return{ret}.values")), None)
                scol = next((n for n in names if n.endswith(f"range_image_return{ret}.shape")), None)
                if vcol is None or r[vcol] is None or len(r[vcol]) == 0:
                    continue
                p, a, _, rc = decode_row(r[vcol], r[scol], cal[ln], ret)
                if (ln == 1 and ret == 1 and pose_img is not None
                        and pose_img.shape[:2] == tuple(int(x) for x in r[scol][:2])):
                    p, _ = compensate_top(p, rc[:, 0], rc[:, 1], pose_img, poses[int(ts)])   # per-pixel vehicle pose
                xs.append(p)
                at.append(a)
                sn.append(np.full(len(p), ln, np.uint8))
                if ret == 1:
                    lab = seg_gt.get((int(ts), ln))
                    if lab is not None and lab.shape == tuple(int(x) for x in r[scol][:2]):
                        sm.append(lab[rc[:, 0], rc[:, 1]])
                    elif seg_mod is not None:
                        sm.append(seg_mod.predict_range_image(r[vcol], r[scol], cal[ln], ln))
                    else:
                        sm.append(np.zeros(len(p), np.uint8))
                else:
                    sm.append(np.zeros(len(p), np.uint8))
        yield {"ts": int(ts),
               "xyz_vehicle": np.concatenate(xs) if xs else np.zeros((0, 3), np.float32),
               "attrs": np.concatenate(at) if at else np.zeros((0, 3), np.float32),
               "sensor": np.concatenate(sn) if sn else np.zeros((0,), np.uint8),
               "sem": np.concatenate(sm) if sm else np.zeros((0,), np.uint8),
               "sem_is_gt": any((int(ts), int(l)) in seg_gt for l in cal),
               "T_world_vehicle": poses[int(ts)]}


def to_world(frame):
    return transform(frame["T_world_vehicle"], frame["xyz_vehicle"].astype(np.float64)).astype(np.float64)


def voxel_downsample(xyz: np.ndarray, voxel: float, *arrays):
    """Keep one point per voxel. Returns (xyz, *arrays) filtered."""
    if voxel <= 0 or len(xyz) == 0:
        return (xyz,) + arrays
    k = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    idx.sort()
    return (xyz[idx],) + tuple(a[idx] for a in arrays)
