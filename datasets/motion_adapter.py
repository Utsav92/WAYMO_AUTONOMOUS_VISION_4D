"""Waymo Open MOTION dataset (scenario protos) -> the same processed layout the TouchDesigner project reads.

Adds what Perception v2 does not have: vector MAP polylines (lanes, road lines/edges, crosswalks, speed bumps,
driveways, stop signs) and ground-truth FUTURE trajectories of every agent.  Motion scenarios are 9.1 s (91 steps at
10 Hz, current time index 10) and are NOT the same recordings as the Perception segments - there is no published
mapping, so they are shown as a separate source and labelled MOTION.

    python motion_adapter.py --shard D:/WAYMO_DATA/motion/validation.tfrecord-00000-of-00150 --count 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "waymo_pb"))
from tfrecord import iter_records  # noqa: E402

PROJ = os.path.dirname(HERE)
MOTION_TYPE_TO_BOX = {1: 1, 2: 2, 3: 4}      # motion: 1 vehicle, 2 pedestrian, 3 cyclist -> perception ids (4 = cyclist)
MAP_KINDS = ("lane", "road_line", "road_edge", "stop_sign", "crosswalk", "speed_bump", "driveway")


def _scenario_cls():
    from waymo_open_dataset.protos import scenario_pb2
    return scenario_pb2.Scenario


def parse_scenario(raw: bytes):
    s = _scenario_cls()()
    s.ParseFromString(raw)
    return s


def map_polylines(sc):
    """kind -> list of (N,3) arrays (polygons are closed by repeating the first point)."""
    out = {k: [] for k in MAP_KINDS}
    for mf in sc.map_features:
        which = mf.WhichOneof("feature_data")
        if which not in out:
            continue
        d = getattr(mf, which)
        if which == "stop_sign":
            p = d.position
            out[which].append(np.array([[p.x, p.y, p.z]], np.float32))
            continue
        pts = d.polygon if which in ("crosswalk", "speed_bump", "driveway") else d.polyline
        a = np.array([[p.x, p.y, p.z] for p in pts], np.float32)
        if which in ("crosswalk", "speed_bump", "driveway") and len(a):
            a = np.vstack([a, a[:1]])
        if len(a) >= 1:
            out[which].append(a)
    return out


def tracks_array(sc):
    """-> arrays (T, K, field) float32: x,y,z,l,w,h,heading,vx,vy,valid ; plus types[K], ids[K]."""
    K, T = len(sc.tracks), len(sc.timestamps_seconds)
    A = np.zeros((T, K, 10), np.float32)
    types = np.zeros(K, np.int8)
    ids = np.zeros(K, np.int64)
    for k, tr in enumerate(sc.tracks):
        types[k], ids[k] = tr.object_type, tr.id
        for t, st in enumerate(tr.states):
            A[t, k] = (st.center_x, st.center_y, st.center_z, st.length, st.width, st.height, st.heading,
                       st.velocity_x, st.velocity_y, 1.0 if st.valid else 0.0)
    return A, types, ids


def export_scenario(sc, out_root, label=None):
    sid = sc.scenario_id
    name = "MOTION_" + sid
    odir = os.path.join(out_root, name)
    os.makedirs(os.path.join(odir, "frames"), exist_ok=True)
    A, types, ids = tracks_array(sc)
    T, K, _ = A.shape
    sdc = int(sc.sdc_track_index)
    ego = A[:, sdc]
    first = int(np.argmax(ego[:, 9] > 0))
    origin = ego[first, :3].copy()
    poses = []
    for t in range(T):
        c, s = np.cos(ego[t, 6]), np.sin(ego[t, 6])
        M = np.eye(4)
        M[:2, :2] = [[c, -s], [s, c]]
        M[:3, 3] = ego[t, :3] - origin
        poses.append(M.tolist())
    ts = [int(round(x * 1e6)) for x in sc.timestamps_seconds]
    empty = dict(pts=np.zeros((0, 3), np.float32), inten=np.zeros(0, np.float32), rng=np.zeros(0, np.float32),
                 rgb=np.zeros((0, 3), np.uint8), cam=np.zeros(0, np.int8), sensor=np.zeros(0, np.uint8))
    for t in range(T):
        rows = []
        for k in range(K):
            if A[t, k, 9] < 0.5 or types[k] not in MOTION_TYPE_TO_BOX:
                continue
            x, y, z, l, w, h, hd, vx, vy, _ = A[t, k]
            rows.append([x - origin[0], y - origin[1], z - origin[2], l, w, h, hd, MOTION_TYPE_TO_BOX[int(types[k])],
                         k, float(np.hypot(vx, vy))])
        np.savez_compressed(os.path.join(odir, "frames", f"f{t:04d}.npz"),
                            boxes=np.array(rows, np.float32).reshape(-1, 10), **empty)
    mp = map_polylines(sc)
    arrs = {}
    for kind, lst in mp.items():
        if lst:
            arrs[kind + "_pts"] = (np.concatenate(lst) - origin.astype(np.float32)).astype(np.float32)
            arrs[kind + "_off"] = np.cumsum([0] + [len(a) for a in lst]).astype(np.int64)
    np.savez_compressed(os.path.join(odir, "map.npz"), **arrs)
    traj = A[:, :, :3] - origin
    np.savez_compressed(os.path.join(odir, "future.npz"), traj=traj.astype(np.float32), valid=A[:, :, 9] > 0.5,
                        types=types, ids=ids, sdc=sdc,
                        to_predict=np.array([p.track_index for p in sc.tracks_to_predict], np.int32),
                        current_index=int(sc.current_time_index))
    meta = {"context": name, "kind": "motion", "frames": T, "timestamps_micros": ts,
            "origin_world": origin.tolist(), "poses_world_vehicle": poses,
            "cam_files": [{} for _ in range(T)], "max_points": 0, "synthetic": False,
            "current_time_index": int(sc.current_time_index), "n_tracks": int(K),
            "map_counts": {k: len(v) for k, v in mp.items()}, "scenario_id": sid, "dataset": "Waymo Open Motion"}
    with open(os.path.join(odir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def score(sc):
    """Prefer busy scenarios with lots of map + agents."""
    return len(sc.tracks) + 0.02 * len(sc.map_features) + 3 * len(sc.tracks_to_predict)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--scan", type=int, default=60, help="scenarios to scan when picking the busiest")
    ap.add_argument("--out", default=os.path.join(PROJ, "data", "processed"))
    a = ap.parse_args()
    cands = []
    for i, raw in enumerate(iter_records(a.shard, limit=a.scan)):
        sc = parse_scenario(raw)
        cands.append((score(sc), i, sc))
    cands.sort(key=lambda c: -c[0])
    for sc_score, i, sc in cands[: a.count]:
        m = export_scenario(sc, a.out)
        print(m["context"], "tracks", m["n_tracks"], "map", m["map_counts"], "score %.0f" % sc_score)
