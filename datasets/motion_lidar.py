"""Decode the delta-compressed LiDAR of Waymo Open Motion (`lidar_and_camera` files) and merge it into an exported
Motion scenario.  Only the first 11 timesteps (1 s of history) carry LiDAR in this dataset.

Delta format (verified against plausible ranges, see README): zlib -> DeltaEncodedData.
  mask     alternating run lengths (valid, invalid, valid, ...) over the flattened CHANNEL-MAJOR array [C, H, W]
  residual sint64, one per valid cell; value_q = cumsum(residual); value = value_q * quant_precision[channel]
"""
from __future__ import annotations

import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(HERE, "waymo_pb"), os.path.join(PROJ, "preprocessing")]
from tfrecord import iter_records  # noqa: E402
from calibration import range_image_to_points, beam_inclinations, transform  # noqa: E402
from motion_compensation import rpy_to_matrix  # noqa: E402


def delta_decode(buf) -> np.ndarray:
    """-> float32 array [H, W, C]."""
    from waymo_open_dataset.protos import compressed_lidar_pb2 as cl
    d = cl.DeltaEncodedData()
    d.ParseFromString(zlib.decompress(bytes(buf)))
    H, W, C = (int(x) for x in d.metadata.shape)
    tot = H * W * C
    runs = np.cumsum(np.asarray(d.mask, dtype=np.int64))
    starts = np.concatenate([[0], runs[:-1]])
    valid = np.zeros(tot, bool)
    for k in range(0, len(runs), 2):                  # even runs are valid cells
        valid[starts[k]:runs[k]] = True
    q = np.cumsum(np.asarray(d.residual, dtype=np.int64))
    prec = np.asarray(d.metadata.quant_precision, np.float32)
    chan = np.nonzero(valid)[0] // (H * W)
    out = np.zeros(tot, np.float32)
    out[valid] = q.astype(np.float32) * prec[chan]
    return out.reshape(C, H, W).transpose(1, 2, 0)


def frames_from_file(path, origin, max_range=75.0, voxel=0.1, max_points=131072, seed=0):
    """-> list (one per lidar timestep) of dict(pts [N,3] world-relative, inten, rng, sensor)."""
    from motion_adapter import parse_scenario
    rng = np.random.default_rng(seed)
    sc = parse_scenario(next(iter_records(path, limit=1)))
    out = []
    for fr in sc.compressed_frame_laser_data:
        T = np.array(fr.pose.transform, dtype=np.float64).reshape(4, 4)
        cal = {int(c.name): c for c in fr.laser_calibrations}
        P, I, R, S = [], [], [], []
        for L in fr.lasers:
            ln = int(L.name)
            c = cal.get(ln)
            if c is None or not len(L.ri_return1.range_image_delta_compressed):
                continue
            img = delta_decode(L.ri_return1.range_image_delta_compressed)
            ext = np.array(c.extrinsic.transform, dtype=np.float64).reshape(4, 4)
            vals = np.asarray(c.beam_inclinations, np.float64)
            inc = beam_inclinations(img.shape[0], c.beam_inclination_min, c.beam_inclination_max,
                                    vals if len(vals) == img.shape[0] else None)
            pts, attrs, valid, rc = range_image_to_points(img, ext, inc)
            if ln == 1 and len(L.ri_return1.range_image_pose_delta_compressed):     # per-pixel vehicle pose (TOP)
                pimg = delta_decode(L.ri_return1.range_image_pose_delta_compressed)
                p = pimg[rc[:, 0], rc[:, 1]].astype(np.float64)
                Rm = rpy_to_matrix(p[:, 0], p[:, 1], p[:, 2])
                world = np.einsum("nij,nj->ni", Rm, pts.astype(np.float64)) + p[:, 3:6]
                bad = ~np.isfinite(world).all(axis=1)
                if bad.any():
                    world[bad] = transform(T, pts[bad].astype(np.float64))
            else:
                world = transform(T, pts.astype(np.float64))
            P.append(world - origin)
            I.append(np.clip(attrs[:, 0], 0, 2.0))
            R.append(attrs[:, 2])
            S.append(np.full(len(pts), ln, np.uint8))
        if not P:
            out.append(None)
            continue
        P, I, R, S = np.concatenate(P), np.concatenate(I), np.concatenate(R), np.concatenate(S)
        k = R <= max_range
        P, I, R, S = P[k], I[k], R[k], S[k]
        key = np.floor(P / voxel).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        idx.sort()
        P, I, R, S = P[idx], I[idx], R[idx], S[idx]
        if len(P) > max_points:
            s = rng.choice(len(P), max_points, replace=False)
            P, I, R, S = P[s], I[s], R[s], S[s]
        out.append({"pts": P.astype(np.float32), "inten": I.astype(np.float32), "rng": R.astype(np.float32), "sensor": S})
    return out


def merge_into(context_dir, lidar_file):
    """Write the decoded LiDAR into frames f0000..f00NN of an exported MOTION_<id> directory."""
    import json
    with open(os.path.join(context_dir, "meta.json")) as f:
        meta = json.load(f)
    origin = np.array(meta["origin_world"], dtype=np.float64)
    frames = frames_from_file(lidar_file, origin)
    n_ok = 0
    for i, fr in enumerate(frames):
        if fr is None:
            continue
        p = os.path.join(context_dir, "frames", f"f{i:04d}.npz")
        d = dict(np.load(p))
        n = len(fr["pts"])
        d.update(pts=fr["pts"], inten=fr["inten"], rng=fr["rng"], sensor=fr["sensor"],
                 rgb=np.zeros((n, 3), np.uint8), cam=np.full(n, -1, np.int8), sem=np.zeros(n, np.uint8))
        np.savez_compressed(p, **d)
        n_ok += 1
    # static world memory: the real LiDAR of the first second, minus every point on a moving agent, held afterwards
    import sys as _s
    _s.path.insert(0, os.path.join(PROJ, "preprocessing"))
    from temporal_history import static_mask
    keep_p, keep_i, keep_r, keep_s = [], [], [], []
    for i in range(n_ok):
        d = np.load(os.path.join(context_dir, "frames", f"f{i:04d}.npz"))
        P = d["pts"].astype(np.float64)
        b = d["boxes"]
        mov = b[(b[:, 9] > 0.5) | np.isin(b[:, 7], (2, 4))] if len(b) else b
        if len(mov):
            m = static_mask(P, mov[:, :3], mov[:, 3:6], mov[:, 6], margin=0.5)
        else:
            m = np.ones(len(P), bool)
        keep_p.append(d["pts"][m]); keep_i.append(d["inten"][m]); keep_r.append(d["rng"][m]); keep_s.append(d["sensor"][m])
    if keep_p:
        P = np.concatenate(keep_p); I = np.concatenate(keep_i); Rg = np.concatenate(keep_r); Sn = np.concatenate(keep_s)
        key = np.floor(P / 0.15).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        idx.sort()
        P, I, Rg, Sn = P[idx], I[idx], Rg[idx], Sn[idx]
        if len(P) > 131072:
            sel = np.random.default_rng(0).choice(len(P), 131072, replace=False)
            P, I, Rg, Sn = P[sel], I[sel], Rg[sel], Sn[sel]
        n = len(P)
        np.savez_compressed(os.path.join(context_dir, "static_memory.npz"), pts=P, inten=I, rng=Rg, sensor=Sn,
                            rgb=np.zeros((n, 3), np.uint8), cam=np.full(n, -1, np.int8), sem=np.zeros(n, np.uint8))
        meta["static_memory_points"] = int(n)
    meta["lidar_frames"] = n_ok
    meta["max_points"] = 131072
    with open(os.path.join(context_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return n_ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=os.path.join(PROJ, "data", "processed"))
    ap.add_argument("--lidar-dir", default=r"D:\WAYMO_DATA\motion_lidar")
    a = ap.parse_args()
    for d in sorted(os.listdir(a.processed)):
        if d.startswith("MOTION_"):
            f = os.path.join(a.lidar_dir, d.split("_", 1)[1] + ".tfrecord")
            if os.path.exists(f):
                print(d, "lidar frames merged:", merge_into(os.path.join(a.processed, d), f))
            else:
                print(d, "no lidar_and_camera file at", f)
