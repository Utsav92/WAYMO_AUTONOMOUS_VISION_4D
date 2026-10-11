"""Decode one Waymo context into TouchDesigner-ready per-frame files.

    python export_frames.py --root ../data/raw --context NAME [--split validation] [--max-frames 60] [--out ../data/processed]

Output  <out>/<context>/meta.json, frames/f0000.npz (pts world-relative, inten, rgb, sensor, boxes), cam/f0000_<CAM>.jpg
World coordinates = Waymo world frame minus the first frame's vehicle position (float precision), z up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "datasets"))
from decode_lidar import decode_frames, to_world, voxel_downsample  # noqa: E402
from decode_camera import load_camera_calibration, frame_images, decode_jpeg  # noqa: E402
from object_labels import load_boxes, speed_from_track  # noqa: E402
from ego_motion import boxes_to_world  # noqa: E402
from rgb_fusion import colorize  # noqa: E402

MAX_POINTS = 512 * 256


def export(root, context, out, split=None, max_frames=None, max_range=75.0, voxel=0.08, stride=1,
           with_rgb=True, seed=0, rolling_shutter=True, motion_comp=True, sem_model=None):
    rng = np.random.default_rng(seed)
    cams = load_camera_calibration(root, context, split)
    images = frame_images(root, context, split) if with_rgb else {}
    boxes, _ids = load_boxes(root, context, split)
    odir = os.path.join(out, context)
    os.makedirs(os.path.join(odir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(odir, "cam"), exist_ok=True)
    origin, ts_list, poses, cam_files = None, [], [], []
    prev_world = {}                       # track -> (ts, world xyz) for speed
    t0 = time.time()
    for fi, fr in enumerate(decode_frames(root, context, split, max_frames, frame_stride=stride, motion_comp=motion_comp,
                              sem_model=sem_model)):
        Twv = fr["T_world_vehicle"]
        if origin is None:
            origin = Twv[:3, 3].copy()
        xyz = to_world(fr) - origin
        rng_m = fr["attrs"][:, 2]
        keep = rng_m <= max_range
        xyz, att, sen, sem = xyz[keep], fr["attrs"][keep], fr["sensor"][keep], fr["sem"][keep]
        xyz, att, sen, sem = voxel_downsample(xyz, voxel, att, sen, sem)
        if len(xyz) > MAX_POINTS:
            sel = rng.choice(len(xyz), MAX_POINTS, replace=False)
            xyz, att, sen, sem = xyz[sel], att[sel], sen[sel], sem[sel]
        rgb = np.zeros((len(xyz), 3), np.uint8)
        cam_idx = np.full(len(xyz), -1, np.int8)
        files = {}
        if with_rgb and fr["ts"] in images:
            cl = []
            for cid, (jpg, Timg, rs) in images[fr["ts"]].items():
                if cid not in cams:
                    continue
                cl.append(dict(image=decode_jpeg(jpg), T_world_vehicle=(Timg if Timg is not None else Twv),
                               rs=rs if rolling_shutter else None, **cams[cid]))
                fn = f"f{fi:04d}_{cid}.jpg"
                with open(os.path.join(odir, "cam", fn), "wb") as f:
                    f.write(bytes(jpg))
                files[cid] = fn
            rgb, cam_idx, _ = colorize(xyz + origin, cl)
        b = boxes[boxes["ts"] == fr["ts"]]
        cw, hw = boxes_to_world(b, Twv)
        rows = np.zeros((len(b), 10), np.float32)
        for i, bi in enumerate(b):
            p = cw[i] - origin
            sp = -1.0
            tr = int(bi["track"])
            if tr in prev_world:
                pts, pp = prev_world[tr]
                dt = (fr["ts"] - pts) * 1e-6
                if 0 < dt <= 0.5:
                    sp = float(np.linalg.norm(cw[i] - pp) / dt)
            prev_world[tr] = (fr["ts"], cw[i].copy())
            if np.isfinite(bi["spd"]):        # prefer the dataset's own speed metadata over finite differences
                sp = float(bi["spd"])
            rows[i] = [p[0], p[1], p[2], bi["l"], bi["w"], bi["h"], hw[i], bi["type"], tr, sp]
        np.savez_compressed(os.path.join(odir, "frames", f"f{fi:04d}.npz"),
                            pts=xyz.astype(np.float32), inten=att[:, 0].astype(np.float32),
                            rng=att[:, 2].astype(np.float32), rgb=rgb, cam=cam_idx, sensor=sen, boxes=rows,
                            sem=sem.astype(np.uint8), sem_gt=np.uint8(1 if fr["sem_is_gt"] else 0))
        ts_list.append(fr["ts"])
        T = Twv.copy()
        T[:3, 3] -= origin
        poses.append(T.tolist())
        cam_files.append(files)
    meta = {"context": context, "split": split, "frames": len(ts_list), "timestamps_micros": ts_list,
            "origin_world": origin.tolist() if origin is not None else None, "poses_world_vehicle": poses,
            "cam_files": cam_files, "max_points": MAX_POINTS, "rolling_shutter": rolling_shutter,
            "top_pixel_pose": motion_comp, "synthetic": context.startswith("SYNTHETIC"),
            "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(odir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--split")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "data", "processed"))
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-rgb", action="store_true")
    ap.add_argument("--semantic", help="trained segmentation checkpoint (models/seg.pt)")
    a = ap.parse_args()
    m = export(a.root, a.context, a.out, a.split, a.max_frames, stride=a.stride, with_rgb=not a.no_rgb,
               sem_model=a.semantic)
    print("exported", m["frames"], "frames in", m["elapsed_s"], "s")
