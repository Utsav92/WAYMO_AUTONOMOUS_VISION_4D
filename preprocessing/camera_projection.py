"""Calibration-debug view: project LiDAR points onto the camera image, coloured by depth.
    python camera_projection.py --root ../data/raw --context NAME --frame 0 --camera 1 --out dbg.png
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "datasets"))
from calibration import project_to_camera  # noqa: E402


def overlay(image_rgb, pts_vehicle, cam, max_depth=60.0):
    u, v, d, ok = project_to_camera(pts_vehicle, cam["T_vehicle_camera"], cam["intrinsic"], cam["width"], cam["height"])
    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    t = np.clip(d[ok] / max_depth, 0, 1)
    cols = cv2.applyColorMap((255 * (1 - t)).astype(np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0]
    for x, y, c in zip(u[ok].astype(int), v[ok].astype(int), cols):
        cv2.circle(out, (x, y), 2, tuple(int(k) for k in c), -1)
    return out, int(ok.sum())


if __name__ == "__main__":
    from decode_camera import load_camera_calibration, frame_images, decode_jpeg
    from decode_lidar import decode_frames
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--split")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--out", default="calibration_debug.png")
    a = ap.parse_args()
    cams = load_camera_calibration(a.root, a.context, a.split)
    imgs = frame_images(a.root, a.context, a.split)
    fr = list(decode_frames(a.root, a.context, a.split, a.frame + 1))[a.frame]
    jpg = imgs[fr["ts"]][a.camera][0]
    o, n = overlay(decode_jpeg(jpg), fr["xyz_vehicle"], cams[a.camera])
    cv2.imwrite(a.out, o)
    print("projected points in image:", n, "->", a.out)
