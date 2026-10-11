"""LiDAR -> camera colouring with temporally correct poses and depth-aware visibility."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import project_to_camera, inv_rigid, transform  # noqa: E402
from motion_compensation import project_rolling  # noqa: E402


def colorize(pts_world, cams, cell=4, depth_tol=0.6):
    """pts_world: [N,3] world coords.
    cams: list of dict(image HxWx3 uint8, T_world_vehicle (vehicle pose AT THE IMAGE TIME), T_vehicle_camera,
          intrinsic dict, width, height).
    Each point takes its colour from the camera in which it is visible (not occluded by a nearer return) and
    closest to the image centre.  Returns (rgb uint8 [N,3], cam_index int8 [N] (-1 = no camera), score).
    """
    N = len(pts_world)
    rgb = np.zeros((N, 3), np.uint8)
    best = np.full(N, np.inf)
    which = np.full(N, -1, np.int8)
    for ci, c in enumerate(cams):
        if c.get("rs") is not None:      # rolling shutter: pose extrapolated to each pixel exposure time
            u, v, d = project_rolling(pts_world, c["rs"], c["T_vehicle_camera"], c["intrinsic"], c["width"],
                                      c["height"], c.get("direction", 5), ang_in_world=False, time_mode="none")
            inside = (d > 0.1) & (u >= 0) & (u < c["width"] - 1) & (v >= 0) & (v < c["height"] - 1)
        else:
            T_v_w = inv_rigid(c["T_world_vehicle"])
            pv = transform(T_v_w, pts_world)
            u, v, d, inside = project_to_camera(pv, c["T_vehicle_camera"], c["intrinsic"], c["width"], c["height"])
        idx = np.nonzero(inside)[0]
        if len(idx) == 0:
            continue
        gx = (u[idx] / cell).astype(np.int32)
        gy = (v[idx] / cell).astype(np.int32)
        W = c["width"] // cell + 2
        key = gy * W + gx
        zmin = np.full((c["height"] // cell + 2) * W, np.inf)
        np.minimum.at(zmin, key, d[idx])
        vis = d[idx] <= zmin[key] + depth_tol
        sel = idx[vis]
        cx, cy = c["width"] / 2.0, c["height"] / 2.0
        score = np.hypot((u[sel] - cx) / c["width"], (v[sel] - cy) / c["height"])
        better = score < best[sel]
        sel2 = sel[better]
        px = np.clip(u[sel2].astype(np.int32), 0, c["width"] - 1)
        py = np.clip(v[sel2].astype(np.int32), 0, c["height"] - 1)
        rgb[sel2] = c["image"][py, px, :3]
        best[sel2] = score[better]
        which[sel2] = ci
    return rgb, which, best
