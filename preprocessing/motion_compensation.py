"""TOP-LiDAR per-pixel pose compensation and camera rolling-shutter projection.

* TOP lidar spins ~0.1 s per sweep while the car moves (~1.7 m at 17 m/s); `lidar_pose` stores the vehicle pose
  (roll, pitch, yaw, x, y, z) for every range-image pixel.  Points are placed with their own pixel pose instead of
  the single frame pose.
* Cameras are rolling shutter: every image row/column is exposed at a slightly different time.  The camera pose and
  velocity at `pose_timestamp` are extrapolated to the exposure time of the pixel a point lands on (fixed-point
  iteration).  Conventions that Waymo does not state unambiguously (velocity frame, exposure-time mapping) are
  selected by the validation script `validate_projection.py`, which compares with `lidar_camera_projection`.
"""
from __future__ import annotations

import numpy as np

from calibration import inv_rigid, transform


def rpy_to_matrix(roll, pitch, yaw):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), vectorised over arrays -> (...,3,3)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.empty(np.shape(roll) + (3, 3), np.float64)
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def compensate_top(xyz_vehicle, rows, cols, pose_image, T_world_vehicle_frame):
    """xyz_vehicle: [N,3] TOP points in the vehicle frame of the frame timestamp (from the range image + extrinsic).
    pose_image: (H,W,6) roll,pitch,yaw,x,y,z.  Returns points in the *frame* vehicle coordinates, i.e. positions
    that, transformed with T_world_vehicle_frame, land at the motion-compensated world position."""
    p = pose_image[rows, cols]                                         # (N,6)
    R = rpy_to_matrix(p[:, 0], p[:, 1], p[:, 2])
    world = np.einsum("nij,nj->ni", R, xyz_vehicle.astype(np.float64)) + p[:, 3:6]
    bad = ~np.isfinite(world).all(axis=1)                  # pixels without a valid pose: use the frame pose
    if bad.any():
        world[bad] = transform(T_world_vehicle_frame, xyz_vehicle[bad].astype(np.float64))
    return transform(inv_rigid(T_world_vehicle_frame), world).astype(np.float32), world


# ------------------------------------------------------------------------------------------------ rolling shutter
def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def rodrigues(w, dt):
    th = np.linalg.norm(w) * dt
    if th < 1e-12:
        return np.eye(3)
    k = _skew(w / np.linalg.norm(w))
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)


def pose_at(T_world_vehicle, lin_vel, ang_vel, dt, ang_in_world=True):
    """Vehicle pose extrapolated by dt seconds (constant velocity).  Velocities in the world frame."""
    T = np.eye(4)
    dR = rodrigues(np.asarray(ang_vel, float), dt)
    R0 = T_world_vehicle[:3, :3]
    T[:3, :3] = dR @ R0 if ang_in_world else R0 @ dR
    T[:3, 3] = T_world_vehicle[:3, 3] + np.asarray(lin_vel, float) * dt
    return T


def exposure_offset(u, v, width, height, meta, direction, mode="center"):
    """Seconds, relative to `pose_timestamp`, at which pixel (u, v) was exposed.
    direction: Waymo RollingShutterReadOutDirection  (1 TOP_TO_BOTTOM, 2 LEFT_TO_RIGHT, 3 BOTTOM_TO_TOP,
    4 RIGHT_TO_LEFT, 5 GLOBAL_SHUTTER).  Readout runs over [camera_trigger_time, camera_readout_done_time]."""
    t0, t1 = meta["camera_trigger_time"], meta["camera_readout_done_time"]
    if direction == 5:
        frac = np.full_like(u, 0.5, dtype=float)
    elif direction == 1:
        frac = v / height
    elif direction == 3:
        frac = 1.0 - v / height
    elif direction == 2:
        frac = u / width
    else:
        frac = 1.0 - u / width
    frac = np.clip(frac, 0.0, 1.0)                       # only the physical readout window exists
    t = t0 + frac * (t1 - t0)
    if mode == "center":
        t = t + 0.5 * meta.get("shutter", 0.0)
    return t - meta["pose_timestamp"]


def project_rolling(pts_world, meta, T_vehicle_camera, intr, width, height, direction, ang_in_world=True,
                    iters=3, time_mode="center"):
    """Project world points with a per-pixel exposure-time camera pose.  meta: pose (4x4 world_from_vehicle at
    pose_timestamp), lin_vel[3], ang_vel[3], pose_timestamp, camera_trigger_time, camera_readout_done_time, shutter.
    Returns (u, v, depth)."""
    from calibration import project_to_camera
    N = len(pts_world)
    dt = np.zeros(N)
    u = v = d = None
    T_v_cam = T_vehicle_camera
    for _ in range(iters):
        out_u, out_v, out_d = np.zeros(N), np.zeros(N), np.zeros(N)
        # points grouped by quantised dt so we do few matrix inversions
        q = np.round(np.nan_to_num(dt) * 2e3).astype(np.int64)
        for key in np.unique(q):
            sel = q == key
            T = pose_at(meta["pose"], meta["lin_vel"], meta["ang_vel"], key / 2e3, ang_in_world)
            pv = transform(inv_rigid(T), pts_world[sel])
            uu, vv, dd, _ = project_to_camera(pv, T_v_cam, intr, width, height)
            out_u[sel], out_v[sel], out_d[sel] = uu, vv, dd
        u, v, d = out_u, out_v, out_d
        dt = np.clip(np.nan_to_num(exposure_offset(u, v, width, height, meta, direction, time_mode)), -0.2, 0.2)
    return u, v, d
