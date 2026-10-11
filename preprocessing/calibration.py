"""Geometry core: range image -> XYZ, frames, camera projection.  Pure numpy (no TensorFlow).

Conventions (Waymo Open Dataset):
  * range image (H, W, C): C0=range[m], C1=intensity, C2=elongation, C3=in-no-label-zone; range<=0 is invalid.
  * LiDAR extrinsic: 4x4 sensor->vehicle.   vehicle_pose: 4x4 vehicle->world.
  * camera extrinsic: 4x4 camera->vehicle, camera frame x forward, y left, z up.
  * camera projection: u = f_u * (-y/x) + c_u,  v = f_v * (-z/x) + c_v (after k1,k2,p1,p2,k3 distortion).
  * P_world = T_world_vehicle @ T_vehicle_sensor @ P_sensor
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------------------------------------ LiDAR
def beam_inclinations(height: int, bi_min: float = None, bi_max: float = None, values=None) -> np.ndarray:
    """Per-row inclination [rad].  Row 0 is the TOP of the image => highest inclination.  Explicit ``values``
    (TOP lidar) are re-ordered so that inclination decreases with the row index, whatever order they are stored in."""
    if values is not None and len(values) == height:
        v = np.asarray(values, dtype=np.float64)
        return np.sort(v)[::-1].copy()
    if bi_min is None or bi_max is None:
        raise ValueError("need beam_inclination min/max or per-beam values")
    ratio = (np.arange(height, 0, -1, dtype=np.float64) - 0.5) / height
    return ratio * (bi_max - bi_min) + bi_min


def range_image_to_points(range_image: np.ndarray, extrinsic: np.ndarray, inclination: np.ndarray):
    """Returns (xyz_vehicle[N,3], attrs[N,3] = (intensity, elongation, range), valid_mask[H,W], pix_rc[N,2])."""
    H, W = range_image.shape[:2]
    r = range_image[..., 0].astype(np.float64)
    az_corr = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    ratio = (np.arange(W, 0, -1, dtype=np.float64) - 0.5) / W
    az = (ratio * 2.0 - 1.0) * np.pi - az_corr
    AZ, INC = np.meshgrid(az, inclination)
    ci = np.cos(INC)
    xyz = np.stack([r * ci * np.cos(AZ), r * ci * np.sin(AZ), r * np.sin(INC)], axis=-1)
    valid = r > 0.0
    pts = xyz[valid]
    pts_h = pts @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    rr, cc = np.nonzero(valid)
    attrs = np.stack([range_image[..., 1][valid], range_image[..., 2][valid], r[valid]], axis=-1)
    return pts_h.astype(np.float32), attrs.astype(np.float32), valid, np.stack([rr, cc], axis=-1)


def points_to_range_image(xyz_sensor: np.ndarray, H: int, W: int, inclination: np.ndarray):
    """Inverse of range_image_to_points for points in the SENSOR frame ignoring extrinsic az correction.
    Used by unit tests (round trip).  Returns (range image[H,W], row, col)."""
    r = np.linalg.norm(xyz_sensor, axis=1)
    az = np.arctan2(xyz_sensor[:, 1], xyz_sensor[:, 0])
    inc = np.arcsin(xyz_sensor[:, 2] / np.maximum(r, 1e-9))
    ratio = (az / np.pi + 1.0) / 2.0
    col = np.clip(np.round(W - ratio * W - 0.5).astype(int), 0, W - 1)
    row = np.abs(inc[:, None] - inclination[None, :]).argmin(axis=1)
    img = np.zeros((H, W), dtype=np.float32)
    img[row, col] = r
    return img, row, col


# ------------------------------------------------------------------------------------------------ transforms
def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def inv_rigid(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def sensor_to_world(pts_sensor, T_vehicle_sensor, T_world_vehicle):
    return transform(T_world_vehicle @ T_vehicle_sensor, pts_sensor)


# ------------------------------------------------------------------------------------------------ camera
def project_to_camera(pts_vehicle: np.ndarray, T_vehicle_camera: np.ndarray, intr: dict, width: int, height: int):
    """pts in the vehicle frame (of the *camera's* timestamp).  Returns (u, v, depth, in_image_mask)."""
    pc = transform(inv_rigid(T_vehicle_camera), pts_vehicle.astype(np.float64))
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    front = x > 0.1
    xs = np.where(front, x, 1.0)
    xn, yn = -y / xs, -z / xs
    r2 = xn * xn + yn * yn
    k1, k2, p1, p2, k3 = (intr.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3"))
    rad = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    u = intr["f_u"] * xd + intr["c_u"]
    v = intr["f_v"] * yd + intr["c_v"]
    inside = front & (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
    inside &= r2 < distortion_r2_limit(intr)                 # beyond the fold of the polynomial it maps back into frame
    return u, v, x, inside


def distortion_r2_limit(intr: dict) -> float:
    """Largest normalised r^2 for which r*(1 + k1 r^2 + k2 r^4 + k3 r^6) is still increasing (valid FOV)."""
    k1, k2, k3 = intr.get("k1", 0.0), intr.get("k2", 0.0), intr.get("k3", 0.0)
    # d/dr [r (1 + k1 t + k2 t^2 + k3 t^3)] with t = r^2  ->  1 + 3 k1 t + 5 k2 t^2 + 7 k3 t^3
    roots = np.roots([7 * k3, 5 * k2, 3 * k1, 1.0]) if (k3 or k2 or k1) else np.array([])
    pos = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    return min(pos) * 0.98 if pos else np.inf


def box_corners(center, size, heading) -> np.ndarray:
    """8 corners (z-up, heading = yaw about +z). Order: bottom 0-3 ccw, top 4-7."""
    l, w, h = size
    sx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    sy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    sz = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2
    c, s = np.cos(heading), np.sin(heading)
    x = c * sx - s * sy + center[0]
    y = s * sx + c * sy + center[1]
    return np.stack([x, y, sz + center[2]], axis=1)


BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
