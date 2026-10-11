"""Vehicle poses: interpolation and world <-> local helpers."""
from __future__ import annotations

import numpy as np


def _rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def _quat_to_rot(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def interpolate_pose(ts_a, T_a, ts_b, T_b, ts):
    """Pose at ``ts`` between two frame poses (slerp rotation, lerp translation).  Only valid inside [a,b]."""
    if not (ts_a <= ts <= ts_b) or ts_b == ts_a:
        raise ValueError("timestamp outside the interpolation interval")
    u = (ts - ts_a) / (ts_b - ts_a)
    qa, qb = _rot_to_quat(T_a[:3, :3]), _rot_to_quat(T_b[:3, :3])
    if np.dot(qa, qb) < 0:
        qb = -qb
    d = np.clip(np.dot(qa, qb), -1, 1)
    th = np.arccos(d)
    q = qa * (1 - u) + qb * u if th < 1e-6 else (np.sin((1 - u) * th) * qa + np.sin(u * th) * qb) / np.sin(th)
    T = np.eye(4)
    T[:3, :3] = _quat_to_rot(q)
    T[:3, 3] = T_a[:3, 3] * (1 - u) + T_b[:3, 3] * u
    return T


def yaw_of(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


def boxes_to_world(boxes, T_world_vehicle):
    """Box centres/headings (vehicle frame) -> world frame."""
    c = np.stack([boxes["cx"], boxes["cy"], boxes["cz"]], 1)
    cw = c @ T_world_vehicle[:3, :3].T + T_world_vehicle[:3, 3]
    return cw, boxes["heading"] + yaw_of(T_world_vehicle)
