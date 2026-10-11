"""Timestamp-based history weights + world-space accumulation of static returns."""
from __future__ import annotations

import numpy as np


def age_fade(ts_now: int, ts_hist, decay_s: float = 2.0):
    """Fade weight per historical observation from real elapsed time (not frame count)."""
    age = (np.asarray(ts_now, dtype=np.float64) - np.asarray(ts_hist, dtype=np.float64)) * 1e-6
    w = np.exp(-np.clip(age, 0, None) / decay_s)
    w[age < 0] = 0.0
    return w, age


def static_mask(points_world, box_centers_world, box_sizes, box_headings, margin=0.3):
    """True for points NOT inside any (inflated) tracked box -> candidate static background."""
    keep = np.ones(len(points_world), bool)
    for c, s, h in zip(box_centers_world, box_sizes, box_headings):
        d = points_world - c
        ch, sh = np.cos(-h), np.sin(-h)
        lx = ch * d[:, 0] - sh * d[:, 1]
        ly = sh * d[:, 0] + ch * d[:, 1]
        inside = (np.abs(lx) <= s[0] / 2 + margin) & (np.abs(ly) <= s[1] / 2 + margin) & \
                 (np.abs(d[:, 2]) <= s[2] / 2 + margin)
        keep &= ~inside
    return keep


class RollingMap:
    """Voxel-hashed static map around the ego vehicle with a hard point budget."""

    def __init__(self, voxel=0.2, radius=60.0, max_points=400_000):
        self.voxel, self.radius, self.max_points = voxel, radius, max_points
        self.store = {}

    def add(self, pts, ego_xy):
        k = np.floor(pts / self.voxel).astype(np.int64)
        for key, p in zip(map(tuple, k), pts):
            self.store[key] = p
        self.prune(ego_xy)

    def prune(self, ego_xy):
        if not self.store:
            return
        P = np.array(list(self.store.values()))
        d = np.hypot(P[:, 0] - ego_xy[0], P[:, 1] - ego_xy[1])
        keep = d <= self.radius
        if keep.sum() > self.max_points:
            order = np.argsort(d)
            nearest = np.zeros(len(d), bool)
            nearest[order[:self.max_points]] = True
            keep &= nearest
        self.store = {tuple(np.floor(p / self.voxel).astype(np.int64)): p for p in P[keep]}

    def points(self):
        return np.array(list(self.store.values()), dtype=np.float32).reshape(-1, 3)
