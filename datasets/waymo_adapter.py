"""Thin reader for Waymo Open Dataset Perception v2.x (modular Parquet).

Column names in v2 parquet files look like ``key.segment_context_name`` and
``[LiDARBoxComponent].box.center.x``.  Because the exact naming must come from the real files, every lookup goes
through :func:`find_col`, which matches by suffix and raises a clear error listing the actual columns.  Run
``datasets/inspect_components.py`` on a real download to see the true schema.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pyarrow.parquet as pq

BUCKET = "gs://waymo_open_dataset_v_2_0_1"
SPLITS = ("training", "validation", "testing")

# component folder name -> purpose
COMPONENTS = {
    "camera_image": "camera JPEG images + per-image vehicle pose + rolling shutter params",
    "camera_calibration": "camera intrinsics / extrinsics",
    "lidar": "LiDAR range images (return 1 / return 2)",
    "lidar_calibration": "LiDAR extrinsics + beam inclinations",
    "vehicle_pose": "world_from_vehicle 4x4 per frame",
    "lidar_box": "3D boxes + tracking ids (ground truth)",
    "camera_box": "2D camera boxes",
    "camera_to_lidar_box_association": "2D<->3D correspondence",
    "lidar_segmentation": "LiDAR semantic segmentation (subset of frames)",
    "lidar_pose": "per-pixel vehicle pose for TOP lidar range images (motion compensation)",
    "lidar_camera_projection": "Waymo-provided LiDAR->camera pixel projection (validation reference)",
    "stats": "scene stats",
}
REQUIRED = ("camera_image", "camera_calibration", "lidar", "lidar_calibration", "vehicle_pose", "lidar_box")

LASER_NAMES = {1: "TOP", 2: "FRONT", 3: "SIDE_LEFT", 4: "SIDE_RIGHT", 5: "REAR"}
CAMERA_NAMES = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
BOX_TYPES = {1: "VEHICLE", 2: "PEDESTRIAN", 3: "SIGN", 4: "CYCLIST"}


class MissingComponent(RuntimeError):
    pass


def find_col(columns: Iterable[str], *suffixes: str, contains: Optional[str] = None) -> str:
    """First column ending with any of ``suffixes`` (and containing ``contains`` when given)."""
    cols = list(columns)
    for suf in suffixes:
        for c in cols:
            if c.endswith(suf) and (contains is None or contains in c):
                return c
    raise KeyError(f"no column ending with {suffixes} (contains={contains}); available: {cols}")


def component_path(root: str, component: str, context: str, split: Optional[str] = None) -> Optional[str]:
    """Locate ``<root>/[split/]<component>/<context>.parquet``."""
    cands = []
    if split:
        cands.append(os.path.join(root, split, component, context + ".parquet"))
    cands.append(os.path.join(root, component, context + ".parquet"))
    cands += glob.glob(os.path.join(root, "*", component, context + ".parquet"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def read_component(root: str, component: str, context: str, split: Optional[str] = None,
                   columns: Optional[List[str]] = None):
    p = component_path(root, component, context, split)
    if p is None:
        raise MissingComponent(f"component '{component}' not found for context {context} under {root}")
    return pq.read_table(p, columns=columns).to_pandas()


def have_components(root: str, context: str, split: Optional[str] = None) -> Dict[str, bool]:
    return {c: component_path(root, c, context, split) is not None for c in COMPONENTS}


def list_contexts(root: str) -> List[str]:
    out = set()
    for p in glob.glob(os.path.join(root, "**", "vehicle_pose", "*.parquet"), recursive=True):
        out.add(os.path.splitext(os.path.basename(p))[0])
    return sorted(out)


def as_matrix(flat) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(4, 4)


def key_cols(df) -> Dict[str, str]:
    d = {"context": find_col(df.columns, "segment_context_name")}
    for k, suf in (("ts", "frame_timestamp_micros"), ("laser", "laser_name"), ("camera", "camera_name")):
        try:
            d[k] = find_col(df.columns, suf)
        except KeyError:
            pass
    return d
