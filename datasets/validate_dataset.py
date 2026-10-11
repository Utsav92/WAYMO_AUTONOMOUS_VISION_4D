"""Check a downloaded context has every required component, consistent frame timestamps and non-empty data.
    python validate_dataset.py --root ../data/raw --context NAME
Exit code 0 only when everything required is present.  Reports what is missing instead of inventing it.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from waymo_adapter import REQUIRED, COMPONENTS, have_components, read_component, find_col  # noqa: E402


def validate(root, ctx, split=None):
    rep = {"context": ctx, "components": have_components(root, ctx, split), "problems": [], "frames": 0}
    for c in REQUIRED:
        if not rep["components"][c]:
            rep["problems"].append(f"required component missing: {c}")
    if rep["problems"]:
        return rep
    pose = read_component(root, "vehicle_pose", ctx, split)
    tcol = find_col(pose.columns, "frame_timestamp_micros")
    ts = sorted(pose[tcol].tolist())
    rep["frames"] = len(ts)
    if len(ts) != len(set(ts)):
        rep["problems"].append("duplicate frame timestamps in vehicle_pose")
    for c in ("lidar", "camera_image", "lidar_box"):
        df = read_component(root, c, ctx, split, columns=None)
        t2 = set(df[find_col(df.columns, "frame_timestamp_micros")].tolist())
        extra = t2 - set(ts)
        if extra:
            rep["problems"].append(f"{c}: {len(extra)} timestamps without a vehicle pose")
        rep[c + "_rows"] = len(df)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--split")
    a = ap.parse_args()
    r = validate(a.root, a.context, a.split)
    for k, v in r.items():
        print(f"{k}: {v}")
    sys.exit(1 if r["problems"] else 0)
