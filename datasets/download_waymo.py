"""Download ONE Waymo Perception v2.0.1 context (only the components we need).

Waymo data is gated: you must (1) accept the license at https://waymo.com/open/terms with the Google account you
use for gcloud, and (2) have valid gcloud credentials (``gcloud auth login``).  Nothing is downloaded without
``--yes`` after the size estimate is shown.

    python download_waymo.py --list --split validation --limit 10
    python download_waymo.py --context <context_name> --split validation --yes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from waymo_adapter import BUCKET, COMPONENTS, REQUIRED  # noqa: E402

OPTIONAL = ("camera_box", "lidar_segmentation", "camera_to_lidar_box_association", "lidar_pose",
            "lidar_camera_projection")
LICENSE_NOTE = (
    "Waymo Open Dataset is licensed for NON-COMMERCIAL use only. By downloading you confirm you have accepted the\n"
    "license at https://waymo.com/open/terms/ with the Google account used by gcloud."
)


def gsutil() -> str:
    exe = shutil.which("gsutil") or shutil.which("gsutil.cmd")
    if not exe:
        sys.exit("gsutil not found - install the Google Cloud SDK")
    return exe


def run(args):
    return subprocess.run([gsutil()] + args, capture_output=True, text=True)


def check_auth() -> None:
    r = run(["ls", BUCKET + "/"])
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip()
        sys.exit("Cannot read %s:\n%s\n\nRun `gcloud auth login` and make sure the Waymo license is accepted for "
                 "that account." % (BUCKET, msg[:600]))


def list_versions() -> None:
    print("Bucket:", BUCKET)
    r = run(["ls", BUCKET + "/"])
    print(r.stdout)


def list_contexts(split: str, limit: int):
    r = run(["ls", f"{BUCKET}/{split}/vehicle_pose/"])
    if r.returncode != 0:
        sys.exit(r.stderr)
    names = [os.path.splitext(os.path.basename(x.strip()))[0] for x in r.stdout.splitlines() if x.strip()]
    return names[:limit] if limit else names


def remote_size(split, comp, ctx):
    r = run(["du", f"{BUCKET}/{split}/{comp}/{ctx}.parquet"])
    try:
        return int(r.stdout.split()[0])
    except Exception:
        return None


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["training", "validation", "testing"])
    ap.add_argument("--context")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "data", "raw"))
    ap.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    a = ap.parse_args()

    print(LICENSE_NOTE, "\n")
    check_auth()
    if a.list or not a.context:
        list_versions()
        print("Perception v2 components:")
        for k, v in COMPONENTS.items():
            print(f"  {k:36s} {v}")
        print(f"\nContexts in {a.split}:")
        for n in list_contexts(a.split, a.limit):
            print("  ", n)
        return

    comps = list(REQUIRED) + list(OPTIONAL)
    total, plan, missing = 0, [], []
    for c in comps:
        sz = remote_size(a.split, c, a.context)
        if sz is None:
            missing.append(c)
        else:
            plan.append((c, sz))
            total += sz
    print("Will download:")
    for c, sz in plan:
        print(f"  {c:36s} {sz/1e6:9.1f} MB")
    print(f"  total {total/1e6:.1f} MB")
    if missing:
        print("MISSING components for this context:", missing)
    gone = [m for m in missing if m in REQUIRED]
    if gone:
        sys.exit(f"required components absent: {gone}")
    if not a.yes and input("Proceed? [y/N] ").lower() != "y":
        return

    manifest = {"context": a.context, "split": a.split, "bucket": BUCKET, "files": {}, "missing": missing}
    for c, _ in plan:
        dst_dir = os.path.join(a.out, a.split, c)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, a.context + ".parquet")
        if os.path.exists(dst) and os.path.getsize(dst) == dict(plan)[c]:      # already downloaded
            print(f"  {c:36s} present, skipped")
            manifest["files"][c] = {"bytes": os.path.getsize(dst), "md5": md5_file(dst), "md5_matches_remote": True}
            continue
        r = run(["cp", f"{BUCKET}/{a.split}/{c}/{a.context}.parquet", dst])
        if r.returncode != 0:
            sys.exit(f"download failed for {c}: {r.stderr[:400]}")
        # integrity: compare gsutil's md5 listing with the local file
        ls = run(["ls", "-L", f"{BUCKET}/{a.split}/{c}/{a.context}.parquet"]).stdout
        md5_remote = None
        for line in ls.splitlines():
            if "Hash (md5)" in line:
                import base64
                md5_remote = base64.b64decode(line.split(":", 1)[1].strip()).hex()
        md5_local = md5_file(dst)
        ok = md5_remote is None or md5_remote == md5_local
        manifest["files"][c] = {"bytes": os.path.getsize(dst), "md5": md5_local, "md5_matches_remote": ok}
        print(f"  {c:36s} ok={ok}")
        if not ok:
            sys.exit(f"checksum mismatch for {c}")
    with open(os.path.join(a.out, f"{a.context}.manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
