"""Bake a photo-textured road for every frame of a Perception context.

1. For every frame a robust local ground plane (+-0.2 m relief from the lowest LiDAR returns) is evaluated on a
   world-aligned grid of 0.35 m cells around the ego car.
2. Each cell centre is projected into the five camera images (frame vehicle pose; camera lens-fold region excluded)
   and the real pixel colour is sampled; cells covered by labelled vehicles / pedestrians are skipped.
3. Observations are accumulated into one world-space road map for the whole sequence (nearer, sharper observations
   win), so cells that no camera sees in one frame - under and behind the car - are still textured.
4. Each frame's texture is a crop of that road map around the ego car.

    python bake_ground.py --root D:/WAYMO_DATA/raw --context NAME --split validation
Output: <processed>/<context>/ground_tex.npz   rgba uint8 [F, N, N, 4], centres [F, 2] (world-relative x, y), n, cell
Note: the road map uses observations from the whole recording (static world memory), including frames after the one
displayed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(PROJ, "datasets")]
from decode_camera import load_camera_calibration  # noqa: E402
from calibration import project_to_camera, inv_rigid, transform  # noqa: E402

N, C = 384, 0.35
CAM_PRIORITY = (1, 2, 3, 4, 5)
MAX_DEPTH = 30.0


def _nsmooth(F, M, sig):
    num = cv2.GaussianBlur(np.where(M, F, 0.0).astype(np.float32), (0, 0), sig)
    den = cv2.GaussianBlur(M.astype(np.float32), (0, 0), sig)
    return num / np.maximum(den, 1e-6), den


def _fill(F, M, sigmas=(2.0, 6.0, 16.0, 40.0)):
    out, got = F.copy(), M.copy()
    for s in sigmas:
        if got.all():
            break
        v, d = _nsmooth(F, M, s)
        take = (~got) & (d > 1e-3)
        out[take] = v[take]
        got |= take
    if not got.all():
        out[~got] = float(F[M].mean()) if M.any() else 0.0
    return out


def local_plane(P, ego, centre):
    """Local ground height on the N x N grid at `centre` (plane +- 0.2 m relief) and the measured-ground mask."""
    cx, cy = centre
    ix = np.floor((P[:, 0] - cx) / C + N / 2).astype(np.int64)
    iy = np.floor((P[:, 1] - cy) / C + N / 2).astype(np.int64)
    ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
    ix, iy, z = ix[ok], iy[ok], P[ok, 2]
    k = iy * N + ix
    zmin = np.full(N * N, np.inf, np.float32)
    np.minimum.at(zmin, k, z)
    sel = z <= zmin[k] + 0.22
    k, z = k[sel], z[sel]
    cnt = np.bincount(k, minlength=N * N).astype(np.float32)
    H = (np.bincount(k, weights=z, minlength=N * N) / np.maximum(cnt, 1)).reshape(N, N).astype(np.float32)
    valid = (cnt > 0).reshape(N, N) & (H > ego[2] - 2.0) & (H < ego[2] + 1.2)
    for _ in range(2):
        Hs, den = _nsmooth(H, valid, 2.0)
        valid &= (np.abs(H - Hs) < 0.30) | (den < 0.05)
    Hs, _ = _nsmooth(H, valid, 1.2)
    Hf = _fill(Hs, valid)
    off = (np.arange(N) + 0.5 - N / 2) * C
    XX, YY = np.meshgrid(off + cx - ego[0], off + cy - ego[1])        # offsets from the ego car
    near = valid & (np.hypot(XX, YY) < 30.0)
    coef = np.array([0.0, 0.0, float(ego[2])])
    if near.sum() > 50:
        A_ = np.stack([XX[near], YY[near], np.ones(near.sum())], 1)
        z_ = Hf[near]
        w_ = np.ones(len(z_))
        for _ in range(4):
            coef, *_ = np.linalg.lstsq(A_ * w_[:, None], z_ * w_, rcond=None)
            r_ = np.abs(A_ @ coef - z_)
            w_ = 1.0 / np.maximum(r_ / (np.median(r_) + 0.02), 1.0)
    plane = (coef[0] * XX + coef[1] * YY + coef[2]).astype(np.float32)
    H_out = plane + np.clip(Hf - plane, -0.20, 0.20)
    meas = cv2.dilate(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return H_out, meas


def vehicle_mask(boxes, centre):
    m = np.zeros((N, N), np.uint8)
    cx, cy = centre
    for r in boxes:
        if int(r[7]) not in (1, 2, 4):
            continue
        x, y, l, w, h = float(r[0]), float(r[1]), float(r[3]) + 0.8, float(r[4]) + 0.8, float(r[6])
        c, s = np.cos(h), np.sin(h)
        corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
        pts = np.stack([x + c * corners[:, 0] - s * corners[:, 1], y + s * corners[:, 0] + c * corners[:, 1]], 1)
        px = ((pts - np.array([cx, cy])) / C + N / 2).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [px], 1)
    return m > 0


def observe_frame(i, meta, cams, frames_dir, cam_dir, centre, hist=6):
    """-> rgb [N*N,3] float32, depth [N*N] (inf where unseen)."""
    pts = []
    for j in range(max(0, i - hist), i + 1):
        with np.load(os.path.join(frames_dir, f"f{j:04d}.npz")) as d:
            pts.append(d["pts"])
    P = np.concatenate(pts)
    T_wv = np.array(meta["poses_world_vehicle"][i], dtype=np.float64)
    ego = T_wv[:3, 3]
    with np.load(os.path.join(frames_dir, f"f{i:04d}.npz")) as d:
        boxes = d["boxes"]
    H, meas = local_plane(P, ego, centre)
    cx, cy = centre
    gx = (np.arange(N) + 0.5 - N / 2) * C + cx
    gy = (np.arange(N) + 0.5 - N / 2) * C + cy
    X, Y = np.meshgrid(gx, gy)
    cells = np.stack([X.ravel(), Y.ravel(), H.ravel()], 1)
    skip = vehicle_mask(boxes, centre).ravel()
    meas = meas.ravel()
    rgb = np.zeros((N * N, 3), np.float32)
    best = np.full(N * N, np.inf)
    files = meta["cam_files"][i]
    pv = transform(inv_rigid(T_wv), cells)
    for cid in CAM_PRIORITY:
        fn = files.get(str(cid)) or files.get(cid)
        if not fn or cid not in cams:
            continue
        im = cv2.imread(os.path.join(cam_dir, fn), cv2.IMREAD_REDUCED_COLOR_2)
        if im is None:
            continue
        cal = cams[cid]
        u, v, d, inside = project_to_camera(pv, cal["T_vehicle_camera"], cal["intrinsic"], cal["width"], cal["height"])
        use = inside & meas & (d > 2.0) & (d < MAX_DEPTH) & (~skip) & (d < best)
        if not use.any():
            continue
        H_, W_ = im.shape[:2]
        mapx = (u * W_ / cal["width"]).astype(np.float32).reshape(N, N)
        mapy = (v * H_ / cal["height"]).astype(np.float32).reshape(N, N)
        samp = cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        col = samp.reshape(-1, 3)[:, ::-1].astype(np.float32)
        rgb[use] = col[use]
        best[use] = d[use]
    return rgb, best


def bake(root, context, split, processed, progress=True):
    pdir = os.path.join(processed, context)
    meta = json.load(open(os.path.join(pdir, "meta.json")))
    cams = load_camera_calibration(root, context, split)
    F = meta["frames"]
    egos = np.array([np.array(m)[:2, 3] for m in meta["poses_world_vehicle"]])
    x0, y0 = egos[:, 0].min() - N * C / 2 - 2.0, egos[:, 1].min() - N * C / 2 - 2.0
    W = int((egos[:, 0].max() - x0 + N * C / 2 + 2.0) / C) + 2
    Hh = int((egos[:, 1].max() - y0 + N * C / 2 + 2.0) / C) + 2
    road = np.zeros((Hh, W, 3), np.float32)
    rdepth = np.full((Hh, W), np.inf, np.float32)
    centres = np.zeros((F, 2), np.float32)
    origins = np.zeros((F, 2), np.int64)
    t0 = time.time()
    for i in range(F):
        ox = int(round((egos[i, 0] - N * C / 2 - x0) / C))
        oy = int(round((egos[i, 1] - N * C / 2 - y0) / C))
        centre = (x0 + (ox + N / 2) * C, y0 + (oy + N / 2) * C)         # world-aligned grid => exact paste
        origins[i] = (ox, oy)
        centres[i] = centre
        rgb, dep = observe_frame(i, meta, cams, os.path.join(pdir, "frames"), os.path.join(pdir, "cam"), centre)
        rgb, dep = rgb.reshape(N, N, 3), dep.reshape(N, N)
        sl = (slice(oy, oy + N), slice(ox, ox + N))
        better = dep < rdepth[sl]
        road[sl][better] = rgb[better]
        rdepth[sl][better] = dep[better]
        if progress and i % 20 == 0:
            print(f"  observe {i}/{F}  {time.time() - t0:.0f}s", flush=True)
    seen = np.isfinite(rdepth)
    seen_u8 = cv2.morphologyEx(seen.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    for c in range(3):                                  # close 1-cell pin-holes only; unseen cells stay transparent
        road[..., c] = np.where(seen, road[..., c], _fill(road[..., c], seen, (1.5, 3.0)))
    alpha = (cv2.GaussianBlur(seen_u8.astype(np.float32), (0, 0), 1.0) * 255).astype(np.uint8)
    tex = np.zeros((F, N, N, 4), np.uint8)
    for i in range(F):
        ox, oy = origins[i]
        sl = (slice(oy, oy + N), slice(ox, ox + N))
        tex[i, ..., :3] = np.clip(road[sl], 0, 255).astype(np.uint8)
        tex[i, ..., 3] = alpha[sl]
    np.savez_compressed(os.path.join(pdir, "ground_tex.npz"), rgba=tex, centres=centres, n=N, cell=C)
    meta["ground_tex"] = True
    with open(os.path.join(pdir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return tex, road, seen, (x0, y0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--processed", default=os.path.join(PROJ, "data", "processed"))
    ap.add_argument("--preview")
    a = ap.parse_args()
    tex, road, seen, _ = bake(a.root, a.context, a.split, a.processed)
    print("baked", tex.shape, "road map %dx%d, seen %.0f%%" % (road.shape[1], road.shape[0], 100 * seen.mean()))
    if a.preview:
        v = np.where(seen[..., None], np.clip(road, 0, 255), 40).astype(np.uint8)[::-1]
        cv2.imwrite(a.preview, cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
