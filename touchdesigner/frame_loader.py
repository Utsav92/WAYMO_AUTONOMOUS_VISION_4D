"""Runtime state + data access for the TouchDesigner project.  Imported (not a DAT module) so shared state persists
across reloads of the cooks.  Everything that touches TD goes through `td` (op/absTime are not globals here).

Coordinates: Waymo world (x fwd, y left, z up) -> TD (x, z, -y), metres, relative to the first ego position.
"""
import json
import os
import time

import numpy as np
import td

ROOT = "/project1/WAYMO_AUTONOMOUS_VISION_4D"
TEX_W, TEX_H = 512, 256
N_SLOTS = TEX_W * TEX_H

S = {
    "dir": None, "meta": None, "ts": None, "t_rel": None, "poses": None, "boxes": None, "cache": {}, "cache_order": [],
    "cur": -1, "t": 0.0, "playing": False, "frozen": False, "direction": 1.0, "speed": 1.0, "last_wall": None,
    "selected": -1, "frame_ms": 0.0, "fps": 0.0, "dirty": True, "tex_key": None, "tex": {}, "cam_cache": {},
    "synthetic": False,
}
BOX_NAMES = {1: "VEHICLE", 2: "PEDESTRIAN", 3: "SIGN", 4: "CYCLIST"}


def load(ctx_dir):
    with open(os.path.join(ctx_dir, "meta.json")) as f:
        m = json.load(f)
    S.update(dir=ctx_dir, meta=m, ts=np.asarray(m["timestamps_micros"], dtype=np.int64), cache={}, cache_order=[],
             cam_cache={}, cur=-1, t=0.0, dirty=True, tex_key=None, synthetic=bool(m.get("synthetic")),
             kind=m.get("kind", "perception"), map=None, future=None, traj=None, t_ego=None, raster=None,
             ground_key=None, ground=None)
    S["gtex"] = None
    gp = os.path.join(ctx_dir, "ground_tex.npz")                  # photo-textured road baked from the camera images
    if os.path.exists(gp):
        with np.load(gp) as z:
            S["gtex"] = {"rgba": z["rgba"], "centres": z["centres"]}
    S["static_mem"] = None
    sm = os.path.join(ctx_dir, "static_memory.npz")
    if os.path.exists(sm):
        with np.load(sm) as z:
            S["static_mem"] = {k: z[k] for k in z.files}
    for key, fn in (("map", "map.npz"), ("future", "future.npz"), ("traj", "traj.npz")):
        p = os.path.join(ctx_dir, fn)
        if os.path.exists(p):
            with np.load(p) as z:
                S[key] = {k: z[k] for k in z.files}
    S["t_rel"] = (S["ts"] - S["ts"][0]) * 1e-6
    S["poses"] = np.asarray(m["poses_world_vehicle"], dtype=np.float64)
    S["boxes"] = [np.load(os.path.join(ctx_dir, "frames", f"f{i:04d}.npz"))["boxes"] for i in range(m["frames"])]
    return m


def n_frames():
    return 0 if S["meta"] is None else int(S["meta"]["frames"])


def duration():
    return float(S["t_rel"][-1]) if S["t_rel"] is not None and len(S["t_rel"]) else 0.0


def frame(i):
    i = int(np.clip(i, 0, n_frames() - 1))
    if i in S["cache"]:
        return S["cache"][i]
    src = i
    if S.get("kind") == "e2e":                 # one captured moment: its estimated cloud is shared by the whole timeline
        src = int(S["meta"].get("current_index", 0))
    d = np.load(os.path.join(S["dir"], "frames", f"f{src:04d}.npz"))
    fr = {k: d[k] for k in d.files}
    mem = S.get("static_mem")
    if S.get("kind") == "motion" and mem is not None and i >= int(S["meta"].get("lidar_frames", 11)):
        for k in ("pts", "inten", "rng", "sensor", "rgb", "cam", "sem"):    # real measured background, held
            if k in mem:
                fr[k] = mem[k]
    S["cache"][i] = fr
    S["cache_order"].append(i)
    while len(S["cache_order"]) > 40:
        S["cache"].pop(S["cache_order"].pop(0), None)
    return fr


def to_td(p):
    p = np.asarray(p, dtype=np.float32)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def frame_at_time(t):
    return int(np.clip(np.searchsorted(S["t_rel"], t + 1e-9, side="right") - 1, 0, n_frames() - 1))


def ego_t():
    """Time used for the ego car / chase cameras.  Equals data time unless a scene decouples it (Time Freeze:
    traffic frozen at one frame while the ego keeps driving through it)."""
    return S["t"] if S.get("t_ego") is None else S["t_ego"]


def ego_pose_at(t):
    """Interpolated vehicle pose (4x4, Waymo world-relative) at data time t seconds - used by the cameras only."""
    from ego_motion import interpolate_pose
    i = frame_at_time(t)
    if i >= n_frames() - 1:
        return S["poses"][i]
    return interpolate_pose(S["t_rel"][i], S["poses"][i], S["t_rel"][i + 1], S["poses"][i + 1], float(np.clip(t, S["t_rel"][i], S["t_rel"][i + 1])))


# ----------------------------------------------------------------------------------------------- textures
def _empty():
    return (np.zeros((TEX_H, TEX_W, 4), np.float32), np.zeros((TEX_H, TEX_W, 4), np.float32),
            np.zeros((TEX_H, TEX_W, 4), np.float32))


def _fill(pos, attr, rgb, start, fr, age):
    n = min(len(fr["pts"]), N_SLOTS - start)
    if n <= 0:
        return 0
    flat = lambda a: a.reshape(-1, 4)
    p = to_td(fr["pts"][:n])
    flat(pos)[start:start + n, :3] = p
    flat(pos)[start:start + n, 3] = fr["rng"][:n]
    flat(attr)[start:start + n, 0] = np.clip(fr["inten"][:n], 0, 1)
    flat(attr)[start:start + n, 2] = fr["sensor"][:n]
    if "sem" in fr:
        flat(attr)[start:start + n, 1] = fr["sem"][:n]
    flat(attr)[start:start + n, 3] = age
    flat(rgb)[start:start + n, :3] = fr["rgb"][:n] / 255.0
    flat(rgb)[start:start + n, 3] = (fr["cam"][:n] >= 0)
    return n


def build_textures(i, hist_frames=10):
    """Current frame -> set 'cur'; K previous frames (timestamp aged) -> set 'hist'."""
    key = (i, int(hist_frames))
    if S["tex_key"] == key:
        return
    t0 = time.perf_counter()
    cur = _empty()
    _fill(*cur, 0, frame(i), 0.0)
    hist = _empty()
    K = int(hist_frames)
    if K > 0 and i > 0:
        per = N_SLOTS // K
        pos = 0
        for j in range(i - 1, max(i - K, 0) - 1, -1):
            fr = frame(j)
            n = len(fr["pts"])
            if n == 0:
                continue
            sel = np.linspace(0, n - 1, min(per, n)).astype(int)
            sub = {k: fr[k][sel] for k in ("pts", "inten", "rng", "sensor", "rgb", "cam") + (("sem",) if "sem" in fr else ())}
            age = float((S["ts"][i] - S["ts"][j]) * 1e-6)
            pos += _fill(*hist, pos, sub, age)
    S["tex"] = {"cur": cur, "hist": hist}
    S["tex_key"] = key
    S["frame_ms"] = (time.perf_counter() - t0) * 1e3


def texture(kind, which):
    """kind in cur|hist ; which in pos|attr|rgb"""
    return S["tex"][kind][("pos", "attr", "rgb").index(which)]


# ----------------------------------------------------------------------------------------------- road surface
GROUND_N, GROUND_CELL = 384, 0.35           # 384 x 384 cells of 0.35 m = 134 m square around the ego car
GROUND_EXTENT = GROUND_N * GROUND_CELL


def _raster_map():
    """Motion: rasterise the vector map once into a scene-sized road height field (0.5 m cells)."""
    import cv2
    m = S.get("map")
    if not m:
        return None
    C = GROUND_CELL
    allp = np.concatenate([m[k] for k in m if k.endswith("_pts") and len(m[k])] or [np.zeros((1, 3), np.float32)])
    x0, y0 = allp[:, 0].min() - 20, allp[:, 1].min() - 20
    x1, y1 = allp[:, 0].max() + 20, allp[:, 1].max() + 20
    W, H = int((x1 - x0) / C) + 1, int((y1 - y0) / C) + 1
    W, H = min(W, 2400), min(H, 2400)
    cover = np.zeros((H, W), np.uint8)
    marks = np.zeros((H, W), np.float32)
    height = np.zeros((H, W), np.float32)

    def px(P):
        return np.stack([(P[:, 0] - x0) / C, (P[:, 1] - y0) / C], 1).astype(np.int32).reshape(-1, 1, 2)

    def polylines(kind):
        if kind + "_pts" not in m:
            return
        pts, off = m[kind + "_pts"], m[kind + "_off"]
        for a, b in zip(off[:-1], off[1:]):
            yield pts[a:b]
    zsum, zcnt = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for P in polylines("lane"):                   # drivable lanes ~3.6 m wide
        if len(P) >= 2:
            cv2.polylines(cover, [px(P)], False, 255, thickness=max(int(3.6 / C), 2))
            tmp = np.zeros((H, W), np.uint8)
            cv2.polylines(tmp, [px(P)], False, 1, thickness=max(int(3.6 / C), 2))
            zsum += tmp * float(P[:, 2].mean())
            zcnt += tmp
    for P in polylines("road_line"):
        if len(P) >= 2:
            cv2.polylines(marks, [px(P)], False, 0.8, thickness=1)
    for P in polylines("road_edge"):
        if len(P) >= 2:
            cv2.polylines(marks, [px(P)], False, 0.7, thickness=1)
    for kind, val in (("crosswalk", 0.8), ("speed_bump", 0.55)):
        for P in polylines(kind):
            if len(P) >= 3:
                cv2.fillPoly(cover, [px(P)], 255)
                cv2.fillPoly(marks, [px(P)], val)
                tmp = np.zeros((H, W), np.uint8)
                cv2.fillPoly(tmp, [px(P)], 1)
                zsum += tmp * float(P[:, 2].mean())
                zcnt += tmp
    for P in polylines("driveway"):
        if len(P) >= 3:
            cv2.fillPoly(cover, [px(P)], 255)
            tmp = np.zeros((H, W), np.uint8)
            cv2.fillPoly(tmp, [px(P)], 1)
            zsum += tmp * float(P[:, 2].mean())
            zcnt += tmp
    have = zcnt > 0
    height = np.where(have, zsum / np.maximum(zcnt, 1), 0.0).astype(np.float32)
    # terrain height: extend the road heights outwards (multi-scale normalised blur), then smooth
    out = height.copy()
    got = have.copy()
    for sig in (3.0, 10.0, 30.0, 90.0):
        if got.all():
            break
        num = cv2.GaussianBlur(np.where(got, out, 0.0).astype(np.float32), (0, 0), sig)
        den = cv2.GaussianBlur(got.astype(np.float32), (0, 0), sig)
        take = (~got) & (den > 1e-3)
        out[take] = (num / np.maximum(den, 1e-6))[take]
        got |= take
    if not got.all():
        out[~got] = float(height[have].mean()) if have.any() else 0.0
    height = cv2.GaussianBlur(out, (0, 0), 1.5)
    return {"x0": x0, "y0": y0, "cover": cover > 0, "marks": marks, "height": height, "W": W, "H": H}


def _ground_from_raster(i):
    R = S.get("raster")
    if R is None:
        R = S["raster"] = _raster_map()
    N, C = GROUND_N, GROUND_CELL
    ego = S["poses"][min(max(i, 0), len(S["poses"]) - 1)][:3, 3]
    A = np.zeros((N, N, 4), np.float32)
    B = np.zeros((N, N, 4), np.float32)
    B[..., 3] = 1.0
    if R is None:
        return A, B, (float(ego[0]), -float(ego[1]))
    ox = int(round((ego[0] - N * C / 2 - R["x0"]) / C))
    oy = int(round((ego[1] - N * C / 2 - R["y0"]) / C))
    x_lo, y_lo = max(ox, 0), max(oy, 0)
    x_hi, y_hi = min(ox + N, R["W"]), min(oy + N, R["H"])
    zref = float(R["height"].mean())
    A[..., 0] = zref
    B[..., :3] = (0.030, 0.062, 0.058)                 # terrain outside the mapped road
    A[..., 2] = 1.0                                    # whole window is drawn (terrain), nothing floats
    A[..., 3] = 1.0
    if x_hi > x_lo and y_hi > y_lo:
        sl = (slice(y_lo - oy, y_hi - oy), slice(x_lo - ox, x_hi - ox))
        src = (slice(y_lo, y_hi), slice(x_lo, x_hi))
        A[sl[0], sl[1], 0] = R["height"][src]
        A[sl[0], sl[1], 1] = R["marks"][src]
        road = R["cover"][src]
        B[sl[0], sl[1], :3] = np.where(road[..., None], np.array([0.085, 0.095, 0.115], np.float32),
                                       np.array([0.030, 0.062, 0.058], np.float32))
    # the grid centre is at the ego position quantised to the raster cell so it does not shimmer
    cx = R["x0"] + (ox + N / 2) * C
    cy = R["y0"] + (oy + N / 2) * C
    return A, B, (float(cx), -float(cy))


def _ground_from_corridor(i):
    """E2E: a 3.6 m wide driving corridor painted along the ego's real past/future states and the alternatives."""
    import cv2
    N, C = GROUND_N, GROUND_CELL
    A = np.zeros((N, N, 4), np.float32)
    B = np.zeros((N, N, 4), np.float32)
    B[..., 3] = 1.0
    t = S.get("traj")
    if t is None:
        return A, B, (0.0, 0.0)
    paths = [np.vstack([t["past"], np.zeros((1, 3), np.float32), t["future"]])]
    paths += [np.vstack([np.zeros((1, 3), np.float32), p[np.isfinite(p[:, 0])]]) for p in t["prefs"]]
    cover = np.zeros((N, N), np.uint8)
    marks = np.zeros((N, N), np.float32)

    def px(P):
        return np.stack([P[:, 0] / C + N / 2, P[:, 1] / C + N / 2], 1).astype(np.int32).reshape(-1, 1, 2)
    for k, P in enumerate(paths):
        if len(P) >= 2:
            cv2.polylines(cover, [px(P)], False, 255, thickness=max(int(3.6 / C), 2))
    cf = cv2.GaussianBlur((cover > 0).astype(np.float32), (0, 0), 1.6)          # soft contour -> no stair-steps
    A[..., 2] = np.clip((cf - 0.2) * 5.0, 0.0, 1.0)
    marks = np.clip(1.0 - np.abs(cf - 0.5) * 7.0, 0.0, 1.0) * 0.9                # thin lane-edge band
    A[..., 1] = marks
    A[..., 0] = 0.0
    return A, B, (0.0, 0.0)


def build_ground(i, hist=6):
    """Measured road height field around the ego car: lowest returns per cell of frames i-hist..i (world frame).
    Returns (A, B, centre_td_xz).  A: r height, g intensity, b coverage (1 measured / .5 filled <= 2.5 m), a colour
    fraction.  B: rgb mean camera colour.  Cells with no return stay empty (coverage 0) - nothing is invented."""
    key = ("ground", i, int(hist))
    if S.get("ground_key") == key:
        return S["ground"]
    if S.get("kind") == "motion":                 # no LiDAR road: rasterise the vector map
        S["ground_key"], S["ground"] = key, _ground_from_raster(i)
        return S["ground"]
    if S.get("kind") == "e2e":
        S["ground_key"], S["ground"] = key, _ground_from_corridor(i)
        return S["ground"]
    N, C = GROUND_N, GROUND_CELL
    ego = S["poses"][min(max(i, 0), len(S["poses"]) - 1)][:3, 3]
    cx, cy = float(ego[0]), float(ego[1])
    P, I, RGB, CAM = [], [], [], []
    for j in range(max(0, i - int(hist)), i + 1):
        fr = frame(j)
        if len(fr["pts"]):
            P.append(fr["pts"])
            I.append(fr["inten"])
            RGB.append(fr["rgb"])
            CAM.append(fr["cam"])
    A = np.zeros((N, N, 4), np.float32)
    B = np.zeros((N, N, 4), np.float32)
    B[..., 3] = 1.0
    if not P:
        S["ground_key"], S["ground"] = key, (A, B, (cx, -cy))
        return S["ground"]
    P, I, RGB, CAM = np.concatenate(P), np.concatenate(I), np.concatenate(RGB), np.concatenate(CAM)
    ix = np.floor((P[:, 0] - cx) / C + N / 2).astype(np.int64)
    iy = np.floor((P[:, 1] - cy) / C + N / 2).astype(np.int64)
    ok = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N)
    ix, iy, z, inten, rgb, cam = ix[ok], iy[ok], P[ok, 2], I[ok], RGB[ok], CAM[ok]
    k = iy * N + ix
    zmin = np.full(N * N, np.inf, np.float32)
    np.minimum.at(zmin, k, z)
    sel = z <= zmin[k] + 0.22                         # ground layer of every cell (cars/poles sit above it)
    k, z, inten, rgb, cam = k[sel], z[sel], inten[sel], rgb[sel], cam[sel]
    cnt = np.bincount(k, minlength=N * N).astype(np.float32)
    H = np.bincount(k, weights=z, minlength=N * N) / np.maximum(cnt, 1)
    It = np.bincount(k, weights=np.clip(inten, 0, 2.0), minlength=N * N) / np.maximum(cnt, 1)
    colored = cam >= 0
    cc_ = np.bincount(k[colored], minlength=N * N).astype(np.float32)
    col = np.stack([np.bincount(k[colored], weights=rgb[colored, c], minlength=N * N) for c in range(3)], 1)
    col = col / np.maximum(cc_, 1)[:, None] / 255.0
    import cv2
    valid = (cnt > 0)
    Hn, vn = H.reshape(N, N).astype(np.float32), valid.reshape(N, N)
    ego_z = float(ego[2])
    vn = vn & (Hn > ego_z - 2.0) & (Hn < ego_z + 1.2)               # not tree canopy / overhangs / sky artefacts

    def nsmooth(F, M, sig):                                          # normalised (mask-aware) gaussian
        num = cv2.GaussianBlur(np.where(M, F, 0.0).astype(np.float32), (0, 0), sig)
        den = cv2.GaussianBlur(M.astype(np.float32), (0, 0), sig)
        return num / np.maximum(den, 1e-6), den
    for _ in range(2):                                               # reject cells far from their neighbourhood surface
        Hs, den = nsmooth(Hn, vn, 2.0)
        vn = vn & ((np.abs(Hn - Hs) < 0.30) | (den < 0.05))
    Hs, den = nsmooth(Hn, vn, 1.2)
    It2, _ = nsmooth(It.reshape(N, N), vn, 0.8)
    cmask = (cc_.reshape(N, N) > 0)
    col2 = np.stack([nsmooth(col[:, c].reshape(N, N), cmask, 1.2)[0] for c in range(3)], -1)
    cc2, _ = nsmooth(np.minimum(cc_.reshape(N, N), 1.0), vn, 1.2)
    # fill every hole from its surroundings at increasing scales, so there are no black voids / stair-step edges
    def fill(F, M):
        out, got = F.copy(), M.copy()
        for sig in (2.0, 6.0, 16.0, 40.0):
            if got.all():
                break
            v, d = nsmooth(F, M, sig)
            take = (~got) & (d > 1e-3)
            out[take] = v[take]
            got |= take
        out[~got] = float(F[M].mean()) if M.any() else 0.0
        return out
    Hf = fill(Hs, vn)
    It_f = fill(It2, vn)
    colf = np.stack([fill(col2[..., c], vn | (cc2 > 0.02)) for c in range(3)], -1)
    ccf = fill(cc2, vn)
    # coverage: opaque where measurements are dense, fading with distance from the nearest return (soft contour)
    dens, _ = nsmooth(vn.astype(np.float32), np.ones_like(vn), 7.0)
    cover = np.clip((dens - 0.04) * 14.0, 0.0, 1.0).astype(np.float32).ravel()
    H = Hf.ravel()
    It = It_f.ravel()
    col = colf.reshape(-1, 3)
    cc_ = ccf.ravel() * 2.0
    A.reshape(-1, 4)[:, 0] = H
    A.reshape(-1, 4)[:, 1] = np.clip(It / 0.5, 0.0, 1.0)
    A.reshape(-1, 4)[:, 2] = cover
    A.reshape(-1, 4)[:, 3] = np.clip(cc_ / 2.0, 0.0, 1.0)
    B.reshape(-1, 4)[:, :3] = col
    gt = S.get("gtex")
    if gt is not None and i < len(gt["rgba"]):                    # real camera pixels on the ground (sharp)
        tex = gt["rgba"][i].astype(np.float32) / 255.0
        bx, by = (float(v) for v in gt["centres"][i])
        dx, dy = (cx - bx) / C, (cy - by) / C                     # grid centre is the exact ego position
        xs, ys = np.meshgrid(np.arange(N, dtype=np.float32) + dx, np.arange(N, dtype=np.float32) + dy)
        tex = cv2.remap(tex, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        al = tex[..., 3]
        B[..., :3] = np.where((al > 0.02)[..., None], tex[..., :3], B[..., :3])
        A[..., 3] = np.maximum(A[..., 3], al)
        A[..., 2] = np.maximum(A[..., 2], np.where(al > 0.6, 1.0, 0.0))        # road seen by cameras is drawn
    S["ground_key"], S["ground"] = key, (A, B, (cx, -cy))
    return S["ground"]


# ----------------------------------------------------------------------------------------------- boxes
EDGES_LOOPS = ((0, 1, 2, 3, 0), (4, 5, 6, 7, 4))
EDGES_POSTS = ((0, 4), (1, 5), (2, 6), (3, 7))


def corners_td(row):
    cx, cy, cz, l, w, h, hd = (float(v) for v in row[:7])
    sx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    sy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    sz = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2
    c, s = np.cos(hd), np.sin(hd)
    x = c * sx - s * sy + cx
    y = s * sx + c * sy + cy
    return to_td(np.stack([x, y, sz + cz], 1))


def emit_box(scriptOp, C, y_off=0.0, z_off=0.0):
    for loop in EDGES_LOOPS:
        poly = scriptOp.appendPoly(len(loop), closed=False, addPoints=True)
        for k, idx in enumerate(loop):
            pt = poly[k].point
            pt.x, pt.y, pt.z = float(C[idx, 0]), float(C[idx, 1] + y_off), float(C[idx, 2] + z_off)
    for a, b in EDGES_POSTS:
        poly = scriptOp.appendPoly(2, closed=False, addPoints=True)
        for k, idx in enumerate((a, b)):
            pt = poly[k].point
            pt.x, pt.y, pt.z = float(C[idx, 0]), float(C[idx, 1] + y_off), float(C[idx, 2] + z_off)


def cook_boxes(scriptOp, types, selected_only=False, exclude_selected=True, moving=None):
    scriptOp.clear()
    i = max(S["cur"], 0)
    if S["boxes"] is None:
        return 0
    n = 0
    for row in S["boxes"][i]:
        t, tid = int(row[7]), int(row[8])
        is_sel = tid == S["selected"]
        if selected_only != is_sel and (selected_only or exclude_selected):
            continue
        if t not in types:
            continue
        if moving is not None:
            spd = float(row[9])
            is_moving = spd < 0 or spd > 0.5 or t in (2, 4)       # unknown speed counts as moving; people always do
            if is_moving != moving:
                continue
        emit_box(scriptOp, corners_td(row))
        n += 1
    return n


def cook_echo(scriptOp, step=3, count=6, explode=0.0):
    """Motion echo: ghost boxes of earlier frames; explode = artistic metres of +y per second of age."""
    scriptOp.clear()
    i = max(S["cur"], 0)
    n = 0
    for k in range(1, count + 1):
        j = i - k * step
        if j < 0:
            break
        age = float((S["ts"][i] - S["ts"][j]) * 1e-6)
        for row in S["boxes"][j]:
            if int(row[7]) in (1, 2, 4):
                emit_box(scriptOp, corners_td(row), y_off=age * explode)
                n += 1
    return n


def cook_trails(scriptOp, bucket, history_s=5.0, explode=0.0, types=(1, 2, 4)):
    """bucket 0 recent / 1 intermediate / 2 oldest third of the history window (by real elapsed time)."""
    scriptOp.clear()
    i = max(S["cur"], 0)
    if S["boxes"] is None:
        return 0
    j0 = i
    while j0 > 0 and (S["ts"][i] - S["ts"][j0 - 1]) * 1e-6 <= history_s:
        j0 -= 1
    tr = {}
    for j in range(j0, i + 1):
        age = float((S["ts"][i] - S["ts"][j]) * 1e-6)
        for row in S["boxes"][j]:
            if int(row[7]) in types:
                tr.setdefault(int(row[8]), []).append((age, row[0], row[1], row[2] + row[5] / 2))
    n = 0
    for tid, pts in tr.items():
        u = lambda a: min(int(3 * a / max(history_s, 1e-3)), 2)
        seg = [p for p in pts if u(p[0]) == bucket]
        # one extra point from the next-more-recent bucket so segments join
        nxt = [p for p in pts if u(p[0]) == bucket - 1]
        if bucket > 0 and nxt:
            seg = [nxt[0]] + seg if seg and nxt[0][0] < seg[-1][0] else seg
        if len(seg) < 2:
            continue
        seg.sort(key=lambda p: -p[0])
        poly = scriptOp.appendPoly(len(seg), closed=False, addPoints=True)
        for k, (age, x, y, z) in enumerate(seg):
            q = to_td(np.array([x, y, z]))
            pt = poly[k].point
            pt.x, pt.y, pt.z = float(q[0]), float(q[1] + age * explode), float(q[2])
        n += 1
    return n


def cook_ego(scriptOp):
    scriptOp.clear()
    if S["poses"] is None:
        return 0
    T = ego_pose_at(ego_t())
    L, W, H = 4.9, 2.0, 1.6
    R, t = T[:3, :3], T[:3, 3]
    # local box (vehicle frame, x fwd) through the interpolated pose in Waymo coordinates, then convert to TD axes
    cx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * L / 2
    cy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * W / 2
    cz = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * H
    loc = np.stack([cx, cy, cz], 1)
    w = loc @ R.T + t
    emit_box(scriptOp, to_td(w))
    return 1


MAP_GROUPS = {0: ("lane",), 1: ("road_edge",), 2: ("road_line",), 3: ("crosswalk", "speed_bump", "driveway"),
              4: ("stop_sign",)}


def _poly(scriptOp, P, lift=0.0):
    q = to_td(P)
    poly = scriptOp.appendPoly(len(q), closed=False, addPoints=True)
    for k in range(len(q)):
        pt = poly[k].point
        pt.x, pt.y, pt.z = float(q[k, 0]), float(q[k, 1] + lift), float(q[k, 2])


def cook_map(scriptOp, group):
    """Static vector map (Motion dataset only).  Cooked once when the context is loaded."""
    scriptOp.clear()
    m = S.get("map")
    if not m:
        return 0
    n = 0
    for kind in MAP_GROUPS[group]:
        if kind + "_pts" not in m:
            continue
        pts, off = m[kind + "_pts"], m[kind + "_off"]
        for a, b in zip(off[:-1], off[1:]):
            P = pts[a:b]
            if kind == "stop_sign":
                c = P[0]
                for d in ((1.2, 0.0), (0.0, 1.2)):
                    _poly(scriptOp, np.array([[c[0] - d[0], c[1] - d[1], c[2] + 2.0], [c[0] + d[0], c[1] + d[1], c[2] + 2.0]]))
                _poly(scriptOp, np.array([[c[0], c[1], c[2]], [c[0], c[1], c[2] + 2.0]]))
            elif len(P) >= 2:
                if len(P) > 120:
                    P = P[np.linspace(0, len(P) - 1, 120).astype(int)]
                _poly(scriptOp, P, 0.05)
            n += 1
    return n


def cook_future(scriptOp):
    """Motion: ground-truth future path of the agents the benchmark predicts (+ the SDC), from now to the end."""
    scriptOp.clear()
    f = S.get("future")
    if not f:
        return 0
    i = max(S["cur"], 0)
    ids = set(int(k) for k in f["to_predict"]) | {int(f["sdc"])}
    n = 0
    for k in ids:
        ok = f["valid"][i:, k]
        P = f["traj"][i:, k][ok]
        if len(P) >= 2:
            _poly(scriptOp, P, 0.3)
            n += 1
    return n


def cook_e2e(scriptOp, which):
    """E2E: 0 = ego past, 1 = ego future (ground truth), 2 = alternative trajectories.  Frame = ego at t=0."""
    scriptOp.clear()
    t = S.get("traj")
    if not t:
        return 0
    z = np.array([0.0, 0.0, 0.0], np.float32)
    if which == 0:
        P = np.vstack([t["past"], z[None]])
        if len(P) >= 2:
            _poly(scriptOp, P, 0.35)
    elif which == 1:
        P = np.vstack([z[None], t["future"]])
        if len(P) >= 2:
            _poly(scriptOp, P, 0.35)
    else:
        for row in t["prefs"]:
            P = row[np.isfinite(row[:, 0])]
            if len(P) >= 2:
                _poly(scriptOp, np.vstack([z[None], P]), 0.45)
    return 1


def cook_grid(scriptOp, half=60.0, step=10.0):
    scriptOp.clear()
    T = ego_pose_at(ego_t()) if S["poses"] is not None else np.eye(4)
    cx = round(T[0, 3] / step) * step
    cz = round(-T[1, 3] / step) * step
    n = int(half / step)
    for k in range(-n, n + 1):
        for a in (0, 1):
            poly = scriptOp.appendPoly(2, closed=False, addPoints=True)
            for e, s in enumerate((-half, half)):
                pt = poly[e].point
                if a == 0:
                    pt.x, pt.y, pt.z = cx + k * step, 0.0, cz + s
                else:
                    pt.x, pt.y, pt.z = cx + s, 0.0, cz + k * step
    return 2 * (2 * n + 1)


# ----------------------------------------------------------------------------------------------- camera mosaic
CAM_ORDER = (2, 1, 3, 4, 5)
CAM_ORDER_8 = (2, 1, 3, 5, 4, 6, 7, 8)        # 4 x 2: front row / side+rear row


def mosaic(i, tw=320, th=240):
    import cv2
    if float(S.get("ctrl", {}).get("show_depth", 0)) > 0.5:       # estimated-depth vs LiDAR error overlay (FRONT)
        fn = os.path.join(S["dir"], "depth_err", f"f{i:04d}.jpg")
        if os.path.exists(fn):
            im = cv2.imread(fn)
            if im is not None:
                im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (3 * tw, 2 * th))
                out = np.ones((2 * th, 3 * tw, 4), np.float32)
                out[..., :3] = im[::-1] / 255.0
                return out
    cols = 4 if S.get("kind") == "e2e" else 3
    order = CAM_ORDER_8 if cols == 4 else CAM_ORDER
    out = np.zeros((2 * th, cols * tw, 4), np.float32)  # empty camera slots stay fully transparent
    files = S["meta"]["cam_files"][i] if S["meta"] else {}
    for slot, cid in enumerate(order):
        fn = files.get(str(cid)) or files.get(cid)
        if not fn:
            continue
        key = (i, cid)
        if key not in S["cam_cache"]:
            big = float(S.get("ctrl", {}).get("inset_scale", 0.3)) > 0.6      # full-frame camera scene needs detail
            im = cv2.imread(os.path.join(S["dir"], "cam", fn),
                            cv2.IMREAD_REDUCED_COLOR_4 if big else cv2.IMREAD_REDUCED_COLOR_8)
            if im is None:
                continue
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (tw, th))
            S["cam_cache"] = {k: v for k, v in S["cam_cache"].items() if k[0] == i}
            S["cam_cache"][key] = im
        im = S["cam_cache"][key]
        r, c = divmod(slot, cols)
        out[(1 - r) * th:(2 - r) * th, c * tw:(c + 1) * tw, :3] = im[::-1] / 255.0   # numpyArray rows are bottom-up
        out[(1 - r) * th:(2 - r) * th, c * tw:(c + 1) * tw, 3] = 1.0
    return out
