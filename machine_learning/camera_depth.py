"""Monocular depth (Depth Anything V2 Small, pretrained, CPU) vs projected LiDAR.

The network predicts RELATIVE inverse depth.  It is converted to metric depth by ONE global affine fit per image
(least squares on inverse depth against LiDAR, robust to outliers) -- this scale/shift alignment is explicit and
reported.  Everything it produces is ESTIMATED camera depth, never measured.  Metrics are computed on valid pixels
only: projected TOP-LiDAR returns inside the image that are not occluded (nearest return per 5x5 px) and within 70 m.

    python camera_depth.py --root D:/WAYMO_DATA/raw --context NAME --frames 5 60 120 --camera 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(PROJ, "preprocessing"), os.path.join(PROJ, "datasets"), HERE]
from evaluation import depth_metrics  # noqa: E402

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_pipe = None


def estimate_inverse_depth(image_rgb: np.ndarray) -> np.ndarray:
    """-> (H,W) relative inverse depth (larger = closer), float32, same size as the image."""
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("depth-estimation", model=MODEL_ID, device=-1)
    from PIL import Image
    out = _pipe(Image.fromarray(image_rgb))
    d = np.asarray(out["predicted_depth"].squeeze().cpu().numpy() if hasattr(out["predicted_depth"], "cpu")
                   else out["predicted_depth"], np.float32)
    import cv2
    return cv2.resize(d, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


def lidar_depth_map(pts_vehicle, cam, shape, cell=4, max_depth=70.0):
    """Sparse LiDAR depth image + validity mask (nearest return per cell => occluded points dropped)."""
    from calibration import project_to_camera
    H, W = shape
    u, v, d, ok = project_to_camera(pts_vehicle, cam["T_vehicle_camera"], cam["intrinsic"], W, H)
    ok &= d < max_depth
    # Returns above the horizon in clear sky are sensor artefacts (the image shows empty sky there): evaluate only
    # at or below the camera horizon (infinite point straight ahead in the vehicle frame).
    _, vh, _, _ = project_to_camera(np.array([[1.0e4, 0.0, 0.0]]), cam["T_vehicle_camera"], cam["intrinsic"], W, H)
    keep = v[ok] >= float(vh[0]) - 3.0
    u, v, d = u[ok][keep].astype(int), v[ok][keep].astype(int), d[ok][keep]
    depth = np.zeros((H, W), np.float32)
    order = np.argsort(-d)                       # far first so near overwrites
    depth[v[order], u[order]] = d[order]
    # occlusion rejection: a return is valid only if no nearer return is within +-2 px
    import cv2
    near = cv2.erode(np.where(depth > 0, depth, 1e6).astype(np.float32), np.ones((5, 5), np.uint8))
    valid = (depth > 0) & (depth <= near + 0.5)
    return depth, valid


def align_inverse(pred_inv, lidar_depth, valid, iters=3):
    """Fit lidar_inv ~ s * pred_inv + t (robust, IRLS) -> metric depth map and (s, t)."""
    x = pred_inv[valid].astype(np.float64)
    y = 1.0 / np.maximum(lidar_depth[valid], 0.5)
    w = np.ones_like(x)
    s, t = 1.0, 0.0
    for _ in range(iters):
        A = np.stack([x * w, w], 1)
        (s, t), *_ = np.linalg.lstsq(A, y * w, rcond=None)
        r = np.abs(s * x + t - y)
        w = 1.0 / np.maximum(r / (np.median(r) + 1e-6), 1.0)
    inv = np.maximum(s * pred_inv + t, 1e-3)
    return (1.0 / inv).astype(np.float32), float(s), float(t)


def evaluate_frame(image_rgb, pts_vehicle, cam):
    inv = estimate_inverse_depth(image_rgb)
    ld, valid = lidar_depth_map(pts_vehicle, cam, image_rgb.shape[:2])
    metric, s, t = align_inverse(inv, ld, valid)
    m = depth_metrics(metric, ld, valid)
    m.update(scale=s, shift=t, n_valid=int(valid.sum()))
    return metric, ld, valid, m


def error_image(image_rgb, metric, ld, valid, max_err=0.35):
    """Colour-coded |relative error| at valid LiDAR pixels over the dimmed camera image (blue ok -> red bad)."""
    import cv2
    out = (image_rgb.astype(np.float32) * 0.45).astype(np.uint8)
    ys, xs = np.nonzero(valid)
    e = np.abs(metric[ys, xs] - ld[ys, xs]) / np.maximum(ld[ys, xs], 1.0)
    c = cv2.applyColorMap((np.clip(e / max_err, 0, 1) * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    for x, y, col in zip(xs, ys, c):
        cv2.circle(out, (int(x), int(y)), 3, (int(col[2]), int(col[1]), int(col[0])), -1)
    return out


if __name__ == "__main__":
    import cv2
    from decode_lidar import decode_frames
    from decode_camera import load_camera_calibration, frame_images, decode_jpeg
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--frames", type=int, nargs="+", default=[5, 60, 120])
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(PROJ, "output", "screenshots"))
    ap.add_argument("--save-processed", help="processed context dir: write depth_err/fNNNN.jpg for every frame given")
    a = ap.parse_args()
    cams = load_camera_calibration(a.root, a.context, a.split)
    imgs = frame_images(a.root, a.context, a.split)
    rows = []
    if a.save_processed:       # every frame: error overlays for the TouchDesigner depth-disagreement layer
        a.frames = list(range(len(imgs)))
        os.makedirs(os.path.join(a.save_processed, "depth_err"), exist_ok=True)
    frames = {f: fr for f, fr in zip(range(max(a.frames) + 1), decode_frames(a.root, a.context, a.split, max(a.frames) + 1))
              if f in a.frames}
    for f in a.frames:
        fr = frames[f]
        im = decode_jpeg(imgs[fr["ts"]][a.camera][0])
        top = fr["sensor"] == 1            # TOP 64-beam lidar only: the secondary lidars add spurious near-range returns
        metric, ld, valid, m = evaluate_frame(im, fr["xyz_vehicle"][top], cams[a.camera])
        rows.append(m)
        print(f"frame {f}: MAE {m['mae']:.2f} m  RMSE {m['rmse']:.2f} m  AbsRel {m['absrel']:.3f} "
              f"(median {m['median_absrel']:.3f})  d<1.25 {100 * m['delta125']:.0f}%  "
              f"valid {m['valid_pct']:.2f}% ({m['n_valid']} px)  scale {m['scale']:.3f} shift {m['shift']:.4f}")
        os.makedirs(a.out, exist_ok=True)
        eimg = cv2.cvtColor(error_image(im, metric, ld, valid), cv2.COLOR_RGB2BGR)
        if a.save_processed:
            cv2.imwrite(os.path.join(a.save_processed, "depth_err", f"f{f:04d}.jpg"), cv2.resize(eimg, (960, 640)),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
        else:
            cv2.imwrite(os.path.join(a.out, f"depth_error_f{f:04d}.jpg"), eimg)
    print("mean:", {k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("mae", "rmse", "absrel", "median_absrel", "delta125", "valid_pct")})
    with open(os.path.join(a.out, "depth_eval.json"), "w") as fh:
        json.dump({"model": MODEL_ID, "context": a.context, "camera": a.camera, "frames": a.frames, "metrics": rows}, fh)
