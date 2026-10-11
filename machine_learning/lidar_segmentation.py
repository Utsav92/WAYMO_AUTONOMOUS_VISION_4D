"""LiDAR semantic segmentation: a small per-point MLP trained on Waymo's `lidar_segmentation` labels.

Features (from the range image only, no neighbours search): vehicle-frame xyz, range, intensity, elongation,
horizontal/vertical range-gradient to the 4 neighbouring pixels, height above the lowest return in a 3 m column
neighbourhood proxy, laser one-hot.  Labels are Waymo's 23-class taxonomy (0 = undefined, ignored).
Evaluation is on a held-out CONTEXT (a different drive segment), never on training frames.

    python lidar_segmentation.py --root D:/WAYMO_DATA/raw --train A B C --test D --out models/seg.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(PROJ, "preprocessing"), os.path.join(PROJ, "datasets")]

CLASSES = ["UNDEFINED", "CAR", "TRUCK", "BUS", "OTHER_VEHICLE", "MOTORCYCLIST", "BICYCLIST", "PEDESTRIAN", "SIGN",
           "TRAFFIC_LIGHT", "POLE", "CONSTRUCTION_CONE", "BICYCLE", "MOTORCYCLE", "BUILDING", "VEGETATION",
           "TREE_TRUNK", "CURB", "ROAD", "LANE_MARKER", "OTHER_GROUND", "WALKABLE", "SIDEWALK"]
N_CLASS = len(CLASSES)
N_FEAT = 15 + 5


def features(img, extrinsic, inclination, laser):
    """img (H,W,C>=3 range,intensity,elongation).  Returns pts_vehicle [N,3], F [N,N_FEAT], rc [N,2]."""
    from calibration import range_image_to_points
    pts, attrs, valid, rc = range_image_to_points(img, extrinsic, inclination)
    r = img[..., 0].astype(np.float32)
    pad = np.pad(r, 1, mode="edge")
    nb = [pad[1:-1, :-2], pad[1:-1, 2:], pad[:-2, 1:-1], pad[2:, 1:-1]]
    g = np.stack([np.clip(n[valid] - r[valid], -5, 5) * (n[valid] > 0) for n in nb], 1)
    # crude height-above-ground proxy: z minus the lowest return of the same azimuth column in the same laser
    z_img = np.full(r.shape, np.inf, np.float32)
    z_img[rc[:, 0], rc[:, 1]] = pts[:, 2]
    col_min = np.min(z_img, axis=0)
    h_above = pts[:, 2] - col_min[rc[:, 1]]
    onehot = np.zeros((len(pts), 5), np.float32)
    onehot[:, int(laser) - 1] = 1.0
    F = np.concatenate([pts, attrs[:, 2:3] / 75.0, np.tanh(attrs[:, 0:1]), np.tanh(attrs[:, 1:2]), g,
                        np.clip(h_above, 0, 6)[:, None], np.sin(np.arctan2(pts[:, 1], pts[:, 0]))[:, None],
                        np.cos(np.arctan2(pts[:, 1], pts[:, 0]))[:, None], np.tanh(pts[:, 2:3] / 2.0),
                        np.linalg.norm(pts[:, :2], axis=1, keepdims=True) / 75.0, onehot], 1).astype(np.float32)
    return pts, F, rc


def load_labelled(root, context, split="validation", max_frames=None):
    """-> X [N,F], y [N] int, frame_id [N] for every labelled (frame, laser) range image of a context."""
    import pyarrow.parquet as pq
    from waymo_adapter import component_path, read_component, find_col
    from decode_lidar import load_lidar_calibration
    from calibration import beam_inclinations
    cal = load_lidar_calibration(read_component(root, "lidar_calibration", context, split))
    seg_p = component_path(root, "lidar_segmentation", context, split)
    if seg_p is None:
        return None
    seg = pq.read_table(seg_p).to_pylist()
    X, Y, FID = [], [], []
    stamps = sorted({r["key.frame_timestamp_micros"] for r in seg})[:max_frames]
    lid_p = component_path(root, "lidar", context, split)
    for fi, ts in enumerate(stamps):
        lrows = pq.read_table(lid_p, filters=[("key.frame_timestamp_micros", "=", int(ts))]).to_pylist()
        for lr in lrows:
            ln = lr["key.laser_name"]
            sr = next((s for s in seg if s["key.frame_timestamp_micros"] == ts and s["key.laser_name"] == ln), None)
            if sr is None:
                continue
            img = np.asarray(lr[next(k for k in lr if k.endswith("range_image_return1.values"))], np.float32)
            img = img.reshape(tuple(lr[next(k for k in lr if k.endswith("range_image_return1.shape"))]))
            lab = np.asarray(sr[next(k for k in sr if k.endswith("range_image_return1.values"))], np.int32)
            lab = lab.reshape(tuple(sr[next(k for k in sr if k.endswith("range_image_return1.shape"))]))
            inc = beam_inclinations(img.shape[0], cal[ln]["bi_min"], cal[ln]["bi_max"], cal[ln]["values"])
            pts, F, rc = features(img, cal[ln]["extrinsic"], inc, ln)
            y = lab[rc[:, 0], rc[:, 1], 1]                      # channel 1 = semantic class (0 = instance id)
            X.append(F)
            Y.append(y)
            FID.append(np.full(len(y), fi))
    if not X:
        return None
    return np.concatenate(X), np.concatenate(Y).astype(np.int64), np.concatenate(FID)


def build_model(n_in=N_FEAT, hidden=192):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, N_CLASS))


def train(X, y, epochs=12, batch=4096, lr=2e-3, seed=0):
    import torch
    torch.manual_seed(seed)
    keep = y > 0
    X, y = X[keep], y[keep]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = torch.from_numpy((X - mu) / sd)
    yt = torch.from_numpy(y)
    cnt = np.bincount(y, minlength=N_CLASS).astype(np.float64)
    w = torch.tensor((1.0 / np.sqrt(np.maximum(cnt, 1))) / (1.0 / np.sqrt(np.maximum(cnt, 1))).mean(), dtype=torch.float32)
    net = build_model()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(yt) + batch - 1) // batch))
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    for ep in range(epochs):
        perm = torch.randperm(len(yt))
        tot = 0.0
        for i in range(0, len(yt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xn[idx]), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        print(f"  epoch {ep + 1}/{epochs} loss {tot / len(yt):.4f}", flush=True)
    return net, mu, sd


class Predictor:
    def __init__(self, path):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = build_model(ck.get("n_feat", N_FEAT))
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.mu, self.sd = ck["mu"], ck["sd"]

    def predict_range_image(self, values, shape, calib, laser):
        from calibration import beam_inclinations
        img = np.asarray(values, np.float32).reshape(tuple(int(s) for s in shape))
        inc = beam_inclinations(img.shape[0], calib["bi_min"], calib["bi_max"], calib["values"])
        _, F, _ = features(img, calib["extrinsic"], inc, laser)
        return predict(self.net, self.mu, self.sd, F).astype(np.uint8)


def load(path):
    return Predictor(path)


def predict(net, mu, sd, X):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 65536):
            out.append(net(torch.from_numpy((X[i:i + 65536] - mu) / sd)).argmax(1).numpy())
    return np.concatenate(out)


def iou_report(y_true, y_pred):
    keep = y_true > 0
    yt, yp = y_true[keep], y_pred[keep]
    per, present = {}, []
    for c in range(1, N_CLASS):
        inter = int(((yt == c) & (yp == c)).sum())
        union = int(((yt == c) | (yp == c)).sum())
        n = int((yt == c).sum())
        if n > 0:
            per[CLASSES[c]] = {"iou": inter / max(union, 1), "support": n}
            present.append(inter / max(union, 1))
    return {"accuracy": float((yt == yp).mean()), "mIoU": float(np.mean(present)) if present else 0.0,
            "classes_evaluated": len(present), "per_class": per, "n_points": int(keep.sum())}


if __name__ == "__main__":
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-pts", type=int, default=600_000, help="random training points kept per context")
    ap.add_argument("--out", default=os.path.join(PROJ, "models", "seg.pt"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    Xs, Ys = [], []
    for c in a.train:
        d = load_labelled(a.root, c, a.split)
        print("train", c[:24], None if d is None else (d[0].shape, "labelled pts", int((d[1] > 0).sum())), flush=True)
        if d is not None:
            lab = np.nonzero(d[1] > 0)[0]
            if len(lab) > a.max_pts:
                lab = np.random.default_rng(0).choice(lab, a.max_pts, replace=False)
            Xs.append(d[0][lab])
            Ys.append(d[1][lab])
    X, y = np.concatenate(Xs), np.concatenate(Ys)
    net, mu, sd = train(X, y, epochs=a.epochs)
    res = {"train_contexts": a.train, "test_contexts": a.test, "train_points": int((y > 0).sum()), "tests": {}}
    for c in a.test:
        d = load_labelled(a.root, c, a.split)
        if d is None:
            continue
        pred = predict(net, mu, sd, d[0])
        res["tests"][c] = iou_report(d[1], pred)
        print("TEST", c[:24], "acc %.3f mIoU %.3f (%d classes)" % (res["tests"][c]["accuracy"], res["tests"][c]["mIoU"],
                                                                res["tests"][c]["classes_evaluated"]))
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "classes": CLASSES, "n_feat": N_FEAT}, a.out)
    with open(os.path.splitext(a.out)[0] + "_metrics.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved", a.out)
