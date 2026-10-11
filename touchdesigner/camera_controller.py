"""Five virtual cameras.  Output is a (position, target) pair in TD space; the geometry is never touched.
Modes: 0 driver POV, 1 chase, 2 bird's-eye, 3 free-fly, 4 cinematic orbit."""
import numpy as np

import frame_loader as fl

MODE_NAMES = ("DRIVER POV", "CHASE", "BIRD'S EYE", "FREE FLY", "CINEMATIC ORBIT")
ST = {"mode": 1, "pos": np.array([-14.0, 7.0, 0.0]), "tgt": np.array([0.0, 1.5, 0.0]), "fly_pos": np.array([-10.0, 5.0, 8.0]),
      "yaw": 0.0, "pitch": -0.25, "orbit_a": 0.6, "init": False, "orbit_target": -1}


def set_mode(m):
    ST["mode"] = int(m) % 5
    if ST["mode"] == 3:   # free-fly starts where the camera currently is
        ST["fly_pos"] = ST["pos"].copy()
        d = ST["tgt"] - ST["pos"]
        ST["yaw"] = float(np.arctan2(d[2], d[0]))
        ST["pitch"] = float(np.arcsin(np.clip(d[1] / (np.linalg.norm(d) + 1e-9), -1, 1)))


def fly(forward=0.0, right=0.0, up=0.0, dyaw=0.0, dpitch=0.0, zoom=0.0, boost=False):
    """Free-fly input (W/A/S/D = forward/right, Q/E = up, mouse drag = yaw/pitch, wheel = zoom)."""
    k = 3.0 if boost else 1.0
    ST["yaw"] += dyaw
    ST["pitch"] = float(np.clip(ST["pitch"] + dpitch, -1.5, 1.5))
    f = np.array([np.cos(ST["yaw"]) * np.cos(ST["pitch"]), np.sin(ST["pitch"]), np.sin(ST["yaw"]) * np.cos(ST["pitch"])])
    r = np.array([-np.sin(ST["yaw"]), 0.0, np.cos(ST["yaw"])])
    ST["fly_pos"] = ST["fly_pos"] + k * (forward * f + right * r + np.array([0, up, 0.0]) + zoom * f)


def _ego(t):
    T = fl.ego_pose_at(t)
    p = fl.to_td(T[:3, 3])
    fwd = fl.to_td(T[:3, 0])        # vehicle +x in TD axes
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    return p, fwd


def desired(t, ctrl):
    p, fwd = _ego(t)
    mode = ST["mode"]
    if mode == 0:                                   # driver: roof-height, looking along the road
        return p + np.array([0, 1.6, 0]) + fwd * 0.8, p + np.array([0, 1.5, 0]) + fwd * 30.0
    if mode == 1:                                   # chase
        return p - fwd * 13.0 + np.array([0, 5.5, 0]), p + fwd * 12.0 + np.array([0, 1.0, 0])
    if mode == 2:                                   # bird's-eye (slight tilt keeps the up vector valid)
        h = float(ctrl.get("orbit_radius", 40.0))
        return p + np.array([-0.1, h, 0.0]) - fwd * 0.5, p + fwd * 6.0
    if mode == 3:
        f = np.array([np.cos(ST["yaw"]) * np.cos(ST["pitch"]), np.sin(ST["pitch"]), np.sin(ST["yaw"]) * np.cos(ST["pitch"])])
        return ST["fly_pos"], ST["fly_pos"] + f * 10.0
    # orbit around the selected object (if it exists in this frame) or the ego car
    centre = p + np.array([0, 1.0, 0])
    i = max(fl.S["cur"], 0)
    if fl.S["boxes"] is not None and fl.S["selected"] >= 0:
        for row in fl.S["boxes"][i]:
            if int(row[8]) == fl.S["selected"]:
                centre = fl.to_td(np.array([row[0], row[1], row[2]]))
    ST["orbit_a"] += 0.012 * float(ctrl.get("orbit_speed", 1.0))
    R = float(ctrl.get("orbit_radius", 22.0))
    return centre + np.array([np.cos(ST["orbit_a"]) * R, 0.45 * R, np.sin(ST["orbit_a"]) * R]), centre


def update(t, ctrl, smooth=0.18, dt=None):
    """dt (s) makes the follow smoothing frame-rate independent (time constant ~0.14 s)."""
    import math
    if dt:
        smooth = 1.0 - math.exp(-7.0 * min(dt, 0.5))
    pos, tgt = desired(t, ctrl)
    if not ST["init"]:
        ST["pos"], ST["tgt"], ST["init"] = pos, tgt, True
    k = 1.0 if ST["mode"] in (3,) else smooth
    ST["pos"] = ST["pos"] + (pos - ST["pos"]) * k
    ST["tgt"] = ST["tgt"] + (tgt - ST["tgt"]) * k
    return ST["pos"], ST["tgt"]
