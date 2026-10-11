"""Timestamp-based playback.  Data time is real elapsed seconds since the first frame; frames are picked from the
recorded timestamps, so uneven frame spacing plays at true speed."""
import time

import numpy as np

import frame_loader as fl

S = fl.S


def play():
    S["playing"], S["frozen"], S["direction"] = True, False, 1.0


def reverse():
    S["playing"], S["frozen"], S["direction"] = True, False, -1.0


def pause():
    S["playing"] = False


def freeze(on=None):
    S["frozen"] = (not S["frozen"]) if on is None else bool(on)


def reset():
    S["t"], S["playing"], S["direction"], S["dirty"] = 0.0, False, 1.0, True
    S["cam_snap"] = True


def step(n=1):
    i = int(np.clip(S["cur"] + n, 0, fl.n_frames() - 1))
    S["t"], S["dirty"], S["playing"] = float(S["t_rel"][i]), True, False


def seek(t):
    new = float(np.clip(t, 0.0, fl.duration()))
    if abs(new - S["t"]) > 1.0:           # a jump in time: the camera must not fly there from the old place
        S["cam_snap"] = True
    S["t"] = new
    S["dirty"] = True


def advance(wall_dt, loop=True):
    """Move data time by wall_dt * speed.  Frozen time does not advance (the camera may still move)."""
    if S["meta"] is None or not S["playing"] or S["frozen"]:
        return
    t = S["t"] + S["direction"] * S["speed"] * wall_dt
    D = fl.duration()
    if t > D or t < 0:
        S["cam_snap"] = True              # loop wrap: ego teleports 300 m, snap the camera with it
    if t > D:
        t = 0.0 if loop else D
        if not loop:
            S["playing"] = False
    elif t < 0:
        t = D if loop else 0.0
        if not loop:
            S["playing"] = False
    S["t"] = t
    S["dirty"] = True


def current_frame():
    return fl.frame_at_time(S["t"])


def tick(ctrl):
    """Called every TD frame.  ctrl: dict of control values.  Returns True when the displayed frame changed."""
    now = time.perf_counter()
    dt = 0.0 if S["last_wall"] is None else min(now - S["last_wall"], 0.25)
    S["last_wall"] = now
    S["dt"] = S.get("dt_override") or dt
    if dt > 0:
        S["fps"] = 0.9 * S["fps"] + 0.1 * (1.0 / dt) if S["fps"] else 1.0 / dt
    S["speed"] = float(ctrl.get("speed", 1.0))
    advance(dt)
    i = current_frame()
    changed = i != S["cur"] or S["dirty"]
    S["cur"], S["dirty"] = i, False
    return changed
