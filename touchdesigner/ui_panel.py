"""Left-side control box: layout spec, action dispatch and live state refresh.

Action strings stored on every widget (`act`):
    T:<ctrl>      toggle a 0/1 control            V:<name>   view preset        M:<name>  temporal preset
    R:<name>      render style preset             A:<CMD>    ui_controller command (PLAY, CAM1 ...)
    S:<ctrl>      slider bound to a control (value range below)
"""
import frame_loader as fl
import ui_controller as uic

PANEL_W, PANEL_H = 340, 720
ROW_H, GAP, PAD = 24, 3, 8

# sections: (title, columns, [(label, act), ...])
SECTIONS = [
    ("RENDER STYLE", 2, [("SCIENTIFIC", "R:sci"), ("CINEMATIC", "R:cine")]),
    ("LAYERS", 3, [
        ("RGB POINTS", "T:show_rgb"), ("LIDAR", "T:show_lidar"), ("DEPTH ERROR", "T:show_depth"),
        ("3D BOXES", "T:show_boxes"), ("3D MODELS", "T:show_models"), ("ROAD", "T:show_ground"),
        ("LABELS", "T:show_labels"), ("GRID", "T:show_grid"), ("TRAJECTORY", "T:show_trails"),
        ("HISTORY", "T:show_history"), ("GLOW", "G:glow"), ("STATISTICS", "T:show_hud"),
        ("MAP", "T:show_map"), ("FUTURE PATHS", "T:show_future"), ("SEMANTIC", "C:5")]),
    ("VIEW PRESETS", 4, [("LIDAR", "V:lidar"), ("RGB", "V:rgb"), ("BOTH", "V:both"), ("BOTH + ERROR", "V:error")]),
    ("TEMPORAL 4D", 4, [("OFF", "M:off"), ("ECHO", "M:echo"), ("TRAILS", "M:trails"), ("EXPLOSION", "M:explode")]),
    ("CAMERA", 5, [("DRIVER", "A:CAM0"), ("CHASE", "A:CAM1"), ("FREE", "A:CAM3"), ("BIRD", "A:CAM2"),
                   ("ORBIT", "A:CAM4")]),
    ("TIMELINE", 4, [("PLAY", "A:PLAY"), ("PAUSE", "A:PAUSE"), ("REVERSE", "A:REVERSE"), ("RESET", "A:RESET"),
                     ("STEP -", "A:STEP-"), ("STEP +", "A:STEP+"), ("FREEZE TIME", "A:FREEZE"),
                     ("NEXT SOURCE", "A:SOURCE")]),
]

# (label, control, lo, hi, format)
SLIDERS = [
    ("Point density", "density", 0.05, 1.0, "%.2f"),
    ("LiDAR point size", "point_size", 1.0, 8.0, "%.2f"),
    ("RGB point size", "rgb_size", 1.0, 8.0, "%.2f"),
    ("LiDAR opacity", "opacity", 0.05, 1.0, "%.2f"),
    ("Max visible range (m)", "max_range", 10.0, 120.0, "%.0f"),
    ("History (frames)", "hist_frames", 0.0, 40.0, "%.0f"),
    ("Trail length (s)", "trail_s", 1.0, 10.0, "%.1f"),
    ("Trail decay (s)", "trail_decay", 0.5, 6.0, "%.1f"),
    ("Orbit / bird radius (m)", "orbit_radius", 8.0, 90.0, "%.0f"),
    ("Camera speed", "cam_speed", 0.2, 5.0, "%.1f"),
    ("Playback speed", "speed", 0.125, 4.0, "%.2f"),
    ("Glow intensity", "glow", 0.0, 1.5, "%.2f"),
    ("Time explosion (m/s)", "explode", 0.0, 2.0, "%.2f"),
]
SLIDER_BY_CTRL = {s[1]: s for s in SLIDERS}

COL_ON_LAYER = (0.05, 0.62, 0.30)      # green  = layer on
COL_ON_MODE = (0.95, 0.45, 0.0)        # orange = active mode / preset
COL_OFF = (0.16, 0.20, 0.24)
COL_TIME = (0.10, 0.35, 0.65)          # blue   = timeline commands


def layout():
    """-> list of dict(kind, x, y, w, h, label, act, ctrl) in container coordinates (origin bottom-left)."""
    items = []
    y = PANEL_H - PAD
    inner = PANEL_W - 2 * PAD
    for title, cols, buttons in SECTIONS:
        y -= 16
        items.append(dict(kind="label", x=PAD, y=y, w=inner, h=14, label=title, act=""))
        y -= 3
        bw = (inner - (cols - 1) * GAP) / cols
        for k, (lab, act) in enumerate(buttons):
            r, c = divmod(k, cols)
            items.append(dict(kind="button", x=PAD + c * (bw + GAP), y=y - (r + 1) * (ROW_H + GAP) + GAP,
                              w=bw, h=ROW_H, label=lab, act=act))
        rows = (len(buttons) + cols - 1) // cols
        y -= rows * (ROW_H + GAP)
    y -= 16
    items.append(dict(kind="label", x=PAD, y=y, w=inner, h=14, label="CONTROLS", act=""))
    y -= 3
    for lab, ctrl, lo, hi, fmt in SLIDERS:
        y -= 20
        items.append(dict(kind="text", x=PAD, y=y, w=140, h=18, label=lab, act=""))
        items.append(dict(kind="slider", x=PAD + 142, y=y + 2, w=inner - 142 - 44, h=14, label=ctrl, act="S:" + ctrl))
        items.append(dict(kind="value", x=PAD + inner - 42, y=y, w=42, h=18, label=ctrl, act=""))
    return items


# ------------------------------------------------------------------------------------------------ actions
def _set(rt, **kw):
    for k, v in kw.items():
        rt.set_ctrl(k, v)


def apply(act, rt):
    c = rt.ctrl()
    kind, _, arg = act.partition(":")
    if kind == "T":
        rt.set_ctrl(arg, 0.0 if c[arg] > 0.5 else 1.0)
    elif kind == "G":                                   # glow toggle
        rt.set_ctrl("glow", 0.0 if c["glow"] > 0.05 else 0.45)
    elif kind == "C":                                   # colour mode toggle (semantic)
        rt.set_ctrl("color_mode", float(arg) if c["color_mode"] != float(arg) else 0.0)
    elif kind == "R":
        if arg == "sci":
            _set(rt, cinematic=0, glow=0.0, vignette=0.0)
        else:
            _set(rt, cinematic=1, glow=0.45, vignette=0.25)
    elif kind == "V":
        if arg == "lidar":
            _set(rt, show_lidar=1, show_rgb=0, color_mode=0, show_depth=0)
        elif arg == "rgb":
            _set(rt, show_lidar=0, show_rgb=1, color_mode=4, show_depth=0)
        elif arg == "both":
            _set(rt, show_lidar=1, show_rgb=1, color_mode=4, show_depth=0)
        elif arg == "error":
            _set(rt, show_lidar=1, show_rgb=1, color_mode=4, show_depth=1)
    elif kind == "M":
        if arg == "off":
            _set(rt, show_trails=0, show_history=0, echo=0, explode=0.0)
        elif arg == "echo":
            _set(rt, echo=6, explode=0.0)
        elif arg == "trails":
            _set(rt, show_trails=1, show_history=1, echo=0, explode=0.0, hist_frames=max(c["hist_frames"], 10))
        elif arg == "explode":
            _set(rt, show_trails=1, show_history=1, explode=0.9, echo=6)
    elif kind == "A":
        uic.command(arg, rt)
    fl.S["dirty"] = True
    S_ui_dirty()


def S_ui_dirty():
    fl.S["ui_dirty"] = True


def is_on(act, c, S):
    kind, _, arg = act.partition(":")
    if kind == "T":
        return c.get(arg, 0) > 0.5, COL_ON_LAYER
    if kind == "G":
        return c["glow"] > 0.05, COL_ON_LAYER
    if kind == "C":
        return int(c["color_mode"]) == int(arg), COL_ON_LAYER
    if kind == "R":
        return (c["cinematic"] < 0.5) == (arg == "sci"), COL_ON_MODE
    if kind == "V":
        l, r, d = c["show_lidar"] > 0.5, c["show_rgb"] > 0.5, c["show_depth"] > 0.5
        return {"lidar": l and not r and not d, "rgb": r and not l and not d, "both": l and r and not d,
                "error": d}[arg], COL_ON_MODE
    if kind == "M":
        t, h, e, x = c["show_trails"] > 0.5, c["show_history"] > 0.5, c["echo"] > 0, c["explode"] > 0
        on = {"off": not (t or h or e or x), "echo": e and not x, "trails": t and h and not e and not x,
              "explode": x}[arg]
        return on, COL_ON_MODE
    if kind == "A":
        import camera_controller as cc
        if arg.startswith("CAM"):
            return cc.ST["mode"] == int(arg[3:]), COL_ON_MODE
        if arg == "PLAY":
            return bool(S["playing"]) and S["direction"] > 0 and not S["frozen"], COL_TIME
        if arg == "PAUSE":
            return not S["playing"], COL_TIME
        if arg == "REVERSE":
            return bool(S["playing"]) and S["direction"] < 0, COL_TIME
        if arg == "FREEZE":
            return bool(S["frozen"]), COL_TIME
    return False, COL_ON_LAYER


def slider_from_value(ctrl, v01):
    _, _, lo, hi, _ = SLIDER_BY_CTRL[ctrl]
    return lo + (hi - lo) * float(v01)


def slider_to_value(ctrl, val):
    _, _, lo, hi, _ = SLIDER_BY_CTRL[ctrl]
    return (float(val) - lo) / (hi - lo)


_last = {}


def poll_sliders(panel, rt):
    """Slider values are polled every tick (TD's value-change callback is not delivered for slider COMPs)."""
    import time
    for w in panel.children:
        if w.fetch("kind", "") != "slider":
            continue
        ctrl = w.fetch("ctrl", "")
        v = float(w.par.value0.eval())
        last = _last.get(ctrl)
        if last is None:
            _last[ctrl] = v
        elif abs(v - last) > 1e-4:                    # moved by the user
            _last[ctrl] = v
            rt.set_ctrl(ctrl, slider_from_value(ctrl, v))
            fl.S["slider_t"] = time.time()
            fl.S["dirty"] = True


def refresh(panel, rt):
    """Recolour toggles and sync slider positions / value texts with the controls (cheap; called ~4x per second)."""
    c = rt.ctrl()
    S = fl.S
    for w in panel.children:
        act = w.fetch("act", "")
        k = w.fetch("kind", "")
        if k == "button" and act:
            on, col = is_on(act, c, S)
            rgb = col if on else COL_OFF
            w.par.colorr, w.par.colorg, w.par.colorb = rgb
        elif k == "slider":
            ctrl = w.fetch("ctrl", "")
            if ctrl in c and (__import__("time").time() - S.get("slider_t", 0)) > 0.6:
                v = slider_to_value(ctrl, c[ctrl])
                if abs(float(w.par.value0.eval()) - v) > 1e-3:
                    w.par.value0 = v
                    _last[ctrl] = v
        elif k == "value":
            ctrl = w.fetch("ctrl", "")
            if ctrl in c:
                w.par.text = SLIDER_BY_CTRL[ctrl][4] % c[ctrl]
