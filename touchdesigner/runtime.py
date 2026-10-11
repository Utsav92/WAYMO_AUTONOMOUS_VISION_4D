"""Per-frame driver + cook entry points for the Waymo TD network (imported module; reached via sys.path)."""
import os
import sys

import numpy as np
import td

import frame_loader as fl
import timeline_controller as tl
import camera_controller as cc

ROOT = fl.ROOT
S = fl.S
CTRL_DEFAULTS = dict(
    speed=1.0, point_size=1.8, rgb_size=2.2, density=1.0, max_range=75.0, hist_frames=10, trail_s=5.0, trail_decay=3.0,
    orbit_radius=24.0, orbit_speed=1.0, cam_speed=1.0, glow=0.35, explode=0.0, echo=0,
    show_lidar=1, show_rgb=0, show_camera=1, show_boxes=1, show_labels=1, show_trails=1, show_history=1,
    show_ped_traj=1, show_veh_traj=1, show_grid=1, show_ego=1, show_seg=0, show_depth=0,
    color_mode=0, cinematic=0, sensor_mask=62, vignette=0.25, inset_scale=0.3, show_hud=1, show_models=1, show_map=1, show_future=1,
    show_ground=1, ground_rgb=1, ground_marks=1, ground_hist=6, opacity=1.0, show_panel=1,
)
CTRL_NAMES = tuple(CTRL_DEFAULTS)


def ctrl():
    c = td.op(ROOT + "/UI/controls")
    d = dict(CTRL_DEFAULTS)
    if c is not None:
        for ch in c.chans():
            d[ch.name] = float(ch.eval())
    S["ctrl"] = d
    return d


def set_ctrl(name, value):
    c = td.op(ROOT + "/UI/controls")
    c.seq.const[CTRL_NAMES.index(name)].par.value = float(value)


# what each source can really provide (shown in the HUD; nothing is faked)
AVAILABLE = {
    "perception": "LiDAR  RGB-points  3D boxes  road(measured)  semantic  depth-error | no map / future in this dataset",
    "motion": "MAP  3D boxes  road+terrain(from map)  future paths  LiDAR(1 s live, then static memory) | no camera images -> no RGB points",
    "e2e": "8 cameras  ego past/future  alternatives  road(corridor)  RGB-points(ESTIMATED) | no LiDAR / boxes / map",
}
DEFAULT_LAYERS = {
    "perception": dict(show_lidar=1, show_rgb=0, show_ground=1, show_boxes=1, show_models=1, show_trails=1,
                       show_history=1, show_map=0, show_future=0, color_mode=0),
    "motion": dict(show_lidar=1, show_rgb=0, show_ground=1, show_boxes=1, show_models=1, show_trails=1,
                   show_history=0, show_map=1, show_future=1, color_mode=0),
    "e2e": dict(show_lidar=1, show_rgb=1, show_ground=1, show_boxes=0, show_models=1, show_trails=0,
                show_history=0, show_map=0, show_future=1, color_mode=4),
}


def use_context(path):
    """Switch to another processed context.  Transactional: if loading fails the previous source stays intact."""
    old = dict(S)
    try:
        fl.load(path)
        if fl.n_frames() < 1 or not os.path.exists(os.path.join(path, "frames", "f0000.npz")):
            raise FileNotFoundError("incomplete context: " + path)
    except Exception as e:                       # roll back: never leave the scene half-loaded
        S.clear()
        S.update(old)
        S["source_error"] = "%s: %s" % (os.path.basename(path), e)
        return None
    S.pop("source_error", None)
    cc.ST["init"] = False
    S["selected"], S["cur"] = -1, 0
    tl.reset()
    c = ctrl()
    kind = S.get("kind", "perception")
    for k, v in CTRL_DEFAULTS.items():             # normal view: no leftover film state (full-screen camera, echo ...)
        set_ctrl(k, v)
    S.pop("dt_override", None)
    S["t_ego"] = None
    t_ = td.op(ROOT + "/UI/title_text")
    if t_ is not None:
        t_.par.text = ""
    for k, v in DEFAULT_LAYERS.get(kind, DEFAULT_LAYERS["perception"]).items():
        set_ctrl(k, v)
    tl.play()
    set_ctrl("show_future", DEFAULT_LAYERS[kind]["show_future"])
    for k in STATIC_SOPS:
        n = _sop_node(k)
        if n is not None:
            n.cook(force=True)
    S["hist_key"] = None
    return kind


def contexts():
    """Complete, non-synthetic processed contexts (meta.json + every frame file present), perception first."""
    import glob, json, os
    base = os.path.join(td.op(ROOT).fetch("project_dir"), "data", "processed")
    out = []
    for p in sorted(glob.glob(os.path.join(base, "*", "meta.json"))):
        d = os.path.dirname(p)
        try:
            m = json.load(open(p))
            if m.get("synthetic"):
                continue
            fdir = os.path.join(d, "frames")
            if os.path.isdir(fdir) and len([f for f in os.listdir(fdir) if f.endswith(".npz")]) >= int(m["frames"]):
                out.append(d)
        except Exception:
            continue
    rank = {"perception": 0, "motion": 1, "e2e": 2}
    out.sort(key=lambda d: (rank.get(json.load(open(os.path.join(d, "meta.json"))).get("kind", "perception"), 3), d))
    return out


def reset_defaults():
    """Interactive state: default controls, chase camera, time 0, no title card."""
    for k, v in CTRL_DEFAULTS.items():
        set_ctrl(k, v)
    cc.set_mode(1)
    cc.ST["init"] = False
    S["selected"] = -1
    S["t_ego"] = None
    tl.reset()
    t = td.op(ROOT + "/UI/title_text")
    if t is not None:
        t.par.text = ""


# ------------------------------------------------------------------------------ cook entry points
def cook_top(scriptOp, kind):
    """kind: cur_pos|cur_attr|cur_rgb|hist_pos|hist_attr|hist_rgb|mosaic"""
    scriptOp.par.outputresolution = "custom"
    if S["meta"] is None:
        arr = np.zeros((4, 4, 4), np.float32)
    elif kind == "mosaic":
        arr = fl.mosaic(max(S["cur"], 0))
    elif kind in ("ground_a", "ground_b"):
        if False:
            pass
        else:
            a, b, ctr = fl.build_ground(max(S["cur"], 0), int(S.get("ctrl", CTRL_DEFAULTS)["ground_hist"]))
            g_ = td.op(ROOT + "/GROUND")
            if g_ is not None:
                g_.store("gcx", float(ctr[0]))
                g_.store("gcz", float(ctr[1]))
            arr = a if kind == "ground_a" else b
    else:
        a, b = kind.split("_")
        fl.build_textures(max(S["cur"], 0), int(S.get("ctrl", CTRL_DEFAULTS)["hist_frames"]))
        arr = fl.texture(a, b)
    scriptOp.par.resolutionw = arr.shape[1]
    scriptOp.par.resolutionh = arr.shape[0]
    scriptOp.copyNumpyArray(np.ascontiguousarray(arr, dtype=np.float32))


def cook_sop(scriptOp, kind):
    c = S.get("ctrl", CTRL_DEFAULTS)
    if S["meta"] is None:
        scriptOp.clear()
        return
    if kind.startswith("box_"):
        names = {"box_vehicle": (1,), "box_vehicle_static": (1,), "box_pedestrian": (2,), "box_cyclist": (4,),
                 "box_sign": (3,)}
        if kind == "box_selected":
            fl.cook_boxes(scriptOp, (1, 2, 3, 4), selected_only=True)
        elif kind == "box_vehicle":
            fl.cook_boxes(scriptOp, (1,), moving=True)
        elif kind == "box_vehicle_static":
            fl.cook_boxes(scriptOp, (1,), moving=False)
        else:
            fl.cook_boxes(scriptOp, names[kind])
    elif kind == "echo":
        fl.cook_echo(scriptOp, step=3, count=int(c["echo"]) or 0, explode=c["explode"])
    elif kind.startswith("trail"):
        types = []
        if c["show_veh_traj"] > 0.5:
            types.append(1)
        if c["show_ped_traj"] > 0.5:
            types += [2, 4]
        fl.cook_trails(scriptOp, int(kind[-1]), history_s=c["trail_s"], explode=c["explode"], types=tuple(types))
    elif kind.startswith("map"):
        fl.cook_map(scriptOp, int(kind[3:]))
    elif kind == "future":
        fl.cook_future(scriptOp)
    elif kind.startswith("e2e"):
        fl.cook_e2e(scriptOp, int(kind[3:]))
    elif kind == "ego":
        if cc.ST["mode"] == 0 or c.get("show_models", 1) > 0.5:   # inside the car / model shown: no ego box
            scriptOp.clear()
        else:
            fl.cook_ego(scriptOp)
    elif kind == "grid":
        fl.cook_grid(scriptOp)


MODELS = ("waymo_car", "car", "truck", "pedestrian", "cyclist", "sign")
_dims = {}


def _model_dims(name):
    if name not in _dims:
        import json, os
        p = os.path.join(td.op(ROOT).fetch("project_dir"), "assets", "models", name + ".json")
        _dims[name] = json.load(open(p))["dims"]
    return _dims[name]


def cook_chop(scriptOp, model):
    """Instance transforms (tx,ty,tz,ry,sx,sy,sz) for one road-user model, from the current frame labels."""
    import math
    c = S.get("ctrl", CTRL_DEFAULTS)
    rows = []
    if S["meta"] is not None and c.get("show_models", 1) > 0.5:
        d = _model_dims(model)
        if model == "waymo_car":
            if cc.ST["mode"] != 0 and c.get("show_ego", 1) > 0.5:
                T = fl.ego_pose_at(fl.ego_t())
                yaw = math.atan2(T[1, 0], T[0, 0])
                p = fl.to_td(T[:3, 3] + np.array([math.cos(yaw), math.sin(yaw), 0.0]) * 1.4)   # axle -> body centre
                rows.append((p[0], p[1], p[2], math.degrees(yaw), 1.0, 1.0, 1.0))
        else:
            want = {"car": (1,), "truck": (1,), "pedestrian": (2,), "cyclist": (4,), "sign": (3,)}[model]
            for r in S["boxes"][max(S["cur"], 0)]:
                t = int(r[7])
                if t not in want:
                    continue
                big = r[3] >= 6.5
                if (model == "truck") != big and t == 1:
                    continue
                p = fl.to_td(r[:3])
                if model == "sign":                       # plate at the labelled centre, pole down to the ground
                    s_ = float(np.clip(max(r[3], r[4], r[5]) / 0.62, 0.8, 3.0))
                    rows.append((p[0], p[1], p[2], math.degrees(float(r[6])), s_, s_, s_))
                    continue
                rows.append((p[0], p[1] - r[5] / 2.0, p[2], math.degrees(float(r[6])),
                             max(r[3], 0.3) / d["L"], max(r[5], 0.3) / d["H"], max(r[4], 0.3) / d["W"]))
    if not rows:
        rows = [(0.0, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)]    # one degenerate instance keeps the CHOP valid
    scriptOp.clear()
    scriptOp.numSamples = len(rows)
    for k, name in enumerate(("tx", "ty", "tz", "ry", "sx", "sy", "sz")):
        ch = scriptOp.appendChan(name)
        ch.vals = [float(r[k]) for r in rows]
    for name in ("rx", "rz"):                  # explicit zero channels: all three rotation axes are mapped
        ch = scriptOp.appendChan(name)
        ch.vals = [0.0] * len(rows)


STATIC_SOPS = ("map0", "map1", "map2", "map3", "map4", "e2e0", "e2e1", "e2e2")
SOPS = ("future", "box_vehicle", "box_vehicle_static", "box_pedestrian", "box_cyclist", "box_sign", "box_selected", "echo", "trail0", "trail1",
        "trail2", "ego", "grid")
TOPS = ("cur_pos", "cur_attr", "cur_rgb", "hist_pos", "hist_attr", "hist_rgb", "mosaic", "ground_a", "ground_b")


def _path(kind):
    d = {"cur_pos": "LIDAR_RENDERER/position_texture", "cur_attr": "LIDAR_RENDERER/attribute_texture",
         "cur_rgb": "RGB_POINTCLOUD/rgb_colors", "hist_pos": "TEMPORAL_4D/history_positions",
         "hist_attr": "TEMPORAL_4D/history_attributes", "hist_rgb": "TEMPORAL_4D/history_colors",
         "mosaic": "CAMERA_SYSTEM/camera_mosaic", "ground_a": "GROUND/ground_data",
         "ground_b": "GROUND/ground_color"}
    d.update({"map0": "OBJECT_TRACKING/map_lanes", "map1": "OBJECT_TRACKING/map_road_edges",
              "map2": "OBJECT_TRACKING/map_road_lines", "map3": "OBJECT_TRACKING/map_crossings",
              "map4": "OBJECT_TRACKING/map_stop_signs", "e2e0": "OBJECT_TRACKING/e2e_ego_past",
              "e2e1": "OBJECT_TRACKING/e2e_ego_future", "e2e2": "OBJECT_TRACKING/e2e_alternatives",
              "future": "OBJECT_TRACKING/future_paths",
              "echo": "TEMPORAL_4D/motion_echo", "ego": "OBJECT_TRACKING/ego_vehicle",
              "grid": "OBJECT_TRACKING/ground_grid", "trail0": "OBJECT_TRACKING/motion_paths_0",
              "trail1": "OBJECT_TRACKING/motion_paths_1", "trail2": "OBJECT_TRACKING/motion_paths_2"})
    return d.get(kind) or "OBJECT_TRACKING/" + kind


def _sop_node(kind):
    base = _path(kind)
    geo = td.op(ROOT + "/" + base)
    return geo.op("sop") if geo is not None else None


# ------------------------------------------------------------------------------ keyboard / mouse
_in = {"tx": None, "ty": None, "wheel": None}


def poll_input():
    """W/A/S/D move, Q/E up/down, arrows look, Shift = fast, left-drag look, wheel zoom.  Any movement key switches
    to the free-fly camera.  Reads the Keyboard In / Mouse In CHOPs (only live while the TD window has focus)."""
    k = td.op(ROOT + "/UI/keys_in")
    if k is None:
        return
    ks = td.op(ROOT + "/UI/keys_shift")
    mo = td.op(ROOT + "/UI/mouse_in")
    v = lambda op_, n: float(op_[n][0]) if op_ is not None and op_[n] is not None else 0.0
    fwd = v(k, "kw") - v(k, "ks")
    right = v(k, "kd") - v(k, "ka")
    up = v(k, "kq") - v(k, "ke")
    # note: Q = up, E = down
    up = v(k, "kq") - v(k, "ke")
    boost = any(v(ks, n) > 0.5 for n in ("kw", "ka", "ks", "kd", "kq", "ke")) if ks is not None else False
    look_x = v(k, "kright") - v(k, "kleft")
    look_y = v(k, "kup") - v(k, "kdown")
    dyaw = 0.03 * look_x
    dpitch = 0.025 * look_y
    if mo is not None and v(mo, "lbutton") > 0.5:
        tx, ty = v(mo, "tx"), v(mo, "ty")
        if _in["tx"] is not None:
            dyaw += (tx - _in["tx"]) * 2.5
            dpitch += (ty - _in["ty"]) * 1.6
        _in["tx"], _in["ty"] = tx, ty
    else:
        _in["tx"] = _in["ty"] = None
    wheel = v(mo, "wheel")
    zoom = 0.0 if _in["wheel"] is None else (wheel - _in["wheel"]) * 1.0     # first read only seeds the baseline
    _in["wheel"] = wheel
    speed = float(S.get("ctrl", CTRL_DEFAULTS).get("cam_speed", 1.0))
    moving = any((fwd, right, up, dyaw, dpitch, zoom))
    if moving:
        if cc.ST["mode"] != 3:
            cc.set_mode(3)
        cc.fly(forward=0.35 * fwd * speed, right=0.35 * right * speed, up=0.3 * up * speed, dyaw=dyaw, dpitch=dpitch,
               zoom=zoom * speed, boost=boost)
    S["input"] = {"fwd": fwd, "right": right, "up": up, "boost": boost, "dyaw": dyaw, "dpitch": dpitch}
    pk = S.setdefault("input_peak", {"fwd": 0.0, "boost": False})     # lets a test read what happened during a key hold
    pk["fwd"] = max(pk["fwd"], abs(fwd))
    pk["boost"] = pk["boost"] or boost


# ------------------------------------------------------------------------------ main tick
def tick():
    c = ctrl()
    poll_input()
    changed = tl.tick(c)
    if S["meta"] is None:
        return
    i = S["cur"]
    # camera (runs every frame, even when time is frozen)
    ego_now = fl.to_td(fl.ego_pose_at(fl.ego_t())[:3, 3])
    if S.pop("cam_snap", False):
        cc.ST["init"] = False                      # time discontinuity: no fly-back across the map
        prev = S.get("ego_prev")
        if cc.ST["mode"] == 3 and prev is not None:  # free camera keeps its offset from the ego car
            cc.ST["fly_pos"] = cc.ST["fly_pos"] + (ego_now - prev)
    S["ego_prev"] = ego_now
    pos, tgt = cc.update(fl.ego_t(), c, dt=S.get("dt"))
    cam = td.op(ROOT + "/VIRTUAL_CAMERAS/main_cam")
    aim = td.op(ROOT + "/VIRTUAL_CAMERAS/look_target")
    if cam is not None and aim is not None:
        cam.par.tx, cam.par.ty, cam.par.tz = float(pos[0]), float(pos[1]), float(pos[2])
        aim.par.tx, aim.par.ty, aim.par.tz = float(tgt[0]), float(tgt[1]), float(tgt[2])
    if changed or S.get("hist_key") != (c["hist_frames"], c["trail_s"], c["explode"], c["echo"], c["show_veh_traj"],
                                         c["show_ped_traj"]):
        S["hist_key"] = (c["hist_frames"], c["trail_s"], c["explode"], c["echo"], c["show_veh_traj"], c["show_ped_traj"])
        for k in TOPS:
            if k == "mosaic" and not changed:
                continue
            o = td.op(ROOT + "/" + _path(k))
            if o is not None:
                o.cook(force=True)
        for k in SOPS:
            n = _sop_node(k)
            if n is not None:
                n.cook(force=True)
    elif S.get("ego_dirty", True):
        pass
    if changed or S["playing"] or S.get("model_mode") != cc.ST["mode"] or S.get("model_key") != c["show_models"]:
        S["model_mode"], S["model_key"] = cc.ST["mode"], c["show_models"]
        for m in MODELS:
            n = td.op(ROOT + "/OBJECT_TRACKING/MODELS/inst_" + m)
            if n is not None:
                n.cook(force=True)
    for k in ("ego", "grid"):          # follow the interpolated pose every frame
        n = _sop_node(k)
        if n is not None and (S["playing"] or S.pop("ego_dirty", False) or S.get("last_mode") != cc.ST["mode"]):
            S["last_mode"] = cc.ST["mode"]
            n.cook(force=True)
    now = __import__("time").time()
    pnl0 = td.op(ROOT + "/UI/ui_root/panel")
    if pnl0 is not None:
        import ui_panel
        ui_panel.poll_sliders(pnl0, sys.modules[__name__])
    if now - S.get("ui_t", 0) > 0.25 or S.pop("ui_dirty", False):          # keep the control box in sync (4 Hz)
        S["ui_t"] = now
        pnl = td.op(ROOT + "/UI/ui_root/panel")
        if pnl is not None:
            import ui_panel
            ui_panel.refresh(pnl, sys.modules[__name__])
    hud(c)


def hud(c):
    i = max(S["cur"], 0)
    m = S["meta"]
    rows = S["boxes"][i] if S["boxes"] is not None else np.zeros((0, 10))
    ts = int(S["ts"][i])
    nv = int((rows[:, 7] == 1).sum()) if len(rows) else 0
    npd = int((rows[:, 7] == 2).sum()) if len(rows) else 0
    ncy = int((rows[:, 7] == 4).sum()) if len(rows) else 0
    state = "FROZEN" if S["frozen"] else ("PLAY" if S["playing"] and S["direction"] > 0 else
                                          ("REVERSE" if S["playing"] else "PAUSE"))
    lines = [
        "WAYMO AUTONOMOUS VISION 4D" + ("   [SYNTHETIC SELF-TEST DATA - NOT WAYMO]" if S["synthetic"] else ""),
        "context %s   [%s]" % (m["context"], (m.get("dataset") or S.get("kind", "perception")).upper()),
        "available: " + AVAILABLE.get(S.get("kind", "perception"), ""),
        ("SOURCE ERROR: " + S["source_error"]) if S.get("source_error") else "",
        "frame %d / %d   t = %.2f s   ts = %d us   %s x%.1f" % (i, m["frames"] - 1, S["t"], ts, state, S["speed"]),
        "objects %d  (vehicles %d, pedestrians %d, cyclists %d)   cam: %s" % (len(rows), nv, npd, ncy,
                                                                           cc.MODE_NAMES[cc.ST["mode"]]),
        "pts %d   render FPS %.1f   upload %.1f ms" % (len(fl.frame(i)["pts"]), S["fps"], S["frame_ms"]),
    ]
    ego = fl.to_td(fl.ego_pose_at(fl.ego_t())[:3, 3])
    near = []
    for row in rows:
        if int(row[7]) in (1, 2, 4):
            p = fl.to_td(row[:3])
            d = float(np.linalg.norm(p - ego))
            sp = float(row[9])
            near.append((d, "%s %d  %.0f m%s" % (fl.BOX_NAMES.get(int(row[7]), "?"), int(row[8]), d,
                                                 ("  %.1f m/s" % sp) if sp >= 0 else "")))
    near.sort()
    if c["show_labels"] > 0.5:
        lines += [" ", "NEAREST TRACKED"] + [t for _, t in near[:6]]
    t = td.op(ROOT + "/UI/hud_text")
    if t is not None:
        t.par.text = "\n".join(lines) if c["show_hud"] > 0.5 else ""
