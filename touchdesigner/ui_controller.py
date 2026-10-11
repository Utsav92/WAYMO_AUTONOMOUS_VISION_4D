"""Button commands for the on-screen control panel + keyboard free-fly.  Imported module; `runtime.tick` polls keys."""
import frame_loader as fl
import timeline_controller as tl
import camera_controller as cc

# (name, label, row, col)
BUTTONS = [
    ("PLAY", "PLAY", 0, 0), ("PAUSE", "PAUSE", 0, 1), ("REVERSE", "REVERSE", 0, 2), ("STEP-", "STEP -", 0, 3),
    ("STEP+", "STEP +", 0, 4), ("FREEZE", "FREEZE", 0, 5), ("RESET", "RESET", 0, 6),
    ("CAM0", "DRIVER", 1, 0), ("CAM1", "CHASE", 1, 1), ("CAM2", "BIRD", 1, 2), ("CAM3", "FREE", 1, 3),
    ("CAM4", "ORBIT", 1, 4), ("SPD-", "SPEED -", 1, 5), ("SPD+", "SPEED +", 1, 6),
    ("LIDAR", "LIDAR", 2, 0), ("RGB", "RGB PTS", 2, 1), ("BOXES", "BOXES", 2, 2), ("TRAILS", "TRAILS", 2, 3),
    ("HISTORY", "HISTORY", 2, 4), ("COLOR", "COLOR", 2, 5), ("CINE", "CINEMATIC", 2, 6),
    ("HIST-", "HIST -", 3, 0), ("HIST+", "HIST +", 3, 1), ("PT-", "PT SIZE -", 3, 2), ("PT+", "PT SIZE +", 3, 3),
    ("GLOW", "GLOW", 3, 4), ("ECHO", "ECHO", 3, 5), ("EXPLODE", "TIME EXPLODE", 3, 6),
    ("SOURCE", "NEXT SOURCE", 4, 0), ("MODELS", "3D MODELS", 4, 1), ("MAP", "MAP", 4, 2),
    ("FUTURE", "FUTURE PATHS", 4, 3), ("BOXES2", "BOXES", 4, 4), ("DEPTH", "DEPTH ERR", 4, 5),
    ("SEMANTIC", "SEMANTIC", 4, 6),
]


def command(name, rt):
    c = rt.ctrl()
    g = lambda k: float(c[k])
    s = lambda k, v: rt.set_ctrl(k, v)
    tog = lambda k: s(k, 0.0 if g(k) > 0.5 else 1.0)
    if name == "PLAY":
        tl.play()
    elif name == "PAUSE":
        tl.pause()
    elif name == "REVERSE":
        tl.reverse()
    elif name == "STEP+":
        tl.step(1)
    elif name == "STEP-":
        tl.step(-1)
    elif name == "FREEZE":
        tl.freeze()
    elif name == "RESET":
        tl.reset()
    elif name.startswith("CAM"):
        cc.set_mode(int(name[3:]))
        fl.S["ego_dirty"] = True
    elif name == "SPD+":
        s("speed", min(g("speed") * 2.0, 8.0))
    elif name == "SPD-":
        s("speed", max(g("speed") / 2.0, 0.125))
    elif name == "LIDAR":
        tog("show_lidar")
    elif name == "RGB":
        tog("show_rgb")
    elif name == "BOXES":
        tog("show_boxes")
    elif name == "TRAILS":
        tog("show_trails")
    elif name == "HISTORY":
        tog("show_history")
    elif name == "COLOR":
        s("color_mode", (g("color_mode") + 1) % 6)
    elif name == "CINE":
        tog("cinematic")
    elif name == "HIST+":
        s("hist_frames", min(g("hist_frames") + 5, 50))
    elif name == "HIST-":
        s("hist_frames", max(g("hist_frames") - 5, 0))
    elif name == "PT+":
        s("point_size", min(g("point_size") + 0.5, 8))
    elif name == "PT-":
        s("point_size", max(g("point_size") - 0.5, 1))
    elif name == "GLOW":
        s("glow", 0.0 if g("glow") > 0.05 else 0.6)
    elif name == "ECHO":
        s("echo", 0 if g("echo") > 0 else 6)
    elif name == "SOURCE":
        ctxs = rt.contexts()
        cur = fl.S["dir"]
        i = ctxs.index(cur) if cur in ctxs else -1
        rt.use_context(ctxs[(i + 1) % len(ctxs)])
    elif name == "MODELS":
        tog("show_models")
    elif name == "MAP":
        tog("show_map")
    elif name == "FUTURE":
        tog("show_future")
    elif name == "DEPTH":
        tog("show_depth")
    elif name == "SEMANTIC":
        s("color_mode", 5 if g("color_mode") != 5 else 0)
    elif name == "BOXES2":
        tog("show_boxes")
    elif name == "EXPLODE":
        s("explode", 0.0 if g("explode") > 0 else 0.8)
    fl.S["dirty"] = True
    return name
