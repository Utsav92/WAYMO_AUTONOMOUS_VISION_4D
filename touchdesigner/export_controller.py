"""Eight-scene, 60 s film timeline.  `apply(f, n, scientific)` sets controls / camera / time for output frame f.
Data time sweeps the whole recording once (the Time Freeze scene holds it).  Geometry is never altered."""
import numpy as np

import frame_loader as fl
import runtime as rt
import camera_controller as cc
import timeline_controller as tl

TOTAL_S = 60.0
SCENES = [  # (start_s, end_s, title)
    (0, 8, "VISION AWAKENS"),
    (8, 16, "LIDAR ACTIVATION"),
    (16, 24, "OBJECT TRACKING"),
    (24, 32, "RGB SPATIAL FUSION"),
    (32, 40, "TEMPORAL RECONSTRUCTION"),
    (40, 48, "TIME FREEZE"),
    (48, 55, "3D FLY-THROUGH"),
    (55, 60, "WAYMO AUTONOMOUS VISION 4D"),
]


def _ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def scene_of(t):
    for i, (a, b, _) in enumerate(SCENES):
        if a <= t < b:
            return i, (t - a) / (b - a)
    return len(SCENES) - 1, 1.0


def fl_path(u):
    p = fl.to_td(fl.ego_pose_at(fl.ego_t())[:3, 3])
    return p + np.array([-6.0 + 14.0 * u, 3.0, 4.0 - 8.0 * u])


def apply(f, n=1800, scientific=False, fps=30.0):
    if f == 0:                                    # film always starts from the Perception source with defaults
        ctxs = rt.contexts()
        if ctxs and fl.S.get("kind") != "perception":
            rt.use_context(ctxs[0])
        rt.reset_defaults()
        tl.pause()
    t = f / fps
    si, u = scene_of(t)
    D = fl.duration()
    base = dict(show_lidar=1, show_rgb=0, show_boxes=0, show_trails=0, show_history=0, show_grid=1, show_ego=1,
                show_camera=0, inset_scale=0.3, color_mode=0, hist_frames=0, echo=0, explode=0.0, density=1.0,
                cinematic=0 if scientific else 1, glow=0.0 if scientific else 0.5, show_hud=0, point_size=2.5,
                rgb_size=3.0, show_labels=0, show_models=1, show_ground=1, ground_rgb=1, show_map=0, show_future=0,
                show_depth=0, opacity=1.0, vignette=0.25)
    cam_mode, data_t = 1, D * min(t / TOTAL_S, 1.0)
    ego_t = None                                  # None = ego follows data time
    if si == 0:        # camera footage -> RGB point cloud
        base.update(show_camera=1, inset_scale=1.0 - 0.7 * _ease((u - 0.5) / 0.5), show_rgb=1, show_lidar=0,
                    density=0.2 + 0.8 * _ease(u), rgb_size=3.5)
        cam_mode = 0
    elif si == 1:      # orange LiDAR reveals the scene
        base.update(density=_ease(u * 1.5))
        cam_mode = 1 if u < 0.6 else 4
    elif si == 2:      # glowing boxes follow vehicles / pedestrians
        base.update(show_boxes=1, show_labels=1, show_trails=1, hist_frames=0)
        cam_mode = 4
    elif si == 3:      # camera-coloured LiDAR
        base.update(show_boxes=1, show_rgb=1, show_lidar=0 if u > 0.2 else 1, color_mode=4, rgb_size=3.2)
        cam_mode = 4
    elif si == 4:      # cyan / purple historical trails
        base.update(show_boxes=1, show_rgb=1, color_mode=4, show_trails=1, show_history=1,
                    hist_frames=int(10 + 15 * u), echo=4)
        cam_mode = 1 if u < 0.5 else 4
    elif si == 5:      # frozen traffic, orbiting camera
        base.update(show_boxes=1, show_rgb=1, color_mode=4, show_trails=1, show_history=1, hist_frames=25, echo=6,
                    explode=0.9 * _ease(u * 2.0))
        cam_mode = 4
        data_t = D * (40 / TOTAL_S)               # traffic frozen at this frame ...
        ego_t = D * (40 + 8 * u) / TOTAL_S        # ... while the ego car keeps driving through it
    elif si == 6:      # fly through the reconstruction
        base.update(show_boxes=1, show_rgb=1, show_lidar=1, color_mode=4, show_history=1, hist_frames=15)
        cam_mode = 3
        data_t = D * (48 + 7 * u) / TOTAL_S
    else:              # aerial overview
        base.update(show_boxes=1, show_rgb=1, show_trails=1, show_history=1, hist_frames=25, color_mode=4)
        cam_mode = 2 if u > 0.25 else 4
        data_t = D * (55 + 5 * u) / TOTAL_S
    for k, v in base.items():
        rt.set_ctrl(k, v)
    rt.set_ctrl("orbit_radius", 16 + 20 * u if si in (1, 2, 3, 5) else (60.0 if cam_mode == 2 else 24.0))
    tl.seek(data_t)
    if cc.ST["mode"] != cam_mode:
        cc.set_mode(cam_mode)
    if cam_mode == 3:
        cc.ST["fly_pos"] = fl_path(u)
        cc.ST["yaw"] = 0.0
        cc.ST["pitch"] = -0.12
    fl.S["selected"] = -1
    fl.S["dt_override"] = 1.0 / fps                # offline render: smoothing runs on film time, not wall time
    fl.S["t_ego"] = ego_t
    for _ in range(2):
        rt.tick()
    title_a = _ease(min(u * 5, 1.0)) * _ease(min((1 - u) * 5, 1.0))
    sub = {0: "CAMERA + LIDAR FUSION", 7: "CAMERA + LIDAR FUSION\n3D OBJECT TRACKING\nSPATIAL RECONSTRUCTION\n"
                                         "FOUR-DIMENSIONAL PERCEPTION"}
    txt = SCENES[si][2] + (("\n" + sub[si]) if si in sub else "")
    tt = rt.td.op(rt.ROOT + "/UI/title_text")
    tt.par.text = txt
    tt.par.fontalpha = float(title_a)
    e = fl.ego_pose_at(fl.ego_t())[:2, 3]
    return {"scene": si, "u": round(u, 2), "data_t": round(data_t, 2), "ego_xy": [round(float(e[0]), 1), round(float(e[1]), 1)]}
