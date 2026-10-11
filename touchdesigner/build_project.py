"""Build /project1/WAYMO_AUTONOMOUS_VISION_4D inside a running TouchDesigner (run via tools/tdrun.py).

  python touchdesigner/tools/tdrun.py touchdesigner/build_project.py

Environment: WAYMO_CONTEXT_DIR = processed context dir (meta.json + frames/); defaults to the newest real context,
falling back to the synthetic self-test context (flagged on screen).
"""
import glob
import importlib
import json
import os
import sys

PROJECT_DIR = os.environ.get("WAYMO_DIR", r"C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D")
TD_DIR = os.path.join(PROJECT_DIR, "touchdesigner")
SHADER_DIR = os.path.join(PROJECT_DIR, "shaders")
ROOT_PATH = "/project1/WAYMO_AUTONOMOUS_VISION_4D"
WARN = []

PANEL_EXEC_SRC = """import sys, time
_p = r'%s'
if _p not in sys.path: sys.path.insert(0, _p)

def onOffToOn(panelValue):
    import runtime, ui_panel
    act = panelValue.owner.fetch('act', '')
    if act and not act.startswith('S:'):
        ui_panel.apply(act, runtime)
    return

def onValueChange(panelValue, prev):
    import runtime, ui_panel, frame_loader
    o = panelValue.owner
    if o.fetch('kind', '') == 'slider' and panelValue.name == 'u':
        ctrl = o.fetch('ctrl', '')
        frame_loader.S['slider_t'] = time.time()
        runtime.set_ctrl(ctrl, ui_panel.slider_from_value(ctrl, panelValue.val))
        frame_loader.S['dirty'] = True
    return
"""

SHOW_EXEC_SRC = """import sys
_p = r'%s'
if _p not in sys.path: sys.path.insert(0, _p)

def onOffToOn(panelValue):
    import runtime, ui_panel
    ui_panel.apply(panelValue.owner.fetch('act', ''), runtime)
    return
"""

OUT_W, OUT_H = 1280, 720          # non-commercial TouchDesigner caps at 1280 px


def say(m):
    print(m)


def warn(m):
    WARN.append(m)
    print("WARNING:", m)


def fresh(parent, optype, name):
    old = parent.op(name)
    if old is not None:
        old.destroy()
    return parent.create(optype, name)


def setp(o, **kw):
    for k, v in kw.items():
        try:
            setattr(o.par, k, v)
        except Exception as e:
            warn(f"{o.path}.{k}={v!r}: {e}")


def text_dat(parent, name, path=None, text=None):
    d = fresh(parent, textDAT, name)
    if path:
        d.par.file = path
        d.par.loadonstart = True
        d.par.syncfile = True
        d.par.loadonstartpulse.pulse()
    if text is not None:
        d.text = text
    return d


def set_vec(mat, i, name, *vals):
    try:
        if mat.seq.vec.numBlocks <= i:
            mat.seq.vec.numBlocks = i + 1
        blk = mat.seq.vec[i]
        blk.par.name = name
        for k, v in zip("xyzw", vals):
            getattr(blk.par, "value" + k).val = v
    except Exception as e:
        warn(f"{mat.name}: uniform {name}: {e}")


def vec_expr(mat, i, comps):
    for k, expr in zip("xyzw", comps):
        try:
            getattr(mat.seq.vec[i].par, "value" + k).expr = expr
        except Exception as e:
            warn(f"{mat.name}: expr {expr}: {e}")


def set_sampler(mat, i, name, top):
    try:
        if mat.seq.sampler.numBlocks <= i:
            mat.seq.sampler.numBlocks = i + 1
        blk = mat.seq.sampler[i]
        blk.par.name = name
        blk.par.top = top
    except Exception as e:
        warn(f"{mat.name}: sampler {name}: {e}")


class Uniforms:
    def __init__(self, op_):
        self.op, self.i = op_, 0

    def f(self, name, value):
        set_vec(self.op, self.i, name, 0.0)
        if isinstance(value, str):
            vec_expr(self.op, self.i, [value])
        else:
            set_vec(self.op, self.i, name, float(value))
        self.i += 1

    def v3(self, name, exprs):
        set_vec(self.op, self.i, name, 0.0, 0.0, 0.0)
        vec_expr(self.op, self.i, list(exprs))
        self.i += 1


def make_mat(parent, name, vert, frag):
    m = fresh(parent, glslMAT, name)
    m.par.vdat = text_dat(parent, name + "_vert", os.path.join(SHADER_DIR, vert))
    m.par.pdat = text_dat(parent, name + "_frag", os.path.join(SHADER_DIR, frag))
    return m


def script_top(parent, name, kind):
    cb = text_dat(parent, name + "_callbacks", text=(
        "def onSetupParameters(scriptOp):\n    return\n\ndef onPulse(par):\n    return\n\n"
        "def onCook(scriptOp):\n    import runtime\n    runtime.cook_top(scriptOp, %r)\n    return\n" % kind))
    t = fresh(parent, scriptTOP, name)
    t.par.callbacks = cb
    try:
        t.par.format = "rgba32float"
    except Exception as e:
        warn(f"{name}: format: {e}")
    return t


def grid_points_geo(parent, name, cols, rows):
    geo = fresh(parent, geometryCOMP, name)
    for c in list(geo.children):
        c.destroy()
    g = geo.create(gridSOP, "grid1")
    setp(g, rows=rows, cols=cols, sizex=10.0, sizey=10.0, texture="rowcol")
    try:
        g.par.surftype = "triangles"
    except Exception as e:
        warn(f"{name}: surftype: {e}")
    conv = geo.create(convertSOP, "to_points")
    conv.inputConnectors[0].connect(g)
    setp(conv, totype="part", prtype="pointsprites")
    conv.display = conv.render = True
    g.display = g.render = False
    return geo


def script_sop_geo(parent, name, kind, mat=None):
    geo = fresh(parent, geometryCOMP, name)
    for c in list(geo.children):
        c.destroy()
    cb = text_dat(geo, "callbacks", text=(
        "def onSetupParameters(scriptOp):\n    return\n\ndef onCook(scriptOp):\n"
        "    import runtime\n    runtime.cook_sop(scriptOp, %r)\n    return\n" % kind))
    sop = geo.create(scriptSOP, "sop")
    sop.par.callbacks = cb
    sop.display = sop.render = True
    if mat is not None:
        geo.par.material = mat
    return geo, sop


def line_mat(parent, name, rgb, width=2.0, alpha=1.0):
    m = fresh(parent, lineMAT, name)
    if alpha < 1.0:
        try:
            m.par.blending = True
        except Exception as e:
            warn(f"{name}.blending: {e}")
    for nm, v in (("widthnear", width), ("widthfar", width), ("linenearcolorr", rgb[0]), ("linenearcolorg", rgb[1]),
                  ("linenearcolorb", rgb[2]), ("linenearalpha", alpha), ("linefarcolorr", rgb[0]),
                  ("linefarcolorg", rgb[1]), ("linefarcolorb", rgb[2]), ("linefaralpha", alpha)):
        try:
            setattr(m.par, nm, v)
        except Exception as e:
            warn(f"{name}.{nm}: {e}")
    return m


def pick_context():
    c = os.environ.get("WAYMO_CONTEXT_DIR")
    if c:
        return c
    ds = [os.path.dirname(p) for p in glob.glob(os.path.join(PROJECT_DIR, "data", "processed", "*", "meta.json"))]
    metas = {d: json.load(open(os.path.join(d, "meta.json"))) for d in ds}
    real = [d for d in ds if not metas[d].get("synthetic")]
    perception = [d for d in real if metas[d].get("kind", "perception") == "perception"]   # default source
    pool = perception or real or ds
    if not pool:
        raise FileNotFoundError("no processed context under data/processed - run preprocessing/export_frames.py")
    return max(pool, key=os.path.getmtime)


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def build():
    for p in (TD_DIR, os.path.join(PROJECT_DIR, "preprocessing"), os.path.join(PROJECT_DIR, "datasets")):
        if p not in sys.path:
            sys.path.insert(0, p)
    for m in ("frame_loader", "timeline_controller", "camera_controller", "runtime", "ego_motion", "ui_controller",
              "ui_panel", "export_controller"):
        if m in sys.modules:
            del sys.modules[m]
    import frame_loader as fl

    ctx_dir = pick_context()
    meta = fl.load(ctx_dir)
    say(f"context {meta['context']}  frames {meta['frames']}  synthetic={meta.get('synthetic')}")

    proj = op("/project1") or root.create(baseCOMP, "project1")
    rv = fresh(proj, baseCOMP, "WAYMO_AUTONOMOUS_VISION_4D")
    rv.store("project_dir", PROJECT_DIR)
    rv.store("context_dir", ctx_dir)
    P = lambda rel: f"{ROOT_PATH}/{rel}"
    CH = lambda n: f"op('{P('UI/controls')}')['{n}'].eval()"
    sec = {n: fresh(rv, baseCOMP, n) for n in ("DATA_INPUT", "CAMERA_SYSTEM", "LIDAR_RENDERER", "RGB_POINTCLOUD",
                                               "OBJECT_TRACKING", "TEMPORAL_4D", "VIRTUAL_CAMERAS", "UI", "OUTPUT")}

    # ------------------------------------------------------------------ UI / controls (created first: others read it)
    sys.path.insert(0, TD_DIR)
    import runtime as rt
    ui = sec["UI"]
    ctrl = fresh(ui, constantCHOP, "controls")
    names = rt.CTRL_NAMES
    try:
        ctrl.seq.const.numBlocks = len(names)
        for i, nm in enumerate(names):
            ctrl.seq.const[i].par.name = nm
            ctrl.seq.const[i].par.value = float(rt.CTRL_DEFAULTS[nm])
    except Exception as e:
        warn(f"controls: {e}")
    hud = fresh(ui, textTOP, "hud_text")
    setp(hud, outputresolution="custom", resolutionw=OUT_W, resolutionh=OUT_H, fontsizexunit="pixels",
         positionunit="pixels")
    setp(hud, fontsizex=15, positionx=352, positiony=-14, alignx="left", aligny="top", fontcolorr=0.0,
         fontcolorg=1.0, fontcolorb=0.9, bgalpha=0.0, text="WAYMO AUTONOMOUS VISION 4D")

    # ------------------------------------------------------------------ DATA_INPUT
    di = sec["DATA_INPUT"]
    text_dat(di, "waymo_loader", os.path.join(TD_DIR, "frame_loader.py"))
    text_dat(di, "timeline_controller", os.path.join(TD_DIR, "timeline_controller.py"))
    ep = fresh(di, tableDAT, "ego_pose")
    ep.appendRow(["frame", "ts_micros", "x", "y", "z"])
    for i, (ts, T) in enumerate(zip(meta["timestamps_micros"], meta["poses_world_vehicle"])):
        ep.appendRow([i, ts, T[0][3], T[1][3], T[2][3]])
    oa = fresh(di, tableDAT, "object_annotations")
    oa.appendRow(["frame", "track", "type", "x", "y", "z", "l", "w", "h", "heading", "speed_mps"])
    for i, rows in enumerate(fl.S["boxes"]):
        for r in rows:
            oa.appendRow([i, int(r[8]), int(r[7])] + [round(float(v), 3) for v in (r[0], r[1], r[2], r[3], r[4], r[5], r[6])]
                         + [round(float(r[9]), 3) if r[9] >= 0 else "n/a"])
    cf = fresh(di, tableDAT, "camera_frames")
    cf.appendRow(["frame", "camera_id", "file"])
    for i, d in enumerate(meta["cam_files"]):
        for k, v in d.items():
            cf.appendRow([i, k, v])
    lf = fresh(di, tableDAT, "lidar_frames")
    lf.appendRow(["frame", "ts_micros", "points_stored"])
    for i in range(meta["frames"]):
        lf.appendRow([i, meta["timestamps_micros"][i], int(len(fl.frame(i)["pts"]))])
    cal = fresh(di, tableDAT, "calibration")
    cal.appendRow(["note"])
    cal.appendRow(["calibration is applied during preprocessing (preprocessing/calibration.py); processed frames are "
                   "already in the world frame"])

    # ------------------------------------------------------------------ LIDAR_RENDERER
    lr = sec["LIDAR_RENDERER"]
    for nm, kind in (("position_texture", "cur_pos"), ("attribute_texture", "cur_attr")):
        script_top(lr, nm, kind)
    rg = sec["RGB_POINTCLOUD"]
    script_top(rg, "rgb_colors", "cur_rgb")
    nulltop = lambda parent, name, src: (lambda n: (setattr(n.par, "top", src), n)[1])(fresh(parent, selectTOP, name))
    nulltop(rg, "rgb_positions", lr.op("position_texture"))
    lmat = make_mat(lr, "lidar_shader", "lidar.vert", "lidar.frag")
    rmat = make_mat(rg, "rgb_shader", "lidar.vert", "rgb_points.frag")
    tp = sec["TEMPORAL_4D"]
    for nm, kind in (("history_positions", "hist_pos"), ("history_attributes", "hist_attr"), ("history_colors", "hist_rgb")):
        script_top(tp, nm, kind)
    tmat = make_mat(tp, "temporal_shader", "lidar.vert", "temporal.frag")
    cam_path = P("VIRTUAL_CAMERAS/main_cam")
    for mat, kind in ((lmat, "cur"), (rmat, "cur"), (tmat, "hist")):
        mat.par.blending = True
        try:
            if mat is tmat:                                             # history trails are emissive: additive
                mat.par.srcblend, mat.par.destblend = "one", "one"
            else:
                mat.par.srcblend, mat.par.destblend = "sa", "omsa"      # true alpha blending
        except Exception as e:
            warn(f"{mat.name}: blend: {e}")
        mat.par.depthwriting = (mat is not tmat)                  # sprites are discarded outside the disc; sky must never paint over them
        if kind == "cur":
            set_sampler(mat, 0, "sPos", lr.op("position_texture"))
            set_sampler(mat, 1, "sAttr", lr.op("attribute_texture"))
            set_sampler(mat, 2, "sRgb", rg.op("rgb_colors"))
        else:
            set_sampler(mat, 0, "sPos", tp.op("history_positions"))
            set_sampler(mat, 1, "sAttr", tp.op("history_attributes"))
            set_sampler(mat, 2, "sRgb", tp.op("history_colors"))
        u = Uniforms(mat)
        u.f("uPointSize", CH("rgb_size") if mat is rmat else (f"0.7 * {CH('point_size')}" if mat is tmat
                                                              else CH("point_size")))
        u.f("uRefDist", 20.0)
        u.f("uMaxRange", CH("max_range"))
        u.f("uDensity", f"{CH('density')} * (0.4 if {CH('show_rgb')} > 0.5 else 1.0)" if mat is lmat else CH("density"))
        u.f("uAgeMax", 1.0e6 if mat is not tmat else CH("trail_decay"))
        u.f("uSensorMask", CH("sensor_mask"))
        u.f("uHeightLo", -2.0)
        u.f("uHeightHi", 8.0)
        u.v3("uCamPos", [f"op('{cam_path}').par.tx.eval()", f"op('{cam_path}').par.ty.eval()",
                         f"op('{cam_path}').par.tz.eval()"])
        u.f("uOpacity", (CH("opacity") if mat is lmat else 1.0) if mat is not tmat else 0.22)
        u.f("uCinematic", CH("cinematic"))
        u.f("uGain", 1.0)
        if mat is lmat:
            u.f("uColorMode", CH("color_mode"))
    lgeo = grid_points_geo(lr, "lidar_geometry", fl.TEX_W, fl.TEX_H)
    lgeo.par.material = lmat
    rgeo = grid_points_geo(rg, "rgb_geometry", fl.TEX_W, fl.TEX_H)
    rgeo.par.material = rmat
    hgeo = grid_points_geo(tp, "history_geometry", fl.TEX_W, fl.TEX_H)
    hgeo.par.material = tmat
    # the history layer has its own age (trail_decay); mark hist frag uniform uAgeMax via the same name

    # ------------------------------------------------------------------ GROUND (measured road surface)
    gr = fresh(rv, baseCOMP, "GROUND")
    script_top(gr, "ground_data", "ground_a")
    script_top(gr, "ground_color", "ground_b")
    gmat = make_mat(gr, "ground_shader", "ground.vert", "ground.frag")
    gmat.par.blending = False
    gmat.par.depthwriting = True
    set_sampler(gmat, 0, "sGround", gr.op("ground_data"))
    set_sampler(gmat, 1, "sColor", gr.op("ground_color"))
    gu = Uniforms(gmat)
    gu.f("uExtent", fl.GROUND_EXTENT)
    gu.f("uDrop", 0.20)               # road sits just below the LiDAR returns that measured it
    gu.f("uRgbMix", CH("ground_rgb"))
    gu.f("uMark", CH("ground_marks"))
    gu.f("uCinematic", CH("cinematic"))
    gu.f("uMaxDist", CH("max_range"))
    gu.v3("uCenter", ["op('%s').fetch('gcx', 0)" % gr.path, "op('%s').fetch('gcz', 0)" % gr.path, "0"])
    ggeo = grid_points_geo(gr, "ground_geometry", 2 * fl.GROUND_N, 2 * fl.GROUND_N)
    for c_ in list(ggeo.children):
        if c_.OPType == "convertSOP":
            c_.destroy()
    gg = ggeo.op("grid1")
    setp(gg, rows=2 * fl.GROUND_N, cols=2 * fl.GROUND_N, sizex=10.0, sizey=10.0, texture="rowcol")
    gg.display = gg.render = True
    ggeo.par.material = gmat
    ggeo.par.render.expr = CH("show_ground") + " > 0.5"
    ggeo.par.display.expr = CH("show_ground") + " > 0.5"

    # ------------------------------------------------------------------ OBJECT_TRACKING
    ot = sec["OBJECT_TRACKING"]
    cols = {"box_vehicle": hexrgb("#24FF60"), "box_vehicle_static": (0.06, 0.42, 0.26),
            "box_pedestrian": hexrgb("#FFCC00"), "box_cyclist": hexrgb("#00D8FF"), "box_selected": (1.0, 1.0, 1.0)}
    geos = [lgeo, rgeo, hgeo, ggeo]
    for k, c in cols.items():
        g, _ = script_sop_geo(ot, k, k, line_mat(ot, k + "_mat", c, 2.5))
        geos.append(g)
    ego, _ = script_sop_geo(ot, "ego_vehicle", "ego", line_mat(ot, "ego_mat", (1.0, 1.0, 1.0), 3.0))
    grid, _ = script_sop_geo(ot, "ground_grid", "grid", line_mat(ot, "grid_mat", (0.06, 0.16, 0.22), 1.0, 0.55))
    geos += [ego, grid]
    for b, c in enumerate(((0.0, 0.9, 1.0), (0.62, 0.2, 1.0), (0.8, 0.8, 0.95))):
        g, _ = script_sop_geo(ot, f"motion_paths_{b}", f"trail{b}", line_mat(ot, f"trail_mat_{b}", c, 3.0 - b * 0.6))
        geos.append(g)
    extra_show = {}
    for g_, k_, c_, w_ in (("map_lanes", "map0", (0.05, 0.35, 0.5), 1.0), ("map_road_edges", "map1", (1.0, 0.15, 0.55), 2.0),
                          ("map_road_lines", "map2", (0.85, 0.9, 1.0), 1.5), ("map_crossings", "map3", (1.0, 0.75, 0.0), 1.5),
                          ("map_stop_signs", "map4", (1.0, 0.1, 0.1), 2.0)):
        g, _ = script_sop_geo(ot, g_, k_, line_mat(ot, g_ + "_mat", c_, w_))
        geos.append(g)
        extra_show[g_] = "show_map"
    g, _ = script_sop_geo(ot, "future_paths", "future", line_mat(ot, "future_mat", (1.0, 0.3, 1.0), 4.0))
    geos.append(g)
    extra_show["future_paths"] = "show_future"
    for g_, k_, c_, w_ in (("e2e_ego_past", "e2e0", (0.0, 0.9, 1.0), 5.0), ("e2e_ego_future", "e2e1", (1.0, 1.0, 1.0), 5.0),
                          ("e2e_alternatives", "e2e2", (0.3, 1.0, 0.4), 3.0)):
        g, _ = script_sop_geo(ot, g_, k_, line_mat(ot, g_ + "_mat", c_, w_))
        geos.append(g)
        extra_show[g_] = "show_future"
    echo, _ = script_sop_geo(tp, "motion_echo", "echo", line_mat(tp, "echo_mat", (0.55, 0.25, 1.0), 1.5, 0.7))
    geos.append(echo)
    show = {"lidar_geometry": "show_lidar", "rgb_geometry": "show_rgb", "history_geometry": "show_history",
            "box_vehicle": "show_boxes", "box_vehicle_static": "show_boxes", "box_pedestrian": "show_boxes",
            "box_cyclist": "show_boxes", "box_selected": "show_boxes", "ego_vehicle": "show_ego",
            "ground_grid": "show_grid", "motion_paths_0": "show_trails", "motion_paths_1": "show_trails",
            "motion_paths_2": "show_trails"}
    show.update(extra_show)
    for g in geos:
        n = show.get(g.name)
        if n:
            try:
                g.par.render.expr = CH(n) + " > 0.5"
                g.par.display.expr = CH(n) + " > 0.5"
            except Exception as e:
                warn(f"{g.name}: visibility expr: {e}")

    # ------------------------------------------------------------------ MODELS (instanced road-user meshes)
    mdl = fresh(ot, baseCOMP, "MODELS")
    mdir = os.path.join(PROJECT_DIR, "assets", "models")
    light = fresh(mdl, lightCOMP, "key_light")
    setp(light, lighttype="distant", rx=-52, ry=-28, dimmer=1.15)
    glowy = {"head": 1.0, "tail": 1.0, "accent": 0.9, "lidar": 0.35, "vest": 0.45, "bike": 0.5,
              "signface": 0.75}
    for mname in rt.MODELS:
        jm = json.load(open(os.path.join(mdir, mname + ".json")))
        cb = text_dat(mdl, f"inst_{mname}_cb", text=(
            "def onSetupParameters(scriptOp):\n    return\n\ndef onCook(scriptOp):\n"
            "    import runtime\n    runtime.cook_chop(scriptOp, %r)\n    return\n" % mname))
        ch = fresh(mdl, scriptCHOP, "inst_" + mname)
        ch.par.callbacks = cb
        for mat, col in jm["parts"].items():
            g = fresh(mdl, geometryCOMP, f"m_{mname}_{mat}")
            for c_ in list(g.children):
                c_.destroy()
            f_ = g.create(fileSOP, "obj")
            f_.par.file = os.path.join(mdir, f"{mname}__{mat}.obj")
            f_.display = f_.render = True
            pm = fresh(mdl, phongMAT, f"mat_{mname}_{mat}")
            e = glowy.get(mat, 0.0)
            setp(pm, diffr=col[0], diffg=col[1], diffb=col[2], ambr=col[0] * 0.55, ambg=col[1] * 0.55,
                 ambb=col[2] * 0.55, specr=0.55, specg=0.55, specb=0.6, shininess=40 if mat in ("body", "glass") else 12,
                 emitr=col[0] * e, emitg=col[1] * e, emitb=col[2] * e)
            g.par.material = pm
            setp(g, instancing=True, instanceop=ch, instancetx="tx", instancety="ty", instancetz="tz",
                 instancerx="rx", instancery="ry", instancerz="rz", instancesx="sx", instancesy="sy", instancesz="sz")
            try:
                g.par.render.expr = CH("show_models") + " > 0.5"
                g.par.display.expr = CH("show_models") + " > 0.5"
            except Exception as e_:
                warn(f"{g.name}: visibility: {e_}")
            geos.append(g)

    # ------------------------------------------------------------------ SKY dome (follows the camera)
    skyc = fresh(rv, baseCOMP, "SKY")
    skymat = make_mat(skyc, "sky_shader", "sky.vert", "sky.frag")
    skymat.par.blending = False
    skymat.par.depthwriting = True
    skygeo = fresh(skyc, geometryCOMP, "sky_dome")
    for c_ in list(skygeo.children):
        c_.destroy()
    sp = skygeo.create(sphereSOP, "sphere1")
    setp(sp, radx=300.0, rady=300.0, radz=300.0, rows=24, cols=48)
    sp.display = sp.render = True
    skygeo.par.material = skymat
    setp(skygeo, sx=-1.0, sy=-1.0, sz=-1.0)                              # inside-out: visible from within
    for a_ in ("tx", "ty", "tz"):
        try:
            getattr(skygeo.par, a_).expr = f"op('{ROOT_PATH}/VIRTUAL_CAMERAS/main_cam').par.{a_}.eval()"
        except Exception as e:
            warn(f"sky.{a_}: {e}")
    geos.insert(0, skygeo)

    # ------------------------------------------------------------------ VIRTUAL_CAMERAS
    vc = sec["VIRTUAL_CAMERAS"]
    aim = fresh(vc, nullCOMP, "look_target")
    cam = fresh(vc, cameraCOMP, "main_cam")
    setp(cam, fov=62.0, near=0.2, far=400.0, lookat=aim)
    cam.par.tx, cam.par.ty, cam.par.tz = -14.0, 7.0, 0.0

    # ------------------------------------------------------------------ CAMERA_SYSTEM
    cs = sec["CAMERA_SYSTEM"]
    script_top(cs, "camera_mosaic", "mosaic")
    cs_front = fresh(cs, cropTOP, "front_camera")
    try:
        cs_front.inputConnectors[0].connect(cs.op("camera_mosaic"))
        setp(cs_front, cropleft=0.3333, cropright=0.6666, cropbottom=0.5, croptop=1.0)
    except Exception as e:
        warn(f"front_camera crop: {e}")

    # ------------------------------------------------------------------ OUTPUT
    out = sec["OUTPUT"]
    rnd = fresh(out, renderTOP, "render_scene")
    setp(rnd, outputresolution="custom", resolutionw=OUT_W, resolutionh=OUT_H, camera=cam,
         geometry=" ".join(g.path for g in geos), bgcolorr=0.0, bgcolorg=0.0, bgcolorb=0.0, bgcolora=1.0)
    try:
        rnd.par.lights = light.path
    except Exception as e:
        warn(f"render lights: {e}")
    blur = fresh(out, blurTOP, "glow_blur")
    blur.inputConnectors[0].connect(rnd)
    setp(blur, size=14)
    gpix = text_dat(out, "glow_pixel", os.path.join(SHADER_DIR, "glow.frag"))
    glow = fresh(out, glslTOP, "glow")
    glow.inputConnectors[0].connect(rnd)
    glow.inputConnectors[1].connect(blur)
    setp(glow, pixeldat=gpix, outputresolution="custom", resolutionw=OUT_W, resolutionh=OUT_H)
    gu = Uniforms(glow)
    gu.f("uGlow", f"{CH('glow')} * (0.5 + {CH('cinematic')})")
    gu.f("uVignette", CH("vignette"))
    msel = fresh(out, selectTOP, "mosaic_select")     # cross-COMP wires silently fail: use a Select TOP
    msel.par.top = cs.op("camera_mosaic")
    tr = fresh(out, transformTOP, "mosaic_inset")
    tr.inputConnectors[0].connect(msel)
    setp(tr, outputresolution="custom", resolutionw=OUT_W, resolutionh=OUT_H, sx=0.3, sy=0.3, tunit="fraction")
    # inset: 0.3 = corner thumbnail, 1.0 = full frame (scene 1 of the film); show_camera=0 hides it
    for par_, ex_ in (("sx", f"{CH('inset_scale')} * {CH('show_camera')}"), ("sy", f"{CH('inset_scale')} * {CH('show_camera')}"),
                      ("tx", f"0.33 * (1 - ({CH('inset_scale')} - 0.3) / 0.7)"),
                      ("ty", f"-0.33 * (1 - ({CH('inset_scale')} - 0.3) / 0.7)")):
        try:
            getattr(tr.par, par_).expr = ex_
        except Exception as e:
            warn(f"mosaic_inset.{par_}: {e}")
    title = fresh(ui, textTOP, "title_text")
    setp(title, outputresolution="custom", resolutionw=OUT_W, resolutionh=OUT_H, fontsizexunit="pixels",
         positionunit="pixels")
    setp(title, fontsizex=44, positionx=0, positiony=-250, alignx="center", aligny="center", fontcolorr=0.0,
         fontcolorg=0.95, fontcolorb=1.0, bgalpha=0.0, text="")
    tsel = fresh(out, selectTOP, "title")
    tsel.par.top = title
    comp = fresh(out, compositeTOP, "composite")
    setp(comp, outputresolution="custom", resolutionw=OUT_W, resolutionh=OUT_H)
    hudsel = fresh(out, selectTOP, "hud")
    hudsel.par.top = hud
    try:                                            # input 0 is ON TOP for 'over'
        comp.inputConnectors[0].connect(hudsel)
        comp.inputConnectors[0].connect(hudsel)
        comp.inputConnectors[1].connect(tsel)
        comp.inputConnectors[2].connect(tr)
        comp.inputConnectors[3].connect(glow)
        comp.par.operand = "over"
    except Exception as e:
        warn(f"composite: {e}")
    fin = fresh(out, nullTOP, "final")
    fin.inputConnectors[0].connect(comp)
    mo = fresh(out, moviefileoutTOP, "movie_out")
    mo.inputConnectors[0].connect(fin)
    setp(mo, file=os.path.join(PROJECT_DIR, "output", "videos", "waymo_av4d.mp4"))

    # ------------------------------------------------------------------ control box (child of ui_root => visible in the window)
    import ui_panel as upl
    root_ui = fresh(ui, containerCOMP, "ui_root")
    setp(root_ui, w=OUT_W, h=OUT_H)
    try:
        root_ui.par.top = fin
    except Exception as e:
        warn(f"ui_root.top: {e}")
    panel = fresh(root_ui, containerCOMP, "panel")
    setp(panel, x=0, y=0, w=upl.PANEL_W, h=upl.PANEL_H, bgcolorr=0.015, bgcolorg=0.03, bgcolorb=0.05, bgalpha=0.80)
    try:
        panel.par.display.expr = CH("show_panel") + " > 0.5"
    except Exception as e:
        warn(f"panel display: {e}")
    n_w = 0
    for it in upl.layout():
        kind = it["kind"]
        nm = f"w{n_w:03d}"
        n_w += 1
        if kind in ("label", "text", "value"):
            t = panel.create(textCOMP, nm)
            setp(t, x=it["x"], y=it["y"], w=it["w"], h=it["h"], text=it["label"] if kind != "value" else "",
                 fontsize=11 if kind == "label" else 10, fontcolorr=1.0 if kind == "label" else 0.82,
                 fontcolorg=0.62 if kind == "label" else 0.88, fontcolorb=0.0 if kind == "label" else 0.92,
                 bgalpha=0.0)
            t.store("kind", kind)
            t.store("ctrl", it["label"] if kind == "value" else "")
        elif kind == "button":
            b = panel.create(buttonCOMP, nm)
            setp(b, buttontype="momentary", label=it["label"], x=it["x"], y=it["y"], w=it["w"], h=it["h"], fontsize=10,
                 colorr=upl.COL_OFF[0], colorg=upl.COL_OFF[1], colorb=upl.COL_OFF[2])
            b.store("kind", "button")
            b.store("act", it["act"])
        elif kind == "slider":
            sl = panel.create(sliderCOMP, nm)
            setp(sl, slidertype="slideru", x=it["x"], y=it["y"], w=it["w"], h=it["h"], colorr=0.95, colorg=0.55,
                 colorb=0.05, bgcolorr=0.10, bgcolorg=0.12, bgcolorb=0.15, bgalpha=1.0)
            sl.store("kind", "slider")
            sl.store("ctrl", it["label"])
            sl.store("act", it["act"])
    # small re-open button, visible only while the box is hidden
    hide = panel.create(buttonCOMP, "w_hide")
    setp(hide, buttontype="momentary", label="HIDE", x=upl.PANEL_W - 54, y=upl.PANEL_H - 22, w=48, h=18, fontsize=9)
    hide.store("kind", "button")
    hide.store("act", "T:show_panel")
    show = root_ui.create(buttonCOMP, "w_show")
    setp(show, buttontype="momentary", label="UI", x=6, y=OUT_H - 28, w=40, h=22, fontsize=10)
    show.store("kind", "button")
    show.store("act", "T:show_panel")
    try:
        show.par.display.expr = CH("show_panel") + " < 0.5"
    except Exception as e:
        warn(f"show button: {e}")
    pex = fresh(ui, panelexecuteDAT, "panel_exec")
    pex.text = PANEL_EXEC_SRC % TD_DIR
    setp(pex, panels=f"{panel.path}/*", offtoon=True, valuechange=True)
    sh = fresh(root_ui, panelexecuteDAT, "panel_exec_show")
    sh.text = SHOW_EXEC_SRC % TD_DIR
    setp(sh, panels=f"{show.path}", offtoon=True)
    # keyboard / mouse for the free-fly camera (W/A/S/D move, Q/E up/down, Shift fast, drag = look, wheel = zoom)
    kb = fresh(ui, keyboardinCHOP, "keys_in")
    setp(kb, keys="w a s d q e up down left right", modifiers="ignore")      # global: any focused TD window
    kbs = fresh(ui, keyboardinCHOP, "keys_shift")
    setp(kbs, keys="w a s d q e up down left right", modifiers="shift")
    mi = fresh(ui, mouseinCHOP, "mouse_in")
    setp(mi, output="normal", lbuttonname="lbutton", rbuttonname="rbutton", wheel="wheel")
    win = fresh(ui, windowCOMP, "perform_window")
    setp(win, winop=root_ui, winw=OUT_W, winh=OUT_H)

    # ------------------------------------------------------------------ frame driver
    ex = fresh(rv, executeDAT, "frame_exec")
    ex.text = ("import sys\n_p = r'%s'\nif _p not in sys.path: sys.path.insert(0, _p)\n"
               "_q = r'%s'\nif _q not in sys.path: sys.path.insert(0, _q)\n\n"
               "def onFrameEnd(frame):\n    import runtime\n    runtime.tick()\n    return\n" %
               (TD_DIR, os.path.join(PROJECT_DIR, "preprocessing")))
    setp(ex, active=True, frameend=True)
    fl.S["cur"] = 0
    importlib.reload(rt)
    import timeline_controller as tl_
    tl_.play()                                    # start moving as soon as the project is built / opened
    say("built; warnings: %d" % len(WARN))
    return rv


build()
for w in WARN:
    print("W:", w)
