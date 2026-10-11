"""Headless Blender script: procedural road-user models -> OBJ per material (TouchDesigner axes: x forward, y up).

    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" --background --python make_models.py -- OUT_DIR

These are my own procedural models in the style of a white SUV robotaxi (roof lidar dome, side sensor pods, teal
accent).  They are NOT Waymo's proprietary assets.
"""
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(os.getcwd(), "models")
os.makedirs(OUT, exist_ok=True)

COL = {  # name -> (rgb, comment)
    "body": (0.93, 0.95, 0.97), "body_dark": (0.55, 0.60, 0.66), "glass": (0.03, 0.05, 0.08),
    "tire": (0.02, 0.02, 0.025), "hub": (0.55, 0.57, 0.6), "trim": (0.06, 0.07, 0.08),
    "lidar": (0.04, 0.10, 0.14), "accent": (0.0, 0.76, 0.72), "head": (0.85, 0.95, 1.0), "tail": (1.0, 0.05, 0.05),
    "vest": (1.0, 0.72, 0.0), "cloth": (0.08, 0.09, 0.12), "skin": (0.75, 0.55, 0.42), "bike": (0.0, 0.85, 1.0),
    "cargo": (0.82, 0.84, 0.88), "signface": (1.0, 0.27, 0.83), "pole": (0.62, 0.66, 0.72),
}


def new_obj(name):
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


class Part:
    """bmesh accumulator keyed by material name."""

    def __init__(self):
        self.bm = {}

    def _bm(self, mat):
        return self.bm.setdefault(mat, bmesh.new())

    def box(self, mat, c, s, rot_z=0.0):
        b = self._bm(mat)
        m = Matrix.Translation(c) @ Matrix.Rotation(rot_z, 4, "Z") @ Matrix.Diagonal((s[0], s[1], s[2], 1.0))
        bmesh.ops.create_cube(b, size=1.0, matrix=m)

    def cyl(self, mat, c, r, h, axis="Z", seg=20, r2=None):
        b = self._bm(mat)
        rot = {"Z": Matrix.Identity(4), "X": Matrix.Rotation(math.pi / 2, 4, "Y"),
               "Y": Matrix.Rotation(math.pi / 2, 4, "X")}[axis]
        m = Matrix.Translation(c) @ rot
        bmesh.ops.create_cone(b, cap_ends=True, cap_tris=False, segments=seg, radius1=r, radius2=r if r2 is None else r2,
                              depth=h, matrix=m)

    def sphere(self, mat, c, r, seg=16):
        b = self._bm(mat)
        bmesh.ops.create_uvsphere(b, u_segments=seg, v_segments=max(seg // 2, 6), radius=r, matrix=Matrix.Translation(c))

    def torus_ring(self, mat, c, R, r, axis="Y", seg=28, tube=8):
        b = self._bm(mat)
        verts = []
        rot = {"Y": Matrix.Rotation(math.pi / 2, 4, "X"), "X": Matrix.Rotation(math.pi / 2, 4, "Y"),
               "Z": Matrix.Identity(4)}[axis]
        M = Matrix.Translation(c) @ rot
        rings = []
        for i in range(seg):
            a = 2 * math.pi * i / seg
            ring = []
            for j in range(tube):
                bj = 2 * math.pi * j / tube
                p = Vector(((R + r * math.cos(bj)) * math.cos(a), (R + r * math.cos(bj)) * math.sin(a), r * math.sin(bj)))
                ring.append(b.verts.new(M @ p))
            rings.append(ring)
        for i in range(seg):
            for j in range(tube):
                try:
                    b.faces.new((rings[i][j], rings[(i + 1) % seg][j], rings[(i + 1) % seg][(j + 1) % tube],
                                 rings[i][(j + 1) % tube]))
                except ValueError:
                    pass

    def loft(self, mat_side, mat_top, stations, y_scale=1.0, cap=True):
        """stations: list of (x, half_width, z_bottom, z_top).  8-point ring per station.  Side faces use mat_side,
        top face strip uses mat_top."""
        sm = self._bm(mat_side)
        tm = self._bm(mat_top) if mat_top != mat_side else sm
        rings = []
        for (x, hw, zb, zt) in stations:
            hw *= y_scale
            h = zt - zb
            pts = [(-hw * 0.82, zb), (hw * 0.82, zb), (hw, zb + 0.18 * h), (hw, zt - 0.18 * h), (hw * 0.80, zt),
                   (-hw * 0.80, zt), (-hw, zt - 0.18 * h), (-hw, zb + 0.18 * h)]
            rings.append([(x, y, z) for y, z in pts])
        n = len(rings)

        def v(bm, p):
            return bm.verts.new(p)
        for i in range(n - 1):
            for k in range(8):
                k2 = (k + 1) % 8
                quad = [rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]]
                bm = tm if k == 4 else sm
                try:
                    bm.faces.new([v(bm, p) for p in quad])
                except ValueError:
                    pass
        if cap:
            for ring in (rings[0], rings[-1]):
                try:
                    sm.faces.new([v(sm, p) for p in ring])
                except ValueError:
                    pass

    def build(self, name, subsurf=("body", "body_dark", "cargo")):
        objs = {}
        for mat, b in self.bm.items():
            bmesh.ops.remove_doubles(b, verts=b.verts[:], dist=1e-4)    # weld loft quads so subsurf is smooth
            bmesh.ops.recalc_face_normals(b, faces=b.faces[:])
            me = bpy.data.meshes.new(f"{name}_{mat}")
            b.to_mesh(me)
            b.free()
            ob = bpy.data.objects.new(f"{name}_{mat}", me)
            bpy.context.scene.collection.objects.link(ob)
            for p in me.polygons:
                p.use_smooth = mat in subsurf or mat in ("lidar", "tire", "hub", "vest", "cloth", "skin", "bike", "head")
            if mat in subsurf:
                ob.modifiers.new("ss", "SUBSURF").levels = 2
            objs[mat] = ob
        return objs


def export(name, objs, dims):
    for mat, ob in objs.items():
        bpy.ops.object.select_all(action="DESELECT")
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        path = os.path.join(OUT, f"{name}__{mat}.obj")
        bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True, forward_axis="NEGATIVE_Z", up_axis="Y",    # Blender (x,y,z) -> OBJ (x, z, -y): car front stays +X
                              export_materials=False, apply_modifiers=True, export_uv=False)
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"dims": dims, "parts": {m: COL.get(m, (1, 1, 1)) for m in objs}}, f)


def wheels(p, xs, y, r=0.37, w=0.26):
    for x in xs:
        for s in (-1, 1):
            p.cyl("tire", (x, s * y, r), r, w, axis="Y", seg=24)
            p.cyl("hub", (x, s * (y + w * 0.5 + 0.005), r), r * 0.6, 0.02, axis="Y", seg=16)


def car(name, waymo):
    p = Part()
    lower = [(-2.35, 0.70, 0.50, 0.92), (-2.22, 0.88, 0.32, 1.00), (-1.9, 0.96, 0.22, 1.02), (-1.0, 0.98, 0.20, 1.00),
             (0.0, 0.98, 0.20, 0.99), (1.0, 0.98, 0.20, 0.98), (1.8, 0.94, 0.22, 0.92), (2.2, 0.86, 0.30, 0.80),
             (2.35, 0.70, 0.44, 0.68)]
    p.loft("body" if waymo else "body_dark", "body" if waymo else "body_dark", lower)
    cabin = [(-1.62, 0.64, 0.98, 1.28), (-1.15, 0.78, 0.98, 1.50), (-0.2, 0.82, 0.98, 1.57), (0.75, 0.78, 0.98, 1.50),
             (1.20, 0.68, 0.98, 1.20)]
    p.loft("glass", "body" if waymo else "body_dark", cabin)
    wheels(p, (-1.42, 1.42), 0.80)
    p.box("trim", (0.0, 0.0, 0.17), (4.3, 1.5, 0.08))                           # under-body / sill
    p.box("head", (2.28, 0.62, 0.78), (0.10, 0.34, 0.07))
    p.box("head", (2.28, -0.62, 0.78), (0.10, 0.34, 0.07))
    p.box("tail", (-2.31, 0.0, 0.90), (0.06, 1.46, 0.06))
    if waymo:
        for s_ in (-1, 1):                                                      # teal rocker stripes (sides only)
            p.box("accent", (0.0, s_ * 0.985, 0.40), (3.5, 0.02, 0.07))
        p.cyl("trim", (0.15, 0.0, 1.60), 0.22, 0.08, seg=28)                    # sensor pod base
        p.cyl("lidar", (0.15, 0.0, 1.78), 0.17, 0.30, seg=28)                   # spinning lidar dome
        p.cyl("lidar", (0.15, 0.0, 1.96), 0.14, 0.06, seg=28, r2=0.06)
        p.cyl("accent", (0.15, 0.0, 1.80), 0.178, 0.025, seg=28)                # glowing ring
        for s in (-1, 1):
            p.cyl("lidar", (1.02, s * 1.02, 1.02), 0.065, 0.14, seg=14)         # side perimeter sensors
            p.cyl("lidar", (-1.5, s * 0.74, 1.26), 0.05, 0.10, seg=12)
            p.box("lidar", (2.30, s * 0.30, 0.55), (0.06, 0.22, 0.10))
    else:
        p.box("trim", (2.32, 0.0, 0.52), (0.08, 1.1, 0.12))                     # grille
    objs = p.build(name)
    export(name, objs, {"L": 4.70, "W": 2.0, "H": 1.96 if waymo else 1.57})


def truck(name):
    p = Part()
    p.loft("body_dark", "body_dark", [(-4.0, 1.25, 0.55, 3.2), (-3.9, 1.25, 0.50, 3.25), (1.5, 1.25, 0.50, 3.25),
                                      (1.55, 1.2, 0.5, 3.2)], cap=True)
    p.loft("cargo", "cargo", [(-3.9, 1.18, 0.62, 3.15), (1.45, 1.18, 0.62, 3.15)])
    p.loft("glass", "body_dark", [(1.6, 1.15, 1.3, 2.1), (2.35, 1.15, 1.3, 2.75), (3.0, 1.15, 1.3, 2.6)])
    p.loft("body_dark", "body_dark", [(1.6, 1.2, 0.5, 1.4), (3.6, 1.18, 0.55, 1.35), (4.0, 1.1, 0.7, 1.1)])
    for x in (-2.6, -1.3, 2.6):
        for s in (-1, 1):
            p.cyl("tire", (x, s * 1.05, 0.5), 0.5, 0.32, axis="Y", seg=24)
    p.box("head", (3.98, 0.8, 0.9), (0.1, 0.4, 0.12))
    p.box("head", (3.98, -0.8, 0.9), (0.1, 0.4, 0.12))
    p.box("tail", (-4.0, 0.0, 0.9), (0.06, 2.0, 0.1))
    export(name, p.build(name), {"L": 8.0, "W": 2.5, "H": 3.25})


def pedestrian(name):
    p = Part()
    for s in (-1, 1):
        p.cyl("cloth", (0.0, s * 0.10, 0.45), 0.075, 0.9, seg=10)               # legs
        p.cyl("vest", (0.0, s * 0.27, 1.22), 0.055, 0.62, seg=10)               # arms
    p.cyl("vest", (0.0, 0.0, 1.18), 0.19, 0.62, seg=14)                          # torso
    p.sphere("skin", (0.0, 0.0, 1.62), 0.115, seg=14)
    p.sphere("cloth", (0.0, 0.0, 1.67), 0.12, seg=12)                            # hair/cap
    export(name, p.build(name), {"L": 0.6, "W": 0.6, "H": 1.78})


def cyclist(name):
    p = Part()
    for x in (-0.55, 0.55):
        p.torus_ring("tire", (x, 0.0, 0.34), 0.32, 0.025, axis="Y")
    p.cyl("bike", (0.0, 0.0, 0.55), 0.02, 1.1, axis="X", seg=8)                 # top tube
    p.cyl("bike", (0.35, 0.0, 0.62), 0.02, 0.62, seg=8)                         # fork
    p.cyl("bike", (-0.15, 0.0, 0.58), 0.02, 0.5, seg=8)                         # seat tube
    p.cyl("bike", (0.35, 0.0, 0.98), 0.02, 0.5, axis="Y", seg=8)                # handlebar
    p.cyl("bike", (-0.18, 0.12, 0.62), 0.07, 0.62, seg=10)                       # near leg
    p.cyl("bike", (-0.18, -0.12, 0.62), 0.07, 0.62, seg=10)
    p.cyl("bike", (0.05, 0.0, 1.1), 0.17, 0.68, seg=14)                          # torso (leans below)
    p.sphere("skin", (0.18, 0.0, 1.55), 0.11, seg=12)
    p.sphere("bike", (0.18, 0.0, 1.60), 0.125, seg=12)                           # helmet
    export(name, p.build(name), {"L": 1.8, "W": 0.6, "H": 1.75})


def sign(name):
    """Origin = centre of the sign plate; the pole hangs down 2.4 m to the ground."""
    p = Part()
    p.box("signface", (0.0, 0.0, 0.0), (0.05, 0.62, 0.62))
    p.box("pole", (-0.045, 0.0, 0.0), (0.03, 0.66, 0.66))
    p.cyl("pole", (-0.06, 0.0, -1.45), 0.035, 2.5, seg=10)
    export(name, p.build(name), {"L": 0.2, "W": 0.62, "H": 0.62})


for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
sign("sign")
car("waymo_car", True)
car("car", False)
truck("truck")
pedestrian("pedestrian")
cyclist("cyclist")
print("MODELS_DONE", OUT)
