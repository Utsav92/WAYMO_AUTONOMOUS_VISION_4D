import json, math, os, sys
import bpy
from mathutils import Vector
D = sys.argv[sys.argv.index("--") + 1]
OUTPNG = sys.argv[sys.argv.index("--") + 2]
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
x = 0.0
for name, dx in (("waymo_car", 0.0), ("car", 6.0), ("pedestrian", 10.5), ("cyclist", 13.0)):
    meta = json.load(open(os.path.join(D, name + ".json")))
    for mat, col in meta["parts"].items():
        bpy.ops.wm.obj_import(filepath=os.path.join(D, f"{name}__{mat}.obj"), forward_axis="NEGATIVE_Z", up_axis="Y")
        ob = bpy.context.selected_objects[0]
        ob.location.x += dx
        ob.color = (*col, 1.0)
scn = bpy.context.scene
scn.render.engine = "BLENDER_WORKBENCH"
sh = scn.display.shading
sh.light = "STUDIO"
sh.color_type = "OBJECT"
scn.render.resolution_x, scn.render.resolution_y = 1500, 560
cam = bpy.data.objects.new("c", bpy.data.cameras.new("c"))
scn.collection.objects.link(cam)
cam.location = (5.0, -21.0, 6.5)
direction = Vector((5.5, 0.0, 0.9)) - cam.location
cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
cam.data.lens = 40
scn.camera = cam
scn.world = bpy.data.worlds.new("w")
scn.world.color = (0.02, 0.03, 0.05)
scn.render.filepath = OUTPNG
bpy.ops.render.render(write_still=True)
