import sys, os
sys.path[:0] = [r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\touchdesigner', r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\preprocessing']
import frame_loader as fl, runtime as rt, camera_controller as cc, timeline_controller as tl
R = '/project1/WAYMO_AUTONOMOUS_VISION_4D'
OUT = r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\output\screenshots\audit_'
shots = [('perc', '1020365', 70, 1, 24, {}), ('perc_bird', '1020365', 120, 2, 50, {}), ('perc_driver', '1020365', 40, 0, 24, {}),
         ('motion', 'MOTION_308716', 6, 1, 24, {}), ('motion_bird', 'MOTION_a3c1aa', 40, 2, 60, {}),
         ('e2e', 'E2E_11d68b', 16, 1, 24, {}), ('e2e_driver', 'E2E_7ad8f3', 16, 0, 24, {})]
done = []
ctxs = rt.contexts()
for name, pref, fi, mode, rad, extra in shots:
    ctx = [c for c in ctxs if os.path.basename(c).startswith(pref)]
    if not ctx:
        continue
    rt.use_context(ctx[0]); tl.pause()
    rt.set_ctrl('show_hud', 0); rt.set_ctrl('show_camera', 0)
    tl.seek(float(fl.S['t_rel'][fi])); cc.set_mode(mode); cc.ST['init'] = False; rt.set_ctrl('orbit_radius', rad)
    for _ in range(5):
        rt.tick()
    f = op(R + '/OUTPUT/final'); f.cook(force=True)
    f.save(OUT + name + '.png'); done.append(name)
rt.use_context(ctxs[0])      # back to the normal interactive view
result = done
