import sys
sys.path[:0] = [r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\touchdesigner', r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\preprocessing']
import os
import frame_loader as fl, runtime as rt, camera_controller as cc, timeline_controller as tl
R = '/project1/WAYMO_AUTONOMOUS_VISION_4D'
out = []
want = ('1020365', 'MOTION_308716', 'MOTION_a3c1aa', 'E2E_11d68b', 'E2E_7ad8f3')
for ctx in rt.contexts():
    b = os.path.basename(ctx)
    if not any(b.startswith(w) for w in want):
        continue
    kind = rt.use_context(ctx)
    tl.pause()
    mid = {'perception': 70, 'motion': 14, 'e2e': 16}[kind]
    tl.seek(float(fl.S['t_rel'][mid]))
    cc.set_mode(1 if kind != 'motion' else 2)
    rt.set_ctrl('orbit_radius', 55 if kind == 'motion' else 24)
    rt.set_ctrl('show_camera', 0)
    cc.ST['init'] = False
    for _ in range(4):
        rt.tick()
    f = op(R + '/OUTPUT/final'); f.cook(force=True)
    f.save(r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\output\screenshots\src_%s.png' % b[:14])
    out.append(b[:14])
result = out
