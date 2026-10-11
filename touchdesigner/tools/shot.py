"""usage (via tdrun): set variables NAME, FRAME, MODE, CM(color mode), HIST before exec.  Two-phase: setup then save."""
import sys
for p in (r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\touchdesigner', r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\preprocessing'):
    if p not in sys.path:
        sys.path.insert(0, p)
import frame_loader as fl, runtime as rt, camera_controller as cc, timeline_controller as tl
R = '/project1/WAYMO_AUTONOMOUS_VISION_4D'
try:
    NAME
except NameError:
    NAME, FRAME, MODE, CM, HIST = 'shot', 10, 1, 0, 8
rt.set_ctrl('color_mode', CM)
rt.set_ctrl('hist_frames', HIST)
tl.seek(float(fl.S['t_rel'][FRAME]))
cc.set_mode(MODE)
cc.ST['init'] = False
for _ in range(3):
    rt.tick()
fin = op(R + '/OUTPUT/final')
fin.cook(force=True)
fin.save(r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\output\screenshots\%s.png' % NAME)
result = {'frame': fl.S['cur'], 'cams': cc.MODE_NAMES[cc.ST['mode']], 'img': NAME}
