import sys
for p in (r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\touchdesigner', r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\preprocessing'):
    if p not in sys.path:
        sys.path.insert(0, p)
import importlib
import frame_loader as fl, runtime as rt, camera_controller as cc, timeline_controller as tl
R = '/project1/WAYMO_AUTONOMOUS_VISION_4D'
OUT = r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\output\screenshots\\'
fin = op(R + '/OUTPUT/final')


def shot(name, frame, mode, **ctl):
    for k, v in dict(color_mode=0, hist_frames=10, show_rgb=0, show_lidar=1, show_history=1, cinematic=0, glow=0.6,
                     echo=0, explode=0, show_trails=1).items():
        rt.set_ctrl(k, ctl.get(k, v))
    tl.seek(float(fl.S['t_rel'][frame]))
    cc.set_mode(mode)
    cc.ST['init'] = False
    for _ in range(3):
        rt.tick()
    fin.cook(force=True)
    fin.save(OUT + name + '.png')


shot('s_birdseye_rgb', 18, 2, color_mode=4, show_history=0)
shot('s_driver_range', 14, 0, color_mode=1, show_history=0)
shot('s_orbit_4d', 20, 4, hist_frames=14, echo=5, explode=0.8, cinematic=1, glow=1.0, color_mode=0)
result = 'ok'
