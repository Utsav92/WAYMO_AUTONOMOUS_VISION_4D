import sys, traceback
sys.path[:0] = [r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\touchdesigner', r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\preprocessing']
import numpy as np
import frame_loader as fl, runtime as rt, camera_controller as cc, timeline_controller as tl
R = '/project1/WAYMO_AUTONOMOUS_VISION_4D'
rt.reset_defaults(); tl.pause()
bad, rows = [], []
for i in range(0, fl.n_frames(), 6):
    try:
        tl.seek(float(fl.S['t_rel'][i]))
        rt.tick()
        ch = op(R + '/OBJECT_TRACKING/MODELS/inst_waymo_car'); ch.cook(force=True)
        ego_inst = [round(float(ch['tx'][0]), 1), round(float(ch['tz'][0]), 1), round(float(ch['sx'][0]), 2)]
        a = fl.S['ground'][0] if fl.S.get('ground') else None
        cover = float((a[..., 2] == 1).mean()) if a is not None else -1
        cam = cc.ST['pos']
        ego = fl.to_td(fl.ego_pose_at(fl.ego_t())[:3, 3])
        rows.append((i, ego_inst, round(cover, 2), round(float(np.linalg.norm(cam - ego)), 1)))
    except Exception as e:
        bad.append((i, repr(e)[:120], traceback.format_exc()[-300:]))
result = {'errors': bad[:3], 'n_ok': len(rows), 'sample': [str(r) for r in rows[::5]],
          'min_cover': min(r[2] for r in rows), 'max_cam_dist': max(r[3] for r in rows)}
