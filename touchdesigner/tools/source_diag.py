import sys, traceback
sys.path[:0] = [r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\touchdesigner', r'C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\preprocessing']
import numpy as np
import frame_loader as fl, runtime as rt, camera_controller as cc, timeline_controller as tl
R = '/project1/WAYMO_AUTONOMOUS_VISION_4D'
rep = []
rt.reset_defaults()
for ctx in rt.contexts():
    name = __import__('os').path.basename(ctx)[:26]
    try:
        kind = rt.use_context(ctx)
        tl.play()
        for _ in range(4):
            rt.tick()
        fin = op(R + '/OUTPUT/final'); fin.cook(force=True)
        a = fin.numpyArray()[:, :, :3]
        nb = len(fl.S['boxes'][max(fl.S['cur'], 0)])
        chans = {}
        for m in rt.MODELS:
            ch = op(R + '/OBJECT_TRACKING/MODELS/inst_' + m); ch.cook(force=True)
            chans[m] = int((np.array(ch['sy'].vals) > 0).sum()) if ch.numChans else 0
        sop = lambda n: op(R + '/OBJECT_TRACKING/' + n).op('sop')
        lanes = sop('map_lanes').numPoints
        fut = sop('future_paths').numPrims
        c = rt.ctrl()
        rep.append('%-26s kind=%-10s boxes=%3d models=%s lanePts=%5d futurePaths=%2d e2ePrims=%d pts=%6d | nonblack=%.2f | vis: lidar=%d rgb=%d map=%d ground=%d boxes=%d models=%d' % (
            name, kind, nb, chans, lanes, fut, sop('e2e_ego_future').numPrims + sop('e2e_alternatives').numPrims,
            len(fl.frame(max(fl.S['cur'], 0))['pts']), float((a.mean(2) > 0.05).mean()), c['show_lidar'], c['show_rgb'],
            c['show_map'], c['show_ground'], c['show_boxes'], c['show_models']))
    except Exception as e:
        rep.append('%s FAILED: %s' % (name, traceback.format_exc()[-300:]))
result = rep
