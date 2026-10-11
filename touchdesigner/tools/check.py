import json
R = '/project1/WAYMO_AUTONOMOUS_VISION_4D'
base = op(R)
errs, n = [], 0


def walk(c):
    global n
    for ch in c.children:
        n += 1
        try:
            e = ch.errors(recurse=False)
        except Exception:
            e = ''
        w = ''
        try:
            w = ch.warnings(recurse=False)
        except Exception:
            pass
        if e or w:
            errs.append((ch.path.replace(R, ''), (e or '')[:200], (w or '')[:160]))
        if ch.isCOMP:
            walk(ch)


walk(base)
res = {'ops': n, 'issues': errs[:40]}
for k in ('LIDAR_RENDERER/lidar_shader', 'RGB_POINTCLOUD/rgb_shader', 'TEMPORAL_4D/temporal_shader'):
    o = op(R + '/' + k)
    res[k] = {'compile': o.compileResult if hasattr(o, 'compileResult') else None}
res['cur'] = [op(R + '/OUTPUT/final').width, op(R + '/OUTPUT/final').height]
result = res
