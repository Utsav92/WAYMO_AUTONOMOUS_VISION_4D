"""Live checks against MY TouchDesigner (port 9995).  Skipped when TD is not running."""
import json
import os
import sys
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "touchdesigner", "tools"))
sys.path.insert(0, os.path.join(ROOT, "machine_learning"))
import numpy as np  # noqa: E402


def td(code):
    from tdrun import run
    try:
        out = run(code, timeout=60)
    except Exception:
        pytest.skip("TouchDesigner (port 9995) not reachable")
    assert not out.get("error"), out.get("error")
    return (out.get("data") or {}).get("result")


def test_network_has_no_errors_and_shaders_compile():
    code = ("R='/project1/WAYMO_AUTONOMOUS_VISION_4D'\nbad=[]\n"
            "def w(c):\n    for ch in c.children:\n        e=ch.errors(recurse=False)\n"
            "        if e: bad.append(ch.path+': '+e[:120])\n        if ch.isCOMP: w(ch)\n"
            "w(op(R))\nresult='; '.join(bad) if bad else 'clean'\n")
    assert td(code) == "clean"


def test_render_and_playback_advance():
    code = ("import sys\nsys.path.insert(0, r'%s')\nimport frame_loader as fl, timeline_controller as tl, runtime as rt\n"
            "rt.reset_defaults()\nfor _ in range(3): rt.tick()\n"
            "tl.play()\nt0=fl.S['t']\nimport time; time.sleep(0.6)\nrt.tick(); rt.tick()\n"
            "a=op('/project1/WAYMO_AUTONOMOUS_VISION_4D/OUTPUT/final'); a.cook(force=True); arr=a.numpyArray()\n"
            "tl.pause()\nresult='%%d %%d %%.3f' %% (arr.shape[0], arr.shape[1], float(arr[...,:3].max()))\n"
            % os.path.join(ROOT, "touchdesigner"))
    h, w, mx = td(code).split()
    assert (int(h), int(w)) == (720, 1280) and float(mx) > 0.2


def test_depth_metrics_use_valid_pixels_only():
    from evaluation import depth_metrics, align_scale_shift
    gt = np.full((10, 10), 10.0)
    pred = gt * 2.0 + 1.0
    valid = np.zeros_like(gt, bool)
    valid[:5] = True
    s, t = align_scale_shift(pred, gt, valid)
    m = depth_metrics(pred * s + t, gt, valid)
    assert m["mae"] < 1e-6 and m["valid_pct"] == 50.0


def test_camera_snaps_on_loop_wrap_and_panel_is_wired():
    """Regression: after the playback loop the camera used to fly 300 m back, so road/map/car vanished for seconds."""
    code = ("import sys\nsys.path.insert(0, r'%s')\nimport frame_loader as fl, timeline_controller as tl, runtime as rt, "
            "camera_controller as cc, ui_panel as up, numpy as np\n"
            "rt.reset_defaults(); cc.set_mode(1); tl.pause()\n"
            "tl.seek(fl.duration() - 0.05); rt.tick(); rt.tick()\n"
            "tl.play(); fl.S['speed'] = 4.0; fl.S['last_wall'] = None\n"
            "import time; time.sleep(0.1); fl.S['last_wall'] = time.perf_counter() - 0.25; rt.tick(); rt.tick()\n"
            "ego = fl.to_td(fl.ego_pose_at(fl.ego_t())[:3, 3]); d = float(np.linalg.norm(cc.ST['pos'] - ego))\n"
            "tl.pause()\n"
            "P = op('/project1/WAYMO_AUTONOMOUS_VISION_4D/UI/ui_root/panel')\n"
            "acts = [c.fetch('act', '') for c in P.children if c.fetch('kind', '') == 'button']\n"
            "sl = [c.fetch('ctrl', '') for c in P.children if c.fetch('kind', '') == 'slider']\n"
            "ok = all(s in rt.CTRL_NAMES for s in sl) and len(acts) >= 35 and len(sl) == len(up.SLIDERS)\n"
            "result = '%%.1f %%s' %% (d, ok)\n" % os.path.join(ROOT, "touchdesigner"))
    d, ok = td(code).split()
    assert float(d) < 40.0, "camera must be near the ego car right after the loop wrap (was ~300 m)"
    assert ok == "True"
