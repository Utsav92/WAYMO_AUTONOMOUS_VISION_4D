"""Render the 60 s film: python export_cinematic.py [--frames 1800] [--scientific] [--name waymo_av4d] [--encode]
Frames are saved as PNG by TouchDesigner, then encoded with ffmpeg (H.264, yuv420p, 30 fps).
Resolution is whatever render_scene is set to (1280x720 under non-commercial TD)."""
import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdrun import run  # noqa: E402

PROJ = r"C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D"
TD = PROJ + r"\touchdesigner"
PRE = ("import sys, importlib\nfor _p in (r'%s', r'%s\\preprocessing'):\n    if _p not in sys.path: sys.path.insert(0, _p)\n"
       "import export_controller as ec\nimportlib.reload(ec)\n" % (TD, PROJ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1800)
    ap.add_argument("--scientific", action="store_true")
    ap.add_argument("--name", default="waymo_av4d")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--total", type=int, default=1800)
    a = ap.parse_args()
    fdir = os.path.join(PROJ, "output", "videos", "_frames_" + a.name)
    shutil.rmtree(fdir, ignore_errors=True)
    os.makedirs(fdir)
    for f in range(a.frames):
        code = PRE + ("info = ec.apply(%d, %d, %s)\nfin = op('/project1/WAYMO_AUTONOMOUS_VISION_4D/OUTPUT/final')\n"
                      "fin.cook(force=True)\nfin.save(r'%s\\f%04d.png')\nresult = info\n"
                      % (f, a.total, a.scientific, fdir, f))
        out = run(code, timeout=120)
        if out.get("error"):
            sys.exit("frame %d: %s" % (f, out["error"]))
        if f % 30 == 0:
            print("frame", f, out.get("data", {}).get("result"), flush=True)
    # leave the project in its normal interactive state (no full-screen camera, no film overrides)
    restore = ("import runtime as rt, frame_loader as fl\n"
               "fl.S.pop('dt_override', None)\n"
               "rt.use_context(fl.S['dir'])\n"
               "result = 'restored'\n")
    run(PRE + restore)
    if a.encode:
        mp4 = os.path.join(PROJ, "output", "videos", a.name + ".mp4")
        subprocess.check_call(["ffmpeg", "-y", "-framerate", "30", "-i", os.path.join(fdir, "f%04d.png"),
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", mp4])
        print("wrote", mp4)


if __name__ == "__main__":
    main()
