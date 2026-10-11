import numpy as np

import calibration as cal
from rgb_fusion import colorize
from decode_camera import load_camera_calibration, frame_images, decode_jpeg
from decode_lidar import decode_frames, to_world
from camera_projection import overlay


def test_known_point_hits_principal_point(ctx):
    cam = load_camera_calibration(ctx["root"], ctx["name"], ctx["split"])[1]
    u, v, d, ok = cal.project_to_camera(np.array([[11.5, 0, 1.6]]), cam["T_vehicle_camera"], cam["intrinsic"],
                                        cam["width"], cam["height"])
    assert ok[0] and abs(u[0] - cam["width"] / 2) < 1e-6 and abs(v[0] - cam["height"] / 2) < 1e-6
    assert np.isclose(d[0], 10.0)


def test_left_is_smaller_u_and_behind_rejected(ctx):
    cam = load_camera_calibration(ctx["root"], ctx["name"], ctx["split"])[1]
    P = np.array([[11.5, 2.0, 1.6], [11.5, -2.0, 1.6], [-5.0, 0, 1.6]])   # +y is LEFT in Waymo
    u, v, d, ok = cal.project_to_camera(P, cam["T_vehicle_camera"], cam["intrinsic"], cam["width"], cam["height"])
    assert u[0] < u[1]
    assert ok[0] and ok[1] and not ok[2]


def test_distortion_moves_points_outward():
    intr = dict(f_u=500, f_v=500, c_u=320, c_v=240, k1=0.1, k2=0, p1=0, p2=0, k3=0)
    p = np.array([[10.0, 3.0, 0.0]])
    u1, *_ = cal.project_to_camera(p, np.eye(4), intr, 640, 480)
    intr0 = dict(intr, k1=0.0)
    u0, *_ = cal.project_to_camera(p, np.eye(4), intr0, 640, 480)
    assert abs(u1[0] - 320) > abs(u0[0] - 320)


def test_rgb_colouring_matches_image_and_occlusion(ctx):
    cams = load_camera_calibration(ctx["root"], ctx["name"], ctx["split"])
    imgs = frame_images(ctx["root"], ctx["name"], ctx["split"])
    fr = next(decode_frames(ctx["root"], ctx["name"], ctx["split"], 1))
    jpg, T, _rs = imgs[fr["ts"]][1]
    cam = dict(image=decode_jpeg(jpg), T_world_vehicle=T, **cams[1])
    w = to_world(fr)
    rgb, which, _ = colorize(w, [cam])
    seen = which >= 0
    assert seen.mean() > 0.05 and not seen.all()      # points behind the camera stay uncoloured
    # colour of a point must equal the pixel it projects to
    u, v, d, ok = cal.project_to_camera(cal.transform(cal.inv_rigid(T), w[seen][:50]), cam["T_vehicle_camera"],
                                        cam["intrinsic"], cam["width"], cam["height"])
    px = cam["image"][v.astype(int), u.astype(int)]
    assert (np.abs(px.astype(int) - rgb[seen][:50].astype(int)).max(axis=1) <= 2).mean() > 0.95
    # occluded: a far point directly behind a near one in the same pixel cell must not take the near colour
    near = np.array([[8.0, 0.0, 1.6]])
    far = np.array([[30.0, 0.0, 1.6]])
    cam_t = dict(cam, image=np.zeros_like(cam["image"]), T_world_vehicle=np.eye(4))
    cam_t["image"][:] = (10, 20, 30)
    _, wh, _ = colorize(np.vstack([near, far]), [cam_t])
    assert wh[0] == 0 and wh[1] == -1


def test_overlay_runs(ctx):
    cams = load_camera_calibration(ctx["root"], ctx["name"], ctx["split"])
    imgs = frame_images(ctx["root"], ctx["name"], ctx["split"])
    fr = next(decode_frames(ctx["root"], ctx["name"], ctx["split"], 1))
    out, n = overlay(decode_jpeg(imgs[fr["ts"]][1][0]), fr["xyz_vehicle"], cams[1])
    assert n > 500 and out.shape[:2] == (480, 640)
