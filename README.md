# WAYMO AUTONOMOUS VISION 4D

Autonomous-driving perception visualiser built on **real Waymo data** and rendered live in **TouchDesigner**:

| Source | What it adds | Dataset |
|---|---|---|
| Perception | calibrated 5-LiDAR clouds, 5 cameras, RGB-coloured points, 3D boxes + track ids, ego poses, measured road surface, semantic labels, depth-vs-LiDAR error | Waymo Open **Perception v2.0.1** (Parquet) |
| Motion | vector **map** (lanes, road edges/lines, crosswalks, speed bumps, driveways, stop signs), 9 s agent tracks, **ground-truth future paths** | Waymo Open **Motion v1.3.1** (scenario protos) |
| End-to-end driving | 8 cameras, 4 s ego past, 5 s ego future, route intent, alternative trajectories | Waymo Open **WOD-E2E camera v1.0.0** |

### What each source can really show (layers that a dataset does not contain are never faked)

| Layer | Perception | Motion | End-to-end |
|---|---|---|---|
| LiDAR points | 5 sensors, every frame | **first 1 s only** (11 frames; delta-compressed, decoded here) | none in dataset |
| RGB-coloured points | LiDAR + calibrated cameras | none (no camera images in the file) | **ESTIMATED** from monocular depth, metric scale from the road plane |
| 3D boxes / tracks / speeds | ground-truth labels | ground-truth tracks | none in dataset |
| Road surface | measured from LiDAR, camera colours, lane paint from intensity | rasterised from the vector map | driving corridor along the real ego trajectories |
| Map polylines | not in this dataset | lanes, road edges/lines, crosswalks, speed bumps, driveways, stop signs | none |
| Future paths | not in this dataset | ground-truth futures (to-predict agents + SDC) | ego future + alternatives |
| Semantic labels / depth error | yes | no | no |

The Motion and E2E sources are separate recordings from the Perception segments (no public mapping), so they are shown as
separate sources (NEXT SOURCE button) and labelled. The HUD prints what the current source provides.

Waymo data is licensed **non-commercial**. Do not redistribute the data or frames rendered from it.

## Run

```powershell
pip install -r requirements.txt
gcloud auth login                         # an account that accepted https://waymo.com/open/terms/
python datasets/download_waymo.py --list --split validation --limit 10
python datasets/download_waymo.py --context <NAME> --split validation --out D:/WAYMO_DATA/raw --yes
python datasets/inspect_components.py --root D:/WAYMO_DATA/raw --context <NAME>
python preprocessing/export_frames.py --root D:/WAYMO_DATA/raw --context <NAME> --split validation [--semantic models/seg.pt]
python preprocessing/validate_projection.py --root D:/WAYMO_DATA/raw --context <NAME> --frames 5 60 120
python machine_learning/camera_depth.py --root D:/WAYMO_DATA/raw --context <NAME> --save-processed data/processed/<NAME>
python datasets/motion_adapter.py --shard <validation.tfrecord-...> --count 3
python datasets/e2e_adapter.py --shard <val_...tfrecord-...> --count 3
# TouchDesigner (own instance, MCP web server port 9995)
TouchDesigner.exe touchdesigner\base.tox
python touchdesigner/tools/tdrun.py touchdesigner/build_project.py        # builds + starts playing
python touchdesigner/tools/export_cinematic.py --encode --name waymo_av4d # 60 s film (add --scientific for no effects)
```

Saved project: `touchdesigner/WAYMO_AUTONOMOUS_VISION_4D.toe`.

## Control box (left side of the output window)

Render style (SCIENTIFIC / CINEMATIC) - 15 layer toggles (RGB points, LiDAR, depth error, 3D boxes, 3D models, road, labels,
grid, trajectory, history, glow, statistics, map, future paths, semantic) - view presets - temporal 4D (off / echo / trails /
explosion) - camera (driver, chase, free, bird, orbit) - timeline (play, pause, reverse, reset, step, freeze, next source) -
13 sliders. Free-fly: W/A/S/D, Q/E, arrows or left-drag to look, wheel to zoom, Shift = fast.

## Visual quality pass (what was fixed)

* Road: holes are filled from the surroundings at several scales (no black voids), soft coverage contour, 0.35 m cells,
  distance fog into the sky colour; Motion gets a lit road + terrain; E2E a smooth corridor.
* Sky dome + horizon glow replace the black background; ground fades into it.
* LiDAR points: true alpha blending (crisp orange, not pale additive), distance-attenuated 1.2-5 px sprites, depth-written
  so nothing is painted over; camera-colour layer shown at 40 % LiDAR density when both are on.
* Objects: real sign model (plate + pole) instead of cubes; static vehicles get dim boxes, moving ones neon boxes.
* Motion: LiDAR of the first second is kept as *static world memory* (moving agents removed) for the rest of the scenario.
* E2E: estimated cloud is edge-filtered (no flying pixels), sky-removed, bilateral-smoothed; labelled ESTIMATED.

## What each layer is

* **LiDAR / RGB points** - 5 LiDARs decoded from range images with real calibration; TOP lidar uses its per-pixel vehicle
  pose; camera colours use the vehicle pose + rolling-shutter exposure time of each image.
* **Road** - height field rebuilt each frame from the lowest returns of the last few frames, coloured by cameras, lane paint lit
  from LiDAR intensity. Cells with no return stay empty (gaps <= 2.5 m are interpolated, flagged 0.5 coverage).
* **3D models** - procedural meshes made with Blender (`assets/make_models.py`): a white robotaxi-style SUV with roof lidar dome
  and teal accent, generic car, truck, pedestrian, cyclist, one instanced draw per class. They are **not** Waymo's assets.
* **Boxes / speeds** - Waymo ground-truth labels (not model output); speed is Waymo's own `speed.x/y` metadata.
* **Depth error** - Depth Anything V2 Small (pretrained) aligned with one affine fit per image to TOP-LiDAR depth; *estimated*
  depth, never shown as measured.
* **Semantic** - Waymo labels where present, otherwise predictions of a small model I trained (see metrics file).

## Verified results (real data, this machine)

| Check | Result |
|---|---|
| Projection vs Waymo `lidar_camera_projection` (FRONT camera) | frame pose + global shutter: median **7.95 px**, p90 33.7 px -> TOP pixel pose + rolling shutter: median **1.56 px**, p90 3.4 px |
| Monocular depth vs TOP LiDAR (context 10203656..., below horizon, un-occluded) | median AbsRel **0.03-0.07**, 76-84 % of pixels within delta<1.25; mean AbsRel ~0.5-0.6 (outliers: thin poles, sky returns) |
| Tests | 30 pass (26 pipeline on a synthetic fixture + 4 live-TD: no errors/shaders compile, playback renders, depth metrics, camera snaps on loop wrap / panel wired) |
| Playback | ~7 data frames/s with history on an Intel UHD 620 while other jobs run; real key press (W) verified through `keybd_event`; panel buttons + sliders verified through TD panel clicks |

| LiDAR semantic segmentation (small per-point MLP, 2.4 M training points from 4 contexts, tested on a **held-out context**, 4.46 M labelled points) | accuracy **51.3 %**, mIoU **18.4 %** over 14 classes present. Best: road 0.70, walkable 0.51, vegetation 0.41; poor: poles 0.04, trunks 0.03, lane markers 0.10. Weak model - used only to fill frames Waymo did not label; Waymo's own labels are used where present. |

## Not done / limits

* Real-time at 1080p on a 12 GB RTX is **not measured** (iGPU only, non-commercial TD caps at 1280 px).
* The film has been rendered only for one 20 s Perception segment (slowed to 60 s, eight scenes).
* LiDAR returns above the horizon in clear sky are Waymo-side artefacts; they are drawn, but excluded from the depth metric.
* Waymo E2E preference scores are -1 in this shard (not released); the three alternative trajectories are shown unscored.
* Free-fly **mouse drag** is wired (Mouse In CHOP) but only the keyboard was verified with real input events.
* `camera_segmentation`, `*_hkp` (keypoints) and `camera_to_lidar_box_association` components are downloaded/available but unused.

## Repository contents / what is NOT included

This repository holds code, shaders and my own procedural assets only.  **Not included** (by design):

* Waymo Open Dataset files and anything derived from them (processed frames, screenshots, the rendered film, the
  segmentation weights trained on Waymo labels) - the Waymo licence is non-commercial and forbids redistribution.
  Download the data yourself after accepting the licence (`datasets/download_waymo.py`).
* The TouchDesigner MCP web-server bridge (`touchdesigner/base.tox`, `touchdesigner/modules/`) and the saved `.toe`;
  start TouchDesigner with your own MCP-enabled `base.tox` on port 9995 and run `build_project.py` to regenerate the project.
* `datasets/waymo_pb/` contains protobuf Python modules generated from Waymo's Apache-2.0 `.proto` files; regenerate with
  `grpc_tools.protoc` if you prefer.
