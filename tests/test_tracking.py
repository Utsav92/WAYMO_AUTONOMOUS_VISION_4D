import numpy as np

from object_labels import load_boxes, track_gaps, tracks, speed_from_track
from export_frames import export
import waymo_adapter as wa


def test_track_ids_continuous(ctx):
    b, ids = load_boxes(ctx["root"], ctx["name"], ctx["split"])
    assert len(ids) == 4
    ts = sorted(set(b["ts"].tolist()))
    assert track_gaps(b, ts) == {}
    for tid, bs in tracks(b).items():
        assert len(bs) == len(ts)


def test_box_types_and_sizes(ctx):
    b, _ = load_boxes(ctx["root"], ctx["name"], ctx["split"])
    assert set(b["type"].tolist()) <= set(wa.BOX_TYPES)
    assert (b["l"] > 0).all() and (b["h"] > 0).all()


def test_exported_speed_matches_truth(ctx, tmp_path):
    m = export(ctx["root"], ctx["name"], str(tmp_path), ctx["split"], max_frames=6)
    d = np.load(tmp_path / ctx["name"] / "frames" / "f0004.npz")
    sp = d["boxes"][:, 9]
    typ = d["boxes"][:, 7]
    veh = sp[typ == 1]
    assert len(veh) == 2
    assert sorted(np.round(veh, 1)) == [2.0, 3.0]
    first = np.load(tmp_path / ctx["name"] / "frames" / "f0000.npz")["boxes"][:, 9]
    assert (first == -1).all()                       # no speed without two observations


def test_speed_none_when_gap():
    bs = [{"ts": 0}, {"ts": 2_000_000}]
    assert speed_from_track(bs, 1, [[0, 0, 0], [1, 0, 0]]) is None
