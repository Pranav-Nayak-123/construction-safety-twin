"""Approximate twin for a steady camera the workers could not calibrate."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pipeline.mapping.camera_model import Camera, default_camera, visible_ground
from pipeline.output.twin_data import FALLBACK_FAR_WH, TwinRecorder


def _worker(x: float, y: float, relative=None):  # noqa: ANN001, ANN202
    return SimpleNamespace(relative_position=relative, anchor_reference=SimpleNamespace(x=x, y=y))


def test_default_camera_sees_ground_from_the_bottom_edge_outwards() -> None:
    camera = Camera(default_camera(1920, 1080))
    near, far = camera.ground(np.array([[960.0, 1080.0], [960.0, 300.0]]))
    assert 0 < near[1] < far[1]  # lower in the image is nearer the camera
    assert abs(near[0]) < 1e-6  # the centre column stays on the camera's axis
    ring = visible_ground(camera, 1920, 1080, far_wh=FALLBACK_FAR_WH)
    assert len(ring) == 4 and max(y for _, y in ring) <= FALLBACK_FAR_WH


def test_fallback_places_uncalibrated_workers_but_never_overrides_calibration() -> None:
    camera = Camera(default_camera(1920, 1080))
    twin = TwinRecorder(fps=25.0, fallback=camera.ground)
    placed = twin._position(_worker(960.0, 1000.0))
    assert placed is not None and placed[1] > 0
    calibrated = twin._position(_worker(960.0, 1000.0, SimpleNamespace(x=1.5, y=2.5)))
    assert calibrated == (1.5, 2.5)
    assert twin._position(_worker(960.0, 0.0)) is None  # near the horizon: too far to trust
    assert TwinRecorder(fps=25.0)._position(_worker(960.0, 1000.0)) is None  # no fallback


def test_r7_ignores_load_tracks_that_never_move() -> None:
    from pipeline.rules.r7 import fixed_structures

    def obs(t: float, x: float):  # noqa: ANN202
        box = SimpleNamespace(x1=x, y1=100.0, x2=x + 80.0, y2=200.0)
        return SimpleNamespace(video_time_s=t, box_oriented=None, box_reference=box)

    boom = [obs(t, 1300.0 + (t % 2)) for t in range(0, 31)]  # 30 s, jitters by 1 px
    hoisted = [obs(t, 400.0 + 20.0 * t) for t in range(0, 31)]  # swings across the frame
    short = [obs(t, 900.0) for t in range(0, 9)]  # still, but only 8 s (a SynthSite clip)
    assert fixed_structures({"boom": boom, "hoisted": hoisted, "short": short}) == {"boom"}


def test_r7_pools_fragmented_tracks_of_one_fixed_structure() -> None:
    from pipeline.rules.r7 import fixed_structures

    def obs(t: float, x: float, y: float = 100.0):  # noqa: ANN202
        box = SimpleNamespace(x1=x, y1=y, x2=x + 500.0, y2=y + 30.0)
        return SimpleNamespace(video_time_s=t, box_oriented=None, box_reference=box)

    # one building ledge, tracked as three short pieces across 40 s
    pieces = {
        "a": [obs(t, 100.0) for t in range(0, 5)],
        "b": [obs(t, 101.0) for t in range(18, 22)],
        "c": [obs(t, 99.0) for t in range(36, 41)],
        "elsewhere": [obs(t, 900.0, 600.0) for t in range(0, 4)],
    }
    assert fixed_structures(pieces) == {"a", "b", "c"}
