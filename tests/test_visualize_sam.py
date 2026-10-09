"""Unit smoke for SAM+gaze overlay drawing helpers (no GPU / no video file)."""

from __future__ import annotations

import numpy as np

from gaze_objects.visualize import _blend_mask, _color_for_label, draw_gaze_marker


def test_color_for_known_label():
    assert _color_for_label("screwdriver") == (40, 180, 255)


def test_draw_gaze_large_marker():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    out = draw_gaze_marker(
        img,
        80,
        60,
        src_width=160,
        src_height=120,
        radius=28,
        thickness=3,
        marker_size=40,
        color=(0, 255, 0),
    )
    assert out.shape == img.shape
    # Marker should paint some non-zero pixels near center.
    assert out[60, 80].sum() > 0 or out[60 - 10 : 60 + 10, 80 - 10 : 80 + 10].sum() > 0


def test_blend_mask_changes_pixels():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    _blend_mask(img, mask, (0, 255, 0), alpha=0.5)
    assert img[15, 15, 1] > 0
    assert img[0, 0, 1] == 0
