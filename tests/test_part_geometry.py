"""Geometric tip/handle/body classification from gaze + mask."""

from __future__ import annotations

import numpy as np

from gaze_objects.part_geometry import classify_gaze_part


def _elongated_mask(h=80, w=200, thickness=8) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    cy = h // 2
    m[cy - thickness // 2 : cy + thickness // 2 + 1, 10:190] = True
    # Thicker handle on the right
    m[cy - thickness : cy + thickness + 1, 150:190] = True
    return m


def test_screwdriver_tip_vs_handle():
    mask = _elongated_mask()
    # Left end ≈ tip (thinner)
    tip = classify_gaze_part(20, 40, mask, "screwdriver")
    assert tip["part_zone"] == "tip"
    assert tip["attended_part_label"] == "screwdriver.tip"
    # Right end ≈ handle (thicker)
    handle = classify_gaze_part(170, 40, mask, "screwdriver")
    assert handle["part_zone"] == "handle"
    assert handle["attended_part_label"] == "screwdriver.handle"


def test_angle_grinder_body_near_center():
    m = np.zeros((100, 100), dtype=bool)
    m[20:80, 20:80] = True
    out = classify_gaze_part(50, 50, m, "angle grinder")
    assert out["part_zone"] == "body"
    assert out["attended_part_label"] == "angle grinder.body"


def test_non_part_object_skipped():
    m = np.ones((50, 50), dtype=bool)
    out = classify_gaze_part(10, 10, m, "storage box")
    assert out["attended_part_label"] is None
