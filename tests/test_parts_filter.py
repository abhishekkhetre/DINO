"""Parent-overlap filter for orphan part detections."""

from __future__ import annotations

import pandas as pd

from gaze_objects.parts_filter import filter_parts_requiring_parent


def test_orphan_screw_dropped_without_parent_grinder():
    df = pd.DataFrame(
        [
            {
                "detection_id": "a",
                "frame_index": 1,
                "normalized_label": "screw",
                "x_min": 10,
                "y_min": 10,
                "x_max": 20,
                "y_max": 20,
            },
            {
                "detection_id": "b",
                "frame_index": 1,
                "normalized_label": "screwdriver",
                "x_min": 100,
                "y_min": 100,
                "x_max": 140,
                "y_max": 180,
            },
        ]
    )
    out, stats = filter_parts_requiring_parent(df)
    assert list(out["detection_id"]) == ["b"]
    assert stats["n_parts_removed"] == 1


def test_side_handle_kept_when_overlaps_grinder():
    df = pd.DataFrame(
        [
            {
                "detection_id": "ag",
                "frame_index": 2,
                "normalized_label": "angle grinder",
                "x_min": 0,
                "y_min": 0,
                "x_max": 200,
                "y_max": 200,
            },
            {
                "detection_id": "h",
                "frame_index": 2,
                "normalized_label": "side handle",
                "x_min": 40,
                "y_min": 40,
                "x_max": 80,
                "y_max": 90,
            },
        ]
    )
    out, stats = filter_parts_requiring_parent(df)
    assert set(out["detection_id"]) == {"ag", "h"}
    assert stats["n_parts_removed"] == 0
