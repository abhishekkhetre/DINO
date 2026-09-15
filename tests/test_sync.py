"""Synthetic timing and association tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gaze_objects.sync import TimeMapping, associate_gaze_to_frames, representative_gaze_per_frame
from gaze_objects.visualize import draw_gaze_marker


def test_time_mapping_offset_not_first_gaze():
    # Clip begins 30s into recording; first gaze in export may be 24.735 — that is NOT the origin.
    mapping = TimeMapping(
        timestamp_unit="milliseconds",
        timestamp_unit_status="provisional",
        video_start_recording_s=30.0,
        time_mapping_status="provisional",
    )
    assert abs(mapping.recording_raw_to_seconds(24735) - 24.735) < 1e-9
    assert mapping.recording_time_to_video_time(30.0) == 0.0
    assert mapping.recording_time_to_video_time(40.0) == 10.0


def test_handoff_example_ms_and_zero_origin():
    mapping = TimeMapping(
        timestamp_unit="milliseconds",
        timestamp_unit_status="provisional",
        video_start_recording_s=0.0,
        time_mapping_status="provisional",
    )
    rec_s = mapping.recording_raw_to_seconds(24735)
    assert abs(rec_s - 24.735) < 1e-9
    assert abs(mapping.recording_time_to_video_time(rec_s) - 24.735) < 1e-9


def test_bounded_nearest_rejects_distant_frames():
    frames = np.array([0.0, 0.04, 0.08, 0.12])
    gaze = np.array([0.041, 0.50])  # second is far
    assoc = associate_gaze_to_frames(gaze, frames, max_abs_residual_s=0.05)
    assert assoc.loc[0, "association_status"] == "matched"
    assert assoc.loc[0, "frame_index"] == 1
    assert assoc.loc[1, "association_status"] == "unmatched_time"
    assert assoc.loc[1, "frame_index"] == -1


def test_irregular_frame_times():
    frames = np.array([0.0, 0.01, 0.05, 0.20])
    gaze = np.array([0.049])
    assoc = associate_gaze_to_frames(gaze, frames, max_abs_residual_s=0.02)
    assert assoc.loc[0, "frame_index"] == 2
    assert abs(assoc.loc[0, "temporal_residual_s"]) < 0.02


def test_empty_frames_all_unmatched():
    assoc = associate_gaze_to_frames(np.array([1.0, 2.0]), np.array([]), max_abs_residual_s=0.05)
    assert list(assoc["association_status"]) == ["unmatched_time", "unmatched_time"]


def test_representative_gaze_picks_smallest_residual():
    gaze = pd.DataFrame(
        {
            "eligible_gaze_default_policy": [True, True, False],
            "gaze_x_px": [1.0, 2.0, 3.0],
            "gaze_y_px": [1.0, 2.0, 3.0],
        }
    )
    assoc = pd.DataFrame(
        {
            "gaze_order": [0, 1, 2],
            "frame_index": [5, 5, 5],
            "frame_time_s": [1.0, 1.0, 1.0],
            "temporal_residual_s": [0.04, 0.01, 0.0],
            "association_status": ["matched", "matched", "matched"],
            "max_abs_residual_s": 0.05,
            "method": "nearest_bounded",
        }
    )
    rep = representative_gaze_per_frame(assoc, gaze)
    assert len(rep) == 1
    assert float(rep.iloc[0]["gaze_x_px"]) == 2.0


def test_draw_gaze_scales_with_resize():
    import numpy as np

    img = np.zeros((540, 960, 3), dtype=np.uint8)
    out = draw_gaze_marker(img, 960, 540, src_width=1920, src_height=1080)
    # Marker near centre of resized image; ensure we did not crash and wrote pixels.
    assert out.shape == img.shape
    assert out.sum() > 0
