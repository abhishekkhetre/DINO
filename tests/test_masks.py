"""Gaze-in-mask assignment and mask archive helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gaze_objects.assign import assign_gaze_to_detections, assign_table, gaze_hits_detection
from gaze_objects.masks import (
    load_detection_masks,
    point_in_mask,
    save_detection_masks,
)


def test_point_in_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:6] = 1
    assert point_in_mask(4, 3, mask)
    assert not point_in_mask(0, 0, mask)
    assert not point_in_mask(4, 3, None)


def test_gaze_hits_mask_not_box_padding():
    # Gaze inside box but outside silhouette → mask miss, box hit.
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1
    det = {
        "detection_id": "d0",
        "x_min": 0,
        "y_min": 0,
        "x_max": 100,
        "y_max": 100,
        "mask": mask,
        "score": 0.9,
        "normalized_label": "screwdriver",
        "raw_label": "screwdriver",
    }
    assert gaze_hits_detection(10, 10, det, hit_test="box")
    assert not gaze_hits_detection(10, 10, det, hit_test="mask")
    assert gaze_hits_detection(50, 50, det, hit_test="mask")


def test_assign_mask_mode():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:40, 20:40] = 1
    det = {
        "detection_id": "d0",
        "x_min": 0,
        "y_min": 0,
        "x_max": 80,
        "y_max": 80,
        "raw_label": "screwdriver",
        "normalized_label": "screwdriver",
        "score": 0.9,
        "mask": mask,
    }
    hit = assign_gaze_to_detections(
        30,
        30,
        has_gaze_coordinates=True,
        in_frame=True,
        out_of_frame=False,
        association_status="matched",
        frame_processed=True,
        detections=[det],
        hit_test="mask",
    )
    assert hit["assignment_status"] == "assigned"
    assert hit["assignment_reason"] == "single_containing_mask"

    miss = assign_gaze_to_detections(
        5,
        5,
        has_gaze_coordinates=True,
        in_frame=True,
        out_of_frame=False,
        association_status="matched",
        frame_processed=True,
        detections=[det],
        hit_test="mask",
    )
    assert miss["assignment_status"] == "no_detected_target"
    assert miss["assignment_reason"] == "no_mask_contains_gaze"


def test_assign_table_loads_masks(tmp_path):
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    save_detection_masks(tmp_path, {"f1_sam3_0": mask})
    assert "f1_sam3_0" in load_detection_masks(tmp_path)

    gaze = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "frame_index": 1,
                "recording_time_s_provisional": 1.0,
                "video_time_s": 1.0,
                "temporal_residual_s": 0.0,
                "gaze_x_px": 15,
                "gaze_y_px": 15,
                "has_gaze_coordinates": True,
                "in_frame": True,
                "out_of_frame": False,
                "association_status": "matched",
            }
        ]
    )
    dets = pd.DataFrame(
        [
            {
                "detection_id": "f1_sam3_0",
                "frame_index": 1,
                "x_min": 0,
                "y_min": 0,
                "x_max": 50,
                "y_max": 50,
                "raw_label": "storage box",
                "normalized_label": "storage box",
                "score": 0.8,
            }
        ]
    )
    out = assign_table(gaze, dets, hit_test="mask", masks_dir=tmp_path, label_map_mode="fine")
    assert out.iloc[0]["assignment_status"] == "assigned"
    assert out.iloc[0]["selected_normalized_label"] == "storage box"
