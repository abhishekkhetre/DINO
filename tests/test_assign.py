"""Tests for gaze-to-box assignment statuses."""

from __future__ import annotations

from gaze_objects.assign import assign_gaze_to_detections, point_in_box


def test_point_in_box_bounds():
    assert point_in_box(10, 10, 0, 0, 100, 100)
    assert not point_in_box(100, 10, 0, 0, 100, 100)  # exclusive max
    assert not point_in_box(-1, 10, 0, 0, 100, 100)


def _det(i, x0, y0, x1, y1, label="Tools", score=0.9):
    return {
        "detection_id": f"d{i}",
        "x_min": x0,
        "y_min": y0,
        "x_max": x1,
        "y_max": y1,
        "raw_label": label,
        "normalized_label": label,
        "score": score,
    }


def test_assigned_single_candidate():
    out = assign_gaze_to_detections(
        50,
        50,
        has_gaze_coordinates=True,
        in_frame=True,
        out_of_frame=False,
        association_status="matched",
        frame_processed=True,
        detections=[_det(0, 0, 0, 100, 100)],
    )
    assert out["assignment_status"] == "assigned"
    assert out["selected_detection_id"] == "d0"


def test_no_detected_target():
    out = assign_gaze_to_detections(
        50,
        50,
        has_gaze_coordinates=True,
        in_frame=True,
        out_of_frame=False,
        association_status="matched",
        frame_processed=True,
        detections=[_det(0, 80, 80, 100, 100)],
    )
    assert out["assignment_status"] == "no_detected_target"


def test_ambiguous_overlap():
    out = assign_gaze_to_detections(
        50,
        50,
        has_gaze_coordinates=True,
        in_frame=True,
        out_of_frame=False,
        association_status="matched",
        frame_processed=True,
        detections=[_det(0, 0, 0, 100, 100, "Manual"), _det(1, 40, 40, 60, 60, "Tools")],
    )
    assert out["assignment_status"] == "ambiguous"
    assert out["n_candidates"] == 2
    assert out["selected_detection_id"] is None


def test_score_threshold_filters():
    out = assign_gaze_to_detections(
        50,
        50,
        has_gaze_coordinates=True,
        in_frame=True,
        out_of_frame=False,
        association_status="matched",
        frame_processed=True,
        detections=[_det(0, 0, 0, 100, 100, score=0.1)],
        score_threshold=0.3,
    )
    assert out["assignment_status"] == "no_detected_target"


def test_invalid_and_unmatched_and_unprocessed():
    assert (
        assign_gaze_to_detections(
            None,
            None,
            has_gaze_coordinates=False,
            in_frame=False,
            out_of_frame=False,
            association_status="matched",
            frame_processed=True,
            detections=[],
        )["assignment_status"]
        == "invalid_gaze"
    )
    assert (
        assign_gaze_to_detections(
            1,
            1,
            has_gaze_coordinates=True,
            in_frame=True,
            out_of_frame=False,
            association_status="unmatched_time",
            frame_processed=True,
            detections=[],
        )["assignment_status"]
        == "unmatched_time"
    )
    assert (
        assign_gaze_to_detections(
            1,
            1,
            has_gaze_coordinates=True,
            in_frame=True,
            out_of_frame=False,
            association_status="matched",
            frame_processed=False,
            detections=[],
        )["assignment_status"]
        == "frame_not_processed"
    )
    assert (
        assign_gaze_to_detections(
            -5,
            10,
            has_gaze_coordinates=True,
            in_frame=False,
            out_of_frame=True,
            association_status="matched",
            frame_processed=True,
            detections=[],
        )["assignment_status"]
        == "out_of_frame"
    )
