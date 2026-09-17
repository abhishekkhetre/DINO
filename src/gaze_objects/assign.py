"""Gaze-to-detection assignment with explicit uncertainty statuses."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from gaze_objects.detectors.label_map import normalize_label


ASSIGNMENT_STATUSES = (
    "assigned",
    "no_detected_target",
    "ambiguous",
    "invalid_gaze",
    "out_of_frame",
    "unmatched_time",
    "frame_not_processed",
)


def point_in_box(x: float, y: float, x_min: float, y_min: float, x_max: float, y_max: float) -> bool:
    """Inclusive min, exclusive max — consistent with in-frame gaze policy."""
    return (x >= x_min) and (x < x_max) and (y >= y_min) and (y < y_max)


def assign_gaze_to_detections(
    gaze_x: float | None,
    gaze_y: float | None,
    *,
    has_gaze_coordinates: bool,
    in_frame: bool,
    out_of_frame: bool,
    association_status: str,
    frame_processed: bool,
    detections: Iterable[dict[str, Any]],
    score_threshold: float = 0.3,
    require_normalized_label: bool = False,
) -> dict[str, Any]:
    """
    Baseline assignment for one gaze sample.

    Candidate boxes must pass score_threshold (and optional label requirement).
    Exactly one containing candidate -> assigned.
    None -> no_detected_target (only if frame was processed and gaze eligible).
    Several -> ambiguous (no automatic nearest/largest/smallest choice).
    """
    base = {
        "selected_detection_id": None,
        "selected_raw_label": None,
        "selected_normalized_label": None,
        "selected_score": None,
        "candidate_detection_ids": [],
        "n_candidates": 0,
        "assignment_status": None,
        "assignment_reason": None,
    }

    if association_status != "matched":
        base["assignment_status"] = "unmatched_time"
        base["assignment_reason"] = "gaze_not_matched_to_frame"
        return base

    if not has_gaze_coordinates:
        base["assignment_status"] = "invalid_gaze"
        base["assignment_reason"] = "missing_gaze_coordinates"
        return base

    if out_of_frame or not in_frame:
        base["assignment_status"] = "out_of_frame"
        base["assignment_reason"] = "gaze_outside_scene_bounds"
        return base

    if not frame_processed:
        base["assignment_status"] = "frame_not_processed"
        base["assignment_reason"] = "no_detections_cached_for_frame"
        return base

    if gaze_x is None or gaze_y is None or pd.isna(gaze_x) or pd.isna(gaze_y):
        base["assignment_status"] = "invalid_gaze"
        base["assignment_reason"] = "gaze_xy_nan"
        return base

    candidates: list[dict[str, Any]] = []
    for det in detections:
        score = float(det.get("score", 0.0))
        if score < score_threshold:
            continue
        if require_normalized_label and not det.get("normalized_label"):
            continue
        if point_in_box(
            float(gaze_x),
            float(gaze_y),
            float(det["x_min"]),
            float(det["y_min"]),
            float(det["x_max"]),
            float(det["y_max"]),
        ):
            candidates.append(det)

    ids = [str(d["detection_id"]) for d in candidates]
    base["candidate_detection_ids"] = ids
    base["n_candidates"] = len(candidates)

    if len(candidates) == 0:
        base["assignment_status"] = "no_detected_target"
        base["assignment_reason"] = "no_box_contains_gaze"
        return base

    if len(candidates) > 1:
        base["assignment_status"] = "ambiguous"
        base["assignment_reason"] = "multiple_boxes_contain_gaze"
        return base

    chosen = candidates[0]
    base["selected_detection_id"] = str(chosen["detection_id"])
    base["selected_raw_label"] = chosen.get("raw_label")
    base["selected_normalized_label"] = chosen.get("normalized_label")
    base["selected_score"] = float(chosen.get("score", 0.0))
    base["assignment_status"] = "assigned"
    base["assignment_reason"] = "single_containing_box"
    return base


def assign_table(
    gaze_frame_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    *,
    score_threshold: float = 0.3,
    require_normalized_label: bool = False,
    frame_id_col: str = "frame_index",
) -> pd.DataFrame:
    """
    Assign each gaze row using detections grouped by frame_index.

    ``gaze_frame_df`` should already include association + eligibility columns.
    """
    processed_frames = set()
    by_frame: dict[int, list[dict[str, Any]]] = {}
    if len(detections_df):
        for _, row in detections_df.iterrows():
            fi = int(row[frame_id_col])
            processed_frames.add(fi)
            det = row.to_dict()
            # Re-apply current phrase→AOI map so map fixes work without re-detect.
            raw = det.get("raw_label")
            if raw is not None and str(raw).strip():
                det["normalized_label"] = normalize_label(str(raw))
            by_frame.setdefault(fi, []).append(det)

    records = []
    for _, g in gaze_frame_df.iterrows():
        fi = int(g[frame_id_col]) if pd.notna(g.get(frame_id_col)) and int(g.get(frame_id_col, -1)) >= 0 else -1
        result = assign_gaze_to_detections(
            g.get("gaze_x_px"),
            g.get("gaze_y_px"),
            has_gaze_coordinates=bool(g.get("has_gaze_coordinates", False)),
            in_frame=bool(g.get("in_frame", False)),
            out_of_frame=bool(g.get("out_of_frame", False)),
            association_status=str(g.get("association_status", "unmatched_time")),
            frame_processed=fi in processed_frames,
            detections=by_frame.get(fi, []),
            score_threshold=score_threshold,
            require_normalized_label=require_normalized_label,
        )
        records.append(
            {
                "source_row_id": g.get("source_row_id"),
                "frame_index": fi,
                "recording_time_s": g.get("recording_time_s_provisional", g.get("recording_time_s")),
                "video_time_s": g.get("video_time_s"),
                "temporal_residual_s": g.get("temporal_residual_s"),
                "gaze_x_px": g.get("gaze_x_px"),
                "gaze_y_px": g.get("gaze_y_px"),
                **result,
                "candidate_detection_ids": "|".join(result["candidate_detection_ids"]),
            }
        )
    return pd.DataFrame.from_records(records)
