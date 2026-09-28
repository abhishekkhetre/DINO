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

# When gaze lands only on a child tool box that sits inside a parent AOI box,
# prefer the parent (Tobii AOIs are often broad work regions).
NESTED_CHILD_TO_PARENT = {
    "Tools": "Angle Grinder",
}


def point_in_box(x: float, y: float, x_min: float, y_min: float, x_max: float, y_max: float) -> bool:
    """Inclusive min, exclusive max — consistent with in-frame gaze policy."""
    return (x >= x_min) and (x < x_max) and (y >= y_min) and (y < y_max)


def _norm_label(det: dict[str, Any]) -> str | None:
    lab = det.get("normalized_label")
    if lab is None or (isinstance(lab, float) and pd.isna(lab)):
        return None
    text = str(lab).strip()
    return text or None


def _box_center_in(inner: dict[str, Any], outer: dict[str, Any]) -> bool:
    cx = (float(inner["x_min"]) + float(inner["x_max"])) / 2.0
    cy = (float(inner["y_min"]) + float(inner["y_max"])) / 2.0
    return point_in_box(
        cx,
        cy,
        float(outer["x_min"]),
        float(outer["y_min"]),
        float(outer["x_max"]),
        float(outer["y_max"]),
    )


def _passes_det_filters(
    det: dict[str, Any],
    *,
    score_threshold: float,
    require_normalized_label: bool,
) -> bool:
    if float(det.get("score", 0.0)) < score_threshold:
        return False
    if require_normalized_label and not _norm_label(det):
        return False
    return True


def _find_nested_parent(
    child: dict[str, Any],
    detections: Iterable[dict[str, Any]],
    *,
    score_threshold: float,
    require_normalized_label: bool,
) -> dict[str, Any] | None:
    parent_lab = NESTED_CHILD_TO_PARENT.get(_norm_label(child) or "")
    if not parent_lab:
        return None
    parents: list[dict[str, Any]] = []
    for det in detections:
        if not _passes_det_filters(
            det,
            score_threshold=score_threshold,
            require_normalized_label=require_normalized_label,
        ):
            continue
        if _norm_label(det) != parent_lab:
            continue
        if _box_center_in(child, det):
            parents.append(det)
    if not parents:
        return None
    return max(parents, key=lambda d: float(d.get("score", 0.0)))


def _fill_assigned(base: dict[str, Any], chosen: dict[str, Any], reason: str) -> dict[str, Any]:
    base["selected_detection_id"] = str(chosen["detection_id"])
    base["selected_raw_label"] = chosen.get("raw_label")
    base["selected_normalized_label"] = chosen.get("normalized_label")
    base["selected_score"] = float(chosen.get("score", 0.0))
    base["assignment_status"] = "assigned"
    base["assignment_reason"] = reason
    return base


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
    prefer_nested_parent: bool = False,
) -> dict[str, Any]:
    """
    Baseline assignment for one gaze sample.

    Candidate boxes must pass score_threshold (and optional label requirement).
    Exactly one containing candidate -> assigned.
    None -> no_detected_target (only if frame was processed and gaze eligible).
    Several -> ambiguous (no automatic nearest/largest/smallest choice), unless
    ``prefer_nested_parent`` resolves Tools nested under Angle Grinder.
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

    dets = list(detections)
    candidates: list[dict[str, Any]] = []
    for det in dets:
        if not _passes_det_filters(
            det,
            score_threshold=score_threshold,
            require_normalized_label=require_normalized_label,
        ):
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
        # If every containing box maps to the same study AOI (e.g. grinder /
        # electric grinder synonyms), treat as a single assignment — pick
        # highest score. Different AOIs still count as ambiguous.
        norm_labels = [_norm_label(d) for d in candidates]
        unique_norms = {n for n in norm_labels if n is not None}
        if len(unique_norms) == 1 and None not in norm_labels:
            chosen = max(candidates, key=lambda d: float(d.get("score", 0.0)))
            return _fill_assigned(base, chosen, "same_normalized_label_highest_score")

        if prefer_nested_parent and unique_norms == {"Tools", "Angle Grinder"}:
            tools = [c for c in candidates if _norm_label(c) == "Tools"]
            grinders = [c for c in candidates if _norm_label(c) == "Angle Grinder"]
            if any(_box_center_in(t, g) for t in tools for g in grinders):
                chosen = max(grinders, key=lambda d: float(d.get("score", 0.0)))
                return _fill_assigned(base, chosen, "nested_tools_under_angle_grinder")

        base["assignment_status"] = "ambiguous"
        base["assignment_reason"] = "multiple_boxes_contain_gaze"
        return base

    chosen = candidates[0]
    if prefer_nested_parent:
        parent = _find_nested_parent(
            chosen,
            dets,
            score_threshold=score_threshold,
            require_normalized_label=require_normalized_label,
        )
        if parent is not None:
            return _fill_assigned(base, parent, "nested_tools_under_angle_grinder")
    return _fill_assigned(base, chosen, "single_containing_box")


def assign_table(
    gaze_frame_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    *,
    score_threshold: float = 0.3,
    require_normalized_label: bool = False,
    prefer_nested_parent: bool = False,
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
            prefer_nested_parent=prefer_nested_parent,
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
