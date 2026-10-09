"""Overlays and review visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Distinct BGR colors for common fine / AOI labels.
_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "screwdriver": (40, 180, 255),
    "storage box": (255, 160, 40),
    "instruction manual": (80, 220, 80),
    "manual": (80, 220, 80),
    "angle grinder": (60, 60, 255),
    "grinding disc": (180, 80, 255),
    "wrench": (255, 80, 180),
    "pliers": (0, 200, 200),
    "hex key": (200, 200, 0),
    "hammer": (120, 120, 255),
    "tools": (40, 180, 255),
    "boxes": (255, 160, 40),
}


def _color_for_label(label: str | None) -> tuple[int, int, int]:
    if not label:
        return (200, 200, 200)
    key = str(label).strip().lower()
    if key in _LABEL_COLORS:
        return _LABEL_COLORS[key]
    # Stable hash color for unknown labels.
    h = abs(hash(key)) % (256 * 256 * 256)
    return (h & 255, (h >> 8) & 255, (h >> 16) & 255)


def draw_gaze_marker(
    image_bgr: np.ndarray,
    x: float,
    y: float,
    *,
    src_width: int,
    src_height: int,
    color: tuple[int, int, int] = (0, 0, 255),
    radius: int = 8,
    thickness: int = 2,
    marker_size: int = 18,
) -> np.ndarray:
    """Draw gaze in scene coordinates, scaling if the display image was resized."""
    import cv2

    out = image_bgr.copy()
    h, w = out.shape[:2]
    sx = w / float(src_width)
    sy = h / float(src_height)
    px = int(round(x * sx))
    py = int(round(y * sy))
    if px < 0 or py < 0 or px >= w or py >= h:
        # Still mark near edge for review, but annotate OOB.
        cv2.putText(out, "OOB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        return out
    cv2.circle(out, (px, py), radius, color, thickness)
    cv2.drawMarker(
        out,
        (px, py),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=marker_size,
        thickness=thickness,
    )
    return out


def _blend_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float = 0.35,
) -> None:
    """In-place semi-transparent mask overlay."""
    import cv2

    m = np.asarray(mask)
    if m.ndim != 2:
        m = np.squeeze(m)
    if m.shape[:2] != image_bgr.shape[:2]:
        m = cv2.resize(
            m.astype(np.uint8),
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    sel = m.astype(bool)
    if not sel.any():
        return
    overlay = image_bgr.copy()
    overlay[sel] = color
    image_bgr[sel] = (
        (1.0 - alpha) * image_bgr[sel].astype(np.float32)
        + alpha * overlay[sel].astype(np.float32)
    ).astype(np.uint8)
    # Contour for readability
    contours, _ = cv2.findContours(
        sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image_bgr, contours, -1, color, 2)


def write_overlay_clip(
    video_path: str | Path,
    frame_gaze: pd.DataFrame,
    out_path: str | Path,
    *,
    src_width: int,
    src_height: int,
    start_s: float,
    end_s: float,
    video_fps_hint: float | None = None,
) -> dict[str, Any]:
    """
    Render a short overlay clip using representative gaze per frame.

    ``frame_gaze`` must contain frame_index, gaze_x_px, gaze_y_px, video_time_s.
    """
    import cv2

    from gaze_objects.video import iter_frames_with_time

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_frame = {
        int(r.frame_index): r
        for r in frame_gaze.itertuples(index=False)
        if int(r.frame_index) >= 0
    }

    writer = None
    n_written = 0
    n_with_gaze = 0
    for frame_index, _pts, video_time_s, frame in iter_frames_with_time(
        video_path, start_s=start_s, end_s=end_s, normalize_pts_to_first=True
    ):
        if writer is None:
            h, w = frame.shape[:2]
            fps = video_fps_hint or 25.0
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
        row = by_frame.get(frame_index)
        if row is not None and pd.notna(row.gaze_x_px) and pd.notna(row.gaze_y_px):
            frame = draw_gaze_marker(
                frame,
                float(row.gaze_x_px),
                float(row.gaze_y_px),
                src_width=src_width,
                src_height=src_height,
                radius=22,
                thickness=3,
                marker_size=36,
            )
            n_with_gaze += 1
            label = f"t={video_time_s:.3f}s gaze=({row.gaze_x_px:.0f},{row.gaze_y_px:.0f})"
        else:
            label = f"t={video_time_s:.3f}s no eligible gaze"
        import cv2 as _cv2

        _cv2.putText(frame, label, (20, 40), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        n_written += 1

    if writer is not None:
        writer.release()

    return {
        "out_path": str(out_path),
        "frames_written": n_written,
        "frames_with_gaze": n_with_gaze,
        "start_s": start_s,
        "end_s": end_s,
    }


def write_sam_gaze_overlay(
    video_path: str | Path,
    *,
    detections: pd.DataFrame,
    assignments: pd.DataFrame,
    masks_by_id: dict[str, np.ndarray],
    out_path: str | Path,
    src_width: int,
    src_height: int,
    start_s: float,
    end_s: float,
    video_fps_hint: float | None = None,
    gaze_radius: int = 28,
) -> dict[str, Any]:
    """
    Illustration clip: SAM masks/boxes + large gaze + hit highlight + timestamp.

    For each video frame:
    - draw all detections for that frame (mask fill + box outline + label)
    - draw representative gaze (largest / first assigned sample on that frame)
    - if gaze is assigned, bright green ring + HIT label
    """
    import cv2

    from gaze_objects.video import iter_frames_with_time

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    det_by_frame: dict[int, pd.DataFrame] = {}
    if len(detections) and "frame_index" in detections.columns:
        for fi, sub in detections.groupby(detections["frame_index"].astype(int)):
            det_by_frame[int(fi)] = sub

    # Representative gaze + assignment per frame (prefer assigned).
    gaze_by_frame: dict[int, dict[str, Any]] = {}
    if len(assignments):
        work = assignments.copy()
        work = work[pd.to_numeric(work["frame_index"], errors="coerce").fillna(-1) >= 0]
        work["_assigned"] = (work.get("assignment_status") == "assigned").astype(int)
        work = work.sort_values(["frame_index", "_assigned"], ascending=[True, False])
        for fi, sub in work.groupby(work["frame_index"].astype(int)):
            row = sub.iloc[0]
            if pd.isna(row.get("gaze_x_px")) or pd.isna(row.get("gaze_y_px")):
                continue
            gaze_by_frame[int(fi)] = row.to_dict()

    writer = None
    n_written = 0
    n_with_gaze = 0
    n_hits = 0
    last_dets: pd.DataFrame | None = None

    for frame_index, _pts, video_time_s, frame in iter_frames_with_time(
        video_path, start_s=start_s, end_s=end_s, normalize_pts_to_first=True
    ):
        if writer is None:
            h, w = frame.shape[:2]
            fps = video_fps_hint or 25.0
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )

        # Carry forward last SAM detections between sparse detect frames.
        if frame_index in det_by_frame:
            last_dets = det_by_frame[frame_index]
        dets = last_dets

        canvas = frame.copy()
        h, w = canvas.shape[:2]
        sx = w / float(src_width)
        sy = h / float(src_height)

        if dets is not None and len(dets):
            for _, det in dets.iterrows():
                label = det.get("normalized_label") or det.get("raw_label") or "?"
                color = _color_for_label(str(label))
                det_id = str(det.get("detection_id", ""))
                mask = masks_by_id.get(det_id)
                if mask is not None:
                    # Scale mask if needed
                    m = mask
                    if m.shape[0] != h or m.shape[1] != w:
                        m = cv2.resize(
                            m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                        )
                    _blend_mask(canvas, m, color, alpha=0.40)
                x0 = int(round(float(det["x_min"]) * sx))
                y0 = int(round(float(det["y_min"]) * sy))
                x1 = int(round(float(det["x_max"]) * sx))
                y1 = int(round(float(det["y_max"]) * sy))
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
                cv2.putText(
                    canvas,
                    str(label)[:28],
                    (x0, max(22, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        gaze = gaze_by_frame.get(frame_index)
        hit = False
        hit_label = None
        if gaze is not None:
            n_with_gaze += 1
            gx, gy = float(gaze["gaze_x_px"]), float(gaze["gaze_y_px"])
            hit = str(gaze.get("assignment_status", "")) == "assigned"
            hit_label = gaze.get("selected_normalized_label") or gaze.get("selected_raw_label")
            if hit:
                n_hits += 1
                # Bright green outer ring when gaze is on a labelled object.
                canvas = draw_gaze_marker(
                    canvas,
                    gx,
                    gy,
                    src_width=src_width,
                    src_height=src_height,
                    color=(0, 255, 0),
                    radius=gaze_radius + 10,
                    thickness=4,
                    marker_size=gaze_radius + 18,
                )
                canvas = draw_gaze_marker(
                    canvas,
                    gx,
                    gy,
                    src_width=src_width,
                    src_height=src_height,
                    color=(0, 255, 255),
                    radius=gaze_radius,
                    thickness=3,
                    marker_size=gaze_radius + 8,
                )
            else:
                canvas = draw_gaze_marker(
                    canvas,
                    gx,
                    gy,
                    src_width=src_width,
                    src_height=src_height,
                    color=(0, 0, 255),
                    radius=gaze_radius,
                    thickness=3,
                    marker_size=gaze_radius + 8,
                )

        # Timestamp + status banner
        ts = f"t={video_time_s:.2f}s  frame={frame_index}"
        cv2.rectangle(canvas, (8, 8), (min(w - 8, 720), 78), (0, 0, 0), -1)
        cv2.putText(
            canvas, ts, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA
        )
        if hit and hit_label:
            status = f"HIT: {hit_label}"
            status_color = (0, 255, 0)
        elif gaze is not None:
            status = f"gaze ({gaze.get('assignment_status', 'n/a')})"
            status_color = (0, 200, 255)
        else:
            status = "no gaze on frame"
            status_color = (180, 180, 180)
        cv2.putText(
            canvas, status, (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA
        )

        writer.write(canvas)
        n_written += 1

    if writer is not None:
        writer.release()

    return {
        "out_path": str(out_path),
        "frames_written": n_written,
        "frames_with_gaze": n_with_gaze,
        "frames_with_hit": n_hits,
        "start_s": start_s,
        "end_s": end_s,
        "gaze_radius": gaze_radius,
    }
