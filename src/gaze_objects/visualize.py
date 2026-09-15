"""Overlays and review visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def draw_gaze_marker(
    image_bgr: np.ndarray,
    x: float,
    y: float,
    *,
    src_width: int,
    src_height: int,
    color: tuple[int, int, int] = (0, 0, 255),
    radius: int = 8,
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
    cv2.circle(out, (px, py), radius, color, 2)
    cv2.drawMarker(out, (px, py), color, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    return out


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
