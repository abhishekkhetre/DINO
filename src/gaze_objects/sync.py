"""Timestamp units, offsets and gaze/frame association."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

UnitName = Literal["milliseconds", "microseconds", "seconds"]
UnitStatus = Literal["provisional", "verified"]
TimeMapStatus = Literal["provisional", "verified", "unavailable"]


UNIT_TO_SECONDS = {
    "milliseconds": 1e-3,
    "microseconds": 1e-6,
    "seconds": 1.0,
}


@dataclass
class TimeMapping:
    """
    Explicit recording-to-video time transform.

    Simple offset model:
        video_time_s = recording_time_s - video_start_recording_s

    ``video_start_recording_s`` is the recording time that corresponds to the
    first presented video frame (after optional PTS normalization), not
    necessarily the first gaze timestamp in an export clip.
    """

    timestamp_unit: UnitName
    timestamp_unit_status: UnitStatus
    video_start_recording_s: float
    time_mapping_status: TimeMapStatus
    drift_slope: float | None = None  # optional affine; None => pure offset
    notes: str = ""

    def recording_raw_to_seconds(self, raw: float) -> float:
        return float(raw) * UNIT_TO_SECONDS[self.timestamp_unit]

    def recording_time_to_video_time(self, recording_time_s: float) -> float:
        video_t = recording_time_s - self.video_start_recording_s
        if self.drift_slope is not None:
            video_t = video_t * self.drift_slope
        return video_t

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def associate_gaze_to_frames(
    gaze_times_s: np.ndarray,
    frame_times_s: np.ndarray,
    *,
    max_abs_residual_s: float,
    method: str = "nearest_bounded",
) -> pd.DataFrame:
    """
    Associate each gaze time with a frame using a bounded nearest-neighbour join.

    Unmatched gazes receive frame_index=-1 and status ``unmatched_time``.
    """
    if method != "nearest_bounded":
        raise ValueError(f"Unsupported association method: {method}")

    gaze_times_s = np.asarray(gaze_times_s, dtype=float)
    frame_times_s = np.asarray(frame_times_s, dtype=float)
    n = len(gaze_times_s)
    if len(frame_times_s) == 0:
        return pd.DataFrame(
            {
                "gaze_order": np.arange(n),
                "frame_index": np.full(n, -1, dtype=int),
                "frame_time_s": np.full(n, np.nan),
                "temporal_residual_s": np.full(n, np.nan),
                "association_status": np.full(n, "unmatched_time"),
                "max_abs_residual_s": max_abs_residual_s,
                "method": method,
            }
        )

    # searchsorted based nearest
    idx = np.searchsorted(frame_times_s, gaze_times_s, side="left")
    idx0 = np.clip(idx - 1, 0, len(frame_times_s) - 1)
    idx1 = np.clip(idx, 0, len(frame_times_s) - 1)
    d0 = np.abs(frame_times_s[idx0] - gaze_times_s)
    d1 = np.abs(frame_times_s[idx1] - gaze_times_s)
    choose1 = d1 < d0
    nearest = np.where(choose1, idx1, idx0)
    residual = frame_times_s[nearest] - gaze_times_s
    ok = np.abs(residual) <= max_abs_residual_s

    frame_index = np.where(ok, nearest, -1).astype(int)
    frame_time = np.where(ok, frame_times_s[nearest], np.nan)
    status = np.where(ok, "matched", "unmatched_time")

    # Approximate presentation interval around chosen frame for diagnostics.
    intervals = np.full(n, np.nan)
    if len(frame_times_s) >= 2:
        diffs = np.diff(frame_times_s)
        # interval after frame i, last uses previous
        frame_intervals = np.empty(len(frame_times_s))
        frame_intervals[:-1] = diffs
        frame_intervals[-1] = diffs[-1]
        intervals = np.where(ok, frame_intervals[nearest], np.nan)

    return pd.DataFrame(
        {
            "gaze_order": np.arange(n),
            "frame_index": frame_index,
            "frame_time_s": frame_time,
            "temporal_residual_s": np.where(ok, residual, np.nan),
            "frame_interval_s": intervals,
            "association_status": status,
            "max_abs_residual_s": max_abs_residual_s,
            "method": method,
        }
    )


def representative_gaze_per_frame(
    associations: pd.DataFrame,
    gaze: pd.DataFrame,
    *,
    eligible_col: str = "eligible_gaze_default_policy",
) -> pd.DataFrame:
    """
    Pick one eligible gaze sample per matched frame for display overlays.

    Scientific outputs should retain all eligible samples; this is display-only.
    Policy: among eligible matched samples for a frame, choose the one with
    smallest absolute temporal residual.
    """
    merged = associations.merge(
        gaze.reset_index(drop=True).reset_index(names="_gaze_idx"),
        left_on="gaze_order",
        right_on="_gaze_idx",
        how="left",
    )
    matched = merged[
        (merged["association_status"] == "matched")
        & (merged[eligible_col].fillna(False))
        & (merged["frame_index"] >= 0)
    ].copy()
    if matched.empty:
        return matched
    matched["_abs_res"] = matched["temporal_residual_s"].abs()
    matched = matched.sort_values(["frame_index", "_abs_res", "gaze_order"])
    return matched.groupby("frame_index", as_index=False).first()
