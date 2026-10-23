"""Post-detect filters for granular / part labels."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

# Part / small-hardware labels that should sit on / near a parent angle grinder.
DEFAULT_PART_LABELS = frozenset(
    {
        "grinder body",
        "motor housing",
        "gear housing",
        "side handle",
        "main handle",
        "grinder guard",
        "battery pack",
        "grinding disc",
        "power switch",
        "trigger",
        "spindle lock",
        "clamping flange",
        "flange nut",
        "screw",
    }
)

DEFAULT_PARENT_LABELS = frozenset({"angle grinder"})


def _boxes_overlap(a: dict[str, Any], b: dict[str, Any], *, expand_px: float = 40.0) -> bool:
    """True if axis-aligned boxes overlap after expanding each by ``expand_px``."""
    pad = max(0.0, float(expand_px))
    ax0, ay0 = float(a["x_min"]) - pad, float(a["y_min"]) - pad
    ax1, ay1 = float(a["x_max"]) + pad, float(a["y_max"]) + pad
    bx0, by0 = float(b["x_min"]) - pad, float(b["y_min"]) - pad
    bx1, by1 = float(b["x_max"]) + pad, float(b["y_max"]) + pad
    return (min(ax1, bx1) > max(ax0, bx0)) and (min(ay1, by1) > max(ay0, by0))


def filter_parts_requiring_parent(
    detections: pd.DataFrame,
    *,
    part_labels: Iterable[str] | None = None,
    parent_labels: Iterable[str] | None = None,
    expand_px: float = 40.0,
    label_col: str = "normalized_label",
    frame_col: str = "frame_index",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Drop part detections that do not overlap any parent (angle grinder) on
    the same frame. Whole-object rows are kept unchanged.

    Returns (filtered_df, stats).
    """
    if detections is None or len(detections) == 0:
        return detections if detections is not None else pd.DataFrame(), {
            "n_in": 0,
            "n_out": 0,
            "n_parts_removed": 0,
        }

    parts = {str(x).strip().lower() for x in (part_labels or DEFAULT_PART_LABELS)}
    parents = {str(x).strip().lower() for x in (parent_labels or DEFAULT_PARENT_LABELS)}
    n_in = int(len(detections))
    keep_idx: list[Any] = []
    n_removed = 0

    for fi, sub in detections.groupby(detections[frame_col].astype(int), sort=False):
        parent_rows = [
            r
            for _, r in sub.iterrows()
            if str(r.get(label_col) or "").strip().lower() in parents
        ]
        for idx, row in sub.iterrows():
            lab = str(row.get(label_col) or "").strip().lower()
            if lab not in parts:
                keep_idx.append(idx)
                continue
            if not parent_rows:
                n_removed += 1
                continue
            det = row.to_dict()
            if any(_boxes_overlap(det, p.to_dict(), expand_px=expand_px) for p in parent_rows):
                keep_idx.append(idx)
            else:
                n_removed += 1

    out = detections.loc[keep_idx].reset_index(drop=True)
    stats = {
        "n_in": n_in,
        "n_out": int(len(out)),
        "n_parts_removed": int(n_removed),
        "expand_px": float(expand_px),
    }
    return out, stats
