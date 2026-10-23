"""Gaze-relative part labels on whole-object masks (tip / handle / body).

Does not rely on SAM text for "side handle". Uses the Tobii gaze point inside
an existing object silhouette and the mask's principal axis / centroid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Elongated hand tools only. Angle grinder parts come from SAM text prompts
# (side handle / body / guard / …), not PCA geometry zones.
DEFAULT_PART_OBJECTS = frozenset(
    {
        "screwdriver",
        "phillips screwdriver",
        "flathead screwdriver",
        "wrench",
    }
)


def _mask_bool(mask: Any) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None
    return arr.astype(bool)


def _principal_axis(mask: np.ndarray) -> dict[str, Any] | None:
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return None
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    if not np.isfinite(cov).all():
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    # Consistent orientation
    if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
        axis = -axis
    proj = centered @ axis
    pmin = float(proj.min())
    pmax = float(proj.max())
    span = pmax - pmin
    if span < 1e-3:
        return None
    # Thickness proxy at each end (std of coordinate perpendicular to axis).
    perp = np.array([-axis[1], axis[0]], dtype=np.float64)
    band = max(span * 0.12, 2.0)

    def _end_thickness(center_p: float) -> float:
        sel = np.abs(proj - center_p) <= band
        if int(sel.sum()) < 5:
            return float("inf")
        return float(np.std(centered[sel] @ perp))

    t_lo = _end_thickness(pmin + band)
    t_hi = _end_thickness(pmax - band)
    # Tip = thinner end for elongated hand tools.
    tip_at_low = t_lo <= t_hi
    return {
        "mean": mean,
        "axis": axis,
        "pmin": pmin,
        "pmax": pmax,
        "span": span,
        "tip_at_low": tip_at_low,
        "centroid": mean,
    }


def _axis_t(gx: float, gy: float, stats: dict[str, Any]) -> float:
    """Gaze position along principal axis mapped to [0, 1] with tip at 0."""
    mean = stats["mean"]
    axis = stats["axis"]
    p = float((np.array([gx, gy], dtype=np.float64) - mean) @ axis)
    t = (p - stats["pmin"]) / stats["span"]
    t = float(np.clip(t, 0.0, 1.0))
    if not stats["tip_at_low"]:
        t = 1.0 - t
    return t


def _elongated_part(t: float, *, tip_max: float = 0.33, handle_min: float = 0.67) -> str:
    if t <= tip_max:
        return "tip"
    if t >= handle_min:
        return "handle"
    return "shaft"


def _grinder_part(gx: float, gy: float, mask: np.ndarray, stats: dict[str, Any]) -> str:
    """
    Angle grinder: body near centroid; otherwise which end of the long axis
    (side_region vs head_region). Tip-at-thin end ≈ disc/head side.
    """
    ys, xs = np.where(mask)
    # Median distance of mask pixels from centroid → radius scale
    c = stats["centroid"]
    d = np.hypot(xs.astype(np.float64) - c[0], ys.astype(np.float64) - c[1])
    med = float(np.median(d)) if len(d) else 1.0
    gaze_d = float(np.hypot(gx - c[0], gy - c[1]))
    if med > 1e-3 and gaze_d <= 0.45 * med:
        return "body"
    t = _axis_t(gx, gy, stats)
    if t <= 0.35:
        return "head"  # thinner / disc side of axis
    if t >= 0.65:
        return "side_handle_zone"  # opposite end (often side-handle side)
    return "body"


def classify_gaze_part(
    gaze_x: float,
    gaze_y: float,
    mask: Any,
    object_label: str | None,
    *,
    part_objects: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """
    Return geometric part fields for one gaze+mask hit.

    Keys: part_label, part_zone, part_axis_t, attended_part_label
    attended_part_label is ``object.zone`` (e.g. screwdriver.tip).
    """
    empty = {
        "part_label": None,
        "part_zone": None,
        "part_axis_t": None,
        "attended_part_label": None,
    }
    lab = (object_label or "").strip().lower()
    allowed = part_objects if part_objects is not None else DEFAULT_PART_OBJECTS
    if lab not in allowed:
        return empty
    m = _mask_bool(mask)
    if m is None or not m.any():
        return empty

    h, w = m.shape
    px = int(round(float(gaze_x)))
    py = int(round(float(gaze_y)))
    if px < 0 or py < 0 or px >= w or py >= h or not bool(m[py, px]):
        return empty

    stats = _principal_axis(m)
    if stats is None:
        # Fallback: only body/center vs edge by distance to centroid.
        ys, xs = np.where(m)
        c = np.array([xs.mean(), ys.mean()], dtype=np.float64)
        d = float(np.hypot(gaze_x - c[0], gaze_y - c[1]))
        med = float(np.median(np.hypot(xs - c[0], ys - c[1])))
        zone = "body" if med > 1e-3 and d <= 0.5 * med else "edge"
        part = f"{lab}.{zone}"
        return {
            "part_label": zone,
            "part_zone": zone,
            "part_axis_t": None,
            "attended_part_label": part,
        }

    if lab == "angle grinder":
        zone = _grinder_part(float(gaze_x), float(gaze_y), m, stats)
    elif "screwdriver" in lab or lab == "wrench":
        t = _axis_t(float(gaze_x), float(gaze_y), stats)
        zone = _elongated_part(t)
        return {
            "part_label": zone,
            "part_zone": zone,
            "part_axis_t": float(t),
            "attended_part_label": f"{lab}.{zone}",
        }
    else:
        t = _axis_t(float(gaze_x), float(gaze_y), stats)
        zone = _elongated_part(t)

    t = _axis_t(float(gaze_x), float(gaze_y), stats)
    return {
        "part_label": zone,
        "part_zone": zone,
        "part_axis_t": float(t),
        "attended_part_label": f"{lab}.{zone}",
    }
