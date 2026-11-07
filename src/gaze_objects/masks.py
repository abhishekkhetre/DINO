"""Persist / load SAM binary masks keyed by detection_id."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


MASK_ARCHIVE_NAME = "detection_masks.npz"


def masks_archive_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / MASK_ARCHIVE_NAME


def save_detection_masks(
    out_dir: str | Path,
    masks_by_id: dict[str, np.ndarray],
) -> Path | None:
    """Write uint8 HxW masks to a compressed npz. Returns path or None if empty."""
    if not masks_by_id:
        return None
    path = masks_archive_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(k): np.asarray(v, dtype=np.uint8)
        for k, v in masks_by_id.items()
        if v is not None
    }
    if not payload:
        return None
    np.savez_compressed(path, **payload)
    return path


def load_detection_masks(out_dir: str | Path) -> dict[str, np.ndarray]:
    """Load masks archive → {detection_id: bool HxW}."""
    path = masks_archive_path(out_dir)
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].astype(bool) for k in data.files}


def attach_masks_to_detections(
    detections: list[dict[str, Any]],
    masks_by_id: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Copy detections and attach ``mask`` arrays when present."""
    out: list[dict[str, Any]] = []
    for det in detections:
        d = dict(det)
        mid = str(d.get("detection_id", ""))
        if mid in masks_by_id:
            d["mask"] = masks_by_id[mid]
        out.append(d)
    return out


def point_in_mask(x: float, y: float, mask: np.ndarray | None) -> bool:
    """True if integer pixel (x,y) is inside a boolean mask (row=y, col=x)."""
    if mask is None:
        return False
    arr = np.asarray(mask)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        return False
    h, w = arr.shape
    px = int(round(float(x)))
    py = int(round(float(y)))
    if px < 0 or py < 0 or px >= w or py >= h:
        return False
    return bool(arr[py, px])
