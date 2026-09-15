"""Mock detector for CPU/laptop testing of the Stage C data path."""

from __future__ import annotations

from typing import Any

import numpy as np

from gaze_objects.detectors import Detection
from gaze_objects.detectors.label_map import load_class_definitions, normalize_label


def mock_detect_frame(
    frame_bgr: np.ndarray,
    frame_key: str,
    *,
    class_map_file: str | None = None,
) -> list[Detection]:
    """
    Produce a few deterministic fake boxes from image size.

    Not for scientific claims — only to exercise assignment/export on machines
    without CUDA/DINO.
    """
    h, w = frame_bgr.shape[:2]
    class_defs = load_class_definitions(class_map_file)
    raw_boxes = [
        ("Manual", 0.55 * w, 0.35 * h, 0.95 * w, 0.95 * h, 0.91),
        ("Tools", 0.05 * w, 0.55 * h, 0.35 * w, 0.95 * h, 0.72),
        ("Boxes", 0.35 * w, 0.05 * h, 0.65 * w, 0.35 * h, 0.64),
    ]
    dets: list[Detection] = []
    for i, (raw, x0, y0, x1, y1, score) in enumerate(raw_boxes):
        dets.append(
            Detection(
                detection_id=f"{frame_key}_mock_{i}",
                frame_key=frame_key,
                x_min=float(x0),
                y_min=float(y0),
                x_max=float(x1),
                y_max=float(y1),
                raw_label=raw,
                normalized_label=normalize_label(raw, class_defs),
                score=float(score),
                model_provenance={
                    "backend": "mock",
                    "model_name": "mock_boxes",
                    "checkpoint": None,
                },
            )
        )
    return dets


def mock_provenance() -> dict[str, Any]:
    return {"backend": "mock", "note": "Synthetic boxes for pipeline testing only"}
