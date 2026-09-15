"""Shared detector output contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Detection:
    detection_id: str
    frame_key: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    raw_label: str
    normalized_label: str | None
    score: float
    model_provenance: dict[str, Any] = field(default_factory=dict)
    coordinate_space: str = "native_scene_pixels"

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        # Flatten provenance lightly for CSV
        prov = row.pop("model_provenance", {}) or {}
        row["model_name"] = prov.get("model_name")
        row["checkpoint"] = prov.get("checkpoint")
        row["backend"] = prov.get("backend")
        return row


def detections_to_frame(dets: list[Detection]) -> list[dict[str, Any]]:
    return [d.to_row() for d in dets]
