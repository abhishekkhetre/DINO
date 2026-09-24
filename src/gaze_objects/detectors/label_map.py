"""Map detector raw labels/phrases to study category IDs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Provisional COCO-name heuristics for pilot only — confirm with supervisor.
# These do NOT claim angle-grinder domain accuracy.
DEFAULT_COCO_TO_STUDY = {
    "book": "Manual",
    "laptop": None,  # do not invent
    "scissors": "Tools",
    "knife": "Tools",
    "fork": "Tools",
    "spoon": "Tools",
    "bottle": None,
    "cup": None,
    "bowl": None,
    "handbag": None,
    "suitcase": "Boxes",
    "backpack": None,
    "box": "Boxes",  # not always a COCO name; kept for mock/Grounding later
    "tv": None,
    "cell phone": None,
    "remote": "Tools",
    "toothbrush": None,
    "hair drier": None,
}

# Grounding-DINO / prompt phrase heuristics (provisional).
DEFAULT_PHRASE_TO_STUDY = {
    "angle grinder": "Angle Grinder",
    "grinder": "Angle Grinder",
    "electric grinder": "Angle Grinder",
    "disk grinder": "Angle Grinder",
    "disc grinder": "Angle Grinder",
    "instruction manual": "Manual",
    "manual": "Manual",
    "booklet": "Manual",
    "instructions": "Manual",
    "instruction": "Manual",  # Grounding often returns truncated prompt token
    "storage box": "Boxes",
    "box": "Boxes",
    "boxes": "Boxes",
    "bin": "Boxes",
    "screwdriver": "Tools",
    "wrench": "Tools",
    "spanner": "Tools",
    "tool": "Tools",
    "tools": "Tools",
    "hammer": "Tools",
    "plier": "Tools",
    "pliers": "Tools",
}


def load_class_definitions(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"categories": [], "status": "missing"}
    path = Path(path)
    if not path.is_file():
        return {"categories": [], "status": "missing_file", "path": str(path)}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_label(raw_label: str, class_defs: dict[str, Any] | None = None) -> str | None:
    """
    Map a raw detector label to a study AOI-style name when possible.

    Returns None if no explicit mapping exists (keep raw_label separately).
    """
    if raw_label is None:
        return None
    text = str(raw_label).strip()
    if not text:
        return None

    known = {"Angle Grinder", "Boxes", "Manual", "Tools"}
    if text in known:
        return text
    lower = text.lower()
    for k in known:
        if k.lower() == lower:
            return k

    # From class_definitions normalized_id reverse map
    if class_defs:
        for cat in class_defs.get("categories", []) or []:
            if cat.get("raw_name") == text:
                return cat.get("raw_name")
            if str(cat.get("normalized_id", "")).lower() == lower:
                return cat.get("raw_name")

    mapped = DEFAULT_COCO_TO_STUDY.get(lower)
    if mapped is not None or lower in DEFAULT_COCO_TO_STUDY:
        return mapped

    # Phrase contains / exact match for Grounding DINO outputs
    if lower in DEFAULT_PHRASE_TO_STUDY:
        return DEFAULT_PHRASE_TO_STUDY[lower]
    for phrase, study in DEFAULT_PHRASE_TO_STUDY.items():
        if phrase in lower:
            return study
    return None
