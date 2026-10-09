"""Map detector raw labels/phrases to study category IDs or fine SAM3 labels."""

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

# Grounding-DINO / prompt phrase heuristics (provisional) → Tobii study AOIs.
DEFAULT_PHRASE_TO_STUDY = {
    "angle grinder": "Angle Grinder",
    "grinder": "Angle Grinder",
    "electric grinder": "Angle Grinder",
    "disk grinder": "Angle Grinder",
    "disc grinder": "Angle Grinder",
    "cordless grinder": "Angle Grinder",
    "power tool": "Angle Grinder",
    "angle grinder disc": "Angle Grinder",
    "angle grinder handle": "Angle Grinder",
    "angle grinder guard": "Angle Grinder",
    "grinder disc": "Angle Grinder",
    "grinder handle": "Angle Grinder",
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

# Fine-grained SAM3 approach: keep object-level labels (not Tobii AOI collapse).
# Synonyms only → stable canonical names for sequences.
# Part / appearance phrases map to the whole AG — egocentric frames often
# miss the bare "angle grinder" prompt when hands occlude the body.
FINE_PHRASE_TO_CANONICAL = {
    # whole device (+ egocentric / appearance synonyms)
    "angle grinder": "angle grinder",
    "grinder": "angle grinder",
    "electric grinder": "angle grinder",
    "disk grinder": "angle grinder",
    "disc grinder": "angle grinder",
    "cordless grinder": "angle grinder",
    "power tool": "angle grinder",  # kit's only powered hand tool
    "angle grinder disc": "angle grinder",
    "angle grinder handle": "angle grinder",
    "angle grinder guard": "angle grinder",
    "grinder disc": "angle grinder",
    "grinder handle": "angle grinder",
    # distinct consumable / disc (not housing parts)
    "grinding disc": "grinding disc",
    "cutting disc": "grinding disc",
    "abrasive disc": "grinding disc",
    # manual
    "instruction manual": "instruction manual",
    "manual": "instruction manual",
    "booklet": "instruction manual",
    "instructions": "instruction manual",
    # containers
    "storage box": "storage box",
    "toolbox": "storage box",
    "tool box": "storage box",
    "box": "storage box",
    "boxes": "storage box",
    # hand tools (kept separate for gaze-order analysis)
    "screwdriver": "screwdriver",
    "phillips screwdriver": "screwdriver",
    "flathead screwdriver": "screwdriver",
    "wrench": "wrench",
    "spanner": "wrench",
    "open end wrench": "wrench",
    "combination wrench": "wrench",
    "allen key": "hex key",
    "hex key": "hex key",
    "hex wrench": "hex key",
    "pliers": "pliers",
    "needle nose pliers": "pliers",
    "locking pliers": "pliers",
    "hammer": "hammer",
}


def load_class_definitions(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"categories": [], "status": "missing"}
    path = Path(path)
    if not path.is_file():
        return {"categories": [], "status": "missing_file", "path": str(path)}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_fine_label(raw_label: str) -> str | None:
    """Map a SAM3 concept to a stable fine object name (not Tobii AOI)."""
    if raw_label is None:
        return None
    text = str(raw_label).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in FINE_PHRASE_TO_CANONICAL:
        return FINE_PHRASE_TO_CANONICAL[lower]
    for phrase, canon in FINE_PHRASE_TO_CANONICAL.items():
        if phrase in lower:
            return canon
    # Unknown concept: keep cleaned raw text so sequences still work.
    return lower


def normalize_label(
    raw_label: str,
    class_defs: dict[str, Any] | None = None,
    *,
    mode: str = "study_aoi",
) -> str | None:
    """
    Map a raw detector label.

    Modes:
    - ``study_aoi`` (default): map to Tobii-style AOIs (Angle Grinder / Manual / …).
    - ``fine`` / ``sam3`` / ``identity``: keep fine object labels for gaze-order analysis.
    """
    if raw_label is None:
        return None
    text = str(raw_label).strip()
    if not text:
        return None

    mode_norm = str(mode or "study_aoi").strip().lower()
    if mode_norm in {"fine", "sam3", "identity", "raw"}:
        return normalize_fine_label(text)

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
