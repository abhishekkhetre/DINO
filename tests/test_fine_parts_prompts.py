"""Granular parts prompt list: kit-safe max detail, no grinding disc."""

from __future__ import annotations

from pathlib import Path

from gaze_objects.detectors.sam3_meta import load_text_concepts

ROOT = Path(__file__).resolve().parents[1]


def test_parts_prompts_max_granular_no_grinding_disc():
    concepts = load_text_concepts(
        concepts_file=ROOT / "metaSAM3" / "prompts_fine_parts.txt"
    )
    assert "angle grinder" in concepts
    assert "side handle" in concepts
    assert "grinder guard" in concepts
    assert "grinder body" in concepts
    assert "battery pack" in concepts
    assert "screw" in concepts
    assert "phillips screwdriver" in concepts
    # Not in study videos / kit FPs
    assert "grinding disc" not in concepts
    assert "pliers" not in concepts
    assert "hammer" not in concepts
    assert len(concepts) <= 16
