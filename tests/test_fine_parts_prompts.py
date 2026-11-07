"""Granular parts prompt list: kit-safe, no grinding disc / bare grinder flood."""

from __future__ import annotations

from pathlib import Path

from gaze_objects.detectors.sam3_meta import load_text_concepts

ROOT = Path(__file__).resolve().parents[1]


def test_parts_prompts_max_granular_no_grinding_disc():
    concepts = load_text_concepts(
        concepts_file=ROOT / "metaSAM3" / "prompts_fine_parts.txt"
    )
    assert "angle grinder" in concepts
    assert "electric grinder" in concepts
    assert "power tool" in concepts
    assert "side handle" in concepts
    assert "grinder guard" in concepts
    assert "grinder body" in concepts
    assert "battery pack" in concepts
    # Flood / FP sources removed for cleaner video
    assert "grinder" not in concepts  # bare word only
    assert "grinding disc" not in concepts
    assert "phillips screwdriver" not in concepts
    assert "screw" not in concepts
    assert "pliers" not in concepts
    assert len(concepts) <= 14
