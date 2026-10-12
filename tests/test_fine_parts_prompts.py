"""Parts-tier prompt list stays lean and includes coarse AG parts."""

from __future__ import annotations

from pathlib import Path

from gaze_objects.detectors.sam3_meta import load_text_concepts

ROOT = Path(__file__).resolve().parents[1]


def test_parts_prompts_include_coarse_ag_parts():
    concepts = load_text_concepts(
        concepts_file=ROOT / "metaSAM3" / "prompts_fine_parts.txt"
    )
    assert "angle grinder" in concepts
    assert "side handle" in concepts
    assert "grinder guard" in concepts
    assert "grinding disc" in concepts
    assert "grinder body" in concepts
    assert "battery pack" in concepts
    # Screws commented out by default (FP risk).
    assert "screw" not in concepts
    assert "pliers" not in concepts
    assert len(concepts) <= 16
