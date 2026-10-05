"""Fine-prompt list stays lean for 12GB VRAM (one forward pass per concept)."""

from __future__ import annotations

from pathlib import Path

from gaze_objects.detectors.sam3_meta import load_text_concepts

ROOT = Path(__file__).resolve().parents[1]


def test_fine_prompts_are_lean():
    concepts = load_text_concepts(
        concepts_file=ROOT / "metaSAM3" / "prompts_fine_objects.txt"
    )
    assert "angle grinder" in concepts
    assert "instruction manual" in concepts
    assert "screwdriver" in concepts
    # Synonym / extra disc prompts inflate VRAM — keep them out of the lean set.
    assert "grinder" not in concepts
    assert "electric grinder" not in concepts
    assert "cutting disc" not in concepts
    assert len(concepts) <= 10
