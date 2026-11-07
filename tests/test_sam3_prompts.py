"""Fine-prompt list: AG synonyms OK; no phantom rare tools."""

from __future__ import annotations

from pathlib import Path

from gaze_objects.detectors.sam3_meta import load_text_concepts

ROOT = Path(__file__).resolve().parents[1]


def test_fine_prompts_cover_grinder_and_drop_phantom_tools():
    concepts = load_text_concepts(
        concepts_file=ROOT / "metaSAM3" / "prompts_fine_objects.txt"
    )
    assert "angle grinder" in concepts
    assert "power tool" in concepts
    assert "electric grinder" in concepts
    assert "instruction manual" in concepts
    assert "screwdriver" in concepts
    # AG family first — recover handheld egocentric misses before other objects.
    assert concepts.index("angle grinder") < concepts.index("screwdriver")
    # Bare "grinder" flooded AM07; kit phantoms polluted sheets.
    assert "grinder" not in concepts
    assert "disk grinder" not in concepts
    assert "pliers" not in concepts
    assert "hex key" not in concepts
    assert "hammer" not in concepts
    assert "cutting disc" not in concepts
    assert "grinding disc" not in concepts
    assert len(concepts) <= 10
