"""Tests for fine vs study_aoi label mapping."""

from __future__ import annotations

from gaze_objects.detectors.label_map import normalize_fine_label, normalize_label


def test_study_aoi_collapses_tools():
    assert normalize_label("screwdriver", mode="study_aoi") == "Tools"
    assert normalize_label("angle grinder", mode="study_aoi") == "Angle Grinder"


def test_fine_keeps_tool_names():
    assert normalize_label("screwdriver", mode="fine") == "screwdriver"
    assert normalize_label("wrench", mode="fine") == "wrench"
    assert normalize_label("hex key", mode="fine") == "hex key"
    assert normalize_label("phillips screwdriver", mode="fine") == "screwdriver"


def test_fine_grinder_synonyms():
    assert normalize_fine_label("electric grinder") == "angle grinder"
    assert normalize_fine_label("power tool") == "angle grinder"
    assert normalize_fine_label("disk grinder") == "angle grinder"
    assert normalize_fine_label("angle grinder handle") == "angle grinder"
    assert normalize_fine_label("grinding disc") == "grinding disc"
    assert normalize_fine_label("cutting disc") == "grinding disc"
    assert normalize_fine_label("instruction manual") == "instruction manual"
    assert normalize_fine_label("storage box") == "storage box"
