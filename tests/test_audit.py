"""Tests for TSV discovery, flags and reference extraction (synthetic fixtures)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gaze_objects.io import discover_columns, file_sha256, validate_required_columns
from gaze_objects.reference import extract_reference_labels, list_aoi_columns
from gaze_objects.audit import prepare_eye_tracker_table


FIXTURE_HEADER = [
    "Recording timestamp",
    "Computer timestamp",
    "Sensor",
    "Project name",
    "Export date",
    "Participant name",
    "Recording name",
    "Recording date",
    "Recording date UTC",
    "Recording start time",
    "Recording start time UTC",
    "Recording duration",
    "Recording Fixation filter name",
    "Event",
    "Event value",
    "Gaze point X",
    "Gaze point Y",
    "Validity left",
    "Validity right",
    "Assisted mapping gaze point X [Screenshot TEST]",
    "Assisted mapping gaze point Y [Screenshot TEST]",
    "Manually mapped gaze point X [Screenshot TEST]",
    "Manually mapped gaze point Y [Screenshot TEST]",
    "Mapped gaze point X [Screenshot TEST]",
    "Mapped gaze point Y [Screenshot TEST]",
    "Assisted mapping gaze point score[Screenshot TEST]",
    "Recording media name",
    "Recording media width",
    "Recording media height",
    "Width [Screenshot TEST]",
    "Height [Screenshot TEST]",
    "Eye movement type",
    "Gaze event duration",
    "Eye movement type index",
    "Fixation point X",
    "Fixation point Y",
    "AOI hit [Recording 31 - Manual]",
    "AOI hit [Screenshot TEST - Angle Grinder]",
    "AOI hit [Screenshot TEST - Boxes]",
    "AOI hit [Screenshot TEST - Manual]",
    "AOI hit [Screenshot TEST - Tools]",
]


def _row(**overrides):
    base = {c: "" for c in FIXTURE_HEADER}
    base.update(
        {
            "Recording timestamp": "1000",
            "Computer timestamp": "1000",
            "Sensor": "Eye Tracker",
            "Participant name": "TESTPART",
            "Recording name": "Recording 1",
            "Recording duration": "5000",
            "Recording Fixation filter name": "Tobii I-VT (Attention)",
            "Gaze point X": "100",
            "Gaze point Y": "200",
            "Validity left": "Valid",
            "Validity right": "Valid",
            "Recording media name": "scenevideo.mp4",
            "Recording media width": "1920",
            "Recording media height": "1080",
            "Eye movement type": "Fixation",
            "Gaze event duration": "120",
            "Eye movement type index": "1",
            "AOI hit [Recording 31 - Manual]": "",
            "AOI hit [Screenshot TEST - Angle Grinder]": "0",
            "AOI hit [Screenshot TEST - Boxes]": "0",
            "AOI hit [Screenshot TEST - Manual]": "0",
            "AOI hit [Screenshot TEST - Tools]": "0",
        }
    )
    base.update(overrides)
    return base


def test_discover_columns_finds_screenshot_and_empty_recording31():
    discovered = discover_columns(FIXTURE_HEADER)
    assert discovered.screenshot_ids == ["Screenshot TEST"]
    aoi = list_aoi_columns(FIXTURE_HEADER)
    labels = {a["label"]: a for a in aoi}
    assert "Manual" in labels or any(a["label"] == "Manual" for a in aoi)
    # Recording 31 column must remain discoverable and marked non-screenshot.
    rec31 = [a for a in aoi if a["column"].startswith("AOI hit [Recording 31")]
    assert len(rec31) == 1
    assert rec31[0]["is_screenshot_source"] is False


def test_validate_required_columns():
    missing = validate_required_columns(["Sensor"])
    assert "Recording timestamp" in missing


def test_out_of_frame_and_invalid_not_repaired(tmp_path: Path):
    rows = [
        _row(**{"Gaze point X": "100", "Gaze point Y": "200"}),
        _row(**{"Recording timestamp": "1010", "Gaze point X": "50", "Gaze point Y": "-3"}),
        _row(**{"Recording timestamp": "1020", "Gaze point X": "", "Gaze point Y": "", "Validity left": "Invalid", "Validity right": "Invalid"}),
        _row(**{"Recording timestamp": "1030", "Sensor": "Gyroscope", "Gaze point X": "", "Gaze point Y": ""}),
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "source_row_id", range(len(df)))
    # numeric convert like io.read_tsv
    for col in ["Recording timestamp", "Gaze point X", "Gaze point Y", "Recording media width", "Recording media height", "Gaze event duration", "Eye movement type index"]:
        df[col] = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")
    discovered = discover_columns(list(df.columns))
    eye = prepare_eye_tracker_table(df, discovered, media_width=1920, media_height=1080)
    assert len(eye) == 3
    assert bool(eye.loc[0, "in_frame"]) is True
    assert bool(eye.loc[1, "out_of_frame"]) is True
    assert bool(eye.loc[1, "eligible_gaze_default_policy"]) is False
    # Negative Y must not be clamped to an object / in-frame.
    assert eye.loc[1, "gaze_y_px"] == -3
    assert bool(eye.loc[2, "has_gaze_coordinates"]) is False


def test_reference_prefers_screenshot_aoi_not_empty_recording31():
    rows = [
        _row(**{"AOI hit [Screenshot TEST - Manual]": "1", "Mapped gaze point X [Screenshot TEST]": "10", "Mapped gaze point Y [Screenshot TEST]": "20"}),
        _row(**{"Recording timestamp": "1010", "AOI hit [Screenshot TEST - Angle Grinder]": "1", "AOI hit [Screenshot TEST - Boxes]": "1", "Mapped gaze point X [Screenshot TEST]": "1", "Mapped gaze point Y [Screenshot TEST]": "2"}),
        _row(**{"Recording timestamp": "1020", "Mapped gaze point X [Screenshot TEST]": "1", "Mapped gaze point Y [Screenshot TEST]": "2"}),
        _row(**{"Recording timestamp": "1030"}),
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "source_row_id", range(len(df)))
    for col in df.columns:
        if col.startswith("AOI hit") or "gaze point" in col.lower() or col in ["Recording timestamp", "Gaze point X", "Gaze point Y", "Mapped gaze point X [Screenshot TEST]", "Mapped gaze point Y [Screenshot TEST]"]:
            df[col] = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")
    discovered = discover_columns(list(df.columns))
    # Ensure mapping numeric cols exist
    for c in ["Mapped gaze point X [Screenshot TEST]", "Mapped gaze point Y [Screenshot TEST]"]:
        df[c] = pd.to_numeric(df[c].replace("", pd.NA), errors="coerce")
    ref = extract_reference_labels(df, discovered, eye_only=True)
    assert ref.loc[0, "selected_reference_label"] == "Manual"
    assert ref.loc[0, "reference_status"] == "single_hit"
    assert ref.loc[1, "reference_status"] == "multiple_hits"
    assert ref.loc[2, "reference_status"] == "zero_hit_with_mapping"
    assert ref.loc[3, "reference_status"] == "zero_hit_without_mapping"


def test_file_sha256_roundtrip(tmp_path: Path):
    path = tmp_path / "tiny.tsv"
    path.write_bytes(b"a\tb\n1\t2\n")
    digest = file_sha256(path)
    assert len(digest) == 64
    assert digest == file_sha256(path)
