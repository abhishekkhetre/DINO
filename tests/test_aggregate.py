"""Tests for corpus sequence aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gaze_objects.aggregate import (
    aggregate_sequences_from_output_root,
    sequence_string_from_table,
)
from gaze_objects.stage_c import resolve_clip_window


def test_sequence_string_from_table():
    seq = pd.DataFrame(
        {
            "sequence_id": [1, 2, 3],
            "attended_label": ["Angle Grinder", "Manual", "Angle Grinder"],
        }
    )
    assert sequence_string_from_table(seq) == "Angle Grinder > Manual > Angle Grinder"


def test_resolve_clip_full(tmp_path: Path):
    eye = pd.DataFrame({"recording_time_s_provisional": [1.5, 10.0, 40.0, 120.25]})
    start, end, mode = resolve_clip_window(eye, {"clip": {"mode": "full"}})
    assert mode == "full"
    assert start == 1.5
    assert end == 120.25


def test_resolve_clip_range():
    eye = pd.DataFrame({"recording_time_s_provisional": [0.0, 100.0]})
    start, end, mode = resolve_clip_window(
        eye, {"clip": {"recording_start_s": 30.0, "recording_end_s": 60.0}}
    )
    assert mode == "range"
    assert start == 30.0
    assert end == 60.0


def test_aggregate_sequences_from_output_root(tmp_path: Path):
    a = tmp_path / "REC_A"
    b = tmp_path / "REC_B"
    a.mkdir()
    b.mkdir()
    pd.DataFrame(
        {
            "sequence_id": [1, 2],
            "attended_label": ["Angle Grinder", "Manual"],
            "n_fixations": [3, 2],
        }
    ).to_csv(a / "attention_sequences.csv", index=False)
    pd.DataFrame(
        {
            "sequence_id": [1, 2, 3],
            "attended_label": ["Manual", "Boxes", "Manual"],
            "n_fixations": [1, 1, 1],
        }
    ).to_csv(b / "attention_sequences.csv", index=False)

    summary = aggregate_sequences_from_output_root(tmp_path)
    assert summary["n_recordings_with_sequences"] == 2
    strings = pd.read_csv(tmp_path / "corpus_sequence_strings.csv")
    by_id = dict(zip(strings.recording_id, strings.sequence_string))
    assert by_id["REC_A"] == "Angle Grinder > Manual"
    assert by_id["REC_B"] == "Manual > Boxes > Manual"
    transitions = pd.read_csv(tmp_path / "corpus_transitions.csv")
    assert len(transitions) >= 2
