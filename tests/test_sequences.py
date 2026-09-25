"""Tests for Stage E fixation → attention sequences."""

from __future__ import annotations

import pandas as pd

from gaze_objects.sequences import (
    build_attention_sequences,
    build_fixation_table,
    majority_attended_label,
)


def test_majority_assigned_label():
    rows = pd.DataFrame(
        [
            {"assignment_status": "assigned", "selected_normalized_label": "Manual", "selected_score": 0.5},
            {"assignment_status": "assigned", "selected_normalized_label": "Manual", "selected_score": 0.6},
            {"assignment_status": "assigned", "selected_normalized_label": "Tools", "selected_score": 0.9},
            {"assignment_status": "ambiguous", "selected_normalized_label": "Tools", "selected_score": 0.9},
        ]
    )
    out = majority_attended_label(rows)
    assert out["attended_label"] == "Manual"
    assert out["label_support"] == 2
    assert out["attended_status"] == "labelled"


def test_build_fixation_and_sequences():
    assignments = pd.DataFrame(
        {
            "source_row_id": [1, 2, 3, 4, 5, 6],
            "assignment_status": [
                "assigned",
                "assigned",
                "assigned",
                "assigned",
                "no_detected_target",
                "assigned",
            ],
            "selected_normalized_label": [
                "Manual",
                "Manual",
                "Tools",
                "Tools",
                None,
                "Manual",
            ],
            "selected_score": [0.8, 0.7, 0.9, 0.85, None, 0.6],
            # Overlaps association columns — must not break merge.
            "video_time_s": [30.0, 30.1, 31.0, 31.1, 32.0, 33.0],
            "recording_time_s_provisional": [30.0, 30.1, 31.0, 31.1, 32.0, 33.0],
        }
    )
    associations = pd.DataFrame(
        {
            "source_row_id": [1, 2, 3, 4, 5, 6],
            "eye_movement_type": ["Fixation"] * 6,
            "eye_movement_type_index": [10, 10, 11, 11, 12, 13],
            "gaze_event_duration_raw": [100, 100, 200, 200, 50, 80],
            "recording_time_s_provisional": [30.0, 30.1, 31.0, 31.1, 32.0, 33.0],
            "video_time_s": [30.0, 30.1, 31.0, 31.1, 32.0, 33.0],
        }
    )
    fix = build_fixation_table(assignments, associations)
    assert list(fix["fixation_index"]) == [10, 11, 12, 13]
    assert fix.loc[fix["fixation_index"] == 10, "attended_label"].iloc[0] == "Manual"
    assert fix.loc[fix["fixation_index"] == 11, "attended_label"].iloc[0] == "Tools"
    assert fix.loc[fix["fixation_index"] == 12, "attended_status"].iloc[0] == "no_assigned_label"
    assert fix.loc[fix["fixation_index"] == 10, "video_start_s"].iloc[0] == 30.0

    seq = build_attention_sequences(fix)
    # Manual(10), Tools(11), gap(12), Manual(13) → 3 sequences
    assert len(seq) == 3
    assert list(seq["attended_label"]) == ["Manual", "Tools", "Manual"]
    assert int(seq.iloc[0]["n_fixations"]) == 1
