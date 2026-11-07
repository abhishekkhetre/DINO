"""Tests for Stage D evaluation denominators."""

from __future__ import annotations

import pandas as pd

from gaze_objects.evaluate import evaluate_assignments


def test_evaluate_separates_overall_and_conditional():
    assignments = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "assignment_status": "assigned",
                "selected_normalized_label": "Manual",
                "selected_raw_label": "book",
            },
            {
                "source_row_id": 2,
                "assignment_status": "assigned",
                "selected_normalized_label": "Tools",
                "selected_raw_label": "scissors",
            },
            {
                "source_row_id": 3,
                "assignment_status": "no_detected_target",
                "selected_normalized_label": None,
                "selected_raw_label": None,
            },
            {
                "source_row_id": 4,
                "assignment_status": "frame_not_processed",
                "selected_normalized_label": None,
                "selected_raw_label": None,
            },
        ]
    )
    reference = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "selected_reference_label": "Manual",
                "reference_status": "single_hit",
                "aoi_hits": "Manual",
                "n_aoi_hits": 1,
                "annotation_review_status": "unverified",
            },
            {
                "source_row_id": 2,
                "selected_reference_label": "Manual",
                "reference_status": "single_hit",
                "aoi_hits": "Manual",
                "n_aoi_hits": 1,
                "annotation_review_status": "unverified",
            },
            {
                "source_row_id": 3,
                "selected_reference_label": "Tools",
                "reference_status": "single_hit",
                "aoi_hits": "Tools",
                "n_aoi_hits": 1,
                "annotation_review_status": "unverified",
            },
            {
                "source_row_id": 4,
                "selected_reference_label": "Tools",
                "reference_status": "single_hit",
                "aoi_hits": "Tools",
                "n_aoi_hits": 1,
                "annotation_review_status": "unverified",
            },
        ]
    )
    _review, summary = evaluate_assignments(assignments, reference)
    # eligible: rows 1,2,3 (frame processed statuses)
    assert summary["metrics_eligible_labelled_frame_processed"]["n"] == 3
    assert summary["metrics_eligible_labelled_frame_processed"]["n_agree"] == 1
    # conditional: only assigned rows 1,2
    assert summary["metrics_conditional_on_assignment"]["n"] == 2


def test_ignore_reference_labels_tools():
    assignments = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "assignment_status": "assigned",
                "selected_normalized_label": "Manual",
                "selected_raw_label": "instruction manual",
            },
            {
                "source_row_id": 2,
                "assignment_status": "assigned",
                "selected_normalized_label": "Manual",
                "selected_raw_label": "instruction manual",
            },
        ]
    )
    reference = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "selected_reference_label": "Tools",
                "reference_status": "single_hit",
                "aoi_hits": "Tools",
                "n_aoi_hits": 1,
                "annotation_review_status": "unverified",
            },
            {
                "source_row_id": 2,
                "selected_reference_label": "Angle Grinder",
                "reference_status": "single_hit",
                "aoi_hits": "Angle Grinder",
                "n_aoi_hits": 1,
                "annotation_review_status": "unverified",
            },
        ]
    )
    _review, summary = evaluate_assignments(
        assignments, reference, ignore_reference_labels=["Tools"]
    )
    # Tools row excluded; only AG→Manual remains in conditional denom
    assert summary["metrics_conditional_on_assignment"]["n"] == 1
    assert summary["metrics_conditional_on_assignment"]["n_agree"] == 0
    assert summary["ignore_reference_labels"] == ["Tools"]
    assert summary["metrics_conditional_on_assignment"]["accuracy"] == 0.0
