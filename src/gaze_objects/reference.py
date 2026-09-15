"""Candidate reference AOI labels, isolated from inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gaze_objects.io import DiscoveredColumns, AOI_HIT_PATTERN


def _label_from_aoi_header(col: str) -> tuple[str, str, bool]:
    """Return (label, raw_bracket, is_screenshot_source)."""
    match = AOI_HIT_PATTERN.match(col)
    raw = match.group(1) if match else col
    is_screenshot = raw.startswith("Screenshot ")
    if " - " in raw:
        label = raw.rsplit(" - ", 1)[1].strip()
    else:
        label = raw.strip()
    return label, raw, is_screenshot


def list_aoi_columns(columns: list[str]) -> list[dict[str, Any]]:
    items = []
    for col in columns:
        if not col.startswith("AOI hit ["):
            continue
        label, raw, is_screenshot = _label_from_aoi_header(col)
        items.append(
            {
                "column": col,
                "label": label,
                "raw_bracket": raw,
                "is_screenshot_source": is_screenshot,
            }
        )
    return items


def extract_reference_labels(
    df: pd.DataFrame,
    discovered: DiscoveredColumns,
    *,
    eye_only: bool = True,
) -> pd.DataFrame:
    """
    Build a reference-label table from AOI hit columns.

    Screenshot AOIs are preferred for ``selected_reference_label`` when both
    screenshot and other-recording columns exist for similar names. Zero-hit
    rows are not forced to a verified background label.
    """
    work = df.copy()
    if eye_only:
        work = work[work["Sensor"] == "Eye Tracker"].copy()

    aoi_meta = list_aoi_columns(list(work.columns))
    # Prefer screenshot columns for primary study labels.
    screenshot_aois = [m for m in aoi_meta if m["is_screenshot_source"]]
    primary = screenshot_aois if screenshot_aois else aoi_meta

    hit_matrix = pd.DataFrame(index=work.index)
    for meta in primary:
        col = meta["column"]
        values = pd.to_numeric(work[col], errors="coerce")
        hit_matrix[meta["label"]] = values.eq(1.0)

    n_hits = hit_matrix.sum(axis=1).astype(int)
    # Selected label only when exactly one primary hit.
    selected = pd.Series(pd.NA, index=work.index, dtype="object")
    single_mask = n_hits == 1
    if primary:
        for label in hit_matrix.columns:
            selected.loc[single_mask & hit_matrix[label]] = label

    mapping_available = pd.Series(False, index=work.index)
    assisted_available = pd.Series(False, index=work.index)
    manual_available = pd.Series(False, index=work.index)
    for mapping in discovered.mapping.values():
        mx, my = mapping.get("mapped_x"), mapping.get("mapped_y")
        if mx and my and mx in work.columns and my in work.columns:
            mapping_available |= work[mx].notna() & work[my].notna()
        ax, ay = mapping.get("assisted_x"), mapping.get("assisted_y")
        if ax and ay and ax in work.columns and ay in work.columns:
            assisted_available |= work[ax].notna() & work[ay].notna()
        manx, many = mapping.get("manual_x"), mapping.get("manual_y")
        if manx and many and manx in work.columns and many in work.columns:
            manual_available |= work[manx].notna() & work[many].notna()

    status = np.where(
        n_hits == 1,
        "single_hit",
        np.where(
            n_hits > 1,
            "multiple_hits",
            np.where(mapping_available, "zero_hit_with_mapping", "zero_hit_without_mapping"),
        ),
    )

    aoi_hits = hit_matrix.apply(
        lambda row: "|".join([c for c in hit_matrix.columns if bool(row[c])]),
        axis=1,
    )

    out = pd.DataFrame(
        {
            "source_row_id": work["source_row_id"].astype(int),
            "participant_id": work["Participant name"],
            "recording_id": work["Recording name"],
            "recording_timestamp_raw": work["Recording timestamp"],
            "sensor": work["Sensor"],
            "n_aoi_hits": n_hits.to_numpy(),
            "aoi_hits": aoi_hits.to_numpy(),
            "selected_reference_label": selected.to_numpy(),
            "reference_status": status,
            "combined_mapping_available": mapping_available.to_numpy(),
            "assisted_mapping_available": assisted_available.to_numpy(),
            "manual_mapping_available": manual_available.to_numpy(),
            "annotation_review_status": "unverified",
        }
    )
    return out.reset_index(drop=True)


def aoi_coverage_summary(
    reference: pd.DataFrame,
    df: pd.DataFrame,
    *,
    aoi_meta: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize AOI coverage using eye-tracker denominators."""
    eye_df = df[df["Sensor"] == "Eye Tracker"]
    if aoi_meta is None:
        aoi_meta = list_aoi_columns(list(df.columns))

    positive_counts: dict[str, int] = {}
    nonempty_counts: dict[str, int] = {}
    for meta in aoi_meta:
        col = meta["column"]
        values = pd.to_numeric(eye_df[col], errors="coerce")
        positive_counts[meta["column"]] = int(values.eq(1.0).sum())
        nonempty_counts[meta["column"]] = int(values.notna().sum())

    status_counts = reference["reference_status"].value_counts(dropna=False).to_dict()
    selected_counts = (
        reference["selected_reference_label"].value_counts(dropna=False).to_dict()
        if len(reference)
        else {}
    )
    return {
        "n_eye_rows": int(len(eye_df)),
        "aoi_column_positive_counts": positive_counts,
        "aoi_column_nonempty_counts": nonempty_counts,
        "selected_label_counts": {str(k): int(v) for k, v in selected_counts.items()},
        "reference_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "aoi_columns": aoi_meta,
    }
