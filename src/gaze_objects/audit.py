"""Quality checks and audit reports for Tobii exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gaze_objects.io import DiscoveredColumns, read_tsv
from gaze_objects.reference import aoi_coverage_summary, extract_reference_labels, list_aoi_columns


def prepare_eye_tracker_table(
    df: pd.DataFrame,
    discovered: DiscoveredColumns,
    *,
    timestamp_unit_hypothesis: str = "milliseconds",
    media_width: float | None = None,
    media_height: float | None = None,
) -> pd.DataFrame:
    """
    Derived eye-tracker table with quality flags.

    Does not repair invalid or out-of-frame gaze. Timestamp conversion is tagged
    as a hypothesis until verified against export settings / video.
    """
    eye = df[df["Sensor"] == "Eye Tracker"].copy()
    width = media_width
    height = media_height
    if width is None:
        width = float(pd.to_numeric(eye["Recording media width"], errors="coerce").dropna().iloc[0])
    if height is None:
        height = float(pd.to_numeric(eye["Recording media height"], errors="coerce").dropna().iloc[0])

    x = pd.to_numeric(eye["Gaze point X"], errors="coerce")
    y = pd.to_numeric(eye["Gaze point Y"], errors="coerce")
    has_xy = x.notna() & y.notna()
    in_frame = has_xy & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    out_of_frame = has_xy & ~in_frame

    unit_scale = {"milliseconds": 1e-3, "microseconds": 1e-6, "seconds": 1.0}.get(
        timestamp_unit_hypothesis
    )
    if unit_scale is None:
        raise ValueError(f"Unsupported timestamp_unit_hypothesis: {timestamp_unit_hypothesis}")

    ts = pd.to_numeric(eye["Recording timestamp"], errors="coerce")
    out = pd.DataFrame(
        {
            "source_row_id": eye["source_row_id"].astype(int),
            "participant_id": eye["Participant name"],
            "recording_id": eye["Recording name"],
            "recording_timestamp_raw": ts,
            "recording_time_s_provisional": ts * unit_scale,
            "timestamp_unit_hypothesis": timestamp_unit_hypothesis,
            "timestamp_unit_status": "provisional",
            "gaze_x_px": x,
            "gaze_y_px": y,
            "has_gaze_coordinates": has_xy,
            "in_frame": in_frame.fillna(False),
            "out_of_frame": out_of_frame.fillna(False),
            "validity_left": eye["Validity left"],
            "validity_right": eye["Validity right"],
            "eye_movement_type": eye["Eye movement type"],
            "eye_movement_type_index": pd.to_numeric(eye["Eye movement type index"], errors="coerce"),
            "gaze_event_duration_raw": pd.to_numeric(eye["Gaze event duration"], errors="coerce"),
            "gaze_event_duration_unit_hypothesis": timestamp_unit_hypothesis,
            "fixation_point_x": pd.to_numeric(eye.get("Fixation point X"), errors="coerce")
            if "Fixation point X" in eye.columns
            else np.nan,
            "fixation_point_y": pd.to_numeric(eye.get("Fixation point Y"), errors="coerce")
            if "Fixation point Y" in eye.columns
            else np.nan,
            "media_width": width,
            "media_height": height,
        }
    )

    # Both-eyes-valid is informative, not an eligibility requirement by default.
    out["both_eyes_valid"] = (out["validity_left"] == "Valid") & (out["validity_right"] == "Valid")
    out["eligible_gaze_default_policy"] = out["has_gaze_coordinates"] & out["in_frame"]
    return out.reset_index(drop=True)


def _timestamp_stats(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    diffs = s.diff().dropna()
    return {
        "n": int(len(s)),
        "min": float(s.min()),
        "max": float(s.max()),
        "span": float(s.max() - s.min()),
        "median_increment": float(diffs.median()) if len(diffs) else None,
        "min_increment": float(diffs.min()) if len(diffs) else None,
        "max_increment": float(diffs.max()) if len(diffs) else None,
        "n_nonpositive_increments": int((diffs <= 0).sum()) if len(diffs) else 0,
    }


def build_audit_report(
    df: pd.DataFrame,
    meta: dict[str, Any],
    eye: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, Any]:
    discovered: DiscoveredColumns = meta["discovered"]
    sensor_counts = df["Sensor"].replace("", "(empty)").value_counts(dropna=False).to_dict()
    event_counts = df.loc[df["Sensor"].isin(["", "(empty)"]) | (df["Sensor"] == ""), "Event"].value_counts(
        dropna=False
    ).to_dict()
    # Cleaner empty-sensor event accounting
    empty_sensor = df["Sensor"].fillna("") == ""
    empty_sensor_events = df.loc[empty_sensor, "Event"].value_counts(dropna=False).to_dict()

    eye_raw = df[df["Sensor"] == "Eye Tracker"]
    x = pd.to_numeric(eye_raw["Gaze point X"], errors="coerce")
    y = pd.to_numeric(eye_raw["Gaze point Y"], errors="coerce")
    has_xy = x.notna() & y.notna()
    width = meta.get("media_width") or float("nan")
    height = meta.get("media_height") or float("nan")
    in_frame = has_xy & (x >= 0) & (x < width) & (y >= 0) & (y < height)

    validity = (
        eye_raw.assign(_vl=eye_raw["Validity left"], _vr=eye_raw["Validity right"])
        .groupby(["Validity left", "Validity right"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    validity_combos = {
        f"{r['Validity left']}|{r['Validity right']}": int(r["count"]) for _, r in validity.iterrows()
    }

    movement = eye_raw["Eye movement type"].value_counts(dropna=False).to_dict()
    fixation_rows = eye_raw[eye_raw["Eye movement type"] == "Fixation"]
    fixation_indices = pd.to_numeric(fixation_rows["Eye movement type index"], errors="coerce").dropna()
    unique_fixations = sorted({int(v) for v in fixation_indices})

    # Mapping coverage on eye rows for each screenshot id.
    mapping_coverage: dict[str, Any] = {}
    for sid, mapping in discovered.mapping.items():
        def pair_count(x_key: str, y_key: str) -> int:
            xc, yc = mapping.get(x_key), mapping.get(y_key)
            if not xc or not yc:
                return 0
            return int((eye_raw[xc].notna() & eye_raw[yc].notna()).sum())

        mapping_coverage[sid] = {
            "assisted_complete_pairs": pair_count("assisted_x", "assisted_y"),
            "manual_complete_pairs": pair_count("manual_x", "manual_y"),
            "combined_mapped_complete_pairs": pair_count("mapped_x", "mapped_y"),
            "width_col": mapping.get("width"),
            "height_col": mapping.get("height"),
        }

    aoi_summary = aoi_coverage_summary(reference, df, aoi_meta=list_aoi_columns(list(df.columns)))

    # Computer vs recording timestamp equality on this sample.
    rt = pd.to_numeric(df["Recording timestamp"], errors="coerce")
    ct = pd.to_numeric(df["Computer timestamp"], errors="coerce")
    both = rt.notna() & ct.notna()
    equal_share = float((rt[both] == ct[both]).mean()) if both.any() else None

    # Duplicate timestamps within eye tracker only (informational).
    eye_ts = pd.to_numeric(eye_raw["Recording timestamp"], errors="coerce")
    eye_dup = int(eye_ts.duplicated(keep=False).sum())

    # Longest reported fixation duration among fixation eye rows (do not sum across rows).
    dur = pd.to_numeric(fixation_rows["Gaze event duration"], errors="coerce")
    # Per fixation index, take the max reported duration (repeated on samples).
    if len(fixation_rows):
        per_fix = (
            fixation_rows.assign(
                _idx=pd.to_numeric(fixation_rows["Eye movement type index"], errors="coerce"),
                _dur=dur,
            )
            .dropna(subset=["_idx"])
            .groupby("_idx")["_dur"]
            .max()
        )
        longest_fix_dur = float(per_fix.max()) if len(per_fix) else None
    else:
        longest_fix_dur = None

    report: dict[str, Any] = {
        "provenance": {
            "tsv_path": meta["tsv_path"],
            "size_bytes": meta["size_bytes"],
            "sha256": meta["sha256"],
            "n_rows_excluding_header": meta["n_rows"],
            "n_columns": meta["n_columns"],
            "participant_id": meta["participant_id"],
            "recording_id": meta["recording_id"],
            "project_name": meta["project_name"],
            "media_name": meta["media_name"],
            "media_width": meta["media_width"],
            "media_height": meta["media_height"],
            "recording_duration_raw": meta["recording_duration_raw"],
            "fixation_filter": meta["fixation_filter"],
            "screenshot_ids": discovered.screenshot_ids,
        },
        "sensor_counts": {str(k): int(v) for k, v in sensor_counts.items()},
        "empty_sensor_event_counts": {str(k): int(v) for k, v in empty_sensor_events.items()},
        "timing": {
            "all_rows": _timestamp_stats(df["Recording timestamp"]),
            "eye_tracker": _timestamp_stats(eye_raw["Recording timestamp"]),
            "computer_equals_recording_fraction": equal_share,
            "eye_tracker_duplicate_timestamp_row_count": eye_dup,
            "timestamp_unit_hypothesis": "milliseconds",
            "timestamp_unit_status": "provisional",
            "note": (
                "Unit interpretation is a working hypothesis from spacing ~10 and duration "
                "scale; not verified against Tobii export settings or video PTS."
            ),
        },
        "gaze_availability": {
            "eye_tracker_rows": int(len(eye_raw)),
            "rows_with_both_xy": int(has_xy.sum()),
            "rows_with_neither_xy": int((x.isna() & y.isna()).sum()),
            "rows_in_frame": int(in_frame.sum()),
            "rows_out_of_frame": int((has_xy & ~in_frame).sum()),
            "coordinate_availability_pct": float(100.0 * has_xy.mean()) if len(eye_raw) else None,
            "validity_combinations": validity_combos,
            "eligibility_policy": (
                "Default scientific eligibility uses coordinate availability and in-frame "
                "bounds; both-eyes-valid is not required."
            ),
        },
        "movement_types": {str(k): int(v) for k, v in movement.items()},
        "fixations": {
            "n_distinct_fixation_indices": len(unique_fixations),
            "fixation_index_min": unique_fixations[0] if unique_fixations else None,
            "fixation_index_max": unique_fixations[-1] if unique_fixations else None,
            "longest_reported_duration_raw": longest_fix_dur,
            "duration_aggregation_note": (
                "Gaze event duration repeats across samples/sensors; do not sum the column."
            ),
        },
        "mapping_coverage_eye_rows": mapping_coverage,
        "aoi": aoi_summary,
        "prepared_eye_rows": int(len(eye)),
        "reference_rows": int(len(reference)),
    }
    return report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_audit(
    tsv_path: str | Path,
    out_dir: str | Path,
    *,
    compute_hash: bool = True,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, meta = read_tsv(tsv_path, compute_hash=compute_hash)
    discovered: DiscoveredColumns = meta["discovered"]

    eye = prepare_eye_tracker_table(
        df,
        discovered,
        media_width=meta["media_width"],
        media_height=meta["media_height"],
    )
    reference = extract_reference_labels(df, discovered, eye_only=True)
    report = build_audit_report(df, meta, eye, reference)

    reconciliation: dict[str, Any] = {"expected_sha256": expected_sha256}
    if expected_sha256 and meta["sha256"]:
        reconciliation["sha256_match"] = meta["sha256"] == expected_sha256
    report["reconciliation"] = reconciliation

    eye.to_csv(out_dir / "prepared_gaze.csv", index=False)
    reference.to_csv(out_dir / "reference_labels.csv", index=False)
    write_json(out_dir / "audit_report.json", report)

    # Readable summary
    summary_lines = [
        f"TSV: {meta['tsv_path']}",
        f"SHA-256: {meta['sha256']}",
        f"Rows: {meta['n_rows']}  Columns: {meta['n_columns']}",
        f"Participant: {meta['participant_id']}  Recording: {meta['recording_id']}",
        f"Media: {meta['media_name']} ({meta['media_width']}x{meta['media_height']})",
        f"Sensors: {report['sensor_counts']}",
        f"Eye rows with XY: {report['gaze_availability']['rows_with_both_xy']}",
        f"In-frame: {report['gaze_availability']['rows_in_frame']}  "
        f"Out-of-frame: {report['gaze_availability']['rows_out_of_frame']}",
        f"Distinct fixations: {report['fixations']['n_distinct_fixation_indices']}",
        f"AOI positive counts: {report['aoi']['aoi_column_positive_counts']}",
        f"Reference status: {report['aoi']['reference_status_counts']}",
        f"Timestamp unit: provisional milliseconds "
        f"(eye span raw={report['timing']['eye_tracker'].get('span')})",
    ]
    (out_dir / "audit_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    run_meta = {
        "command": "audit",
        "tsv_path": meta["tsv_path"],
        "sha256": meta["sha256"],
        "outputs": [
            "audit_report.json",
            "audit_summary.txt",
            "prepared_gaze.csv",
            "reference_labels.csv",
        ],
    }
    write_json(out_dir / "run_metadata.json", run_meta)
    return report
