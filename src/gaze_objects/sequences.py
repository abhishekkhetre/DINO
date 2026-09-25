"""Stage E: attended-object sequences from Tobii fixations + Stage C assignments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gaze_objects.audit import write_json


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _safe_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def majority_attended_label(
    fixation_rows: pd.DataFrame,
    *,
    label_policy: str = "assigned_majority",
) -> dict[str, Any]:
    """
    Pick one attended label for a single Tobii fixation.

    assigned_majority: mode of selected_normalized_label among status==assigned.
    Ties broken by higher sum(score), then label name.
    """
    if label_policy != "assigned_majority":
        raise ValueError(f"Unsupported sequences.label_policy: {label_policy}")

    assigned = fixation_rows[fixation_rows["assignment_status"] == "assigned"].copy()
    labels = assigned["selected_normalized_label"].map(_safe_str)
    assigned = assigned.assign(_label=labels)
    assigned = assigned[assigned["_label"].notna()]

    n_samples = int(len(fixation_rows))
    n_assigned = int((fixation_rows["assignment_status"] == "assigned").sum())
    n_ambiguous = int((fixation_rows["assignment_status"] == "ambiguous").sum())
    n_labelled = int(len(assigned))

    if assigned.empty:
        return {
            "attended_label": None,
            "attended_status": "no_assigned_label",
            "n_samples": n_samples,
            "n_assigned": n_assigned,
            "n_ambiguous": n_ambiguous,
            "n_labelled_assigned": 0,
            "label_support": 0,
            "label_fraction": None,
            "mean_score": None,
        }

    assigned["_score"] = pd.to_numeric(assigned.get("selected_score"), errors="coerce").fillna(0.0)
    grouped = (
        assigned.groupby("_label", sort=False)
        .agg(support=("_label", "size"), score_sum=("_score", "sum"))
        .reset_index()
        .sort_values(["support", "score_sum", "_label"], ascending=[False, False, True])
    )
    top = grouped.iloc[0]
    support = int(top["support"])
    return {
        "attended_label": str(top["_label"]),
        "attended_status": "labelled",
        "n_samples": n_samples,
        "n_assigned": n_assigned,
        "n_ambiguous": n_ambiguous,
        "n_labelled_assigned": n_labelled,
        "label_support": support,
        "label_fraction": float(support / n_labelled) if n_labelled else None,
        "mean_score": float(assigned.loc[assigned["_label"] == top["_label"], "_score"].mean()),
    }


def _series_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Always return a numeric Series (empty if column missing)."""
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def build_fixation_table(
    assignments: pd.DataFrame,
    associations: pd.DataFrame,
    *,
    movement_type: str = "Fixation",
    label_policy: str = "assigned_majority",
) -> pd.DataFrame:
    """One row per Tobii eye-movement index of the given type (default Fixation)."""
    asg = assignments.copy()
    assoc = associations.copy()
    asg["source_row_id"] = pd.to_numeric(asg["source_row_id"], errors="coerce").astype("Int64")
    assoc["source_row_id"] = pd.to_numeric(assoc["source_row_id"], errors="coerce").astype("Int64")

    keep_cols = [
        c
        for c in [
            "source_row_id",
            "eye_movement_type",
            "eye_movement_type_index",
            "gaze_event_duration_raw",
            "recording_time_s_provisional",
            "video_time_s",
            "gaze_x_px",
            "gaze_y_px",
        ]
        if c in assoc.columns
    ]
    # Drop overlap so merge does not create video_time_s_x / _y and hide columns.
    overlap = [c for c in keep_cols if c != "source_row_id" and c in asg.columns]
    asg_slim = asg.drop(columns=overlap)
    merged = asg_slim.merge(assoc[keep_cols], on="source_row_id", how="left", validate="one_to_one")

    fix = merged[merged["eye_movement_type"] == movement_type].copy()
    fix["eye_movement_type_index"] = pd.to_numeric(
        fix["eye_movement_type_index"], errors="coerce"
    )
    fix = fix[fix["eye_movement_type_index"].notna()]

    records: list[dict[str, Any]] = []
    for fix_idx, sub in fix.groupby("eye_movement_type_index", sort=True):
        label_info = majority_attended_label(sub, label_policy=label_policy)
        t_rec = _series_numeric(sub, "recording_time_s_provisional")
        t_vid = _series_numeric(sub, "video_time_s")
        dur = _series_numeric(sub, "gaze_event_duration_raw")
        records.append(
            {
                "fixation_index": int(fix_idx),
                "eye_movement_type": movement_type,
                "recording_start_s": float(t_rec.min()) if t_rec.notna().any() else None,
                "recording_end_s": float(t_rec.max()) if t_rec.notna().any() else None,
                "video_start_s": float(t_vid.min()) if t_vid.notna().any() else None,
                "video_end_s": float(t_vid.max()) if t_vid.notna().any() else None,
                # Tobii repeats Gaze event duration on samples — take max, do not sum.
                "gaze_event_duration_raw": float(dur.max()) if dur.notna().any() else None,
                **label_info,
            }
        )
    return pd.DataFrame.from_records(records)


def apply_fixation_quality_gates(
    fixation_table: pd.DataFrame,
    *,
    min_assigned_samples: int = 1,
    min_label_fraction: float = 0.0,
) -> pd.DataFrame:
    """
    Demote weak fixation labels so they do not enter attention sequences.

    Keeps ``attended_label_candidate`` for inspection; clears ``attended_label``
    when gates fail and sets ``attended_status`` to ``below_quality_gate``.
    """
    out = fixation_table.copy()
    if out.empty:
        out["attended_label_candidate"] = pd.Series(dtype=object)
        out["passes_quality_gate"] = pd.Series(dtype=bool)
        return out

    out["attended_label_candidate"] = out["attended_label"]
    labelled = out["attended_status"] == "labelled"
    n_lab = pd.to_numeric(out["n_labelled_assigned"], errors="coerce").fillna(0)
    frac = pd.to_numeric(out["label_fraction"], errors="coerce")
    frac_ok = frac.isna() | (frac >= float(min_label_fraction))
    sample_ok = n_lab >= int(min_assigned_samples)
    passes = labelled & sample_ok & frac_ok
    out["passes_quality_gate"] = passes

    weak = labelled & ~passes
    out.loc[weak, "attended_status"] = "below_quality_gate"
    out.loc[weak, "attended_label"] = None
    return out


def build_attention_sequences(fixation_table: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse consecutive fixations that share the same attended_label.

    Fixations with no label break the run (not merged across gaps).
    """
    if fixation_table.empty:
        return pd.DataFrame(
            columns=[
                "sequence_id",
                "attended_label",
                "n_fixations",
                "fixation_index_start",
                "fixation_index_end",
                "recording_start_s",
                "recording_end_s",
                "video_start_s",
                "video_end_s",
                "total_gaze_event_duration_raw",
            ]
        )

    rows = fixation_table.sort_values("fixation_index").reset_index(drop=True)
    sequences: list[dict[str, Any]] = []
    seq_id = 0
    cur_label: str | None = None
    buf: list[dict[str, Any]] = []

    def _flush() -> None:
        nonlocal seq_id, buf
        if not buf or cur_label is None:
            buf = []
            return
        seq_id += 1
        durations = [
            r["gaze_event_duration_raw"]
            for r in buf
            if r.get("gaze_event_duration_raw") is not None
        ]
        sequences.append(
            {
                "sequence_id": seq_id,
                "attended_label": cur_label,
                "n_fixations": len(buf),
                "fixation_index_start": buf[0]["fixation_index"],
                "fixation_index_end": buf[-1]["fixation_index"],
                "recording_start_s": buf[0].get("recording_start_s"),
                "recording_end_s": buf[-1].get("recording_end_s"),
                "video_start_s": buf[0].get("video_start_s"),
                "video_end_s": buf[-1].get("video_end_s"),
                "total_gaze_event_duration_raw": float(sum(durations)) if durations else None,
            }
        )
        buf = []

    for _, row in rows.iterrows():
        label = _safe_str(row.get("attended_label"))
        rec = row.to_dict()
        if label is None:
            _flush()
            cur_label = None
            continue
        if cur_label is None:
            cur_label = label
            buf = [rec]
            continue
        if label == cur_label:
            buf.append(rec)
        else:
            _flush()
            cur_label = label
            buf = [rec]
    _flush()
    return pd.DataFrame.from_records(sequences)


def run_sequences(config_path: str | Path) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    out_dir = Path(cfg["output_dir"])
    asg_path = out_dir / "gaze_assignments.csv"
    assoc_path = out_dir / "gaze_frame_associations.csv"
    if not asg_path.is_file():
        raise FileNotFoundError(f"Missing {asg_path}. Run assign first.")
    if not assoc_path.is_file():
        raise FileNotFoundError(f"Missing {assoc_path}. Run detect/assign first.")

    seq_cfg = cfg.get("sequences") or {}
    movement_type = str(seq_cfg.get("movement_type", "Fixation"))
    label_policy = str(seq_cfg.get("label_policy", "assigned_majority"))
    min_assigned_samples = int(seq_cfg.get("min_assigned_samples", 1))
    min_label_fraction = float(seq_cfg.get("min_label_fraction", 0.0))

    assignments = pd.read_csv(asg_path)
    associations = pd.read_csv(assoc_path)
    fixations = build_fixation_table(
        assignments,
        associations,
        movement_type=movement_type,
        label_policy=label_policy,
    )
    fixations = apply_fixation_quality_gates(
        fixations,
        min_assigned_samples=min_assigned_samples,
        min_label_fraction=min_label_fraction,
    )
    sequences = build_attention_sequences(fixations)

    fixations.to_csv(out_dir / "fixation_attended_objects.csv", index=False)
    sequences.to_csv(out_dir / "attention_sequences.csv", index=False)

    labelled = fixations[fixations["attended_status"] == "labelled"]
    weak = fixations[fixations["attended_status"] == "below_quality_gate"]
    label_counts = (
        labelled["attended_label"].value_counts(dropna=False).to_dict() if len(labelled) else {}
    )
    seq_label_counts = (
        sequences["attended_label"].value_counts(dropna=False).to_dict() if len(sequences) else {}
    )
    summary = {
        "movement_type": movement_type,
        "label_policy": label_policy,
        "min_assigned_samples": min_assigned_samples,
        "min_label_fraction": min_label_fraction,
        "n_fixations": int(len(fixations)),
        "n_fixations_labelled": int(len(labelled)),
        "n_fixations_below_quality_gate": int(len(weak)),
        "n_sequences": int(len(sequences)),
        "fixation_attended_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "sequence_label_counts": {str(k): int(v) for k, v in seq_label_counts.items()},
        "output_dir": str(out_dir),
        "notes": [
            "Fixations use Tobii Eye movement type index; Gaze event duration is max-per-index (not summed).",
            "attended_label = majority selected_normalized_label among assignment_status==assigned.",
            "Quality gates: min_assigned_samples and min_label_fraction; failures → below_quality_gate (excluded from sequences).",
            "Sequences merge consecutive fixations with the same attended_label; unlabelled/weak fixations break runs.",
        ],
    }
    write_json(out_dir / "sequences_summary.json", summary)
    return summary
