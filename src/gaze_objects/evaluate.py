"""Stage D: compare gaze assignments to Tobii AOI reference labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gaze_objects.audit import write_json
from gaze_objects.io import read_tsv
from gaze_objects.reference import extract_reference_labels


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _safe_label(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def evaluate_assignments(
    assignments: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Join model assignments to reference AOI labels on source_row_id.

    Reports both:
    - overall metrics over eligible labelled observations
    - accuracy conditional on making an assignment
    AOI labels are reference annotations, not absolute ground truth.
    """
    ref = reference.copy()
    asg = assignments.copy()
    asg["source_row_id"] = pd.to_numeric(asg["source_row_id"], errors="coerce").astype("Int64")
    ref["source_row_id"] = pd.to_numeric(ref["source_row_id"], errors="coerce").astype("Int64")

    merged = asg.merge(
        ref[
            [
                "source_row_id",
                "selected_reference_label",
                "reference_status",
                "aoi_hits",
                "n_aoi_hits",
                "annotation_review_status",
            ]
        ],
        on="source_row_id",
        how="left",
        validate="one_to_one",
    )

    merged["pred_label"] = merged["selected_normalized_label"].where(
        merged["selected_normalized_label"].notna(),
        merged["selected_raw_label"],
    )
    merged["pred_label"] = merged["pred_label"].map(_safe_label)
    merged["ref_label"] = merged["selected_reference_label"].map(_safe_label)

    # Denominator policies
    has_ref = merged["ref_label"].notna() & (merged["reference_status"] == "single_hit")
    is_assigned = merged["assignment_status"] == "assigned"
    frame_ready = merged["assignment_status"].isin(
        ["assigned", "no_detected_target", "ambiguous"]
    )

    eligible_labelled = has_ref & frame_ready
    assigned_and_labelled = has_ref & is_assigned

    def _agree(mask: pd.Series) -> dict[str, Any]:
        sub = merged.loc[mask]
        n = int(len(sub))
        if n == 0:
            return {"n": 0, "n_agree": 0, "accuracy": None}
        agree = sub["pred_label"] == sub["ref_label"]
        # For no_detected_target / ambiguous, pred_label is null → disagree with a ref label
        n_agree = int(agree.fillna(False).sum())
        return {"n": n, "n_agree": n_agree, "accuracy": float(n_agree / n)}

    # Confusion on assigned ∩ labelled only
    confusion: dict[str, dict[str, int]] = {}
    sub = merged.loc[assigned_and_labelled].copy()
    for _, row in sub.iterrows():
        r = row["ref_label"] or "None"
        p = row["pred_label"] or "None"
        confusion.setdefault(r, {})
        confusion[r][p] = confusion[r].get(p, 0) + 1

    # Per-class precision/recall on assigned ∩ labelled
    labels = sorted(
        {
            *(sub["ref_label"].dropna().unique().tolist() if len(sub) else []),
            *(sub["pred_label"].dropna().unique().tolist() if len(sub) else []),
        }
    )
    per_class: dict[str, Any] = {}
    for label in labels:
        tp = int(((sub["ref_label"] == label) & (sub["pred_label"] == label)).sum())
        fp = int(((sub["ref_label"] != label) & (sub["pred_label"] == label)).sum())
        fn = int(((sub["ref_label"] == label) & (sub["pred_label"] != label)).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) else None
        recall = float(tp / (tp + fn)) if (tp + fn) else None
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if precision is not None and recall is not None and (precision + recall)
            else None
        )
        per_class[label] = {
            "support_ref": int((sub["ref_label"] == label).sum()),
            "predicted": int((sub["pred_label"] == label).sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    status_counts = merged["assignment_status"].value_counts(dropna=False).to_dict()
    summary = {
        "n_joined_rows": int(len(merged)),
        "assignment_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "n_reference_single_hit": int(has_ref.sum()),
        "n_frame_processed_statuses": int(frame_ready.sum()),
        "metrics_eligible_labelled_frame_processed": _agree(eligible_labelled),
        "metrics_conditional_on_assignment": _agree(assigned_and_labelled),
        "per_class_assigned_and_labelled": per_class,
        "confusion_assigned_and_labelled": confusion,
        "notes": [
            "AOI reference labels are not absolute ground truth.",
            "eligible_labelled = single-hit AOI AND assignment_status in {assigned, no_detected_target, ambiguous}.",
            "conditional_on_assignment = single-hit AOI AND status==assigned (abstentions excluded).",
            "COCO-pretrained DINO labels may not match study AOIs; low agreement is informative, not final.",
        ],
    }

    # Compact review table
    review_cols = [
        "source_row_id",
        "frame_index",
        "video_time_s",
        "gaze_x_px",
        "gaze_y_px",
        "assignment_status",
        "pred_label",
        "selected_raw_label",
        "selected_score",
        "ref_label",
        "reference_status",
    ]
    present = [c for c in review_cols if c in merged.columns]
    review = merged.loc[:, present].copy()
    review["labels_agree"] = (review["pred_label"] == review["ref_label"]) & review["pred_label"].notna()
    return review, summary


def run_evaluate(config_path: str | Path) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    out_dir = Path(cfg["output_dir"])
    asg_path = out_dir / "gaze_assignments.csv"
    if not asg_path.is_file():
        raise FileNotFoundError(f"Missing {asg_path}. Run assign first.")

    assignments = pd.read_csv(asg_path)
    df, meta = read_tsv(cfg["tsv_path"], compute_hash=False)
    reference = extract_reference_labels(df, meta["discovered"], eye_only=True)
    reference.to_csv(out_dir / "reference_labels_eval.csv", index=False)

    review, summary = evaluate_assignments(assignments, reference)
    review.to_csv(out_dir / "evaluation_joined.csv", index=False)
    write_json(out_dir / "evaluation_summary.json", summary)
    return summary
