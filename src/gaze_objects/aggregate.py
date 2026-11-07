"""Corpus-level attended-object sequence aggregation (gaze order / transitions)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gaze_objects.audit import write_json


def _safe_label(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def sequence_string_from_table(sequences: pd.DataFrame) -> str:
    """Build 'Angle Grinder > Manual > Boxes' from an attention_sequences table."""
    if sequences is None or len(sequences) == 0 or "attended_label" not in sequences.columns:
        return ""
    ordered = sequences.sort_values("sequence_id") if "sequence_id" in sequences.columns else sequences
    labels: list[str] = []
    for lab in ordered["attended_label"].tolist():
        s = _safe_label(lab)
        if s is None:
            continue
        if not labels or labels[-1] != s:
            labels.append(s)
    return " > ".join(labels)


def transitions_from_string(seq: str) -> list[tuple[str, str]]:
    parts = [p.strip() for p in seq.split(">") if p.strip()]
    return list(zip(parts, parts[1:]))


def aggregate_sequences_from_output_root(output_root: str | Path) -> dict[str, Any]:
    """
    Scan ``*/attention_sequences.csv`` under a batch output root.

    Writes:
    - corpus_attention_sequences.csv — all sequence rows + recording_id
    - corpus_sequence_strings.csv — one gaze-order string per recording
    - corpus_transitions.csv — from→to transition counts
    - corpus_sequences_summary.json
    """
    root = Path(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"output_root not found: {root}")

    all_rows: list[pd.DataFrame] = []
    string_rows: list[dict[str, Any]] = []
    transition_counts: dict[tuple[str, str], int] = {}

    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        path = sub / "attention_sequences.csv"
        if not path.is_file():
            continue
        try:
            seq = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        if seq.empty:
            string_rows.append(
                {
                    "recording_id": sub.name,
                    "n_sequences": 0,
                    "sequence_string": "",
                    "n_unique_labels": 0,
                }
            )
            continue
        seq = seq.copy()
        seq.insert(0, "recording_id", sub.name)
        all_rows.append(seq)
        s = sequence_string_from_table(seq)
        labels = [p.strip() for p in s.split(">") if p.strip()]
        string_rows.append(
            {
                "recording_id": sub.name,
                "n_sequences": int(len(seq)),
                "sequence_string": s,
                "n_unique_labels": int(len(set(labels))),
            }
        )
        for a, b in transitions_from_string(s):
            transition_counts[(a, b)] = transition_counts.get((a, b), 0) + 1

    events = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    strings = pd.DataFrame.from_records(string_rows)
    transitions = pd.DataFrame(
        [
            {"from_label": a, "to_label": b, "count": c}
            for (a, b), c in sorted(transition_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
        ]
    )

    events_path = root / "corpus_attention_sequences.csv"
    strings_path = root / "corpus_sequence_strings.csv"
    transitions_path = root / "corpus_transitions.csv"
    events.to_csv(events_path, index=False)
    strings.to_csv(strings_path, index=False)
    transitions.to_csv(transitions_path, index=False)

    # One-row-per-recording results table (QC metrics + gaze-order string).
    results_path = root / "corpus_results.csv"
    qc_path = root / "batch_qc_summary.csv"
    results = strings.copy()
    if qc_path.is_file() and len(results):
        qc = pd.read_csv(qc_path)
        if "recording_id" in qc.columns:
            keep = [
                c
                for c in (
                    "recording_id",
                    "status",
                    "n_frames_processed",
                    "n_detections",
                    "n_fixations",
                    "n_fixations_labelled",
                    "n_sequences",
                    "conditional_accuracy",
                    "conditional_n",
                    "error",
                )
                if c in qc.columns
            ]
            results = qc[keep].merge(results, on="recording_id", how="outer", suffixes=("", "_agg"))
            # Prefer QC n_sequences when both present.
            if "n_sequences_agg" in results.columns:
                results = results.drop(columns=["n_sequences_agg"])
    results.to_csv(results_path, index=False)

    summary = {
        "n_recordings_with_sequences": int(len(strings)),
        "n_recordings_nonempty": int((strings["n_sequences"] > 0).sum()) if len(strings) else 0,
        "n_sequence_events": int(len(events)),
        "n_transition_types": int(len(transitions)),
        "corpus_attention_sequences_csv": str(events_path),
        "corpus_sequence_strings_csv": str(strings_path),
        "corpus_transitions_csv": str(transitions_path),
        "corpus_results_csv": str(results_path),
        "example_sequence_strings": strings.loc[strings["sequence_string"] != "", "sequence_string"]
        .head(10)
        .tolist()
        if len(strings)
        else [],
        "top_transitions": transitions.head(20).to_dict(orient="records") if len(transitions) else [],
        "notes": [
            "sequence_string = ordered attended labels across Stage E sequences "
            "(fine mode: screwdriver / angle grinder / …; AOI mode: Angle Grinder / Manual / …).",
            "Only labelled fixations that pass quality gates enter sequences.",
            "Unlabelled gaps break runs; they do not appear as tokens in the string.",
            "corpus_results.csv joins batch_qc_summary with sequence_string (one row per recording).",
        ],
    }
    write_json(root / "corpus_sequences_summary.json", summary)
    return summary
