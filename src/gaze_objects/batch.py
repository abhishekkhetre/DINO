"""Batch runner: dense SAM3 pipeline over multiple Tobii recordings."""

from __future__ import annotations

import json
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from gaze_objects.audit import write_json
from gaze_objects.evaluate import run_evaluate
from gaze_objects.sequences import run_sequences
from gaze_objects.stage_c import run_assign, run_detect


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def discover_recording_pairs(data_dir: str | Path) -> list[dict[str, str]]:
    """
    Pair ``*_data_export.tsv`` with a matching scene video.

    Video name may be ``{stem}_scenevideo.mp4`` or ``{stem} scenevideo.mp4``.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir not found: {root}")

    pairs: list[dict[str, str]] = []
    for tsv in sorted(root.glob("*_data_export.tsv")):
        stem = tsv.name[: -len("_data_export.tsv")]
        candidates = [
            root / f"{stem}_scenevideo.mp4",
            root / f"{stem} scenevideo.mp4",
            root / f"{stem}_01_scenevideo.mp4",  # unlikely but harmless
        ]
        video = next((p for p in candidates if p.is_file()), None)
        if video is None:
            continue
        pairs.append(
            {
                "id": stem,
                "tsv_path": str(tsv.resolve()),
                "video_path": str(video.resolve()),
            }
        )
    return pairs


def _resolve_existing_path(path: str | Path, data_dir: str | Path | None = None) -> Path:
    p = Path(path)
    if not p.is_absolute() and data_dir:
        p = Path(data_dir) / p
    if p.is_file():
        return p.resolve()
    # Common Tobii naming: underscore vs space before scenevideo
    name = p.name
    alts: list[Path] = []
    if "_scenevideo.mp4" in name:
        alts.append(p.with_name(name.replace("_scenevideo.mp4", " scenevideo.mp4")))
    if " scenevideo.mp4" in name:
        alts.append(p.with_name(name.replace(" scenevideo.mp4", "_scenevideo.mp4")))
    for alt in alts:
        if alt.is_file():
            return alt.resolve()
    return p.resolve()


def resolve_batch_recordings(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Build the recording list from explicit entries and/or discover-first-n."""
    data_dir = cfg.get("data_dir")
    select = cfg.get("select") or {}
    recordings: list[dict[str, str]] = []

    for item in cfg.get("recordings") or []:
        rec_id = str(item.get("id") or item.get("stem") or "").strip()
        tsv = item.get("tsv_path") or item.get("tsv")
        video = item.get("video_path") or item.get("video")
        if not rec_id and tsv:
            name = Path(tsv).name
            rec_id = (
                name[: -len("_data_export.tsv")]
                if name.endswith("_data_export.tsv")
                else Path(tsv).stem
            )
        if not tsv or not video:
            raise ValueError(f"Recording entry missing tsv/video: {item}")
        tsv_p = _resolve_existing_path(tsv, data_dir)
        video_p = _resolve_existing_path(video, data_dir)
        recordings.append(
            {"id": rec_id, "tsv_path": str(tsv_p), "video_path": str(video_p)}
        )

    mode = str(select.get("mode", "")).strip().lower()
    if mode in {"first_n", "discover_first_n"}:
        if not data_dir:
            raise ValueError("select.mode=first_n requires data_dir")
        n = int(select.get("n", 5))
        discovered = discover_recording_pairs(data_dir)
        have = {r["id"] for r in recordings}
        for pair in discovered:
            if pair["id"] in have:
                continue
            recordings.append(pair)
            if len(recordings) >= n:
                break
        recordings = recordings[:n]

    if not recordings:
        raise ValueError("No recordings resolved. Set recordings: and/or select: first_n.")
    return recordings


def build_recording_config(
    batch_cfg: dict[str, Any],
    recording: dict[str, str],
) -> dict[str, Any]:
    """Merge batch defaults with one recording's paths."""
    defaults = deepcopy(batch_cfg.get("defaults") or {})
    rec_id = recording["id"]
    out_root = Path(batch_cfg.get("output_root", "outputs/batch_sam3_dense"))
    cfg = defaults
    cfg["tsv_path"] = recording["tsv_path"]
    cfg["video_path"] = recording["video_path"]
    cfg["output_dir"] = str(out_root / rec_id)
    cfg["participant_id"] = cfg.get("participant_id") or rec_id
    cfg["recording_id"] = cfg.get("recording_id") or rec_id
    cfg["_batch_recording_id"] = rec_id
    return cfg


def _write_temp_config(cfg: dict[str, Any], path: Path) -> Path:
    payload = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _safe_get(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = d or {}
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def run_one_recording(
    recording_cfg: dict[str, Any],
    *,
    stages: list[str],
    skip_if_done: bool = True,
) -> dict[str, Any]:
    """Run selected pipeline stages for one recording config dict."""
    rec_id = str(recording_cfg.get("_batch_recording_id") or recording_cfg.get("participant_id"))
    out_dir = Path(recording_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = _write_temp_config(recording_cfg, out_dir / "run_config.yaml")

    row: dict[str, Any] = {
        "recording_id": rec_id,
        "tsv_path": recording_cfg["tsv_path"],
        "video_path": recording_cfg["video_path"],
        "output_dir": str(out_dir),
        "status": "ok",
        "error": None,
    }

    want = [s.strip().lower() for s in stages]
    try:
        if not Path(recording_cfg["tsv_path"]).is_file():
            raise FileNotFoundError(f"TSV missing: {recording_cfg['tsv_path']}")
        if not Path(recording_cfg["video_path"]).is_file():
            raise FileNotFoundError(f"Video missing: {recording_cfg['video_path']}")

        det_summary = None
        if "detect" in want:
            det_done = (out_dir / "detections.csv").is_file() and (out_dir / "detect_summary.json").is_file()
            if skip_if_done and det_done:
                det_summary = json.loads((out_dir / "detect_summary.json").read_text(encoding="utf-8"))
            else:
                det_summary = run_detect(cfg_path)
        elif (out_dir / "detect_summary.json").is_file():
            det_summary = json.loads((out_dir / "detect_summary.json").read_text(encoding="utf-8"))

        if "assign" in want:
            asg_done = (out_dir / "gaze_assignments.csv").is_file()
            if not (skip_if_done and asg_done):
                run_assign(cfg_path)

        eval_summary = None
        if "evaluate" in want:
            ev_done = (out_dir / "evaluation_summary.json").is_file()
            if skip_if_done and ev_done:
                eval_summary = json.loads((out_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
            else:
                eval_summary = run_evaluate(cfg_path)
        elif (out_dir / "evaluation_summary.json").is_file():
            eval_summary = json.loads((out_dir / "evaluation_summary.json").read_text(encoding="utf-8"))

        seq_summary = None
        if "sequences" in want:
            seq_done = (out_dir / "sequences_summary.json").is_file()
            if skip_if_done and seq_done:
                seq_summary = json.loads((out_dir / "sequences_summary.json").read_text(encoding="utf-8"))
            else:
                seq_summary = run_sequences(cfg_path)
        elif (out_dir / "sequences_summary.json").is_file():
            seq_summary = json.loads((out_dir / "sequences_summary.json").read_text(encoding="utf-8"))

        row["n_frames_processed"] = _safe_get(det_summary, "n_frames_processed")
        row["n_detections"] = _safe_get(det_summary, "n_detections")
        row["conditional_accuracy"] = _safe_get(
            eval_summary, "metrics_conditional_on_assignment", "accuracy"
        )
        row["conditional_n"] = _safe_get(eval_summary, "metrics_conditional_on_assignment", "n")
        row["n_fixations"] = _safe_get(seq_summary, "n_fixations")
        row["n_fixations_labelled"] = _safe_get(seq_summary, "n_fixations_labelled")
        row["n_sequences"] = _safe_get(seq_summary, "n_sequences")
    except Exception as exc:  # noqa: BLE001 — batch must continue
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        (out_dir / "batch_error.txt").write_text(
            row["error"] + "\n\n" + traceback.format_exc(), encoding="utf-8"
        )
    return row


def run_batch(config_path: str | Path) -> dict[str, Any]:
    batch_cfg = _load_yaml(config_path)
    recordings = resolve_batch_recordings(batch_cfg)
    stages = list(batch_cfg.get("stages") or ["detect", "assign", "evaluate", "sequences"])
    skip_if_done = bool(batch_cfg.get("skip_if_done", True))
    continue_on_error = bool(batch_cfg.get("continue_on_error", True))

    out_root = Path(batch_cfg.get("output_root", "outputs/batch_sam3_dense"))
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for recording in recordings:
        rec_cfg = build_recording_config(batch_cfg, recording)
        print(f"[batch] start {recording['id']}", flush=True)
        row = run_one_recording(rec_cfg, stages=stages, skip_if_done=skip_if_done)
        rows.append(row)
        print(f"[batch] done  {recording['id']} status={row['status']}", flush=True)
        if row["status"] != "ok" and not continue_on_error:
            break

    qc = pd.DataFrame.from_records(rows)
    qc_path = out_root / "batch_qc_summary.csv"
    qc.to_csv(qc_path, index=False)

    summary = {
        "n_recordings": int(len(rows)),
        "n_ok": int(sum(1 for r in rows if r["status"] == "ok")),
        "n_error": int(sum(1 for r in rows if r["status"] != "ok")),
        "stages": stages,
        "output_root": str(out_root),
        "qc_csv": str(qc_path),
        "recordings": rows,
    }
    write_json(out_root / "batch_summary.json", summary)
    return summary
