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

    Searches ``data_dir`` recursively so nested Tobii export folders work.
    Video may sit beside the TSV as ``{stem}_scenevideo.mp4`` or
    ``{stem} scenevideo.mp4``.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir not found: {root}")

    pairs: list[dict[str, str]] = []
    seen: set[str] = set()
    for tsv in sorted(root.rglob("*_data_export.tsv")):
        stem = tsv.name[: -len("_data_export.tsv")]
        if stem in seen:
            continue
        # Tobii "fehlerhaft" = faulty export; skip by default.
        if "fehlerhaft" in stem.lower():
            continue
        parent = tsv.parent
        candidates = [
            parent / f"{stem}_scenevideo.mp4",
            parent / f"{stem} scenevideo.mp4",
            parent / f"{stem}_01_scenevideo.mp4",
        ]
        video = next((p for p in candidates if p.is_file()), None)
        if video is None:
            continue
        seen.add(stem)
        pairs.append(
            {
                "id": stem,
                "tsv_path": str(tsv.resolve()),
                "video_path": str(video.resolve()),
            }
        )
    return pairs


def _resolve_existing_path(path: str | Path, data_dir: str | Path | None = None) -> Path | None:
    """Return resolved file path if it exists (incl. Tobii space/underscore variants)."""
    p = Path(path)
    if not p.is_absolute() and data_dir:
        p = Path(data_dir) / p
    candidates = [p]
    name = p.name
    if "_scenevideo.mp4" in name:
        candidates.append(p.with_name(name.replace("_scenevideo.mp4", " scenevideo.mp4")))
    if " scenevideo.mp4" in name:
        candidates.append(p.with_name(name.replace(" scenevideo.mp4", "_scenevideo.mp4")))
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def resolve_batch_recordings(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Build the recording list from explicit entries and/or discover-first-n."""
    data_dir = cfg.get("data_dir")
    select = cfg.get("select") or {}
    recordings: list[dict[str, str]] = []
    discovered = discover_recording_pairs(data_dir) if data_dir else []
    by_id = {p["id"]: p for p in discovered}

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
        if rec_id and rec_id in by_id:
            recordings.append(by_id[rec_id])
            continue
        if not tsv or not video:
            print(f"[batch] skip preferred recording (missing paths): {item}", flush=True)
            continue
        tsv_p = _resolve_existing_path(tsv, data_dir)
        video_p = _resolve_existing_path(video, data_dir)
        if tsv_p is None or video_p is None:
            print(
                f"[batch] skip preferred recording (not found under data_dir): {rec_id or item}",
                flush=True,
            )
            continue
        recordings.append(
            {"id": rec_id, "tsv_path": str(tsv_p), "video_path": str(video_p)}
        )

    mode = str(select.get("mode", "")).strip().lower()
    if mode in {"first_n", "discover_first_n", "all", "discover_all"}:
        if not data_dir:
            raise ValueError(f"select.mode={mode} requires data_dir")
        have = {r["id"] for r in recordings}
        for pair in discovered:
            if pair["id"] in have:
                continue
            recordings.append(pair)
        if mode in {"first_n", "discover_first_n"}:
            n = int(select.get("n", 5))
            recordings = recordings[:n]
        # mode all / discover_all: keep every discovered pair (preferred first)

    if not recordings:
        raise ValueError(
            "No recordings resolved. Check data_dir for *_data_export.tsv + scenevideo pairs."
        )
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
        tb = traceback.format_exc()
        msg = str(exc).strip() or "(no message)"
        # Bare asserts (common in SAM3) print as "AssertionError:" — include last frame.
        last_frames = [ln for ln in tb.strip().splitlines() if ln.strip()][-4:]
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {msg}"
        if type(exc) is AssertionError and not str(exc).strip():
            row["error"] += " | " + " || ".join(last_frames)
        (out_dir / "batch_error.txt").write_text(
            row["error"] + "\n\n" + tb, encoding="utf-8"
        )
        # Failed detect may leave SAM3 weights in this process's CUDA cache.
        try:
            from gaze_objects.detectors.sam3_meta import release_cuda_memory

            release_cuda_memory()
        except Exception:  # noqa: BLE001
            pass
    return row


def _write_batch_qc(
    out_root: Path,
    rows: list[dict[str, Any]],
    stages: list[str],
    *,
    merge_existing: bool = True,
) -> dict[str, Any]:
    """Persist QC CSV + summary (safe to call after each recording).

    When ``merge_existing`` is True, update rows by ``recording_id`` in any
    existing ``batch_qc_summary.csv`` so subset re-runs do not wipe the corpus QC.
    """
    qc_path = out_root / "batch_qc_summary.csv"
    new_df = pd.DataFrame.from_records(rows)
    if merge_existing and qc_path.is_file() and "recording_id" in new_df.columns:
        old = pd.read_csv(qc_path)
        if "recording_id" in old.columns and len(old):
            updated_ids = set(new_df["recording_id"].astype(str))
            keep = old[~old["recording_id"].astype(str).isin(updated_ids)]
            qc = pd.concat([keep, new_df], ignore_index=True)
        else:
            qc = new_df
    else:
        qc = new_df
    if "recording_id" in qc.columns:
        qc = qc.sort_values("recording_id").reset_index(drop=True)
    qc.to_csv(qc_path, index=False)

    records = qc.to_dict(orient="records")
    summary = {
        "n_recordings": int(len(records)),
        "n_ok": int(sum(1 for r in records if r.get("status") == "ok")),
        "n_error": int(sum(1 for r in records if r.get("status") != "ok")),
        "stages": stages,
        "output_root": str(out_root),
        "qc_csv": str(qc_path),
        "recordings": records,
    }
    write_json(out_root / "batch_summary.json", summary)
    return summary


def rebuild_qc_from_output_root(out_root: str | Path) -> dict[str, Any]:
    """Rebuild corpus QC by scanning per-recording summaries under ``out_root``."""
    root = Path(out_root)
    if not root.is_dir():
        raise FileNotFoundError(f"output_root not found: {root}")

    rows: list[dict[str, Any]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        rec_id = sub.name
        err_path = sub / "batch_error.txt"
        det_path = sub / "detect_summary.json"
        eval_path = sub / "evaluation_summary.json"
        seq_path = sub / "sequences_summary.json"
        cfg_path = sub / "run_config.yaml"

        row: dict[str, Any] = {
            "recording_id": rec_id,
            "tsv_path": None,
            "video_path": None,
            "output_dir": str(sub),
            "status": "ok",
            "error": None,
        }
        if cfg_path.is_file():
            cfg = _load_yaml(cfg_path)
            row["tsv_path"] = cfg.get("tsv_path")
            row["video_path"] = cfg.get("video_path")

        if err_path.is_file() and not eval_path.is_file():
            row["status"] = "error"
            row["error"] = err_path.read_text(encoding="utf-8").splitlines()[0][:500]
        elif not det_path.is_file() and not eval_path.is_file():
            continue

        if det_path.is_file():
            det = json.loads(det_path.read_text(encoding="utf-8"))
            row["n_frames_processed"] = det.get("n_frames_processed")
            row["n_detections"] = det.get("n_detections")
        if eval_path.is_file():
            ev = json.loads(eval_path.read_text(encoding="utf-8"))
            m = ev.get("metrics_conditional_on_assignment") or {}
            row["conditional_accuracy"] = m.get("accuracy")
            row["conditional_n"] = m.get("n")
        if seq_path.is_file():
            seq = json.loads(seq_path.read_text(encoding="utf-8"))
            row["n_fixations"] = seq.get("n_fixations")
            row["n_fixations_labelled"] = seq.get("n_fixations_labelled")
            row["n_sequences"] = seq.get("n_sequences")
        rows.append(row)

    return _write_batch_qc(root, rows, stages=[], merge_existing=False)


def run_batch(config_path: str | Path) -> dict[str, Any]:
    batch_cfg = _load_yaml(config_path)
    recordings = resolve_batch_recordings(batch_cfg)
    stages = list(batch_cfg.get("stages") or ["detect", "assign", "evaluate", "sequences"])
    skip_if_done = bool(batch_cfg.get("skip_if_done", True))
    continue_on_error = bool(batch_cfg.get("continue_on_error", True))

    out_root = Path(batch_cfg.get("output_root", "outputs/batch_sam3_dense"))
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[batch] n_recordings={len(recordings)} output_root={out_root}", flush=True)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for i, recording in enumerate(recordings, start=1):
        rec_cfg = build_recording_config(batch_cfg, recording)
        print(f"[batch] ({i}/{len(recordings)}) start {recording['id']}", flush=True)
        row = run_one_recording(rec_cfg, stages=stages, skip_if_done=skip_if_done)
        rows.append(row)
        # Keep allocator free between recordings even after skip_if_done paths.
        try:
            from gaze_objects.detectors.sam3_meta import release_cuda_memory

            release_cuda_memory()
        except Exception:  # noqa: BLE001
            pass
        print(
            f"[batch] ({i}/{len(recordings)}) done  {recording['id']} status={row['status']}"
            + (f" error={row['error']}" if row.get("error") else ""),
            flush=True,
        )
        summary = _write_batch_qc(out_root, rows, stages, merge_existing=True)
        if row["status"] != "ok" and not continue_on_error:
            break

    return summary
