"""Stage C orchestration: select frames, detect, assign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from gaze_objects.assign import assign_table
from gaze_objects.audit import prepare_eye_tracker_table, write_json
from gaze_objects.detectors import Detection
from gaze_objects.io import read_tsv
from gaze_objects.sync import TimeMapping, associate_gaze_to_frames
from gaze_objects.video import build_frame_table, iter_frames_with_time


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_synced_gaze(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, meta = read_tsv(cfg["tsv_path"], compute_hash=bool(cfg.get("compute_hash", True)))
    eye = prepare_eye_tracker_table(
        df,
        meta["discovered"],
        timestamp_unit_hypothesis=cfg.get("timestamp_unit", "milliseconds"),
        media_width=meta["media_width"],
        media_height=meta["media_height"],
    )
    mapping = TimeMapping(
        timestamp_unit=cfg.get("timestamp_unit", "milliseconds"),
        timestamp_unit_status=cfg.get("timestamp_unit_status", "provisional"),
        video_start_recording_s=float(cfg.get("video_start_recording_s", 0.0)),
        time_mapping_status=cfg.get("time_mapping_status", "provisional"),
    )
    clip = cfg.get("clip", {})
    rec_start = float(clip["recording_start_s"])
    rec_end = float(clip["recording_end_s"])
    eye_clip = eye[
        (eye["recording_time_s_provisional"] >= rec_start)
        & (eye["recording_time_s_provisional"] <= rec_end)
    ].copy()
    eye_clip["video_time_s"] = eye_clip["recording_time_s_provisional"].map(
        mapping.recording_time_to_video_time
    )

    video_start_clip = mapping.recording_time_to_video_time(rec_start)
    video_end_clip = mapping.recording_time_to_video_time(rec_end)
    max_residual = float(cfg.get("max_abs_residual_s", 0.05))
    pad = max_residual + 0.05
    frames = build_frame_table(
        cfg["video_path"],
        normalize_pts_to_first=True,
        start_s=max(0.0, video_start_clip - pad),
        end_s=video_end_clip + pad,
    )
    frame_times = [f["video_time_s"] for f in frames]
    frame_indices = [f["frame_index"] for f in frames]
    assoc = associate_gaze_to_frames(
        eye_clip["video_time_s"].to_numpy(),
        frame_times,
        max_abs_residual_s=max_residual,
    )
    mapped_idx = assoc["frame_index"].to_numpy().copy()
    absolute = mapped_idx.copy()
    absolute[:] = -1
    ok = mapped_idx >= 0
    if ok.any():
        import numpy as np

        absolute[ok] = np.asarray(frame_indices, dtype=int)[mapped_idx[ok]]
    assoc["frame_index"] = absolute

    eye_clip = eye_clip.reset_index(drop=True)
    joined = assoc.join(eye_clip, how="left")
    ctx = {
        "meta": meta,
        "mapping": mapping.to_dict(),
        "video_start_clip": video_start_clip,
        "video_end_clip": video_end_clip,
    }
    return joined, ctx


def select_frame_indices(joined: pd.DataFrame, cfg: dict[str, Any]) -> list[int]:
    sel = cfg.get("frame_selection", {})
    mode = sel.get("mode", "unique_matched_frames")
    stride = int(sel.get("stride", 1))
    max_frames = sel.get("max_frames")
    if mode != "unique_matched_frames":
        raise ValueError(f"Unsupported frame_selection.mode: {mode}")

    matched = joined[
        (joined["association_status"] == "matched")
        & (joined["frame_index"] >= 0)
        & (joined["eligible_gaze_default_policy"].fillna(False))
    ]
    frames = sorted({int(v) for v in matched["frame_index"].tolist()})
    if stride > 1:
        frames = frames[::stride]
    if max_frames is not None:
        frames = frames[: int(max_frames)]
    return frames


def _build_detector(cfg: dict[str, Any]):
    det_cfg = cfg.get("detector") or {}
    if not isinstance(det_cfg, dict):
        raise ValueError(
            "Config key 'detector' must be a mapping. Check YAML indentation under detector:."
        )
    backend = det_cfg.get("backend", "mock")
    if backend == "mock":
        return "mock", None
    if backend == "idea_dino":
        from gaze_objects.detectors.dino_idea import IdeaDinoDetector

        checkpoint = det_cfg.get("checkpoint")
        if not checkpoint:
            raise ValueError("detector.checkpoint is required for backend idea_dino")
        detector = IdeaDinoDetector(
            dino_repo=det_cfg.get("dino_repo", "DINO"),
            config_file=det_cfg.get("config_file", "DINO/config/DINO/DINO_4scale.py"),
            checkpoint=checkpoint,
            device=det_cfg.get("device", "cuda"),
            score_threshold=float(det_cfg.get("score_threshold", 0.3)),
            class_map_file=det_cfg.get("class_map_file"),
        )
        return "idea_dino", detector
    if backend == "grounding_dino":
        from gaze_objects.detectors.grounding_dino import GroundingDinoDetector

        checkpoint = det_cfg.get("checkpoint")
        if not checkpoint:
            raise ValueError("detector.checkpoint is required for backend grounding_dino")
        detector = GroundingDinoDetector(
            grounding_repo=det_cfg.get("grounding_repo", "GroundingDINO"),
            config_file=det_cfg.get(
                "config_file",
                "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            ),
            checkpoint=checkpoint,
            text_prompt=det_cfg.get(
                "text_prompt",
                "angle grinder . instruction manual . storage box . screwdriver . wrench . tool",
            ),
            device=det_cfg.get("device", "cuda"),
            box_threshold=float(det_cfg.get("box_threshold", det_cfg.get("score_threshold", 0.3))),
            text_threshold=float(det_cfg.get("text_threshold", 0.25)),
            class_map_file=det_cfg.get("class_map_file"),
        )
        return "grounding_dino", detector
    raise ValueError(f"Unknown detector.backend: {backend}")


def run_detect(config_path: str | Path) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    joined, ctx = build_synced_gaze(cfg)
    joined.to_csv(out_dir / "gaze_frame_associations.csv", index=False)
    frame_ids = select_frame_indices(joined, cfg)
    backend, detector = _build_detector(cfg)

    want = set(frame_ids)
    rows: list[dict[str, Any]] = []
    n_frames = 0

    if backend == "mock":
        from gaze_objects.detectors.mock import mock_detect_frame

        class_map = cfg.get("detector", {}).get("class_map_file")
        for frame_index, _pts, video_time_s, frame in iter_frames_with_time(
            cfg["video_path"],
            start_s=ctx["video_start_clip"],
            end_s=ctx["video_end_clip"],
            normalize_pts_to_first=True,
        ):
            if frame_index not in want:
                continue
            frame_key = f"f{frame_index}"
            dets = mock_detect_frame(frame, frame_key, class_map_file=class_map)
            for d in dets:
                row = d.to_row()
                row["frame_index"] = frame_index
                row["video_time_s"] = video_time_s
                rows.append(row)
            n_frames += 1
    else:
        assert detector is not None
        detector.load()
        for frame_index, _pts, video_time_s, frame in iter_frames_with_time(
            cfg["video_path"],
            start_s=ctx["video_start_clip"],
            end_s=ctx["video_end_clip"],
            normalize_pts_to_first=True,
        ):
            if frame_index not in want:
                continue
            frame_key = f"f{frame_index}"
            dets: list[Detection] = detector.detect_frame(frame, frame_key)
            for d in dets:
                row = d.to_row()
                row["frame_index"] = frame_index
                row["video_time_s"] = video_time_s
                rows.append(row)
            n_frames += 1

    detections_df = pd.DataFrame(rows)
    detections_df.to_csv(out_dir / "detections.csv", index=False)
    summary = {
        "backend": backend,
        "n_selected_frames": len(frame_ids),
        "n_frames_processed": n_frames,
        "n_detections": int(len(detections_df)),
        "frame_indices": frame_ids,
        "time_mapping": ctx["mapping"],
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "detect_summary.json", summary)
    return summary


def run_assign(config_path: str | Path) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    assoc_path = out_dir / "gaze_frame_associations.csv"
    det_path = out_dir / "detections.csv"
    if not det_path.is_file():
        raise FileNotFoundError(f"Missing {det_path}. Run detect first.")

    if assoc_path.is_file():
        joined = pd.read_csv(assoc_path)
    else:
        joined, _ctx = build_synced_gaze(cfg)
        joined.to_csv(assoc_path, index=False)

    detections = pd.read_csv(det_path)
    asg_cfg = cfg.get("assignment", {})
    assignments = assign_table(
        joined,
        detections,
        score_threshold=float(asg_cfg.get("score_threshold", cfg.get("detector", {}).get("score_threshold", 0.3))),
        require_normalized_label=bool(asg_cfg.get("require_normalized_label", False)),
    )
    assignments.to_csv(out_dir / "gaze_assignments.csv", index=False)
    status_counts = assignments["assignment_status"].value_counts(dropna=False).to_dict()
    summary = {
        "n_gaze_rows": int(len(assignments)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "assign_summary.json", summary)
    return summary
