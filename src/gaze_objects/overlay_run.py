"""Render SAM mask + gaze overlay from existing detect/assign outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from gaze_objects.audit import write_json
from gaze_objects.io import read_tsv
from gaze_objects.masks import load_detection_masks
from gaze_objects.video import inspect_video
from gaze_objects.visualize import transcode_to_h264, write_sam_gaze_overlay


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_overlay(config_path: str | Path) -> dict[str, Any]:
    """
    Write sam_gaze_overlay.mp4 from detections + assignments + masks.

    Requires prior detect (detections.csv, detection_masks.npz) and assign
    (gaze_assignments.csv). Does not run SAM again.
    """
    cfg = _load_yaml(config_path)
    out_dir = Path(cfg["output_dir"])
    det_path = out_dir / "detections.csv"
    asg_path = out_dir / "gaze_assignments.csv"
    if not det_path.is_file():
        raise FileNotFoundError(f"Missing {det_path}. Run detect first.")
    if not asg_path.is_file():
        raise FileNotFoundError(f"Missing {asg_path}. Run assign first.")

    detections = pd.read_csv(det_path)
    assignments = pd.read_csv(asg_path)
    masks = load_detection_masks(out_dir)

    _, meta = read_tsv(cfg["tsv_path"], compute_hash=False)
    vinfo = inspect_video(cfg["video_path"], max_pts_samples=int(cfg.get("max_pts_samples", 2000)))

    start_s = 0.0
    end_s = float(vinfo.duration_s or 30.0)
    det_sum_path = out_dir / "detect_summary.json"
    if det_sum_path.is_file():
        ds = json.loads(det_sum_path.read_text(encoding="utf-8"))
        start_s = float(ds.get("video_start_clip", start_s))
        end_s = float(ds.get("video_end_clip", end_s))

    out_mp4 = out_dir / str(cfg.get("overlay_filename", "sam_gaze_overlay.mp4"))
    print(f"[overlay] rendering {out_mp4.name} …", flush=True)
    overlay_stats = write_sam_gaze_overlay(
        cfg["video_path"],
        detections=detections,
        assignments=assignments,
        masks_by_id=masks,
        out_path=out_mp4,
        src_width=int(meta["media_width"]),
        src_height=int(meta["media_height"]),
        start_s=start_s,
        end_s=end_s,
        video_fps_hint=vinfo.average_fps,
        gaze_radius=int(cfg.get("overlay_gaze_radius", 28)),
    )
    playable = transcode_to_h264(out_mp4)
    summary = {
        "overlay": overlay_stats,
        "playable_path": str(playable),
        "n_masks_loaded": len(masks),
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "overlay_summary.json", summary)
    return summary
