"""Command-line interface for incremental pipeline stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def cmd_audit(args: argparse.Namespace) -> int:
    from gaze_objects.audit import run_audit

    report = run_audit(
        args.tsv,
        args.out,
        compute_hash=not args.skip_hash,
        expected_sha256=args.expected_sha256,
    )
    print(f"Wrote audit outputs to {args.out}")
    print(f"SHA-256: {report['provenance']['sha256']}")
    print(f"Eye rows: {report['gaze_availability']['eye_tracker_rows']}")
    print(f"In-frame XY: {report['gaze_availability']['rows_in_frame']}")
    if report.get("reconciliation", {}).get("sha256_match") is False:
        print("WARNING: SHA-256 does not match expected value.", file=sys.stderr)
        return 2
    return 0


def cmd_inspect_video(args: argparse.Namespace) -> int:
    from gaze_objects.video import inspect_video, write_video_info

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = inspect_video(args.video, max_pts_samples=args.max_pts_samples)
    write_video_info(info, out_dir / "video_info.json")
    print(json.dumps(info.__dict__, indent=2))
    return 0


def cmd_overlay(args: argparse.Namespace) -> int:
    from gaze_objects.audit import prepare_eye_tracker_table, write_json
    from gaze_objects.io import read_tsv
    from gaze_objects.sync import TimeMapping, associate_gaze_to_frames, representative_gaze_per_frame
    from gaze_objects.video import build_frame_table, inspect_video
    from gaze_objects.visualize import write_overlay_clip

    cfg = _load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = cfg["tsv_path"]
    video_path = cfg["video_path"]
    unit = cfg.get("timestamp_unit", "milliseconds")
    unit_status = cfg.get("timestamp_unit_status", "provisional")
    video_start = float(cfg.get("video_start_recording_s", 0.0))
    map_status = cfg.get("time_mapping_status", "provisional")
    max_residual = float(cfg.get("max_abs_residual_s", 0.05))
    clip = cfg.get("clip", {})
    # Clip is expressed in original recording time seconds.
    rec_start = float(clip["recording_start_s"])
    rec_end = float(clip["recording_end_s"])

    df, meta = read_tsv(tsv_path, compute_hash=bool(cfg.get("compute_hash", True)))
    eye = prepare_eye_tracker_table(
        df,
        meta["discovered"],
        timestamp_unit_hypothesis=unit,
        media_width=meta["media_width"],
        media_height=meta["media_height"],
    )

    mapping = TimeMapping(
        timestamp_unit=unit,
        timestamp_unit_status=unit_status,
        video_start_recording_s=video_start,
        time_mapping_status=map_status,
        notes=cfg.get(
            "time_mapping_notes",
            "Offset model hypothesis; verify against Tobii playback before treating as validated.",
        ),
    )

    eye_clip = eye[
        (eye["recording_time_s_provisional"] >= rec_start)
        & (eye["recording_time_s_provisional"] <= rec_end)
    ].copy()
    eye_clip["video_time_s"] = eye_clip["recording_time_s_provisional"].map(
        mapping.recording_time_to_video_time
    )

    video_start_clip = mapping.recording_time_to_video_time(rec_start)
    video_end_clip = mapping.recording_time_to_video_time(rec_end)

    vinfo = inspect_video(video_path, max_pts_samples=int(cfg.get("max_pts_samples", 3000)))
    # Frame table for the clip window only (plus small pad for association edges).
    pad = max_residual + 0.05
    frames = build_frame_table(
        video_path,
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
    # Remap local nearest indices back to absolute frame_index from the video.
    mapped_idx = assoc["frame_index"].to_numpy().copy()
    absolute = np.full_like(mapped_idx, -1)
    ok = mapped_idx >= 0
    if ok.any():
        absolute[ok] = np.asarray(frame_indices, dtype=int)[mapped_idx[ok]]
    assoc["frame_index"] = absolute

    eye_clip = eye_clip.reset_index(drop=True)
    joined = assoc.join(eye_clip, how="left")
    joined.to_csv(out_dir / "gaze_frame_associations.csv", index=False)

    rep = representative_gaze_per_frame(assoc, eye_clip)
    if not rep.empty:
        overlay_df = rep[
            [
                "frame_index",
                "gaze_x_px",
                "gaze_y_px",
                "video_time_s",
                "temporal_residual_s",
            ]
        ].copy()
    else:
        overlay_df = rep

    overlay_stats = write_overlay_clip(
        video_path,
        overlay_df,
        out_dir / "gaze_overlay_clip.mp4",
        src_width=int(meta["media_width"]),
        src_height=int(meta["media_height"]),
        start_s=video_start_clip,
        end_s=video_end_clip,
        video_fps_hint=vinfo.average_fps,
    )

    diagnostics = {
        "time_mapping": mapping.to_dict(),
        "clip_recording_s": {"start": rec_start, "end": rec_end},
        "clip_video_s": {"start": video_start_clip, "end": video_end_clip},
        "n_gaze_in_clip": int(len(eye_clip)),
        "n_matched": int((assoc["association_status"] == "matched").sum()),
        "n_unmatched": int((assoc["association_status"] == "unmatched_time").sum()),
        "max_abs_residual_s": max_residual,
        "video_info": vinfo.__dict__,
        "overlay": overlay_stats,
        "example_row_check": {
            "description": (
                "Handoff example: raw timestamp 24735, gaze (947, 613). "
                "With ms units and video_start_recording_s=0 => video 24.735s."
            ),
            "assumptions_status": map_status,
        },
        "validation_label": (
            "provisional_overlay"
            if map_status != "verified" or unit_status != "verified"
            else "verified_overlay"
        ),
    }
    write_json(out_dir / "sync_diagnostics.json", diagnostics)
    print(f"Wrote overlay and diagnostics to {out_dir}")
    print(f"Validation label: {diagnostics['validation_label']}")
    print(
        f"Matched {diagnostics['n_matched']} / {diagnostics['n_gaze_in_clip']} "
        f"gaze samples in clip"
    )
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    from gaze_objects.stage_c import run_detect

    summary = run_detect(args.config)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_assign(args: argparse.Namespace) -> int:
    from gaze_objects.stage_c import run_assign

    summary = run_assign(args.config)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from gaze_objects.evaluate import run_evaluate

    summary = run_evaluate(args.config)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_sequences(args: argparse.Namespace) -> int:
    from gaze_objects.sequences import run_sequences

    summary = run_sequences(args.config)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_discover_pairs(args: argparse.Namespace) -> int:
    from gaze_objects.batch import discover_recording_pairs

    pairs = discover_recording_pairs(args.data_dir)
    if args.limit is not None:
        pairs = pairs[: int(args.limit)]
    print(json.dumps(pairs, indent=2))
    print(f"n_pairs={len(pairs)}", file=sys.stderr)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from gaze_objects.batch import run_batch

    summary = run_batch(args.config)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("n_error", 0) == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gaze_objects", description="Gaze-to-object pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Stage A: TSV audit and reference extraction")
    p_audit.add_argument("--tsv", required=True, help="Path to Tobii TSV export")
    p_audit.add_argument("--out", required=True, help="Output run directory")
    p_audit.add_argument("--expected-sha256", default=None)
    p_audit.add_argument("--skip-hash", action="store_true")
    p_audit.set_defaults(func=cmd_audit)

    p_vid = sub.add_parser("inspect-video", help="Stage B: inspect video timing metadata")
    p_vid.add_argument("--video", required=True)
    p_vid.add_argument("--out", required=True)
    p_vid.add_argument("--max-pts-samples", type=int, default=5000)
    p_vid.set_defaults(func=cmd_inspect_video)

    p_over = sub.add_parser("overlay", help="Stage B: provisional/verified gaze overlay")
    p_over.add_argument("--config", required=True)
    p_over.set_defaults(func=cmd_overlay)

    p_det = sub.add_parser("detect", help="Stage C: run detector on selected frames")
    p_det.add_argument("--config", required=True)
    p_det.set_defaults(func=cmd_detect)

    p_asg = sub.add_parser("assign", help="Stage C: assign gaze to cached detections")
    p_asg.add_argument("--config", required=True)
    p_asg.set_defaults(func=cmd_assign)

    p_eval = sub.add_parser("evaluate", help="Stage D: compare assignments to AOI reference labels")
    p_eval.add_argument("--config", required=True)
    p_eval.set_defaults(func=cmd_evaluate)

    p_seq = sub.add_parser(
        "sequences",
        help="Stage E: Tobii fixation → attended-object sequences from assignments",
    )
    p_seq.add_argument("--config", required=True)
    p_seq.set_defaults(func=cmd_sequences)

    p_disc = sub.add_parser(
        "discover-pairs",
        help="List TSV+scenevideo pairs under a data directory",
    )
    p_disc.add_argument("--data-dir", required=True)
    p_disc.add_argument("--limit", type=int, default=None)
    p_disc.set_defaults(func=cmd_discover_pairs)

    p_batch = sub.add_parser(
        "batch",
        help="Run detect/assign/evaluate/sequences over a batch manifest",
    )
    p_batch.add_argument("--config", required=True)
    p_batch.set_defaults(func=cmd_batch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
