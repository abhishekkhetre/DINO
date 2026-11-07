#!/usr/bin/env python3
"""AM07 sample video: SAM masks + gaze + geometric tip/handle/body for visual QC.

Workstation only (CUDA + Tobii paths). Produces a playable H.264 overlay so you
can manually verify detailed labelling:

  - coloured SAM masks / boxes for kit objects (+ optional part prompts)
  - cyan/green gaze ring on hits
  - banner text ``GEOMETRY: screwdriver.tip`` (etc.) when part_geometry applies

Usage:
  conda activate sam3
  cd ~/KHETRE/DINO_KHETRE/DINO
  git pull origin cursor/geometry-overlay-sample-e54c   # or fine-parts branch
  export PYTHONPATH=src:$PYTHONPATH
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  # Full recording (reuse existing detect if present):
  python scripts/run_am07_geometry_sample_video.py

  # Shorter review window (faster; good for manual scrubbing):
  python scripts/run_am07_geometry_sample_video.py --start-s 60 --end-s 180

Output (playable):
  outputs/am07_sam3_granular_overlay/sam_gaze_overlay_geometry.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "am07_sam3_granular_overlay.yaml"
OUT_DIR = Path("outputs/am07_sam3_granular_overlay")
OVERLAY_NAME = "sam_gaze_overlay_geometry.mp4"


def _have_detect() -> bool:
    return (OUT_DIR / "detections.csv").is_file() and (OUT_DIR / "detection_masks.npz").is_file()


def _print_seek_hints() -> None:
    asg_path = OUT_DIR / "gaze_assignments.csv"
    if not asg_path.is_file():
        return
    try:
        import pandas as pd
    except ImportError:
        return
    asg = pd.read_csv(asg_path)
    if "attended_part_label" not in asg.columns:
        print("[geometry-sample] no attended_part_label column — geometry did not run", flush=True)
        return
    parts = asg[
        asg["attended_part_label"].notna()
        & (asg["attended_part_label"].astype(str).str.strip() != "")
        & (asg["attended_part_label"].astype(str).str.lower() != "nan")
    ]
    if "assignment_status" in parts.columns:
        parts = parts[parts["assignment_status"] == "assigned"]
    if parts.empty:
        print("[geometry-sample] zero geometry hits — check gaze lands on screwdriver/wrench/AG masks", flush=True)
        return
    print("[geometry-sample] attended_part_label counts:", flush=True)
    print(parts["attended_part_label"].value_counts().head(20).to_string(), flush=True)
    tcol = None
    for c in ("video_time_s", "recording_time_s", "gaze_time_s"):
        if c in parts.columns:
            tcol = c
            break
    if tcol is None:
        return
    print(f"[geometry-sample] seek hints (first hit per label, column={tcol}):", flush=True)
    for lab, sub in parts.groupby(parts["attended_part_label"].astype(str), sort=False):
        t = pd.to_numeric(sub[tcol], errors="coerce").dropna()
        if t.empty:
            continue
        print(f"  {lab:40s}  t≈{float(t.iloc[0]):7.1f}s  (n={len(sub)})", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-redetect",
        action="store_true",
        help="Delete outputs and re-run SAM detect before geometry overlay",
    )
    parser.add_argument("--start-s", type=float, default=None, help="Overlay window start (s)")
    parser.add_argument("--end-s", type=float, default=None, help="Overlay window end (s)")
    parser.add_argument(
        "--skip-sequences",
        action="store_true",
        help="Skip Stage E part_majority sequences (overlay only)",
    )
    args = parser.parse_args(argv)

    try:
        import torch

        if not torch.cuda.is_available():
            print("ERROR: need CUDA (conda activate sam3 on the GPU workstation).", file=sys.stderr)
            return 3
    except ImportError:
        print("ERROR: torch missing — conda activate sam3", file=sys.stderr)
        return 3

    if not CONFIG.is_file():
        print(f"ERROR: missing {CONFIG}", file=sys.stderr)
        return 2

    if args.force_redetect and OUT_DIR.exists():
        import shutil

        print(f"[geometry-sample] removing {OUT_DIR.resolve()}", flush=True)
        shutil.rmtree(OUT_DIR)

    if not _have_detect():
        print("[geometry-sample] no detect cache — running granular SAM overlay first…", flush=True)
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_am07_granular_overlay.py"), "--keep-output"],
            cwd=str(ROOT),
        ).returncode
        if rc != 0:
            return int(rc)
    else:
        print(f"[geometry-sample] reusing detect cache under {OUT_DIR}", flush=True)

    cmd = [
        sys.executable,
        "-m",
        "gaze_objects.cli",
        "refine-parts",
        "--config",
        str(CONFIG),
        "--overlay-name",
        OVERLAY_NAME,
    ]
    if not args.skip_sequences:
        cmd.append("--part-sequences")
    if args.start_s is not None:
        cmd.extend(["--overlay-start-s", str(args.start_s)])
    if args.end_s is not None:
        cmd.extend(["--overlay-end-s", str(args.end_s)])

    print("[geometry-sample] " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        return int(rc)

    mp4 = (OUT_DIR / OVERLAY_NAME).resolve()
    # refine-parts transcodes via ffmpeg and replaces the mp4 in place when possible.
    print(f"\n[geometry-sample] PLAY THIS VIDEO:\n  {mp4}\n", flush=True)
    if not mp4.is_file():
        print(f"ERROR: expected overlay missing: {mp4}", file=sys.stderr)
        return 4
    print(
        "If it will not play in Windows Media Player:\n"
        f"  bash scripts/make_overlay_playable.sh {mp4}\n",
        flush=True,
    )
    print(
        "What to check manually:\n"
        "  1) Mask colours track tools (screwdriver / wrench / angle grinder / box / manual).\n"
        "  2) When gaze is on a tool, banner shows GEOMETRY: <object>.<zone>\n"
        "       screwdriver/wrench → tip | shaft | handle\n"
        "       angle grinder     → body | head | side_handle_zone\n"
        "  3) Green/cyan ring = assigned hit; red ring = gaze without label.\n",
        flush=True,
    )
    _print_seek_hints()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
