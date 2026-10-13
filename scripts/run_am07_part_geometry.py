#!/usr/bin/env python3
"""Refine AM07 granular overlay with gaze tip/handle/body — NO re-detect.

Requires an existing outputs/am07_sam3_granular_overlay/ with detections + masks.

Usage (workstation):
  conda activate sam3
  cd ~/KHETRE/DINO_KHETRE/DINO
  git pull origin cursor/fine-parts-thresholds-e54c
  export PYTHONPATH=src:$PYTHONPATH
  python scripts/run_am07_part_geometry.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "am07_sam3_granular_overlay.yaml"


def main() -> int:
    out = Path("outputs/am07_sam3_granular_overlay")
    if not (out / "detections.csv").is_file() or not (out / "detection_masks.npz").is_file():
        print(
            "ERROR: need existing granular detect outputs.\n"
            "  First run: python scripts/run_am07_granular_overlay.py\n"
            f"  Missing under {out.resolve()}",
            file=sys.stderr,
        )
        return 2
    cmd = [
        sys.executable,
        "-m",
        "gaze_objects.cli",
        "refine-parts",
        "--config",
        str(CONFIG),
        "--part-sequences",
        "--overlay-name",
        "sam_gaze_overlay_parts.mp4",
    ]
    print("[part-geometry] " + " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, cwd=str(ROOT)).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
