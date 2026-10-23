#!/usr/bin/env python3
"""Run AM07 granular-label SAM mask + gaze overlay for video verification.

Must run on the lab workstation (CUDA + Tobii paths). This cloud agent cannot
see the scene video or GPU.

Usage:
  conda activate sam3
  cd ~/KHETRE/DINO_KHETRE/DINO
  git pull origin cursor/fine-parts-thresholds-e54c
  export PYTHONPATH=src:$PYTHONPATH
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python scripts/run_am07_granular_overlay.py

Output:
  outputs/am07_sam3_granular_overlay/sam_gaze_overlay.mp4
  outputs/am07_sam3_granular_overlay/detections.csv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "am07_sam3_granular_overlay.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Granular overlay YAML (default: am07_sam3_granular_overlay.yaml)",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Do not delete the output_dir before running",
    )
    args = parser.parse_args(argv)

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.is_file():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    # Soft preflight — full detect still needs sam3 env + CUDA.
    try:
        import torch

        if not torch.cuda.is_available():
            print(
                "ERROR: torch.cuda is not available. "
                "Activate conda env `sam3` on the workstation GPU machine.",
                file=sys.stderr,
            )
            return 3
    except ImportError:
        print(
            "ERROR: torch not installed in this Python. "
            "Run: conda activate sam3 && export PYTHONPATH=src:$PYTHONPATH",
            file=sys.stderr,
        )
        return 3

    # Read output_dir from YAML for clean re-runs.
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    out_dir = Path(cfg.get("output_dir", "outputs/am07_sam3_granular_overlay"))
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    if not args.keep_output and out_dir.exists():
        print(f"[granular-overlay] removing {out_dir}", flush=True)
        shutil.rmtree(out_dir)

    cmd = [
        sys.executable,
        "-m",
        "gaze_objects.cli",
        "sam-overlay",
        "--config",
        str(cfg_path),
    ]
    print("[granular-overlay] " + " ".join(cmd), flush=True)
    print(
        "[granular-overlay] labels: whole AG + body/side handle/guard/battery "
        "+ screw + kit objects (NO grinding disc)",
        flush=True,
    )
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        return int(proc.returncode)

    mp4 = out_dir / str(cfg.get("overlay_filename", "sam_gaze_overlay.mp4"))
    det = out_dir / "detections.csv"
    print(f"[granular-overlay] video → {mp4}", flush=True)
    print(f"[granular-overlay] detections → {det}", flush=True)
    if det.is_file():
        try:
            import pandas as pd

            d = pd.read_csv(det)
            print("[granular-overlay] label counts:", flush=True)
            print(d["normalized_label"].value_counts().to_string(), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[granular-overlay] could not summarize detections: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
