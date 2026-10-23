#!/usr/bin/env python3
"""Count Tobii TSV/video inventory vs discoverable pairs (workstation helper)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional dir for recording_ids.txt + recording_pairs.csv (paired only)",
    )
    args = parser.parse_args()
    root = Path(args.data_dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"data_dir not found: {root}")

    tsvs = sorted(root.rglob("*_data_export.tsv"))
    videos = sorted(
        set(root.rglob("*_scenevideo.mp4")) | set(root.rglob("* scenevideo.mp4"))
    )
    fehlerhaft = [p for p in tsvs if "fehlerhaft" in p.name.lower()]
    usable_tsvs = [p for p in tsvs if "fehlerhaft" not in p.name.lower()]

    paired = []
    unpaired_tsv = []
    for tsv in usable_tsvs:
        stem = tsv.name[: -len("_data_export.tsv")]
        parent = tsv.parent
        candidates = [
            parent / f"{stem}_scenevideo.mp4",
            parent / f"{stem} scenevideo.mp4",
            parent / f"{stem}_01_scenevideo.mp4",
        ]
        video = next((p for p in candidates if p.is_file()), None)
        if video is None:
            unpaired_tsv.append(tsv)
        else:
            paired.append((stem, tsv, video))

    video_stems = set()
    for v in videos:
        name = v.name
        if name.endswith("_scenevideo.mp4"):
            video_stems.add(name[: -len("_scenevideo.mp4")])
        elif name.endswith(" scenevideo.mp4"):
            video_stems.add(name[: -len(" scenevideo.mp4")])

    tsv_stems = {p.name[: -len("_data_export.tsv")] for p in usable_tsvs}
    unpaired_video_stems = sorted(video_stems - tsv_stems)

    print(f"data_dir: {root}")
    print(f"TSV total:              {len(tsvs)}")
    print(f"TSV fehlerhaft skipped: {len(fehlerhaft)}")
    print(f"TSV usable:             {len(usable_tsvs)}")
    print(f"scenevideo files:       {len(videos)}")
    print(f"paired (pipeline):      {len(paired)}")
    print(f"TSV without video:      {len(unpaired_tsv)}")
    print(f"video stems w/o TSV:    {len(unpaired_video_stems)}")
    print()
    print("--- all paired recording ids ---")
    for stem, _tsv, _video in paired:
        print(stem)
    print()
    print("--- first 20 TSV without matching scenevideo ---")
    for p in unpaired_tsv[:20]:
        print(p.relative_to(root))
    if len(unpaired_tsv) > 20:
        print(f"... +{len(unpaired_tsv) - 20} more")
    print()
    print("--- first 20 video stems without TSV ---")
    for s in unpaired_video_stems[:20]:
        print(s)
    if len(unpaired_video_stems) > 20:
        print(f"... +{len(unpaired_video_stems) - 20} more")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        ids_path = out / "recording_ids.txt"
        csv_path = out / "recording_pairs.csv"
        ids_path.write_text(
            "\n".join(stem for stem, _, _ in paired) + ("\n" if paired else ""),
            encoding="utf-8",
        )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "tsv_path", "video_path"])
            writer.writeheader()
            for stem, tsv, video in paired:
                writer.writerow(
                    {
                        "id": stem,
                        "tsv_path": str(tsv.resolve()),
                        "video_path": str(video.resolve()),
                    }
                )
        print()
        print(f"Wrote {len(paired)} ids → {ids_path}")
        print(f"Wrote pairs CSV → {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
