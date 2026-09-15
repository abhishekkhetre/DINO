"""Video metadata, decoding and timing helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    n_frames: int | None
    duration_s: float | None
    average_fps: float | None
    time_base: str | None
    codec: str | None
    pix_fmt: str | None
    rotation: int | None
    first_pts_s: float | None
    last_pts_s: float | None
    pts_start_assumption: str
    notes: list[str]


def inspect_video(path: str | Path, *, max_pts_samples: int = 5000) -> VideoInfo:
    """Inspect container metadata and sample presentation timestamps via PyAV."""
    try:
        import av
    except ImportError as exc:
        raise ImportError("PyAV is required for video inspection. Install with: pip install av") from exc

    path = Path(path)
    notes: list[str] = []
    container = av.open(str(path))
    stream = container.streams.video[0]
    codec = stream.codec_context.name if stream.codec_context else None
    pix_fmt = stream.codec_context.pix_fmt if stream.codec_context else None
    width = int(stream.codec_context.width)
    height = int(stream.codec_context.height)
    n_frames = int(stream.frames) if stream.frames else None
    average_fps = float(stream.average_rate) if stream.average_rate else None
    time_base = str(stream.time_base) if stream.time_base else None
    duration_s = None
    if stream.duration is not None and stream.time_base is not None:
        duration_s = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        duration_s = float(container.duration / av.time_base)

    rotation = 0
    try:
        if stream.side_data and "DISPLAYMATRIX" in str(stream.side_data):
            notes.append("DISPLAYMATRIX side data present; verify orientation.")
    except Exception:
        pass
    # Prefer stream metadata rotation if present.
    rot_meta = stream.metadata.get("rotate") if stream.metadata else None
    if rot_meta is not None:
        try:
            rotation = int(rot_meta)
        except ValueError:
            notes.append(f"Unparsed rotate metadata: {rot_meta}")

    pts_list: list[float] = []
    for i, frame in enumerate(container.decode(video=0)):
        if frame.pts is None or stream.time_base is None:
            continue
        pts_list.append(float(frame.pts * stream.time_base))
        if i + 1 >= max_pts_samples:
            notes.append(
                f"PTS sampling stopped after {max_pts_samples} frames; "
                "full last_pts may be incomplete."
            )
            break
    container.close()

    first_pts = pts_list[0] if pts_list else None
    last_pts = pts_list[-1] if pts_list else None
    if first_pts is not None and abs(first_pts) > 1e-6:
        notes.append(
            f"First sampled PTS is {first_pts:.6f}s, not zero. "
            "Normalization must be explicit."
        )
    notes.append(
        "Average FPS alone is insufficient for synchronization; use presentation timestamps."
    )

    return VideoInfo(
        path=str(path.resolve()),
        width=width,
        height=height,
        n_frames=n_frames,
        duration_s=duration_s,
        average_fps=average_fps,
        time_base=time_base,
        codec=codec,
        pix_fmt=pix_fmt,
        rotation=rotation,
        first_pts_s=first_pts,
        last_pts_s=last_pts,
        pts_start_assumption=(
            "normalize_to_first_presented_frame"
            if first_pts is not None
            else "unknown"
        ),
        notes=notes,
    )


def write_video_info(info: VideoInfo, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")


def iter_frames_with_time(
    path: str | Path,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
    normalize_pts_to_first: bool = True,
) -> Iterator[tuple[int, float, float, np.ndarray]]:
    """
    Yield (frame_index, pts_s_original, video_time_s, bgr_array).

    ``video_time_s`` subtracts the first presented frame PTS when
    ``normalize_pts_to_first`` is True. Association with recording time is
    handled in ``sync.py``, not here.
    """
    import av
    import cv2

    container = av.open(str(path))
    stream = container.streams.video[0]
    first_pts: float | None = None
    index = 0
    for frame in container.decode(video=0):
        if frame.pts is None or stream.time_base is None:
            continue
        pts_s = float(frame.pts * stream.time_base)
        if first_pts is None:
            first_pts = pts_s
        video_time_s = pts_s - first_pts if normalize_pts_to_first else pts_s
        if start_s is not None and video_time_s < start_s:
            index += 1
            continue
        if end_s is not None and video_time_s > end_s:
            break
        rgb = frame.to_ndarray(format="rgb24")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        yield index, pts_s, video_time_s, bgr
        index += 1
    container.close()


def build_frame_table(
    path: str | Path,
    *,
    max_frames: int | None = None,
    normalize_pts_to_first: bool = True,
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev_t: float | None = None
    for idx, pts_s, video_time_s, _frame in iter_frames_with_time(
        path,
        start_s=start_s,
        end_s=end_s,
        normalize_pts_to_first=normalize_pts_to_first,
    ):
        interval = None if prev_t is None else video_time_s - prev_t
        rows.append(
            {
                "frame_index": idx,
                "pts_s_original": pts_s,
                "video_time_s": video_time_s,
                "presentation_interval_s": interval,
            }
        )
        prev_t = video_time_s
        if max_frames is not None and len(rows) >= max_frames:
            break
    return rows
