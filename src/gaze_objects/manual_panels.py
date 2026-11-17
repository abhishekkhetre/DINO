"""Which manual step is being read: gaze on the printed manual -> 'Schritt' panel.

For every attention episode labelled as the instruction manual, the scene
frame is matched against the four manual page images (ORB features + RANSAC
homography). The gaze point is projected into page coordinates and looked up
in the step panels of ``configs/manual_steps.yaml``.

Feasibility depends on how large and sharp the manual appears in the scene
video; episodes without a confident page match get no panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MANUAL_SYMBOL_LABELS = {"instruction manual", "manual", "booklet"}


def panel_for_page_point(
    page: int, u: float, v: float, panels: dict[Any, dict[str, list[float]]]
) -> str | None:
    """Step id whose panel contains normalised page point (u, v)."""
    page_panels = panels.get(page) or panels.get(str(page)) or {}
    for step_id, (x0, y0, x1, y1) in page_panels.items():
        if x0 <= u <= x1 and y0 <= v <= y1:
            return str(step_id)
    return None


class ManualPageMatcher:
    def __init__(
        self,
        page_images: dict[int, np.ndarray],
        *,
        n_features: int = 3000,
        min_inliers: int = 18,
        ratio: float = 0.75,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_inliers = min_inliers
        self.ratio = ratio
        self.pages: dict[int, dict[str, Any]] = {}
        for page, img in page_images.items():
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)
            self.pages[int(page)] = {"kp": kp, "des": des, "shape": gray.shape[:2]}

    @classmethod
    def from_directory(cls, directory: str | Path, **kwargs: Any) -> "ManualPageMatcher":
        import cv2

        images: dict[int, np.ndarray] = {}
        for path in sorted(Path(directory).glob("page*.jpg")) + sorted(Path(directory).glob("page*.png")):
            digits = "".join(ch for ch in path.stem if ch.isdigit())
            if digits:
                images[int(digits)] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if not images:
            raise FileNotFoundError(f"no page*.jpg/png in {directory}")
        return cls(images, **kwargs)

    def locate(self, frame: np.ndarray, gaze_xy: tuple[float, float]) -> dict[str, Any]:
        """Best page + gaze position in normalised page coordinates."""
        cv2 = self._cv2
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_f, des_f = self.orb.detectAndCompute(gray, None)
        best: dict[str, Any] = {"page": None, "u": None, "v": None, "n_inliers": 0}
        if des_f is None or len(kp_f) < 8:
            return best
        for page, ref in self.pages.items():
            if ref["des"] is None:
                continue
            pairs = self.matcher.knnMatch(des_f, ref["des"], k=2)
            good = [m for m, *rest in pairs if rest and m.distance < self.ratio * rest[0].distance]
            if len(good) < self.min_inliers:
                continue
            src = np.float32([kp_f[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([ref["kp"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            h, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if h is None or inl is None:
                continue
            n_in = int(inl.sum())
            if n_in < self.min_inliers or n_in <= best["n_inliers"]:
                continue
            pt = cv2.perspectiveTransform(np.float32([[gaze_xy]]), h)[0, 0]
            hgt, wid = ref["shape"]
            best = {
                "page": page,
                "u": float(pt[0] / wid),
                "v": float(pt[1] / hgt),
                "n_inliers": n_in,
            }
        return best


def _read_frame(cap: Any, frame_index: int) -> np.ndarray | None:
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def manual_panels_for_recording(
    recording_dir: str | Path,
    video_path: str | Path,
    matcher: ManualPageMatcher,
    panels: dict[Any, dict[str, list[float]]],
    *,
    label_column: str = "attended_label",
    max_frames_per_episode: int = 3,
) -> pd.DataFrame:
    """One row per manual episode: page, gaze page coords, step panel read."""
    import cv2

    rec = Path(recording_dir)
    seq = pd.read_csv(rec / "attention_sequences.csv")
    asg = pd.read_csv(rec / "gaze_assignments.csv")
    labels = seq[label_column].astype(str).str.lower().str.split(".").str[0]
    manual_eps = seq[labels.isin(MANUAL_SYMBOL_LABELS)]
    rows: list[dict[str, Any]] = []
    if manual_eps.empty:
        return pd.DataFrame(rows)
    asg = asg.dropna(subset=["video_time_s", "gaze_x_px", "gaze_y_px", "frame_index"])
    cap = cv2.VideoCapture(str(video_path))
    try:
        for ep in manual_eps.itertuples(index=False):
            t0, t1 = float(ep.video_start_s), float(ep.video_end_s)
            inside = asg[(asg["video_time_s"] >= t0 - 1e-3) & (asg["video_time_s"] <= t1 + 1e-3)]
            if inside.empty:
                continue
            picks = inside.iloc[np.linspace(0, len(inside) - 1, min(max_frames_per_episode, len(inside))).astype(int)]
            votes: dict[str, int] = {}
            best: dict[str, Any] = {"page": None, "u": None, "v": None, "n_inliers": 0}
            for p in picks.itertuples(index=False):
                frame = _read_frame(cap, int(p.frame_index))
                if frame is None:
                    continue
                loc = matcher.locate(frame, (float(p.gaze_x_px), float(p.gaze_y_px)))
                if loc["page"] is None:
                    continue
                step = panel_for_page_point(loc["page"], loc["u"], loc["v"], panels)
                if step:
                    votes[step] = votes.get(step, 0) + 1
                if loc["n_inliers"] > best["n_inliers"]:
                    best = loc
            panel = max(votes.items(), key=lambda kv: kv[1])[0] if votes else None
            rows.append(
                {
                    "sequence_id": ep.sequence_id,
                    "video_start_s": t0,
                    "video_end_s": t1,
                    "manual_page": best["page"],
                    "page_u": best["u"],
                    "page_v": best["v"],
                    "n_inliers": best["n_inliers"],
                    "manual_panel_step": panel,
                    "panel_votes": sum(votes.values()),
                }
            )
    finally:
        cap.release()
    return pd.DataFrame(rows)


def attach_panels(
    sequences: pd.DataFrame, panel_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Add manual_panel_step / manual_page to a corpus sequences table."""
    parts = []
    for rid, table in panel_tables.items():
        if table is None or table.empty:
            continue
        t = table[["sequence_id", "manual_page", "manual_panel_step"]].copy()
        t.insert(0, "recording_id", rid)
        parts.append(t)
    if not parts:
        out = sequences.copy()
        out["manual_panel_step"] = None
        out["manual_page"] = None
        return out
    panels = pd.concat(parts, ignore_index=True)
    return sequences.merge(panels, on=["recording_id", "sequence_id"], how="left")


def run_manual_panels(
    output_root: str | Path,
    data_dir: str | Path,
    pages_dir: str | Path,
    panels: dict[Any, dict[str, list[float]]],
    *,
    label_column: str = "attended_label",
    recordings: list[str] | None = None,
) -> dict[str, Any]:
    """Write <recording>/manual_panels.csv for every recording with manual gaze."""
    root = Path(output_root)
    matcher = ManualPageMatcher.from_directory(pages_dir)
    videos = {p.name.replace("_scenevideo.mp4", ""): p for p in Path(data_dir).rglob("*_scenevideo.mp4")}
    done, missing_video, n_panel = 0, [], 0
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if recordings and sub.name not in recordings:
            continue
        if not (sub / "attention_sequences.csv").is_file() or not (sub / "gaze_assignments.csv").is_file():
            continue
        video = videos.get(sub.name)
        if video is None:
            missing_video.append(sub.name)
            continue
        table = manual_panels_for_recording(sub, video, matcher, panels, label_column=label_column)
        table.to_csv(sub / "manual_panels.csv", index=False)
        done += 1
        if len(table):
            n_panel += int(table["manual_panel_step"].notna().sum())
    return {
        "recordings_processed": done,
        "recordings_missing_video": missing_video,
        "manual_episodes_with_panel": n_panel,
    }


def load_panel_tables(output_root: str | Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for path in Path(output_root).glob("*/manual_panels.csv"):
        try:
            tables[path.parent.name] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
    return tables
