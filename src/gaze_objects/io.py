"""Read Tobii Pro Lab TSV exports and discover recording-specific columns."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = (
    "Recording timestamp",
    "Computer timestamp",
    "Sensor",
    "Participant name",
    "Recording name",
    "Recording duration",
    "Gaze point X",
    "Gaze point Y",
    "Validity left",
    "Validity right",
    "Eye movement type",
    "Gaze event duration",
    "Eye movement type index",
    "Recording media name",
    "Recording media width",
    "Recording media height",
)

SCREENSHOT_COORD_PATTERNS = {
    "assisted_x": re.compile(r"^Assisted mapping gaze point X \[(.+)\]$"),
    "assisted_y": re.compile(r"^Assisted mapping gaze point Y \[(.+)\]$"),
    "manual_x": re.compile(r"^Manually mapped gaze point X \[(.+)\]$"),
    "manual_y": re.compile(r"^Manually mapped gaze point Y \[(.+)\]$"),
    "mapped_x": re.compile(r"^Mapped gaze point X \[(.+)\]$"),
    "mapped_y": re.compile(r"^Mapped gaze point Y \[(.+)\]$"),
    "assisted_score": re.compile(r"^Assisted mapping gaze point score\[(.+)\]$"),
    "width": re.compile(r"^Width \[(.+)\]$"),
    "height": re.compile(r"^Height \[(.+)\]$"),
}

AOI_HIT_PATTERN = re.compile(r"^AOI hit \[(.+)\]$")


@dataclass
class DiscoveredColumns:
    """Recording-specific screenshot mapping and AOI columns."""

    screenshot_ids: list[str] = field(default_factory=list)
    mapping: dict[str, dict[str, str]] = field(default_factory=dict)
    aoi_columns: dict[str, str] = field(default_factory=dict)  # label -> column name
    aoi_sources: dict[str, str] = field(default_factory=dict)  # label -> raw bracket text


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def discover_columns(columns: list[str]) -> DiscoveredColumns:
    discovered = DiscoveredColumns()
    screenshot_ids: set[str] = set()

    for col in columns:
        for key, pattern in SCREENSHOT_COORD_PATTERNS.items():
            match = pattern.match(col)
            if match:
                sid = match.group(1)
                screenshot_ids.add(sid)
                discovered.mapping.setdefault(sid, {})[key] = col
                break
        else:
            aoi_match = AOI_HIT_PATTERN.match(col)
            if aoi_match:
                raw = aoi_match.group(1)
                # Prefer the text after the last " - " as the AOI label.
                if " - " in raw:
                    label = raw.rsplit(" - ", 1)[1].strip()
                else:
                    label = raw.strip()
                # Keep first occurrence; later duplicates stay discoverable via aoi_sources.
                if label not in discovered.aoi_columns:
                    discovered.aoi_columns[label] = col
                discovered.aoi_sources[col] = raw

    discovered.screenshot_ids = sorted(screenshot_ids)
    return discovered


def validate_required_columns(columns: list[str]) -> list[str]:
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    return missing


def _to_numeric(series: pd.Series) -> pd.Series:
    """Parse numeric Tobii fields; keep original non-numeric as NA without silent coercion of commas."""
    return pd.to_numeric(series, errors="coerce")


def read_tsv(path: str | Path, *, compute_hash: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Read a Tobii export as tab-separated UTF-8 (BOM-tolerant).

    Returns the full dataframe plus provenance metadata. Adds ``source_row_id``
    (0-based index among data rows) without altering original columns.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TSV not found: {path}")

    meta: dict[str, Any] = {
        "tsv_path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": None,
        "delimiter": "\t",
        "encoding": "utf-8-sig",
    }
    if compute_hash:
        meta["sha256"] = file_sha256(path)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        engine="c",
    )
    # Empty strings stay as ""; convert known numeric columns after discovery.
    missing = validate_required_columns(list(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    discovered = discover_columns(list(df.columns))
    meta["discovered"] = discovered
    meta["n_rows"] = len(df)
    meta["n_columns"] = len(df.columns)
    meta["columns"] = list(df.columns)

    df = df.copy()
    df.insert(0, "source_row_id", range(len(df)))

    numeric_base = [
        "Recording timestamp",
        "Computer timestamp",
        "Recording duration",
        "Gaze point X",
        "Gaze point Y",
        "Gaze event duration",
        "Eye movement type index",
        "Recording media width",
        "Recording media height",
        "Fixation point X",
        "Fixation point Y",
    ]
    for col in numeric_base:
        if col in df.columns:
            df[col] = _to_numeric(df[col].replace("", pd.NA))

    for sid, mapping in discovered.mapping.items():
        for key, col in mapping.items():
            if key == "assisted_score" or key.endswith(("_x", "_y")) or key in {"width", "height"}:
                df[col] = _to_numeric(df[col].replace("", pd.NA))

    for col in discovered.aoi_columns.values():
        df[col] = _to_numeric(df[col].replace("", pd.NA))

    # Preserve empty Sensor / Event as empty string rather than NaN for filtering.
    for col in ("Sensor", "Event", "Event value", "Participant name", "Recording name", "Recording media name"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    meta["participant_id"] = _first_nonempty(df["Participant name"])
    meta["recording_id"] = _first_nonempty(df["Recording name"])
    meta["project_name"] = _first_nonempty(df.get("Project name", pd.Series(dtype=str)))
    meta["media_name"] = _first_nonempty(df["Recording media name"])
    meta["media_width"] = _first_numeric(df["Recording media width"])
    meta["media_height"] = _first_numeric(df["Recording media height"])
    meta["recording_duration_raw"] = _first_numeric(df["Recording duration"])
    meta["fixation_filter"] = _first_nonempty(df.get("Recording Fixation filter name", pd.Series(dtype=str)))

    return df, meta


def _first_nonempty(series: pd.Series) -> str | None:
    if series is None or len(series) == 0:
        return None
    for value in series:
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return None


def _first_numeric(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.iloc[0])
