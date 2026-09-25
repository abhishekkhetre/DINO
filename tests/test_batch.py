"""Tests for batch recording discovery and config merge."""

from __future__ import annotations

from pathlib import Path

from gaze_objects.batch import (
    build_recording_config,
    discover_recording_pairs,
    resolve_batch_recordings,
)


def test_discover_recording_pairs(tmp_path: Path):
    (tmp_path / "AA01_data_export.tsv").write_text("x\n", encoding="utf-8")
    (tmp_path / "AA01_scenevideo.mp4").write_bytes(b"0")
    (tmp_path / "BB02_data_export.tsv").write_text("x\n", encoding="utf-8")
    (tmp_path / "BB02 scenevideo.mp4").write_bytes(b"0")
    (tmp_path / "CC03_data_export.tsv").write_text("x\n", encoding="utf-8")  # no video

    pairs = discover_recording_pairs(tmp_path)
    ids = [p["id"] for p in pairs]
    assert ids == ["AA01", "BB02"]


def test_resolve_first_n_with_preferred(tmp_path: Path):
    (tmp_path / "AM07AM07_01_data_export.tsv").write_text("x\n", encoding="utf-8")
    (tmp_path / "AM07AM07_01_scenevideo.mp4").write_bytes(b"0")
    (tmp_path / "KI05KO01_01_data_export.tsv").write_text("x\n", encoding="utf-8")
    (tmp_path / "KI05KO01_01_scenevideo.mp4").write_bytes(b"0")
    (tmp_path / "ZZ99_data_export.tsv").write_text("x\n", encoding="utf-8")
    (tmp_path / "ZZ99_scenevideo.mp4").write_bytes(b"0")

    cfg = {
        "data_dir": str(tmp_path),
        "recordings": [
            {
                "id": "AM07AM07_01",
                "tsv": "AM07AM07_01_data_export.tsv",
                "video": "AM07AM07_01_scenevideo.mp4",
            }
        ],
        "select": {"mode": "first_n", "n": 2},
        "defaults": {"detector": {"backend": "sam3"}},
        "output_root": "outputs/tmp_batch",
    }
    recs = resolve_batch_recordings(cfg)
    assert len(recs) == 2
    assert recs[0]["id"] == "AM07AM07_01"
    assert recs[1]["id"] in {"KI05KO01_01", "ZZ99"}

    one = build_recording_config(cfg, recs[0])
    assert one["detector"]["backend"] == "sam3"
    assert one["output_dir"].endswith("AM07AM07_01")
    assert Path(one["tsv_path"]).name == "AM07AM07_01_data_export.tsv"
