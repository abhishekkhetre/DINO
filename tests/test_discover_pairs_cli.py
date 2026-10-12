"""CLI discover-pairs can export a flat ID list."""

from __future__ import annotations

from pathlib import Path

from gaze_objects.cli import cmd_discover_pairs


class _Args:
    def __init__(self, data_dir: str, out: str | None = None, ids_only: bool = False):
        self.data_dir = data_dir
        self.out = out
        self.ids_only = ids_only
        self.limit = None


def test_discover_pairs_writes_id_list(tmp_path: Path, capsys):
    (tmp_path / "AA01_data_export.tsv").write_text("x\n", encoding="utf-8")
    (tmp_path / "AA01_scenevideo.mp4").write_bytes(b"0")
    (tmp_path / "BB02_data_export.tsv").write_text("x\n", encoding="utf-8")
    (tmp_path / "BB02_scenevideo.mp4").write_bytes(b"0")

    out = tmp_path / "inventory"
    assert cmd_discover_pairs(_Args(str(tmp_path), out=str(out), ids_only=True)) == 0

    ids = (out / "recording_ids.txt").read_text(encoding="utf-8").strip().splitlines()
    assert ids == ["AA01", "BB02"]
    csv_text = (out / "recording_pairs.csv").read_text(encoding="utf-8")
    assert "AA01" in csv_text and "BB02" in csv_text

    captured = capsys.readouterr()
    assert "AA01\nBB02" in captured.out
