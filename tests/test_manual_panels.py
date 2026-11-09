from pathlib import Path

import numpy as np
import pytest

from gaze_objects.manual_panels import panel_for_page_point
from gaze_objects.manual_steps import load_manual_config

ROOT = Path(__file__).resolve().parents[1]
PANELS = load_manual_config(ROOT / "configs" / "manual_steps.yaml")["manual_panels"]


def test_panel_lookup_matches_page_layout():
    assert panel_for_page_point(1, 0.5, 0.10, PANELS) == "D1"
    assert panel_for_page_point(1, 0.3, 0.90, PANELS) == "D4"
    assert panel_for_page_point(1, 0.8, 0.90, PANELS) == "D5"
    assert panel_for_page_point(2, 0.5, 0.40, PANELS) == "D7"
    assert panel_for_page_point(3, 0.8, 0.35, PANELS) == "A3"
    assert panel_for_page_point(4, 0.3, 0.75, PANELS) == "A8"
    assert panel_for_page_point(2, 0.5, 0.97, PANELS) is None


def test_matcher_reads_panel_from_warped_page():
    cv2 = pytest.importorskip("cv2")
    from gaze_objects.manual_panels import ManualPageMatcher

    pages_dir = ROOT / "docs" / "manual" / "pages"
    matcher = ManualPageMatcher.from_directory(pages_dir)
    page = cv2.imread(str(pages_dir / "page1.jpg"), cv2.IMREAD_GRAYSCALE)
    h, w = page.shape
    # Simulated head-camera view: page shrunk, tilted, placed on a darker table.
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[300, 80], [900, 120], [860, 940], [260, 900]])
    hmat = cv2.getPerspectiveTransform(src, dst)
    canvas = cv2.warpPerspective(page, hmat, (1280, 1024), borderValue=60)
    target_page_pt = np.float32([[[0.5 * w, 0.62 * h]]])  # middle of Schritt 3
    gaze = cv2.perspectiveTransform(target_page_pt, hmat)[0, 0]
    loc = matcher.locate(canvas, (float(gaze[0]), float(gaze[1])))
    assert loc["page"] == 1
    assert panel_for_page_point(loc["page"], loc["u"], loc["v"], PANELS) == "D3"
