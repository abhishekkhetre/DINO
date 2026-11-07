"""SAM3 processor resolution must match the stock ViT img_size (1008)."""

from __future__ import annotations

import pytest

from gaze_objects.detectors.sam3_meta import Sam3Detector


def test_resolution_must_be_1008():
    with pytest.raises(ValueError, match="1008"):
        Sam3Detector(resolution=784, device="cpu", load_from_hf=False)
    with pytest.raises(ValueError, match="1008"):
        Sam3Detector(resolution=768, device="cpu", load_from_hf=False)


def test_resolution_1008_accepted():
    det = Sam3Detector(resolution=1008, device="cpu", load_from_hf=False)
    assert det.resolution == 1008
