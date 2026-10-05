"""SAM3 resolution must match ViT patch size."""

from __future__ import annotations

import pytest

from gaze_objects.detectors.sam3_meta import Sam3Detector


def test_resolution_must_be_multiple_of_14():
    with pytest.raises(ValueError, match="multiple of 14"):
        Sam3Detector(resolution=768, device="cpu", load_from_hf=False)


def test_resolution_784_accepted():
    det = Sam3Detector(resolution=784, device="cpu", load_from_hf=False)
    assert det.resolution == 784
