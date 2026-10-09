"""Canonical threshold / min-area lookup for AG synonyms."""

from __future__ import annotations

import numpy as np

from gaze_objects.detectors.sam3_meta import Sam3Detector


def test_threshold_follows_canonical_label():
    det = Sam3Detector(
        device="cpu",
        score_threshold=0.50,
        score_threshold_by_concept={"angle grinder": 0.12},
        text_concepts=["power tool", "screwdriver"],
        label_map_mode="fine",
        load_from_hf=False,
    )
    assert det._threshold_for("power tool") == 0.12
    assert det._threshold_for("disk grinder") == 0.12
    assert det._threshold_for("screwdriver") == 0.50


def test_min_area_frac_follows_canonical_label():
    det = Sam3Detector(
        device="cpu",
        text_concepts=["power tool"],
        label_map_mode="fine",
        load_from_hf=False,
        min_box_area_frac_by_concept={"angle grinder": 0.008},
    )
    assert det._min_area_frac_for("power tool") == 0.008
    assert det._min_area_frac_for("screwdriver") == 0.0


def test_min_area_rejects_tiny_boxes_without_model(monkeypatch):
    """Unit-level area gate: simulate processor outputs without loading SAM3."""
    det = Sam3Detector(
        device="cpu",
        score_threshold=0.10,
        score_threshold_by_concept={"angle grinder": 0.10},
        text_concepts=["power tool"],
        label_map_mode="fine",
        load_from_hf=False,
        min_box_area_frac_by_concept={"angle grinder": 0.05},
    )

    class _FakeProc:
        def set_image(self, _img):
            return {}

        def reset_all_prompts(self, state):
            return None

        def set_text_prompt(self, prompt, state):
            import torch

            # Tiny box (~1% of 100x100) vs large box (~30%).
            state["boxes"] = torch.tensor(
                [[1.0, 1.0, 11.0, 11.0], [10.0, 10.0, 70.0, 60.0]],
                dtype=torch.float32,
            )
            state["scores"] = torch.tensor([0.9, 0.8], dtype=torch.float32)
            state["masks"] = None
            return state

    det.model = object()
    det.processor = _FakeProc()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        import torch  # noqa: F401
    except ImportError:
        return  # skip if no torch in this env
    outs = det.detect_frame(frame, "f0")
    assert len(outs) == 1
    assert outs[0].x_min == 10.0
