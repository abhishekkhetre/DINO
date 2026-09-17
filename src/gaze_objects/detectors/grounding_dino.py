"""Grounding DINO adapter (text-prompted detection).

Uses IDEA-Research/GroundingDINO. Imports are lazy so CPU audit/assign stay light.
Prompts are hypotheses — confirm object wording with your supervisor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from gaze_objects.detectors import Detection
from gaze_objects.detectors.label_map import load_class_definitions, normalize_label

# Default pilot prompt. Periods separate categories (Grounding DINO convention).
DEFAULT_TEXT_PROMPT = (
    "angle grinder . instruction manual . storage box . screwdriver . wrench . tool"
)


class GroundingDinoDetector:
    def __init__(
        self,
        *,
        grounding_repo: str | Path,
        config_file: str | Path,
        checkpoint: str | Path,
        text_prompt: str = DEFAULT_TEXT_PROMPT,
        device: str = "cuda",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        class_map_file: str | Path | None = None,
    ) -> None:
        self.grounding_repo = Path(grounding_repo).expanduser().resolve()
        self.config_file = Path(config_file).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.text_prompt = text_prompt
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.class_defs = load_class_definitions(class_map_file)
        self.model = None

    def _ensure_repo_on_path(self) -> None:
        repo = str(self.grounding_repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)

    def load(self) -> None:
        import torch

        self._ensure_repo_on_path()
        from groundingdino.util.inference import load_model  # type: ignore

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "grounding_dino backend requested CUDA, but torch.cuda.is_available() is False."
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Grounding DINO checkpoint not found: {self.checkpoint}. "
                "Download groundingdino_swint_ogc.pth (see README / guide)."
            )
        if not self.config_file.is_file():
            raise FileNotFoundError(f"Grounding DINO config not found: {self.config_file}")

        self.model = load_model(str(self.config_file), str(self.checkpoint), device=self.device)

    def detect_frame(self, frame_bgr: np.ndarray, frame_key: str) -> list[Detection]:
        if self.model is None:
            self.load()
        assert self.model is not None

        from groundingdino.util.inference import predict  # type: ignore
        import groundingdino.datasets.transforms as T  # type: ignore
        from PIL import Image

        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        image_pil = Image.fromarray(rgb)

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image_pil, None)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        # boxes: cxcywh normalized 0-1
        dets: list[Detection] = []
        for i in range(len(boxes)):
            cx, cy, bw, bh = [float(v) for v in boxes[i].tolist()]
            score = float(logits[i])
            phrase = str(phrases[i]).strip()
            x_min = (cx - bw / 2.0) * w
            y_min = (cy - bh / 2.0) * h
            x_max = (cx + bw / 2.0) * w
            y_max = (cy + bh / 2.0) * h
            dets.append(
                Detection(
                    detection_id=f"{frame_key}_gdino_{i}",
                    frame_key=frame_key,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    raw_label=phrase,
                    normalized_label=normalize_label(phrase, self.class_defs),
                    score=score,
                    model_provenance={
                        "backend": "grounding_dino",
                        "model_name": "IDEA-Research/GroundingDINO",
                        "checkpoint": str(self.checkpoint),
                        "config_file": str(self.config_file),
                        "text_prompt": self.text_prompt,
                        "box_threshold": self.box_threshold,
                        "text_threshold": self.text_threshold,
                    },
                )
            )
        return dets
