"""Meta SAM 3 adapter (text-prompted concept segmentation → boxes).

Uses facebookresearch/sam3. Imports are lazy so CPU audit/assign stay light.
One short noun phrase per forward pass (SAM 3 convention); results are merged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gaze_objects.detectors import Detection
from gaze_objects.detectors.label_map import load_class_definitions, normalize_label

DEFAULT_CONCEPTS = (
    "angle grinder",
    "instruction manual",
    "storage box",
    "screwdriver",
    "wrench",
    "tool",
)


def load_text_concepts(
    concepts: list[str] | None = None,
    concepts_file: str | Path | None = None,
) -> list[str]:
    if concepts:
        return [c.strip() for c in concepts if str(c).strip()]
    if concepts_file:
        path = Path(concepts_file)
        if not path.is_file():
            raise FileNotFoundError(f"SAM3 text_concepts_file not found: {path}")
        lines: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("```"):
                continue
            # Skip markdown prose lines that are not short noun phrases
            if line.startswith("|") or line.lower().startswith("run once"):
                continue
            if "→" in line or line.endswith(":"):
                continue
            lines.append(line)
        if lines:
            return lines
    return list(DEFAULT_CONCEPTS)


class Sam3Detector:
    def __init__(
        self,
        *,
        device: str = "cuda",
        score_threshold: float = 0.30,
        text_concepts: list[str] | None = None,
        text_concepts_file: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        load_from_hf: bool = True,
        class_map_file: str | Path | None = None,
        resolution: int = 1008,
    ) -> None:
        self.device = device
        self.score_threshold = float(score_threshold)
        self.concepts = load_text_concepts(text_concepts, text_concepts_file)
        self.checkpoint_path = (
            str(Path(checkpoint_path).expanduser().resolve())
            if checkpoint_path
            else None
        )
        self.load_from_hf = bool(load_from_hf)
        self.class_defs = load_class_definitions(class_map_file)
        self.resolution = int(resolution)
        self.model = None
        self.processor = None

    def load(self) -> None:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "sam3 backend requested CUDA, but torch.cuda.is_available() is False."
            )
        if not self.concepts:
            raise ValueError("sam3 backend needs at least one text concept.")

        self.model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=self.checkpoint_path,
            load_from_HF=self.load_from_hf and self.checkpoint_path is None,
            enable_segmentation=True,
        )
        self.processor = Sam3Processor(
            self.model,
            resolution=self.resolution,
            device=self.device,
            confidence_threshold=self.score_threshold,
        )

    def detect_frame(self, frame_bgr: np.ndarray, frame_key: str) -> list[Detection]:
        if self.model is None or self.processor is None:
            self.load()
        assert self.processor is not None

        from PIL import Image

        rgb = frame_bgr[:, :, ::-1]
        image_pil = Image.fromarray(rgb)
        state = self.processor.set_image(image_pil)

        dets: list[Detection] = []
        det_i = 0
        for concept in self.concepts:
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt=concept, state=state)
            boxes = state.get("boxes")
            scores = state.get("scores")
            if boxes is None or scores is None or len(boxes) == 0:
                continue

            boxes_np = boxes.detach().cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            for j in range(len(boxes_np)):
                score = float(scores_np[j])
                if score < self.score_threshold:
                    continue
                x_min, y_min, x_max, y_max = [float(v) for v in boxes_np[j].tolist()]
                dets.append(
                    Detection(
                        detection_id=f"{frame_key}_sam3_{det_i}",
                        frame_key=frame_key,
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                        raw_label=concept,
                        normalized_label=normalize_label(concept, self.class_defs),
                        score=score,
                        model_provenance={
                            "backend": "sam3",
                            "model_name": "facebook/sam3",
                            "checkpoint": self.checkpoint_path or "huggingface:facebook/sam3",
                            "text_concept": concept,
                            "score_threshold": self.score_threshold,
                        },
                    )
                )
                det_i += 1
        return dets
