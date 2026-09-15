"""IDEA-Research DINO adapter (CUDA workstation).

Imports torch/DINO only inside methods so audit/overlay/assign stay CPU-safe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from gaze_objects.detectors import Detection
from gaze_objects.detectors.label_map import load_class_definitions, normalize_label

COCO_LABELS = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


class IdeaDinoDetector:
    """Load IDEA-Research DINO once; run per-frame inference."""

    def __init__(
        self,
        *,
        dino_repo: str | Path,
        config_file: str | Path,
        checkpoint: str | Path,
        device: str = "cuda",
        score_threshold: float = 0.3,
        class_map_file: str | Path | None = None,
    ) -> None:
        self.dino_repo = Path(dino_repo).expanduser().resolve()
        self.config_file = Path(config_file).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = device
        self.score_threshold = float(score_threshold)
        self.class_defs = load_class_definitions(class_map_file)
        self.model = None
        self.postprocessors = None
        self.transform = None
        self._torch = None
        self._Image = None

    def _ensure_repo_on_path(self) -> None:
        repo = str(self.dino_repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)

    def load(self) -> None:
        import torch
        from PIL import Image

        self._ensure_repo_on_path()
        import datasets.transforms as T  # type: ignore
        from util.slconfig import SLConfig  # type: ignore
        from util.misc import clean_state_dict  # type: ignore
        from models.registry import MODULE_BUILD_FUNCS  # type: ignore

        self._torch = torch
        self._Image = Image

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "idea_dino backend requires CUDA, but torch.cuda.is_available() is False. "
                "Use the workstation dino_gpu env, or set detector.backend: mock on CPU."
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"DINO checkpoint not found: {self.checkpoint}. "
                "Download a ResNet-50 4-scale weight before running detect."
            )
        if not self.config_file.is_file():
            raise FileNotFoundError(f"DINO config not found: {self.config_file}")

        args = SLConfig.fromfile(str(self.config_file))
        args.device = self.device
        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        model, _criterion, postprocessors = build_func(args)
        ckpt = torch.load(str(self.checkpoint), map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(clean_state_dict(state), strict=False)
        model.eval()
        model.to(self.device)

        normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])
        self.model = model
        self.postprocessors = postprocessors

    def detect_frame(self, frame_bgr: np.ndarray, frame_key: str) -> list[Detection]:
        if self.model is None:
            self.load()
        assert self.model is not None and self.transform is not None
        assert self._torch is not None and self._Image is not None
        torch = self._torch

        rgb = frame_bgr[:, :, ::-1]
        image = self._Image.fromarray(rgb)
        orig_w, orig_h = image.size
        tensor, _ = self.transform(image, None)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model([tensor])
            target_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)
            results = self.postprocessors["bbox"](outputs, target_sizes)[0]

        scores = results["scores"].detach().cpu()
        labels = results["labels"].detach().cpu()
        boxes = results["boxes"].detach().cpu()

        dets: list[Detection] = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.score_threshold:
                continue
            label_idx = int(labels[i])
            raw = COCO_LABELS[label_idx] if 0 <= label_idx < len(COCO_LABELS) else str(label_idx)
            x0, y0, x1, y1 = [float(v) for v in boxes[i].tolist()]
            dets.append(
                Detection(
                    detection_id=f"{frame_key}_dino_{i}",
                    frame_key=frame_key,
                    x_min=x0,
                    y_min=y0,
                    x_max=x1,
                    y_max=y1,
                    raw_label=raw,
                    normalized_label=normalize_label(raw, self.class_defs),
                    score=score,
                    model_provenance={
                        "backend": "idea_dino",
                        "model_name": "IDEA-Research/DINO",
                        "checkpoint": str(self.checkpoint),
                        "config_file": str(self.config_file),
                        "label_index": label_idx,
                    },
                )
            )
        return dets
