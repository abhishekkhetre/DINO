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


def _cuda_mem_mib() -> tuple[float, float, float] | None:
    """Return (free_MiB, total_MiB, allocated_MiB) or None if CUDA unavailable."""
    import torch

    if not torch.cuda.is_available():
        return None
    free_b, total_b = torch.cuda.mem_get_info()
    allocated_b = torch.cuda.memory_allocated()
    return free_b / (1024**2), total_b / (1024**2), allocated_b / (1024**2)


def assert_cuda_headroom(min_free_gib: float = 6.0) -> None:
    """Fail fast when another process has drained the GPU (common OOM cause).

    SAM3 weights alone take ~3.4 GiB at res 1008; inference needs several more
    GiB of free headroom. On a 12GB card that means roughly one Python job.
    """
    import torch

    if not torch.cuda.is_available():
        return
    mem = _cuda_mem_mib()
    if mem is None:
        return
    free_mib, total_mib, allocated_mib = mem
    free_gib = free_mib / 1024.0
    print(
        f"[sam3] CUDA mem free={free_mib:.0f}MiB "
        f"total={total_mib:.0f}MiB allocated={allocated_mib:.0f}MiB",
        flush=True,
    )
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"CUDA only has {free_gib:.2f} GiB free (need ≥{min_free_gib:.1f} GiB "
            "before load). Another process is likely holding the GPU — run "
            "`nvidia-smi`, kill every other `python` PID (Type C), then start "
            "exactly one batch. Dual SAM3 jobs on a 12GB card will OOM."
        )


def release_cuda_memory() -> None:
    """Best-effort free of cached CUDA blocks (call after unloading a model)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001 — cleanup must never raise
        pass


class Sam3Detector:
    def __init__(
        self,
        *,
        device: str = "cuda",
        score_threshold: float = 0.30,
        score_threshold_by_concept: dict[str, float] | None = None,
        text_concepts: list[str] | None = None,
        text_concepts_file: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        load_from_hf: bool = True,
        class_map_file: str | Path | None = None,
        resolution: int = 1008,
        label_map_mode: str = "study_aoi",
        enable_segmentation: bool = True,
        min_free_vram_gib: float = 6.0,
    ) -> None:
        self.device = device
        self.score_threshold = float(score_threshold)
        self.score_threshold_by_concept = {
            str(k).strip().lower(): float(v)
            for k, v in (score_threshold_by_concept or {}).items()
        }
        self.concepts = load_text_concepts(text_concepts, text_concepts_file)
        self.checkpoint_path = (
            str(Path(checkpoint_path).expanduser().resolve())
            if checkpoint_path
            else None
        )
        self.load_from_hf = bool(load_from_hf)
        self.class_defs = load_class_definitions(class_map_file)
        self.resolution = int(resolution)
        # Stock facebook/sam3 ViT is built with img_size=1008 and RoPE freqs for that
        # size. Sam3Processor.resolution must match or reshape_for_broadcast asserts.
        if self.resolution != 1008:
            raise ValueError(
                f"sam3 detector.resolution={self.resolution} is unsupported. "
                "Meta's image model is built for img_size=1008 only; other sizes "
                "break RoPE (vitdet.reshape_for_broadcast AssertionError). "
                "Use resolution: 1008 and free VRAM by running a single GPU job."
            )
        self.label_map_mode = str(label_map_mode or "study_aoi")
        # Sam3Processor._forward_grounding always reads pred_masks — keep True.
        # We discard masks after copying boxes to CPU to limit peak VRAM.
        self.enable_segmentation = bool(enable_segmentation)
        self.min_free_vram_gib = float(min_free_vram_gib)
        self.model = None
        self.processor = None

    def _threshold_for(self, concept: str) -> float:
        return float(
            self.score_threshold_by_concept.get(
                concept.strip().lower(), self.score_threshold
            )
        )

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

        # Match Meta example notebooks (Ampere+ / SAM3 image inference).
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            assert_cuda_headroom(self.min_free_vram_gib)

        self.model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            checkpoint_path=self.checkpoint_path,
            load_from_HF=self.load_from_hf and self.checkpoint_path is None,
            enable_segmentation=self.enable_segmentation,
        )
        self.processor = Sam3Processor(
            self.model,
            resolution=self.resolution,
            device=self.device,
            # Processor gate uses the global floor; per-concept cuts applied below.
            confidence_threshold=min(
                [self.score_threshold, *self.score_threshold_by_concept.values()]
                or [self.score_threshold]
            ),
        )
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            mem = _cuda_mem_mib()
            if mem is not None:
                free_mib, total_mib, allocated_mib = mem
                print(
                    f"[sam3] model loaded segmentation={self.enable_segmentation} "
                    f"resolution={self.resolution} concepts={len(self.concepts)} "
                    f"free={free_mib:.0f}MiB allocated={allocated_mib:.0f}MiB "
                    f"total={total_mib:.0f}MiB",
                    flush=True,
                )

    def unload(self) -> None:
        """Drop model/processor references and release CUDA cache."""
        self.processor = None
        self.model = None
        if self.device.startswith("cuda"):
            release_cuda_memory()
            mem = _cuda_mem_mib()
            if mem is not None:
                free_mib, _total_mib, allocated_mib = mem
                print(
                    f"[sam3] unloaded free={free_mib:.0f}MiB allocated={allocated_mib:.0f}MiB",
                    flush=True,
                )

    def detect_frame(self, frame_bgr: np.ndarray, frame_key: str) -> list[Detection]:
        if self.model is None or self.processor is None:
            self.load()
        assert self.processor is not None

        import torch
        from PIL import Image

        rgb = frame_bgr[:, :, ::-1]
        image_pil = Image.fromarray(rgb)

        # Official SAM3 image demos run under bfloat16 autocast; without it,
        # linear layers can mix BFloat16 weights with Float32 activations.
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else torch.autocast("cpu", enabled=False)
        )

        dets: list[Detection] = []
        det_i = 0
        with torch.inference_mode():
            with autocast_ctx:
                state = self.processor.set_image(image_pil)
                for concept in self.concepts:
                    concept_thr = self._threshold_for(concept)
                    self.processor.reset_all_prompts(state)
                    state = self.processor.set_text_prompt(prompt=concept, state=state)
                    boxes = state.get("boxes")
                    scores = state.get("scores")
                    # Drop full-res masks immediately — assign uses boxes only.
                    if self.device.startswith("cuda"):
                        for key in ("masks", "masks_logits"):
                            if key in state:
                                state[key] = None
                    if boxes is None or scores is None or len(boxes) == 0:
                        if self.device.startswith("cuda"):
                            for key in ("boxes", "scores", "masks", "masks_logits"):
                                state[key] = None
                        continue

                    boxes_np = boxes.detach().float().cpu().numpy()
                    scores_np = scores.detach().float().cpu().numpy()
                    # Release GPU tensors before the next concept forward.
                    if self.device.startswith("cuda"):
                        for key in ("boxes", "scores", "masks", "masks_logits"):
                            state[key] = None
                        del boxes, scores
                        torch.cuda.empty_cache()
                    for j in range(len(boxes_np)):
                        score = float(scores_np[j])
                        if score < concept_thr:
                            continue
                        x_min, y_min, x_max, y_max = [
                            float(v) for v in boxes_np[j].tolist()
                        ]
                        dets.append(
                            Detection(
                                detection_id=f"{frame_key}_sam3_{det_i}",
                                frame_key=frame_key,
                                x_min=x_min,
                                y_min=y_min,
                                x_max=x_max,
                                y_max=y_max,
                                raw_label=concept,
                                normalized_label=normalize_label(
                                    concept,
                                    self.class_defs,
                                    mode=self.label_map_mode,
                                ),
                                score=score,
                                model_provenance={
                                    "backend": "sam3",
                                    "model_name": "facebook/sam3",
                                    "checkpoint": self.checkpoint_path
                                    or "huggingface:facebook/sam3",
                                    "text_concept": concept,
                                    "score_threshold": concept_thr,
                                },
                            )
                        )
                        det_i += 1

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return dets
