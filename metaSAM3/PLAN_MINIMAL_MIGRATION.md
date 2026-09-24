# Minimal-effort plan: Grounding DINO → Meta SAM 3

## 1. How SAM 3 maps onto our pipeline

| Our stage | Change for SAM 3? | Notes |
| --- | --- | --- |
| A — TSV audit | No | Unchanged |
| B — sync + overlay | No | Unchanged |
| C — **detect** | **Yes** | New backend `sam3` producing same `Detection` rows |
| C — **assign** | No* | Same gaze-in-box; *optional later: gaze-in-mask |
| D — evaluate | No | Same AOI comparison; reuse `label_map.py` |
| E — sequences | Later bonus | SAM 3 video tracker can help identity over time |

**Contract to keep (do not redesign):** each detection still has  
`x_min,y_min,x_max,y_max`, `raw_label`, `normalized_label`, `score`  
→ existing `assign` / `evaluate` / configs keep working.

Grounding DINO:

```text
frame → text prompt (multi-phrase) → boxes + labels → detections.csv
```

SAM 3 (image mode, minimal):

```text
frame → set_image → for each noun phrase: set_text_prompt → boxes + masks + scores
      → convert boxes to Detection (raw_label = phrase) → same detections.csv
```

Official sketch:

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)
state = processor.set_image(image)
out = processor.set_text_prompt(state=state, prompt="angle grinder")
masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
```

---

## 2. Why this is a small shift (reuse almost everything)

**Keep as-is**

- `configs/am07_recording50.yaml` sync fields  
- `gaze_objects.assign` / `evaluate` / CLI `assign` `evaluate`  
- `label_map.py` (and Manual/`instruction` fix)  
- AM07 pilot protocol: same 30–60 s, 60 frames, Stage D metrics  

**Add only**

1. `src/gaze_objects/detectors/sam3_meta.py` — adapter implementing `detect_frame()` like `grounding_dino.py`  
2. One branch in `stage_c._build_detector` for `backend: sam3`  
3. Config `configs/am07_stage_c_sam3.yaml` (copy of grounding config; change backend + paths)  
4. Env on workstation: conda `sam3`, HF login, weights  

**Do not redo:** Grounding install, COCO IDEA-DINO path, Stage A/B.

---

## 3. Effort-minimizing choices

| Choice | Recommendation | Why |
| --- | --- | --- |
| Image vs video API first | **Image** per selected frame | Matches current Stage C; fewer moving parts |
| Multi-phrase | Loop phrases from `prompts_pilot.txt` | SAM 3 likes one short noun phrase per call |
| Masks | Store optional; **assign with boxes first** | Zero change to assign; add gaze-in-mask later if needed |
| SAM 3 vs 3.1 | Start with **sam3** HF access you can get; try 3.1 after | Less dependency churn |
| Parallel Grounding | Keep grounding config runnable | Fair A/B on same clip |
| Fine-tune SAM 3 | **Not** in first migration | Pilot + Stage D first |

---

## 4. Phased plan (minimal path)

### Phase 0 — Done on laptop (this folder)
- [x] Create `metaSAM3/`  
- [x] Clone `facebookresearch/sam3`  
- [x] Document checkpoints + migration plan  

### Phase 1 — Access & env (workstation)
- [x] User: installation + HF access done  

### Phase 2 — Thin adapter (code ready; run on workstation)
- [x] `src/gaze_objects/detectors/sam3_meta.py`  
- [x] `backend: sam3` in `_build_detector`  
- [x] `configs/am07_stage_c_sam3.yaml`  
- [ ] Run: `detect` → `assign` → `evaluate`  
- [ ] Compare `evaluation_summary.json` to Grounding

**Success criterion:** same Stage D JSON schema; conditional AOI accuracy reported for SAM 3 on AM07 pilot.

### Phase 3 — Decide
- If SAM 3 ≥ Grounding on conditional accuracy (and fewer Manual naming bugs): make `sam3` the default pilot backend; keep Grounding as baseline in STATUS.  
- If worse: tune phrases/thresholds; only then consider masks or video mode.  
- Parts / assembly ontology: same as discussed earlier — after coarse SAM 3 baseline.

### Phase 4 — Optional upgrades (only if Phase 2 works)
- Gaze-in-**mask** assignment for overlapping parts  
- Video predictor for identity across frames (Stage E)  
- Exemplar prompts (crop of “this is a disc”) for hard parts  

---

## 5. Config sketch (future `am07_stage_c_sam3.yaml`)

```yaml
# Same as am07_stage_c_grounding.yaml for tsv/video/clip/sync...
output_dir: "outputs/am07_stage_c_sam3"

detector:
  backend: sam3
  sam3_repo: "metaSAM3/sam3"   # or workstation absolute path
  device: cuda
  score_threshold: 0.30
  # One phrase per line; adapter loops
  text_concepts_file: "metaSAM3/prompts_pilot.txt"
  class_map_file: "configs/class_definitions.yaml"
```

---

## 6. Comparison matrix (for supervisor)

| | Grounding DINO (done) | SAM 3 (planned) |
| --- | --- | --- |
| Role | Text → boxes | Text → boxes + masks (+ track) |
| AM07 pilot | 60 frames, Stage D done | Same protocol |
| Integration cost | High (done) | **Low** if adapter only |
| Env risk | Known working | New CUDA/torch pins |
| Parts / fine classes | Prompts | Prompts + optional exemplars/masks |
| Video sequences | External | Built-in tracker |

---

## 7. What we are *not* doing in the first shift

- Replacing Stages A–D architecture  
- Dropping Grounding results (keep for comparison)  
- Fine-tuning SAM 3  
- Full ~400-video batch  
- Laptop GPU inference  

---

## 8. Immediate next actions checklist

1. **You:** Request HF access to https://huggingface.co/facebook/sam3  
2. **Workstation:** Create `sam3` conda env; install repo; login; one-image smoke test  
3. **Code (when ready):** Add `sam3_meta.py` + config + `_build_detector` branch  
4. **Run:** AM07 detect → assign → evaluate → side-by-side with `outputs/am07_stage_c_grounding`  

Estimated engineering for Phase 2: **~1 adapter file + ~20 lines wiring + 1 YAML** — same CLI you already know.
