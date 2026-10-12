# Finer labelling + reviewed thresholds

How to go beyond whole-object labels (angle grinder / screwdriver / …), and which
score gates to use next — based on all past AM07 and corpus batch evidence.

## 1. What is realistic with SAM3 (mask path)

| Tier | Labels | How | Ready? |
|------|--------|-----|--------|
| **A. Whole objects** (current fine corpus) | `angle grinder`, `screwdriver`, `storage box`, … | Text prompts + gaze-in-mask | **Yes — shipped** |
| **B. Coarse AG parts** | `grinder body`, `side handle`, `grinder guard`, `grinding disc`, `battery pack` | Extra prompts + **prefer smallest mask** when gaze hits AG+part | **Pilot ready** (`am07_sam3_parts_pilot.yaml`) |
| **C. Screws / spatial parts** | `screw`, `top-left screw`, `screw 1` | Open-vocab SAM3 alone is **not reliable** | Needs Tobii AOIs, click/point prompts, or a trained part model |

**Important:** SAM3 does **not** invent “screw 1” vs “screw 2”. Numbered / spatial names need tracking or human annotation. Coarse parts (handle, guard, disc) are the next honest step.

## 2. Recommended workflow for more detail

1. **Keep whole-object detect** (so sequences still work when parts miss).
2. **Add coarse part prompts** only on a pilot (AM07 first) — see `metaSAM3/prompts_fine_parts.txt`.
3. **Assign with `prefer_smallest_containing: true`** — if gaze is on both the whole grinder and the side handle, pick the **smaller** mask (the part).
4. **Do not put `screw` in the full 290 batch yet** — validate on AM07 overlay; screws false-fire on any small metal.
5. Optional later: only keep part detections that **overlap** a parent `angle grinder` mask (parent filter) to cut FPs.
6. For true screw identity: draw Tobii AOIs or run an interactive/point-prompt pass on fixation frames.

## 3. Threshold review (past figures → recommended)

### Evidence

| Era | Default | AG | Manual | Assign | What we learned |
|-----|--------:|---:|-------:|-------:|-----------------|
| SAM3 AOI dense / 5-vid / ~291 corpus | 0.55 | **0.25** | **0.70** | 0.25 | High Manual cut AG→Manual FPs; AG 0.25 OK on trimmed clips; AA05 Tools fixed by **removing** screwdriver prompts in AOI mode |
| Fine full-video 290 (first pass) | 0.50 | 0.18→**0.12** | **0.55** | **0.12** | Handheld AG scores often **0.25–0.35** (AM07: 96 dets / 51 frames, mean 0.35, **0 AG before ~77s** at t=34s miss). Gate 0.12 + synonyms + min area 0.8% needed |
| Sheet pollution | — | — | — | — | `pliers` / `hammer` / standalone `grinding disc` → **remove** from object prompts |

### Recommended gates for next runs

| Concept | Detect gate | Why |
|---------|------------:|-----|
| **angle grinder** (+ synonyms) | **0.12** | Best recall for egocentric/handheld; pair with `min_box_area_frac: 0.008` |
| **instruction manual** | **0.60** | Between fine 0.55 and AOI 0.70 — fewer false manuals without starving recall |
| **screwdriver** | **0.48** | Slightly above 0.45; still below default |
| **storage box** | **0.45** | Boxes score well (~0.59 mean on AM07) |
| **wrench** | **0.50** | Default |
| **Default** (unknown concepts) | **0.50** | Keep |
| **Assign** | **0.12** | Must be ≤ AG detect gate or weak grinders never become attended |
| **Sequence quality** | ≥2 samples, ≥50% agreement | Unchanged — kept weak fixations out of sheets |

### Part gates (pilot only — stricter)

| Part | Detect gate | Notes |
|------|------------:|-------|
| grinder body | 0.25 | |
| side handle | 0.28 | |
| grinder guard | 0.28 | |
| grinding disc | 0.32 | Only in **parts** prompt file; was FP as whole-object |
| battery pack | 0.35 | |
| screw *(optional)* | 0.40 | Off by default in prompt file comments |

## 4. Configs / commands

**Object-level rebatch (reviewed thresholds, no parts):**

```bash
# configs/batch_sam3_full_fine.yaml  — thresholds updated to Section 3
python -m gaze_objects.cli batch --config configs/batch_sam3_full_fine.yaml
python -m gaze_objects.cli aggregate-sequences --output-root outputs/batch_sam3_full_fine
```

**AM07 parts pilot (whole objects + coarse AG parts):**

```bash
rm -rf outputs/am07_sam3_parts_pilot
python -m gaze_objects.cli sam-overlay --config configs/am07_sam3_parts_pilot.yaml
# Check: side handle / guard / disc masks while holding the grinder
python - <<'PY'
import pandas as pd
d=pd.read_csv('outputs/am07_sam3_parts_pilot/detections.csv')
print(d['normalized_label'].value_counts())
PY
```

## 5. VRAM note

Each extra prompt = one SAM3 forward pass per frame. Parts file adds ~5 concepts → slower / more VRAM pressure. Pilot on AM07 before scaling to 290. Keep `resolution: 1008`, one GPU job, `stride` 5–8.
