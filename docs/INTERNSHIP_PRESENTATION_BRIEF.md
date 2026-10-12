# Internship Presentation Brief — Gaze-to-Object Pipeline for Angle-Grinder Assembly

**Document purpose:** Feed this file to a presentation-building AI agent. It is a complete briefing of the internship work (September–October 2026): problem, pipeline, results, demos, and talking points for office colleagues.

**Author / intern:** Abhishek Khetre (KIT M.Sc. CS; HiWi / internship context)  
**Project:** Offline attended-object identification from Tobii Pro Lab exports + egocentric scene video  
**Repository:** https://github.com/abhishekkhetre/DINO  
**Primary code package:** `src/gaze_objects/`  
**Lab workstation:** Ubuntu (`ifab-ws22-09`), NVIDIA **RTX 3080 Ti 12 GB**, conda env `sam3`  
**Pilot recording:** AM07AM07 / Recording 50 / CRC 1574 / scene video 1920×1080  

**How an AI presentation agent should use this file**
- Prefer **Section 1–4 and 7–9** for the main talk (problem → pipeline → results → demo → takeaways).
- Use **Section 5–6** for optional “journey / methods evolution” slides if time allows.
- Use **Section 10–12** for appendix / appendix / Q&A prep.
- Keep slides visual: pipeline diagram, one overlay screenshot, one sequence string, one transition table, one QC number.
- Audience is office/research colleagues — explain Tobii, SAM3, and “attended object” in plain language once, then use consistent terms.

---

## 1. One-sentence outcome

We built a reproducible offline pipeline that takes **Tobii eye-tracking exports + first-person scene video** and delivers **which assembly object the participant looked at, in what order, across ~290 recordings**, using **Meta SAM 3 masks** and Tobii fixations — ready for psychology / workflow analysis of angle-grinder assembly and disassembly.

---

## 2. Problem statement (why this work exists)

### Research need
In CRC / assembly studies, participants wear Tobii glasses while assembling or disassembling an **angle grinder** and related kit (manual, storage boxes, screwdrivers, wrench, …). Tobii already records:

- where the eyes look (pixel coordinates on the scene camera),
- when a **fixation** starts/ends (stable look),
- optionally manual **AOI** (Area of Interest) labels drawn in Pro Lab.

What was missing for automated analysis:

> At each moment in the **scene video**, which **object** was under the gaze — and across the whole task, what was the **order of attended objects**?

Manual AOI labelling in Pro Lab does not scale well to hundreds of videos and does not automatically give fine tool names (screwdriver vs wrench vs grinder).

### Internship goal (confirmed)
Build a **workflow for real-world gaze recordings** whose immediate focus is **attended-object identification in the scene video** — not Pro Lab writeback or reference-image geometry as a prerequisite.

### Non-goals (scope discipline)
- We do **not** estimate gaze from eye images; Tobii provides gaze.
- We do **not** claim Tobii AOIs are perfect ground truth; they are used for **QC** when labels match.
- Full instance IDs (`storage box 1`) and screw-level parts are **future work**, not the shipped outcome.

---

## 3. What colleagues should remember (3 bullets)

1. **Pipeline works end-to-end:** Tobii TSV → sync → SAM3 detect (masks) → gaze-in-mask assign → fixation sequences → corpus sheets.
2. **Scaled result:** **290/290** full-video recordings processed with fine object labels → **18,028** attention episodes and gaze-order strings (e.g. `storage box > angle grinder > screwdriver > instruction manual`).
3. **Current technical stance:** assignment is **mask-based** (silhouette), not loose boxes; illustration overlay exists for AM07; prompts/thresholds tuned for egocentric angle grinders.

---

## 4. Current pipeline — what it is and how it delivers

### 4.1 Conceptual pipeline (use this diagram on a slide)

```text
 Tobii Pro Lab export (.tsv)          Scene video (.mp4)
            │                                  │
            ▼                                  ▼
   ┌─────────────────┐               ┌─────────────────┐
   │ Stage A — Audit │               │ Stage B — Video │
   │ prepare gaze    │               │ timing / frames │
   └────────┬────────┘               └────────┬────────┘
            │                                  │
            └────────────┬─────────────────────┘
                         ▼
              ┌─────────────────────┐
              │ Stage B — Sync      │
              │ gaze sample ↔ frame │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Stage C — Detect    │
              │ SAM3 text → masks   │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Stage C — Assign    │
              │ gaze-in-MASK hit    │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Stage E — Sequences │
              │ Tobii fixations →   │
              │ attended-object run │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Aggregate (corpus)  │
              │ strings + transitions│
              └─────────────────────┘
                         │
                         ▼
         Psychology / workflow sheets + optional overlay video
```

Optional Stage D **evaluate** compares assignments to Tobii AOIs when running in AOI mode (QC only).

### 4.2 Division of labor (critical clarification slide)

| System | Answers |
|--------|---------|
| **Tobii** | *When* and *where* the eyes looked (fixation index, duration, XY pixels) |
| **SAM3 + assign** | *What object* was under that gaze point (mask / silhouette) |
| **Sequences + aggregate** | *In what order* labelled looks happened across the task |

**SAM3 never sees eye movements.** It only segments objects on video frames. Gaze comes only from Tobii.

### 4.3 Key concepts in plain language

**Fixation (`fixation_index`)**  
Tobii’s `Eye movement type index` for rows where `Eye movement type == Fixation`. Many raw sample rows share one index = one stable look. We do **not** invent fixations.

**Attended object**  
For a fixation, majority vote of gaze samples that landed on a SAM mask (status `assigned`), after quality gates.

**Sequence / `sequence_id`**  
Consecutive fixations with the **same** attended label are merged into one attention episode. Example: three fixations on `screwdriver` in a row → one sequence with `n_fixations=3`. An unlabelled gap **breaks** the run.

**Sequence string**  
Ordered labels across episodes, e.g.  
`storage box > angle grinder > screwdriver > instruction manual > storage box`.

**Mask vs box**  
- **Box:** axis-aligned rectangle (cheaper, looser).  
- **Mask:** pixel silhouette of the object (tighter; preferred for tools and overlap).  
**Current policy:** `hit_test: mask` — gaze must land on the silhouette (falls back to box only if mask missing).

### 4.4 Stages and code map

| Stage | CLI command | Main modules | Primary outputs |
|-------|-------------|--------------|-----------------|
| A Audit | `audit` | `io.py`, `audit.py`, `reference.py` | prepared gaze, AOI reference tables, audit JSON |
| B Video / sync / overlay | `inspect-video`, `overlay` | `video.py`, `sync.py`, `visualize.py` | frame table, gaze↔frame associations, sync check video |
| C Detect | `detect` | `stage_c.py`, `detectors/sam3_meta.py`, `label_map.py` | `detections.csv`, optional `detection_masks.npz` |
| C Assign | `assign` | `assign.py`, `masks.py` | `gaze_assignments.csv` |
| D Evaluate (optional) | `evaluate` | `evaluate.py` | AOI agreement metrics (AOI mode only) |
| E Sequences | `sequences` | `sequences.py` | `fixation_attended_objects.csv`, `attention_sequences.csv` |
| Batch | `batch`, `discover-pairs` | `batch.py` | per-recording folders + `batch_qc_summary.csv` |
| Aggregate | `aggregate-sequences` | `aggregate.py` | `corpus_*.csv` sheets |
| Demo overlay | `sam-overlay` | detect+assign+`visualize.py` | `sam_gaze_overlay.mp4` (H.264) |

Package entry: `python -m gaze_objects.cli <command> --config <yaml>`.

### 4.5 Current production configs

| Config | Role |
|--------|------|
| `configs/batch_sam3_full_fine.yaml` | **Full corpus** fine-label batch (detect → assign → sequences) |
| `configs/am07_sam3_mask_overlay.yaml` | **AM07 demo:** full-video mask detect + assign + illustration mp4 |
| `metaSAM3/prompts_fine_objects.txt` | Text concepts SAM3 is allowed to find |
| `configs/class_definitions.yaml` | Study class / mapping definitions |

**Fine prompts currently (kit-aligned, FP-cleaned):**

```text
angle grinder
electric grinder
grinder
disk grinder
power tool
instruction manual
storage box
screwdriver
wrench
```

Removed from prompts because they polluted sheets / are not reliably in the kit: `pliers`, `hex key`, `hammer`, `grinding disc`.

Synonyms (`power tool`, `disk grinder`, …) map to canonical `angle grinder` in `label_map.py`.

### 4.6 Thresholds (current) — easy language + numbers

Thresholds = **how picky** we are before believing a detection or a gaze hit.

| Knob | Value | Effect |
|------|------:|--------|
| Default detect score | **0.50** | Keep medium–high confidence objects |
| Angle grinder detect | **0.12** | Much looser — handheld / egocentric AG often scores low |
| Instruction manual detect | **0.55** | Stricter — reduce false “manual” |
| Screwdriver / storage box | **0.45** | Slightly loose |
| Wrench | **0.50** | Default |
| Min AG box area | **0.8% of frame** | Drop tiny corner false AG blobs when gate is low |
| Assign score gate | **0.12** | Gaze may attach only to dets above this score |
| Sequence quality | **≥2** assigned samples **and ≥50%** label agreement | Weak fixations excluded from sheets |

**Impact rule of thumb**
- Lower detect threshold → more objects found (and more false positives).  
- Higher → cleaner sheets, but miss hard cases (early handheld grinder).  
- Changing prompts/thresholds requires **re-detect** (then assign → sequences → aggregate). Changing only assign/sequence knobs can reuse detections.

### 4.7 Hardware / runtime constraints (colleagues care about “will it run?”)

- GPU: **RTX 3080 Ti 12 GB** — run **exactly one** heavy SAM3 job.
- SAM3 image resolution **must be 1008** (model RoPE / ViT size); other sizes crash.
- Segmentation head must stay **enabled** (`Sam3Processor` needs masks).
- Env: `conda activate sam3`, `PYTHONPATH=src:$PYTHONPATH`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Batch uses `skip_if_done: true` (safe resume) and unloads the model between recordings.

### 4.8 What the pipeline delivers (artifacts)

**Per recording** (under e.g. `outputs/batch_sam3_full_fine/<id>/`):

| File | Meaning |
|------|---------|
| `detections.csv` | Every SAM object instance on processed frames (label, score, box; masks in npz) |
| `gaze_assignments.csv` | Each gaze sample → selected object or status (assigned / ambiguous / miss…) |
| `fixation_attended_objects.csv` | One row per Tobii fixation with attended label + quality |
| `attention_sequences.csv` | Merged attention episodes + duration stats |
| `sequences_summary.json` | Counts for that recording |

**Corpus-level** (after aggregate):

| File | Meaning |
|------|---------|
| `corpus_sequence_strings.csv` | **Primary product:** one gaze-order string per recording |
| `corpus_attention_sequences.csv` | All episodes across corpus (+ duration columns) |
| `corpus_transitions.csv` | From→to transition counts (behavior graph) |
| `corpus_results.csv` | QC join + sequence string |
| `batch_qc_summary.csv` | Per-recording success / metrics |
| `corpus_sequences_summary.json` | Rollup stats |

**Illustration**

| File | Meaning |
|------|---------|
| `outputs/am07_sam3_mask_overlay/sam_gaze_overlay.mp4` | Full AM07 video with colored SAM masks, gaze crosshair, green HIT when gaze on object |

**Duration columns on sequences** (for psychology sheets):

- `total_gaze_event_duration_raw` — sum of fixation durations in the episode  
- `mean_fixation_duration_raw` / `median_fixation_duration_raw`  
- `fixation_durations_raw` — pipe-separated list (`100|200|150`)

---

## 5. Chronological journey (methods evolution)

Use this section for 2–4 “how we got here” slides if the audience wants the story.

### Phase 0 — Handoff & sync proof (9–11 Sep 2026)
- Scaffolded Python package `gaze_objects` from internship handoff.
- Stage A audit of AM07 TSV; Stage B gaze overlay on 30–60 s clip.
- **User verified overlay vs Tobii playback: sync perfect** → time mapping marked verified (milliseconds, video start 0).

AM07 audit snapshot: ~132k TSV rows; ~37.8k eye rows; ~37.1k in-frame gaze; 532 fixation indices; Tobii AOIs: Angle Grinder, Boxes, Manual, Tools.

### Phase 1 — GPU workstation (11–15 Sep)
- Miniconda under home; CUDA stack; IDEA-Research DINO ops compiled.
- Confirmed RTX 3080 Ti; code pushed to GitHub; Tobii corpus under `~/KHETRE/Tobii_Data`.

### Phase 2 — COCO IDEA-DINO pilot (16 Sep)
- Same AM07 30–60 s clip, 60 frames.
- **Conditional AOI agreement: 0%** — pretrained COCO names (`hair drier`, `refrigerator`, …) ≠ study AOIs.
- Scientific success: pipeline wiring works; detector vocabulary wrong → do not scale.

### Phase 3 — Grounding DINO (16–22 Sep)
- Promptable open-vocab detector.
- Conditional AOI agreement **~55%** (60/109), later phrase map `instruction`→Manual projected ~**99%** on that matrix.
- Team decision: move to **Meta SAM 3**.

### Phase 4 — SAM3 adoption + Stage E (22–25 Sep)
- `sam3_meta.py` adapter; bfloat16 autocast fix; prompt/threshold tuning.
- SAM3 dense AM07: **2972 detections**, conditional AOI acc **~99.7%** (318/319).
- Stage E: Tobii fixations → attended-object sequences.

### Phase 5 — 5-video batch (25–28 Sep)
- Batch runner + QC summary; recursive Tobii pair discovery.
- Hard case `AA05ED02_03`: AG→Tools confusion (**screwdriver** stealing AG) — fixed for AOI mode by removing screwdriver/wrench prompts and nested-parent logic; accuracy **~0.44 → ~0.91**.
- Nested Manual→AG preference tried then **reverted** (broke AM07 where Manual is real).

### Phase 6 — ~291-video AOI dense corpus (28–30 Sep)
- Discoverable paired recordings ≈ **291** (not all ~400 files — unpaired / fehlerhaft skipped).
- QC: up to **290/291 ok**; mean conditional AOI accuracy ~**0.94** (median 1.0); after ignoring Tools in scoring ~**0.96**.
- Still a 30–60 s / max-150-frame pilot style — not whole-task sequences yet.

### Phase 7 — Full-video sequences (30 Sep+)
- `clip.mode: full`, no max_frames; aggregate-sequences for corpus strings.
- User pivot: **Approach 2 — fine object names** for psychology, not Tobii AOI collapse.

### Phase 8 — Fine-label full-video corpus (2–7 Oct) — **main deliverable**
- Config `batch_sam3_full_fine.yaml`; output `outputs/batch_sam3_full_fine` / `output_SAM_labelled_290`.
- CUDA OOM saga on 12 GB → one GPU job; keep masks; **force resolution 1008**; stride 8; unload between IDs.
- Result: **290/290 ok**, aggregate **289** nonempty sequence strings, **18,028** episodes, **36** transition types.

### Phase 9–11 — Masks, overlay, AG recovery, sheet cleanup (9–12 Oct)
- Gaze-in-mask assign + full-recording H.264 overlay (PR #5).
- User screenshot t≈34.54 s: grinder in hands under gaze, **no mask**; CSV had **0 AG before ~77 s**.
- Fixes: AG synonyms + gate 0.12 + min area + magenta; remove pliers/hammer/grinding disc from prompts; add per-sequence fixation duration stats.
- Feasibility of instance/part labels discussed (class OK; instances need tracking; screws not reliable with SAM3 alone).

---

## 6. Detector evolution — numbers for a comparison slide

AM07, typically 30–60 s clip, 60 frames (unless noted “dense”):

| Detector | Detections | Cond. AOI agreement | Takeaway |
|----------|----------:|--------------------:|----------|
| IDEA COCO-DINO | 135 | **0%** (0/73) | Wrong vocabulary |
| Grounding DINO | 597 | **~55%** → ~99% after phrase map | Promptable, weaker than SAM3 path |
| SAM3 (tuned v2) | ~1198 | **~95–97%** | Adopted |
| SAM3 dense (150 frames) | **2972** | **~99.7%** (318/319) | Ready for sequences |

**Important:** Fine-label corpus does **not** report Tobii AOI `conditional_accuracy` — names differ on purpose (`screwdriver` ≠ `Tools`).

---

## 7. Scale results — what to show as internship outcome

### 7.1 Fine full-video corpus (Approach 2) — headline numbers

| Metric | Value |
|--------|------:|
| Recordings processed successfully | **290 / 290** |
| Nonempty gaze-order strings | **289** (1 empty: `ER05RG05_02`) |
| Attention episodes (sequence events) | **18,028** |
| Distinct transition types | **36** |
| Approx fixations → labelled → episodes | ~126k → ~26k → ~18k |
| Mean distinct objects per string | ~4.4 |

### 7.2 Episode counts by object (behavior relevance)

| Attended label | Episodes |
|----------------|--------:|
| screwdriver | 5,203 |
| instruction manual | 5,005 |
| storage box | 4,902 |
| angle grinder | 2,510 |
| wrench | 263 |
| grinding disc | 133 *(prompt later removed — treat as noisy)* |
| pliers | 12 *(prompt removed — false positive class)* |

### 7.3 Top transitions (workflow graph)

| From → To | Count |
|-----------|------:|
| screwdriver → storage box | 892 |
| storage box → screwdriver | 834 |
| storage box → instruction manual | 802 |
| instruction manual → screwdriver | 793 |
| screwdriver → instruction manual | 760 |
| instruction manual → storage box | 742 |
| storage box → angle grinder | 489 |
| angle grinder → screwdriver | 481 |
| angle grinder → storage box | 435 |
| screwdriver → angle grinder | 414 |

**Interpretation for colleagues:** participants’ labelled attention heavily cycles among **screwdriver ↔ storage box ↔ instruction manual**, with the **angle grinder** as a secondary but important hub — consistent with fetch-parts / read-manual / use-tool assembly behavior.

### 7.4 Example sequence string (slide-friendly)

```text
storage box > angle grinder > screwdriver > storage box > instruction manual > screwdriver > …
```

### 7.5 AOI-mode corpus (earlier QC track)

- ~291 discoverable TSV+video pairs; up to **290 ok**.
- Mean conditional AOI accuracy ~**0.94** (median **1.0**) on dense trimmed pilots.
- Useful as **detector QC against Tobii AOIs**, not as the primary psychology product.

### 7.6 Overlay / AG diagnostics (honesty slide)

On one AM07 overlay run before AG recovery:

- 8,142 detections / 1,844 frames  
- Angle grinder: only **96** dets on **51** frames (~2.8%), mean score **0.35**, first hit ~**77 s**  
- At **t = 34.54 s** user screenshot: grinder clearly held under gaze — **no segmentation**  
- Prompt/threshold recovery pushed; **re-run overlay** required to refresh the video  
- Sheet FPs (pliers, grinding disc) cleaned in prompts; **full corpus re-detect** needed if sheets must be fully clean

---

## 8. How to run (demo / reproducibility)

### 8.1 Environment

```bash
conda activate sam3
cd ~/KHETRE/DINO_KHETRE/DINO
git checkout cursor/sam3-mask-assign-overlay-e54c   # or latest merged branch
git pull
export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Ensure no other Python process holds the GPU (nvidia-smi)
```

### 8.2 One-video illustration (best live demo)

```bash
rm -rf outputs/am07_sam3_mask_overlay
python -m gaze_objects.cli sam-overlay --config configs/am07_sam3_mask_overlay.yaml
# → outputs/am07_sam3_mask_overlay/sam_gaze_overlay.mp4
```

Shows: colored SAM masks, large gaze crosshair, **green HIT** when gaze is on a labelled object, timestamps over the **full** recording (H.264 playable).

### 8.3 Step-by-step for one config

```bash
python -m gaze_objects.cli detect    --config configs/am07_sam3_mask_overlay.yaml
python -m gaze_objects.cli assign    --config configs/am07_sam3_mask_overlay.yaml
python -m gaze_objects.cli sequences --config configs/am07_sam3_mask_overlay.yaml
```

### 8.4 Full corpus batch + sheets

```bash
python -m gaze_objects.cli batch --config configs/batch_sam3_full_fine.yaml
python -m gaze_objects.cli aggregate-sequences --output-root outputs/batch_sam3_full_fine
```

### 8.5 What impacts what (ops cheat-sheet)

| Change | Must re-run |
|--------|-------------|
| Prompts or detect thresholds | detect → assign → sequences → aggregate (+ overlay) |
| Assign threshold / `hit_test` | assign → sequences → aggregate |
| Sequence quality gates / duration columns | sequences → aggregate |
| Overlay colors only | sam-overlay (or redraw) |

---

## 9. Suggested presentation outline (for the AI slide agent)

Aim ~12–15 slides for a 15–20 minute talk.

1. **Title** — Gaze-to-object pipeline for angle-grinder assembly (Tobii + SAM3)  
2. **Motivation** — Need automated “what did they look at, in what order?”  
3. **Inputs** — Tobii TSV + scene video; what Tobii already gives  
4. **Pipeline diagram** — Section 4.1  
5. **Division of labor** — Tobii vs SAM3 vs sequences  
6. **Why not COCO-DINO?** — 0% AOI agreement (learning slide)  
7. **Why SAM3 + masks** — open vocab + silhouette precision  
8. **Scale outcome** — 290 videos, 18k episodes, example string  
9. **Transition graph** — top screwdriver/box/manual/AG flows  
10. **Demo** — overlay video / screenshot (masks + gaze HIT)  
11. **Engineering constraints** — 12 GB GPU, res 1008, one job  
12. **Quality & limits** — AG hard when handheld; FP prompt cleanup; no screw-level labels yet  
13. **Deliverables & repo** — corpus CSVs, configs, PRs  
14. **Next steps** — clean re-batch; optional instance tracking; part labels only if needed  
15. **Q&A**

**Speaker notes cues**
- Emphasize sync was **human-verified** before claiming detector accuracy.  
- Emphasize fine labels are for **behavior**, AOI accuracy is for **QC**.  
- Be honest about early AG misses and tray FPs — show the fix path.  
- Offer the overlay as the visual proof; corpus CSVs as the analysis product.

---

## 10. GitHub trail (evidence of work)

Draft PRs under https://github.com/abhishekkhetre/DINO :

| PR | Branch | Topic |
|----|--------|-------|
| #1 | `cursor/sam3-dense-stage-e-quality-e54c` | Dense SAM3 + Stage E quality gates |
| #2 | `cursor/batch-5-sam3-dense-e54c` | Multi-video batch runner + QC |
| #3 | `cursor/full-video-sequences-e54c` | Full-video AOI sequences + aggregate |
| #4 | `cursor/sam3-fine-labels-full-e54c` | Fine labels full corpus (290) |
| #5 | `cursor/sam3-mask-assign-overlay-e54c` | Gaze-in-mask + overlay + AG/FP fixes + duration stats |

Cloud agent workspace for this work: https://cursor.com/agents/bc-0d91caf7-3f2a-40b1-b5ce-a4e17122e54c  

---

## 11. Future work discussed (do not over-promise)

| Idea | Feasibility with current stack |
|------|--------------------------------|
| Class-level masks + sequences | **Done / current** |
| Stable instance IDs (`storage box #3`) | Feasible with **tracking** after masks — not from text prompts alone |
| Coarse AG parts (body / side handle / disc) | Partial; noisy under hands; needs strict gates |
| Fine parts (`screw 1`, `top-left screw`) | **Not reliable** with open-vocab SAM3 alone; needs annotation or a part model |
| Full 290 re-detect with cleaned prompts | Recommended before final psychology export |
| SAM video tracking (SAM 3.1 style) | Optional speed/consistency upgrade; not required for current sheets |

---

## 12. Glossary

| Term | Meaning |
|------|---------|
| Tobii / Pro Lab | Eye-tracking glasses + software; exports gaze TSV and scene video |
| AOI | Area of Interest — coarse study labels (Angle Grinder, Manual, Boxes, Tools) |
| Fixation | Period of relatively stable gaze (Tobii I-VT filter) |
| SAM3 | Meta Segment Anything Model 3 — text-prompted detection + segmentation |
| Attended object | Object whose mask contains the gaze during a fixation (after gates) |
| Sequence string | Ordered list of attended objects across a recording |
| Conditional AOI accuracy | Among gaze rows with both a Tobii AOI and a pipeline assignment, fraction that match |
| Egocentric | First-person wearable camera view |
| Hit-test mask | Gaze counts only if the pixel lies inside the object silhouette |

---

## 13. Suggested visual assets to attach in the deck

1. Pipeline flowchart (Section 4.1).  
2. Overlay frame with masks + red gaze + green HIT (AM07).  
3. Optional “failure then fix” pair: t≈34 s AG missing vs post-rerun (if available).  
4. Table of top transitions (Section 7.3).  
5. One example `sequence_string` from `corpus_sequence_strings.csv`.  
6. Hardware slide: RTX 3080 Ti, res 1008, one-job rule.  
7. Repo / output folder screenshot of `output_SAM_labelled_290` or `batch_sam3_full_fine`.

---

## 14. Closing statement (usable as last slide text)

This internship delivered a **working, documented, GPU-runnable research pipeline** that turns Tobii assembly recordings into **object-level attention sequences at corpus scale**. We moved from a zero-accuracy COCO baseline to **SAM3 mask-based assignment**, processed **290 full videos**, and produced **gaze-order and transition sheets** suitable for studying how people visually navigate angle-grinder assembly — with clear limits, reproducible configs, and a path for cleaner re-exports and richer labels later.

---

## Appendix A — File tree (current tip branch)

```text
configs/
  batch_sam3_full_fine.yaml          # corpus production
  am07_sam3_mask_overlay.yaml        # demo overlay
  class_definitions.yaml
  am07_stage_c*.yaml                 # historical pilots
metaSAM3/
  prompts_fine_objects.txt           # current fine concepts
  prompts_assembly_aoi.txt           # AOI-mode concepts
  prompts_pilot*.txt
src/gaze_objects/
  cli.py  audit.py  sync.py  video.py
  stage_c.py  assign.py  masks.py  evaluate.py
  sequences.py  aggregate.py  batch.py  visualize.py
  detectors/sam3_meta.py  label_map.py  grounding_dino.py  dino_idea.py  mock.py
tests/                               # unit tests for assign, sequences, SAM gates, …
docs/
  INTERNSHIP_PRESENTATION_BRIEF.md   # this file
```

## Appendix B — Timeline (compact)

| Date (2026) | Milestone |
|-------------|-----------|
| 9–11 Sep | Handoff, Stage A/B, sync verified |
| 11–15 Sep | Workstation GPU / DINO env |
| 16 Sep | COCO-DINO 0% AOI; start Grounding |
| 16–22 Sep | Grounding ~55–99%; decide SAM3 |
| 22–25 Sep | SAM3 pilot + Stage E; dense ~99.7% |
| 25–28 Sep | 5-video batch; AA05 Tools fix |
| 28–30 Sep | ~291 AOI corpus QC |
| 30 Sep–2 Oct | Full-video path; fine-label approach |
| 2–7 Oct | Fine corpus 290/290; 18k episodes |
| 9–12 Oct | Masks + overlay; AG/FP fixes; duration columns; presentation brief |

---

*End of briefing. Generated for presentation use from project transcript (Sep–Oct 2026), repository history, PRs #1–#5, and workstation corpus results.*
