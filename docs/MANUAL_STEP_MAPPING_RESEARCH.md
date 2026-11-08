# Gaze results mapped to the instruction manual — analysis and insights

**Sources:**
- The manual: `Anleitung Demontage 16.03.2025`, 4 pages, in `docs/manual/`.
- The October 2026 corpus results: 36 participants (17 experts, 19 non-experts), 360 recordings. These are the aggregate tables of the *Manual-aligned gaze research* report, in `docs/manual_step_results/source/`.
- The recording rule: odd recording = disassembly, even = assembly.

All numbers below are real results from these tables. Shares are of *labelled* fixation time, averaged over participants. Derived tables are in `docs/manual_step_results/derived/`.

---

## 0. Findings in brief

1. **The manual is a fixed 8-step script, and assembly is disassembly in reverse.** Gaze follows that structure. The manual's five phases can be recognised from today's labels: handle, screw block, wrench, armature, sort. The exact step within the screw block cannot yet be recognised.
2. **Manual use is a learning curve, and the learning happens inside each disassembly → assembly pair.** Non-experts cut manual use by 30% from a disassembly to its assembly, but only by 6% from that assembly to the next disassembly. Reassembling right after taking the device apart is done largely from memory.
3. **Exposure, not direction, sets the level.** Non-expert manual share is 52 → 37 → 37 → 22 → 21 → 16 → 14 → 10% over recordings 1–8. It halves about every 3 recordings. Only at recording 8 do non-experts read as little as experts did on recording 1.
4. **Recordings 9–10 break the curve.** Manual share falls to ~0.5% in both groups, a 20-fold drop for non-experts. This is a change of condition or of data, not learning (Q1).
5. **Each direction has its own gaze signature that matches the manual.**
   - Assembly: +9–10 pp grinder, because A1–A3 are hands-only work inside the housing.
   - Disassembly: +7–9 pp screwdriver, because D2–D5 are four unscrew steps in a row.
6. **Gaze routes reverse with the task, as the manual does.**
   - Disassembly: *tool → grinder → box* (unscrew, remove, deposit) leads 1.33 : 1.
   - Assembly: *box → grinder → tool* (fetch, place, fasten) leads 1.54 : 1.
   - The task can therefore be read from routes alone.
7. **"Deposit, then read" marks a step boundary.** *grinder → box → manual* occurs in 42% of disassembly recordings and in 30 of 36 participants. This is the manual's rhythm of "part into Box n, then the next Schritt".
8. **Experts do, non-experts read.** In disassembly, experts look 19.5 pp less at the manual and about 10 pp more at boxes and at the screwdriver. Experts keep a stable box protocol (24% box share in both directions). Non-experts pay for skipping it with box search in assembly (+5.4 pp).
9. **Assembly has a slow front end.** The first wrench gaze comes later at A3 (the 3rd assembly step, median 180 s) than at D6 (the 6th disassembly step, median 132 s). Inserting the armature and seating the gear (A1–A2) costs more time than the first five disassembly steps.
10. **Tool steps are watched through the target, not the tool.** Wrench gaze is 0.3–0.7% of labelled time and appears in only a third of recordings, although every recording contains the wrench step.

---

## 1. The manual as a process

The PDF has pictures only. The step readings below come from the pictures; the full catalogue is `docs/manual_step_results/manual_step_catalog.csv`.

| Disassembly (p. 1–2) | Tool | Box | ↔ | Assembly (p. 3–4) | Tool | Box |
|---|---|---|---|---|---|---|
| D1 side handle off | hand | Box 1 red | ↔ A8 | side handle on | hand | Box 1 |
| D2 lever bracket screws | green screwdriver | Box 2 red | ↔ A7 | bracket + spring + screw | – | Box 2 |
| D3 gear-head flange + bevel gear | green screwdriver | Box 3 red | ↔ A6 | refit flange | green screwdriver | Box 3 |
| D4 housing-ring screws | red screwdriver | (Box 4) | ↔ A4 | fit ring, tighten | red screwdriver | Box 4 |
| D5 rear side screw | – | – | ↔ A5 | rear side screw | – | – |
| D6 wrench inside housing | wrench | – | ↔ A3 | wrench inside housing | wrench | – |
| D7 pull armature + white gear | hands | – | ↔ A1 | insert armature | hands | Box 5 blue |
| D8 sort to Box 4 (red) / Box 5 (blue) | hands | Box 4, 5 | | A2 turn gear to seat | hands | – |

Points that matter for interpreting gaze:

- **Phases.** Each direction groups into handle → screw block → wrench → armature → sort (disassembly), or the reverse with an insert phase first (assembly). The screw block is D2–D5 and A4–A7.
- **Colour separates steps that look alike.** Within the screw block, the screwdriver handle colour (green for D2/D3/A6, red for D4/A4) and the box number distinguish steps. The current labels have only "screwdriver" and "storage box", so these steps merge.
- **The kit shows no guard, battery or grinding disc.** Gaze on those labels, where it exists in other prompt sets, is probably detection noise (Q2).

---

## 2. Insights

### 2.1 Manual use: a learning curve that is learned in pairs

| Recording | Task | Expert | Non-expert | Gap (pp) |
|---:|---|---:|---:|---:|
| 1 | Dis | 11.5% | **52.0%** | 40.5 |
| 2 | Asm | 9.2% | 36.8% | 27.6 |
| 3 | Dis | 4.8% | 37.2% | 32.4 |
| 4 | Asm | 2.9% | 22.0% | 19.1 |
| 5 | Dis | 7.5% | 20.6% | 13.1 |
| 6 | Asm | 3.9% | 16.4% | 12.5 |
| 7 | Dis | 2.4% | 14.4% | 12.0 |
| 8 | Asm | 3.3% | **10.0%** | 6.7 |
| 9 | Dis | 0.7% | 0.5% | −0.2 |
| 10 | Asm | 0.5% | 0.3% | −0.2 |

**Shape.**
- Recordings 1–8 follow an exponential decay. Manual use falls ~20% per recording for non-experts and ~17% for experts, a half-life of about 3 recordings in both groups.
- The rank correlation with recording order is −0.98 for non-experts and −0.76 for experts.
- The gap between the groups shrinks from 40 pp to 7 pp.

**Where the drop happens.** For non-experts, the change from one recording to the next is:

| Transition | Kind | Change |
|---|---|---:|
| D1 → A1 (rec 1 → 2) | within pair | −29% |
| A1 → D2 (rec 2 → 3) | between pairs | +1% |
| D2 → A2 (rec 3 → 4) | within pair | −41% |
| A2 → D3 (rec 4 → 5) | between pairs | −6% |
| D3 → A3 (rec 5 → 6) | within pair | −20% |
| A3 → D4 (rec 6 → 7) | between pairs | −12% |
| D4 → A4 (rec 7 → 8) | within pair | −31% |

The within-pair mean is **−30%** and the between-pair mean is **−6%**.

**Interpretation.**
- The assembly pages are the disassembly pages read backwards. Right after taking the grinder apart, participants rebuild it from fresh episodic memory: the last part out is the first part in, and it sits in Box 5.
- That memory does not fully carry over to the next disassembly, which again needs the manual nearly as much as the previous assembly did. The step order learned in the moment is short-lived. What persists from one repetition to the next is a slower, general familiarity.
- Experts show the same within-pair drop at a much lower level, with a median of −30%. Their between-pair changes are noisy. The clearest example is a rise at recording 5 (4.8 → 7.5%), the third disassembly. That is a candidate for a step that experts deliberately re-check.

**Exposure, not direction.** Non-expert assembly 1 (36.8%) is equal to disassembly 2 (37.2%). Both are the second contact with the device. Disassembly is not intrinsically more manual-heavy; it simply comes first in each pair.

**Catch-up.** Non-experts first come within 25% of the experts' first-recording level (11.5%) at recording 8. That is about seven practice runs to reach expert first-contact behaviour.

**Recordings 9–10.** No learning curve predicts a 20-fold drop between recordings 8 and 9, and both groups land at the same ~0.5%. Possible explanations:
- The manual was removed or hidden in the 5th repetition.
- The camera framing changed so the manual was out of view.
- These recordings were processed differently.

If the manual was removed, recordings 9–10 are a valuable **no-guidance test**: they show what each group does from memory alone. Until this is confirmed they should not be pooled with 1–8 (Q1).

### 2.2 Direction signatures (assembly − disassembly)

| Object | Experts | Non-experts | Manual explanation |
|---|---:|---:|---|
| angle grinder | **+10.3 pp** | **+9.0 pp** | A1–A3 are hands-only work inside the housing: insert the armature, seat the gear, wrench. No screwdriver is involved. |
| screwdriver | **−9.1 pp** | **−6.7 pp** | Disassembly starts with four consecutive unscrew steps (D2–D5). In assembly, screws come only at the end (A4–A7). |
| storage box | +0.3 pp | **+5.4 pp** | Assembly takes parts *out of* Boxes 5 → 1. Those who did not sort deliberately have to search. |
| manual | −1.4 pp | −8.0 pp | The within-pair learning described in 2.1. |

Together, the grinder and screwdriver rows say that **disassembly is tool-led and assembly is workpiece-led**. Seen in gaze terms: removing a part needs the tool on the fastener, while fitting a part needs the eyes on the housing until it seats.

### 2.3 Expertise: experts execute, non-experts consult

| Object | Expert − non-expert, disassembly | Expert − non-expert, assembly |
|---|---:|---:|
| manual | **−19.5 pp** | −12.9 pp |
| storage box | **+9.7 pp** | +4.6 pp |
| screwdriver | **+9.5 pp** | +7.1 pp |
| angle grinder | +0.3 pp | +1.6 pp |

- **The gap is not about the workpiece.** Both groups give the grinder the same share. The difference is entirely *manual versus box and tool*.
- **Attention budget.** Execution (grinder + tools) takes ~70% of labelled gaze for experts in both directions and 60–63% for non-experts.
- **Experts follow the box protocol.** Their box share is 24.3% in disassembly and 24.6% in assembly. They look where the part goes as part of the routine. Non-experts look at boxes less in disassembly (14.6%) and more in assembly (20.0%): they sort less carefully and then search.
- **The gap is largest in disassembly**, the first contact in each pair, which is where expertise matters most.

### 2.4 Gaze routes follow the manual's action grammar

Role letters: W = grinder (workpiece), S = storage box, T = tool, C = manual. There are 180 recordings per direction.

| Direction | Route | Per recording | Recordings | Reading | Manual steps |
|---|---|---:|---:|---|---|
| Dis | W → S → W | 1.32 | 57% | deposit part, back to grinder | D1–D3 deposits, D8 |
| Dis | T → W → S | 0.89 | 51% | unscrew → remove → deposit | D2, D3 |
| Dis | S → W → T | 0.67 | 40% | deposit done → next screw | D2 → D3, D3 → D4 |
| Dis | **W → S → C** | 0.81 | 42% | **deposit → read next step** | boundary after D1 / D2 / D3 |
| Asm | **S → W → S** | **2.68** | **81%** | fetch → fit → fetch | A1 (Box 5), A4, A6–A8 |
| Asm | S → W → T | 1.19 | 61% | fetch → place → fasten | A4, A6, A7 |
| Asm | T → W → S | 0.78 | 48% | fasten → next part | A4 → A5, A6 → A7, A7 → A8 |

- **Mirror structure.** The two screw routes swap dominance between directions:
  - Disassembly: T → W → S : S → W → T = 160 : 120.
  - Assembly: the same ratio is 140 : 215.
  - The manual's reversal ("remove, then store" against "fetch, then fasten") is visible directly in eye movements.
- **Assembly is driven by retrieval.** The fetch–fit–fetch loop is 2.3× more frequent in assembly (2.68 vs 1.17 per recording) and occurs in 81% of assembly recordings.
- **Step boundaries in disassembly.** *Deposit → read* (W → S → C) is the only top route that includes the manual, and it is specific to disassembly. Its position fits the end of the box steps D1–D3: the part is stored, then the next Schritt is looked up. It is the best boundary marker the current labels offer.

### 2.5 A general model of a manual step in gaze

Every manual step that involves a box resolves into the same four-part cycle:

| Element | Gaze | Where it dominates |
|---|---|---|
| **Consult** | manual | step onset (planning) or mid-step (uncertainty) |
| **Fetch** | box → grinder | assembly: parts from Box 5 → 1 |
| **Operate** | tool ↔ grinder (tip-focused) | screw block, wrench |
| **Deposit / check** | grinder → box, often followed by Consult | disassembly: part into Box n |

What changes the cycle:
- **Expertise** removes most of *Consult*. Experts go straight to *Fetch* or *Operate*.
- **Learning** shrinks *Consult* from one recording to the next, fastest inside each pair.
- **Direction** reverses the cycle: Operate → Deposit in disassembly, Fetch → Operate in assembly.
- **Tool use is monitored at the target.** The wrench takes 0.3–0.7% of labelled gaze and the screwdriver 8–25%, mostly the tip. Once a tool is in hand, people watch where it acts, not the tool itself.

### 2.6 Timing: anchoring steps in time

The wrench step is the only step with its own label (D6 and A3).

| | Disassembly (D6, step 6 of 8) | Assembly (A3, step 3 of 8) |
|---|---:|---:|
| recordings with wrench gaze | 61 / 180 (34%) | 54 / 180 (30%) |
| median first wrench gaze (report excerpt) | **132 s** (n = 8) | **180 s** (n = 8) |

- **Assembly is slow at the start.** Reaching the 3rd assembly step takes longer than reaching the 6th disassembly step. Inserting the armature with its white gear and turning it until it seats (A1–A2) is the most demanding manipulation in the task. This matches the +10 pp grinder share in assembly.
- **Practice and expertise effects are large in timing.** For example, AM07's first assembly reaches the wrench at 608 s, while the same participant's third assembly reaches it at 167 s. Some recordings show wrench gaze very early (23–33 s), which points to looking at the wrench lying in place rather than using it.
- **Caveats.** These are 16 recordings from 5 participants, so the timing is an indication only. For two-thirds of recordings there is no wrench gaze at all, so the wrench cannot anchor most timelines.

---

## 3. Step by step: what the gaze shows today

Recognisability is simulated from each step's profile in the manual (5-label gaze windows, chance 12.5%). It shows what the labels *can* separate; it is not measured accuracy.

### Disassembly

| Step | Expected gaze signature | Corpus evidence | Phase recognised | Exact step |
|---|---|---|---:|---:|
| D1 side handle | grinder + box (Box 1), hand | W → S → W deposit loop (57% of recordings) | 53% | 53% |
| D2 lever bracket | green screwdriver tip, box (Box 2) | T → W → S unscrew–store (51%); screwdriver +9.5 pp for experts | 92% | 54% |
| D3 gear-head flange | green screwdriver, box (Box 3) | same as D2; S → W → T continuation | 88% | **0%** (merges with D2) |
| D4 housing ring | red screwdriver, hand holding ring | screwdriver-heavy disassembly front | 91% | 61% |
| D5 rear screw | screwdriver at rear | — | 77% | 20% |
| D6 wrench | wrench, grinder interior | wrench gaze in 34% of recordings, median 132 s | 90% | 90% |
| D7 pull armature | grinder interior, hands | unlabelled hand work | 48% | 48% |
| D8 sort | boxes (red Box 4, blue Box 5) | W → S → W, W → S → C at the end | 73% | 73% |

### Assembly

| Step | Expected gaze signature | Corpus evidence | Phase recognised | Exact step |
|---|---|---|---:|---:|
| A1 insert armature | blue Box 5, grinder interior | S → W → S fetch loop (81%); grinder +10 pp | 90% | 63% |
| A2 seat gear | grinder interior, hands | slow front end (wrench at 180 s) | 89% | 66% |
| A3 wrench | wrench | wrench gaze in 30% of recordings | 92% | 92% |
| A4 housing ring | red screwdriver, red Box 4 | S → W → T fetch–fasten (61%) | 89% | 44% |
| A5 rear screw | screwdriver at rear | — | 77% | 38% |
| A6 flange | green screwdriver, Box 3 | as A4 | 86% | **1%** |
| A7 lever bracket | screw, hand, Box 2 | T → W → S fasten–next (48%) | 82% | 24% |
| A8 side handle | Box 1, grinder | — | **0%** | **0%** |

**Weak spots and what would fix them.**

| Weakness | Cause | Fix |
|---|---|---|
| D2 ↔ D3, A6 ↔ A7 | same tool, same box colour | box number, part identity (bracket vs flange), or which manual panel was just read |
| D1 / A8 | without a side-handle label, they look like generic grinder + box work | a side-handle label (D1 rises to 86% and A8 to 86% in simulation) |
| D7, A1, A2 | hand work | hand and armature labels (D7 rises to 78%) |

With step-aware labels (side handle, armature, hand, green vs red screwdriver, red vs blue box), 10-label windows reach in simulation:

| | Exact step | Phase |
|---|---:|---:|
| Today's labels | 48–60% | 82–86% |
| Step-aware labels | 73–77% | 95–96% |

---

## 4. Predicting the manual step from a gaze sequence

Today's labels support rules at the **phase** level. Given a short window of labelled gaze:

| If the window contains… | Most likely |
|---|---|
| any wrench | D6 / A3 (wrench) |
| screwdriver → grinder → box | disassembly screw block (D2–D5) |
| box → grinder → screwdriver | assembly screw block (A4–A7) |
| box → grinder → box, no tool, early in assembly | A1 (armature from Box 5) |
| grinder → box → manual | a disassembly step boundary (just after D1 / D2 / D3) |
| long grinder-only stretch, early in assembly | A1–A2 (insert and seat) |
| grinder + box at the very start or end, no tool | D1 / A8 (handle) or D8 (sort) |
| mostly manual | non-expert, early recording, step onset |

Two worked examples:

- The five-label route that the earlier HTML report proposed as "step two" (*screwdriver tip → grinder → screwdriver handle → box → grinder*) maps to the **disassembly screw block with 99.8% probability**. Within the block it is split between D3 (39%) and D2 (37%). The phase is certain but the step is not.
- A window of *red screwdriver → grinder → red box → red screwdriver* in assembly maps to **A4** (housing ring, Box 4) with **91% probability**. This shows that colour alone is enough to resolve the step.

Order adds information that a single window lacks. Knowing the previous phase narrows the next step: for example, after the wrench in disassembly, the next step can only be D7 or D8. Following a recording through the manual's order should therefore do better than the per-window figures above.

---

## 5. What can and cannot be claimed

**Supported now:**
- Gaze encodes the manual's structure: phases, direction reversal, and box logic.
- Manual dependence follows a learning curve that drops mainly within each disassembly → assembly pair.
- Experts differ from non-experts mainly in *manual versus box and tool* attention, most of all in disassembly.
- *Deposit → read* marks disassembly step boundaries.

**Not yet supported:**
- Exact per-step start and end times for each recording. Only wrench gaze anchors a step, and only in a third of recordings.
- Separating D2 from D3, A6 from A7, and A8 from A1. Report these as phases.
- Any claim about manual use that pools recordings 9–10 with 1–8.
- Treating `angle grinder.side_handle_zone` as the side handle. It is a geometric region of the grinder mask, not the removable side handle.

**General caveats:**
- About 20% of fixations carry a label, so hand-only steps are under-observed.
- The numbers here are participant-averaged descriptives. Formal tests need the participant-level tables.

---

## 6. Open questions

- **Q1.** Recordings 09/10 show ~0.5% manual use in both groups. Was the manual removed or out of view in the 5th repetition?
- **Q2.** Does the kit match the manual: green and red screwdriver, flat wrench, red Boxes 1–4, blue Box 5, and no guard, battery or disc?
- **Q3.** Is the manual a printed sheet lying in view during the task? This decides whether gaze on the manual can be resolved to the Schritt panel being read.
- **Q4.** Are there experimenter logs or completion times for any recordings? They would turn the step estimates into validated timestamps.
- **Q5.** NA05AN05 numbers its recordings 00–09. Is 00 the first disassembly? It is excluded until this is answered.
- **Q6.** Where is the expert / non-expert assignment kept? The tables above use the report's grouping.

---

## Appendix: material on this branch

No action is needed. The batch that is already queued on bwUniCluster produces the episode-level sheets that this material can use later.

- `configs/manual_steps.yaml` holds the manual as data: steps, tools, boxes, reverse pairs, expected gaze profiles and page panels.
- `src/gaze_objects/manual_steps.py` aligns a recording's labelled episodes to the eight steps in manual order. It produces a per-step timeline with uncertainty, per-step metrics, manual lookups and a past-only "current step" estimate.
- `src/gaze_objects/manual_panels.py` maps gaze on a manual page to the Schritt panel being read.
- `metaSAM3/prompts_manual_steps.txt` is a step-aware label set: side handle, armature, white gear, green and red screwdriver, red and blue box, hand.
- `docs/manual_step_results/` contains the source and derived tables behind every number in this document.
