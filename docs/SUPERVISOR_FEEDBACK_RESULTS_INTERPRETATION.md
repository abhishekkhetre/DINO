# Interpreting the Final Gaze Results — Answers to Supervisor Feedback

**Prepared for:** supervisor feedback on *Final Gaze Visual Report* (16 Oct 2026 snapshot)  
**Author context:** internship pipeline (Tobii → SAM3 mask assign → fixation sequences → corpus sheets)  
**Source analysed:** `Final_Gaze_Visual_Report.html` + underlying corpus CSVs referenced therein  

This note answers the feedback questions directly: what the results **mean for us**, what we **can and cannot** claim, which findings are **worth keeping**, and how they **connect** into a coherent research story — rather than treating the report as a catalogue of charts.

---

## 1. What this work actually delivered (one paragraph)

We now have, for ~370 processed recordings (17 experts, 20 non-experts), an automated map from **Tobii fixations → attended kit objects** (angle grinder, screwdriver, storage box, manual, …) and, where available, **finer labels** (e.g. screwdriver tip; grinder body / side-handle-zone). From that map we can measure **who looked at what for how long**, **in what order**, and **whether experts and non-experts differ** — without manually drawing AOIs on hundreds of videos. That is the operational breakthrough. The scientific value is not “we detected objects,” but that we can finally ask **behavioural and workflow questions** at corpus scale with a transparent, reproducible label stream.

**Coverage caveat (must stay visible in every claim):** only **~19.7%** of fixation records received an object label (~31.7k labelled / ~160k fixations). Everything below is about the **labelled subset** unless stated otherwise. Unlabelled gaze is unknown, not “looking at nothing.”

---

## 2. Direct answers to the feedback questions

### 2.1 What insight do they offer? What can we do now?

| Insight | What we can do now |
|--------|---------------------|
| Attention is **object-structured**, not only pixel-structured | Analyse **time budgets** and **gaze order** over kit objects instead of only raw XY heatmaps |
| Experts and non-experts show a **stable difference in manual use** | Use **manual-looking share** as a candidate expertise marker (see §2.2) |
| The grinder dominates labelled attention (~50% equal-participant; ~58% if pooling time) | Treat the grinder as the **task anchor**; study other objects as *relative to* grinder time |
| Recurring routes (grinder↔box, grinder↔screwdriver) appear across almost everyone | Use motifs as **process signatures** for comparing sessions / groups (see §2.4) |
| Fine labels add *within-tool* structure (e.g. tip vs handle) | Separate **tool switches** from **within-tool part switches** in analysis |
| Coverage and unlabelled gaps limit path claims | Design the next study with **stage annotations** + denser labelling (hand/table/parts batch) |

**Concrete “now” uses (without new data collection):**

1. **Screen / stratify participants** by manual share and screwdriver share (exploratory biomarkers).  
2. **Describe workflow boundaries** (storage box often first/last labelled object).  
3. **Prioritise human validation** where it matters most (manual vs AG confusion; part/zone labels).  
4. **Feed psychology / CRC tables** with equal-participant object budgets and transition matrices instead of anecdotal AM07 clips.  
5. **Plan interventions** (e.g. whether reduced manual dependence tracks with training) as *hypotheses* for the next experiment — not as proven effects from this snapshot alone.

---

### 2.2 Can we differentiate experts and non-experts based on some patterns?

**Yes — partially, and with a clear ranking of strength.**

#### Strongest current separator (keep this)

**Manual-looking share (labelled time):**  
**experts ~4.7% vs non-experts ~20.6%** (−15.9 pp; exploratory adjusted *q* ≈ 0.00025).  
Bootstrap interval roughly −20.6 to −10.9 pp. The direction survives leave-one-participant-out and coverage filters (≥10/20/30% labelled).

**Interpretation for us:** non-experts spend a much larger fraction of *labelled* attention on the instruction manual. A natural behavioural reading is **greater reliance on written guidance**; alternatives (manual visibility, labelling bias, task variant) are not ruled out. This is an **association with expertise group**, not a proof of better performance — we have **no completion time or error rates** in these files.

#### Secondary separator

**Screwdriver share:** experts **20.0%** vs non-experts **12.7%** (+7.3 pp; *q* ≈ 0.0416). Compatible with more tool-use attention among experts, but weaker than the manual gap.

#### What does *not* separate groups cleanly

- **Angle grinder share** is similar (~50.6% experts vs ~49.1% non-experts) — both groups look at the device a lot; expertise is not “who looks at the grinder.”  
- Broad claims that experts “always look longer” or “switch faster” **do not survive** participant-level multiple-comparison correction across the duration / return / switching family.  
- **Grinder–box–grinder** route rate is higher in experts (12.4 vs 8.1 per 100 eligible triples; *q* ≈ 0.0129), but **return-to-first-object** habit is similar (~52% vs ~51%). Sequence differences exist; they are not a full cognitive story by themselves.

#### Practical answer for the supervisor

> We can already **distinguish groups on average** using a small feature set — especially **manual share**, secondarily **screwdriver share** and some **grinder↔box** motif rates — with participants as the unit. We cannot yet ship a high-accuracy individual classifier, and we must not equate “looks less at the manual” with “is better at the task” until performance labels exist.

---

### 2.3 Can we recognise which step of a disassembly process a person is in right now?

**Not yet as true process steps — only as coarse temporal structure of labelled attention.**

Why:

1. The report’s “start / middle / end” are **equal thirds of the labelled span** (first→last labelled episode), **not** annotated assembly/disassembly stages.  
2. Only **92 recordings** have explicit timing bounds in the log (12 people, **2 experts**) — not representative for stage models.  
3. Recording suffixes (`_01`…`_10`) are **not confirmed** chronological task stages.  
4. ~**84%** of neighbouring detailed episodes have **unlabelled fixation indices between them**, so “now” is a **partial trace**.

What we *can* say today (descriptive, useful for design):

- **Screwdriver attention rises mid-span** (14.2% → 18.8% → 13.2%; middle−start +4.7 pp, *q* ≈ 0.0095) — consistent with a mid-task **fastening / tool-use phase** hypothesis, not proof of a named step.  
- **Storage box** is often the **first** (52.6%) and **last** (45.0%) recognised object — consistent with **kit open / kit close** boundaries of the *observed label stream*.  
- Motifs like **grinder → box → grinder** suggest **workpiece ↔ parts storage** cycling rather than a single linear script.

**To recognise steps “right now” we still need:** human or logged **task-stage annotations** (or synchronized procedure checklist), then supervised mapping from (object sequence + timing) → stage. The current pipeline supplies the **features**; it does not yet supply the **stage ground truth**.

---

### 2.4 Can we use prevalent patterns to infer anything?

**Yes — limited but meaningful inferences, if we stay honest about gaps.**

| Pattern | Careful inference | Over-claim to avoid |
|--------|-------------------|---------------------|
| Grinder↔box loops are nearly universal (36/37 people; hundreds of recordings) | Kit interaction is **central** to the observed workflow, not incidental | “Everyone follows the same procedure” |
| Experts show more grinder–box–grinder triples | Experts may **cycle workpiece↔box** more densely in labelled data | “Experts are more efficient” |
| Manual routes depend on how often the manual appears | Sequence stats involving Manual partly **reflect exposure**, not only strategy | Treat manual sequences as pure strategy |
| Long shared substrings (up to 27 whole-object labels) exist between some recordings | Some **local loops** repeat across people | Identical full-task behaviour |
| Median detailed episode ~0.46 s; 73% &lt; 1 s | Attention episodes are **brief**; long looks are rare | Long look = confusion |
| 32% of detailed transitions stay within the same parent tool | Much “detail” is **within-tool**, not tool switching | Count every part change as a new action |

**Inference we endorse for the group:**  
The corpus supports a **grinder-centred, box-coupled, short-episode** attention regime, with expertise linked most clearly to **reduced manual share** and somewhat **increased screwdriver / grinder–box cycling** — all within labelled gaze.

---

### 2.5 How is this data used now, and how could it be used later?

#### Used now (internship / CRC analysis path)

1. **Corpus psychology sheets:** object time budgets, transition matrices, sequence strings.  
2. **Expertise comparison** (exploratory, participant-weighted).  
3. **QC of the detector pipeline** (coverage, failed TSVs, sensitivity to coverage filters).  
4. **Prioritisation** of what to validate manually (manual detections; part/zone labels; mid-task screwdriver rise).

#### Near-term (with little new labelling)

1. **Simple expert/non-expert scorecards** (manual %, screwdriver %, selected motif rates) for internal discussion.  
2. **Recording triage:** flag sessions with extreme manual share or near-zero tool labels for review.  
3. **Hypothesis generation** for the next Tobii study (stage markers; performance metrics).

#### Medium-term (needs extra annotation or denser labels)

1. **Step recognition models** once stages are coded.  
2. **Gaze–performance models** once time-on-task / errors exist.  
3. **Assistance / AR triggers** (e.g. prolonged manual-looking → offer guidance) — *only after* validation; not justified by this snapshot alone.  
4. **Training evaluation:** does a course reduce manual share toward the expert profile?

#### Longer-term product / lab vision

A reusable **attended-object layer** on egocentric assembly recordings: same pipeline, new kits; human-in-the-loop correction of hard classes (screws, true motor region); closed-loop experiments linking gaze patterns to assistance.

---

## 3. Curating what is meaningful (why *these* results, not others)

The report can feel like “many plots.” Below is an intentional **shortlist** — what to keep in a talk or paper draft — and why each earns its place.

### Tier A — Lead with these (connected story)

1. **Manual share separates expertise**  
   *Why:* Clearest group signal; robust; maps to a human-interpretable behaviour (guidance seeking).  
   *Connection:* Anchors the claim “gaze object streams carry expertise-relevant structure.”

2. **Grinder dominates labelled attention (~½)**  
   *Why:* Defines the task’s visual centre of gravity.  
   *Connection:* Explains why motifs orbit the grinder; expertise is *not* “who looks at AG.”

3. **Grinder↔box (and grinder↔screwdriver) routes are widespread**  
   *Why:* Turns static shares into **process**.  
   *Connection:* Links object budgets to workflow cycling; experts densify some of these loops.

4. **Coverage ~20% + unlabelled gaps**  
   *Why:* Scientific honesty; prevents over-reading transitions as continuous eye paths.  
   *Connection:* Motivates the denser (stride-2, hand/table/parts) rebatch and stage annotation.

### Tier B — Supporting (keep if time)

5. **Screwdriver mid-span rise** — temporal structure without claiming named steps.  
6. **Box often first/last labelled object** — observed session boundaries.  
7. **Within-tool part structure** (tip-heavy screwdriver; AG body/side-handle-zone) — shows fine labels add information, pending manual validation.  
8. **Short episode durations** — sets expectation that “attention units” are brief.

### Tier C — Demote or appendix-only

- Sparse wrench-part group tests (only ~146 wrench fixations).  
- Longest shared substrings / identical full strings (interesting, easy to over-interpret).  
- Broad duration/switching “experts always …” claims that fail correction.  
- Logged-window subset (92 recordings, 2 experts) as if it were the main result.

### How Tier A connects (one narrative)

```text
We can label what people look at on the kit
        ↓
Attention is grinder-centred, with box/tool cycling
        ↓
Experts look far less at the manual (and somewhat more at the screwdriver)
        ↓
So object-level gaze streams are already useful for group comparison
        ↓
But ~80% of fixations are still unlabelled and stages are unannotated
        ↓
Next: denser labels + stage/performance ground truth → step recognition & assistance
```

That is why these results are shown together: they are not random KPIs; they are the **minimum set** that (a) proves corpus-scale utility, (b) yields a usable expertise signal, (c) sketches process structure, and (d) states the limit that defines the next experiment.

---

## 4. What we should *not* claim (explicit)

- That we measure **workload, confusion, or skill** directly.  
- That start/middle/end thirds are **disassembly steps**.  
- That part/zone labels (especially geometric AG zones) are **anatomically verified**.  
- That transition matrices are **continuous gaze paths** (unlabelled gaps intervene).  
- That detecting objects alone is the scientific contribution — the contribution is **scalable attended-object structure for analysis**.

---

## 5. Recommended talking points for the supervisor meeting

1. **Meaning:** We turned hundreds of Tobii videos into **object attention budgets and sequences** usable for CRC/psychology analysis.  
2. **Expertise:** The standout finding is **much less manual looking in experts**; screwdriver share and some grinder–box motifs support a secondary contrast.  
3. **Process:** We see **recurring grinder↔box / grinder↔tool cycles**, not yet named procedure steps.  
4. **Now:** Use these metrics for **group comparison, triage, and hypothesis design**.  
5. **Next:** Add **task-stage + performance labels**, validate hard classes, and push labelling coverage (hand / surfaces / AG parts, denser frames) so step recognition becomes testable.

---

## 6. Open points (queries back to supervisor)

If useful, clarification on these would sharpen the next analysis cycle:

1. **Is the supplied expert list the definitive grouping** for publication, or should expertise be redefined (years, certification, task score)?  
2. **Do recordings mix assembly and disassembly**, and can we get stage timestamps or checklist codes?  
3. For “differentiate experts,” is the goal **descriptive group difference**, **individual classification**, or **training outcome tracking**?  
4. Should **manual-looking reduction** be treated as a candidate **training KPI** for the next study design?

---

## 7. Bottom line

The results are impressive as engineering, but their value for the group is this:

> We can already **see expertise-related differences in how attention is allocated across the kit**, especially **manual avoidance in experts**, against a **grinder-centred, box-coupled workflow**. We **cannot yet** say which disassembly step someone is in, or that patterns prove cognitive strategy or performance. The curated findings above are the ones that matter because they jointly answer: *what the stream measures, who differs, what process looks like, and what must come next.*

*Document version: answers to supervisor feedback on Final Gaze Visual Report (Oct 2026). Numbers cited from that report’s equal-participant / exploratory statistics unless noted.*
