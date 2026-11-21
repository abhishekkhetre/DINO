# How the percentages on slide 7 were calculated

**What the graph answers:**  
Given a short sequence of things the person looked at, *how likely is each of the 8 manual steps?*

This note explains every percentage on that slide in plain language. No training on our gaze data is involved. The numbers come from matching the gaze sequence to what each manual step is expected to look like.

---

## The idea in three sentences

1. For each manual step we wrote down, from the instruction pictures, how often you would expect to look at the grinder, screwdriver, box, wrench, etc.
2. For a short gaze sequence we multiply those chances together for each step (one product per step).
3. We turn the eight products into percentages that add up to 100%. The tallest bar is the best-matching step.

That is all. The bars are **not** “how often this happened in the corpus.” They are “how well this gaze window matches each step of the manual.”

---

## Step A — Expected looking mix for each manual step

From the manual pictures we assign **weights** (relative importance), then convert them to probabilities that sum to 1 for each step.

**Example: assembly step “Fit housing ring” (A4)**  
Manual shows: red-handled screwdriver, housing ring, Box 4 (red).

| Object | Weight we set | After converting to a probability |
|---|---:|---:|
| Screwdriver | 4.0 | 0.271 |
| Angle grinder | 4.0 (default) | 0.271 |
| Box | 1.5 | 0.102 |
| Screw / hand | 1.5 each | 0.102 each |
| Manual | 1.0 | 0.068 |
| Other objects | small | small |

**Colour (only when the label knows colour):**

| | Green | Red |
|---|---:|---:|
| Screwdriver at A4 | 10% | **90%** |
| So chance of a *red* screwdriver at A4 | | \(0.271 × 0.9 = 0.244\) |

Other steps have their own weights. For example “Refit gear-head cover” (A6) looks similar for screwdriver/grinder/box, but the screwdriver colour is flipped (**90% green**, 10% red). That is why colour separates A4 from A6 later.

We do this once for all 8 disassembly steps (D1–D8) and all 8 assembly steps (A1–A8).

---

## Step B — Score one gaze sequence against every step

Take the objects in order. For each step, multiply that step’s probability for each object:

\[
\text{score(step)} = P(\text{1st object} \mid \text{step}) × P(\text{2nd} \mid \text{step}) × \cdots × P(\text{last} \mid \text{step})
\]

A step that expects those objects gets a larger product. A step that almost never looks at them (e.g. “attach side handle” when we see a screwdriver) gets a tiny product.

---

## Step C — Turn scores into the percentages on the graph

\[
\text{percentage(step)} = \frac{\text{score(step)}}{\text{sum of scores of all 8 steps}} × 100\%
\]

The eight bars always add to 100%.

**Phase certainty** (e.g. “unscrew block 99.8%”) is just the sum of the bars that belong to that phase.

---

# Worked example 1 — Taking the grinder apart

**What the person looked at:**  
screwdriver → grinder → screwdriver → box → grinder

(These are the objects after simplifying tip/handle/body labels.)

### Probabilities used for the four unscrewing steps

| Looked at | D2 Lever bracket | D3 Gear-head cover | D4 Housing ring | D5 Rear screw |
|---|---:|---:|---:|---:|
| screwdriver | 0.262 | 0.266 | 0.293 | 0.215 |
| grinder | 0.262 | 0.266 | 0.293 | 0.343 |
| screwdriver | 0.262 | 0.266 | 0.293 | 0.215 |
| box | 0.131 | 0.133 | 0.029 | 0.034 |
| grinder | 0.262 | 0.266 | 0.293 | 0.343 |

### Products (scores)

| Step | Calculation | Score |
|---|---|---:|
| D2 | 0.262 × 0.262 × 0.262 × 0.131 × 0.262 | 0.000621 |
| D3 | 0.266 × 0.266 × 0.266 × 0.133 × 0.266 | 0.000663 |
| D4 | 0.293 × 0.293 × 0.293 × 0.029 × 0.293 | 0.000216 |
| D5 | 0.215 × 0.343 × 0.215 × 0.034 × 0.343 | 0.000186 |
| D1, D6, D7, D8 | (almost never expect this mix) | ≈ 0.000003 together |

Sum of all eight scores ≈ **0.001690**

### Percentages on the graph

| Step | Score ÷ 0.001690 | Bar on slide |
|---|---:|---:|
| D2 Unscrew lever bracket | 0.000621 / 0.001690 | **36.7%** |
| D3 Unscrew gear-head cover | 0.000663 / 0.001690 | **39.2%** |
| D4 Unscrew housing ring | 0.000216 / 0.001690 | **12.8%** |
| D5 Unscrew rear screw | 0.000186 / 0.001690 | **11.0%** |
| All other steps | tiny | ~0% |

**Why D3 slightly beats D2:** the product for D3 is a little larger (0.000663 vs 0.000621).  
**Why D4/D5 are lower:** they expect the box much less often (~0.03 vs ~0.13), which shrinks the product.  
**Why “unscrew block” is 99.8%:** 36.7 + 39.2 + 12.8 + 11.0. The kind of work is clear; the exact screw step is not.

---

# Worked example 2 — Putting together, without colour labels

**What the person looked at:**  
screwdriver → grinder → box → screwdriver

Same method. Leading scores become:

| Step | Plain name | Percentage |
|---|---|---:|
| A6 | Refit gear-head cover | **37.7%** |
| A4 | Fit housing ring | **32.8%** |
| A7 | Refit lever bracket | 17.0% |
| A5 | Tighten rear screw | 12.3% |

Again the **phase** (screwing) is almost certain (~99.8%), but the probability is spread across four similar steps because a plain “screwdriver” and “box” fit several manual steps equally well.

---

# Worked example 3 — Same moment, with colour (housing ring)

**What the person looked at:**  
**red** screwdriver → grinder → **red** box → **red** screwdriver

### Weights that matter for “Fit housing ring” (A4)

From Step A above:

- P(red screwdriver | A4) = **0.244**
- P(grinder | A4) = **0.271**
- P(red box | A4) = **0.102**

### Score for A4

\[
0.244 × 0.271 × 0.102 × 0.244 ≈ 0.001643
\]

### Score for A6 (gear-head cover — expects a *green* screwdriver)

- P(red screwdriver | A6) = only **0.028**
- Product ≈ 0.000023  

So A4’s score is about **70 times** A6’s, almost entirely because of the two red-screwdriver looks.

### After dividing by the sum of all 8 assembly scores

| Step | Plain name | Percentage |
|---|---|---:|
| **A4** | **Fit housing ring** | **91%** |
| A5 | Tighten rear screw | 5% |
| A7 | Refit lever bracket | 2% |
| A6 | Refit gear-head cover | 1% |

**Takeaway for the supervisor:** colour is what turns “somewhere in the screwing phase” into “Fit housing ring at 91%.” The red handle is doing most of the work; the red box alone would not separate A4 from A6 (both use a red box in the manual).

---

## What these percentages are — and are not

| They **are** | They **are not** |
|---|---|
| A match score between one gaze window and each manual step | Counts from the 360 recordings |
| Normalised so the 8 steps add to 100% | Trained classifier accuracy |
| Based on object lists read from the manual | Proof of the true step without video annotation |

To measure real accuracy later we would mark true step times on a sample of recordings and compare. Slide 7 shows what the label set *can* tell apart in principle.

---

## One sentence you can say in the meeting

> “Each bar is: multiply the step’s expected chance for every object in the gaze window, then scale the eight products so they add to 100%. That is how 39%, 37%, and the 91% for Fit housing ring were obtained.”
