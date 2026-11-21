# Related literature (closest to our Tobii + angle-grinder assembly/disassembly study)

Curated for: mobile eye tracking, procedural manuals, expert vs novice, learning over repetitions, object/AOI gaze sequences, assembly & disassembly.

## Closest matches

### 1. Expert vs novice use of procedural documentation (maintenance)
**Differences between Experts and Novices in the Use of Aircraft Maintenance Documentation: Evidence from Eye Tracking**  
Lima Souza et al., *Applied Sciences* 14(3), 1251 (2024).  
Open access: https://doi.org/10.3390/app14031251  
Also: https://www.mdpi.com/2076-3417/14/3/1251  

**Why it matches:** Real maintenance with a long procedural document; mobile eye tracking; experts faster and spend less time on documentation; more consultation early in the task (planning). Closest parallel to our “manual share drops with expertise / early in the process”.

Related short paper from the same line of work:  
**Characterizing the expertise of Aircraft Maintenance Technicians using eye-tracking**  
https://dl.acm.org/doi/fullHtml/10.1145/3517031.3532199  

---

### 2. Object-related gaze *sequences* and expertise development in assembly
**An algorithmic approach to determine expertise development using object-related gaze pattern sequences**  
Hosp et al., *Behavior Research Methods* (2022).  
PMC (open): https://pmc.ncbi.nlm.nih.gov/articles/PMC8863757/  

**Why it matches:** Assembly task with a manual; AOI sequences turned into strings / k-mers; repeated trials; novices’ gaze patterns become more “expert-like” with practice; experts trained until they can assemble **without the manual**. Directly supports our route / sequence analysis and learning-curve story.

---

### 3. Eye tracking in *disassembly* (process dynamics)
**Eye and Gaze Behavior During Human Disassembly Procedures: Effects of Task Conditions and Process Dynamics**  
(2025/2026).  
PMC (open): https://pmc.ncbi.nlm.nih.gov/articles/PMC13514580/  

**Why it matches:** Human disassembly (not only assembly); gaze metrics under uncertainty; revisit/fixation dynamics. Useful contrast: they found **weak** skill-level gaze differences after correction — good for discussing what our expert–non-expert gaps do and do not claim.

---

### 4. Learning curve + declining attention to instructions over repetitions
**Identification of key visual areas that guide an assembly process in real and virtual environments**  
(University of Iowa thesis / dissertation).  
PDF: https://ir.uiowa.edu/cgi/viewcontent.cgi?article=7470&context=etd  

**Why it matches:** Ten assembly cycles with eye tracking; clear drop in attention to visual instructions (~35–55%) and distractors with practice; links learning curves to eye-scan efficiency. Closest to our “manual use halves across recordings”.

---

### 5. AOIs: manual + parts/object area + assembly/workpiece
**Detecting Cognitive States in Assembly Tasks: A Gaze-Based Approach to Confusion and Smoothness of Progress**  
Ozimek et al., OzCHI (2025).  
https://dl.acm.org/doi/10.1145/3764687.3769931  

**Why it matches:** Explicit AOIs almost identical to ours: **manual**, **object/parts**, **assembly** workspace; gaze sequences used to detect confusion / progress. Useful for step/phase prediction framing.

Also related (carburetor / real product AOIs):  
**Analysis of the Gaze Behaviour of the Worker on the Carburetor Assembly Task**  
https://doi.org/10.12777/jati.10.2.63-70  
PDF: https://ejournal.undip.ac.id/index.php/jgti/article/download/8600/7006  

---

### 6. Mobile eye tracking + computer vision for assembly AOIs (expert vs novice + learning)
**Statistical assessment of differences in visual strategies and within-session learning dynamics of operators in industrial assembly based on mobile eye tracking and computer vision**  
Gurbanov (2026).  
https://doi.org/10.26102/2310-6018/2026.61.10.006  

**Why it matches:** Combines mobile eye tracking with CV object/hand detection (methodologically close to Tobii + SAM masks); AOI switching entropy; novices get faster across repetitions, experts do not — parallel to our learning vs expertise contrast. (Paper is in Russian; DOI above.)

---

### 7. Phase-wise attention in complex assembly (step bottlenecks)
**Understanding Assembly Process Quality through Product Complexity and Cognitive Demand**  
Gervasi et al., ICQEM 2026.  
PDF: https://iris.polito.it/retrieve/961badc3-71f5-47b8-9419-fcc3acd779c5/Gervasi%20et%20al%20-%20Understanding%20Assembly%20Process%20Quality%20through%20Product%20Complexity%20and%20Cognitive%20Demand%20-%20ICQEM2026.pdf  

**Why it matches:** Assembly broken into macro-phases; eye tracking shows uneven attentional load across steps — supports our “map gaze to manual steps / find bottleneck steps” goal.

---

## How these map to our claims

| Our finding | Best supporting papers |
|---|---|
| Experts look less at the manual | (1) Lima Souza et al. 2024 |
| Manual use falls with practice / repetitions | (4) Iowa thesis; (2) Hosp et al.; (6) Gurbanov |
| Gaze routes / object sequences encode the procedure | (2) Hosp et al. k-mers; (5) Ozimek AOI sequences |
| AOIs: manual, tools/parts, workpiece, boxes | (5) Ozimek; carburetor AOI paper |
| Disassembly-specific gaze | (3) Disassembly PMC paper |
| Step / phase difficulty varies | (7) Gervasi et al. |

## Suggested starting set (read first)
1. Lima Souza et al. 2024 (documentation + expertise)  
2. Hosp et al. 2022 (gaze sequence patterns + learning)  
3. Disassembly PMC paper (2025)  
4. Iowa thesis (learning curve of instruction looking)  
5. Ozimek et al. 2025 (manual / object / assembly AOIs)
