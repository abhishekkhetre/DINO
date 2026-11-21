# Empirical re-analysis from latest successful HPC sheets pack

## Hard rules followed
- Numbers come only from `corpus_attention_sequences.csv` (HPC sheets, 2 Nov 2026) + expertise file.
- No slide-7 style step weights or made-up match percentages.
- Previous report graphs are cross-checked, not copied as truth for this batch.

## Coverage
- Participants in this attention sheet: **24** (14 Expert, 10 Non-expert).
- Recordings: **238**; episodes: **21420**.
- Labels present: **Angle Grinder, Boxes, Manual** only (no screwdriver/wrench in this sheet).
- Learning tests use recordings **1–8**; 9–10 are no-manual trials.
- Alias applied: `{'LE07UF1': 'LE07UF17'}`.

## Important: why old graph numbers will not match
Previous report graphs used a 5-label vocabulary (incl. screwdriver, wrench) and ~17/19 participants. This sheet has 3 labels and 24 participants. Manual share here = Manual / (Manual+Boxes+Angle Grinder) duration. It is NOT the same denominator as the old report. Do not equate bar heights to old slides.

## Tests run
Total tests: **33**. Significant after Holm correction within family (α=0.05): **12**.

### Significant results (Holm)
- **welch_exp_slope_expert_vs_nonexpert_task_duration_s**. Expert mean=-0.01367 (n=14) vs Non-expert mean=-0.09871 (n=10). t=5.807, p=5.855e-05, p_Holm=0.0004684. Cohen d=2.631. Why: Do experts and non-experts learn (task_duration_s) at different rates?
- **one_sample_exp_slope_lt0_task_duration_s**. mean slope k=-0.04911 (n=24). t=-4.519, p=7.715e-05, p_Holm=0.0005401. Why: Supervisor asked for logarithmic decay. Fit log(task_duration_s+eps)=a+k*recording per person; test whether mean k < 0 across participants.
- **one_sample_exp_slope_lt0_manual_share**. mean slope k=-0.3509 (n=24). t=-3.486, p=0.0009968, p_Holm=0.005981. Why: Supervisor asked for logarithmic decay. Fit log(manual_share+eps)=a+k*recording per person; test whether mean k < 0 across participants.
- **welch_expert_vs_nonexpert_manual_share_assembly**. Expert mean=0.02856 (n=14) vs Non-expert mean=0.1216 (n=10). t=-3.814, p=0.003181, p_Holm=0.006362. Cohen d=-1.820. Why: Expertise gap may differ by direction. Participant-mean manual share within Assembly only (rec 1–8).
- **welch_expert_vs_nonexpert_manual_share_disassembly**. Expert mean=0.04199 (n=14) vs Non-expert mean=0.1654 (n=10). t=-3.241, p=0.008509, p_Holm=0.008509. Cohen d=-1.549. Why: Expertise gap may differ by direction. Participant-mean manual share within Disassembly only (rec 1–8).
- **welch_expert_vs_nonexpert_task_duration_s**. Expert mean=186.1 (n=14) vs Non-expert mean=283.9 (n=10). t=-3.730, p=0.002664, p_Holm=0.01332. Cohen d=-1.706. Why: Supervisor asked whether expertise differences are real. Unit = participant mean task_duration_s over recordings 1–8. Welch t-test (unequal variance).
- **welch_expert_vs_nonexpert_manual_share**. Expert mean=0.03528 (n=14) vs Non-expert mean=0.1435 (n=10). t=-3.801, p=0.003401, p_Holm=0.01361. Cohen d=-1.826. Why: Supervisor asked whether expertise differences are real. Unit = participant mean manual_share over recordings 1–8. Welch t-test (unequal variance).
- **paired_manual_share_rec8_vs_rec9**. rec8=0.03273 vs rec9=0.004164 (n=23, mean diff=0.02856). t=2.937, p=0.007627, p_Holm=0.007627. dz=0.612. Why: Rec 9–10 had no manual provided. Test that manual share drops from rec8 to rec9.
- **paired_manual_share_rec1_vs_rec8**. rec1=0.1383 vs rec8=0.03136 (n=24, mean diff=0.107). t=3.383, p=0.002563, p_Holm=0.005127. dz=0.691. Why: Learning-curve claim: same person uses the manual less after practice. Paired t-test rec1 vs rec8.
- **paired_route_WSW_disassembly_vs_assembly**. Disassembly=3.719 vs Assembly=7.271 (n=24, mean diff=-3.552). t=-6.436, p=1.441e-06, p_Holm=1.297e-05. dz=-1.314. Why: Does route WSW differ by task direction within person?
- **paired_route_SWS_disassembly_vs_assembly**. Disassembly=3.312 vs Assembly=6.333 (n=24, mean diff=-3.021). t=-5.824, p=6.19e-06, p_Holm=4.952e-05. dz=-1.189. Why: Does route SWS differ by task direction within person?
- **welch_route_SWS_disassembly_expert_vs_nonexpert**. Expert mean rate=4.357/rec (n=14) vs Non-expert=1.85/rec (n=10). t=3.617, p=0.001591, p_Holm=0.01114. Cohen d=1.373. Why: Route SWS (roles W=grinder,S=box,C=manual) rate per recording. Only 3 labels exist in this sheet — no tool role T. Empirical counts after collapsing repeats.

## Headline facts supported by this sheet (with numbers)
- **Expertise — manual gaze:** Expert participant-mean manual share = **3.53%** (n=14) vs Non-expert **14.35%** (n=10); Welch t=-3.80, p_Holm=0.01361.
- **Expertise — speed:** Expert mean task duration = **186.1s** vs Non-expert **283.9s**; Welch significant after Holm.
- **Practice:** Same people drop manual share from rec1 **13.83%** to rec8 **3.14%** (paired n=24, Holm-significant).
- **Log decay:** Mean exponential slope of log(manual_share) across recordings is negative (mean k=-0.351); duration also decays (mean k=-0.049). Non-experts shorten duration faster than experts (Welch on duration slopes Holm-significant).
- **No-manual check:** Rec9 manual share falls vs rec8 (paired n=23), consistent with the experimental note that the manual was not provided.
- **Not supported here:** within-pair vs between-pair absolute manual changes (all three paired tests non-significant after Holm); overall D vs A manual share (p≈0.053 raw, not Holm-sig); screwdriver/wrench routes (labels absent).

### Full machine-readable tables
- `all_ttests.csv` — every test with rationale, raw p, Holm p, means
- `per_recording_shares.csv` — analysis grain (verified: independent recomputation Δ=0 vs this file)
- `per_participant_decay.csv` — log/power fits
- `descriptive_curves_this_batch.csv` — means used for figures
- `crosscheck_vs_previous_report_manual_share.csv` — old vs this batch (mean |Δ| ≈ 0.056; largest gap early non-expert disassembly)
- `figures/` — plots from this batch only

## What previous slides claimed vs this batch
| Previous claim | Supported by this Nov-2 sheet? |
|---|---|
| Experts look less at the manual | **Yes** — 3.5% vs 14.3% participant means, Holm-sig |
| Manual use falls with practice / log decay | **Yes** — rec1 13.8% → rec8 3.1%; mean k&lt;0 Holm-sig |
| Learning mainly within D→A pairs | **Not supported** after Holm on absolute within vs between changes (n=24) |
| Assembly more grinder-heavy / screwdriver-light | **Not supported** for grinder (paired n.s.); **screwdriver cannot** (label absent) |
| Routes T→W→S vs S→W→T | **Not computable** (no tool label T). 3-label SWS/WSW direction effects are Holm-sig |
| Slide 7 step probabilities from manual weights | **Not repeated** — not empirical corpus evidence |

## Bottom line for supervisor
All p-values and means in `03_outputs/empirical_nov2/` are recomputed from the uploaded Nov-2 attention sheet (21 420 episodes, 238 recordings, 24 participants) and the expertise file. An independent recomputation of participant-level shares and the headline Welch/paired/decay tests matched the script CSV to machine precision (max |Δ| share = 0). Old slide bar heights must not be quoted as this batch’s numbers: the cross-check table documents the denominator and sample mismatch. Slide 7-style invented step weights were not regenerated.