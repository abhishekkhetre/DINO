# Facts for supervisor (Nov-2 HPC sheet only)

Source: `corpus_attention_sequences.csv` (mtime 2026-11-02) + `participants_expertise.csv`.  
Independent recomputation matched script outputs (max |Δ| manual_share = 0).  
Slide-7 invented step weights were **not** recomputed.

## Sample
- 24 participants (14 Expert, 10 Non-expert), 238 recordings, 21 420 episodes  
- Labels: Angle Grinder, Boxes, Manual only  
- Manual share = Manual / (Manual + Boxes + Angle Grinder)

## Supported (Holm α=0.05 within family)

| Claim | Numbers | Test |
|---|---|---|
| Experts look at the manual less | Expert **3.53%** vs Non-expert **14.35%** (participant means, rec 1–8) | Welch t=−3.80, p_Holm=0.014 |
| Experts are faster | Expert **186 s** vs Non-expert **284 s** mean task duration | Welch t=−3.73, p_Holm=0.013 |
| Expertise gap in both directions | Disassembly 4.2% vs 16.5%; Assembly 2.9% vs 12.2% | both Holm-sig |
| Practice reduces manual use | Same people: rec1 **13.83%** → rec8 **3.14%** | paired t=3.38, p_Holm=0.005 |
| Log decay of manual use | mean slope k=**−0.351** on log(manual_share) | one-sample t, p_Holm=0.006 |
| Log decay of duration | mean slope k=**−0.049** | one-sample t, p_Holm=0.0005 |
| Non-experts shorten duration faster | duration slope Expert −0.014 vs Non-expert −0.099 | Welch, p_Holm=0.0005 |
| Rec 9–10 no-manual check | rec8 3.27% → rec9 0.42% | paired n=23, p_Holm=0.008 |

## Not supported on this sheet
- Within-pair vs between-pair absolute manual changes (all three tests n.s. after Holm)  
- Overall Disassembly vs Assembly manual share (raw p≈0.053, not Holm-sig)  
- Assembly more grinder-heavy (paired n.s.)  
- Screwdriver/wrench / T-routes (labels absent)  
- Old slide bar heights (different 5-label denominator + different n; see cross-check CSV, mean |Δ|≈0.056)

## Files
- Full report: `EMPIRICAL_REPORT.md`  
- All 33 tests: `all_ttests.csv`  
- Curves: `descriptive_curves_this_batch.csv` + `figures/`  
- Old vs this batch: `crosscheck_vs_previous_report_manual_share.csv`
