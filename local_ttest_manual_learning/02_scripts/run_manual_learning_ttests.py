#!/usr/bin/env python3
"""Participant-level t-tests and log decay for manual gaze learning.

Runs only on real CSVs placed in ../01_inputs_needed/.
Does not invent or simulate participant values from group means.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "01_inputs_needed"
OUT = ROOT / "03_outputs"
LEARNING_RECS = set(range(1, 9))  # 1–8 only; 9–10 = no-manual condition


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def find_input() -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    share = INP / "manual_share_by_participant_recording.csv"
    seq = INP / "corpus_attention_sequences.csv"
    exp = INP / "participants_expertise.csv"
    dur = INP / "recording_duration_by_participant.csv"

    if share.exists():
        df = pd.read_csv(share)
        notes.append(f"loaded {share.name} ({len(df)} rows)")
    elif seq.exists() and exp.exists():
        df = build_from_sequences(seq, exp)
        notes.append(f"built share table from {seq.name} + {exp.name} ({len(df)} rows)")
    else:
        fail(
            "Missing participant-level input.\n"
            f"  Place one of:\n"
            f"    {share.name}\n"
            f"    OR {seq.name} + {exp.name}\n"
            f"  in {INP}\n"
            "Group-mean tables in 00_available_results/ are not enough for t-tests."
        )

    required = ["participant_id", "recording_number", "manual_share"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        fail(f"Input missing columns: {missing}. Found: {list(df.columns)}")

    df["recording_number"] = pd.to_numeric(df["recording_number"], errors="coerce").astype("Int64")
    df["manual_share"] = pd.to_numeric(df["manual_share"], errors="coerce")
    if "task" not in df.columns:
        df["task"] = np.where(df["recording_number"] % 2 == 1, "Disassembly", "Assembly")
    if "task_repetition" not in df.columns:
        df["task_repetition"] = (df["recording_number"] + 1) // 2
    if "expertise" not in df.columns:
        fail("Column 'expertise' missing. Add it or provide participants_expertise.csv.")

    if dur.exists() and "task_duration_s" not in df.columns:
        d = pd.read_csv(dur)
        need = {"participant_id", "recording_number", "task_duration_s"}
        if not need.issubset(d.columns):
            fail(f"{dur.name} needs columns {need}")
        df = df.merge(
            d[["participant_id", "recording_number", "task_duration_s"]],
            on=["participant_id", "recording_number"],
            how="left",
        )
        notes.append(f"merged durations from {dur.name}")

    bad = df["manual_share"].isna().sum()
    if bad:
        fail(f"{bad} rows have missing manual_share; fix the input before testing.")
    if ((df["manual_share"] < 0) | (df["manual_share"] > 1)).any():
        fail("manual_share must be in [0, 1].")

    return df, notes


def build_from_sequences(seq_path: Path, exp_path: Path) -> pd.DataFrame:
    seq = pd.read_csv(seq_path)
    exp = pd.read_csv(exp_path)
    # Accept common column name variants; do not guess values.
    colmap_candidates = {
        "recording_id": ["recording_id", "Recording", "recording"],
        "attended_label": ["attended_label", "label", "object", "attended_object"],
        "duration": [
            "total_gaze_event_duration_raw",
            "gaze_event_duration_raw",
            "duration_ms",
            "duration_s",
            "total_duration_ms",
        ],
    }
    resolved: dict[str, str] = {}
    for key, opts in colmap_candidates.items():
        hit = next((c for c in opts if c in seq.columns), None)
        if hit is None:
            fail(
                f"Cannot build share table: no column matching {key} in {seq_path.name}.\n"
                f"  looked for {opts}\n"
                f"  found {list(seq.columns)}\n"
                "Provide manual_share_by_participant_recording.csv instead."
            )
        resolved[key] = hit

    if "participant_id" not in exp.columns or "expertise" not in exp.columns:
        fail(f"{exp_path.name} needs columns participant_id,expertise. Found: {list(exp.columns)}")

    s = seq.rename(columns={resolved["recording_id"]: "recording_id",
                            resolved["attended_label"]: "attended_label",
                            resolved["duration"]: "duration_raw"})
    s["duration_raw"] = pd.to_numeric(s["duration_raw"], errors="coerce")
    # Tobii raw duration is typically milliseconds; keep unit consistent within file only.
    unit = "ms" if s["duration_raw"].median(skipna=True) > 20 else "s"

    def parse_rec(rid: str) -> tuple[str, int]:
        text = str(rid)
        if "_" not in text:
            fail(f"recording_id not parseable (expected PARTICIPANT_NN): {rid}")
        pid, suffix = text.rsplit("_", 1)
        if not suffix.isdigit():
            fail(f"recording suffix not numeric: {rid}")
        return pid, int(suffix)

    parsed = s["recording_id"].map(parse_rec)
    s["participant_id"] = [p[0] for p in parsed]
    s["recording_number"] = [p[1] for p in parsed]

    # Only labelled episodes contribute to the denominator (matches report definition).
    s = s.dropna(subset=["attended_label", "duration_raw"])
    s = s[s["attended_label"].astype(str).str.len() > 0]

    def is_manual(lab: str) -> bool:
        t = str(lab).strip().lower()
        return "manual" in t or t in {"instruction manual", "instruction_manual"}

    s["is_manual"] = s["attended_label"].map(is_manual)
    g = s.groupby(["participant_id", "recording_number", "recording_id"], as_index=False).agg(
        labelled_duration=("duration_raw", "sum"),
        manual_duration=("duration_raw", lambda x: x[s.loc[x.index, "is_manual"]].sum()),
    )
    g["manual_share"] = np.where(g["labelled_duration"] > 0, g["manual_duration"] / g["labelled_duration"], np.nan)
    g["duration_unit"] = unit
    if unit == "ms":
        g["manual_duration_s"] = g["manual_duration"] / 1000.0
        g["labelled_duration_s"] = g["labelled_duration"] / 1000.0
    else:
        g["manual_duration_s"] = g["manual_duration"]
        g["labelled_duration_s"] = g["labelled_duration"]

    out = g.merge(exp[["participant_id", "expertise"]], on="participant_id", how="left")
    if out["expertise"].isna().any():
        missing = sorted(out.loc[out["expertise"].isna(), "participant_id"].unique())
        fail(f"Expertise missing for participants: {missing}")
    return out


def welch_ttest(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"n_a": len(a), "n_b": len(b), "error": "too few observations"}
    t, p = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
    # Cohen's d (Hedges not required for first pass)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "sd_a": float(a.std(ddof=1)),
        "sd_b": float(b.std(ddof=1)),
        "t": float(t),
        "p": float(p),
        "cohens_d": float(d),
    }


def paired_ttest(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return {"n": int(len(x)), "error": "too few paired observations"}
    t, p = stats.ttest_rel(x, y, alternative="two-sided")
    diff = x - y
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    return {
        "n": int(len(x)),
        "mean_x": float(x.mean()),
        "mean_y": float(y.mean()),
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)),
        "t": float(t),
        "p": float(p),
        "cohens_d_z": float(d),
    }


def fit_log_decay(rec: np.ndarray, y: np.ndarray) -> dict:
    """Exponential: log(y+eps) = a - k * recording. Also power: log(y+eps) = a - b * log(rec)."""
    rec = np.asarray(rec, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(rec) & ~np.isnan(y) & (y >= 0)
    rec, y = rec[m], y[m]
    if len(rec) < 3:
        return {"n": int(len(rec)), "error": "need ≥3 points"}
    eps = 1e-6
    ly = np.log(y + eps)
    # exponential in recording number
    k, a = np.polyfit(rec, ly, 1)  # ly = a + k*rec with k negative when decaying
    # polyfit returns highest power first: ly ≈ k*rec + a
    yhat = a + k * rec
    ss_res = float(np.sum((ly - yhat) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2_exp = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    half_life = (np.log(0.5) / k) if k < 0 else np.nan
    # power law
    bp, ap = np.polyfit(np.log(rec), ly, 1)
    yhat_p = ap + bp * np.log(rec)
    ss_resp = float(np.sum((ly - yhat_p) ** 2))
    r2_pow = 1 - ss_resp / ss_tot if ss_tot > 0 else np.nan
    return {
        "n": int(len(rec)),
        "exp_intercept": float(a),
        "exp_slope_k": float(k),
        "exp_decay_per_recording": float(1 - np.exp(k)),
        "exp_half_life_recordings": float(half_life) if np.isfinite(half_life) else None,
        "exp_r2": float(r2_exp),
        "power_intercept": float(ap),
        "power_slope_b": float(bp),
        "power_r2": float(r2_pow),
        "better_model": "exponential" if r2_exp >= r2_pow else "power",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, notes = find_input()
    df.to_csv(OUT / "analysis_table_participant_recording.csv", index=False)

    learn = df[df["recording_number"].isin(LEARNING_RECS)].copy()
    noman = df[df["recording_number"].isin({9, 10})].copy()

    rows: list[dict] = []

    # 1) Expert vs non-expert (participant mean over rec 1–8)
    pmean = learn.groupby(["participant_id", "expertise"], as_index=False)["manual_share"].mean()
    a = pmean.loc[pmean["expertise"].str.lower().str.startswith("exp"), "manual_share"].to_numpy()
    b = pmean.loc[pmean["expertise"].str.lower().str.contains("non"), "manual_share"].to_numpy()
    r = welch_ttest(a, b)
    r.update({"test": "welch_expert_vs_nonexpert_manual_share_mean_rec1_8", "a": "Expert", "b": "Non-expert"})
    rows.append(r)

    # 2) Paired rec1 vs rec8
    wide = learn.pivot_table(index="participant_id", columns="recording_number", values="manual_share", aggfunc="mean")
    if 1 in wide.columns and 8 in wide.columns:
        r = paired_ttest(wide[1].to_numpy(), wide[8].to_numpy())
        r.update({"test": "paired_manual_share_rec1_vs_rec8", "x": "rec1", "y": "rec8"})
        rows.append(r)

    # 3) Disassembly vs assembly (participant means, rec 1–8)
    ta = learn.groupby(["participant_id", "task"])["manual_share"].mean().unstack()
    if {"Disassembly", "Assembly"}.issubset(ta.columns):
        r = paired_ttest(ta["Disassembly"].to_numpy(), ta["Assembly"].to_numpy())
        r.update({"test": "paired_manual_share_disassembly_vs_assembly_rec1_8", "x": "Disassembly", "y": "Assembly"})
        rows.append(r)

    # 4) Within-pair vs between-pair change (relative), per participant
    def pair_changes(g: pd.DataFrame) -> pd.Series:
        g = g.set_index("recording_number")["manual_share"]
        within, between = [], []
        for d, a in [(1, 2), (3, 4), (5, 6), (7, 8)]:
            if d in g.index and a in g.index and g[d] > 0:
                within.append((g[a] - g[d]) / g[d])
        for a, d in [(2, 3), (4, 5), (6, 7)]:
            if a in g.index and d in g.index and g[a] > 0:
                between.append((g[d] - g[a]) / g[a])
        return pd.Series(
            {
                "within_mean_rel_change": np.mean(within) if within else np.nan,
                "between_mean_rel_change": np.mean(between) if between else np.nan,
                "n_within": len(within),
                "n_between": len(between),
            }
        )

    ch = learn.groupby(["participant_id", "expertise"]).apply(pair_changes).reset_index()
    ch.to_csv(OUT / "within_vs_between_per_participant.csv", index=False)
    for label, sub in [
        ("all", ch),
        ("non-expert", ch[ch["expertise"].str.lower().str.contains("non")]),
        ("expert", ch[ch["expertise"].str.lower().str.startswith("exp")]),
    ]:
        r = paired_ttest(sub["within_mean_rel_change"].to_numpy(), sub["between_mean_rel_change"].to_numpy())
        r.update({"test": f"paired_within_vs_between_rel_change_{label}", "x": "within_pair", "y": "between_pair"})
        rows.append(r)

    # 5) Per-participant decay slopes (rec 1–8)
    slopes = []
    for (pid, ex), g in learn.groupby(["participant_id", "expertise"]):
        fit = fit_log_decay(g["recording_number"].to_numpy(), g["manual_share"].to_numpy())
        fit.update({"participant_id": pid, "expertise": ex, "outcome": "manual_share"})
        slopes.append(fit)
        if "task_duration_s" in g.columns and g["task_duration_s"].notna().sum() >= 3:
            fitd = fit_log_decay(g["recording_number"].to_numpy(), g["task_duration_s"].to_numpy())
            fitd.update({"participant_id": pid, "expertise": ex, "outcome": "task_duration_s"})
            slopes.append(fitd)
    slope_df = pd.DataFrame(slopes)
    slope_df.to_csv(OUT / "per_participant_log_decay.csv", index=False)

    man_slopes = slope_df[slope_df["outcome"] == "manual_share"]
    if not man_slopes.empty and "exp_slope_k" in man_slopes:
        from scipy import stats

        k = man_slopes["exp_slope_k"].dropna().to_numpy()
        if len(k) >= 2:
            t, p = stats.ttest_1samp(k, 0.0, alternative="less")
            rows.append(
                {
                    "test": "one_sample_exp_slope_k_less_than_0_manual_share",
                    "n": int(len(k)),
                    "mean_k": float(k.mean()),
                    "sd_k": float(k.std(ddof=1)),
                    "t": float(t),
                    "p": float(p),
                }
            )
        ke = man_slopes.loc[man_slopes["expertise"].str.lower().str.startswith("exp"), "exp_slope_k"].to_numpy()
        kn = man_slopes.loc[man_slopes["expertise"].str.lower().str.contains("non"), "exp_slope_k"].to_numpy()
        r = welch_ttest(ke, kn)
        r.update({"test": "welch_exp_slope_k_expert_vs_nonexpert", "a": "Expert", "b": "Non-expert"})
        rows.append(r)

    # 6) No-manual condition summary (rec 9–10) — descriptive only from data
    if not noman.empty:
        noman.groupby(["expertise", "recording_number"])["manual_share"].agg(["count", "mean", "std", "min", "max"]).reset_index().to_csv(
            OUT / "no_manual_condition_rec9_10_summary.csv", index=False
        )

    pd.DataFrame(rows).to_csv(OUT / "ttest_results.csv", index=False)
    meta = {
        "notes": notes,
        "n_rows_input": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "learning_recordings": sorted(LEARNING_RECS),
        "no_manual_recordings": [9, 10],
        "assumption": "Recordings 9–10: instruction manual was not provided; excluded from learning fits.",
        "outputs": sorted(p.name for p in OUT.glob("*.csv")),
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("Wrote t-test results to", OUT / "ttest_results.csv")


if __name__ == "__main__":
    main()
