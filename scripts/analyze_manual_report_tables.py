"""Step-oriented analysis of the aggregate tables from the Oct-2026 corpus report.

Inputs (docs/manual_step_results/source/):
  attention_share_by_task_expertise.csv  task x expertise x object share
  manual_share_by_task_repetition.csv    manual share per task x expertise x repetition
  top_routes_by_direction.csv            3-object routes per direction

Outputs (docs/manual_step_results/derived/):
  chronological_manual_curve.csv   manual share over the 10 recordings in time order
  manual_learning_fit.csv          decay fit / catch-up point per group
  direction_effect.csv             assembly minus disassembly per object and group
  expertise_gap.csv                expert minus non-expert per object and direction
  attention_budget.csv             guidance (manual+box) vs execution (grinder+tools)
  routes_mapped_to_manual.csv      routes -> action pattern -> candidate manual steps

Usage:
  python scripts/analyze_manual_report_tables.py [--source DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS_PER_DIRECTION = 180  # 36 participants x 5 repetitions

ROLE = {
    "angle grinder": "W",
    "storage box": "S",
    "screwdriver": "T",
    "wrench": "T",
    "instruction manual": "C",
}

# Action pattern (role trigram) -> reading and the manual steps it can occur in.
PATTERNS = {
    ("Disassembly", "W>S>W"): ("deposit part, return to grinder", "D1, D2, D3 (Box 1-3), D8 sorting"),
    ("Disassembly", "S>W>S"): ("box-grinder-box shuttling", "D8 sorting into Box 4/5; D1-D3 deposits"),
    ("Disassembly", "T>W>S"): ("unscrew -> remove part -> deposit", "D2, D3 (screwdriver then Box 2/3)"),
    ("Disassembly", "W>T>W"): ("operate screwdriver on grinder", "D2-D5 unscrew block"),
    ("Disassembly", "S>W>T"): ("after deposit, start next screw", "D2->D3, D3->D4 hand-over"),
    ("Disassembly", "W>T>S"): ("tool then box", "D2, D3 (screws to box)"),
    ("Disassembly", "W>S>T"): ("deposit then pick tool", "D1->D2, D2->D3 hand-over"),
    ("Disassembly", "W>S>C"): ("deposit then read manual", "step boundary after D1/D2/D3"),
    ("Assembly", "S>W>S"): ("fetch part - fit - fetch again", "A1 (Box 5), A4 (Box 4), A6 (Box 3), A7 (Box 2), A8 (Box 1)"),
    ("Assembly", "W>S>W"): ("look up part in box while fitting", "A1, A4, A6-A8"),
    ("Assembly", "S>W>T"): ("fetch part -> place -> fasten", "A4, A6, A7"),
    ("Assembly", "W>T>W"): ("operate screwdriver on grinder", "A4-A7 screw block"),
    ("Assembly", "T>W>S"): ("fasten -> check -> next part", "A4->A5, A6->A7, A7->A8"),
    ("Assembly", "T>S>W"): ("tool -> fetch -> fit", "A5->A6, A6->A7"),
    ("Assembly", "W>T>S"): ("tool then box", "A4, A6, A7"),
    ("Assembly", "W>S>T"): ("fetch screw/part then tool", "A4, A6, A7"),
}


def chronological_curve(rep: pd.DataFrame) -> pd.DataFrame:
    data = rep.copy()
    data["task_l"] = data["task"].str.lower()
    # Within each repetition, disassembly (odd number) precedes assembly (even).
    data["recording_number"] = np.where(
        data["task_l"] == "disassembly", 2 * data["task_repetition"] - 1, 2 * data["task_repetition"]
    )
    wide = data.pivot_table(index=["recording_number", "task", "task_repetition"], columns="expertise", values="mean").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"Expert": "expert_manual_share", "Non-expert": "non_expert_manual_share"})
    wide["gap_pp"] = 100 * (wide["non_expert_manual_share"] - wide["expert_manual_share"])
    wide["ratio_non_expert_to_expert"] = wide["non_expert_manual_share"] / wide["expert_manual_share"]
    return wide.sort_values("recording_number").reset_index(drop=True)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def learning_fit(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    first8 = curve[curve["recording_number"] <= 8]
    for col, group in (("expert_manual_share", "expert"), ("non_expert_manual_share", "non-expert")):
        x = first8["recording_number"].to_numpy(dtype=float)
        y = first8[col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x - 1, np.log(y), 1)
        rows.append(
            {
                "group": group,
                "share_recording_1": float(curve[col].iloc[0]),
                "share_recording_8": float(curve.loc[curve["recording_number"] == 8, col].iloc[0]),
                "share_recordings_9_10_mean": float(curve.loc[curve["recording_number"] >= 9, col].mean()),
                "spearman_order_vs_share_rec1_8": _spearman(x, y),
                "spearman_order_vs_share_rec1_10": _spearman(
                    curve["recording_number"].to_numpy(dtype=float), curve[col].to_numpy(dtype=float)
                ),
                "decay_per_recording_rec1_8": float(1 - np.exp(slope)),
                "half_life_recordings_rec1_8": float(np.log(2) / -slope) if slope < 0 else np.nan,
                "drop_rec8_to_rec9_fold": float(
                    curve.loc[curve["recording_number"] == 8, col].iloc[0]
                    / curve.loc[curve["recording_number"] == 9, col].iloc[0]
                ),
            }
        )
    out = pd.DataFrame(rows)
    expert_first = float(curve["expert_manual_share"].iloc[0])
    catch = curve.loc[curve["non_expert_manual_share"] <= expert_first * 1.25, "recording_number"]
    out["non_expert_reaches_expert_rec1_level_at"] = int(catch.iloc[0]) if len(catch) else None
    return out


def direction_effect(shares: pd.DataFrame) -> pd.DataFrame:
    wide = shares.pivot_table(index=["expertise", "label"], columns="task", values="mean").reset_index()
    wide.columns.name = None
    wide["assembly_minus_disassembly_pp"] = 100 * (wide["Assembly"] - wide["Disassembly"])
    return wide.rename(columns={"Assembly": "assembly_share", "Disassembly": "disassembly_share"})


def expertise_gap(shares: pd.DataFrame) -> pd.DataFrame:
    wide = shares.pivot_table(index=["task", "label"], columns="expertise", values="mean").reset_index()
    wide.columns.name = None
    wide["expert_minus_non_expert_pp"] = 100 * (wide["Expert"] - wide["Non-expert"])
    return wide.rename(columns={"Expert": "expert_share", "Non-expert": "non_expert_share"})


def attention_budget(shares: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "guidance (manual + box)": {"instruction manual", "storage box"},
        "execution (grinder + tools)": {"angle grinder", "screwdriver", "wrench"},
        "manual only": {"instruction manual"},
        "tools only": {"screwdriver", "wrench"},
    }
    rows = []
    for (task, exp), grp in shares.groupby(["task", "expertise"]):
        for name, labels in groups.items():
            rows.append(
                {"task": task, "expertise": exp, "budget": name, "share": float(grp.loc[grp["label"].isin(labels), "mean"].sum())}
            )
    return pd.DataFrame(rows)


def routes_mapped(routes: pd.DataFrame) -> pd.DataFrame:
    out = routes.copy()
    out["role_pattern"] = out["sequence"].map(lambda s: ">".join(ROLE.get(p.strip(), "?") for p in s.split(">")))
    out["occurrences_per_recording"] = out["occurrences"] / RECORDINGS_PER_DIRECTION
    out["share_of_recordings"] = out["recordings"] / RECORDINGS_PER_DIRECTION
    out["action_reading"] = [PATTERNS.get((t, p), ("", ""))[0] for t, p in zip(out["task"], out["role_pattern"])]
    out["candidate_manual_steps"] = [PATTERNS.get((t, p), ("", ""))[1] for t, p in zip(out["task"], out["role_pattern"])]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "docs/manual_step_results/source"))
    parser.add_argument("--out", default=str(ROOT / "docs/manual_step_results/derived"))
    args = parser.parse_args()
    src, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shares = pd.read_csv(src / "attention_share_by_task_expertise.csv")
    rep = pd.read_csv(src / "manual_share_by_task_repetition.csv")
    routes = pd.read_csv(src / "top_routes_by_direction.csv")

    curve = chronological_curve(rep)
    fit = learning_fit(curve)
    tables = {
        "chronological_manual_curve.csv": curve,
        "manual_learning_fit.csv": fit,
        "direction_effect.csv": direction_effect(shares),
        "expertise_gap.csv": expertise_gap(shares),
        "attention_budget.csv": attention_budget(shares),
        "routes_mapped_to_manual.csv": routes_mapped(routes),
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    print(json.dumps({"written": sorted(tables), "out": str(out)}, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        for name, table in tables.items():
            print(f"\n== {name}")
            print(table.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
