"""Evidence figures for docs/MANUAL_STEP_MAPPING_RESEARCH.md.

Reads only the aggregate tables in docs/manual_step_results/{source,derived}
and writes, per figure, a PNG plus the exact plotted numbers as CSV, then a
combined PDF and a self-contained HTML evidence report.
"""

from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "manual_step_results"
SRC, DER = RES / "source", RES / "derived"
OUT = RES / "figures"
DATA = OUT / "data"

sys.path.insert(0, str(ROOT / "src"))

EXPERT, NONEXPERT = "#2F6DB5", "#E07B39"
DIS, ASM = "#7A4FA3", "#2A9D8F"
GREY, LIGHT = "#8A8A8A", "#E9E9E9"
LABEL_COLOURS = {
    "angle grinder": "#5B7083",
    "screwdriver": "#D4A017",
    "storage box": "#C0504D",
    "instruction manual": "#4F81BD",
    "wrench": "#2E8B57",
}
PHASE_COLOURS = {
    "handle": "#C9DDF2",
    "unscrew_block": "#F6D8AE",
    "screw_block": "#F6D8AE",
    "wrench": "#CDEBC8",
    "extract": "#E4D3F0",
    "insert": "#E4D3F0",
    "sort": "#F3C9C9",
}
COHORT = "36 participants (17 experts, 19 non-experts), 360 recordings"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "figure.dpi": 100,
        "savefig.dpi": 200,
    }
)

FIGURES: list[dict] = []


def footer(fig, text: str) -> None:
    fig.text(0.01, -0.035, text, fontsize=7.5, color=GREY, ha="left", va="top")


def save(fig, slug: str, title: str, insight: str, proof: str, data: pd.DataFrame, sources: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{slug}.png"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    data.to_csv(DATA / f"{slug}.csv", index=False)
    FIGURES.append(
        {"slug": slug, "title": title, "insight": insight, "proof": proof, "data": data, "sources": sources, "fig": fig}
    )


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


# ---------------------------------------------------------------- figure 0
def fig_manual_process() -> None:
    cat = pd.read_csv(RES / "manual_step_catalog.csv")
    dis = cat[cat["task"].str.lower().str.startswith("dis")].reset_index(drop=True)
    asm = cat[cat["task"].str.lower().str.startswith("ass")].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.2, 4.4)
    ax.axis("off")
    w, h = 0.9, 1.15
    y_dis, y_asm = 2.95, 0.25

    def draw(row: pd.Series, i: int, y: float, colour: str) -> tuple[float, float]:
        x = i + 0.05
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
            fc=PHASE_COLOURS.get(row["phase"], LIGHT), ec=colour, lw=1.4,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.16, row["step_id"], ha="center", va="center", fontsize=12, fontweight="bold", color=colour)
        ax.text(x + w / 2, y + h / 2 - 0.08, short_action(row), ha="center", va="center", fontsize=7.3, wrap=True)
        ax.text(x + w / 2, y + 0.12, str(row.get("tools", "")), ha="center", va="center", fontsize=7, color="#444", style="italic")
        return x + w / 2, y

    centres_d = {r["step_id"]: draw(r, i, y_dis, DIS) for i, r in dis.iterrows()}
    centres_a = {r["step_id"]: draw(r, i, y_asm, ASM) for i, r in asm.iterrows()}
    for _, r in asm.iterrows():
        rev = str(r.get("reverse_of", "") or "")
        if rev in centres_d:
            xa, ya = centres_a[r["step_id"]]
            xd, yd = centres_d[rev]
            ax.annotate("", xy=(xa, ya + h + 0.03), xytext=(xd, yd - 0.03),
                        arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.1, alpha=0.8))
    ax.text(0.05, y_dis + h + 0.18, "Disassembly (manual pages 1–2) — odd recordings 1, 3, 5, 7, 9", color=DIS, fontweight="bold", fontsize=11)
    ax.text(0.05, y_asm - 0.2, "Assembly (manual pages 3–4) — even recordings 2, 4, 6, 8, 10", color=ASM, fontweight="bold", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#999") for c in ["#C9DDF2", "#F6D8AE", "#CDEBC8", "#E4D3F0", "#F3C9C9"]]
    ax.legend(handles, ["handle", "screw block", "wrench", "armature (extract / insert)", "sort"],
              loc="upper right", bbox_to_anchor=(1.0, 1.06), ncol=5, frameon=False, fontsize=8.5, title=None)
    fig.suptitle("Figure 0. The manual as a process: assembly is disassembly reversed", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=13)
    footer(fig, "Source: manual_step_catalog.csv (visual reading of Anleitung Demontage 16.03.2025). Arrows join each assembly step to the disassembly step it reverses.")
    save(
        fig, "fig00_manual_process",
        "The manual as a process",
        "Assembly reverses disassembly; steps group into five phases; colour (green vs red screwdriver, red vs blue box) carries step identity.",
        "Seven reverse pairs: D7↔A1, D6↔A3, D4↔A4, D5↔A5, D3↔A6, D2↔A7, D1↔A8. A2 (turn gear to seat) and D8 (sort) have no counterpart.",
        cat[["task", "step_id", "phase", "action", "tools", "box", "reverse_of"]],
        ["manual_step_catalog.csv"],
    )


SHORT_ACTION = {
    "D1": "side handle off\n→ Box 1", "D2": "lever bracket\nscrews → Box 2", "D3": "gear-head flange\n+ bevel gear\n→ Box 3",
    "D4": "housing-ring\nscrews", "D5": "rear side\nscrew", "D6": "wrench inside\ngear housing", "D7": "pull armature\n+ white gear",
    "D8": "sort: Box 4 (red)\nBox 5 (blue)", "A1": "insert armature\n(from Box 5)", "A2": "turn gear\nby hand to seat", "A3": "wrench inside\ngear housing",
    "A4": "fit housing ring,\ntighten (Box 4)", "A5": "rear side\nscrew", "A6": "refit flange\n(Box 3)", "A7": "lever bracket\n+ spring (Box 2)",
    "A8": "side handle on\n(Box 1)",
}


def short_action(row: pd.Series) -> str:
    return SHORT_ACTION.get(row["step_id"], str(row.get("action", ""))[:40])


# ---------------------------------------------------------------- figure 1
def fig_learning_curve() -> None:
    cur = pd.read_csv(DER / "chronological_manual_curve.csv")
    rec = cur["recording_number"].to_numpy()
    ex, ne = cur["expert_manual_share"].to_numpy(), cur["non_expert_manual_share"].to_numpy()
    is_dis = cur["task"].eq("Disassembly").to_numpy()

    fits = {}
    for name, y in [("expert", ex), ("non-expert", ne)]:
        m = rec <= 8
        b, a = np.polyfit(rec[m], np.log(y[m]), 1)
        fits[name] = (a, b)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.45, 1]})
    ax1.axvspan(8.5, 10.5, color=LIGHT, zorder=0)
    ax1.text(9.5, 49, "Recordings 9–10:\nboth groups\n≈ 0.5%\n(condition\nchange? Q1)", ha="center", va="top", fontsize=8.3, color="#555")
    for name, y, col in [("Experts", ex, EXPERT), ("Non-experts", ne, NONEXPERT)]:
        ax1.plot(rec, 100 * y, color=col, lw=2, zorder=2, label=name)
        ax1.scatter(rec[is_dis], 100 * y[is_dis], color=col, marker="o", s=55, zorder=3, edgecolor="white")
        ax1.scatter(rec[~is_dis], 100 * y[~is_dis], color="white", marker="s", s=50, zorder=3, edgecolor=col, lw=1.8)
        for r, v in zip(rec, y):
            ax1.text(r, 100 * v + 1.6, f"{100 * v:.1f}", ha="center", fontsize=7.5, color=col)
    ax1.axhline(100 * ex[0], ls="--", color=EXPERT, lw=1, alpha=0.7)
    ax1.text(10.45, 100 * ex[0] + 0.8, "experts, recording 1 (11.5%)", ha="right", fontsize=8, color=EXPERT)
    ax1.annotate("non-experts first reach\nthe experts' recording-1 level", xy=(8, 10), xytext=(6.3, 30),
                 arrowprops=dict(arrowstyle="->", color=NONEXPERT), fontsize=8.5, color=NONEXPERT)
    ax1.annotate("", xy=(3, 37.2), xytext=(2, 36.8), arrowprops=dict(arrowstyle="-", color=GREY, ls=":"))
    ax1.text(3.9, 42.5, "assembly 1 = disassembly 2\n(both = 2nd contact)", ha="left", fontsize=8, color="#555")
    ax1.set_xticks(rec, [f"{r}\n{'D' if d else 'A'}" for r, d in zip(rec, is_dis)])
    ax1.set_xlabel("Recording number (D = disassembly ●, A = assembly ■)")
    ax1.set_ylabel("Manual share of labelled gaze (%)")
    ax1.set_ylim(0, 58)
    ax1.set_xlim(0.5, 10.5)
    ax1.legend(loc="upper center", frameon=False)
    ax1.set_title("A. Manual use falls with every recording")

    for name, y, col in [("experts", ex, EXPERT), ("non-experts", ne, NONEXPERT)]:
        a, b = fits["expert" if name == "experts" else "non-expert"]
        m = rec <= 8
        ax2.scatter(rec[m], 100 * y[m], color=col, s=40, zorder=3)
        xs = np.linspace(1, 8, 50)
        ax2.plot(xs, 100 * np.exp(a + b * xs), color=col, lw=1.6, ls="--")
        decay = 1 - np.exp(b)
        half = np.log(0.5) / b
        rho = pd.Series(rec[m]).corr(pd.Series(y[m]), method="spearman")
        ax2.text(0.03, 0.12 if name == "non-experts" else 0.05,
                 f"{name}: −{100 * decay:.0f}% per recording, half-life {half:.1f} recordings, Spearman ρ = {rho:.2f}",
                 transform=ax2.transAxes, fontsize=8.3, color=col)
    ax2.set_yscale("log")
    ax2.set_yticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_ylim(1.2, 70)
    ax2.set_xticks(range(1, 9))
    ax2.set_xlabel("Recording number (1–8)")
    ax2.set_ylabel("Manual share (%), log scale")
    ax2.set_title("B. Straight lines on a log scale = steady exponential decay")
    fig.suptitle("Figure 1. Manual use is a learning curve across the ten recordings", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=13)
    footer(fig, f"Source: manual_share_by_task_repetition.csv → chronological_manual_curve.csv. Participant means, {COHORT}. Fit: log-linear least squares, recordings 1–8.")

    data = cur.copy()
    for name, (a, b) in fits.items():
        data[f"fit_{name.replace('-', '_')}"] = np.where(rec <= 8, np.exp(a + b * rec), np.nan)
    save(
        fig, "fig01_manual_learning_curve",
        "Manual use is a learning curve",
        "Manual dependence halves about every three recordings; non-experts reach the experts' first-recording level only at recording 8; recordings 9–10 break the curve.",
        f"Non-experts 52.0% → 10.0% over recordings 1–8 (ρ = −0.98), experts 11.5% → 3.3% (ρ = −0.76). "
        f"Decay per recording −{100 * (1 - np.exp(fits['non-expert'][1])):.0f}% (non-experts) and −{100 * (1 - np.exp(fits['expert'][1])):.0f}% (experts). "
        "Recording 8 → 9: 10.0% → 0.5% (20-fold) for non-experts.",
        data, ["manual_share_by_task_repetition.csv", "chronological_manual_curve.csv", "manual_learning_fit.csv"],
    )


# ---------------------------------------------------------------- figure 2
def fig_within_between() -> None:
    d = pd.read_csv(DER / "manual_change_within_vs_between_pairs.csv")
    d["within"] = d["kind"].str.startswith("within")
    d["change_pct"] = 100 * d["relative_change"]
    d["label"] = d["transition"].str.replace(r" \(rec (\d+)->(\d+)\)", r"\n(rec \1→\2)", regex=True).str.replace("->", "→")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.25, 1.25, 0.8]})
    for ax, grp, title in [(axes[0], "non-expert", "A. Non-experts"), (axes[1], "expert", "B. Experts")]:
        g = d[d["group"] == grp].reset_index(drop=True)
        colours = [DIS if w else "#BDBDBD" for w in g["within"]]
        vals = g["change_pct"].clip(upper=60)
        ax.bar(range(len(g)), vals, color=colours, edgecolor="white")
        for i, (v, raw) in enumerate(zip(vals, g["change_pct"])):
            txt = f"{raw:+.0f}%" + (" (clipped)" if raw > 60 else "")
            ax.text(i, v + (2 if v >= 0 else -2), txt, ha="center", va="bottom" if v >= 0 else "top", fontsize=9,
                    fontweight="bold" if g.loc[i, "within"] else "normal")
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_xticks(range(len(g)), g["label"], fontsize=8)
        ax.set_ylim(-60, 68)
        ax.set_ylabel("Change in manual share to the next recording (%)")
        ax.set_title(title)
    axes[0].legend(
        [plt.Rectangle((0, 0), 1, 1, fc=DIS), plt.Rectangle((0, 0), 1, 1, fc="#BDBDBD")],
        ["within a pair: disassembly → its assembly", "between pairs: assembly → next disassembly"],
        loc="lower left", frameon=False, fontsize=8,
    )

    ax = axes[2]
    rows = []
    for gi, (grp, col) in enumerate([("non-expert", NONEXPERT), ("expert", EXPERT)]):
        for ki, within in enumerate([True, False]):
            v = d[(d["group"] == grp) & (d["within"] == within)]["change_pct"].to_numpy()
            x = gi * 2.6 + ki
            ax.scatter(np.full(len(v), x) + np.linspace(-0.12, 0.12, len(v)), np.clip(v, -60, 60), color=col, s=36, zorder=3,
                       marker="o" if within else "D", alpha=0.9)
            med = float(np.median(v))
            ax.hlines(med, x - 0.3, x + 0.3, color="#222", lw=2)
            ax.text(x, 64, f"median\n{med:+.0f}%", ha="center", fontsize=8)
            rows.append({"group": grp, "kind": "within pair" if within else "between pairs", "n": len(v),
                         "median_change_pct": med, "mean_change_pct": float(np.mean(v)),
                         "n_drops_over_20pct": int((v <= -20).sum())})
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks([0, 1, 2.6, 3.6], ["within", "between", "within", "between"])
    ax.text(0.5, -75, "non-experts", ha="center", color=NONEXPERT, fontweight="bold")
    ax.text(3.1, -75, "experts", ha="center", color=EXPERT, fontweight="bold")
    ax.set_ylim(-62, 76)
    ax.set_title("C. Summary")
    fig.suptitle("Figure 2. Non-experts drop the manual inside each disassembly → assembly pair, not between pairs", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=13)
    footer(fig, "Source: manual_share_by_task_repetition.csv → manual_change_within_vs_between_pairs.csv (recordings 1–8). Change = (next − current) / current, participant means.")
    save(
        fig, "fig02_within_vs_between_pairs",
        "Learning happens within each pair",
        "Non-experts cut manual use mostly between a disassembly and the assembly that follows it, not between repetitions.",
        "Non-experts: all 4 within-pair transitions drop 20–41% (median −30%); between-pair transitions change +1%, −6%, −12% (median −6%). "
        "Experts: within-pair median −30% at a low base level; between-pair changes are noisy (−48%, +159%, −38%).",
        pd.concat([d.drop(columns=["label"]), pd.DataFrame(rows)], ignore_index=True),
        ["manual_share_by_task_repetition.csv", "manual_change_within_vs_between_pairs.csv"],
    )


# ---------------------------------------------------------------- figure 3
def fig_attention() -> None:
    att = pd.read_csv(SRC / "attention_share_by_task_expertise.csv")
    labels = ["angle grinder", "screwdriver", "wrench", "storage box", "instruction manual"]
    groups = [("Expert", "Disassembly"), ("Expert", "Assembly"), ("Non-expert", "Disassembly"), ("Non-expert", "Assembly")]

    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    for yi, (ex, task) in enumerate(groups):
        left = 0.0
        for lab in labels:
            v = float(att[(att.expertise == ex) & (att.task == task) & (att.label == lab)]["mean"].iloc[0])
            ax.barh(yi, v, left=left, color=LABEL_COLOURS[lab], edgecolor="white", height=0.7)
            if v >= 0.03:
                ax.text(left + v / 2, yi, f"{100 * v:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            left += v
    ax.set_yticks(range(4), [f"{e}s · {t}" for e, t in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=LABEL_COLOURS[l]) for l in labels], labels, ncol=5, frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.set_title("A. Where labelled gaze goes", pad=26)

    ax2 = fig.add_subplot(gs[1, 0])
    de = pd.read_csv(DER / "direction_effect.csv")
    order = ["angle grinder", "screwdriver", "storage box", "instruction manual"]
    y = np.arange(len(order))
    for off, ex, col in [(-0.18, "Expert", EXPERT), (0.18, "Non-expert", NONEXPERT)]:
        v = [float(de[(de.expertise == ex) & (de.label == l)]["assembly_minus_disassembly_pp"].iloc[0]) for l in order]
        ax2.barh(y + off, v, height=0.34, color=col, label=f"{ex}s")
        for yy, vv in zip(y + off, v):
            ax2.text(vv + (0.3 if vv >= 0 else -0.3), yy, f"{vv:+.1f}", va="center", ha="left" if vv >= 0 else "right", fontsize=8.5)
    ax2.axvline(0, color="#444", lw=0.8)
    ax2.set_yticks(y, order)
    ax2.invert_yaxis()
    ax2.set_xlim(-13, 14)
    ax2.set_xlabel("Assembly − disassembly (percentage points)")
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax2.set_title("B. Direction: assembly is workpiece-led,\n    disassembly is tool-led")

    ax3 = fig.add_subplot(gs[1, 1])
    eg = pd.read_csv(DER / "expertise_gap.csv")
    order3 = ["instruction manual", "storage box", "screwdriver", "angle grinder"]
    y = np.arange(len(order3))
    for off, task, col in [(-0.18, "Disassembly", DIS), (0.18, "Assembly", ASM)]:
        v = [float(eg[(eg.task == task) & (eg.label == l)]["expert_minus_non_expert_pp"].iloc[0]) for l in order3]
        ax3.barh(y + off, v, height=0.34, color=col, label=task)
        for yy, vv in zip(y + off, v):
            ax3.text(vv + (0.4 if vv >= 0 else -0.4), yy, f"{vv:+.1f}", va="center", ha="left" if vv >= 0 else "right", fontsize=8.5)
    ax3.axvline(0, color="#444", lw=0.8)
    ax3.set_yticks(y, order3)
    ax3.invert_yaxis()
    ax3.set_xlim(-24, 14)
    ax3.set_xlabel("Expert − non-expert (percentage points)")
    ax3.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax3.set_title("C. Expertise: experts look at boxes and tools\n    where non-experts look at the manual")
    fig.suptitle("Figure 3. Attention signatures of direction and expertise", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=13)
    footer(fig, f"Source: attention_share_by_task_expertise.csv → direction_effect.csv, expertise_gap.csv. Share of labelled fixation time, participant means, {COHORT}.")
    save(
        fig, "fig03_attention_direction_expertise",
        "Attention signatures of direction and expertise",
        "Assembly shifts gaze to the grinder, disassembly to the screwdriver; experts replace manual reading with box and tool gaze.",
        "Assembly − disassembly: grinder +10.3 / +9.0 pp, screwdriver −9.1 / −6.7 pp (experts / non-experts). "
        "Expert − non-expert in disassembly: manual −19.5 pp, box +9.7 pp, screwdriver +9.5 pp, grinder +0.3 pp.",
        att, ["attention_share_by_task_expertise.csv", "direction_effect.csv", "expertise_gap.csv"],
    )


# ---------------------------------------------------------------- figure 4
def fig_budget() -> None:
    b = pd.read_csv(DER / "attention_budget.csv")
    att = pd.read_csv(SRC / "attention_share_by_task_expertise.csv")
    groups = [("Expert", "Disassembly"), ("Non-expert", "Disassembly"), ("Expert", "Assembly"), ("Non-expert", "Assembly")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    for i, (ex, task) in enumerate(groups):
        exe = float(b[(b.expertise == ex) & (b.task == task) & (b.budget.str.startswith("execution"))]["share"].iloc[0])
        man = float(b[(b.expertise == ex) & (b.task == task) & (b.budget == "manual only")]["share"].iloc[0])
        gui = 1 - exe
        box = gui - man
        ax1.bar(i, exe, color="#5B7083", width=0.62)
        ax1.bar(i, box, bottom=exe, color=LABEL_COLOURS["storage box"], width=0.62)
        ax1.bar(i, man, bottom=exe + box, color=LABEL_COLOURS["instruction manual"], width=0.62)
        ax1.text(i, exe / 2, f"execution\n{100 * exe:.0f}%", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        ax1.text(i, exe + box / 2, f"box {100 * box:.0f}%", ha="center", va="center", color="white", fontsize=8)
        if man > 0.03:
            ax1.text(i, exe + box + man / 2, f"manual {100 * man:.0f}%", ha="center", va="center", color="white", fontsize=8)
    ax1.set_xticks(range(4), [f"{e}s\n{t}" for e, t in groups], fontsize=8.5)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_ylim(0, 1)
    ax1.set_title("A. Execution (grinder + tools) vs guidance (box + manual)", pad=12)

    for ex, col in [("Expert", EXPERT), ("Non-expert", NONEXPERT)]:
        v = [float(att[(att.expertise == ex) & (att.task == t) & (att.label == "storage box")]["mean"].iloc[0]) for t in ["Disassembly", "Assembly"]]
        ax2.plot([0, 1], [100 * x for x in v], "-o", color=col, lw=2.2, ms=8, label=f"{ex}s")
        for xi, vv in enumerate(v):
            ax2.text(xi + (0.05 if xi else -0.05), 100 * vv, f"{100 * vv:.1f}%", ha="left" if xi else "right", va="center", color=col, fontsize=9)
    ax2.set_xticks([0, 1], ["Disassembly\n(parts go into boxes)", "Assembly\n(parts come out of boxes)"])
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(10, 28)
    ax2.set_ylabel("Storage-box share of labelled gaze (%)")
    ax2.legend(frameon=False, loc="lower right")
    ax2.set_title("B. Experts keep a stable box routine;\n    non-experts sort less and search more")
    fig.suptitle("Figure 4. Experts do, non-experts read", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=13)
    footer(fig, f"Source: attention_budget.csv, attention_share_by_task_expertise.csv. Participant means, {COHORT}.")
    save(
        fig, "fig04_attention_budget_box_protocol",
        "Experts do, non-experts read",
        "Experts spend ~70% of labelled gaze on execution in both directions; non-experts 60–63%. Experts' box share is flat; non-experts' rises in assembly.",
        "Execution share: experts 70.2% / 71.4%, non-experts 60.5% / 63.1% (disassembly / assembly). "
        "Box share: experts 24.3% → 24.6%, non-experts 14.6% → 20.0%.",
        b, ["attention_budget.csv", "attention_share_by_task_expertise.csv"],
    )


# ---------------------------------------------------------------- figure 5
def fig_routes() -> None:
    r = pd.read_csv(DER / "routes_mapped_to_manual.csv")
    fig = plt.figure(figsize=(17, 7.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.7, 1.3, 1.3], wspace=0.75)

    ax = fig.add_subplot(gs[0, 0])
    occ = {t: {p: int(r[(r.task == t) & (r.role_pattern == p)]["occurrences"].iloc[0]) for p in ["T>W>S", "S>W>T"]} for t in ["Disassembly", "Assembly"]}
    x = np.arange(2)
    ax.bar(x - 0.19, [occ[t]["T>W>S"] for t in ["Disassembly", "Assembly"]], 0.36, color="#9E6B2F", label="tool → grinder → box\n(unscrew → remove → deposit)")
    ax.bar(x + 0.19, [occ[t]["S>W>T"] for t in ["Disassembly", "Assembly"]], 0.36, color="#E1B169", label="box → grinder → tool\n(fetch → place → fasten)")
    for xi, t in enumerate(["Disassembly", "Assembly"]):
        a, b = occ[t]["T>W>S"], occ[t]["S>W>T"]
        ax.text(xi - 0.19, a + 4, str(a), ha="center", fontsize=9)
        ax.text(xi + 0.19, b + 4, str(b), ha="center", fontsize=9)
        lead = f"{a / b:.2f} : 1" if a > b else f"1 : {b / a:.2f}"
        ax.text(xi, max(a, b) + 26, lead, ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x, ["Disassembly", "Assembly"])
    ax.set_ylim(0, 330)
    ax.set_ylabel("Occurrences in the corpus (180 recordings per direction)")
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax.set_title("A. The screw route reverses\n    with the task")

    for k, (task, col) in enumerate([("Disassembly", DIS), ("Assembly", ASM)]):
        ax = fig.add_subplot(gs[0, k + 1])
        t = r[r.task == task].sort_values("occurrences_per_recording").reset_index(drop=True)
        ax.barh(range(len(t)), t["occurrences_per_recording"], color=col, alpha=0.85)
        for i, row in t.iterrows():
            ax.text(row["occurrences_per_recording"] + 0.04, i,
                    f"{row['role_pattern'].replace('>', '→')}  {row['occurrences_per_recording']:.2f}/rec\n"
                    f"{100 * row['share_of_recordings']:.0f}% rec. · {row['participants']}/36 part.",
                    va="center", fontsize=7.6)
        ax.set_yticks(range(len(t)), [wrap(s.replace("->", "→"), 26) for s in t["action_reading"]], fontsize=7.8)
        ax.set_xlim(0, max(t["occurrences_per_recording"]) * 2.9)
        ax.set_xlabel("Occurrences per recording")
        highlight = "W>S>C" if task == "Disassembly" else "S>W>S"
        hi = t.index[t.role_pattern == highlight]
        if len(hi):
            ax.get_yticklabels()[hi[0]].set_fontweight("bold")
            ax.barh(hi[0], t.loc[hi[0], "occurrences_per_recording"], color="none", edgecolor="#222", lw=1.5)
        ax.set_title(f"{'B' if k == 0 else 'C'}. Top routes in {task.lower()}")
    fig.text(0.33, 0.02, "W = grinder, S = storage box, T = tool, C = manual. Bold outline: deposit → read (step-boundary marker) and fetch → fit → fetch (assembly retrieval loop).",
             fontsize=8, color="#444")
    fig.suptitle("Figure 5. Gaze routes follow the manual's action order", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=13)
    footer(fig, "Source: top_routes_by_direction.csv → routes_mapped_to_manual.csv. Three-object routes; participants = number of the 36 who show the route at least once.")
    save(
        fig, "fig05_routes_action_grammar",
        "Gaze routes follow the manual's action order",
        "Unscrew → remove → deposit dominates disassembly; fetch → place → fasten dominates assembly; deposit → read marks disassembly step boundaries.",
        "T→W→S : S→W→T = 160 : 120 in disassembly and 140 : 215 in assembly. Fetch–fit–fetch (S→W→S) 2.68 per assembly recording vs 1.17 per disassembly recording (2.3×), in 81% of assembly recordings. "
        "Deposit → read (W→S→C) in 75 of 180 disassembly recordings (42%) and 30 of 36 participants; not among the top assembly routes.",
        r, ["top_routes_by_direction.csv", "routes_mapped_to_manual.csv"],
    )


def wrap(text: str, width: int) -> str:
    words, lines, cur = str(text).split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    lines.append(cur)
    return "\n".join(lines)


# ---------------------------------------------------------------- figure 6
def fig_wrench() -> None:
    w = pd.read_csv(DER / "wrench_anchor_timing_excerpt.csv")
    att = pd.read_csv(SRC / "attention_share_by_task_expertise.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), gridspec_kw={"width_ratios": [0.9, 1.3, 0.8]})

    ax = axes[0]
    parts = sorted(w.participant.unique())
    cmap = plt.get_cmap("tab10")
    pcol = {p: cmap(i) for i, p in enumerate(parts)}
    for xi, step in enumerate(["D6", "A3"]):
        s = w[w.candidate_manual_step == step].reset_index(drop=True)
        jitter = np.linspace(-0.16, 0.16, len(s))
        ax.scatter(xi + jitter, s["first_wrench_gaze_s"], c=[pcol[p] for p in s.participant], s=46, zorder=3, edgecolor="white")
        med = s["first_wrench_gaze_s"].median()
        ax.hlines(med, xi - 0.3, xi + 0.3, color="#222", lw=2.4)
        ax.text(xi + 0.33, med, f"median\n{med:.0f} s", va="center", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1], ["D6 wrench\n(step 6 of 8,\ndisassembly)", "A3 wrench\n(step 3 of 8,\nassembly)"])
    ax.set_xlim(-0.5, 1.75)
    ax.set_ylabel("First wrench gaze (s from recording start)")
    ax.set_title("A. Step 3 of assembly comes later\n    than step 6 of disassembly")

    ax = axes[1]
    for p in parts:
        s = w[w.participant == p].sort_values("rec")
        ax.plot(s.rec, s.first_wrench_gaze_s, "-", color=pcol[p], lw=1, alpha=0.6)
        for _, row in s.iterrows():
            ax.scatter(row.rec, row.first_wrench_gaze_s, color=pcol[p], marker="o" if row.task == "Disassembly" else "s", s=48, zorder=3, edgecolor="white")
    ax.annotate("AM07: first assembly 608 s,\nthird assembly 167 s", xy=(2, 607.8), xytext=(3.4, 520), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Recording number (● disassembly, ■ assembly)")
    ax.set_ylabel("First wrench gaze (s)")
    ax.legend([plt.Line2D([], [], color=pcol[p], marker="o", lw=1) for p in parts], parts, fontsize=7.5, frameon=False, ncol=1, loc="upper right")
    ax.set_title("B. Per participant across recordings")

    ax = axes[2]
    cov = {"Disassembly": 61 / 180, "Assembly": 54 / 180}
    ax.bar([0, 1], [100 * cov["Disassembly"], 100 * cov["Assembly"]], color=[DIS, ASM], width=0.55)
    ax.bar([0, 1], [100 - 100 * cov["Disassembly"], 100 - 100 * cov["Assembly"]], bottom=[100 * cov["Disassembly"], 100 * cov["Assembly"]], color=LIGHT, width=0.55)
    ax.text(0, 100 * cov["Disassembly"] / 2, "61 / 180\nwith wrench\ngaze", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax.text(1, 100 * cov["Assembly"] / 2, "54 / 180\nwith wrench\ngaze", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax.text(0.5, 80, "no wrench gaze although\nthe step is in every recording", ha="center", fontsize=8.5, color="#555")
    wr = att[att.label == "wrench"]
    ax.text(0.5, 62, "wrench share of labelled gaze:\n" + "\n".join(f"{r.expertise}s, {r.task.lower()}: {100 * r['mean']:.1f}%" for _, r in wr.iterrows()),
            ha="center", va="top", fontsize=7.8, color="#444")
    ax.set_xticks([0, 1], ["Disassembly", "Assembly"])
    ax.set_ylabel("Recordings (%)")
    ax.set_ylim(0, 100)
    ax.set_title("C. The wrench is rarely looked at")
    fig.suptitle("Figure 6. Timing anchor: the wrench step (D6 / A3)", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=13)
    footer(fig, "Source: wrench_intervals_report_excerpt.csv → wrench_anchor_timing_excerpt.csv (16 recordings, 5 participants: the excerpt printed in the report); coverage 61/54 of 180 and shares from the report. Indicative, not a full-corpus estimate.")
    save(
        fig, "fig06_wrench_anchor_timing",
        "Timing anchor: the wrench step",
        "Assembly has a slow front end (A1–A2); tool steps are watched at the target, so the wrench anchors only a third of recordings.",
        f"Median first wrench gaze: D6 {w[w.candidate_manual_step == 'D6'].first_wrench_gaze_s.median():.0f} s (n = 8), "
        f"A3 {w[w.candidate_manual_step == 'A3'].first_wrench_gaze_s.median():.0f} s (n = 8). Wrench gaze in 61/180 disassembly and 54/180 assembly recordings; 0.3–0.7% of labelled gaze.",
        w, ["wrench_intervals_report_excerpt.csv", "wrench_anchor_timing_excerpt.csv", "attention_share_by_task_expertise.csv"],
    )


# ---------------------------------------------------------------- figure 7
def fig_recognisability() -> None:
    s = pd.read_csv(DER / "step_discriminability_model.csv")
    s5 = s[s.window_labels == 5]
    order = [f"D{i}" for i in range(1, 9)] + [f"A{i}" for i in range(1, 9)]
    cols = [("current", "phase_accuracy", "phase\ntoday's labels"), ("current", "step_accuracy", "exact step\ntoday's labels"),
            ("extended", "phase_accuracy", "phase\nstep-aware labels"), ("extended", "step_accuracy", "exact step\nstep-aware labels")]
    mat = np.array([[float(s5[(s5.step_id == st) & (s5.vocabulary == v)][c].iloc[0]) for v, c, _ in cols] for st in order])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.45})
    im = ax1.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax1.text(j, i, f"{100 * mat[i, j]:.0f}", ha="center", va="center", fontsize=9,
                     color="black" if 0.25 < mat[i, j] < 0.8 else "white", fontweight="bold")
    phase = {st: s5[s5.step_id == st].phase.iloc[0] for st in order}
    ax1.set_yticks(range(len(order)), [f"{st}  ({phase[st].replace('_', ' ')})" for st in order], fontsize=8.8)
    ax1.set_xticks(range(4), [c for _, _, c in cols], fontsize=8.5)
    ax1.xaxis.tick_top()
    ax1.axhline(7.5, color="white", lw=4)
    ax1.axvline(1.5, color="white", lw=4)
    fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02, label="correct (%)")
    ax1.set_title("A. Per step, 5-label windows (chance 12.5%)", pad=36)

    agg = s.groupby(["task", "vocabulary", "window_labels"])[["step_accuracy", "phase_accuracy"]].mean().reset_index()
    for task, col in [("disassembly", DIS), ("assembly", ASM)]:
        for voc, ls in [("current", "--"), ("extended", "-")]:
            a = agg[(agg.task == task) & (agg.vocabulary == voc)].sort_values("window_labels")
            ax2.plot(a.window_labels, 100 * a.phase_accuracy, ls, color=col, marker="s", lw=1.6, alpha=0.55)
            ax2.plot(a.window_labels, 100 * a.step_accuracy, ls, color=col, marker="o", lw=2.2)
            ax2.text(10.2, 100 * a.step_accuracy.iloc[-1], f"{task[:3]}. step, {'step-aware' if voc == 'extended' else 'today'}", fontsize=7.8, va="center", color=col)
            ax2.text(10.2, 100 * a.phase_accuracy.iloc[-1], f"{task[:3]}. phase, {'step-aware' if voc == 'extended' else 'today'}", fontsize=7.4, va="center", color=col, alpha=0.75)
    ax2.axhline(12.5, color=GREY, ls=":")
    ax2.text(3, 14, "chance (8 steps)", fontsize=8, color=GREY)
    ax2.set_xticks([3, 5, 10])
    ax2.set_xlim(2.5, 13.8)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Window length (labelled fixations)")
    ax2.set_ylabel("Mean correct over the 8 steps (%)")
    ax2.set_title("B. Longer windows and step-aware labels help\n    (● exact step, ■ phase; dashed = today's labels)")
    fig.suptitle("Figure 7. How well can a gaze window name the manual step?", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=13)
    footer(fig, "Source: step_discriminability_model.csv — windows simulated from each step's expected gaze profile (configs/manual_steps.yaml), 400 per step. Shows what a label set can separate, not measured accuracy.")
    save(
        fig, "fig07_step_recognisability",
        "How well can gaze name the step?",
        "Today's five labels identify the phase well but merge steps inside the screw block; step-aware labels (colours, side handle, armature, hand) separate most steps.",
        "5-label windows: D3 0% and A6 1% exact (merge with D2 / A7), A8 0%; wrench steps 90–92%. 10-label means: exact step 48–60% today vs 73–77% step-aware; phase 82–86% vs 95–96%.",
        s, ["step_discriminability_model.csv", "configs/manual_steps.yaml"],
    )


# ---------------------------------------------------------------- figure 8
def fig_prediction_examples() -> None:
    from gaze_objects.manual_steps import build_step_model, load_manual_config, predict_from_sequence

    cfg = load_manual_config(ROOT / "configs" / "manual_steps.yaml")
    rules = cfg["symbols"]
    examples = [
        ("disassembly", "screwdriver.tip > angle grinder.body > screwdriver.handle > storage box > angle grinder.side_handle_zone",
         "A. Route the earlier report called \"step two\""),
        ("assembly", "screwdriver > angle grinder > storage box > screwdriver", "B. Assembly window, today's labels"),
        ("assembly", "red screwdriver.tip > angle grinder > red box > red screwdriver.tip", "C. Same window with colour labels"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    rows = []
    for ax, (task, seq, title) in zip(axes, examples):
        model = build_step_model(cfg, task)
        p = predict_from_sequence(model, [t.strip() for t in seq.split(">")], rules)
        p = p.set_index("step_id").reindex([f"{task[0].upper()}{i}" for i in range(1, 9)]).reset_index()
        colours = [PHASE_COLOURS.get(ph, LIGHT) for ph in p.phase]
        ax.bar(p.step_id, 100 * p.p_step_bag_of_objects, color=colours, edgecolor="#555")
        for i, v in enumerate(p.p_step_bag_of_objects):
            if v >= 0.01:
                ax.text(i, 100 * v + 1.5, f"{100 * v:.0f}%", ha="center", fontsize=9)
        top_phase = p.loc[p.p_step_bag_of_objects.idxmax(), "phase"]
        ax.text(0.98, 0.95, f"phase '{top_phase.replace('_', ' ')}': {100 * p.p_phase.max():.1f}%", transform=ax.transAxes,
                ha="right", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Probability of step (%)")
        ax.set_title(title)
        ax.text(0.0, -0.2, wrap(seq.replace(" > ", " → "), 62), transform=ax.transAxes, fontsize=7.6, color="#333", va="top", family="DejaVu Sans Mono")
        for _, r in p.iterrows():
            rows.append({"example": title, "task": task, "sequence": seq, "step_id": r.step_id, "phase": r.phase,
                         "p_step": r.p_step_bag_of_objects, "p_phase": r.p_phase})
    fig.subplots_adjust(bottom=0.28, wspace=0.3)
    fig.suptitle("Figure 8. Predicting the manual step from a gaze sequence", x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=13)
    footer(fig, "Model: manual step profiles (configs/manual_steps.yaml), posterior over steps from the objects in the window. Sequence A is a real route from five_label_route_occurrences.csv (NG06NG25_07, KJ03JM25_10).")
    save(
        fig, "fig08_step_prediction_examples",
        "Predicting the step from a sequence",
        "Today's labels pin the phase almost certainly but spread the probability over the steps of the block; colour labels resolve the step.",
        "A: unscrew block 99.8%, D3 39% vs D2 37%. B: screw block certain, step spread over A4–A7. C: A4 91%.",
        pd.DataFrame(rows), ["five_label_route_occurrences.csv", "configs/manual_steps.yaml"],
    )


# ---------------------------------------------------------------- outputs
def write_pdf() -> None:
    with PdfPages(OUT / "manual_step_evidence_figures.pdf") as pdf:
        for f in FIGURES:
            pdf.savefig(f["fig"], bbox_inches="tight", facecolor="white")


def write_html() -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Gaze results mapped to the instruction manual — evidence</title>",
        "<style>body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#222;line-height:1.45}"
        "h1{font-size:26px;margin-bottom:4px}h2{font-size:19px;margin-top:44px;border-top:1px solid #ddd;padding-top:22px}"
        ".meta{color:#666}.ins{font-size:15px}.proof{background:#f5f7fa;border-left:4px solid #2F6DB5;padding:10px 14px;margin:10px 0}"
        "img{width:100%;border:1px solid #eee;margin:8px 0}table{border-collapse:collapse;font-size:12px;margin:6px 0 4px}"
        "td,th{border:1px solid #ddd;padding:3px 7px;text-align:left}th{background:#f2f2f2}details{margin:6px 0 0}"
        ".src{color:#777;font-size:12px}.toc td{font-size:13px;border:none;padding:3px 10px 3px 0}</style></head><body>",
        "<h1>Gaze results mapped to the instruction manual — figures and evidence</h1>",
        f"<p class='meta'>{COHORT}. Odd recordings = disassembly, even = assembly. Shares are of labelled fixation time, participant means. "
        "Every figure is followed by the exact numbers it plots.</p>",
        "<table class='toc'>",
    ]
    for f in FIGURES:
        parts.append(f"<tr><td><a href='#{f['slug']}'>{html.escape(f['slug'].split('_')[0].replace('fig', 'Figure '))}</a></td>"
                     f"<td>{html.escape(f['title'])}</td></tr>")
    parts.append("</table>")
    for f in FIGURES:
        b64 = base64.b64encode((OUT / f"{f['slug']}.png").read_bytes()).decode()
        table = f["data"].round(4).to_html(index=False, border=0)
        parts += [
            f"<h2 id='{f['slug']}'>{html.escape(f['slug'].split('_')[0].replace('fig', 'Figure '))}. {html.escape(f['title'])}</h2>",
            f"<p class='ins'><b>Insight.</b> {html.escape(f['insight'])}</p>",
            f"<div class='proof'><b>Proof from our data.</b> {html.escape(f['proof'])}</div>",
            f"<img src='data:image/png;base64,{b64}' alt='{html.escape(f['title'])}'>",
            f"<p class='src'>Source tables: {html.escape(', '.join(f['sources']))}</p>",
            f"<details><summary>Data plotted in this figure</summary>{table}</details>",
        ]
    parts.append("</body></html>")
    (OUT / "manual_step_evidence_report.html").write_text("\n".join(parts), encoding="utf-8")


def write_index() -> None:
    rows = [{"figure": f["slug"], "title": f["title"], "insight": f["insight"], "proof": f["proof"],
             "data_csv": f"data/{f['slug']}.csv", "source_tables": "; ".join(f["sources"])} for f in FIGURES]
    pd.DataFrame(rows).to_csv(OUT / "figure_index.csv", index=False)


def main() -> None:
    for build in [fig_manual_process, fig_learning_curve, fig_within_between, fig_attention, fig_budget,
                  fig_routes, fig_wrench, fig_recognisability, fig_prediction_examples]:
        build()
    write_pdf()
    write_html()
    write_index()
    for f in FIGURES:
        plt.close(f["fig"])
    print(f"wrote {len(FIGURES)} figures to {OUT}")


if __name__ == "__main__":
    main()
