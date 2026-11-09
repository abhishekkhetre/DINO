"""Map attended-object episodes onto the steps of the printed instruction manual.

Each recording is one pass through the manual (odd number = disassembly,
even = assembly). Steps follow the manual order, so a recording's labelled
gaze episodes are aligned to the eight steps with a left-to-right hidden
Markov model whose emission priors come from ``configs/manual_steps.yaml``.

Outputs are evidence-based step *estimates*: posteriors, boundary intervals
and per-step behaviour metrics. They become validated step timestamps only
after comparison with human step annotations (``evaluate_against_annotations``).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

ROLE_OF_SYMBOL = {
    "manual": "CONSULT",
    "box": "STORAGE",
    "box_red": "STORAGE",
    "box_blue": "STORAGE",
    "screwdriver": "TOOL",
    "screwdriver_green": "TOOL",
    "screwdriver_red": "TOOL",
    "wrench": "TOOL",
    "grinder": "WORKPIECE",
    "side_handle": "WORKPIECE",
    "inner_part": "WORKPIECE",
    "screw": "WORKPIECE",
    "hand": "HAND",
    "surface": "CONTEXT",
    "other": "CONTEXT",
}

SHARE_GROUPS = {
    "manual_share": {"manual"},
    "tool_share": {"screwdriver", "screwdriver_green", "screwdriver_red", "wrench"},
    "screwdriver_share": {"screwdriver", "screwdriver_green", "screwdriver_red"},
    "wrench_share": {"wrench"},
    "box_share": {"box", "box_red", "box_blue"},
    "workpiece_share": {"grinder", "side_handle", "inner_part", "screw"},
    "hand_share": {"hand"},
}

_REC_RE = re.compile(r"^(?P<participant>.+)_(?P<number>\d{1,3})$")


# ---------------------------------------------------------------------------
# Config / symbols
# ---------------------------------------------------------------------------


def load_manual_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def split_part(label: str) -> tuple[str, str | None]:
    """'screwdriver.tip' -> ('screwdriver', 'tip')."""
    text = str(label).strip().lower()
    if "." in text:
        base, part = text.rsplit(".", 1)
        if base and part and " " not in part:
            return base.strip(), part.strip()
    return text, None


def label_to_symbol(label: Any, symbol_rules: dict[str, list[str]]) -> str | None:
    """Map an attended label to a manual-model symbol (exact match, then substring)."""
    if label is None or (isinstance(label, float) and math.isnan(label)):
        return None
    base, _ = split_part(str(label))
    if not base or base in {"nan", "none"}:
        return None
    for symbol, phrases in symbol_rules.items():
        if base in {p.lower() for p in phrases}:
            return symbol
    for symbol, phrases in symbol_rules.items():
        for phrase in phrases:
            if phrase.lower() in base:
                return symbol
    return "other"


# ---------------------------------------------------------------------------
# Recording index
# ---------------------------------------------------------------------------


def _normalise_expertise(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if text in {"expert", "experts", "e", "1", "true"}:
        return "expert"
    if text in {"non-expert", "nonexpert", "non-experts", "novice", "ne", "0", "false"}:
        return "non-expert"
    return text


def load_participants(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    table = pd.read_csv(path)
    cols = {c.lower(): c for c in table.columns}
    pid_col = cols.get("participant_id") or cols.get("participant") or table.columns[0]
    exp_col = cols.get("expertise") or cols.get("group") or table.columns[1]
    out: dict[str, str] = {}
    for pid, exp in zip(table[pid_col], table[exp_col]):
        norm = _normalise_expertise(exp)
        if norm:
            out[str(pid).strip()] = norm
    return out


def build_recording_index(
    recording_ids: Iterable[str],
    *,
    expertise: dict[str, str] | None = None,
    zero_based_policy: str = "exclude",
) -> pd.DataFrame:
    """Participant / recording number / task direction / repetition per recording."""
    rows: list[dict[str, Any]] = []
    for rid in sorted(set(map(str, recording_ids))):
        m = _REC_RE.match(rid)
        if not m:
            rows.append({"recording_id": rid, "participant_id": None, "recording_number": None})
            continue
        rows.append(
            {
                "recording_id": rid,
                "participant_id": m.group("participant"),
                "recording_number": int(m.group("number")),
            }
        )
    index = pd.DataFrame(rows)
    if index.empty:
        return index
    zero_based = set(
        index.loc[index["recording_number"] == 0, "participant_id"].dropna().astype(str)
    )
    tasks, reps, notes, included = [], [], [], []
    for _, r in index.iterrows():
        num = r["recording_number"]
        pid = r["participant_id"]
        if num is None or pd.isna(num):
            tasks.append(None)
            reps.append(None)
            notes.append("unparsed_recording_id")
            included.append(False)
            continue
        num = int(num)
        if pid in zero_based:
            if zero_based_policy == "shift":
                num = num + 1
                note = "zero_based_shifted"
            else:
                tasks.append(None)
                reps.append(None)
                notes.append("zero_based_numbering_excluded")
                included.append(False)
                continue
        else:
            note = ""
        tasks.append("disassembly" if num % 2 == 1 else "assembly")
        reps.append((num + 1) // 2)
        notes.append(note)
        included.append(True)
    index["task"] = tasks
    index["repetition"] = reps
    index["chronological_order"] = index["recording_number"].where(
        ~index["participant_id"].isin(zero_based), index["recording_number"] + 1
    )
    index["index_note"] = notes
    index["included"] = included
    exp = expertise or {}
    index["expertise"] = index["participant_id"].map(lambda p: exp.get(str(p)) if p else None)
    return index


# ---------------------------------------------------------------------------
# Step model
# ---------------------------------------------------------------------------


@dataclass
class StepModel:
    task: str
    step_ids: list[str]
    phases: list[str]
    base_symbols: list[str]
    colour_children: dict[str, dict[str, str]]
    base_prob: np.ndarray  # K x B
    colour_share: dict[str, np.ndarray]  # parent -> K x n_colours (ordered as children)
    prior_base: np.ndarray
    prior_colour: dict[str, np.ndarray]
    skip_prob: float = 0.10
    max_start_step: int = 2
    end_penalty: float = 2.0
    panel_boost: float = 6.0
    prior_strength: float = 30.0
    # Expected labelled episodes per step. Fixed per corpus so that online
    # (past-only) decoding does not depend on the recording's total length.
    expected_len: float | None = None
    meta: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.step_ids)

    @property
    def symbols(self) -> list[str]:
        out = list(self.base_symbols)
        for children in self.colour_children.values():
            out.extend(children.keys())
        return out

    def log_emission_matrix(self) -> tuple[np.ndarray, dict[str, int]]:
        """K x S log-likelihood of each observable symbol."""
        cols: list[np.ndarray] = []
        names: list[str] = []
        for j, sym in enumerate(self.base_symbols):
            cols.append(self.base_prob[:, j])
            names.append(sym)
        for parent, children in self.colour_children.items():
            pj = self.base_symbols.index(parent)
            shares = self.colour_share[parent]
            for ci, child in enumerate(children):
                cols.append(self.base_prob[:, pj] * shares[:, ci])
                names.append(child)
        mat = np.log(np.clip(np.stack(cols, axis=1), 1e-12, None))
        return mat, {n: i for i, n in enumerate(names)}

    def phase_of(self, k: int) -> str:
        return self.phases[k]


def build_step_model(config: dict[str, Any], task: str) -> StepModel:
    task_cfg = config["tasks"][task]
    steps = task_cfg["steps"]
    defaults = dict(config.get("default_weights", {}))
    colour_children = {
        parent: dict(children) for parent, children in (config.get("colour_children") or {}).items()
    }
    child_names = {c for ch in colour_children.values() for c in ch}
    base_symbols = [s for s in config["symbols"].keys() if s not in child_names]
    if "other" not in base_symbols:
        base_symbols.append("other")

    base = np.zeros((len(steps), len(base_symbols)))
    colour_share: dict[str, np.ndarray] = {}
    for parent, children in colour_children.items():
        colour_share[parent] = np.zeros((len(steps), len(children)))
    for k, step in enumerate(steps):
        w = dict(defaults)
        w.update(step.get("gaze_weights") or {})
        for j, sym in enumerate(base_symbols):
            base[k, j] = float(w.get(sym, defaults.get("other", 0.3)))
        for parent, children in colour_children.items():
            given = (step.get("colours") or {}).get(parent) or {}
            colours = list(children.values())
            vals = np.array([float(given.get(c, 1.0 / len(colours))) for c in colours])
            colour_share[parent][k] = vals / vals.sum() if vals.sum() > 0 else 1.0 / len(colours)
    base = base / base.sum(axis=1, keepdims=True)

    align = config.get("alignment", {}) or {}
    return StepModel(
        task=task,
        step_ids=[s["id"] for s in steps],
        phases=[s.get("phase", s["id"]) for s in steps],
        base_symbols=base_symbols,
        colour_children=colour_children,
        base_prob=base,
        colour_share=colour_share,
        prior_base=base.copy(),
        prior_colour={k: v.copy() for k, v in colour_share.items()},
        skip_prob=float(align.get("skip_prob", 0.10)),
        max_start_step=int(align.get("max_start_step", 2)),
        end_penalty=float(align.get("end_penalty_per_missing_step", 2.0)),
        panel_boost=float(align.get("panel_boost", 6.0)),
        prior_strength=float(align.get("prior_strength", 30.0)),
        expected_len=(
            float(align["expected_episodes_per_step"])
            if isinstance(align.get("expected_episodes_per_step"), (int, float))
            else None
        ),
        meta=steps,
    )


# ---------------------------------------------------------------------------
# HMM inference
# ---------------------------------------------------------------------------


def _lse(stack: list[np.ndarray]) -> np.ndarray:
    arr = np.stack(stack, axis=0)
    m = np.max(arr, axis=0)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = m_safe + np.log(np.sum(np.exp(arr - m_safe), axis=0))
    return np.where(np.isfinite(m), out, -np.inf)


def _shift(arr: np.ndarray, by: int, fill: float = -np.inf) -> np.ndarray:
    """out[k] = arr[k - by] (move mass to higher steps)."""
    out = np.full_like(arr, fill)
    if by < arr.shape[-1]:
        out[..., by:] = arr[..., : arr.shape[-1] - by]
    return out


def _unshift(arr: np.ndarray, by: int, fill: float = -np.inf) -> np.ndarray:
    """out[k] = arr[k + by]."""
    out = np.full_like(arr, fill)
    if by < arr.shape[-1]:
        out[..., : arr.shape[-1] - by] = arr[..., by:]
    return out


def _transition_logs(model: StepModel, n_obs: int) -> tuple[float, float, float]:
    expected_len = model.expected_len if model.expected_len else n_obs / max(1, model.n_steps)
    expected_len = max(1.0, float(expected_len))
    p_move = min(0.5, 1.0 / expected_len)
    p_skip = p_move * model.skip_prob
    p_adv = p_move - p_skip
    p_stay = 1.0 - p_move
    return math.log(p_stay), math.log(max(p_adv, 1e-12)), math.log(max(p_skip, 1e-12))


def emission_logs(
    model: StepModel,
    symbols: list[str],
    panel_steps: list[str | None] | None = None,
) -> np.ndarray:
    mat, col = model.log_emission_matrix()
    other = col.get("other")
    idx = [col.get(s, other) for s in symbols]
    e = mat[:, idx].T.copy()  # N x K
    if panel_steps is not None and model.panel_boost > 1.0:
        boost = math.log(model.panel_boost)
        for t, ps in enumerate(panel_steps):
            if ps and ps in model.step_ids:
                k = model.step_ids.index(ps)
                e[t, k] += boost
                if k > 0:
                    e[t, k - 1] += boost * 0.5
    return e


@dataclass
class Decoded:
    viterbi: np.ndarray
    posterior: np.ndarray  # N x K (offline, uses full recording)
    online: np.ndarray  # N x K (filtered, past-only)
    log_likelihood: float


def decode(
    model: StepModel,
    log_e: np.ndarray,
    *,
    uniform_start: bool = False,
    use_end_penalty: bool = True,
) -> Decoded:
    n, k = log_e.shape
    stay, adv, skip = _transition_logs(model, n)
    if uniform_start:
        log_pi = np.full(k, -math.log(k))
    else:
        start = max(1, min(model.max_start_step, k))
        log_pi = np.full(k, -np.inf)
        log_pi[:start] = -math.log(start)
    if use_end_penalty:
        log_end = -model.end_penalty * (k - 1 - np.arange(k))
    else:
        log_end = np.zeros(k)

    alpha = np.full((n, k), -np.inf)
    alpha[0] = log_pi + log_e[0]
    delta = alpha.copy()
    back = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        prev = alpha[t - 1]
        alpha[t] = _lse([prev + stay, _shift(prev, 1) + adv, _shift(prev, 2) + skip]) + log_e[t]
        pd_ = delta[t - 1]
        cand = np.stack([pd_ + stay, _shift(pd_, 1) + adv, _shift(pd_, 2) + skip], axis=0)
        choice = np.argmax(cand, axis=0)
        delta[t] = cand[choice, np.arange(k)] + log_e[t]
        back[t] = np.arange(k) - choice

    beta = np.full((n, k), -np.inf)
    beta[n - 1] = log_end
    for t in range(n - 2, -1, -1):
        nxt = log_e[t + 1] + beta[t + 1]
        beta[t] = _lse([nxt + stay, _unshift(nxt, 1) + adv, _unshift(nxt, 2) + skip])

    log_z = float(np.logaddexp.reduce(alpha[n - 1] + log_end))
    post = np.exp(alpha + beta - log_z)
    post = post / np.clip(post.sum(axis=1, keepdims=True), 1e-300, None)
    online = np.exp(alpha - np.logaddexp.reduce(alpha, axis=1, keepdims=True))

    path = np.zeros(n, dtype=int)
    path[n - 1] = int(np.argmax(delta[n - 1] + log_end))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return Decoded(viterbi=path, posterior=post, online=online, log_likelihood=log_z)


# ---------------------------------------------------------------------------
# Re-estimation (EM over corpus, or supervised from annotations)
# ---------------------------------------------------------------------------


def _accumulate(
    model: StepModel,
    symbols: list[str],
    weights: np.ndarray,
    base_counts: np.ndarray,
    colour_counts: dict[str, np.ndarray],
) -> None:
    """weights: N x K soft (or one-hot) step responsibilities."""
    base_index = {s: j for j, s in enumerate(model.base_symbols)}
    child_parent = {
        child: (parent, ci)
        for parent, children in model.colour_children.items()
        for ci, child in enumerate(children)
    }
    other_j = base_index["other"]
    for t, sym in enumerate(symbols):
        w = weights[t]
        if sym in child_parent:
            parent, ci = child_parent[sym]
            base_counts[:, base_index[parent]] += w
            colour_counts[parent][:, ci] += w
        else:
            base_counts[:, base_index.get(sym, other_j)] += w


def _update_from_counts(
    model: StepModel, base_counts: np.ndarray, colour_counts: dict[str, np.ndarray]
) -> None:
    a = model.prior_strength
    base = base_counts + a * model.prior_base
    model.base_prob = base / base.sum(axis=1, keepdims=True)
    for parent, counts in colour_counts.items():
        shares = counts + 0.2 * a * model.prior_colour[parent]
        model.colour_share[parent] = shares / shares.sum(axis=1, keepdims=True)


def _empty_counts(model: StepModel) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    base = np.zeros_like(model.base_prob)
    colours = {p: np.zeros_like(v) for p, v in model.colour_share.items()}
    return base, colours


def em_refine(
    model: StepModel,
    recordings: list[tuple[list[str], list[str | None] | None]],
    iterations: int,
) -> list[float]:
    """Soft-EM re-estimation of emissions (transitions fixed). Returns log-likelihoods."""
    history: list[float] = []
    for _ in range(max(0, iterations)):
        base_counts, colour_counts = _empty_counts(model)
        total = 0.0
        for symbols, panels in recordings:
            if not symbols:
                continue
            dec = decode(model, emission_logs(model, symbols, panels))
            total += dec.log_likelihood
            _accumulate(model, symbols, dec.posterior, base_counts, colour_counts)
        _update_from_counts(model, base_counts, colour_counts)
        history.append(total)
    return history


def fit_supervised(
    model: StepModel, labelled: list[tuple[list[str], list[str]]]
) -> StepModel:
    """Re-estimate emissions from episodes with known (annotated) step ids."""
    base_counts, colour_counts = _empty_counts(model)
    for symbols, steps in labelled:
        onehot = np.zeros((len(symbols), model.n_steps))
        for t, sid in enumerate(steps):
            if sid in model.step_ids:
                onehot[t, model.step_ids.index(sid)] = 1.0
        _accumulate(model, symbols, onehot, base_counts, colour_counts)
    _update_from_counts(model, base_counts, colour_counts)
    return model


def clone_model(model: StepModel) -> StepModel:
    return StepModel(
        task=model.task,
        step_ids=list(model.step_ids),
        phases=list(model.phases),
        base_symbols=list(model.base_symbols),
        colour_children={k: dict(v) for k, v in model.colour_children.items()},
        base_prob=model.base_prob.copy(),
        colour_share={k: v.copy() for k, v in model.colour_share.items()},
        prior_base=model.prior_base.copy(),
        prior_colour={k: v.copy() for k, v in model.prior_colour.items()},
        skip_prob=model.skip_prob,
        max_start_step=model.max_start_step,
        end_penalty=model.end_penalty,
        panel_boost=model.panel_boost,
        prior_strength=model.prior_strength,
        expected_len=model.expected_len,
        meta=model.meta,
    )


def emissions_table(model: StepModel, source: str) -> pd.DataFrame:
    mat, col = model.log_emission_matrix()
    rows = []
    for k, sid in enumerate(model.step_ids):
        for sym, j in col.items():
            rows.append(
                {
                    "task": model.task,
                    "step_id": sid,
                    "phase": model.phases[k],
                    "symbol": sym,
                    "probability": float(np.exp(mat[k, j])),
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Window prediction ("which step is this sequence from?")
# ---------------------------------------------------------------------------


def predict_from_sequence(
    model: StepModel, labels: list[str], symbol_rules: dict[str, list[str]]
) -> pd.DataFrame:
    """Posterior over manual steps at the last token of an isolated label sequence."""
    symbols = [label_to_symbol(lab, symbol_rules) or "other" for lab in labels]
    if not symbols:
        raise ValueError("empty sequence")
    log_e = emission_logs(model, symbols)
    dec = decode(model, log_e, uniform_start=True, use_end_penalty=False)
    last = dec.online[-1]
    bag = log_e.sum(axis=0)
    bag = np.exp(bag - np.logaddexp.reduce(bag))
    out = pd.DataFrame(
        {
            "step_id": model.step_ids,
            "phase": model.phases,
            "p_step_bag_of_objects": bag,
            "p_step_ordered_last_token": last,
        }
    )
    out["p_phase"] = out.groupby("phase")["p_step_bag_of_objects"].transform("sum")
    return out.sort_values("p_step_bag_of_objects", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Episode preparation
# ---------------------------------------------------------------------------


def _episode_times(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    for start_col, end_col in (("video_start_s", "video_end_s"), ("recording_start_s", "recording_end_s")):
        if start_col in frame.columns and frame[start_col].notna().any():
            start = pd.to_numeric(frame[start_col], errors="coerce")
            end = pd.to_numeric(frame.get(end_col, start), errors="coerce").fillna(start)
            return start, end
    seq = pd.Series(np.arange(len(frame), dtype=float), index=frame.index)
    return seq, seq


def _episode_duration_s(frame: pd.DataFrame, start: pd.Series, end: pd.Series) -> pd.Series:
    span = (end - start).clip(lower=0)
    if "total_gaze_event_duration_raw" in frame.columns:
        raw = pd.to_numeric(frame["total_gaze_event_duration_raw"], errors="coerce")
        if raw.notna().any():
            # Tobii exports gaze-event durations in milliseconds.
            scale = 1000.0 if raw.median() > 20 else 1.0
            return (raw / scale).fillna(span)
    return span


def prepare_episodes(
    sequences: pd.DataFrame,
    symbol_rules: dict[str, list[str]],
    *,
    label_column: str = "attended_label",
) -> pd.DataFrame:
    frame = sequences.copy()
    if label_column not in frame.columns:
        raise KeyError(f"missing label column {label_column!r}")
    frame = frame[frame[label_column].notna()].copy()
    frame["label"] = frame[label_column].astype(str)
    frame["symbol"] = frame["label"].map(lambda v: label_to_symbol(v, symbol_rules))
    frame["part"] = frame["label"].map(lambda v: split_part(v)[1])
    frame = frame[frame["symbol"].notna()].copy()
    start, end = _episode_times(frame)
    frame["t_start_s"] = start
    frame["t_end_s"] = end
    frame["duration_s"] = _episode_duration_s(frame, start, end)
    frame["role"] = frame["symbol"].map(lambda s: ROLE_OF_SYMBOL.get(s, "CONTEXT"))
    sort_cols = ["recording_id", "t_start_s"]
    if "sequence_id" in frame.columns:
        sort_cols.append("sequence_id")
    return frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-recording outputs
# ---------------------------------------------------------------------------


def _decode_recording(
    model: StepModel, rec: pd.DataFrame, panel_col: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = rec["symbol"].tolist()
    panels = rec[panel_col].tolist() if panel_col and panel_col in rec.columns else None
    dec = decode(model, emission_logs(model, symbols, panels))
    k_ids = model.step_ids
    out = rec.copy()
    out["step_id"] = [k_ids[i] for i in dec.viterbi]
    out["phase"] = [model.phases[i] for i in dec.viterbi]
    out["step_confidence"] = dec.posterior[np.arange(len(rec)), dec.viterbi]
    phase_names = list(dict.fromkeys(model.phases))
    phase_post = np.stack(
        [dec.posterior[:, [i for i, p in enumerate(model.phases) if p == ph]].sum(axis=1) for ph in phase_names],
        axis=1,
    )
    out["phase_confidence"] = [
        phase_post[t, phase_names.index(model.phases[i])] for t, i in enumerate(dec.viterbi)
    ]
    online_arg = dec.online.argmax(axis=1)
    out["online_step_id"] = [k_ids[i] for i in online_arg]
    out["online_step_prob"] = dec.online.max(axis=1)
    online_phase = np.stack(
        [dec.online[:, [i for i, p in enumerate(model.phases) if p == ph]].sum(axis=1) for ph in phase_names],
        axis=1,
    )
    out["online_phase"] = [phase_names[i] for i in online_phase.argmax(axis=1)]
    out["online_phase_prob"] = online_phase.max(axis=1)
    for i, sid in enumerate(k_ids):
        out[f"p_{sid}"] = dec.posterior[:, i]
    out["boundary_marker"] = _boundary_markers(rec["role"].tolist())
    timeline = _timeline(model, out, dec.posterior)
    return out, timeline


def _boundary_markers(roles: list[str]) -> list[bool]:
    """Storage followed by manual consultation: candidate 'step finished, read next' cue."""
    marks = [False] * len(roles)
    for t in range(1, len(roles)):
        if roles[t] == "CONSULT" and roles[t - 1] == "STORAGE":
            marks[t] = True
    return marks


def _timeline(model: StepModel, ep: pd.DataFrame, post: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    starts = ep["t_start_s"].to_numpy(dtype=float)
    ends = ep["t_end_s"].to_numpy(dtype=float)
    cum_ge = np.cumsum(post[:, ::-1], axis=1)[:, ::-1]  # P(step >= k)
    for k, sid in enumerate(model.step_ids):
        mask = (ep["step_id"] == sid).to_numpy()
        row: dict[str, Any] = {
            "task": model.task,
            "step_id": sid,
            "step_order": k + 1,
            "phase": model.phases[k],
            "n_episodes": int(mask.sum()),
        }
        if mask.any():
            row["first_gaze_s"] = float(np.nanmin(starts[mask]))
            row["last_gaze_s"] = float(np.nanmax(ends[mask]))
            row["mean_confidence"] = float(post[mask, k].mean())
        else:
            row["first_gaze_s"] = np.nan
            row["last_gaze_s"] = np.nan
            row["mean_confidence"] = np.nan
        if k > 0:
            lo = np.where(cum_ge[:, k] >= 0.1)[0]
            hi = np.where(cum_ge[:, k] >= 0.9)[0]
            row["boundary_lo_s"] = float(starts[lo[0]]) if len(lo) else np.nan
            row["boundary_hi_s"] = float(starts[hi[0]]) if len(hi) else np.nan
        else:
            row["boundary_lo_s"] = float(np.nanmin(starts)) if len(starts) else np.nan
            row["boundary_hi_s"] = row["boundary_lo_s"]
        rows.append(row)
    tl = pd.DataFrame(rows)
    # Estimated step span: boundaries at the midpoint of the unlabelled gap
    # between the last gaze of one step and the first gaze of the next.
    est_start = []
    observed = tl[tl["n_episodes"] > 0]
    obs_idx = observed.index.tolist()
    for pos, i in enumerate(obs_idx):
        if pos == 0:
            est_start.append((i, tl.at[i, "first_gaze_s"]))
        else:
            prev = obs_idx[pos - 1]
            est_start.append((i, 0.5 * (tl.at[prev, "last_gaze_s"] + tl.at[i, "first_gaze_s"])))
    tl["est_start_s"] = np.nan
    tl["est_end_s"] = np.nan
    for pos, (i, s) in enumerate(est_start):
        tl.at[i, "est_start_s"] = s
        tl.at[i, "est_end_s"] = (
            est_start[pos + 1][1] if pos + 1 < len(est_start) else tl.at[i, "last_gaze_s"]
        )
    tl["est_duration_s"] = tl["est_end_s"] - tl["est_start_s"]
    tl["status"] = np.where(
        tl["n_episodes"] == 0,
        "not_observed",
        np.where(tl["mean_confidence"] >= 0.5, "observed", "uncertain"),
    )
    return tl


def _step_metrics(ep: pd.DataFrame) -> dict[str, Any]:
    dur = ep["duration_s"].fillna(0).to_numpy(dtype=float)
    total = float(dur.sum())
    syms = ep["symbol"].tolist()
    out: dict[str, Any] = {"labelled_gaze_s": total, "n_episodes": len(ep)}
    for name, group in SHARE_GROUPS.items():
        sel = np.array([s in group for s in syms])
        out[name] = float(dur[sel].sum() / total) if total > 0 else np.nan
    out["n_manual_lookups"] = int(sum(1 for s in syms if s == "manual"))
    out["n_object_switches"] = int(sum(1 for a, b in zip(syms, syms[1:]) if a != b))
    seen: set[str] = set()
    revisits = 0
    prev = None
    for s in syms:
        if s != prev and s in seen:
            revisits += 1
        seen.add(s)
        prev = s
    out["n_revisits"] = revisits
    out["n_unique_objects"] = len(seen)
    tips = ep["part"].fillna("").eq("tip") & ep["role"].eq("TOOL")
    tool_sel = ep["role"].eq("TOOL").to_numpy()
    tool_dur = float(dur[tool_sel].sum())
    out["tool_tip_fraction"] = float(dur[tips.to_numpy()].sum() / tool_dur) if tool_dur > 0 else np.nan
    return out


def _manual_lookups(ep: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    symbols = ep["symbol"].tolist()
    times = ep["t_start_s"].tolist()
    last_manual_t: float | None = None
    spans = timeline.set_index("step_id")[["est_start_s", "est_end_s"]].to_dict("index")
    for t, r in enumerate(ep.itertuples(index=False)):
        if r.symbol != "manual":
            continue
        span = spans.get(r.step_id, {})
        s0, s1 = span.get("est_start_s", np.nan), span.get("est_end_s", np.nan)
        rel = (r.t_start_s - s0) / (s1 - s0) if pd.notna(s0) and pd.notna(s1) and s1 > s0 else np.nan
        if pd.isna(rel):
            kind = "unknown"
        elif rel < 0.2:
            kind = "step_onset"
        elif rel > 0.8:
            kind = "step_closing"
        else:
            kind = "mid_step"
        rows.append(
            {
                "recording_id": r.recording_id,
                "step_id": r.step_id,
                "t_start_s": r.t_start_s,
                "duration_s": r.duration_s,
                "relative_position_in_step": rel,
                "lookup_type": kind,
                "previous_object": symbols[t - 1] if t > 0 else None,
                "next_object": symbols[t + 1] if t + 1 < len(symbols) else None,
                "seconds_since_previous_lookup": (r.t_start_s - last_manual_t)
                if last_manual_t is not None
                else np.nan,
                "panel_step_read": getattr(r, "manual_panel_step", None),
            }
        )
        last_manual_t = r.t_start_s
    return pd.DataFrame(rows)


def _motifs(ep: pd.DataFrame, index: pd.DataFrame, n: int = 3, top: int = 15) -> pd.DataFrame:
    meta = index.set_index("recording_id")[["participant_id", "task"]]
    counts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for rid, rec in ep.groupby("recording_id", sort=False):
        pid = meta.at[rid, "participant_id"] if rid in meta.index else None
        task = meta.at[rid, "task"] if rid in meta.index else None
        for sid, seg in rec.groupby("step_id", sort=False):
            for level, col in (("object", "symbol"), ("role", "role")):
                toks = seg[col].tolist()
                collapsed = [x for i, x in enumerate(toks) if i == 0 or x != toks[i - 1]]
                for i in range(len(collapsed) - n + 1):
                    gram = " > ".join(collapsed[i : i + n])
                    key = (str(task), str(sid), level, gram)
                    d = counts.setdefault(key, {"occurrences": 0, "recs": set(), "pids": set()})
                    d["occurrences"] += 1
                    d["recs"].add(rid)
                    d["pids"].add(pid)
    rows = [
        {
            "task": k[0],
            "step_id": k[1],
            "level": k[2],
            "motif": k[3],
            "occurrences": v["occurrences"],
            "recordings": len(v["recs"]),
            "participants": len(v["pids"]),
        }
        for k, v in counts.items()
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values(
        ["task", "step_id", "level", "participants", "recordings", "occurrences"],
        ascending=[True, True, True, False, False, False],
    )
    return table.groupby(["task", "step_id", "level"], sort=False).head(top).reset_index(drop=True)


ROLE_LETTER = {"WORKPIECE": "W", "STORAGE": "S", "TOOL": "T", "CONSULT": "C", "HAND": "H", "CONTEXT": "X"}
SIGNATURE_MOTIFS = ["T>W>S", "S>W>T", "W>S>W", "S>W>S", "W>S>C", "W>T>W"]


def direction_signature(episodes: pd.DataFrame) -> pd.DataFrame:
    """Per-recording role-trigram counts and the unscrew/fasten asymmetry index.

    Disassembly is the manual read forwards (unscrew -> remove -> deposit,
    T>W>S); assembly reverses it (fetch -> place -> fasten, S>W>T).
    asymmetry = (T>W>S - S>W>T) / (T>W>S + S>W>T): positive = disassembly-like.
    """
    rows = []
    for rid, rec in episodes.groupby("recording_id", sort=False):
        roles = [ROLE_LETTER.get(r, "X") for r in rec["role"]]
        roles = [r for r in roles if r in "WSTC"]
        collapsed = [r for i, r in enumerate(roles) if i == 0 or r != roles[i - 1]]
        grams = [">".join(collapsed[i : i + 3]) for i in range(len(collapsed) - 2)]
        row: dict[str, Any] = {"recording_id": rid, "task": rec["task"].iloc[0], "n_role_trigrams": len(grams)}
        for motif in SIGNATURE_MOTIFS:
            row[motif] = grams.count(motif)
        a, b = row["T>W>S"], row["S>W>T"]
        row["unscrew_vs_fasten_asymmetry"] = (a - b) / (a + b) if a + b else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank AUC of `scores` for `positive` (ties count half)."""
    pos, neg = scores[positive], scores[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# Group summaries
# ---------------------------------------------------------------------------

SUMMARY_METRICS = [
    "est_duration_s",
    "labelled_gaze_s",
    "manual_share",
    "tool_share",
    "box_share",
    "workpiece_share",
    "hand_share",
    "n_manual_lookups",
    "n_object_switches",
    "n_revisits",
    "mean_confidence",
    "observed",
]


def summarise_groups(step_table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Participant-weighted means: recordings averaged within participant first."""
    data = step_table.copy()
    data["observed"] = (data["status"] == "observed").astype(float)
    metrics = [m for m in SUMMARY_METRICS if m in data.columns]
    per_participant = data.groupby(["participant_id"] + by, dropna=False)[metrics].mean().reset_index()
    agg = per_participant.groupby(by, dropna=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_participants"] = per_participant.groupby(by, dropna=False)["participant_id"].nunique()
    return agg.reset_index()


def expertise_contrast(
    step_table: pd.DataFrame,
    metrics: list[str] | None = None,
    *,
    n_boot: int = 2000,
    seed: int = 7,
) -> pd.DataFrame:
    """Expert minus non-expert per task x step, participant bootstrap 95% CI."""
    metrics = metrics or ["manual_share", "tool_share", "box_share", "est_duration_s", "n_manual_lookups"]
    data = step_table[step_table["expertise"].isin(["expert", "non-expert"])]
    rng = np.random.default_rng(seed)
    rows = []
    for (task, sid), grp in data.groupby(["task", "step_id"]):
        pp = grp.groupby(["participant_id", "expertise"])[metrics].mean().reset_index()
        for m in metrics:
            e = pp.loc[pp["expertise"] == "expert", m].dropna().to_numpy()
            ne = pp.loc[pp["expertise"] == "non-expert", m].dropna().to_numpy()
            if len(e) < 2 or len(ne) < 2:
                continue
            diff = float(e.mean() - ne.mean())
            boots = np.array(
                [
                    rng.choice(e, len(e)).mean() - rng.choice(ne, len(ne)).mean()
                    for _ in range(n_boot)
                ]
            )
            rows.append(
                {
                    "task": task,
                    "step_id": sid,
                    "metric": m,
                    "expert_mean": float(e.mean()),
                    "non_expert_mean": float(ne.mean()),
                    "difference": diff,
                    "ci95_lo": float(np.percentile(boots, 2.5)),
                    "ci95_hi": float(np.percentile(boots, 97.5)),
                    "n_expert": len(e),
                    "n_non_expert": len(ne),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model-based discriminability (what the label vocabulary can separate)
# ---------------------------------------------------------------------------


def step_discriminability(
    model: StepModel,
    vocabulary: list[str] | None,
    *,
    window: int = 5,
    n_samples: int = 400,
    seed: int = 11,
) -> pd.DataFrame:
    """Simulated accuracy of naming the step from an isolated window of `window` labels.

    Windows are sampled from the model's own emissions restricted to
    `vocabulary` (None = every symbol incl. colours). This measures what the
    label set *can* separate under the model, not accuracy on real data.
    """
    rng = np.random.default_rng(seed)
    mat, col = model.log_emission_matrix()
    if vocabulary is None:
        observable = [s for s in col if s not in model.colour_children]
    else:
        observable = [s for s in vocabulary if s in col]
    obs_idx = [col[s] for s in observable]
    probs = np.exp(mat[:, obs_idx])
    probs = probs / probs.sum(axis=1, keepdims=True)
    log_p = np.log(np.clip(probs, 1e-12, None))
    rows = []
    for k, sid in enumerate(model.step_ids):
        correct = 0
        phase_correct = 0
        confusions: dict[str, int] = {}
        for _ in range(n_samples):
            draw = rng.choice(len(observable), size=window, p=probs[k])
            score = log_p[:, draw].sum(axis=1)
            pred = int(np.argmax(score))
            correct += pred == k
            phase_correct += model.phases[pred] == model.phases[k]
            if pred != k:
                confusions[model.step_ids[pred]] = confusions.get(model.step_ids[pred], 0) + 1
        top = max(confusions.items(), key=lambda kv: kv[1])[0] if confusions else None
        rows.append(
            {
                "task": model.task,
                "step_id": sid,
                "phase": model.phases[k],
                "vocabulary": "current" if vocabulary is not None else "extended",
                "window_labels": window,
                "step_accuracy": correct / n_samples,
                "phase_accuracy": phase_correct / n_samples,
                "most_confused_with": top,
                "chance": 1.0 / model.n_steps,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Annotation evaluation
# ---------------------------------------------------------------------------


def assign_true_steps(episodes: pd.DataFrame, annotations: pd.DataFrame) -> pd.Series:
    """Ground-truth step per episode by episode midpoint within annotated intervals."""
    truth = pd.Series([None] * len(episodes), index=episodes.index, dtype=object)
    mids = 0.5 * (episodes["t_start_s"] + episodes["t_end_s"])
    for rid, ann in annotations.groupby("recording_id"):
        sel = episodes["recording_id"] == rid
        for a in ann.itertuples(index=False):
            inside = sel & (mids >= float(a.start_s)) & (mids <= float(a.end_s))
            truth[inside] = str(a.step_id)
    return truth


def evaluate_against_annotations(
    episodes: pd.DataFrame,
    annotations: pd.DataFrame,
    timelines: pd.DataFrame,
    step_phase: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = episodes.copy()
    data["true_step_id"] = assign_true_steps(data, annotations)
    data = data[data["true_step_id"].notna()]
    rows = []
    for task, tdata in data.groupby("task"):
        for sid in sorted(tdata["true_step_id"].unique()):
            pred, true = tdata["step_id"] == sid, tdata["true_step_id"] == sid
            tp = int((pred & true).sum())
            fp = int((pred & ~true).sum())
            fn = int((~pred & true).sum())
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / (tp + fn) if tp + fn else np.nan
            rows.append(
                {
                    "task": task,
                    "step_id": sid,
                    "n_episodes": int(true.sum()),
                    "precision": prec,
                    "recall": rec,
                    "f1": 2 * prec * rec / (prec + rec) if prec and rec else np.nan,
                    "online_recall": float((tdata.loc[true, "online_step_id"] == sid).mean()),
                }
            )
    per_step = pd.DataFrame(rows)
    ann_start = annotations.groupby(["recording_id", "step_id"])["start_s"].min()
    tl = timelines.set_index(["recording_id", "step_id"])
    errs = []
    for key, true_start in ann_start.items():
        if key in tl.index and pd.notna(tl.at[key, "est_start_s"]):
            errs.append(abs(float(tl.at[key, "est_start_s"]) - float(true_start)))
    phase_truth = step_phase or dict(zip(episodes["step_id"], episodes["phase"]))
    summary = {
        "n_annotated_episodes": int(len(data)),
        "episode_accuracy_offline": float((data["step_id"] == data["true_step_id"]).mean()) if len(data) else None,
        "episode_accuracy_online": float((data["online_step_id"] == data["true_step_id"]).mean()) if len(data) else None,
        "phase_accuracy_offline": float(
            (data["phase"] == data["true_step_id"].map(phase_truth)).mean()
        )
        if len(data)
        else None,
        "boundary_abs_error_median_s": float(np.median(errs)) if errs else None,
        "boundary_abs_error_mean_s": float(np.mean(errs)) if errs else None,
        "n_boundaries_compared": len(errs),
    }
    return per_step, summary


def leave_participant_out(
    models: dict[str, StepModel],
    episodes: pd.DataFrame,
    annotations: pd.DataFrame,
    panel_col: str | None = None,
) -> pd.DataFrame:
    """Supervised emissions fitted on other participants, decoded on the held-out one."""
    data = episodes.copy()
    data["true_step_id"] = assign_true_steps(data, annotations)
    annotated = data[data["true_step_id"].notna()]
    rows = []
    for pid in sorted(annotated["participant_id"].dropna().unique()):
        for task, base_model in models.items():
            train = annotated[(annotated["participant_id"] != pid) & (annotated["task"] == task)]
            test_ids = annotated.loc[
                (annotated["participant_id"] == pid) & (annotated["task"] == task), "recording_id"
            ].unique()
            if len(test_ids) == 0 or train.empty:
                continue
            m = clone_model(base_model)
            fit_supervised(
                m,
                [
                    (g["symbol"].tolist(), g["true_step_id"].tolist())
                    for _, g in train.groupby("recording_id", sort=False)
                ],
            )
            for rid in test_ids:
                rec = data[data["recording_id"] == rid]
                panels = rec[panel_col].tolist() if panel_col and panel_col in rec.columns else None
                dec = decode(m, emission_logs(m, rec["symbol"].tolist(), panels))
                pred = np.array([m.step_ids[i] for i in dec.viterbi])
                online = np.array([m.step_ids[i] for i in dec.online.argmax(axis=1)])
                truth = rec["true_step_id"].to_numpy()
                has = pd.notna(truth)
                rows.append(
                    {
                        "held_out_participant": pid,
                        "recording_id": rid,
                        "task": task,
                        "n_annotated": int(has.sum()),
                        "accuracy_offline": float((pred[has] == truth[has]).mean()) if has.any() else np.nan,
                        "accuracy_online": float((online[has] == truth[has]).mean()) if has.any() else np.nan,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Corpus runner
# ---------------------------------------------------------------------------


def run_manual_step_mapping(
    sequences_csv: str | Path,
    out_dir: str | Path,
    *,
    model_config: str | Path = "configs/manual_steps.yaml",
    participants_csv: str | Path | None = None,
    annotations_csv: str | Path | None = None,
    em_iterations: int = 5,
    label_column: str = "attended_label",
    panel_column: str | None = "manual_panel_step",
    panels_root: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = load_manual_config(model_config)
    rules = config["symbols"]
    sequences = pd.read_csv(sequences_csv)
    if panels_root:
        from gaze_objects.manual_panels import attach_panels, load_panel_tables

        sequences = attach_panels(sequences, load_panel_tables(panels_root))
    episodes = prepare_episodes(sequences, rules, label_column=label_column)
    expertise = load_participants(participants_csv)
    policy = (config.get("recording_rule") or {}).get("zero_based_policy", "exclude")
    index = build_recording_index(episodes["recording_id"].unique(), expertise=expertise, zero_based_policy=policy)
    index.to_csv(out / "recording_task_index.csv", index=False)
    keep = index.loc[index["included"], ["recording_id", "participant_id", "task", "repetition", "chronological_order", "expertise"]]
    episodes = episodes.merge(keep, on="recording_id", how="inner")
    panel_col = panel_column if panel_column and panel_column in episodes.columns else None

    models = {task: build_step_model(config, task) for task in config["tasks"]}
    for task, model in models.items():
        if model.expected_len is None:
            lengths = episodes[episodes["task"] == task].groupby("recording_id").size()
            if len(lengths):
                model.expected_len = max(1.0, float(lengths.median()) / model.n_steps)
    em_history: dict[str, list[float]] = {}
    emission_tables = []
    for task, model in models.items():
        emission_tables.append(emissions_table(model, "manual_prior"))
        recs = [
            (g["symbol"].tolist(), g[panel_col].tolist() if panel_col else None)
            for _, g in episodes[episodes["task"] == task].groupby("recording_id", sort=False)
        ]
        em_history[task] = em_refine(model, recs, em_iterations)
        if em_iterations > 0:
            emission_tables.append(emissions_table(model, "corpus_em"))
    pd.concat(emission_tables, ignore_index=True).to_csv(out / "step_emissions.csv", index=False)

    decoded_parts, timeline_parts, metric_rows, lookup_parts = [], [], [], []
    for rid, rec in episodes.groupby("recording_id", sort=False):
        task = rec["task"].iloc[0]
        model = models[task]
        ep, tl = _decode_recording(model, rec.reset_index(drop=True), panel_col)
        tl.insert(0, "recording_id", rid)
        decoded_parts.append(ep)
        timeline_parts.append(tl)
        for sid, seg in ep.groupby("step_id", sort=False):
            row = {"recording_id": rid, "task": task, "step_id": sid}
            row.update(_step_metrics(seg))
            metric_rows.append(row)
        lookup_parts.append(_manual_lookups(ep, tl))

    decoded = pd.concat(decoded_parts, ignore_index=True) if decoded_parts else pd.DataFrame()
    timelines = pd.concat(timeline_parts, ignore_index=True) if timeline_parts else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    lookups = pd.concat([x for x in lookup_parts if len(x)], ignore_index=True) if any(len(x) for x in lookup_parts) else pd.DataFrame()

    decoded.to_csv(out / "step_episode_map.csv", index=False)
    timelines.to_csv(out / "step_timeline.csv", index=False)
    lookups.to_csv(out / "manual_lookups.csv", index=False)

    meta_cols = ["recording_id", "participant_id", "repetition", "chronological_order", "expertise"]
    step_table = timelines.merge(
        metrics.drop(columns=["task", "n_episodes"]), on=["recording_id", "step_id"], how="left"
    ).merge(index[meta_cols], on="recording_id", how="left")
    step_table.to_csv(out / "step_metrics.csv", index=False)

    _motifs(decoded, index).to_csv(out / "step_motifs.csv", index=False)

    summaries: dict[str, Any] = {}
    signature = direction_signature(decoded) if len(decoded) else pd.DataFrame()
    if len(signature):
        signature.to_csv(out / "direction_signature.csv", index=False)
        valid = signature.dropna(subset=["unscrew_vs_fasten_asymmetry"])
        summaries["direction_asymmetry_auc_disassembly"] = _auc(
            valid["unscrew_vs_fasten_asymmetry"].to_numpy(dtype=float),
            (valid["task"] == "disassembly").to_numpy(),
        )
        summaries["direction_asymmetry_n_recordings"] = int(len(valid))
    if len(step_table):
        summarise_groups(step_table, ["task", "step_id"]).to_csv(out / "step_summary_all.csv", index=False)
        summarise_groups(step_table, ["task", "step_id", "repetition"]).to_csv(
            out / "step_summary_by_repetition.csv", index=False
        )
        if step_table["expertise"].notna().any():
            summarise_groups(step_table, ["task", "step_id", "expertise"]).to_csv(
                out / "step_summary_by_expertise.csv", index=False
            )
            summarise_groups(step_table, ["task", "step_id", "expertise", "repetition"]).to_csv(
                out / "step_summary_by_expertise_repetition.csv", index=False
            )
            expertise_contrast(step_table).to_csv(out / "step_expertise_contrast.csv", index=False)
        if len(lookups):
            lk = lookups.merge(index[["recording_id", "task", "expertise"]], on="recording_id", how="left")
            (
                lk.groupby(["task", "expertise", "step_id", "lookup_type"], dropna=False)
                .size()
                .rename("n_lookups")
                .reset_index()
                .to_csv(out / "manual_lookup_summary.csv", index=False)
            )
        markers = decoded.groupby("recording_id")["boundary_marker"].sum()
        summaries["boundary_markers_per_recording_median"] = float(markers.median()) if len(markers) else None

    disc = []
    current = config.get("current_vocabulary")
    for model in models.values():
        prior = clone_model(model)
        prior.base_prob = prior.prior_base.copy()
        prior.colour_share = {k: v.copy() for k, v in prior.prior_colour.items()}
        for window in (3, 5, 10):
            disc.append(step_discriminability(prior, current, window=window))
            disc.append(step_discriminability(prior, None, window=window))
    pd.concat(disc, ignore_index=True).to_csv(out / "step_discriminability_model.csv", index=False)

    eval_summary = None
    if annotations_csv:
        ann = pd.read_csv(annotations_csv)
        step_phase = {
            sid: ph for m in models.values() for sid, ph in zip(m.step_ids, m.phases)
        }
        per_step, eval_summary = evaluate_against_annotations(decoded, ann, timelines, step_phase)
        per_step.to_csv(out / "step_eval_per_step.csv", index=False)
        lpo = leave_participant_out(models, decoded, ann, panel_col)
        lpo.to_csv(out / "step_eval_leave_participant_out.csv", index=False)
        if len(lpo):
            eval_summary["lpo_accuracy_offline_mean"] = float(lpo["accuracy_offline"].mean())
            eval_summary["lpo_accuracy_online_mean"] = float(lpo["accuracy_online"].mean())

    summary = {
        "n_recordings_mapped": int(decoded["recording_id"].nunique()) if len(decoded) else 0,
        "n_recordings_excluded": int((~index["included"]).sum()),
        "n_episodes": int(len(decoded)),
        "tasks": {
            t: int(decoded.loc[decoded["task"] == t, "recording_id"].nunique()) if len(decoded) else 0
            for t in models
        },
        "em_log_likelihood": em_history,
        "panel_evidence_used": bool(panel_col),
        "expertise_available": bool(len(expertise)),
        "step_status_counts": timelines["status"].value_counts().to_dict() if len(timelines) else {},
        "annotation_evaluation": eval_summary,
        **summaries,
        "notes": [
            "Step ids/timestamps are model estimates from gaze evidence, not validated action boundaries.",
            "est_start_s/est_end_s split unlabelled gaps at their midpoint; boundary_lo_s/hi_s give the 10-90% posterior interval.",
            "online_* columns use past episodes only (live 'which step now').",
            "step_discriminability_model.csv is simulated from the model, not measured on data.",
        ],
    }
    (out / "manual_step_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
