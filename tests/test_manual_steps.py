from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gaze_objects.manual_steps import (
    build_recording_index,
    build_step_model,
    decode,
    em_refine,
    emission_logs,
    label_to_symbol,
    load_manual_config,
    predict_from_sequence,
    run_manual_step_mapping,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "manual_steps.yaml"


@pytest.fixture(scope="module")
def config():
    return load_manual_config(CONFIG)


def _planted_disassembly():
    blocks = [
        ("D1", ["side handle", "angle grinder", "red box", "angle grinder"]),
        ("D2", ["green screwdriver.tip", "angle grinder", "screw", "red box", "green screwdriver.tip"]),
        ("D4", ["red screwdriver.tip", "hand", "angle grinder", "red screwdriver.handle", "screw"]),
        ("D6", ["wrench", "angle grinder", "wrench"]),
        ("D7", ["armature", "hand", "angle grinder", "white gear"]),
        ("D8", ["blue box", "armature", "red box", "blue box"]),
    ]
    labels, truth = [], []
    for sid, labs in blocks:
        labels += labs
        truth += [sid] * len(labs)
    return labels, truth


def test_recording_index_odd_even_and_zero_based():
    idx = build_recording_index(
        ["AB01CD02_01", "AB01CD02_02", "AB01CD02_09", "AB01CD02_10", "ZZ00ZZ00_00", "ZZ00ZZ00_01"],
        expertise={"AB01CD02": "expert"},
    ).set_index("recording_id")
    assert idx.at["AB01CD02_01", "task"] == "disassembly"
    assert idx.at["AB01CD02_02", "task"] == "assembly"
    assert idx.at["AB01CD02_09", "repetition"] == 5
    assert idx.at["AB01CD02_10", "repetition"] == 5
    assert idx.at["AB01CD02_01", "expertise"] == "expert"
    assert not idx.at["ZZ00ZZ00_00", "included"]
    assert not idx.at["ZZ00ZZ00_01", "included"]


def test_label_to_symbol_parts_and_colours(config):
    rules = config["symbols"]
    assert label_to_symbol("screwdriver.tip", rules) == "screwdriver"
    assert label_to_symbol("red screwdriver.handle", rules) == "screwdriver_red"
    assert label_to_symbol("green screwdriver", rules) == "screwdriver_green"
    assert label_to_symbol("blue box", rules) == "box_blue"
    assert label_to_symbol("storage box", rules) == "box"
    assert label_to_symbol("angle grinder.side_handle_zone", rules) == "grinder"
    assert label_to_symbol("gear housing", rules) == "grinder"
    assert label_to_symbol("white gear", rules) == "inner_part"
    assert label_to_symbol("instruction manual", rules) == "manual"
    assert label_to_symbol("coffee cup", rules) == "other"
    assert label_to_symbol(None, rules) is None


def test_decode_recovers_planted_step_order(config):
    model = build_step_model(config, "disassembly")
    labels, truth = _planted_disassembly()
    symbols = [label_to_symbol(x, config["symbols"]) for x in labels]
    dec = decode(model, emission_logs(model, symbols))
    pred = [model.step_ids[i] for i in dec.viterbi]
    assert np.mean(np.array(pred) == np.array(truth)) >= 0.8
    assert all(b >= a for a, b in zip(dec.viterbi, dec.viterbi[1:]))
    assert np.allclose(dec.posterior.sum(axis=1), 1.0)


def test_online_posterior_uses_past_only(config):
    model = build_step_model(config, "disassembly")
    model.expected_len = 4.0
    labels, _ = _planted_disassembly()
    symbols = [label_to_symbol(x, config["symbols"]) for x in labels]
    full = decode(model, emission_logs(model, symbols))
    prefix = decode(model, emission_logs(model, symbols[:10]))
    assert np.allclose(full.online[:10], prefix.online)


def test_predict_wrench_window_points_to_wrench_step(config):
    for task, expected in (("disassembly", "D6"), ("assembly", "A3")):
        model = build_step_model(config, task)
        table = predict_from_sequence(model, ["wrench", "angle grinder", "wrench"], config["symbols"])
        assert table.sort_values("p_step_bag_of_objects", ascending=False).iloc[0]["step_id"] == expected


def test_em_keeps_distributions_normalised(config):
    model = build_step_model(config, "assembly")
    recs = [(["grinder", "box", "screwdriver", "grinder", "manual", "wrench"] * 5, None)] * 3
    history = em_refine(model, recs, 3)
    assert len(history) == 3
    assert np.allclose(model.base_prob.sum(axis=1), 1.0)
    for shares in model.colour_share.values():
        assert np.allclose(shares.sum(axis=1), 1.0)


def _corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    labels, truth = _planted_disassembly()
    rows, ann = [], []
    for pid in ("AA01AA01", "BB02BB02", "CC03CC03"):
        for num in (1, 2):
            t = 0.0
            seq = labels if num == 1 else list(reversed(labels))
            for i, lab in enumerate(seq):
                rows.append(
                    {
                        "recording_id": f"{pid}_{num:02d}",
                        "sequence_id": i,
                        "attended_label": lab,
                        "video_start_s": t,
                        "video_end_s": t + 0.4,
                        "total_gaze_event_duration_raw": 400.0,
                    }
                )
                t += 1.0
            if num == 1:
                bounds: dict[str, list[float]] = {}
                for i, sid in enumerate(truth):
                    bounds.setdefault(sid, [i * 1.0, i * 1.0 + 0.4])[1] = i * 1.0 + 0.4
                for sid, (s, e) in bounds.items():
                    ann.append({"recording_id": f"{pid}_01", "step_id": sid, "start_s": s, "end_s": e})
    seq_csv = tmp_path / "corpus_attention_sequences.csv"
    pd.DataFrame(rows).to_csv(seq_csv, index=False)
    part_csv = tmp_path / "participants.csv"
    pd.DataFrame(
        {"participant_id": ["AA01AA01", "BB02BB02", "CC03CC03"], "expertise": ["expert", "non-expert", "non-expert"]}
    ).to_csv(part_csv, index=False)
    ann_csv = tmp_path / "annotations.csv"
    pd.DataFrame(ann).to_csv(ann_csv, index=False)
    return seq_csv, part_csv, ann_csv


def test_run_manual_step_mapping_end_to_end(tmp_path):
    seq_csv, part_csv, ann_csv = _corpus(tmp_path)
    out = tmp_path / "out"
    summary = run_manual_step_mapping(
        seq_csv,
        out,
        model_config=CONFIG,
        participants_csv=part_csv,
        annotations_csv=ann_csv,
        em_iterations=2,
    )
    assert summary["n_recordings_mapped"] == 6
    assert summary["tasks"] == {"disassembly": 3, "assembly": 3}
    for name in (
        "step_episode_map.csv",
        "step_timeline.csv",
        "step_metrics.csv",
        "step_summary_by_expertise.csv",
        "step_expertise_contrast.csv",
        "step_motifs.csv",
        "step_emissions.csv",
        "step_discriminability_model.csv",
        "step_eval_per_step.csv",
        "step_eval_leave_participant_out.csv",
        "recording_task_index.csv",
    ):
        assert (out / name).is_file(), name
    timeline = pd.read_csv(out / "step_timeline.csv")
    assert set(timeline["status"]) <= {"observed", "uncertain", "not_observed"}
    assert summary["annotation_evaluation"]["episode_accuracy_offline"] >= 0.7
    episodes = pd.read_csv(out / "step_episode_map.csv")
    assert {"online_step_id", "online_step_prob", "step_confidence", "p_D1"} <= set(episodes.columns)
