# Local folder: manual-learning t-tests

Workspace path on this device:

`/workspace/local_ttest_manual_learning/`

## Status

| Item | Status |
|---|---|
| Group-level tables and evidence figures (from the shared report) | **Present** in `00_available_results/` |
| Participant-level manual share per recording | **Missing** — required for t-tests |
| Expertise list | **Missing** — required for expert vs non-expert tests |
| Recording duration / timestamps per recording | **Missing** — required for duration decay |
| Recordings 9–10 | Confirmed: **manual was not provided** → exclude from learning-curve fits; analyse separately as no-manual trials |

**No t-tests have been run yet.** The only tables on this device are group means (e.g. mean manual share for 17 experts / 19 non-experts). A t-test needs one value per participant. Inventing individual values from the group means would be incorrect, so that is not done.

## Folder layout

```
00_available_results/   # what we already have (group aggregates + figures)
01_inputs_needed/       # put the required CSVs here (see below)
02_scripts/             # analysis script (runs only when inputs are present)
03_outputs/             # t-test tables and figures written here after a successful run
```

## What to place in `01_inputs_needed/`

### Required for manual-share t-tests and log decay

**A. Preferred (one file):** `manual_share_by_participant_recording.csv`

| column | meaning |
|---|---|
| `participant_id` | e.g. AM07AM07 |
| `recording_number` | 1–10 |
| `task` | `Disassembly` or `Assembly` (odd = Disassembly, even = Assembly) |
| `task_repetition` | 1–5 = (recording_number + 1) // 2 |
| `expertise` | `Expert` or `Non-expert` |
| `manual_share` | share of **labelled** fixation time on the instruction manual (0–1) |
| `manual_duration_s` | optional but useful: seconds of gaze on the manual |
| `labelled_duration_s` | optional: total labelled fixation time (s) |

How to build it on HPC from an existing finished batch (not the pending one):

```text
corpus_attention_sequences.csv
  → per recording: sum duration where label = instruction manual
                   / sum duration of all labelled episodes
  → join participants_expertise.csv
  → write one row per recording
```

**B. Or two files:**

1. `corpus_attention_sequences.csv` (episode-level, finished corpus)  
2. `participants_expertise.csv` with columns `participant_id,expertise`

The script can build A from B if the sequence sheet has the expected columns (`recording_id`, `attended_label`, duration column).

### Required for timestamp / duration decay (supervisor request)

**C.** `recording_duration_by_participant.csv`

| column | meaning |
|---|---|
| `participant_id` | |
| `recording_number` | 1–10 |
| `task_duration_s` | recording / task length in seconds (define once and stick to it: e.g. last − first fixation, or video length) |

If duration is already a column in A, file C is not needed.

## What will be computed once inputs are present (recordings 1–8 only)

1. **Welch t-test** — expert vs non-expert mean manual share (participant means).  
2. **Paired t-test** — recording 1 vs recording 8 (same person).  
3. **Paired t-test** — within-pair change vs between-pair change (non-experts; experts reported separately).  
4. **Paired t-test** — disassembly vs assembly mean manual share.  
5. **Per-participant log / power-law decay** of manual share over recordings 1–8; one-sample test that slopes &lt; 0; Welch test expert vs non-expert slopes.  
6. Same decay for **task duration** if duration is available.  
7. **Recordings 9–10** summarised separately (expected near-zero manual share; no-manual condition).

All outputs go to `03_outputs/` with the exact numbers from the input files only.

## Run (after inputs are in place)

```bash
cd /workspace/local_ttest_manual_learning
python3 02_scripts/run_manual_learning_ttests.py
```

If required files are missing, the script exits with a clear list and writes nothing invented.
