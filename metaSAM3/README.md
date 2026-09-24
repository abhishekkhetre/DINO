# metaSAM3 — Meta Segment Anything Model 3

Local workspace for evaluating **SAM 3** as a drop-in replacement for **Grounding DINO** in the Tobii gaze → attended-object pipeline.

## What’s in this folder

| Path | Purpose |
| --- | --- |
| `sam3/` | Official clone of [facebookresearch/sam3](https://github.com/facebookresearch/sam3) (code + examples) |
| `PLAN_MINIMAL_MIGRATION.md` | How SAM 3 maps to our Stages A–D and the lowest-effort shift from Grounding DINO |
| `CHECKPOINTS.md` | Hugging Face weight access / download steps (auth required) |
| `prompts_pilot.txt` | Starting text concepts aligned with our AM07 study AOIs |

## Why SAM 3 fits our work

SAM 3 does **Promptable Concept Segmentation**: given a short text phrase (or exemplars), it returns **boxes + masks** (and video tracking). That is the same role Grounding DINO played for us — open-vocabulary objects → boxes we feed into gaze assignment — with extra upside:

- Stronger open-vocab concept coverage (SA-Co benchmark)
- **Masks** (gaze-in-mask can be stricter than gaze-in-box for parts)
- **Video tracking** (useful later for Stage E sequences)

Our Stages A/B (audit, sync) and assign/evaluate stay the same. Only the **detect** backend changes.

## Laptop vs workstation

| Step | Laptop (this PC) | Workstation (RTX) |
| --- | --- | --- |
| Clone / read docs | Done here | Sync via git / copy `metaSAM3` |
| Install `sam3` + CUDA torch | Optional (no NVIDIA here) | **Required** |
| Download HF checkpoints | Needs HF account + access grant | Same |
| AM07 detect pilot | Not practical without GPU | Run here |

## Official links

- Repo: https://github.com/facebookresearch/sam3  
- Paper: https://arxiv.org/abs/2511.16719  
- Project: https://ai.meta.com/sam3  
- Weights: https://huggingface.co/facebook/sam3 (and SAM 3.1: `facebook/sam3.1`)
