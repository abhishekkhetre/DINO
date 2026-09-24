# SAM 3 checkpoints (download)

Weights are **not** in the GitHub clone. They live on Hugging Face and need access approval.

## Steps (do once)

1. Request access: https://huggingface.co/facebook/sam3  
   (For newer weights also see https://huggingface.co/facebook/sam3.1 — needs latest `metaSAM3/sam3` code.)
2. Create a token: https://huggingface.co/settings/tokens  
3. On the machine that will run inference:

```bash
pip install -U huggingface_hub
hf auth login
# paste token
```

4. First model load auto-downloads (official API):

```python
from sam3.model_builder import build_sam3_image_model
model = build_sam3_image_model()  # load_from_HF=True by default
```

Or use Transformers:

```python
from transformers import Sam3Model, Sam3Processor
model = Sam3Model.from_pretrained("facebook/sam3")
processor = Sam3Processor.from_pretrained("facebook/sam3")
```

5. Optional: pin weights under `metaSAM3/checkpoints/` on the workstation (large files; keep out of git — already covered by `*.pth` / `checkpoints/` in `.gitignore`).

## Status on this laptop

- Repo code: **cloned** under `metaSAM3/sam3/`
- Checkpoints: **not downloaded here** (no GPU; HF access is interactive)
- Next machine action: request HF access → `hf auth login` → first `build_sam3_image_model()` on workstation
