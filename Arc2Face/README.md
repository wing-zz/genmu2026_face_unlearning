# Genμ 2.0 — Task 1: Face Identity Unlearning

## Method Overview

**Approach:** Arc2Face + Cross-Attention Fine-tuning with Anchor Identity Replacement

This submission modifies Arc2Face to forget a target identity by fine-tuning only the cross-attention K/V projection layers in the UNet (~2.2% of parameters). The target identity's ArcFace embedding is redirected to a dissimilar anchor identity from the retain set, while retain identities are preserved through standard diffusion loss.

## Environment Setup

```bash
conda create -n arc2face python=3.10 -y
conda activate arc2face
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Required Models

Before running, download the following models:

### Arc2Face Models (from HuggingFace)
Place under `./models/`:
- `models/arc2face/` — `config.json`, `diffusion_pytorch_model.safetensors`
- `models/encoder/` — `config.json`, `pytorch_model.bin`

### Stable Diffusion v1.5
Place under a local path (e.g., `/data/weight/`) or use the HuggingFace model ID.

### InsightFace antelopev2
Place under `./models/antelopev2/`:
- `1k3d68.onnx`, `2d106det.onnx`, `scrfd_10g_bnkps.onnx`, `genderage.onnx`, `arcface.onnx`

## Dataset

CelebA aligned images at `../Data/CelebA/Img/img_align_celeba/` (202,599 images).
Identity mapping at `../Data/CelebA/Anno/identity_CelebA.txt`.

## Validation Identities

| Set | Forget ID | Retain IDs |
|-----|-----------|------------|
| Face Set 1 | 3422 | 5230, 5239, 1539 |
| Face Set 2 | 3376 | 3602, 608, 7405 |

## Training

```bash
conda activate arc2face
python unlearn_train.py
```

This trains two models sequentially, saving checkpoints to `./outputs/unlearning/{Face_Set_1,Face_Set_2}/final/`.

**Training details:**
- Only UNet cross-attention K/V layers are trainable (19M / 859M parameters)
- 20 epochs per identity set
- Learning rate: 1e-5
- Optimizer: AdamW
- Gradient clipping: max_norm=1.0
- FP32 training
- Anchor selection: farthest retain identity in ArcFace embedding space

## Evaluation

```bash
python evaluate.py
```

Computes FA, EA, RA, and ERB for both original and unlearned models.

### Results

| | Original ERB | Unlearned ERB | FA (unlearned) | RA (unlearned) |
|---|---|---|---|---|
| Face Set 1 | 28.6 | **100.0** | 0.0% | 100.0% |
| Face Set 2 | 26.3 | **93.3** | 0.0% | 87.5% |

- **FA** (Forget Accuracy): percentage of forget-conditioned images still verified as the forget identity
- **EA** (Erasure Accuracy): 100 - FA
- **RA** (Retain Accuracy): percentage of retain-conditioned images verified correctly
- **ERB**: 2 × EA × RA / (EA + RA) — harmonic mean, the official ranking metric

## Unlearned Weights

Located at `./outputs/unlearning/`:

```
outputs/unlearning/
├── Face_Set_1/final/
│   ├── diffusion_pytorch_model.safetensors
│   ├── config.json
│   └── meta.pt
└── Face_Set_2/final/
    ├── diffusion_pytorch_model.safetensors
    ├── config.json
    └── meta.pt
```

To use unlearned weights for inference:
```python
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained("outputs/unlearning/Face_Set_1/final")
```

## Hardware Requirements

- GPU: 1× NVIDIA RTX 3090 (24GB) or equivalent
- Training time: ~5 minutes per identity set
- Inference: ~2 seconds per image (25 steps)

## File Structure

```
├── unlearn_train.py    # Training script
├── evaluate.py         # Evaluation script
├── test_arc2face.py    # Sanity check script
├── arc2face/           # Arc2Face model utilities
│   ├── __init__.py
│   ├── models.py       # CLIPTextModelWrapper
│   └── utils.py        # project_face_embs, image_align
├── requirements.txt    # Python dependencies
├── models/             # Model weights (see above)
└── outputs/unlearning/ # Unlearned model weights
```
