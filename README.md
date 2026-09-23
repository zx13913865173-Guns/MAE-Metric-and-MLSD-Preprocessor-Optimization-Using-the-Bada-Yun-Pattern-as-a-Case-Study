# MAE-Metric-and-MLSD-Preprocessor-Optimization-Using-the-Bada-Yun-Pattern-as-a-Case-Study
A Generative Evaluation Methodology for Line-Based Intangible Cultural Heritage Patterns: MAE Metric and MLSD Preprocessor Optimization Using the Bada Yun Pattern as a Case Study
# Reproducible Materials

Code and configuration for:

**A Generative Evaluation Methodology for Line-Based Intangible Cultural Heritage Patterns: MAE Metric and MLSD Preprocessor Optimization Using the Bada Yun Pattern as a Case Study**

## Repository structure

- `workflows/` – ComfyUI workflows for MLSD, Canny, and Lineart.
- `scripts/` – MAE, PCA, and objective metric evaluation.
- `train_configs/` – LoRA rank-8 training configuration.

## Environment

- Python 3.10.12, PyTorch 2.0.1+cu118, CUDA 11.8
- ComfyUI 0.3.0, controlnet_aux 0.0.7
- SD-Trainer Mikazuki v0.6.5
- Hardware: NVIDIA RTX 3050 6GB Laptop GPU

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproduction workflow

### Step 1: Dataset preparation

- Collect 100 Bada Yun images from the China National Silk Museum and Suzhou Silk Museum.
- Normalize to 512×512 PNG using XnConvert v1.100.0.
- Random 4:1 split with category stratification → 80 train / 20 test.

### Step 2: LoRA fine-tuning

Use `train_configs/lora_rank8.yaml` with SD-Trainer Mikazuki v0.6.5.
For other ranks, change `network_dim` to 4, 16, or 32.

```bash
python train.py --config train_configs/lora_rank8.yaml
```

Key hyperparameters: learning rate 1e-4, batch size 1, AdamW 8bit, fp16, seed 1337, 20 epochs, UNet only, text encoder frozen.

Expected training time / model size:
- rank 4: 1.1 h / 13 MB
- rank 8: 1.3 h / 26 MB
- rank 16: 1.6 h / 52 MB
- rank 32: 2.3 h / 104 MB

### Step 3: Image generation

Import `workflows/badayaoxiang_mlsd.json` into ComfyUI.
Generate seeds 1–40 for each group. Group configurations:

| Group | Configuration | Key parameters |
|-------|---------------|----------------|
| A | Native SD 1.5 | no LoRA, no ControlNet |
| B | LoRA rank 32 only | — |
| C | LoRA rank 8 + Canny | low 100, high 200, scale 0.7 |
| D | LoRA rank 8 + Lineart | LineartCoarse, scale 0.8 |
| E | LoRA rank 8 + MLSD | thresholds 0.1/0.2, scale 0.5 |
| H | LoRA rank 32 + MLSD | same MLSD settings as E |

Common sampler settings: DPM++ 2M Karras, 30 steps, CFG 8, 512×512.

### Step 4: Evaluation

```bash
python scripts/compute_mae.py --image path/to/image.png
python scripts/compute_pca.py --image_dir dataset/train
python scripts/evaluate_metrics.py --gen_dir generated/E --ref_dir dataset/test
```

### Step 5: Statistical analysis

Use the exported CSV (group, metric, value) with scipy/statsmodels:

- One-way ANOVA + Tukey HSD across groups
- Paired t-tests for MAE between groups
- Pearson correlation between MAE and ICH ratings
- Cohen's d with bootstrap 95% CI (1000 resamples)

## Model weights

LoRA weights are available on request or via GitHub Releases. Base model: Stable Diffusion v1.5 (pruned). ControlNet weights from `lllyasviel/ControlNet-v1-1`.
