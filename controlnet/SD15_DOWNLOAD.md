# Stable Diffusion v1.5 Download

This ControlNet training code expects a Diffusers-format Stable Diffusion v1.5
directory. Download the component folders, not the root `.ckpt` checkpoint
files.

## What To Download

Use the full-precision safetensors files from:

```text
stable-diffusion-v1-5/stable-diffusion-v1-5
```

Required files:

```text
model_index.json
scheduler/*
tokenizer/*
text_encoder/config.json
text_encoder/model.safetensors
unet/config.json
unet/diffusion_pytorch_model.safetensors
vae/config.json
vae/diffusion_pytorch_model.safetensors
```

Approximate size:

| Component | Approximate size |
|-----------|------------------|
| UNet full safetensors | 3.44 GB |
| VAE full safetensors | 335 MB |
| Text encoder full safetensors | 492 MB |
| Configs, scheduler, tokenizer | Small |
| Total | About 4.3 GB |

The full Hugging Face repository is much larger because it includes duplicate
formats and precision variants. Do not download the root `.ckpt`, root
`.safetensors`, `*.bin`, or `safety_checker/*` files for this repo's training
path.

## Cache Locations

Put Hugging Face caches and model files on persistent storage with enough quota,
not inside this repository.

Recommended on a Linux cluster:

```bash
export HF_HOME=/scratch/$USER/hf
export HF_HUB_CACHE=$HF_HOME/hub
export SD15_DIR=/scratch/$USER/models/stable-diffusion-v1-5
```

Other reasonable options:

```bash
export HF_HOME=/work/<project>/hf
export HF_HUB_CACHE=$HF_HOME/hub
export SD15_DIR=/work/<project>/models/stable-diffusion-v1-5
```

Use `$HOME/.cache/huggingface` only if your home quota is large enough. Avoid
temporary folders that may be purged while training is running.

## Download

From the project root:

```bash
bash controlnet/scripts/download_sd15_minimal.sh
```

Or override locations:

```bash
HF_HOME=/scratch/$USER/hf \
HF_HUB_CACHE=/scratch/$USER/hf/hub \
SD15_DIR=/scratch/$USER/models/stable-diffusion-v1-5 \
bash controlnet/scripts/download_sd15_minimal.sh
```

Then use:

```bash
--model_dir /scratch/$USER/models/stable-diffusion-v1-5
```

## Authentication

If Hugging Face requires access approval or authentication:

```bash
huggingface-cli login
```

Run that once on the remote server before downloading. The script does not store
tokens in this repository.
