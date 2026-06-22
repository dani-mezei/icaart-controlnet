# ControlNet Module

This module handles the data preparation, training, and inference pipeline for the ControlNet models described in the paper. It is operated via a unified CLI runner designed out-of-the-box for A100 environments.

## Quick Start

All commands should be run from the root of the project directory (e.g., `icaart-controlnet`).

### 0. Python Environment

Set up the Linux CUDA environment before downloading models or starting training:

```bash
bash controlnet/scripts/setup_venv_cu121.sh
source "${VIRTUAL_ENV:-$HOME/.venv}/bin/activate"
```

See `controlnet/ENV_SETUP.md` for scratch-storage cache locations, JupyterHub setup, CUDA wheel details, and verification commands.

### 1. Data Preparation
Prepare your dataset pipeline. This step sets up the HuggingFace dataset builder script and optionally generates BLIP2 captions for your input images.

```bash
python -m controlnet.runner prep --data_dir /path/to/data
```
*(If you need captioning, pass `--blip2_dir /path/to/blip2_model` and `--input_dir /path/to/images`)*

### 2. Training
The training pipeline is pre-configured with A100 optimizations enabled by default (`bf16` mixed precision, `TF32` acceleration, `8-bit Adam`, and `gradient checkpointing`).

Download the Stable Diffusion v1.5 base model first and pass its local path as `--model_dir`. See `controlnet/SD15_DOWNLOAD.md` for the minimal training-compatible download script and Hugging Face cache recommendations.

To reproduce the paper's exact hyperparameter setup (batch size 128, learning rate 1e-4, 20 epochs), use the provided reference config:

```bash
python -m controlnet.runner train \
    --config controlnet/configs/a100_train.json \
    --model_dir /path/to/stable-diffusion-v1-5 \
    --dataset_dir /path/to/data \
    --output_dir /path/to/output
```

> **Note on VRAM Limits:** The original paper achieved a batch size of 128 by distributing the load across 4 GPUs (32 per GPU). If you are training on a single GPU and hit Out-of-Memory (OOM) errors, use gradient accumulation instead of lowering your effective batch size:
> ```bash
> python -m controlnet.runner train \
>     --config controlnet/configs/a100_train.json \
>     --train_batch_size 32 \
>     --gradient_accumulation_steps 4 \
>     ... (paths)
> ```

#### Shared A100 and Streaming Dataset

For a shared 40 GB A100, start with the conservative config. It keeps the paper-style effective batch size of 128 via gradient accumulation while lowering peak VRAM:

```bash
python -m controlnet.runner train \
    --config controlnet/configs/a100_shared_40gb_conservative.json \
    --model_dir /path/to/stable-diffusion-v1-5 \
    --dataset_dir /path/to/data \
    --output_dir /path/to/output
```

If the remote server does not have enough disk space for KITTI-360, prepare and upload a private Hugging Face streaming dataset from a machine with enough storage:

```bash
python controlnet/training/prepare_hf_streaming_dataset.py \
    --train_image_dir /path/to/train/images \
    --train_mask_dir /path/to/train/masks \
    --validation_image_dir /path/to/validation/images \
    --validation_mask_dir /path/to/validation/masks \
    --output_dir /path/to/hf_kitti360_controlnet \
    --repo_id your-user/kitti360-controlnet \
    --private
```

Then train on the remote server with a pinned dataset revision and train sample count:

```bash
python -m controlnet.runner train \
    --config controlnet/configs/a100_streaming_conservative.json \
    --model_dir /path/to/stable-diffusion-v1-5 \
    --output_dir /path/to/output \
    --hf_dataset_name your-user/kitti360-controlnet \
    --dataset_revision <dataset-commit-sha> \
    --dataset_num_samples <train-sample-count>
```

See `controlnet/SHARED_A100.md` for the full shared-GPU and streaming workflow.

### 3. Inference
Generate images using your newly trained ControlNet model.

```bash
python -m controlnet.runner infer \
    --controlnet_dir /path/to/output \
    --stable_diffusion_dir /path/to/stable-diffusion-v1-5 \
    --input_data_dir /path/to/test_data \
    --mask_dir_name mask \
    --prompt_file_name prompt.jsonl \
    --output_dir ./inference_results \
    --batch_size 1 \
    --dtype fp16 \
    --skip_existing \
    --seed 42
```

## Advanced Usage

The CLI supports overwriting any JSON config values directly from the command line. You can disable specific hardware optimizations if you run on different hardware (e.g., `--no_8bit_adam`, `--no_tf32`).

For a full list of arguments for any command, run:
```bash
python -m controlnet.runner {prep,train,infer} --help
```
