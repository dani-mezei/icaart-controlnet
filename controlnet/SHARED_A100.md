# Shared A100 40 GB ControlNet Notes

This repo trains a Stable Diffusion 1.5 ControlNet where only the ControlNet
weights are updated. The VAE, UNet, and text encoder are frozen, but they still
occupy VRAM during the forward/backward pass.

## Do Not Reserve VRAM

Do not intentionally allocate unused VRAM just to hold it. On a shared GPU this
hurts other users and does not make your run safer. Use a cap instead:

```bash
--cuda_memory_fraction 0.85
```

On a 40 GB A100, `0.85` means PyTorch may use roughly 34 GiB before it fails
fast with an OOM inside your process. It does not reserve the memory up front.

The runner also sets this allocator default for subprocesses:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

That reduces allocator fragmentation in long training runs.

## Recommended First Run

Use a smoke test before launching the full paper-reproduction run:

```bash
python -m controlnet.runner train \
    --config controlnet/configs/a100_shared_40gb_conservative.json \
    --model_dir /path/to/stable-diffusion-v1-5 \
    --dataset_dir /path/to/controlnet_kitti360 \
    --output_dir /path/to/output/controlnet-kitti360-smoke \
    --max_train_samples 32 \
    --validation_data_dir /path/to/controlnet_kitti360_val
```

Then watch:

```bash
nvidia-smi -l 2
```

If the peak is below your agreed share, remove `--max_train_samples` and run the
full job.

Two shared-A100 configs are provided:

| Config | Effective batch | VRAM posture |
|--------|-----------------|--------------|
| `a100_shared_40gb_conservative.json` | `2 * 64 = 128` | Safer when others may start jobs during training |
| `a100_shared_40gb.json` | `4 * 32 = 128` | Faster if the GPU is quiet and smoke tests show headroom |

## Paper-Style Effective Batch Size

The repo's paper config uses batch size 128. On one shared A100, treat that as
an effective batch size:

```text
effective_batch = train_batch_size * gradient_accumulation_steps * num_gpus
128 = 4 * 32 * 1
```

The shared config therefore uses:

```json
"train_batch_size": 4,
"gradient_accumulation_steps": 32
```

The conservative shared config uses:

```json
"train_batch_size": 2,
"gradient_accumulation_steps": 64
```

If your smoke test has a large safety margin, try `4 * 32 = 128` or `8 * 16 =
128`. Avoid starting with `32 * 4` or `128 * 1` on a shared 40 GB device.

## Rough VRAM Planning

For SD 1.5 ControlNet training at 512 px with bf16, gradient checkpointing, and
8-bit Adam, plan conservatively:

| Micro-batch | Effective batch with accumulation | Expected use |
|-------------|-----------------------------------|--------------|
| 1 | 128 via accumulation 128 | safest probe, slow |
| 2 | 128 via accumulation 64 | safe starting point |
| 4 | 128 via accumulation 32 | recommended first full run |
| 8 | 128 via accumulation 16 | try only after measuring |
| 16+ | 128 via lower accumulation | not a shared-GPU default |

Real VRAM depends on installed Diffusers/PyTorch versions, xFormers/SDPA,
validation frequency, and image resolution. Measure your local peak; do not rely
only on estimates.

## KITTI-360 ControlNet Dataset Shape

The training loader expects this directory:

```text
controlnet_kitti360/
  image/
    0000000250.png
  mask/
    0000000250.png
  prompt.jsonl
```

Each JSONL row must contain matching image and mask names:

```json
{"image": "0000000250.png", "mask": "0000000250.png", "prompt": "a driving scene"}
```

Use RGB/color semantic masks for ControlNet conditioning. Raw class-index masks
can work technically after `.convert("RGB")`, but they are much less explicit
than a consistent Cityscapes/KITTI color palette.

After extracting KITTI-360, stage matched RGB frames and masks with:

```bash
python controlnet/training/prepare_kitti360_controlnet.py \
    --image_dir /path/to/2013_05_28_drive_0000_sync/image_00/data_rect \
    --mask_dir /path/to/data_2d_semantics/train/2013_05_28_drive_0000_sync/image_00/semantic \
    --output_dir /path/to/controlnet_kitti360 \
    --link
```

Omit `--link` if symlinks are not allowed. The script writes a generic prompt
for each image; replace `prompt.jsonl` with BLIP2-generated captions if you need
to match the paper captioning path.

## Hugging Face Streaming

If the training server does not have enough disk space, prepare and upload a
private Hugging Face ImageFolder dataset from a machine that does have space:

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

After upload, copy the sample count and the pinned dataset commit/revision into
`controlnet/configs/a100_streaming_conservative.json`.

Run streamed training on the server with:

```bash
python -m controlnet.runner train \
    --config controlnet/configs/a100_streaming_conservative.json \
    --model_dir /path/to/stable-diffusion-v1-5 \
    --output_dir /path/to/output/controlnet-kitti360-streaming \
    --hf_dataset_name your-user/kitti360-controlnet \
    --dataset_revision <dataset-commit-sha> \
    --dataset_num_samples <train-sample-count>
```

Streaming keeps the dataset off local disk, but the shuffle is buffer-based
rather than a full local random shuffle. For reproducibility, pin
`dataset_revision`, keep `dataset_num_samples` fixed, and keep the same seed,
batch size, accumulation steps, and number of epochs. The streaming config uses
`dataloader_num_workers: 0` to keep iterable ordering easier to reason about;
raise it later only after validating sample counts and throughput.

## Inference On A Shared GPU

Use batch size 1 first:

```bash
python -m controlnet.runner infer \
    --controlnet_dir /path/to/output \
    --stable_diffusion_dir /path/to/stable-diffusion-v1-5 \
    --input_data_dir /path/to/controlnet_kitti360_val \
    --mask_dir_name mask \
    --prompt_file_name prompt.jsonl \
    --output_dir /path/to/generated \
    --batch_size 1 \
    --dtype fp16 \
    --cuda_memory_fraction 0.85 \
    --skip_existing \
    --seed 42
```

Turn on `--enable_attention_slicing` if another user is active or if generation
fails near the limit.

For long generation runs, shard by index range instead of adding multi-GPU
logic:

```bash
python -m controlnet.runner infer \
    --controlnet_dir /path/to/output \
    --stable_diffusion_dir /path/to/stable-diffusion-v1-5 \
    --input_data_dir /path/to/controlnet_kitti360_val \
    --mask_dir_name mask \
    --prompt_file_name prompt.jsonl \
    --output_dir /path/to/generated \
    --batch_size 2 \
    --dtype fp16 \
    --start_index 0 \
    --end_index 1000 \
    --skip_existing \
    --seed 42 \
    --manifest_file generation_manifest_0000_1000.jsonl
```

`--skip_existing` makes interrupted jobs resumable. `--seed` uses
`seed + absolute_sample_index` for each sample, so index-sharded jobs remain
deterministic. Keep `--latent_denoising_steps` disabled for full runs because it
saves many intermediate images and substantially increases runtime and disk use.
