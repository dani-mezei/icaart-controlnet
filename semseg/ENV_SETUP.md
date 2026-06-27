# DeepLab Python Environment

This guide sets up the Python environment for `semseg/`: DeepLabV3-ResNet101 training, validation, checkpoint testing, and the paper-style synthetic-data experiments.

Use the same Linux CUDA virtual environment as `controlnet/`. The shared setup script installs CUDA PyTorch wheels first, then installs this repo in editable mode with both ControlNet and DeepLab dependencies.

## Quick Setup

Run from the repository root on the remote Linux machine:

```bash
bash controlnet/scripts/setup_venv_cu121.sh
source "${VIRTUAL_ENV:-$HOME/.venv}/bin/activate"
python -m semseg.main --help
```

The script installs:

- `torch==2.5.1`, `torchvision==0.20.1`, and `torchaudio==2.5.1` from the PyTorch CUDA 12.1 wheel index
- the local repo in editable mode
- DeepLab dependencies including PyTorch Lightning, TorchMetrics, OpenCV, Pillow, NumPy, and TensorBoard
- ControlNet dependencies used by the rest of this repository

## Storage-Constrained Servers

If your home directory is small, place the virtual environment and model caches on scratch or project storage:

```bash
export VENV_DIR=/scratch/$USER/venvs/icaart-controlnet
export HF_HOME=/scratch/$USER/hf
export HF_HUB_CACHE=/scratch/$USER/hf/hub

mkdir -p "$VENV_DIR" "$HF_HOME" "$HF_HUB_CACHE"
bash controlnet/scripts/setup_venv_cu121.sh
source "$VENV_DIR/bin/activate"
```

DeepLab uses torchvision ImageNet-pretrained weights for ResNet-101. Keeping the cache variables stable avoids repeated downloads across SSH shells, JupyterHub kernels, and training jobs.

## JupyterHub Kernel

After activating the environment, install a kernel only if JupyterHub does not already expose the venv:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name icaart-controlnet --display-name "ICAART ControlNet"
```

Start JupyterHub with the same cache variables set:

```bash
export HF_HOME=/scratch/$USER/hf
export HF_HUB_CACHE=/scratch/$USER/hf/hub
```

## Verification

The setup script prints Python, Torch, CUDA, GPU, torchvision, PyTorch Lightning, TorchMetrics, OpenCV, and ControlNet dependency versions.

You can rerun a quick DeepLab check later:

```bash
python - <<'PY'
import torch
import torchvision
import pytorch_lightning
import torchmetrics
import cv2

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda runtime", torch.version.cuda)
    print("gpu", torch.cuda.get_device_name(0))
print("torchvision", torchvision.__version__)
print("pytorch_lightning", pytorch_lightning.__version__)
print("torchmetrics", torchmetrics.__version__)
print("cv2", cv2.__version__)
PY
```

Confirm the training entrypoint is available from the repo root:

```bash
python -m semseg.main --help
```

## ResNet-101 25% Synthetic Experiment

The paper uses DeepLabV3 with a ResNet-101 backbone, ImageNet-pretrained torchvision weights, Adam with learning rate `1e-4`, cross-entropy loss with ignore index `255`, input resolution `1408x376`, batch size `128`, and `20` epochs.

DeepLab labels must be single-channel 19-class trainId PNGs with values `0..18` and `255` for ignored pixels. If your dataset has KITTI-360 raw semantic IDs in `mask_semseg`, relabel them first:

```bash
python -m semseg.preprocess.relabel \
    --input_dir /path/to/controlnet_dataset/train/mask_semseg \
    --output_dir /path/to/controlnet_dataset/train/label_19

python -m semseg.preprocess.relabel \
    --input_dir /path/to/controlnet_dataset/val/mask_semseg \
    --output_dir /path/to/controlnet_dataset/val/label_19
```

For one A100 40 GB GPU, start with the provided config. It keeps the effective batch size at `128` via gradient accumulation while lowering peak VRAM:

```bash
python -m semseg.main \
    --load_json semseg/configs/resnet101_25pct_synthetic_a100_40gb.json \
    --train_image_dir /path/to/real/train/images \
    --train_label_dir /path/to/real/train/label_19 \
    --synthetic_image_dir /path/to/synthetic/train/images \
    --synthetic_label_dir /path/to/synthetic/train/label_19 \
    --val_image_dir /path/to/val/images \
    --val_label_dir /path/to/val/label_19 \
    --output_dir /path/to/output/deeplab-resnet101-25pct-synth
```

This config uses `batch_size=8` and `gradient_accumulation_steps=16`, so the effective batch size is `8 * 16 = 128`. It also uses `mixed_batch` with `real_ratio=0.75`, giving each batch 75% real and 25% synthetic samples.

Validation runs every `1000` training batches by default to reduce validation overhead on full A100 runs. Override `--validation_steps` for smoke tests or denser validation curves.

To strictly match the paper's fixed-size replacement setup, make sure the real and synthetic directories passed to the command contain the intended 75/25 split. The command controls the batch ratio; the actual dataset replacement is determined by which files are present in those directories.

## Smoke Test

Before starting a full 20-epoch run on the A100, run a tiny dataloading and training check:

```bash
python -m semseg.main \
    --load_json semseg/configs/resnet101_25pct_synthetic_a100_40gb.json \
    --train_image_dir /path/to/real/train/images \
    --train_label_dir /path/to/real/train/labels_19 \
    --synthetic_image_dir /path/to/synthetic/train/images \
    --synthetic_label_dir /path/to/synthetic/train/labels_19 \
    --val_image_dir /path/to/val/images \
    --val_label_dir /path/to/val/labels_19 \
    --output_dir /path/to/output/deeplab-smoke \
    --max_real_samples 24 \
    --max_synth_samples 8 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --validation_steps 1 \
    --checkpoint_step 10
```

## Common Overrides

Use Python 3.11 instead of Python 3.10:

```bash
PYTHON_BIN=python3.11 bash controlnet/scripts/setup_venv_cu121.sh
```

Install only development and DeepLab extras:

```bash
INSTALL_EXTRAS=dev,semseg bash controlnet/scripts/setup_venv_cu121.sh
```

Use a different CUDA wheel index only after checking the server driver and cluster policy:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash controlnet/scripts/setup_venv_cu121.sh
```
