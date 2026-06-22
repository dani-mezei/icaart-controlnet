# ControlNet Python Environment

This guide sets up the Python environment for everything under `controlnet/`: KITTI-360 preparation, Hugging Face streaming helpers, BLIP2 prompt generation, ControlNet training, ControlNet inference, and tests.

The recommended setup is a Linux `uv` virtual environment with CUDA 12.1 PyTorch wheels.

## Quick Setup

Run from the repository root on the remote Linux machine:

```bash
bash controlnet/scripts/setup_venv_cu121.sh
source "${VIRTUAL_ENV:-$HOME/.venv}/bin/activate"
python -m controlnet --help
```

The script installs:

- `torch==2.5.1`, `torchvision==0.20.1`, and `torchaudio==2.5.1` from the PyTorch CUDA 12.1 wheel index
- the local repo in editable mode
- ControlNet dependencies including Diffusers, Accelerate, Transformers, Datasets, OpenCV, TensorBoard, and BitsAndBytes

## Storage-Constrained Servers

If your home directory is small, put the virtual environment and Hugging Face caches on scratch or project storage:

```bash
export VENV_DIR=/scratch/$USER/venvs/icaart-controlnet
export HF_HOME=/scratch/$USER/hf
export HF_HUB_CACHE=/scratch/$USER/hf/hub

mkdir -p "$VENV_DIR" "$HF_HOME" "$HF_HUB_CACHE"
bash controlnet/scripts/setup_venv_cu121.sh
source "$VENV_DIR/bin/activate"
```

Use the same `HF_HOME` and `HF_HUB_CACHE` values in SSH shells, JupyterHub kernels, training jobs, and inference jobs. This avoids redownloading Stable Diffusion, BLIP2, and dataset files into different cache locations.

## JupyterHub Kernel

After activating the environment, install a kernel only if JupyterHub does not already let you select the venv:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name icaart-controlnet --display-name "ICAART ControlNet"
```

Start JupyterHub with the same cache variables set, or add them to the notebook server environment:

```bash
export HF_HOME=/scratch/$USER/hf
export HF_HUB_CACHE=/scratch/$USER/hf/hub
```

## Why PyTorch Is Installed Separately

`pyproject.toml` contains the normal Python dependencies, but not `torch`. CUDA-enabled PyTorch wheels come from a PyTorch-specific package index, so the setup script installs Torch first with:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

This avoids accidentally installing a CPU-only wheel from PyPI.

## Optional Packages

The default script installs the `dev` and `bnb` extras:

```bash
uv pip install -e ".[dev,bnb]"
```

`bitsandbytes` is included because the shared-A100 configs use `--use_8bit_adam`. `wandb` is not installed by default; add it only if you will report training runs to Weights & Biases:

```bash
uv pip install -e ".[wandb]"
```

`xformers` is intentionally not installed by default. It is tightly coupled to exact Torch and CUDA builds. The inference and training code can run without it, and PyTorch 2.x provides native attention kernels. Only install `xformers` if you have confirmed a compatible wheel for the exact Torch version on the server.

## Verification

The setup script prints the Python, Torch, CUDA, GPU, Diffusers, Accelerate, Transformers, Datasets, Hugging Face Hub, and BitsAndBytes versions.

You can rerun a quick check later:

```bash
python - <<'PY'
import torch
import diffusers
import accelerate
import transformers

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda runtime", torch.version.cuda)
    print("gpu", torch.cuda.get_device_name(0))
print("diffusers", diffusers.__version__)
print("accelerate", accelerate.__version__)
print("transformers", transformers.__version__)
PY
```

For the repo tests:

```bash
python -m pytest tests/controlnet -q
```

## Common Overrides

Use Python 3.11 instead of Python 3.10:

```bash
PYTHON_BIN=python3.11 bash controlnet/scripts/setup_venv_cu121.sh
```

Install without BitsAndBytes:

```bash
INSTALL_EXTRAS=dev bash controlnet/scripts/setup_venv_cu121.sh
```

Use a different CUDA wheel index only after checking the server driver and cluster policy:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash controlnet/scripts/setup_venv_cu121.sh
```
