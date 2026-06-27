#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if [[ -z "${VENV_DIR:-}" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        VENV_DIR="${VIRTUAL_ENV}"
    else
        VENV_DIR="${HOME}/.venv"
    fi
fi
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-dev,bnb,semseg}"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ -x "${VENV_PYTHON}" ]]; then
    SETUP_PYTHON="${VENV_PYTHON}"
else
    SETUP_PYTHON="${PYTHON_BIN}"
fi

if [[ "${SETUP_PYTHON}" == "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable '${PYTHON_BIN}' was not found." >&2
    echo "Set PYTHON_BIN to python3.10 or python3.11 and rerun." >&2
    exit 1
fi

"${SETUP_PYTHON}" - <<'PY'
import sys

version = sys.version_info
if not ((3, 10) <= (version.major, version.minor) < (3, 12)):
    raise SystemExit(
        f"Python {version.major}.{version.minor} is not supported. "
        "Use Python 3.10 or 3.11."
    )
PY

UV_BIN="$(command -v uv || true)"
if [[ -n "${UV_BIN}" ]] && "${UV_BIN}" --version >/dev/null 2>&1; then
    UV_CMD=("${UV_BIN}")
else
    echo "uv was not found or is not executable; installing uv with ${SETUP_PYTHON}."
    "${SETUP_PYTHON}" -m pip install --upgrade "uv>=0.4"
    UV_CMD=("${SETUP_PYTHON}" -m uv)
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Virtual environment: ${VENV_DIR}"
echo "PyTorch CUDA wheel index: ${TORCH_INDEX_URL}"

if [[ -x "${VENV_PYTHON}" ]]; then
    echo "Using existing virtual environment at ${VENV_DIR}"
else
    "${UV_CMD[@]}" venv "${VENV_DIR}" --python "${PYTHON_BIN}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

"${UV_CMD[@]}" pip install \
    --python "${VENV_DIR}/bin/python" \
    --upgrade pip setuptools wheel

"${UV_CMD[@]}" pip install \
    --python "${VENV_DIR}/bin/python" \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

"${UV_CMD[@]}" pip install \
    --python "${VENV_DIR}/bin/python" \
    -e "${PROJECT_ROOT}[${INSTALL_EXTRAS}]"

python - <<'PY'
import importlib
import sys

import torch

packages = [
    "torchvision",
    "diffusers",
    "accelerate",
    "transformers",
    "datasets",
    "huggingface_hub",
    "pytorch_lightning",
    "torchmetrics",
    "cv2",
]

print(f"Python: {sys.version.split()[0]}")
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

for package in packages:
    module = importlib.import_module(package)
    print(f"{package}: {getattr(module, '__version__', 'unknown')}")

try:
    import bitsandbytes as bnb
except Exception as exc:
    print(f"bitsandbytes: unavailable ({exc})")
else:
    print(f"bitsandbytes: {getattr(bnb, '__version__', 'unknown')}")
PY

cat <<EOF

Environment setup complete.

Activate it with:
  source "${VENV_DIR}/bin/activate"

Run the ControlNet CLI with:
  python -m controlnet --help
  icaart-controlnet --help

Run the DeepLab training entrypoint with:
  python -m semseg.main --help
EOF
