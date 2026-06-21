#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-controlnet}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-dev,bnb}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable '${PYTHON_BIN}' was not found." >&2
    echo "Set PYTHON_BIN to python3.10 or python3.11 and rerun." >&2
    exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys

version = sys.version_info
if not ((3, 10) <= (version.major, version.minor) < (3, 12)):
    raise SystemExit(
        f"Python {version.major}.{version.minor} is not supported. "
        "Use Python 3.10 or 3.11."
    )
PY

if command -v uv >/dev/null 2>&1; then
    UV_CMD=(uv)
else
    echo "uv was not found; installing uv into the current user environment."
    "${PYTHON_BIN}" -m pip install --user "uv>=0.4"
    UV_CMD=("${PYTHON_BIN}" -m uv)
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Virtual environment: ${VENV_DIR}"
echo "PyTorch CUDA wheel index: ${TORCH_INDEX_URL}"

"${UV_CMD[@]}" venv "${VENV_DIR}" --python "${PYTHON_BIN}"

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

packages = ["diffusers", "accelerate", "transformers", "datasets", "huggingface_hub"]

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
EOF
