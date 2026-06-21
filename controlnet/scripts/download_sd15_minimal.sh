#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
HF_REVISION="${HF_REVISION:-451f4fe16113bff5a5d2269ed5ad43b0592e9a14}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
SD15_DIR="${SD15_DIR:-$HOME/models/stable-diffusion-v1-5}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Download the minimal training-compatible Stable Diffusion v1.5 Diffusers files.

Environment variables:
  SD15_DIR       Destination model directory.
                 Default: $HOME/models/stable-diffusion-v1-5
  HF_HOME        Hugging Face cache root.
                 Recommended: /scratch/$USER/hf or /work/<project>/hf
                 Default: $HOME/.cache/huggingface
  HF_HUB_CACHE   Hugging Face Hub cache.
                 Recommended: $HF_HOME/hub
                 Default: $HF_HOME/hub
  HF_REVISION    Pinned model revision.
  MODEL_ID       Hugging Face model id.

Example:
  HF_HOME=/scratch/$USER/hf \
  HF_HUB_CACHE=/scratch/$USER/hf/hub \
  SD15_DIR=/scratch/$USER/models/stable-diffusion-v1-5 \
  bash controlnet/scripts/download_sd15_minimal.sh
EOF
  exit 0
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli was not found." >&2
  echo "Install it with: pip install huggingface_hub" >&2
  exit 1
fi

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$SD15_DIR"

export HF_HOME
export HF_HUB_CACHE

echo "Model:       $MODEL_ID"
echo "Revision:    $HF_REVISION"
echo "HF_HOME:     $HF_HOME"
echo "HF_HUB_CACHE:$HF_HUB_CACHE"
echo "SD15_DIR:    $SD15_DIR"
echo

huggingface-cli download "$MODEL_ID" \
  --revision "$HF_REVISION" \
  --local-dir "$SD15_DIR" \
  --local-dir-use-symlinks False \
  --include "model_index.json" \
  --include "scheduler/*" \
  --include "tokenizer/*" \
  --include "text_encoder/config.json" \
  --include "text_encoder/model.safetensors" \
  --include "unet/config.json" \
  --include "unet/diffusion_pytorch_model.safetensors" \
  --include "vae/config.json" \
  --include "vae/diffusion_pytorch_model.safetensors"

echo
echo "Stable Diffusion v1.5 files downloaded."
echo "Use this training argument:"
echo "  --model_dir $SD15_DIR"
