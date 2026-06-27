# Semantic Segmentation Model

This module trains and evaluates the DeepLabV3-ResNet101 semantic segmentation model used in the ICAART experiments.

## Environment

Use the shared CUDA environment setup from the repository root:

```bash
bash controlnet/scripts/setup_venv_cu121.sh
source "${VIRTUAL_ENV:-$HOME/.venv}/bin/activate"
python -m semseg.main --help
```

See `semseg/ENV_SETUP.md` for A100 setup notes, scratch-cache guidance, verification commands, and the ResNet-101 25% synthetic experiment command.

## Paper-Style A100 Training

```bash
python -m semseg.main \
    --load_json semseg/configs/resnet101_25pct_synthetic_a100_40gb.json \
    --train_image_dir /path/to/real/train/images \
    --train_label_dir /path/to/real/train/labels_19 \
    --synthetic_image_dir /path/to/synthetic/train/images \
    --synthetic_label_dir /path/to/synthetic/train/labels_19 \
    --val_image_dir /path/to/val/images \
    --val_label_dir /path/to/val/labels_19 \
    --output_dir /path/to/output/deeplab-resnet101-25pct-synth
```
