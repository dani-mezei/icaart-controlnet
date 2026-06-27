# ICAART ControlNet - Semantic Segmentation and Image Generation

## Overview

This repository contains two main components:

1. **Semantic Segmentation (`semseg/`)**: Training and inference for DeepLabV3-ResNet model on driving scene datasets (KITTI, Cityscapes)
2. **ControlNet (`controlnet/`)**: Training and inference for Stable Diffusion ControlNet conditioned on semantic segmentation masks

## Repository Structure

```
.
├── semseg/                 # Semantic segmentation module
│   ├── data/              # Dataset loaders (KITTI, combined datasets)
│   ├── model/             # DeepLabV3-ResNet101 model
│   ├── inference/         # Inference scripts
│   ├── preprocess/        # Data preprocessing utilities
│   ├── main.py            # Training script
│   └── test_model_from_ckpt.py  # Model evaluation
│
└── controlnet/            # ControlNet module
    ├── training/          # Training scripts and data pipeline
    ├── inference/         # Image generation scripts
    ├── captioning/        # BLIP2-based caption generation
    └── custom/            # Utility functions
```

## Requirements

### Core Dependencies

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning
- Diffusers (Hugging Face)
- Transformers (Hugging Face)
- Accelerate
- torchvision
- torchmetrics
- numpy
- pandas
- opencv-python (cv2)
- Pillow

### Installation

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install pytorch-lightning diffusers transformers accelerate
pip install torchmetrics pandas opencv-python pillow
```

## Usage

### 1. Semantic Segmentation

Set up the shared CUDA environment first:

```bash
bash controlnet/scripts/setup_venv_cu121.sh
source "${VIRTUAL_ENV:-$HOME/.venv}/bin/activate"
python -m semseg.main --help
```

See `semseg/ENV_SETUP.md` for the DeepLab A100 setup guide and the paper-style ResNet-101 25% synthetic experiment command.

#### Training

```bash
python -m semseg.main \
    --train_image_dir /path/to/train/images \
    --train_label_dir /path/to/train/labels \
    --val_image_dir /path/to/val/images \
    --val_label_dir /path/to/val/labels \
    --output_dir /path/to/output \
    --batch_size 8 \
    --num_train_epochs 50 \
    --learning_rate 1e-4
```

Or use JSON configuration:
```bash
python -m semseg.main --load_json config.json
```

Paper-style A100 40 GB run with 25% synthetic data:

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

#### Inference

```bash
cd semseg
python test_model_from_ckpt.py \
    --checkpoint_path /path/to/checkpoint.ckpt \
    --test_image_dir /path/to/test/images \
    --test_label_dir /path/to/test/labels
```

#### Data Preprocessing

Verify and relabel datasets:
```bash
cd semseg/preprocess
python verify.py --image_dir /path/to/images --label_dir /path/to/labels
python relabel.py --input_dir /path/to/labels --output_dir /path/to/relabeled
```

### 2. ControlNet

#### Data Preparation

1. Generate captions for training images:
```bash
cd controlnet/captioning
python caption_generator.py \
    --blip2_dir /path/to/blip2/model \
    --input_dir /path/to/images \
    --output_dir /path/to/captions
```

2. Create custom data pipeline:
```bash
cd controlnet/training
python create_custom_data_pipeline.py \
    --data_dir /path/to/training/data \
    --output_dir /path/to/output
```

#### Training

Training uses Hugging Face Accelerate for distributed training and memory optimization.

```bash
cd controlnet/training
python run.py \
    --model_dir runwayml/stable-diffusion-v1-5 \
    --data_dir /path/to/training/data \
    --output_dir /path/to/output \
    --resolution 512 \
    --train_batch_size 4 \
    --num_train_epochs 100 \
    --learning_rate 1e-5
```

Use JSON configuration:
```bash
python run.py --load_json training_config.json
```

#### Inference (Image Generation)

```bash
cd controlnet/inference
python generate.py \
    --controlnet_dir /path/to/trained/controlnet \
    --base_model_dir runwayml/stable-diffusion-v1-5 \
    --conditioning_image /path/to/mask.png \
    --prompt "Your text prompt" \
    --output_dir /path/to/output
```

## Dataset Format

### Semantic Segmentation
- **Images**: RGB images (PNG/JPG)
- **Labels**: Single-channel PNG with class indices (0-18 for 19 classes)
- **Classes**: 19 Cityscapes classes (road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle)

### ControlNet Training
The data pipeline expects:
```
data_dir/
├── image/          # Training images
├── mask/           # Semantic segmentation masks
└── prompt.jsonl    # Captions (one JSON per line)
```

## Model Architecture

- **Semantic Segmentation**: DeepLabV3 with ResNet-101 or ResNet-50 backbone
  - Pre-trained on ImageNet
  - Fine-tuned on driving scene datasets
  - 19 output classes
  - Mean IoU metric

- **ControlNet**: Stable Diffusion v1.5 with ControlNet conditioning
  - Conditioned on semantic segmentation masks
  - BLIP2-generated captions

## Reproducibility

- Random seed can be set in main (set across NumPy, PyTorch, PyTorch Lightning)
- Deterministic training enabled where possible
- GPU: NVIDIA GPUs with CUDA support recommended

## License

This project uses code adapted from Hugging Face Diffusers (Apache 2.0 License).
