import argparse
import json
import os
import random
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from semseg.data.data_module import MixedDataModule
from semseg.model import DeeplabV3Resnet101


SEED = 42

# Python built-in random
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# PyTorch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# PyTorch Lightning
pl.seed_everything(SEED, workers=True)


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load_json", type=str, help="Path to the json file containing the arguments."
    )
    parser.add_argument(
        "--save_json",
        type=str,
        help="Path to save the arguments in a json file after parsing.",
    )
    parser.add_argument("--train_image_dir", type=str, help="Path to training images.")
    parser.add_argument("--train_label_dir", type=str, help="Path to training labels.")
    parser.add_argument("--val_image_dir", type=str, help="Path to validation images.")
    parser.add_argument("--val_label_dir", type=str, help="Path to validation labels.")
    parser.add_argument(
        "--additional_train_image_dir",
        type=list,
        help="Paths additional training images, which will be added to the training images.",
    )
    parser.add_argument(
        "--additional_train_label_dir",
        type=list,
        help="Paths to additional training labels, which will be added to the training labels.",
    )
    parser.add_argument(
        "--negative_train_image_dir",
        type=list,
        help="Paths negativ training images, which will not be added to the training images.",
    )
    parser.add_argument(
        "--negative_train_label_dir",
        type=list,
        help="Paths to negative training labels, which will not be added to the training labels.",
    )
    parser.add_argument(
        "--additional_val_image_dir",
        type=list,
        help="Paths additional val images, which will be added to the val images.",
    )
    parser.add_argument(
        "--additional_val_label_dir",
        type=list,
        help="Paths to additional val labels, which will be added to the val labels.",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for the training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate used at training.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size used at training."
    )
    parser.add_argument(
        "--use_synthetic_images",
        action="store_true",
        help="Set to true if mixed batches need to be used.",
    )
    parser.add_argument(
        "--synthetic_image_dir", type=str, help="Path to synthetic training images."
    )
    parser.add_argument(
        "--synthetic_label_dir", type=str, help="Path to synthetic training labels."
    )
    parser.add_argument(
        "--max_real_samples",
        type=int,
        default=None,
        help="Maximum number of real images to use.",
    )
    parser.add_argument(
        "--max_synth_samples",
        type=int,
        default=None,
        help="Maximum number of synthetic images to use.",
    )
    parser.add_argument(
        "--mixed_batch",
        action="store_true",
        help=(
            "If set each batch will have real and synthetic images"
            "according to 'real_ratio'."
        ),
    )
    parser.add_argument(
        "--real_ratio",
        type=float,
        default=0.5,
        help="Ratio of real samples in each batch (0.0 to 1.0)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=376,
        help="Height of the image at training and validation.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1408,
        help="Width of the image at training and validation.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--validation_steps", type=int, default=500, help="Step count used to validate."
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--checkpoint_step", type=int, default=500, help="Checkpoint step."
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Set if distributed training is used.",
    )
    parser.add_argument(
        "--finetune_weights_path",
        type=str,
        default=None,
        help="Path to the .ckpt file with weights to load.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.load_json is not None:
        with open(args.load_json, "r") as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(input_args, namespace=t_args)

    if args.save_json is not None:
        with open(args.save_json, "w") as f:
            json.dump(vars(args), f, indent=4)

    required_paths = [
        "train_image_dir",
        "train_label_dir",
        "val_image_dir",
        "val_label_dir",
        "output_dir",
    ]
    if args.use_synthetic_images:
        required_paths.extend(["synthetic_image_dir", "synthetic_label_dir"])

    missing_paths = [name for name in required_paths if not getattr(args, name)]
    if missing_paths:
        missing = ", ".join(missing_paths)
        raise FileNotFoundError(f"Missing required arguments: {missing}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mixed_batch and not args.use_synthetic_images:
        raise ValueError(
            "Cannot use mixed batches, when synthetic images are not used."
        )

    return args


def main(args):

    num_classes = 19

    model = DeeplabV3Resnet101(num_classes=num_classes, lr_rate=args.learning_rate)

    if args.finetune_weights_path:
        print(f"\nLoading weights ONLY from: {args.finetune_weights_path}")
        print("Ignoring hyperparameters stored in the checkpoint.")
        if not os.path.exists(args.finetune_weights_path):
            raise FileNotFoundError(
                f"Weights file not found: {args.finetune_weights_path}"
            )

        # Load the checkpoint dictionary
        checkpoint = torch.load(
            args.finetune_weights_path, map_location=torch.device("cpu")
        )

        # Check if the expected 'state_dict' key exists
        if "state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint dictionary does not contain the key 'state_dict'. "
                "Cannot load weights."
            )

        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["state_dict"], strict=True
        )

        if unexpected_keys:
            print(
                f"Warning: Unexpected keys found in checkpoint's state_dict: {unexpected_keys}"
            )
        if missing_keys:
            print(
                f"Warning: Missing keys in model's state_dict that were not in checkpoint: {missing_keys}"
            )

        print("Successfully loaded weights into the model.")
    else:
        print(
            "\nStarting training from scratch (random initialization or default torchvision weights)."
        )

    logger = TensorBoardLogger(save_dir=args.output_dir, name="logs")

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",  # Directory to save checkpoints
        filename="step-checkpoint-{epoch:02d}-{step:06d}",  # Filename format with epoch and step
        every_n_train_steps=args.checkpoint_step,  # Save checkpoint every N steps
        save_top_k=-1,  # Keep all checkpoints
        verbose=True,  # Enable logging for checkpointing
    )

    trainer = Trainer(
        max_epochs=args.num_train_epochs,
        accelerator=device,
        logger=logger,
        devices="auto",
        strategy="ddp_find_unused_parameters_true" if args.distributed else "auto",
        val_check_interval=args.validation_steps,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        use_distributed_sampler=False,
        callbacks=[checkpoint_callback],
    )

    data_module = MixedDataModule(args=args)

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    args = parse_args()
    main(args)
