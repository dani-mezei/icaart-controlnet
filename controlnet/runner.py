#!/usr/bin/env python
"""Unified CLI for ControlNet pipeline operations.

Usage (from project root):
    python -m controlnet.runner prep  [--config config.json] [args...]
    python -m controlnet.runner train [--config config.json] [args...]
    python -m controlnet.runner infer [--config config.json] [args...]

A100-optimized defaults are baked into the train subcommand:
  - bf16 mixed precision (native A100 support, more stable than fp16)
  - Gradient checkpointing (reduces VRAM at cost of ~20% slower backward)
  - 8-bit Adam via bitsandbytes (halves optimizer state memory)
  - TF32 enabled (faster matmuls on Ampere with negligible precision loss)
  - set_grads_to_none (minor memory saving)

Each optimization can be individually disabled with --no_<flag>.
"""
import argparse
import json
import os
import subprocess
import sys

CONTROLNET_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONTROLNET_DIR)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(args):
    """Merge a JSON config file into *args*.

    Keys present in the JSON but absent (None) on *args* are filled in.
    Non-None CLI values always take precedence over the config file.
    """
    if hasattr(args, "config") and args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    return args


# ---------------------------------------------------------------------------
# Subcommand: prep
# ---------------------------------------------------------------------------

def _build_prep_parser(subparsers):
    p = subparsers.add_parser(
        "prep",
        help="Data preparation — captioning (optional) + data pipeline creation.",
    )
    p.add_argument("--config", type=str, default=None,
                    help="Path to JSON config file. CLI args override config values.")
    # Captioning (optional — skipped if --blip2_dir is not set)
    p.add_argument("--blip2_dir", type=str, default=None,
                    help="Path to BLIP2 model directory. If not set, captioning is skipped.")
    p.add_argument("--input_dir", type=str, default=None,
                    help="Path to input images for captioning.")
    p.add_argument("--caption_output_dir", type=str, default=None,
                    help="Path to save generated captions. Defaults to --data_dir.")
    p.add_argument("--question", type=str, default=None,
                    help="Question for BLIP2 visual QA captioning.")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--prompt_suffix", type=str, default="")
    # Data pipeline
    p.add_argument("--data_dir", type=str, default=None,
                    help="Path to training data directory (must contain image/, mask/, prompt.jsonl).")
    p.add_argument("--dataset_name", type=str, default="data_pipeline",
                    help="Name for the HuggingFace dataset builder.")
    return p


def run_prep(args):
    """Execute data preparation: optional captioning then pipeline configuration."""
    args = load_config(args)

    env = _get_env()

    # Step 1: Captioning (optional)
    if args.blip2_dir:
        script = os.path.join(CONTROLNET_DIR, "captioning", "caption_generator.py")
        cmd = [
            sys.executable, script,
            "--blip2_dir", args.blip2_dir,
            "--input_dir", args.input_dir,
            "--output_dir", args.caption_output_dir or args.data_dir,
        ]
        if args.question:
            cmd.extend(["--question", args.question])
        if args.max_new_tokens:
            cmd.extend(["--max_new_tokens", str(args.max_new_tokens)])
        if args.prompt_suffix:
            cmd.extend(["--prompt_suffix", args.prompt_suffix])

        print(f"[prep] Running captioning:\n  {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)

    # Step 2: Configure the HuggingFace data pipeline script
    if args.data_dir:
        script = os.path.join(CONTROLNET_DIR, "training", "create_custom_data_pipeline.py")
        cmd = [
            sys.executable, script,
            "--data_dir", args.data_dir,
            "--dataset_name", args.dataset_name,
        ]
        print(f"[prep] Creating data pipeline:\n  {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def _build_train_parser(subparsers):
    p = subparsers.add_parser(
        "train",
        help="Train ControlNet with A100-optimized defaults.",
    )
    p.add_argument("--config", type=str, default=None,
                    help="Path to JSON config file. CLI args override config values.")
    # Core paths
    p.add_argument("--model_dir", type=str, default=None,
                    help="Path to pretrained Stable Diffusion model (or HF model id).")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Directory to save trained ControlNet checkpoints.")
    p.add_argument("--dataset_dir", type=str, default=None,
                    help="Path to training data (image/, mask/, prompt.jsonl).")
    p.add_argument("--controlnet_dir", type=str, default=None,
                    help="Resume from a pretrained ControlNet checkpoint.")
    p.add_argument("--validation_data_dir", type=str, default=None,
                    help="Directory with images/ and prompt.jsonl for validation.")
    # Training hyperparameters
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--lr_scheduler", type=str, default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--lr_num_cycles", type=int, default=1)
    p.add_argument("--lr_power", type=float, default=1.0)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--validation_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    # A100 optimizations (on by default — use --no_* to disable)
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                    help="Disable gradient checkpointing.")
    p.add_argument("--no_8bit_adam", action="store_true",
                    help="Use standard AdamW instead of 8-bit Adam.")
    p.add_argument("--no_tf32", action="store_true",
                    help="Disable TF32 matmul acceleration.")
    p.add_argument("--no_set_grads_to_none", action="store_true",
                    help="Use zero_grad() instead of setting grads to None.")
    # Multi-GPU
    p.add_argument("--multi_gpu", action="store_true",
                    help="Enable multi-GPU training via Accelerate.")
    p.add_argument("--num_gpus", type=int, default=1,
                    help="Number of GPUs (only used with --multi_gpu).")
    return p


def build_train_command(args):
    """Build the ``accelerate launch`` command list for training.

    Returns a list of strings suitable for ``subprocess.run()``.
    This function is deterministic and side-effect-free so it can be unit-tested.
    """
    train_script = os.path.join(CONTROLNET_DIR, "training", "train_controlnet.py")
    data_pipeline = os.path.join(CONTROLNET_DIR, "training", "data_pipeline.py")

    # --- accelerate flags ---
    cmd = ["accelerate", "launch"]
    if args.multi_gpu:
        cmd.extend(["--multi_gpu", f"--num_processes={args.num_gpus}"])
    cmd.append("--mixed_precision=bf16")

    # --- training script + core args ---
    cmd.extend([
        train_script,
        f"--pretrained_model_name_or_path={args.model_dir}",
        f"--output_dir={args.output_dir}",
        f"--train_data_dir={data_pipeline}",
        f"--seed={args.seed}",
        f"--resolution={args.resolution}",
        f"--learning_rate={args.learning_rate}",
        f"--lr_scheduler={args.lr_scheduler}",
        f"--lr_warmup_steps={args.lr_warmup_steps}",
        f"--lr_num_cycles={args.lr_num_cycles}",
        f"--lr_power={args.lr_power}",
        f"--train_batch_size={args.train_batch_size}",
        f"--num_train_epochs={args.num_train_epochs}",
        f"--checkpointing_steps={args.checkpointing_steps}",
        f"--validation_steps={args.validation_steps}",
        f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
        f"--dataloader_num_workers={args.dataloader_num_workers}",
    ])

    # --- optional paths ---
    if args.validation_data_dir:
        cmd.extend([
            f"--validation_image={args.validation_data_dir}",
            f"--validation_prompt={args.validation_data_dir}",
        ])
    if args.controlnet_dir:
        cmd.append(f"--controlnet_model_name_or_path={args.controlnet_dir}")

    # --- A100 optimizations (enabled unless explicitly disabled) ---
    if not args.no_gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if not args.no_8bit_adam:
        cmd.append("--use_8bit_adam")
    if not args.no_tf32:
        cmd.append("--allow_tf32")
    if not args.no_set_grads_to_none:
        cmd.append("--set_grads_to_none")

    return cmd


def run_train(args):
    """Configure the data pipeline, then launch ControlNet training."""
    args = load_config(args)

    env = _get_env()

    # Step 1: Patch data_pipeline.py with the dataset directory
    pipeline_script = os.path.join(CONTROLNET_DIR, "training", "create_custom_data_pipeline.py")
    prep_cmd = [
        sys.executable, pipeline_script,
        "--data_dir", args.dataset_dir,
    ]
    print(f"[train] Configuring data pipeline:\n  {' '.join(prep_cmd)}")
    subprocess.run(prep_cmd, check=True, env=env)

    # Step 2: Build and run the accelerate launch command
    cmd = build_train_command(args)
    print(f"[train] Launching training:\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


# ---------------------------------------------------------------------------
# Subcommand: infer
# ---------------------------------------------------------------------------

def _build_infer_parser(subparsers):
    p = subparsers.add_parser(
        "infer",
        help="Generate images with a trained ControlNet.",
    )
    p.add_argument("--config", type=str, default=None,
                    help="Path to JSON config file. CLI args override config values.")
    p.add_argument("--controlnet_dir", type=str, default=None,
                    help="Path to trained ControlNet model directory.")
    p.add_argument("--stable_diffusion_dir", type=str, default=None,
                    help="Path to base Stable Diffusion model.")
    p.add_argument("--input_data_dir", type=str, default=None,
                    help="Path to input data (masks + prompt.jsonl).")
    p.add_argument("--mask_dir_name", type=str, default=None,
                    help="Name of the mask subdirectory inside input_data_dir.")
    p.add_argument("--prompt_file_name", type=str, default=None,
                    help="Name of the prompt JSONL file.")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--num_images_per_prompt", type=int, default=1)
    p.add_argument("--use_sdxl", action="store_true")
    p.add_argument("--vae_dir", type=str, default=None,
                    help="Path to VAE (only for SDXL).")
    return p


def build_infer_command(args):
    """Build the inference command list.

    Returns a list of strings suitable for ``subprocess.run()``.
    """
    script = os.path.join(CONTROLNET_DIR, "inference", "generate.py")
    cmd = [
        sys.executable, script,
        "--controlnet_dir", args.controlnet_dir,
        "--stable_diffusion_dir", args.stable_diffusion_dir,
        "--input_data_dir", args.input_data_dir,
        "--output_dir", args.output_dir,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_inference_steps", str(args.num_inference_steps),
        "--num_images_per_prompt", str(args.num_images_per_prompt),
    ]
    if args.mask_dir_name:
        cmd.extend(["--mask_dir_name", args.mask_dir_name])
    if args.prompt_file_name:
        cmd.extend(["--prompt_file_name", args.prompt_file_name])
    if args.use_sdxl:
        cmd.append("--use_sdxl")
        if args.vae_dir:
            cmd.extend(["--vae_dir", args.vae_dir])
    return cmd


def run_infer(args):
    """Run ControlNet inference to generate images."""
    args = load_config(args)
    cmd = build_infer_command(args)
    print(f"[infer] Running inference:\n  {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=_get_env())


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_env():
    """Return a copy of the environment with PROJECT_ROOT on PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + existing if existing else "")
    return env


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ControlNet Pipeline Runner — data prep, training, and inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    _build_prep_parser(subparsers)
    _build_train_parser(subparsers)
    _build_infer_parser(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {"prep": run_prep, "train": run_train, "infer": run_infer}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
