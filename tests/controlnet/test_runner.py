"""Tests for controlnet.runner - command construction and config loading."""
import argparse
import json
import os
import tempfile
import pytest


def _make_train_args(**overrides):
    """Build a Namespace with A100-default train args. Override any key via kwargs."""
    defaults = dict(
        model_dir="/models/sd15",
        output_dir="/output",
        dataset_dir="/data",
        streaming=False,
        hf_dataset_name=None,
        dataset_revision=None,
        train_split="train",
        validation_split="validation",
        dataset_num_samples=None,
        streaming_shuffle_buffer=10000,
        streaming_validation_samples=3,
        controlnet_dir=None,
        validation_data_dir=None,
        resolution=512,
        train_batch_size=4,
        num_train_epochs=1,
        learning_rate=5e-6,
        lr_scheduler="constant",
        lr_warmup_steps=500,
        lr_num_cycles=1,
        lr_power=1.0,
        checkpointing_steps=500,
        validation_steps=100,
        seed=42,
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
        max_train_samples=None,
        checkpoints_total_limit=None,
        resume_from_checkpoint=None,
        cuda_memory_fraction=None,
        enable_xformers=False,
        multi_gpu=False,
        num_gpus=1,
        no_gradient_checkpointing=False,
        no_8bit_adam=False,
        no_tf32=False,
        no_set_grads_to_none=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBuildTrainCommand:
    """Tests for build_train_command()."""

    def test_a100_defaults_present(self):
        """bf16, gradient checkpointing, 8-bit Adam, TF32, set_grads_to_none are all enabled."""
        from controlnet.runner import build_train_command

        cmd = build_train_command(_make_train_args())
        cmd_str = " ".join(cmd)

        assert "--mixed_precision=bf16" in cmd_str
        assert "--gradient_checkpointing" in cmd_str
        assert "--use_8bit_adam" in cmd_str
        assert "--allow_tf32" in cmd_str
        assert "--set_grads_to_none" in cmd_str

    def test_optimizations_can_be_disabled(self):
        """Each A100 optimization can be individually turned off."""
        from controlnet.runner import build_train_command

        cmd = build_train_command(_make_train_args(
            no_gradient_checkpointing=True,
            no_8bit_adam=True,
            no_tf32=True,
            no_set_grads_to_none=True,
        ))
        cmd_str = " ".join(cmd)

        assert "--gradient_checkpointing" not in cmd_str
        assert "--use_8bit_adam" not in cmd_str
        assert "--allow_tf32" not in cmd_str
        assert "--set_grads_to_none" not in cmd_str
        # bf16 is always on (A100 native)
        assert "--mixed_precision=bf16" in cmd_str

    def test_multi_gpu_flags(self):
        """Multi-GPU adds --multi_gpu and --num_processes."""
        from controlnet.runner import build_train_command

        cmd = build_train_command(_make_train_args(multi_gpu=True, num_gpus=4))
        cmd_str = " ".join(cmd)

        assert "--multi_gpu" in cmd_str
        assert "--num_processes=4" in cmd_str

    def test_single_gpu_no_multi_flags(self):
        """Single-GPU mode omits --multi_gpu and --num_processes."""
        from controlnet.runner import build_train_command

        cmd = build_train_command(_make_train_args(multi_gpu=False))
        cmd_str = " ".join(cmd)

        assert "--multi_gpu" not in cmd_str
        assert "--num_processes" not in cmd_str

    def test_controlnet_resume(self):
        """--controlnet_model_name_or_path is included when controlnet_dir is set."""
        from controlnet.runner import build_train_command

        cmd = build_train_command(_make_train_args(controlnet_dir="/ckpts/controlnet"))
        cmd_str = " ".join(cmd)

        assert "--controlnet_model_name_or_path=/ckpts/controlnet" in cmd_str

    def test_validation_data(self):
        """Validation image/prompt flags point to validation_data_dir when set."""
        from controlnet.runner import build_train_command

        cmd = build_train_command(_make_train_args(validation_data_dir="/data/val"))
        cmd_str = " ".join(cmd)

        assert "--validation_image=/data/val" in cmd_str
        assert "--validation_prompt=/data/val" in cmd_str

    def test_core_training_params_forwarded(self):
        """Core params (LR, batch size, epochs, resolution) are forwarded correctly."""
        from controlnet.runner import build_train_command

        cmd = build_train_command(_make_train_args(
            learning_rate=1e-5,
            train_batch_size=8,
            num_train_epochs=50,
            resolution=768,
        ))
        cmd_str = " ".join(cmd)

        assert "--learning_rate=1e-05" in cmd_str
        assert "--train_batch_size=8" in cmd_str
        assert "--num_train_epochs=50" in cmd_str
        assert "--resolution=768" in cmd_str

    def test_shared_gpu_controls_forwarded(self):
        """Memory cap, sample limit, checkpoint limit, resume, and xFormers are forwarded."""
        from controlnet.runner import build_train_command

        cmd = build_train_command(_make_train_args(
            max_train_samples=32,
            checkpoints_total_limit=3,
            resume_from_checkpoint="latest",
            cuda_memory_fraction=0.85,
            enable_xformers=True,
        ))
        cmd_str = " ".join(cmd)

        assert "--max_train_samples=32" in cmd_str
        assert "--checkpoints_total_limit=3" in cmd_str
        assert "--resume_from_checkpoint=latest" in cmd_str
        assert "--cuda_memory_fraction=0.85" in cmd_str
        assert "--enable_xformers_memory_efficient_attention" in cmd_str

    def test_streaming_dataset_flags(self):
        """Streaming mode uses --dataset_name and does not pass the local data pipeline."""
        from controlnet.runner import build_train_command

        cmd = build_train_command(_make_train_args(
            streaming=True,
            hf_dataset_name="user/private-kitti360-controlnet",
            dataset_revision="abc123",
            dataset_num_samples=5000,
            train_split="train",
            validation_split="validation",
            streaming_shuffle_buffer=2048,
            streaming_validation_samples=2,
        ))
        cmd_str = " ".join(cmd)

        assert "--streaming" in cmd_str
        assert "--dataset_name=user/private-kitti360-controlnet" in cmd_str
        assert "--dataset_revision=abc123" in cmd_str
        assert "--dataset_num_samples=5000" in cmd_str
        assert "--streaming_shuffle_buffer=2048" in cmd_str
        assert "--streaming_validation_samples=2" in cmd_str
        assert "--train_data_dir=" not in cmd_str


class TestBuildInferCommand:
    """Tests for build_infer_command()."""

    def test_basic_infer_command(self):
        """Inference command includes all required paths."""
        from controlnet.runner import build_infer_command

        args = argparse.Namespace(
            controlnet_dir="/ckpts/controlnet",
            stable_diffusion_dir="/models/sd15",
            input_data_dir="/data/test",
            mask_dir_name=None,
            prompt_file_name=None,
            output_dir="./output",
            height=512,
            width=512,
            num_inference_steps=20,
            num_images_per_prompt=1,
            batch_size=1,
            dtype="auto",
            cuda_memory_fraction=None,
            enable_xformers=False,
            enable_attention_slicing=False,
            enable_vae_slicing=False,
            channels_last=False,
            skip_existing=False,
            start_index=0,
            end_index=None,
            max_samples=None,
            seed=None,
            manifest_file="generation_manifest.jsonl",
            use_sdxl=False,
            vae_dir=None,
        )
        cmd = build_infer_command(args)
        cmd_str = " ".join(cmd)

        assert "--controlnet_dir" in cmd_str
        assert "--stable_diffusion_dir" in cmd_str
        assert "--input_data_dir" in cmd_str

    def test_sdxl_flags(self):
        """SDXL mode adds --use_sdxl and --vae_dir."""
        from controlnet.runner import build_infer_command

        args = argparse.Namespace(
            controlnet_dir="/ckpts/controlnet",
            stable_diffusion_dir="/models/sdxl",
            input_data_dir="/data/test",
            mask_dir_name=None,
            prompt_file_name=None,
            output_dir="./output",
            height=1024,
            width=1024,
            num_inference_steps=20,
            num_images_per_prompt=1,
            batch_size=1,
            dtype="auto",
            cuda_memory_fraction=None,
            enable_xformers=False,
            enable_attention_slicing=False,
            enable_vae_slicing=False,
            channels_last=False,
            skip_existing=False,
            start_index=0,
            end_index=None,
            max_samples=None,
            seed=None,
            manifest_file="generation_manifest.jsonl",
            use_sdxl=True,
            vae_dir="/models/sdxl-vae",
        )
        cmd = build_infer_command(args)
        cmd_str = " ".join(cmd)

        assert "--use_sdxl" in cmd_str
        assert "--vae_dir" in cmd_str

    def test_shared_gpu_inference_controls(self):
        """Inference forwards batch, dtype, memory cap, and memory-saving flags."""
        from controlnet.runner import build_infer_command

        args = argparse.Namespace(
            controlnet_dir="/ckpts/controlnet",
            stable_diffusion_dir="/models/sd15",
            input_data_dir="/data/test",
            mask_dir_name="mask",
            prompt_file_name="prompt.jsonl",
            output_dir="./output",
            height=512,
            width=512,
            num_inference_steps=20,
            num_images_per_prompt=1,
            batch_size=1,
            dtype="fp16",
            cuda_memory_fraction=0.85,
            enable_xformers=True,
            enable_attention_slicing=True,
            enable_vae_slicing=True,
            channels_last=True,
            skip_existing=True,
            start_index=5,
            end_index=20,
            max_samples=4,
            seed=123,
            manifest_file="manifest.jsonl",
            use_sdxl=False,
            vae_dir=None,
        )
        cmd = build_infer_command(args)
        cmd_str = " ".join(cmd)

        assert "--batch_size 1" in cmd_str
        assert "--dtype fp16" in cmd_str
        assert "--cuda_memory_fraction 0.85" in cmd_str
        assert "--enable_xformers" in cmd_str
        assert "--enable_attention_slicing" in cmd_str
        assert "--enable_vae_slicing" in cmd_str
        assert "--channels_last" in cmd_str
        assert "--skip_existing" in cmd_str
        assert "--start_index 5" in cmd_str
        assert "--end_index 20" in cmd_str
        assert "--max_samples 4" in cmd_str
        assert "--seed 123" in cmd_str
        assert "--manifest_file manifest.jsonl" in cmd_str


class TestConfigLoading:
    """Tests for config loading via parse_args()."""

    def test_config_overrides_defaults(self):
        """JSON config values override argparse defaults."""
        from controlnet.runner import parse_args

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"num_train_epochs": 100, "seed": 123}, f)
            config_path = f.name

        try:
            _, args = parse_args(["train", "--config", config_path])
            assert args.num_train_epochs == 100
            assert args.seed == 123
        finally:
            os.unlink(config_path)

    def test_cli_overrides_config(self):
        """Explicit CLI values override config values."""
        from controlnet.runner import parse_args

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"num_train_epochs": 100, "seed": 123}, f)
            config_path = f.name

        try:
            _, args = parse_args(["train", "--config", config_path, "--seed", "456"])
            assert args.num_train_epochs == 100
            assert args.seed == 456
        finally:
            os.unlink(config_path)

    def test_unknown_config_key_raises(self):
        """Unknown config keys raise a ValueError."""
        from controlnet.runner import parse_args

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"unknown_key_xyz": "value"}, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Unknown config key: unknown_key_xyz"):
                parse_args(["train", "--config", config_path])
        finally:
            os.unlink(config_path)

    def test_no_config_uses_defaults(self):
        """When no config is provided, defaults are used."""
        from controlnet.runner import parse_args

        _, args = parse_args(["train"])
        assert args.num_train_epochs == 1  # Default value
