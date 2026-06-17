"""Tests for controlnet.runner — command construction and config loading."""
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
            use_sdxl=True,
            vae_dir="/models/sdxl-vae",
        )
        cmd = build_infer_command(args)
        cmd_str = " ".join(cmd)

        assert "--use_sdxl" in cmd_str
        assert "--vae_dir" in cmd_str


class TestLoadConfig:
    """Tests for load_config()."""

    def test_config_fills_none_values(self):
        """JSON config values fill in attributes that are None on the Namespace."""
        from controlnet.runner import load_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model_dir": "/from/config", "seed": 123}, f)
            config_path = f.name

        try:
            args = argparse.Namespace(config=config_path, model_dir=None, seed=None)
            result = load_config(args)
            assert result.model_dir == "/from/config"
            assert result.seed == 123
        finally:
            os.unlink(config_path)

    def test_cli_overrides_config(self):
        """Explicit CLI values (non-None) are not overwritten by config."""
        from controlnet.runner import load_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model_dir": "/from/config", "seed": 123}, f)
            config_path = f.name

        try:
            args = argparse.Namespace(config=config_path, model_dir="/from/cli", seed=None)
            result = load_config(args)
            assert result.model_dir == "/from/cli"
            assert result.seed == 123
        finally:
            os.unlink(config_path)

    def test_no_config_is_noop(self):
        """When config is None, args are returned unchanged."""
        from controlnet.runner import load_config

        args = argparse.Namespace(config=None, model_dir="/test")
        result = load_config(args)
        assert result.model_dir == "/test"
