import importlib
import json
import shutil
import sys
import types
import uuid
from pathlib import Path

import pytest


def _load_semseg_main(monkeypatch):
    sys.modules.pop("semseg.main", None)

    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda *_args, **_kwargs: None
    fake_torch.load = lambda *_args, **_kwargs: {}
    fake_torch.device = lambda name: name
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda *_args, **_kwargs: None,
    )

    fake_pl = types.ModuleType("pytorch_lightning")
    fake_pl.seed_everything = lambda *_args, **_kwargs: None
    fake_pl.Trainer = object

    fake_callbacks = types.ModuleType("pytorch_lightning.callbacks")
    fake_callbacks.ModelCheckpoint = object

    fake_loggers = types.ModuleType("pytorch_lightning.loggers")
    fake_loggers.TensorBoardLogger = object

    fake_data_module = types.ModuleType("semseg.data.data_module")
    fake_data_module.MixedDataModule = object

    fake_model = types.ModuleType("semseg.model")
    fake_model.DeeplabV3Resnet101 = object

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pytorch_lightning", fake_pl)
    monkeypatch.setitem(sys.modules, "pytorch_lightning.callbacks", fake_callbacks)
    monkeypatch.setitem(sys.modules, "pytorch_lightning.loggers", fake_loggers)
    monkeypatch.setitem(sys.modules, "semseg.data.data_module", fake_data_module)
    monkeypatch.setitem(sys.modules, "semseg.model", fake_model)

    return importlib.import_module("semseg.main")


def _test_dir():
    root = Path(".tmp-semseg-tests") / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


def test_json_config_combines_with_cli_path_overrides(monkeypatch):
    semseg_main = _load_semseg_main(monkeypatch)

    root = _test_dir()
    try:
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "learning_rate": 0.0001,
                    "batch_size": 8,
                    "use_synthetic_images": True,
                    "mixed_batch": True,
                    "real_ratio": 0.75,
                    "height": 376,
                    "width": 1408,
                    "num_train_epochs": 20,
                    "gradient_accumulation_steps": 16,
                }
            ),
            encoding="utf-8",
        )

        output_dir = root / "output"
        args = semseg_main.parse_args(
            [
                "--load_json",
                str(config_path),
                "--train_image_dir",
                "/data/real/images",
                "--train_label_dir",
                "/data/real/labels",
                "--synthetic_image_dir",
                "/data/synth/images",
                "--synthetic_label_dir",
                "/data/synth/labels",
                "--val_image_dir",
                "/data/val/images",
                "--val_label_dir",
                "/data/val/labels",
                "--output_dir",
                str(output_dir),
            ]
        )

        assert args.batch_size == 8
        assert args.gradient_accumulation_steps == 16
        assert args.real_ratio == 0.75
        assert args.output_dir == str(output_dir)
        assert output_dir.is_dir()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_synthetic_training_requires_synthetic_paths(monkeypatch):
    semseg_main = _load_semseg_main(monkeypatch)

    root = _test_dir()
    with pytest.raises(FileNotFoundError, match="synthetic_image_dir, synthetic_label_dir"):
        try:
            semseg_main.parse_args(
                [
                    "--train_image_dir",
                    "/data/real/images",
                    "--train_label_dir",
                    "/data/real/labels",
                    "--val_image_dir",
                    "/data/val/images",
                    "--val_label_dir",
                    "/data/val/labels",
                    "--output_dir",
                    str(root / "output"),
                    "--use_synthetic_images",
                ]
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)
