import importlib.util
import json
from pathlib import Path
import sys
import types
import uuid


def load_generate_module():
    diffusers = types.ModuleType("diffusers")
    diffusers.ControlNetModel = object
    diffusers.AutoencoderKL = object
    diffusers.UniPCMultistepScheduler = object
    diffusers.StableDiffusionControlNetPipeline = object
    diffusers.StableDiffusionXLControlNetPipeline = object
    diffusers_utils = types.ModuleType("diffusers.utils")
    diffusers_utils.load_image = lambda path: path
    sys.modules.setdefault("diffusers", diffusers)
    sys.modules.setdefault("diffusers.utils", diffusers_utils)

    spec = importlib.util.spec_from_file_location(
        "generate_helpers", "controlnet/inference/generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_workspace_tmp():
    path = Path("scratch") / f"pytest-generate-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_filter_samples_range_and_max():
    generate = load_generate_module()
    samples = [{"index": i, "image": f"{i}.png"} for i in range(10)]

    selected = generate.filter_samples(samples, start_index=2, end_index=8, max_samples=3)

    assert [sample["index"] for sample in selected] == [2, 3, 4]


def test_filter_existing_samples():
    generate = load_generate_module()
    tmp_path = make_workspace_tmp()
    existing = tmp_path / "exists.png"
    existing.write_text("x")
    samples = [
        {"index": 0, "output_path": str(existing)},
        {"index": 1, "output_path": str(tmp_path / "missing.png")},
    ]

    selected = generate.filter_existing_samples(samples, skip_existing=True)

    assert [sample["index"] for sample in selected] == [1]


def test_manifest_row_and_append():
    generate = load_generate_module()
    tmp_path = make_workspace_tmp()
    manifest = tmp_path / "manifest.jsonl"
    sample = {"image": "0001.png", "prompt": "road", "output_path": "/out/0001.png"}

    row = generate.manifest_row(sample, "success", seed=123, elapsed_seconds=1.5)
    generate.append_manifest(str(manifest), row)

    saved = json.loads(manifest.read_text().strip())
    assert saved["image"] == "0001.png"
    assert saved["prompt"] == "road"
    assert saved["seed"] == 123
    assert saved["status"] == "success"
    assert saved["error"] is None


def test_sample_seed():
    generate = load_generate_module()

    assert generate.sample_seed(100, {"index": 5}) == 105
    assert generate.sample_seed(None, {"index": 5}) is None
