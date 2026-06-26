#!/usr/bin/env python
"""Generate HED edge maps for ControlNet conditioning.

Mirrors the structure of ``controlnet/inference/generate.py`` (argparse +
validation, sample building, skip_existing / range filtering, and a JSONL
manifest), but instead of running a diffusion pipeline it produces HED
(Holistically-Nested Edge Detection) maps from a folder of source images
(e.g. KITTI). The output folder can be used directly as the ``mask/``
conditioning directory of a ControlNet dataset.

Install:
    pip install controlnet_aux

The HED weights (``lllyasviel/Annotators`` -> ``ControlNetHED.pth``) download
from the Hugging Face Hub on first run and cache under
``~/.cache/huggingface``. The machine needs network access the first time
(or pre-seed the cache).

Usage (standalone):
    python generate_hed.py \
        --input_data_dir ~/data/controlnet_dataset/train \
        --image_dir_name image \
        --output_dir ~/data/controlnet_dataset/train \
        --mask_dir_name hed \
        --detect_resolution 512 \
        --skip_existing

Usage (as a module inside the repo, e.g. controlnet/preprocessing/generate_hed.py):
    python -m controlnet.preprocessing.generate_hed --input_data_dir ... --image_dir_name image
"""
import argparse
import json
import os
import time

from PIL import Image
from tqdm.auto import tqdm

# Reuse the repo helpers if available; otherwise fall back to local versions so
# the script also runs standalone outside the controlnet package.
try:
    from controlnet.custom.utils import validate_dir, create_dir_if_not_exists
except Exception:  # pragma: no cover - trivial fallbacks
    def validate_dir(path):
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(f"Directory does not exist: {path}")

    def create_dir_if_not_exists(path):
        os.makedirs(path, exist_ok=True)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Generate HED edge maps for ControlNet conditioning."
    )

    # --- I/O ---
    parser.add_argument("--input_data_dir", type=str, required=True,
                        help="Directory containing the source images (or holding --image_dir_name).")
    parser.add_argument("--image_dir_name", type=str, default=None,
                        help="Subfolder of --input_data_dir holding source images. "
                             "If omitted, images are read directly from --input_data_dir.")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to write HED maps into. Default is ./output.")
    parser.add_argument("--mask_dir_name", type=str, default="hed",
                        help="Subfolder of --output_dir to write HED maps into. "
                             "Set to '' to write directly into --output_dir.")
    parser.add_argument("--output_ext", type=str, default=".png",
                        help="Output file extension. .png is lossless and recommended for edges.")

    # --- HED settings ---
    parser.add_argument("--detect_resolution", type=str, default="512",
                        help="Resolution (shorter side) the HED model runs at internally. "
                             "An integer, or 'native' to use each image's own shorter side "
                             "(minimal resize: only the wrapper's unavoidable 64px rounding). Default 512.")
    parser.add_argument("--output_size", type=str, default="native",
                        choices=["native", "detect", "fixed"],
                        help="native = resize the HED map back to each source image's size (recommended for "
                             "pixel-aligned conditioning); detect = keep the detector's output size; "
                             "fixed = use --width/--height for every image.")
    parser.add_argument("--width", type=int, default=None,
                        help="Output width when --output_size fixed.")
    parser.add_argument("--height", type=int, default=None,
                        help="Output height when --output_size fixed.")
    parser.add_argument("--resample", type=str, default="bilinear",
                        choices=["bilinear", "bicubic", "lanczos", "nearest"],
                        help="Resampling filter used when resizing the HED map. Default bilinear.")
    parser.add_argument("--safe", action="store_true",
                        help="HED safe mode: fewer high-frequency artifacts, slightly less fine detail.")
    parser.add_argument("--scribble", action="store_true",
                        help="Produce thinned, scribble-style edges instead of dense soft edges.")
    parser.add_argument("--annotator_repo", type=str, default="lllyasviel/Annotators",
                        help="Hugging Face repo to load the HED weights from.")

    # --- Runtime ---
    parser.add_argument("--device", type=str, default=None,
                        help="cuda, cpu, or cuda:N. Auto-detected if omitted.")

    # --- Selection / resumption (parity with generate.py) ---
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images whose output file already exists.")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start index, inclusive, after sorting inputs.")
    parser.add_argument("--end_index", type=int, default=None,
                        help="End index, exclusive, after sorting inputs.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of images to process after applying start/end indices.")
    parser.add_argument("--manifest_file", type=str, default="hed_manifest.jsonl",
                        help="JSONL manifest filename or path. Relative paths land inside --output_dir.")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Log per-image failures to the manifest and keep going instead of aborting.")

    args = parser.parse_args(input_args)

    # --- Validation (mirrors generate.py) ---
    validate_dir(args.input_data_dir)
    image_dir = args.input_data_dir if not args.image_dir_name \
        else os.path.join(args.input_data_dir, args.image_dir_name)
    validate_dir(image_dir)
    args.image_dir = image_dir

    if isinstance(args.detect_resolution, str) and args.detect_resolution.lower() == "native":
        args.detect_resolution = "native"
    else:
        try:
            args.detect_resolution = int(args.detect_resolution)
        except (TypeError, ValueError):
            raise ValueError("--detect_resolution must be an integer or 'native'.")
        if args.detect_resolution < 64:
            raise ValueError("--detect_resolution should be >= 64.")
    if args.output_size == "fixed":
        if args.width is None or args.height is None:
            raise ValueError("--width and --height are required when --output_size fixed.")
        if args.width % 8 != 0 or args.height % 8 != 0:
            raise ValueError("--width and --height should be multiples of 8 for ControlNet.")
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative.")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end_index must be >= --start_index.")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max_samples must be > 0.")
    if not args.output_ext.startswith("."):
        args.output_ext = "." + args.output_ext

    out_mask_dir = args.output_dir if not args.mask_dir_name \
        else os.path.join(args.output_dir, args.mask_dir_name)
    args.output_mask_dir = os.path.abspath(out_mask_dir)
    create_dir_if_not_exists(args.output_mask_dir)

    args.output_dir = os.path.abspath(args.output_dir)
    if not os.path.isabs(args.manifest_file):
        args.manifest_file = os.path.join(args.output_dir, args.manifest_file)

    return args


# ---------------------------------------------------------------------------
# Sample handling (parity with generate.py)
# ---------------------------------------------------------------------------

def list_image_names(image_dir):
    names = [
        name for name in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, name))
        and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    return sorted(names)


def build_samples(image_names, image_dir, output_mask_dir, output_ext):
    samples = []
    for index, name in enumerate(image_names):
        stem = os.path.splitext(name)[0]
        samples.append({
            "index": index,
            "image": name,
            "input_path": os.path.join(image_dir, name),
            "output_path": os.path.join(output_mask_dir, stem + output_ext),
        })
    return samples


def filter_samples(samples, start_index=0, end_index=None, max_samples=None):
    selected = samples[start_index:end_index]
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def filter_existing_samples(samples, skip_existing=False):
    if not skip_existing:
        return samples
    return [s for s in samples if not os.path.exists(s["output_path"])]


def manifest_row(sample, status, elapsed_seconds=None, size=None, error=None):
    return {
        "image": sample["image"],
        "output_path": sample["output_path"],
        "status": status,
        "size": size,
        "elapsed_seconds": elapsed_seconds,
        "error": error,
    }


def append_manifest(manifest_file, row):
    create_dir_if_not_exists(os.path.dirname(os.path.abspath(manifest_file)))
    with open(manifest_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# HED processing
# ---------------------------------------------------------------------------

_RESAMPLE = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}


def resolve_device(requested):
    import torch
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def target_size(args, src_w, src_h):
    if args.output_size == "native":
        return src_w, src_h
    if args.output_size == "fixed":
        return args.width, args.height
    return None  # "detect": leave detector output untouched


def process_image(hed, sample, args, resample):
    image = Image.open(sample["input_path"]).convert("RGB")
    src_w, src_h = image.size

    # "native" => run detection at this image's own shorter side, so the only
    # pre-detection resize is the wrapper's unavoidable rounding to a multiple
    # of 64; the post-resize then restores the exact source dimensions.
    detect_res = min(src_w, src_h) if args.detect_resolution == "native" else args.detect_resolution

    edge = hed(
        image,
        detect_resolution=detect_res,
        image_resolution=detect_res,
        safe=args.safe,
        scribble=args.scribble,
        output_type="pil",
    )

    size = target_size(args, src_w, src_h)
    if size is not None and edge.size != tuple(size):
        edge = edge.resize(tuple(size), resample)

    # ControlNet conditioning is expected as 3-channel RGB; the training
    # pipeline calls .convert("RGB") on conditioning images anyway.
    edge = edge.convert("RGB")
    edge.save(sample["output_path"])
    return edge.size


def main(args):
    import torch  # noqa: F401  (device resolution; kept lazy like generate.py's heavy imports)
    from controlnet_aux import HEDdetector

    device = resolve_device(args.device)
    resample = _RESAMPLE[args.resample]

    print(f"[hed] Loading HED detector from {args.annotator_repo} onto {device} ...")
    hed = HEDdetector.from_pretrained(args.annotator_repo)
    hed = hed.to(device)

    image_names = list_image_names(args.image_dir)
    if not image_names:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    samples = build_samples(image_names, args.image_dir, args.output_mask_dir, args.output_ext)
    total_found = len(samples)
    samples = filter_samples(samples, args.start_index, args.end_index, args.max_samples)
    samples = filter_existing_samples(samples, args.skip_existing)

    print(f"[hed] {total_found} images found, {len(samples)} selected for processing.")
    print(f"[hed] Writing HED maps to {args.output_mask_dir}")

    succeeded = 0
    failed = 0
    for sample in tqdm(samples, desc="HED", unit="img"):
        start_time = time.perf_counter()
        try:
            out_size = process_image(hed, sample, args, resample)
            elapsed = time.perf_counter() - start_time
            append_manifest(
                args.manifest_file,
                manifest_row(sample, "success", elapsed_seconds=elapsed, size=list(out_size)),
            )
            succeeded += 1
        except Exception as ex:  # noqa: BLE001
            elapsed = time.perf_counter() - start_time
            append_manifest(
                args.manifest_file,
                manifest_row(sample, "error", elapsed_seconds=elapsed, error=str(ex)),
            )
            failed += 1
            if not args.continue_on_error:
                raise

    print(f"[hed] Done. {succeeded} succeeded, {failed} failed. Manifest: {args.manifest_file}")


if __name__ == "__main__":
    main(parse_args())
