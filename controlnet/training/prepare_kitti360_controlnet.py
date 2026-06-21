import argparse
import json
import os
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage extracted KITTI-360 RGB images and semantic masks for ControlNet training."
    )
    parser.add_argument("--image_dir", required=True, help="Directory containing KITTI-360 RGB frames.")
    parser.add_argument("--mask_dir", required=True, help="Directory containing matching semantic mask frames.")
    parser.add_argument("--output_dir", required=True, help="Output directory with image/, mask/, prompt.jsonl.")
    parser.add_argument(
        "--prompt",
        default="a high quality photo of an urban driving scene",
        help="Prompt used for every row. Replace prompt.jsonl later if using BLIP2 captions.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of matched samples.")
    parser.add_argument(
        "--link",
        action="store_true",
        help="Create symlinks instead of copying files. Recommended on Linux scratch storage.",
    )
    return parser.parse_args()


def list_images(directory):
    paths = {}
    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths[path.name] = path
    return paths


def stage_file(src, dst, link):
    if dst.exists():
        dst.unlink()
    if link:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    args = parse_args()

    image_dir = Path(args.image_dir).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    out_images = output_dir / "image"
    out_masks = output_dir / "mask"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    masks = list_images(mask_dir)
    names = sorted(set(images) & set(masks))
    if args.limit is not None:
        names = names[: args.limit]
    if not names:
        raise ValueError("No matching image/mask filenames found.")

    prompt_path = output_dir / "prompt.jsonl"
    with prompt_path.open("w", encoding="utf-8") as prompt_file:
        for name in names:
            stage_file(images[name], out_images / name, args.link)
            stage_file(masks[name], out_masks / name, args.link)
            row = {"image": name, "mask": name, "prompt": args.prompt}
            prompt_file.write(json.dumps(row) + "\n")

    print(f"Prepared {len(names)} samples in {output_dir}")


if __name__ == "__main__":
    main()
