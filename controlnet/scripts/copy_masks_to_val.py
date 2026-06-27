import json
import random
import shutil
from pathlib import Path


# Set these paths before running.
TRAIN_DIR = Path(r"C:\path\to\train")
VAL_DIR = Path(r"C:\path\to\val")

# Randomly copy this many masks from train/mask into val/images.
NUM_MASKS = 5

# Set to an integer for repeatable random picks, or None for different picks each run.
RANDOM_SEED = None

MASK_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_masks(path):
    return sorted(
        file for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in MASK_EXTENSIONS
    )


def main():
    train_mask_dir = TRAIN_DIR / "mask"
    val_images_dir = VAL_DIR / "images"
    train_prompt_path = TRAIN_DIR / "prompt.jsonl"
    val_prompt_path = VAL_DIR / "prompt.jsonl"

    if not train_mask_dir.exists():
        raise FileNotFoundError(f"Missing train mask directory: {train_mask_dir}")

    val_images_dir.mkdir(parents=True, exist_ok=True)

    train_prompts = read_jsonl(train_prompt_path)
    val_prompts = read_jsonl(val_prompt_path)
    val_by_image = {row["image"]: row for row in val_prompts if "image" in row}

    masks = list_masks(train_mask_dir)
    if len(masks) < NUM_MASKS:
        raise ValueError(f"Asked for {NUM_MASKS} masks, but only found {len(masks)} in {train_mask_dir}")

    rng = random.Random(RANDOM_SEED)
    selected_masks = rng.sample(masks, NUM_MASKS)

    for source_mask in selected_masks:
        name = source_mask.name

        matching_rows = [
            row for row in train_prompts
            if row.get("mask", row.get("image")) == name
        ]
        if not matching_rows:
            raise ValueError(f"No prompt.jsonl row found for mask: {name}")

        shutil.copy2(source_mask, val_images_dir / name)

        val_by_image[name] = {
            "image": name,
            "prompt": matching_rows[0]["prompt"],
        }

    write_jsonl(val_prompt_path, val_by_image.values())
    print(f"Copied {len(selected_masks)} masks and updated {val_prompt_path}")
    print("Selected masks:")
    for mask in selected_masks:
        print(f"  {mask.name}")


if __name__ == "__main__":
    main()
