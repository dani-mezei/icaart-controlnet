import argparse
import json
import os
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and optionally upload a private Hugging Face ImageFolder dataset for streaming."
    )
    parser.add_argument("--train_image_dir", required=True)
    parser.add_argument("--train_mask_dir", required=True)
    parser.add_argument("--validation_image_dir", default=None)
    parser.add_argument("--validation_mask_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repo_id", default=None, help="Optional HF dataset repo id, e.g. user/kitti360-controlnet.")
    parser.add_argument("--private", action="store_true", help="Create/upload the dataset repo as private.")
    parser.add_argument("--prompt", default="a high quality photo of an urban driving scene")
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_validation", type=int, default=None)
    parser.add_argument("--link", action="store_true", help="Use symlinks instead of copying files.")
    return parser.parse_args()


def list_images(directory):
    paths = {}
    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths[path.name] = path.resolve()
    return paths


def stage_file(src, dst, link):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def write_split(split, image_dir, mask_dir, output_dir, prompt, limit, link):
    images = list_images(image_dir)
    masks = list_images(mask_dir)
    names = sorted(set(images) & set(masks))
    if limit is not None:
        names = names[:limit]
    if not names:
        raise ValueError(f"No matching image/mask filenames found for split {split}.")

    split_dir = output_dir / split
    out_images = split_dir / "images"
    out_masks = split_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    metadata_path = split_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata:
        for name in names:
            stage_file(images[name], out_images / name, link)
            stage_file(masks[name], out_masks / name, link)
            row = {
                "image_file_name": f"images/{name}",
                "mask_file_name": f"masks/{name}",
                "prompt": prompt,
            }
            metadata.write(json.dumps(row) + "\n")

    return names


def upload_dataset(repo_id, output_dir, private):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        commit_message="Upload ControlNet streaming dataset",
    )
    return {
        "revision": getattr(commit, "oid", None),
        "url": getattr(commit, "commit_url", str(commit)),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_names = write_split(
        "train",
        Path(args.train_image_dir),
        Path(args.train_mask_dir),
        output_dir,
        args.prompt,
        args.limit_train,
        args.link,
    )

    validation_names = []
    if args.validation_image_dir and args.validation_mask_dir:
        validation_names = write_split(
            "validation",
            Path(args.validation_image_dir),
            Path(args.validation_mask_dir),
            output_dir,
            args.prompt,
            args.limit_validation,
            args.link,
        )

    upload_result = None
    if args.repo_id:
        upload_result = upload_dataset(args.repo_id, output_dir, args.private)

    manifest = {
        "repo_id": args.repo_id,
        "dataset_revision": upload_result["revision"] if upload_result else None,
        "upload_result": upload_result,
        "train_samples": len(train_names),
        "validation_samples": len(validation_names),
        "prompt": args.prompt,
        "train_names": train_names,
        "validation_names": validation_names,
    }
    manifest_path = output_dir / "streaming_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Prepared {len(train_names)} train samples and {len(validation_names)} validation samples.")
    print(f"Manifest: {manifest_path}")
    if upload_result:
        print(f"Dataset revision: {upload_result['revision']}")
        print(f"Upload URL: {upload_result['url']}")


if __name__ == "__main__":
    main()
