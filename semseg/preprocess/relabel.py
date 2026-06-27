import argparse
from pathlib import Path

import cv2
import numpy as np

from semseg.labels import labels


VALID_TRAIN_IDS = set(range(19)) | {255}
ID_TO_TRAIN_ID = {label.id: label.trainId for label in labels if label.id != -1}


def relabel_array(label_image):
    train_id_image = np.full(label_image.shape, 255, dtype=np.uint8)
    for raw_id, train_id in ID_TO_TRAIN_ID.items():
        train_id_image[label_image == raw_id] = train_id
    return train_id_image


def invalid_train_ids(label_image):
    values = np.unique(label_image)
    return [int(value) for value in values if int(value) not in VALID_TRAIN_IDS]


def relabel_file(input_path, output_path):
    label_image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if label_image is None:
        raise ValueError(f"Failed to load label image: {input_path}")

    train_id_image = relabel_array(label_image)
    invalid = invalid_train_ids(train_id_image)
    if invalid:
        raise ValueError(f"Relabeled image still has invalid train IDs: {invalid}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), train_id_image):
        raise ValueError(f"Failed to write relabeled image: {output_path}")

    return train_id_image


def relabel_directory(input_dir, output_dir, pattern="*.png"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    input_paths = sorted(input_dir.glob(pattern))
    if not input_paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {input_dir}")

    processed = 0
    for input_path in input_paths:
        output_path = output_dir / input_path.name
        relabel_file(input_path, output_path)
        processed += 1

    return processed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert KITTI-360 raw semantic label IDs to 19-class trainId PNGs."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Directory containing raw single-channel KITTI-360 semantic label PNGs.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where 19-class trainId PNGs will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.png",
        help="Input file glob pattern relative to input_dir. Defaults to *.png.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processed = relabel_directory(args.input_dir, args.output_dir, args.pattern)
    print(
        f"Relabeled {processed} files from {args.input_dir} to {args.output_dir}. "
        "Valid train IDs are 0..18 and 255."
    )


if __name__ == "__main__":
    main()
