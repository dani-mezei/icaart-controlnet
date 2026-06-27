import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np

from semseg.data.dataset.kitti_dataset import KittiDataset


def _test_dir():
    root = Path(".tmp-semseg-dataset-tests") / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


def test_kitti_dataset_ignores_non_image_entries():
    root = _test_dir()
    try:
        image_dir = root / "images"
        label_dir = root / "labels"
        image_dir.mkdir()
        label_dir.mkdir()

        cv2.imwrite(str(image_dir / "sample.png"), np.zeros((2, 2, 3), dtype=np.uint8))
        cv2.imwrite(str(label_dir / "sample.png"), np.zeros((2, 2), dtype=np.uint8))
        (image_dir / "prompt.jsonl").write_text("{}", encoding="utf-8")
        (label_dir / "README.txt").write_text("ignore me", encoding="utf-8")

        dataset = KittiDataset(image_dir, label_dir)

        assert len(dataset) == 1
        image, label = dataset[0]
        assert image.shape == (2, 2, 3)
        assert label.shape == (2, 2)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_kitti_dataset_skips_unreadable_image():
    root = _test_dir()
    try:
        image_dir = root / "images"
        label_dir = root / "labels"
        image_dir.mkdir()
        label_dir.mkdir()

        (image_dir / "bad.png").write_text("not an image", encoding="utf-8")
        cv2.imwrite(str(label_dir / "bad.png"), np.zeros((2, 2), dtype=np.uint8))
        cv2.imwrite(str(image_dir / "good.png"), np.ones((2, 2, 3), dtype=np.uint8))
        cv2.imwrite(str(label_dir / "good.png"), np.ones((2, 2), dtype=np.uint8))

        dataset = KittiDataset(image_dir, label_dir)
        image, label = dataset[0]

        assert image.shape == (2, 2, 3)
        assert label.shape == (2, 2)
        assert label.max().item() == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
