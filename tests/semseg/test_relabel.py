import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np

from semseg.preprocess.relabel import invalid_train_ids, relabel_directory


def _test_dir():
    root = Path(".tmp-semseg-relabel-tests") / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


def test_relabel_directory_maps_kitti_ids_to_train_ids():
    root = _test_dir()
    try:
        input_dir = root / "raw"
        output_dir = root / "label_19"
        input_dir.mkdir()

        raw = np.array(
            [
                [7, 8, 11, 17, 19],
                [20, 21, 22, 23, 24],
                [25, 26, 27, 28, 31],
                [32, 33, 6, 38, 41],
            ],
            dtype=np.uint8,
        )
        expected = np.array(
            [
                [0, 1, 2, 5, 6],
                [7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16],
                [17, 18, 255, 255, 255],
            ],
            dtype=np.uint8,
        )

        cv2.imwrite(str(input_dir / "sample.png"), raw)

        processed = relabel_directory(input_dir, output_dir)
        relabeled = cv2.imread(str(output_dir / "sample.png"), cv2.IMREAD_GRAYSCALE)

        assert processed == 1
        np.testing.assert_array_equal(relabeled, expected)
        assert invalid_train_ids(relabeled) == []
    finally:
        shutil.rmtree(root, ignore_errors=True)
