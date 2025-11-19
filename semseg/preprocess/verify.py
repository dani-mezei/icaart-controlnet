import os
from collections import defaultdict

import cv2
import numpy as np

# Input directory containing ground truth images
input_dir = "C:/Users/MED7CLJ/OneDrive - Bosch Group/UBB_thesis/deeplabv3_resnet50/dataset/training/label_19"


def count_train_ids(image_path):
    """Count the trainId values in the image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Count the occurrence of each trainId
    train_id_count: Dict[int, int] = defaultdict(int)
    unique_train_ids = np.unique(image)  # Get unique pixel values (IDs)

    for train_id in unique_train_ids:
        train_id_count[train_id] += np.sum(
            image == train_id
        )  # Count occurrences of each trainId

    return train_id_count


# Process each file in the input directory
all_train_ids: Dict[int, int] = defaultdict(int)

for filename in os.listdir(input_dir):
    if filename.endswith(".png"):  # Assuming the files are PNG images
        input_path = os.path.join(input_dir, filename)
        train_id_count = count_train_ids(input_path)

        if train_id_count:
            for train_id, count in train_id_count.items():
                all_train_ids[train_id] += count

# Print the counts of all trainIds
print("trainId counts across all images:")
for train_id, count in sorted(all_train_ids.items()):
    print(f"TrainId {train_id}: {count} pixels")
